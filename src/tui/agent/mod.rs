pub mod edit_summary;
mod events;
mod tool_dispatch;
pub mod turn;

use crate::conversation::RetrySettings;
use crate::mcp::client::McpClient;
use crate::ollama::client::OllamaClient;
use crate::permissions::engine::PermissionEngine;
use crate::session::{storage as session_storage, Session};
use crate::tools::registry::ToolRegistry;
use crate::tui::BackendEvent;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

// Re-export utilities used by conversation.rs and other callers outside this module.
pub use events::select_client;

/// Background agent task — drives the conversation loop and communicates
/// with the TUI via channels.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    mut client: OllamaClient,
    mut tool_client: Option<OllamaClient>,
    mut registry: ToolRegistry,
    mcp_clients: HashMap<String, Arc<Mutex<McpClient>>>,
    system_prompt: String,
    mut session: Session,
    mut engine: PermissionEngine,
    config_dir: PathBuf,
    global_config_dir: PathBuf,
    verbose: bool,
    max_turns: usize,
    staging: bool,
    format_after: bool,
    event_tx: mpsc::Sender<BackendEvent>,
    mut user_rx: mpsc::Receiver<String>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    retry: RetrySettings,
) {
    let hooks_config = crate::tools::hooks::HooksConfig::load(&config_dir);

    // Query the model's context window limit once at startup; fallback 4096.
    let mut context_limit = client.model_context_limit(client.model()).await;

    let mut messages: Vec<serde_json::Value> = if session.messages.is_empty() {
        vec![json!({"role": "system", "content": system_prompt})]
    } else {
        session.messages.clone()
    };

    // Accumulated staged changes awaiting user approval.
    let mut pending: Vec<crate::changeset::PendingChange> = Vec::new();
    let mut plan_mode = false;
    // Token counts from the most recent turn — held until apply/reject resolves.
    let mut held_tokens: Option<(u64, u64)> = None;
    // Snapshot of messages before the last user turn — enables `/undo`.
    let mut undo_snapshot: Option<Vec<serde_json::Value>> = None;

    while let Some(user_input) = user_rx.recv().await {
        // ── Diff-review control signals ───────────────────────────────────────
        if let Some(decisions_str) = user_input.strip_prefix("__apply_selected__:") {
            let decisions: Vec<Option<bool>> = decisions_str
                .split(',')
                .map(|s| match s.trim() {
                    "0" => Some(false),
                    _ => None, // "1" or anything else → undecided (apply)
                })
                .collect();
            let applied_paths: Vec<std::path::PathBuf> = pending
                .iter()
                .enumerate()
                .filter(|(i, _)| decisions.get(*i).copied().flatten() != Some(false))
                .map(|(_, c)| c.path.clone())
                .collect();
            event_tx
                .send(BackendEvent::ChangesetApplied(
                    pending
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| decisions.get(*i).copied().flatten() != Some(false))
                        .map(|(_, c)| c.clone())
                        .collect(),
                ))
                .await
                .ok();
            let (written, skipped, errors) =
                crate::changeset::apply_selected(&pending, &decisions).await;
            let n = pending.len();
            pending.clear();
            if errors.is_empty() {
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[Applied {} of {} file{} ({} rejected)]\n",
                        written,
                        n,
                        if n == 1 { "" } else { "s" },
                        skipped
                    )))
                    .await
                    .ok();
            } else {
                for e in &errors {
                    event_tx
                        .send(BackendEvent::StreamToken(format!("[Error: {}]\n", e)))
                        .await
                        .ok();
                }
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[Applied {}/{} file{}; {} rejected; {} error{}]\n",
                        written,
                        n,
                        if n == 1 { "" } else { "s" },
                        skipped,
                        errors.len(),
                        if errors.len() == 1 { "" } else { "s" }
                    )))
                    .await
                    .ok();
            }
            if format_after {
                for path in &applied_paths {
                    match crate::format::format_file(path).await {
                        Ok(true) => {
                            event_tx
                                .send(BackendEvent::StreamToken(format!(
                                    "[Formatted {}]\n",
                                    path.display()
                                )))
                                .await
                                .ok();
                        }
                        Ok(false) => {}
                        Err(e) => {
                            event_tx
                                .send(BackendEvent::StreamToken(format!(
                                    "[Format error for {}: {}]\n",
                                    path.display(),
                                    e
                                )))
                                .await
                                .ok();
                        }
                    }
                }
            }
            let (pt, ct) = held_tokens.take().unwrap_or((0, 0));
            event_tx
                .send(BackendEvent::TurnComplete {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                })
                .await
                .ok();
            continue;
        }

        if user_input == "__apply__" {
            let applied_paths: Vec<std::path::PathBuf> =
                pending.iter().map(|c| c.path.clone()).collect();
            event_tx
                .send(BackendEvent::ChangesetApplied(pending.clone()))
                .await
                .ok();
            let (written, errors) = crate::changeset::apply_all(&pending).await;
            let n = pending.len();
            pending.clear();
            if errors.is_empty() {
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[Applied {} staged file{}]\n",
                        written,
                        if written == 1 { "" } else { "s" }
                    )))
                    .await
                    .ok();
            } else {
                for e in &errors {
                    event_tx
                        .send(BackendEvent::StreamToken(format!("[Error: {}]\n", e)))
                        .await
                        .ok();
                }
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[Applied {}/{} file{}; {} error{}]\n",
                        written,
                        n,
                        if n == 1 { "" } else { "s" },
                        errors.len(),
                        if errors.len() == 1 { "" } else { "s" }
                    )))
                    .await
                    .ok();
            }
            // --format-after: run formatters on each applied file
            if format_after {
                for path in &applied_paths {
                    match crate::format::format_file(path).await {
                        Ok(true) => {
                            event_tx
                                .send(BackendEvent::StreamToken(format!(
                                    "[Formatted {}]\n",
                                    path.display()
                                )))
                                .await
                                .ok();
                        }
                        Ok(false) => {}
                        Err(e) => {
                            event_tx
                                .send(BackendEvent::StreamToken(format!(
                                    "[Format error for {}: {}]\n",
                                    path.display(),
                                    e
                                )))
                                .await
                                .ok();
                        }
                    }
                }
            }
            let (pt, ct) = held_tokens.take().unwrap_or((0, 0));
            event_tx
                .send(BackendEvent::TurnComplete {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                })
                .await
                .ok();
            continue;
        }

        if user_input == "__reject__" {
            let n = pending.len();
            pending.clear();
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Rejected {} staged change{}]\n",
                    n,
                    if n == 1 { "" } else { "s" }
                )))
                .await
                .ok();
            let (pt, ct) = held_tokens.take().unwrap_or((0, 0));
            event_tx
                .send(BackendEvent::TurnComplete {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                })
                .await
                .ok();
            continue;
        }

        // Plan mode toggle signal
        if user_input == "__plan_mode__:on" {
            plan_mode = true;
            continue;
        }
        if user_input == "__plan_mode__:off" {
            plan_mode = false;
            continue;
        }

        // Discard any stale pending from a previous turn
        pending.clear();
        held_tokens = None;

        if user_input.starts_with('/') {
            events::handle_slash_signal(
                &user_input,
                &mut messages,
                &mut undo_snapshot,
                &mut session,
                &config_dir,
                &global_config_dir,
                &event_tx,
                &mut client,
                &mut tool_client,
                &engine,
                &mcp_clients,
                &mut registry,
                &mut context_limit,
            )
            .await;
            continue;
        }

        let (text, images) = events::parse_image_payload(&user_input);
        let user_msg = if images.is_empty() {
            json!({"role": "user", "content": text})
        } else {
            json!({"role": "user", "content": text, "images": images})
        };
        // Snapshot before this turn so /undo can restore.
        event_tx.send(BackendEvent::TurnStarted).await.ok();
        undo_snapshot = Some(messages.clone());
        messages.push(user_msg);
        session.messages = messages.clone();
        session_storage::save(&config_dir, &session).ok();

        // Warn before the turn if context is ≥80% full.
        {
            let used = crate::tokens::conversation_tokens(&messages);
            let pct = used * 100 / context_limit.max(1);
            if pct >= 95 {
                event_tx
                    .send(BackendEvent::ContextWarning(format!(
                        "Context {}% full (~{} / {} tokens) — compaction will trigger soon",
                        pct, used, context_limit
                    )))
                    .await
                    .ok();
            } else if pct >= 80 {
                event_tx
                    .send(BackendEvent::ContextWarning(format!(
                        "Context {}% full (~{} / {} tokens)",
                        pct, used, context_limit
                    )))
                    .await
                    .ok();
            }
        }

        if let Some((prompt_tokens, completion_tokens)) = turn::run_turn(
            &client,
            tool_client.as_ref(),
            &registry,
            &mcp_clients,
            &hooks_config,
            verbose,
            max_turns,
            context_limit,
            staging,
            &mut pending,
            &mut messages,
            &mut session,
            &mut engine,
            &config_dir,
            &global_config_dir,
            &event_tx,
            &cancel_rx,
            plan_mode,
            &retry,
        )
        .await
        {
            // Persist turn/token counters for the /sessions listing. Saturating
            // arithmetic because the field width (u32/u64) is the defensive
            // ceiling, not something a real session will reach.
            session.turn_count = session.turn_count.saturating_add(1);
            session.prompt_tokens = session.prompt_tokens.saturating_add(prompt_tokens);
            session.completion_tokens = session.completion_tokens.saturating_add(completion_tokens);
            session_storage::save(&config_dir, &session).ok();

            if staging && !pending.is_empty() {
                held_tokens = Some((prompt_tokens, completion_tokens));
                event_tx
                    .send(BackendEvent::StagedChangeset(pending.clone()))
                    .await
                    .ok();
            } else {
                event_tx
                    .send(BackendEvent::TurnComplete {
                        prompt_tokens,
                        completion_tokens,
                    })
                    .await
                    .ok();
            }

            // Emit context usage for the status bar pill.
            let used = crate::tokens::conversation_tokens(&messages);
            event_tx
                .send(BackendEvent::ContextUsage {
                    used,
                    limit: context_limit,
                })
                .await
                .ok();

            // Auto-generate a title after the first turn
            if session.title.is_none() {
                let first_user = session
                    .messages
                    .iter()
                    .find(|m| m["role"].as_str() == Some("user"))
                    .and_then(|m| m["content"].as_str())
                    .unwrap_or("")
                    .chars()
                    .take(200)
                    .collect::<String>();

                if !first_user.is_empty() {
                    let title_client = client.clone();
                    let config_dir_clone = config_dir.clone();
                    let session_id = session.id.clone();
                    let title_event_tx = event_tx.clone();
                    tokio::spawn(async move {
                        let req = vec![serde_json::json!({
                            "role": "user",
                            "content": format!(
                                "Generate a concise 4-6 word title for a conversation that starts with:\n\"{}\"\n\nRespond with ONLY the title, no quotes, no punctuation at the end.",
                                first_user
                            )
                        })];
                        if let Ok(resp) = title_client.chat(&req, &[]).await {
                            let title = resp.message.content.trim().to_string();
                            if !title.is_empty() && title.len() < 80 {
                                if let Ok(mut sess) =
                                    crate::session::storage::load(&config_dir_clone, &session_id)
                                {
                                    if sess.title.as_deref() == Some("") {
                                        sess.title = Some(title.clone());
                                        let _ =
                                            crate::session::storage::save(&config_dir_clone, &sess);
                                        title_event_tx
                                            .send(crate::tui::BackendEvent::TitleGenerated(title))
                                            .await
                                            .ok();
                                    }
                                }
                            }
                        }
                    });
                }

                // Mark as "in progress" so we don't spawn multiple title tasks
                session.title = Some(String::new());
                session_storage::save(&config_dir, &session).ok();
            }
        }
    }
}
