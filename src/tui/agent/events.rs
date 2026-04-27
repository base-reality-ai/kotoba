use crate::compaction::maybe_compact;
use crate::mcp::client::McpClient;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::ToolCall;
use crate::permissions::engine::PermissionEngine;
use crate::session::{short_id, storage as session_storage, Session};
use crate::tools::registry::ToolRegistry;
use crate::tui::BackendEvent;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

// ── Pure helpers ─────────────────────────────────────────────────────────────

/// Select which Ollama client to use for a given round.
/// Public so print-mode `conversation.rs` can use the same routing logic.
/// - Round 0 (first response to a user message) → reasoning model (`client`)
/// - Round N>0 where the previous round made tool calls → tool model if configured
/// - Round N>0 where the previous round produced a final answer → reasoning model
pub fn select_client<'a>(
    client: &'a OllamaClient,
    tool_client: Option<&'a OllamaClient>,
    round: usize,
    last_round_was_tools_only: bool,
) -> &'a OllamaClient {
    match tool_client {
        Some(tc) if round > 0 && last_round_was_tools_only => tc,
        _ => client,
    }
}

/// Returns true if the message list is too large to send to Ollama.
/// Uses character count as a cheap proxy for token count (~4 chars/token).
#[allow(dead_code)]
pub fn context_too_large(messages: &[Value]) -> bool {
    const CAP: usize = 512_000; // ~128k tokens at 4 chars/token
    messages
        .iter()
        .filter_map(|m| m["content"].as_str())
        .map(|s| s.len())
        .sum::<usize>()
        > CAP
}

/// Decode the null-byte image protocol: `\x00IMAGES\x00<json>\x00END\x00<text>`
/// Returns (text, `base64_image_list`). If no protocol prefix, returns (input, empty).
pub fn parse_image_payload(input: &str) -> (String, Vec<String>) {
    if let Some(rest) = input.strip_prefix("\x00IMAGES\x00") {
        if let Some(end_idx) = rest.find("\x00END\x00") {
            let imgs_json = &rest[..end_idx];
            let text = rest[end_idx + "\x00END\x00".len()..].to_string();
            if let Ok(pairs) = serde_json::from_str::<Vec<(String, String)>>(imgs_json) {
                let b64s: Vec<String> = pairs.into_iter().map(|(_, data)| data).collect();
                return (text, b64s);
            }
        }
    }
    (input.to_string(), Vec::new())
}

/// Build the JSON representation of an assistant message, including `tool_calls` if present.
pub fn build_assistant_msg(content: &str, tool_calls: &[ToolCall]) -> Value {
    if tool_calls.is_empty() {
        json!({"role": "assistant", "content": content})
    } else {
        json!({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls.iter().map(|tc| json!({
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            })).collect::<Vec<_>>()
        })
    }
}

// ── Slash command handler ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub async fn handle_slash_signal(
    signal: &str,
    messages: &mut Vec<Value>,
    undo_snapshot: &mut Option<Vec<Value>>,
    session: &mut Session,
    config_dir: &Path,
    event_tx: &mpsc::Sender<BackendEvent>,
    client: &mut OllamaClient,
    tool_client: &mut Option<OllamaClient>,
    engine: &PermissionEngine,
    mcp_clients: &HashMap<String, Arc<Mutex<McpClient>>>,
    registry: &mut ToolRegistry,
    context_limit: &mut usize,
) {
    if signal == "/undo" {
        if let Some(snap) = undo_snapshot.take() {
            *messages = snap;
            session.messages = messages.clone();
            session_storage::save(config_dir, session).ok();
            event_tx.send(BackendEvent::UndoComplete).await.ok();
        } else {
            event_tx.send(BackendEvent::NothingToUndo).await.ok();
        }
        return;
    }

    if signal == "/clear" {
        messages.retain(|m| m["role"].as_str() == Some("system"));
        session.messages = messages.clone();
        session_storage::save(config_dir, session).ok();
    } else if signal == "/compact" {
        let before = messages.len();
        match maybe_compact(messages, client, false).await {
            Ok(compacted) => {
                let after = compacted.len();
                *messages = compacted;
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[Context compacted — {} messages → {} messages]\n",
                        before, after
                    )))
                    .await
                    .ok();
            }
            Err(e) => {
                event_tx
                    .send(BackendEvent::Error(format!("Compact failed: {}", e)))
                    .await
                    .ok();
            }
        }
    } else if signal == "/compact/smart" {
        match compact_smart(messages, client, session, config_dir, event_tx).await {
            Ok(()) => {}
            Err(e) => {
                event_tx
                    .send(BackendEvent::Error(format!("Compact failed: {}", e)))
                    .await
                    .ok();
            }
        }
    } else if let Some(spec) = signal.strip_prefix("/model tool ") {
        let spec = spec.trim();
        let settings_path = config_dir.join("settings.json");
        let existing_json = std::fs::read_to_string(&settings_path).ok();
        let mut v: serde_json::Value = existing_json
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(serde_json::json!({}));

        if spec == "off" {
            *tool_client = None;
            v.as_object_mut().map(|o| o.remove("tool_model"));
            let _ = std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&v).unwrap_or_default(),
            );
            event_tx
                .send(BackendEvent::StreamToken(
                    "[Tool model routing disabled — using reasoning model for all rounds]\n"
                        .to_string(),
                ))
                .await
                .ok();
        } else {
            let base = client.base_url().to_string();
            *tool_client = Some(OllamaClient::new(base, spec.to_string()));
            v["tool_model"] = serde_json::Value::String(spec.to_string());
            let _ = std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&v).unwrap_or_default(),
            );
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Tool model set to {} — saved to settings.json]\n",
                    spec
                )))
                .await
                .ok();
        }
    } else if let Some(name) = signal.strip_prefix("/model embed ") {
        let name = name.trim().to_string();
        if !name.is_empty() {
            let settings_path = config_dir.join("settings.json");
            let existing_json = std::fs::read_to_string(&settings_path).ok();
            let mut v: serde_json::Value = existing_json
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(serde_json::json!({}));
            v["embed_model"] = serde_json::Value::String(name.clone());
            let _ = std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&v).unwrap_or_default(),
            );
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Embed model set to {} — takes effect on next `dm index` run]\n",
                    name
                )))
                .await
                .ok();
        }
    } else if let Some(name) = signal.strip_prefix("/model ") {
        let name = name.trim().to_string();
        if !name.is_empty() {
            let base = client.base_url().to_string();
            *client = OllamaClient::new(base, name.clone());
            *context_limit = client.model_context_limit(&name).await;
            let settings_path = config_dir.join("settings.json");
            let new_entry = serde_json::json!({"model": name});
            let updated = if let Ok(existing) = std::fs::read_to_string(&settings_path) {
                if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&existing) {
                    v["model"] = serde_json::Value::String(name.clone());
                    v
                } else {
                    new_entry
                }
            } else {
                new_entry
            };
            let _ = std::fs::write(
                &settings_path,
                serde_json::to_string_pretty(&updated).unwrap_or_default(),
            );
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Model switched to {} — saved to settings.json]\n",
                    name
                )))
                .await
                .ok();
        }
    } else if signal == "/permissions" {
        let report = engine.describe_rules();
        event_tx
            .send(BackendEvent::PermissionsReport(report))
            .await
            .ok();
    } else if let Some(rest) = signal.strip_prefix("/mcp ") {
        // Forward MCP tool calls from slash commands
        let parts: Vec<&str> = rest.splitn(3, ' ').collect();
        if parts.len() >= 2 {
            let server = parts[0];
            let tool = parts[1];
            let args_str = parts.get(2).unwrap_or(&"{}");
            let args: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            if let Some(mc) = mcp_clients.get(server) {
                let mut locked = mc.lock().await;
                match locked.call_tool(tool, args).await {
                    Ok(output) => {
                        event_tx
                            .send(BackendEvent::StreamToken(format!("{}\n", output)))
                            .await
                            .ok();
                    }
                    Err(e) => {
                        event_tx
                            .send(BackendEvent::Error(format!("MCP error: {}", e)))
                            .await
                            .ok();
                    }
                }
            } else {
                event_tx
                    .send(BackendEvent::Error(format!(
                        "MCP server '{}' not found",
                        server
                    )))
                    .await
                    .ok();
            }
        }
    } else if let Some(name) = signal.strip_prefix("/tool disable ") {
        let name = name.trim();
        if registry.disable(name) {
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Tool '{}' disabled — won't be offered to the model]\n",
                    name
                )))
                .await
                .ok();
        } else {
            event_tx
                .send(BackendEvent::Error(format!(
                    "Unknown tool '{}'. Use /tool list to see available tools.",
                    name
                )))
                .await
                .ok();
        }
    } else if let Some(name) = signal.strip_prefix("/tool enable ") {
        let name = name.trim();
        if registry.enable(name) {
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Tool '{}' re-enabled]\n",
                    name
                )))
                .await
                .ok();
        } else {
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[Tool '{}' was not disabled]\n",
                    name
                )))
                .await
                .ok();
        }
    } else if signal == "/tool list" {
        let defs = registry.definitions();
        let disabled = registry.disabled_names();
        let mut lines = vec![format!(
            "Tools ({} active, {} disabled):",
            defs.len(),
            disabled.len()
        )];
        for d in &defs {
            lines.push(format!("  ✓ {}", d.function.name));
        }
        for name in &disabled {
            lines.push(format!("  ✗ {} (disabled)", name));
        }
        event_tx
            .send(BackendEvent::StreamToken(lines.join("\n") + "\n"))
            .await
            .ok();
    } else if let Some(target_id) = signal.strip_prefix("/resume ") {
        let target_id = target_id.trim();
        match session_storage::list_meta(config_dir) {
            Ok(metas) => {
                let matches: Vec<_> = metas
                    .iter()
                    .filter(|m| m.id.starts_with(target_id))
                    .collect();
                match matches.len() {
                    0 => {
                        event_tx
                            .send(BackendEvent::Error(format!(
                                "No session found matching '{}'",
                                target_id
                            )))
                            .await
                            .ok();
                    }
                    1 => {
                        let target = &matches[0];
                        let new_full_id = target.id.clone();
                        let new_short = short_id(&new_full_id);
                        let title = target.title.clone().unwrap_or_default();
                        let title_display = if title.is_empty() {
                            "(untitled)".to_string()
                        } else {
                            title.clone()
                        };

                        session_storage::save(config_dir, session).ok();
                        let old_id = short_id(&session.id).to_string();

                        match session_storage::load(config_dir, &new_full_id) {
                            Ok(loaded) => {
                                let msg_count = loaded.messages.len();
                                messages.clone_from(&loaded.messages);
                                *session = loaded;
                                *undo_snapshot = None;
                                event_tx
                                    .send(BackendEvent::SessionSwitched {
                                        old_id,
                                        new_id: new_short.to_string(),
                                        new_full_id,
                                        title: title_display,
                                        message_count: msg_count,
                                    })
                                    .await
                                    .ok();
                            }
                            Err(e) => {
                                event_tx
                                    .send(BackendEvent::Error(format!(
                                        "Failed to load session {}: {}",
                                        new_short, e
                                    )))
                                    .await
                                    .ok();
                            }
                        }
                    }
                    n => {
                        let mut lines =
                            vec![format!("Ambiguous: {} sessions match '{}':", n, target_id)];
                        for m in &matches {
                            let short = short_id(&m.id);
                            let t = m.title.as_deref().unwrap_or("(untitled)");
                            lines.push(format!("  [{}] {}", short, t));
                        }
                        event_tx
                            .send(BackendEvent::StreamToken(lines.join("\n") + "\n"))
                            .await
                            .ok();
                    }
                }
            }
            Err(e) => {
                event_tx
                    .send(BackendEvent::Error(format!(
                        "Failed to list sessions: {}",
                        e
                    )))
                    .await
                    .ok();
            }
        }
    }
}

/// Context-aware compaction: keeps the system message and the last 8 messages
/// untouched, summarizes the middle portion with a focused prompt, replaces
/// them with a single synthetic message, and saves the session.
async fn compact_smart(
    messages: &mut Vec<serde_json::Value>,
    client: &OllamaClient,
    session: &mut Session,
    config_dir: &Path,
    event_tx: &mpsc::Sender<BackendEvent>,
) -> anyhow::Result<()> {
    const KEEP_TAIL: usize = 8;

    // Need at least: system + 1 message to summarize + KEEP_TAIL
    if messages.len() <= KEEP_TAIL + 1 {
        event_tx
            .send(BackendEvent::StreamToken(
                "[Nothing to compact — not enough messages]\n".to_string(),
            ))
            .await
            .ok();
        return Ok(());
    }

    let system = messages[0].clone();
    let tail_start = messages.len() - KEEP_TAIL;
    let to_summarize = &messages[1..tail_start];
    let n_summarized = to_summarize.len();

    let history_text = to_summarize
        .iter()
        .map(|m| {
            let role = m["role"].as_str().unwrap_or("unknown");
            let content = m["content"].as_str().unwrap_or("[tool call]");
            let cap = 500usize;
            let mut end = cap.min(content.len());
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            let truncated = &content[..end];
            format!("[{}]: {}", role, truncated)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let summary_request = vec![serde_json::json!({
        "role": "user",
        "content": format!(
            "Summarize this conversation segment in 3-5 sentences. Preserve: decisions made, \
            code written, files changed, errors encountered. Be specific about file names and \
            function names.\n\n{}",
            history_text
        )
    })];

    let summary_resp = client.chat(&summary_request, &[]).await?;
    let summary_text = summary_resp.message.content;

    // Estimate context freed (chars before vs after)
    let chars_before: usize = to_summarize
        .iter()
        .filter_map(|m| m["content"].as_str())
        .map(|s| s.len())
        .sum();
    let chars_after = summary_text.len();
    let freed_chars = chars_before.saturating_sub(chars_after);
    // Very rough token→% estimate: 1 token ≈ 4 chars, typical context ~100k tokens = ~400k chars
    let freed_pct = freed_chars * 100 / (400_000usize.max(chars_before + 1));

    let synthetic = serde_json::json!({
        "role": "system",
        "content": format!(
            "[Compacted {} messages — summary: {}]",
            n_summarized, summary_text
        )
    });

    let tail: Vec<serde_json::Value> = messages[tail_start..].to_vec();
    let mut new_messages = vec![system, synthetic];
    new_messages.extend(tail);
    *messages = new_messages;

    session.messages.clone_from(messages);
    session_storage::save(config_dir, session).ok();

    event_tx
        .send(BackendEvent::StreamToken(format!(
            "[compacted {} messages → 1 summary. Context freed: ~{}%]\n",
            n_summarized, freed_pct
        )))
        .await
        .ok();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_too_large_small_messages() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        assert!(!context_too_large(&msgs));
    }

    #[test]
    fn context_too_large_at_cap() {
        let big = "x".repeat(513_000);
        let msgs = vec![serde_json::json!({"role": "user", "content": big})];
        assert!(context_too_large(&msgs));
    }

    #[test]
    fn context_too_large_sums_all_messages() {
        let chunk = "x".repeat(256_001);
        let msgs = vec![
            serde_json::json!({"role": "user", "content": chunk.clone()}),
            serde_json::json!({"role": "assistant", "content": chunk}),
        ];
        assert!(context_too_large(&msgs));
    }

    #[test]
    fn context_too_large_ignores_non_string_content() {
        let msgs = vec![serde_json::json!({"role": "assistant", "tool_calls": []})];
        assert!(!context_too_large(&msgs));
    }

    #[test]
    fn parse_image_payload_no_images() {
        let (text, imgs) = parse_image_payload("hello world");
        assert_eq!(text, "hello world");
        assert!(imgs.is_empty());
    }

    #[test]
    fn parse_image_payload_with_images() {
        let payload = "\x00IMAGES\x00[[\"file.png\",\"abc123\"]]\x00END\x00describe this";
        let (text, imgs) = parse_image_payload(payload);
        assert_eq!(text, "describe this");
        assert_eq!(imgs, vec!["abc123"]);
    }

    #[test]
    fn parse_image_payload_malformed_json_falls_back() {
        // Has the protocol prefix but the JSON is invalid — falls back to plain text
        let payload = "\x00IMAGES\x00not-json\x00END\x00some text";
        let (text, imgs) = parse_image_payload(payload);
        // Falls back to returning the full input unchanged
        assert_eq!(text, payload);
        assert!(imgs.is_empty());
    }

    #[test]
    fn build_assistant_msg_no_tool_calls() {
        let msg = build_assistant_msg("hello", &[]);
        assert_eq!(msg["role"].as_str(), Some("assistant"));
        assert_eq!(msg["content"].as_str(), Some("hello"));
        assert!(msg["tool_calls"].is_null());
    }

    #[test]
    fn build_assistant_msg_with_tool_calls() {
        use crate::ollama::types::{FunctionCall, ToolCall};
        let tc = ToolCall {
            function: FunctionCall {
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            },
        };
        let msg = build_assistant_msg("", &[tc]);
        assert_eq!(msg["role"].as_str(), Some("assistant"));
        let calls = msg["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"].as_str(), Some("bash"));
    }

    #[test]
    fn select_client_round_zero_always_uses_primary() {
        let client = OllamaClient::new("http://localhost:11434".to_string(), "primary".to_string());
        let tool_client =
            OllamaClient::new("http://localhost:11434".to_string(), "tool".to_string());
        let selected = select_client(&client, Some(&tool_client), 0, true);
        assert_eq!(selected.model(), "primary");
    }

    #[test]
    fn select_client_no_tool_client_always_uses_primary() {
        let client = OllamaClient::new("http://localhost:11434".to_string(), "primary".to_string());
        let selected = select_client(&client, None, 1, true);
        assert_eq!(selected.model(), "primary");
    }

    #[test]
    fn select_client_tool_round_uses_tool_client() {
        let client = OllamaClient::new("http://localhost:11434".to_string(), "primary".to_string());
        let tool_client =
            OllamaClient::new("http://localhost:11434".to_string(), "tool".to_string());
        let selected = select_client(&client, Some(&tool_client), 1, true);
        assert_eq!(selected.model(), "tool");
    }

    #[test]
    fn select_client_non_tool_round_uses_primary() {
        // round > 0 but last_round_was_tools_only = false → use primary (reasoning) model
        let client = OllamaClient::new("http://localhost:11434".to_string(), "primary".to_string());
        let tool_client =
            OllamaClient::new("http://localhost:11434".to_string(), "tool".to_string());
        let selected = select_client(&client, Some(&tool_client), 1, false);
        assert_eq!(selected.model(), "primary");
    }

    #[test]
    fn parse_image_payload_empty_input() {
        let (text, imgs) = parse_image_payload("");
        assert_eq!(text, "");
        assert!(imgs.is_empty());
    }

    #[test]
    fn context_too_large_empty_messages() {
        let msgs: &[Value] = &[];
        assert!(
            !context_too_large(msgs),
            "empty messages should not exceed context"
        );
    }

    #[test]
    fn build_assistant_msg_empty_content() {
        let msg = build_assistant_msg("", &[]);
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "");
    }

    #[tokio::test]
    async fn model_switch_updates_context_limit() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/show")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "model_info": { "llama.context_length": 65536_u64 }
                })
                .to_string(),
            )
            .create_async()
            .await;

        let base = server.url();
        let mut client = OllamaClient::new(format!("{}/api", base), "old-model".to_string());
        let mut tool_client: Option<OllamaClient> = None;
        let engine = PermissionEngine::new(true, vec![]);
        let mcp_clients: HashMap<String, Arc<Mutex<McpClient>>> = HashMap::new();
        let mut registry = ToolRegistry::default();
        let mut messages = vec![json!({"role": "system", "content": "test"})];
        let mut undo_snapshot: Option<Vec<Value>> = None;
        let config_dir = tempfile::TempDir::new().unwrap();
        let mut session = Session {
            id: "test".to_string(),
            messages: messages.clone(),
            title: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            cwd: "/tmp".to_string(),
            host_project: None,
            model: "old-model".to_string(),
            compact_failures: 0,
            turn_count: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: None,
        };
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let mut context_limit: usize = 4096;

        handle_slash_signal(
            "/model new-model",
            &mut messages,
            &mut undo_snapshot,
            &mut session,
            config_dir.path(),
            &event_tx,
            &mut client,
            &mut tool_client,
            &engine,
            &mcp_clients,
            &mut registry,
            &mut context_limit,
        )
        .await;

        assert_eq!(client.model(), "new-model");
        assert_eq!(
            context_limit, 65536,
            "context_limit should be updated from model_info"
        );
        mock.assert_async().await;

        // Drain the event to confirm the switch message was sent
        if let Some(BackendEvent::StreamToken(msg)) = event_rx.recv().await {
            assert!(
                msg.contains("new-model"),
                "event should mention new model name"
            );
        }
    }
}
