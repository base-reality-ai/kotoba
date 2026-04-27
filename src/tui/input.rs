//! TUI keyboard input handling.
//!
//! Maps crossterm key events into App state mutations and backend
//! requests. Owns the slash-command parser, prompt submission, scroll
//! navigation, mode toggles, and permission-prompt resolution. Calls
//! into `tui::commands` for slash dispatch and `OllamaClient` /
//! `DaemonClient` for backend work.

use crate::daemon::client::DaemonClient;
use crate::ollama::client::OllamaClient;
use crate::tui::{
    app::{App, EntryKind, Mode},
    commands, PermissionDecision,
};
use crossterm::event::{KeyCode, KeyModifiers};
use tokio::sync::mpsc;

pub async fn handle_key(
    app: &mut App,
    key: crossterm::event::KeyEvent,
    user_tx: &mpsc::Sender<String>,
    client: &OllamaClient,
    cancel_tx: &tokio::sync::watch::Sender<bool>,
    daemon_client: &mut Option<DaemonClient>,
) -> anyhow::Result<()> {
    use KeyCode::*;

    if key.code == Char('c') && key.modifiers == KeyModifiers::CONTROL {
        let now = std::time::Instant::now();
        if let Some(last) = app.last_ctrl_c {
            if now.duration_since(last).as_millis() < 800 {
                app.should_quit = true;
                return Ok(());
            }
        }
        // First press: cancel agent if busy, otherwise just show hint
        app.last_ctrl_c = Some(now);
        if app.agent_busy {
            cancel_tx.send(true).ok();
            app.push_entry(
                EntryKind::SystemInfo,
                "Interrupted. Press Ctrl+C again to quit.".to_string(),
            );
        } else {
            app.push_entry(
                EntryKind::SystemInfo,
                "Press Ctrl+C again to quit.".to_string(),
            );
        }
        return Ok(());
    }

    match &app.mode {
        Mode::PermissionDialog => {
            if let Some(perm) = app.pending_permission.take() {
                let decision = match key.code {
                    Char('y') | Enter => PermissionDecision::AllowOnce,
                    Char('a') => PermissionDecision::AlwaysAllow,
                    Char('n') | Esc => PermissionDecision::DenyOnce,
                    Char('N') => PermissionDecision::AlwaysDeny,
                    _ => {
                        app.pending_permission = Some(perm);
                        return Ok(());
                    }
                };
                perm.reply.send(decision).ok();
                app.mode = Mode::Input;
            }
        }

        Mode::AskUserQuestion => match key.code {
            Enter => {
                if let Some(q) = app.pending_question.take() {
                    let answer = app.take_input();
                    app.mode = Mode::Input;
                    q.reply.send(answer).ok();
                }
            }
            Esc => {
                if let Some(q) = app.pending_question.take() {
                    app.input.clear();
                    app.cursor = 0;
                    app.mode = Mode::Input;
                    q.reply.send(String::new()).ok();
                }
            }
            Backspace => {
                app.input_backspace();
            }
            Char(c)
                if key.modifiers == KeyModifiers::NONE || key.modifiers == KeyModifiers::SHIFT =>
            {
                app.input_insert(c);
            }
            _ => {}
        },

        Mode::HelpOverlay => {
            if matches!(key.code, Esc | Char('q')) {
                app.mode = Mode::Input;
            }
        }

        Mode::DiffReview => match key.code {
            Char('y') => {
                if let Some(d) = app.diff_file_decisions.get_mut(app.diff_review_idx) {
                    *d = Some(true);
                }
                // Advance to next undecided file
                let n = app.staged_changes.len();
                for offset in 1..n {
                    let next = (app.diff_review_idx + offset) % n;
                    if app
                        .diff_file_decisions
                        .get(next)
                        .copied()
                        .flatten()
                        .is_none()
                    {
                        app.diff_review_idx = next;
                        app.diff_scroll = 0;
                        break;
                    }
                }
            }
            Char('x') => {
                if let Some(d) = app.diff_file_decisions.get_mut(app.diff_review_idx) {
                    *d = Some(false);
                }
                let n = app.staged_changes.len();
                for offset in 1..n {
                    let next = (app.diff_review_idx + offset) % n;
                    if app
                        .diff_file_decisions
                        .get(next)
                        .copied()
                        .flatten()
                        .is_none()
                    {
                        app.diff_review_idx = next;
                        app.diff_scroll = 0;
                        break;
                    }
                }
            }
            Char('a') => {
                let has_rejections = app.diff_file_decisions.contains(&Some(false));
                if has_rejections {
                    let encoded: String = app
                        .diff_file_decisions
                        .iter()
                        .map(|d| if *d == Some(false) { "0" } else { "1" })
                        .collect::<Vec<_>>()
                        .join(",");
                    user_tx
                        .send(format!("__apply_selected__:{}", encoded))
                        .await
                        .ok();
                } else {
                    user_tx.send("__apply__".to_string()).await.ok();
                }
                app.mode = Mode::Input;
                app.agent_busy = true;
                app.turn_start = Some(std::time::Instant::now());
                app.staged_changes.clear();
                app.diff_file_decisions.clear();
            }
            Char('r') | Esc => {
                user_tx.send("__reject__".to_string()).await.ok();
                app.mode = Mode::Input;
                app.staged_changes.clear();
                app.diff_file_decisions.clear();
            }
            Char('n') | Tab => {
                if app.diff_review_idx + 1 < app.staged_changes.len() {
                    app.diff_review_idx += 1;
                    app.diff_scroll = 0;
                }
            }
            Char('p') => {
                if app.diff_review_idx > 0 {
                    app.diff_review_idx -= 1;
                    app.diff_scroll = 0;
                }
            }
            Char('j') | Down => {
                app.diff_scroll = app.diff_scroll.saturating_add(1);
            }
            Char('k') | Up => {
                app.diff_scroll = app.diff_scroll.saturating_sub(1);
            }
            PageDown => {
                app.diff_scroll = app.diff_scroll.saturating_add(10);
            }
            PageUp => {
                app.diff_scroll = app.diff_scroll.saturating_sub(10);
            }
            _ => {}
        },

        Mode::Scrolling => match key.code {
            Up | Char('k') => {
                app.scroll_top = app.scroll_top.saturating_sub(3);
            }
            Down | Char('j') => {
                let max = app
                    .cached_total_lines
                    .get()
                    .saturating_sub(app.cached_visible_height.get());
                app.scroll_top = app.scroll_top.saturating_add(3).min(max);
            }
            PageUp => {
                app.scroll_top = app.scroll_top.saturating_sub(20);
            }
            PageDown => {
                let max = app
                    .cached_total_lines
                    .get()
                    .saturating_sub(app.cached_visible_height.get());
                app.scroll_top = app.scroll_top.saturating_add(20).min(max);
            }
            Home => {
                app.scroll_top = 0;
            }
            Char('G') => {
                let max = app
                    .cached_total_lines
                    .get()
                    .saturating_sub(app.cached_visible_height.get());
                app.scroll_top = max;
            }
            Char('g') | End => {
                app.scroll_locked = false;
                app.mode = Mode::Input;
            }
            Esc | Char('i') | Enter => {
                app.scroll_locked = false;
                app.mode = Mode::Input;
            }
            _ => {}
        },

        Mode::Input => match (key.code, key.modifiers) {
            (Char('y'), KeyModifiers::CONTROL) => {
                app.copy_last_response();
            }
            (Char('k'), KeyModifiers::CONTROL) if app.agent_busy => {
                if crate::tools::bash::kill_running_bash() {
                    app.push_entry(
                        EntryKind::SystemInfo,
                        "Sent SIGTERM to running bash process.".to_string(),
                    );
                } else {
                    app.push_entry(
                        EntryKind::SystemInfo,
                        "No bash process is currently running.".to_string(),
                    );
                }
                return Ok(());
            }
            (Esc, KeyModifiers::NONE) if app.agent_busy => {
                cancel_tx.send(true).ok();
                app.push_entry(EntryKind::SystemInfo, "Interrupting…".to_string());
                return Ok(());
            }
            (Char('d'), KeyModifiers::CONTROL) => {
                if !app.agent_busy {
                    if app.input.is_empty() {
                        app.should_quit = true;
                        return Ok(());
                    }
                    let text = app.take_input();
                    if let Some(cmd) = commands::parse(&text) {
                        handle_slash_result(
                            commands::execute(cmd, app, client, user_tx).await,
                            app,
                        )?;
                        return Ok(());
                    }
                    submit_turn(app, text, user_tx, cancel_tx, daemon_client).await;
                }
            }
            (Enter, KeyModifiers::NONE) => {
                if !app.agent_busy {
                    let text = app.take_input();
                    if text.is_empty() {
                        return Ok(());
                    }
                    if let Some(cmd) = commands::parse(&text) {
                        handle_slash_result(
                            commands::execute(cmd, app, client, user_tx).await,
                            app,
                        )?;
                        return Ok(());
                    }
                    submit_turn(app, text, user_tx, cancel_tx, daemon_client).await;
                }
            }
            // Ctrl+P / Ctrl+N for input history (readline-style),
            // freeing Up/Down for scroll-wheel via alternate-scroll mode.
            (Char('p'), KeyModifiers::CONTROL) => {
                app.history_prev();
            }
            (Char('n'), KeyModifiers::CONTROL) => {
                app.history_next();
            }
            // Up / Shift+Up / PageUp: enter scroll mode
            (Up, _) | (PageUp, _) => {
                app.mode = Mode::Scrolling;
                app.scroll_locked = true;
                let total = app.cached_total_lines.get();
                let visible = app.cached_visible_height.get();
                let max_scroll = total.saturating_sub(visible);
                app.scroll_top = max_scroll.saturating_sub(3);
            }
            // Down when not scrolling is a no-op (auto-follow is already at bottom)
            (Down, _) => {}
            (Backspace, _) => {
                app.input_backspace();
            }
            (Delete, _) => {
                app.input_delete_forward();
            }
            (Home, _) => {
                app.input_move_start();
            }
            (End, _) => {
                app.input_move_end();
            }
            (Left, KeyModifiers::NONE) => {
                if app.cursor > 0 {
                    let prev = app.input[..app.cursor]
                        .char_indices()
                        .last()
                        .map_or(0, |(i, _)| i);
                    app.cursor = prev;
                }
            }
            (Right, KeyModifiers::NONE) => {
                if app.cursor < app.input.len() {
                    if let Some(c) = app.input[app.cursor..].chars().next() {
                        app.cursor += c.len_utf8();
                    }
                }
            }
            (Left, KeyModifiers::CONTROL) | (Left, KeyModifiers::ALT) => {
                app.input_move_word_back();
            }
            (Right, KeyModifiers::CONTROL) | (Right, KeyModifiers::ALT) => {
                app.input_move_word_forward();
            }
            (Char('a'), KeyModifiers::CONTROL) => {
                app.input_move_start();
            }
            (Char('e'), KeyModifiers::CONTROL) => {
                app.input_move_end();
            }
            (Char('k'), KeyModifiers::CONTROL) => {
                app.input_kill_to_end();
            }
            (Char('w'), KeyModifiers::CONTROL) => {
                app.input_delete_word_back();
            }
            (Char('l'), KeyModifiers::CONTROL) => {
                app.entries.clear();
            }
            (Char('g'), KeyModifiers::CONTROL) => {
                if app.chain_state.is_some() || !app.chain_log.is_empty() {
                    app.show_chain_pane = !app.show_chain_pane;
                }
            }
            (Char('u'), KeyModifiers::CONTROL) => {
                app.input.clear();
                app.cursor = 0;
            }
            (Tab, KeyModifiers::NONE) => {
                if app.input.starts_with("/chain start ") {
                    let prefix = &app.input["/chain start ".len()..];
                    let configs = discover_chain_configs(prefix);
                    match configs.len() {
                        0 => {}
                        1 => {
                            app.input = format!("/chain start {}", configs[0]);
                            app.cursor = app.input.len();
                        }
                        _ => {
                            let hints: Vec<String> = configs
                                .iter()
                                .map(|c| format!("/chain start {}", c))
                                .collect();
                            app.push_entry(
                                EntryKind::SystemInfo,
                                format!("Tab: {}", hints.join("  ")),
                            );
                        }
                    }
                } else if complete_chain_node_name(app, "/chain remove ", "/chain remove ")
                    || complete_chain_node_name(app, "/chain talk ", "/chain talk ")
                    || complete_chain_node_name(app, "/chain model ", "/chain model ")
                {
                    // Node name completion handled above
                } else if app.input.starts_with("/chain ") {
                    let prefix = &app.input["/chain ".len()..];
                    let matches = slash_completions(prefix, commands::CHAIN_SUBCOMMAND_NAMES);
                    match matches.len() {
                        0 => {}
                        1 => {
                            app.input = format!("/chain {}", matches[0]);
                            app.cursor = app.input.len();
                        }
                        _ => {
                            let hints: Vec<String> =
                                matches.iter().map(|c| format!("/chain {}", c)).collect();
                            app.push_entry(
                                EntryKind::SystemInfo,
                                format!("Tab: {}", hints.join("  ")),
                            );
                        }
                    }
                } else if app.input.starts_with("/model ") {
                    // Model name completion: /model <prefix>. Mirrors the
                    // three-arm shape of the slash-command branch so the
                    // UX is consistent across completion contexts.
                    let prefix = app.input["/model ".len()..].to_string();
                    if app.available_models.is_empty() {
                        app.push_entry(
                            EntryKind::SystemInfo,
                            "Tab: no models cached. Run /models to refresh, or check `ollama list`.".to_string(),
                        );
                    } else {
                        let matches = model_completions(&prefix, &app.available_models);
                        match matches.len() {
                            0 => {}
                            1 => {
                                app.input = format!("/model {}", matches[0]);
                                app.cursor = app.input.len();
                            }
                            _ => {
                                let hints: Vec<String> =
                                    matches.iter().map(|m| format!("/model {}", m)).collect();
                                app.push_entry(
                                    EntryKind::SystemInfo,
                                    format!("Tab: {}", hints.join("  ")),
                                );
                            }
                        }
                    }
                } else if app.input.starts_with('/') && !app.input.contains(' ') {
                    // Slash command name completion: /prefix → /command
                    let prefix = &app.input[1..]; // strip the leading '/'
                    let matches = slash_completions(prefix, commands::SLASH_COMMAND_NAMES);
                    match matches.len() {
                        0 => {} // no match — no-op
                        1 => {
                            app.input = format!("/{}", matches[0]);
                            app.cursor = app.input.len();
                        }
                        _ => {
                            let hints: Vec<String> =
                                matches.iter().map(|c| format!("/{}", c)).collect();
                            app.push_entry(
                                EntryKind::SystemInfo,
                                format!("Tab: {}", hints.join("  ")),
                            );
                        }
                    }
                } else {
                    // File path Tab completion: complete the last word as a path
                    let last_word_start = app.input[..app.cursor]
                        .rfind(|c: char| c.is_whitespace())
                        .map_or(0, |i| i + 1);
                    let partial = &app.input[last_word_start..app.cursor];
                    if !partial.is_empty() {
                        let matches = complete_file_path(partial);
                        match matches.len() {
                            0 => {}
                            1 => {
                                let before = &app.input[..last_word_start];
                                let after = &app.input[app.cursor..];
                                app.input = format!("{}{}{}", before, matches[0], after);
                                app.cursor = last_word_start + matches[0].len();
                            }
                            _ if matches.len() <= 20 => {
                                let common = longest_common_prefix(&matches);
                                if common.len() > partial.len() {
                                    let before = &app.input[..last_word_start];
                                    let after = &app.input[app.cursor..];
                                    app.input = format!("{}{}{}", before, common, after);
                                    app.cursor = last_word_start + common.len();
                                }
                                app.push_entry(
                                    EntryKind::SystemInfo,
                                    format!("Tab: {}", matches.join("  ")),
                                );
                            }
                            n => {
                                let common = longest_common_prefix(&matches);
                                if common.len() > partial.len() {
                                    let before = &app.input[..last_word_start];
                                    let after = &app.input[app.cursor..];
                                    app.input = format!("{}{}{}", before, common, after);
                                    app.cursor = last_word_start + common.len();
                                }
                                app.push_entry(
                                    EntryKind::SystemInfo,
                                    format!("Tab: {} matches", n),
                                );
                            }
                        }
                    }
                }
            }
            (Enter, KeyModifiers::SHIFT) | (Enter, KeyModifiers::ALT) => {
                app.input_insert('\n');
            }
            (Char(c), KeyModifiers::NONE) | (Char(c), KeyModifiers::SHIFT) => {
                app.input_insert(c);
            }
            _ => {}
        },
    }
    Ok(())
}

/// Shared submit logic for Enter and Ctrl+D.
async fn submit_turn(
    app: &mut App,
    text: String,
    user_tx: &mpsc::Sender<String>,
    cancel_tx: &tokio::sync::watch::Sender<bool>,
    daemon_client: &mut Option<DaemonClient>,
) {
    app.push_history(text.clone());
    // Snapshot display state before the turn so /undo can restore it.
    app.undo_entries_snapshot = Some(app.entries.clone());
    let image_names: Vec<String> = app.pending_images.iter().map(|(f, _)| f.clone()).collect();
    for filename in image_names {
        app.push_entry(EntryKind::ImageAttachment, filename);
    }
    app.push_entry(EntryKind::UserMessage, text.clone());
    app.agent_busy = true;
    app.turn_start = Some(std::time::Instant::now());
    app.streaming_partial = Some(String::new());
    app.perf = None; // cleared at turn start; populated during streaming
    cancel_tx.send(false).ok();

    if let Some(dc) = daemon_client.as_mut() {
        let session_id = app.session_id.clone();
        if let Err(e) = dc
            .send_request(
                "turn.send",
                serde_json::json!({
                    "session_id": session_id,
                    "text": text,
                }),
            )
            .await
        {
            app.push_entry(EntryKind::SystemInfo, format!("Daemon send error: {}", e));
            app.agent_busy = false;
            app.turn_start = None;
        }
    } else {
        // Inject effort hint (persistent — applied on every message send).
        let text = if let Some(hint) = commands::effort_instruction(&app.effort_level) {
            format!("<effort: {}>\n\n{}", hint, text)
        } else {
            text
        };
        // Inject brief-mode hint (persistent — applied on every message send when enabled).
        let text = if let Some(hint) = commands::brief_instruction(app.brief_mode) {
            format!("<brief: {}>\n\n{}", hint, text)
        } else {
            text
        };
        // Inject plan-mode hint (persistent — applied on every message send when enabled).
        let text = if let Some(hint) = commands::plan_instruction(app.plan_mode) {
            format!("<plan_mode: {}>\n\n{}", hint, text)
        } else {
            text
        };
        // Inject pinned file contents (persistent across all messages in this session).
        let text = if !app.pinned_files.is_empty() {
            let mut parts: Vec<String> = app
                .pinned_files
                .iter()
                .filter_map(|path| {
                    std::fs::read_to_string(path).ok().map(|content| {
                        format!("<pinned file: {}>\n{}\n</pinned file>", path, content)
                    })
                })
                .collect();
            parts.push(text);
            parts.join("\n\n")
        } else {
            text
        };
        // Auto-detect file paths mentioned in the message and inject contents.
        let text = {
            let detected = extract_file_paths(&text);
            let pinned_set: std::collections::HashSet<&str> =
                app.pinned_files.iter().map(|s| s.as_str()).collect();
            let new_files: Vec<&String> = detected
                .iter()
                .filter(|p| !pinned_set.contains(p.as_str()))
                .collect();

            if new_files.is_empty() {
                text
            } else {
                let mut budget = 8_000usize;
                let mut parts = Vec::new();
                for path in &new_files {
                    if budget == 0 {
                        break;
                    }
                    if let Ok(content) = std::fs::read_to_string(path.as_str()) {
                        let truncated = crate::util::safe_truncate(&content, budget);
                        parts.push(format!(
                            "<auto-context file=\"{}\">\n{}\n</auto-context>",
                            path, truncated
                        ));
                        budget = budget.saturating_sub(truncated.len());
                    }
                }
                if parts.is_empty() {
                    text
                } else {
                    format!("{}\n\n{}", parts.join("\n\n"), text)
                }
            }
        };
        // One-shot context (e.g. git diff, review prompt).
        let text = if let Some(ctx) = app.pending_context.take() {
            format!("{}\n\n{}", ctx, text)
        } else {
            text
        };
        let payload = if !app.pending_images.is_empty() {
            let imgs_json = serde_json::to_string(&app.pending_images).unwrap_or_default();
            app.clear_images();
            format!("\x00IMAGES\x00{}\x00END\x00{}", imgs_json, text)
        } else {
            text
        };
        user_tx.send(payload).await.ok();
    }
}

/// Return all slash command names from `candidates` that begin with `prefix`.
///
/// Matching is case-insensitive. This is a pure function, decoupled from App
/// state, so it can be called from tests without spinning up a TUI.
/// Tab-complete node names for chain subcommands that take a node as first arg.
/// Returns true if the input matched the prefix pattern (even if no completions found),
/// false if the input didn't match.
fn complete_chain_node_name(app: &mut App, input_prefix: &str, output_prefix: &str) -> bool {
    if !app.input.starts_with(input_prefix) {
        return false;
    }
    let after = &app.input[input_prefix.len()..];
    // Only complete the first word (node name) — don't interfere with later args
    if after.contains(' ') {
        return true;
    }
    let node_names: Vec<String> = app
        .chain_state
        .as_ref()
        .map(|cs| cs.config.nodes.iter().map(|n| n.name.clone()).collect())
        .unwrap_or_default();
    if node_names.is_empty() {
        return true;
    }
    let lower = after.to_lowercase();
    let matches: Vec<&String> = node_names
        .iter()
        .filter(|n| n.to_lowercase().starts_with(&lower))
        .collect();
    match matches.len() {
        0 => {}
        1 => {
            app.input = format!("{}{}", output_prefix, matches[0]);
            app.cursor = app.input.len();
        }
        _ => {
            let hints: Vec<String> = matches.iter().map(|n| n.to_string()).collect();
            app.push_entry(EntryKind::SystemInfo, format!("Tab: {}", hints.join("  ")));
        }
    }
    true
}

pub fn slash_completions<'a>(prefix: &str, candidates: &'a [&'a str]) -> Vec<&'a str> {
    let lower = prefix.to_lowercase();
    candidates
        .iter()
        .copied()
        .filter(|c| c.to_lowercase().starts_with(&lower))
        .collect()
}

/// Case-insensitive prefix match against the cached Ollama model list.
/// Mirrors `slash_completions`'s semantics so `/model <Tab>` behaves
/// consistently with `/<Tab>` — the user should never have to remember
/// which completion paths are case-sensitive. Kept separate from
/// `slash_completions` because models arrive as `Vec<String>` (no static
/// lifetime) and future divergence is likely (tag-aware matching).
pub fn model_completions<'a>(prefix: &str, models: &'a [String]) -> Vec<&'a str> {
    let lower = prefix.to_lowercase();
    models
        .iter()
        .map(|m| m.as_str())
        .filter(|m| m.to_lowercase().starts_with(&lower))
        .collect()
}

fn complete_file_path(partial: &str) -> Vec<String> {
    let path = std::path::Path::new(partial);
    let (dir, prefix) = if partial.ends_with('/')
        || partial.ends_with(std::path::MAIN_SEPARATOR)
        || path.is_dir()
    {
        (std::path::PathBuf::from(partial), String::new())
    } else {
        let dir = path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        let prefix = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        (dir, prefix)
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let lower_prefix = prefix.to_lowercase();
    let mut matches: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }
        if !lower_prefix.is_empty() && !name_str.to_lowercase().starts_with(&lower_prefix) {
            continue;
        }
        let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
        let full = if dir == std::path::Path::new(".") {
            if is_dir {
                format!("{}/", name_str)
            } else {
                name_str.to_string()
            }
        } else {
            let base = dir.join(&*name_str);
            if is_dir {
                format!("{}/", base.display())
            } else {
                base.display().to_string()
            }
        };
        matches.push(full);
    }
    matches.sort();
    matches
}

fn longest_common_prefix(strings: &[String]) -> String {
    if strings.is_empty() {
        return String::new();
    }
    let first = &strings[0];
    let mut len = first.len();
    for s in &strings[1..] {
        len = len.min(s.len());
        for (i, (a, b)) in first.bytes().zip(s.bytes()).enumerate() {
            if a != b {
                len = len.min(i);
                break;
            }
        }
    }
    first[..len].to_string()
}

pub fn discover_chain_configs(prefix: &str) -> Vec<String> {
    let lower = prefix.to_lowercase();
    let mut results = Vec::new();

    let scan_dir = |dir: &std::path::Path, results: &mut Vec<String>, use_full_path: bool| {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                    let display = if use_full_path {
                        path.display().to_string()
                    } else {
                        path.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default()
                    };
                    if display.to_lowercase().starts_with(&lower) {
                        results.push(display);
                    }
                }
            }
        }
    };

    if let Some(home) = dirs::home_dir() {
        let chains_dir = home.join(".dm").join("chains");
        scan_dir(&chains_dir, &mut results, true);
    }

    scan_dir(std::path::Path::new("."), &mut results, false);

    results.sort();
    results.dedup();
    results
}

pub fn handle_slash_result(result: commands::SlashResult, app: &mut App) -> anyhow::Result<()> {
    match result {
        commands::SlashResult::Done => {}
        commands::SlashResult::Info(msg) => {
            app.push_entry(EntryKind::SystemInfo, msg);
        }
        commands::SlashResult::Error(msg) => {
            app.push_entry(EntryKind::SystemInfo, msg);
        }
        commands::SlashResult::EditInEditor { path } => {
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            crossterm::terminal::disable_raw_mode()?;
            std::process::Command::new(&editor).arg(&path).status().ok();
            crossterm::terminal::enable_raw_mode()?;
            app.push_entry(
                EntryKind::SystemInfo,
                format!("Opened {} in {}.", path.display(), editor),
            );
        }
    }
    Ok(())
}

const TEXT_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "go", "c", "h", "cpp", "hpp", "java", "rb", "toml",
    "yaml", "yml", "json", "md", "txt", "html", "css", "sh", "bash", "zsh", "sql", "xml", "proto",
    "graphql", "cfg", "ini", "conf", "lua", "zig", "swift", "kt", "scala", "ex", "exs", "erl",
    "hs", "ml", "svelte", "vue",
];

/// Extract file paths mentioned in user text that exist on disk.
/// Strips trailing punctuation and backticks. Skips URLs and binary extensions.
fn extract_file_paths(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for token in text.split_whitespace() {
        let token = token.trim_matches(|c: char| matches!(c, '`' | '\'' | '"'));
        let token = token.trim_end_matches(&[',', '.', ':', ';', ')', ']'] as &[char]);

        if token.starts_with("http://")
            || token.starts_with("https://")
            || token.starts_with("ftp://")
        {
            continue;
        }

        if !token.contains('/') && !token.starts_with("./") && !token.contains('.') {
            continue;
        }

        let path = std::path::Path::new(token);
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => continue,
        };

        if !TEXT_EXTENSIONS.contains(&ext.as_str()) {
            continue;
        }

        if !path.is_file() {
            continue;
        }

        let canonical = token.to_string();
        if seen.insert(canonical.clone()) {
            result.push(canonical);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[&str] = &[
        "commit", "compact", "context", "copy", "clear", "chain", "diff",
    ];

    #[test]
    fn slash_completions_empty_prefix_returns_all() {
        let result = slash_completions("", SAMPLE);
        assert_eq!(
            result.len(),
            SAMPLE.len(),
            "empty prefix should return everything"
        );
    }

    #[test]
    fn slash_completions_prefix_filters_correctly() {
        // "co" matches commit, compact, context, copy — NOT clear or chain or diff
        let result = slash_completions("co", SAMPLE);
        assert_eq!(
            result.len(),
            4,
            "co should match commit, compact, context, copy: {:?}",
            result
        );
        assert!(result.contains(&"commit"));
        assert!(result.contains(&"compact"));
        assert!(result.contains(&"context"));
        assert!(result.contains(&"copy"));
        assert!(!result.contains(&"clear"));
        assert!(!result.contains(&"chain"));
    }

    #[test]
    fn slash_completions_exact_match_returns_one() {
        let result = slash_completions("commit", SAMPLE);
        assert_eq!(result, vec!["commit"]);
    }

    #[test]
    fn slash_completions_no_match_returns_empty() {
        let result = slash_completions("xyz", SAMPLE);
        assert!(result.is_empty(), "no command starts with 'xyz'");
    }

    #[test]
    fn slash_completions_case_insensitive() {
        // "CO" should match the same as "co"
        let lower = slash_completions("co", SAMPLE);
        let upper = slash_completions("CO", SAMPLE);
        assert_eq!(lower, upper, "matching should be case-insensitive");
    }

    #[test]
    fn slash_command_names_contains_expected_commands() {
        let names = commands::SLASH_COMMAND_NAMES;
        for required in &[
            "help", "clear", "commit", "todo", "pr", "summary", "context", "review",
            // commands added in recent cycles
            "add-dir", "pin", "unpin", "effort", "new",
            // commands that were previously missing from the list
            "brief", "files",
        ] {
            assert!(
                names.contains(required),
                "SLASH_COMMAND_NAMES missing: {}",
                required
            );
        }
    }

    #[test]
    fn slash_completions_full_list_single_letter_h() {
        // "help" and "history" start with "h"
        let result = slash_completions("h", commands::SLASH_COMMAND_NAMES);
        assert_eq!(
            result,
            vec!["help", "history"],
            "expected help + history: {:?}",
            result
        );
    }

    // Cycle 10: `/wi` must complete uniquely to `/wiki`. Locks the acceptance
    // criterion from the builder's report ("typing `/wi` + Tab should complete
    // to `/wiki`") so a future addition of another w-prefix command doesn't
    // silently change completion without the team noticing.
    #[test]
    fn slash_completions_wi_prefix_yields_wiki() {
        let result = slash_completions("wi", commands::SLASH_COMMAND_NAMES);
        assert_eq!(
            result,
            vec!["wiki"],
            "'wi' should uniquely complete to 'wiki': {:?}",
            result
        );
    }

    #[test]
    fn slash_completions_preserves_candidate_order() {
        let candidates = &["alpha", "beta", "almond", "gamma"];
        let result = slash_completions("al", candidates);
        // "alpha" comes before "almond" because that's the order in candidates
        assert_eq!(result, vec!["alpha", "almond"]);
    }

    #[test]
    fn slash_completions_empty_candidates_returns_empty() {
        let result = slash_completions("any", &[]);
        assert!(result.is_empty(), "no candidates → empty result");
    }

    #[test]
    fn model_completions_exact_prefix_one_match() {
        let models = vec!["gemma3:4b".to_string(), "qwen2.5-coder:7b".to_string()];
        let matches = model_completions("gem", &models);
        assert_eq!(matches, vec!["gemma3:4b"]);
    }

    #[test]
    fn model_completions_multiple_matches_preserve_order() {
        let models = vec![
            "gemma3:12b".to_string(),
            "gemma3:4b".to_string(),
            "qwen2.5-coder:7b".to_string(),
        ];
        let matches = model_completions("gemma", &models);
        assert_eq!(matches, vec!["gemma3:12b", "gemma3:4b"]);
    }

    #[test]
    fn model_completions_case_insensitive() {
        // `/model QW<Tab>` must find `qwen2.5-coder` — typing case should
        // never be a reason completion fails.
        let models = vec!["qwen2.5-coder:7b".to_string(), "llama3:8b".to_string()];
        let matches = model_completions("QW", &models);
        assert_eq!(matches, vec!["qwen2.5-coder:7b"]);
    }

    #[test]
    fn model_completions_empty_prefix_returns_all() {
        // `/model <Tab>` (space, nothing after) should show the whole list
        // as a hint — same semantics as `slash_completions("")`.
        let models = vec!["a:1".to_string(), "b:2".to_string()];
        let matches = model_completions("", &models);
        assert_eq!(matches, vec!["a:1", "b:2"]);
    }

    #[test]
    fn model_completions_no_match_returns_empty() {
        let models = vec!["gemma3:4b".to_string()];
        let matches: Vec<&str> = model_completions("xyz", &models);
        assert!(matches.is_empty());
    }

    #[test]
    fn model_completions_empty_cache_returns_empty() {
        let models: Vec<String> = Vec::new();
        let matches: Vec<&str> = model_completions("anything", &models);
        assert!(matches.is_empty());
    }

    #[test]
    fn model_completions_colon_in_prefix_matches_tag() {
        // Model names use `:` as the version separator. Tab-completing past
        // the colon must still work (`/model gemma3:<Tab>` → pick a tag).
        let models = vec![
            "gemma3:4b".to_string(),
            "gemma3:12b".to_string(),
            "qwen2.5-coder:7b".to_string(),
        ];
        let matches = model_completions("gemma3:", &models);
        assert_eq!(matches, vec!["gemma3:4b", "gemma3:12b"]);
    }

    #[test]
    fn chain_subcommand_single_match() {
        let result = slash_completions("star", commands::CHAIN_SUBCOMMAND_NAMES);
        assert_eq!(result, vec!["start"], "star should uniquely match start");
    }

    #[test]
    fn chain_subcommand_multiple_matches() {
        let result = slash_completions("s", commands::CHAIN_SUBCOMMAND_NAMES);
        assert_eq!(
            result,
            vec!["start", "status", "stop"],
            "s should match start, status, stop"
        );
    }

    #[test]
    fn chain_subcommand_no_match() {
        let result = slash_completions("xyz", commands::CHAIN_SUBCOMMAND_NAMES);
        assert!(result.is_empty(), "xyz should match nothing");
    }

    #[test]
    fn chain_subcommand_empty_prefix_returns_all() {
        let result = slash_completions("", commands::CHAIN_SUBCOMMAND_NAMES);
        assert_eq!(result.len(), commands::CHAIN_SUBCOMMAND_NAMES.len());
    }

    #[test]
    fn chain_subcommand_resume_variants() {
        let result = slash_completions("resume", commands::CHAIN_SUBCOMMAND_NAMES);
        assert_eq!(result, vec!["resume", "resume-from"]);
    }

    #[test]
    fn discover_chain_configs_finds_yaml_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("build.yaml"), "name: build").unwrap();
        std::fs::write(dir.path().join("test.yml"), "name: test").unwrap();
        std::fs::write(dir.path().join("readme.md"), "not a config").unwrap();

        let mut results = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir.path()) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                    results.push(path.file_name().unwrap().to_string_lossy().to_string());
                }
            }
        }
        results.sort();
        assert_eq!(results, vec!["build.yaml", "test.yml"]);
    }

    #[test]
    fn discover_chain_configs_prefix_filters() {
        let candidates = ["build.yaml", "build-ci.yml", "test.yaml"];
        let lower = "build".to_lowercase();
        let filtered: Vec<&&str> = candidates
            .iter()
            .filter(|c| c.to_lowercase().starts_with(&lower))
            .collect();
        assert_eq!(filtered, vec![&"build.yaml", &"build-ci.yml"]);
    }

    #[test]
    fn discover_chain_configs_empty_prefix_matches_all() {
        let candidates = ["a.yaml", "b.yml", "c.yaml"];
        let lower = "".to_lowercase();
        let filtered: Vec<&&str> = candidates
            .iter()
            .filter(|c| c.to_lowercase().starts_with(&lower))
            .collect();
        assert_eq!(filtered.len(), 3);
    }

    fn make_app_with_chain() -> App {
        use crate::orchestrate::types::*;
        let config = ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "n1".into(),
                    name: "planner".into(),
                    role: "planner".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n2".into(),
                    name: "builder".into(),
                    role: "builder".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
                ChainNodeConfig {
                    id: "n3".into(),
                    name: "tester".into(),
                    role: "tester".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 0,
                    timeout_secs: 60,
                    max_tool_turns: 10,
                },
            ],
            max_cycles: 5,
            max_total_turns: 100,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: false,
            loop_forever: false,
            directive: None,
        };
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        app.chain_state = Some(ChainState::new(config, "test-chain".into()));
        app
    }

    #[test]
    fn complete_chain_node_name_single_match() {
        let mut app = make_app_with_chain();
        app.input = "/chain remove pl".into();
        app.cursor = app.input.len();
        let matched = complete_chain_node_name(&mut app, "/chain remove ", "/chain remove ");
        assert!(matched);
        assert_eq!(app.input, "/chain remove planner");
    }

    #[test]
    fn complete_chain_node_name_multiple_matches() {
        let mut app = make_app_with_chain();
        // "b" and "t" each only match one, but empty prefix matches all three
        app.input = "/chain talk ".into();
        app.cursor = app.input.len();
        let matched = complete_chain_node_name(&mut app, "/chain talk ", "/chain talk ");
        assert!(matched);
        // With empty prefix, should show all 3 as hints (multiple matches path)
        assert!(app
            .entries
            .iter()
            .any(|e| e.content.contains("planner") && e.content.contains("builder")));
    }

    #[test]
    fn complete_chain_node_name_no_match() {
        let mut app = make_app_with_chain();
        app.input = "/chain model xyz".into();
        app.cursor = app.input.len();
        let matched = complete_chain_node_name(&mut app, "/chain model ", "/chain model ");
        assert!(matched);
        assert_eq!(
            app.input, "/chain model xyz",
            "input should be unchanged on no match"
        );
    }

    #[test]
    fn complete_chain_node_name_wrong_prefix() {
        let mut app = make_app_with_chain();
        app.input = "/chain start foo".into();
        let matched = complete_chain_node_name(&mut app, "/chain remove ", "/chain remove ");
        assert!(!matched, "should not match wrong subcommand");
    }

    #[test]
    fn complete_chain_node_name_skips_after_space() {
        let mut app = make_app_with_chain();
        app.input = "/chain talk planner hello".into();
        app.cursor = app.input.len();
        let matched = complete_chain_node_name(&mut app, "/chain talk ", "/chain talk ");
        assert!(matched, "should match prefix");
        assert_eq!(
            app.input, "/chain talk planner hello",
            "should not modify after first arg"
        );
    }

    #[test]
    fn complete_chain_node_name_no_chain_state() {
        let mut app = App::new(
            "m".into(),
            "h".into(),
            "s".into(),
            std::path::PathBuf::from("/tmp"),
            vec![],
        );
        app.input = "/chain remove pl".into();
        let matched = complete_chain_node_name(&mut app, "/chain remove ", "/chain remove ");
        assert!(matched, "should match prefix even without chain state");
        assert_eq!(
            app.input, "/chain remove pl",
            "should not modify without nodes"
        );
    }

    // ── extract_file_paths ──────────────────────────────────────────────────

    #[test]
    fn extract_finds_existing_rust_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("main.rs");
        std::fs::write(&file, "fn main() {}").unwrap();
        let text = format!("look at {} for the entry point", file.display());
        let paths = extract_file_paths(&text);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], file.to_str().unwrap());
    }

    #[test]
    fn extract_strips_trailing_punctuation() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("lib.rs");
        std::fs::write(&file, "pub mod foo;").unwrap();
        let text = format!("check {},", file.display());
        let paths = extract_file_paths(&text);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn extract_handles_backtick_wrapped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("app.ts");
        std::fs::write(&file, "export {}").unwrap();
        let text = format!("the file `{}` has an issue", file.display());
        let paths = extract_file_paths(&text);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn extract_deduplicates() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("foo.rs");
        std::fs::write(&file, "x").unwrap();
        let p = file.to_str().unwrap();
        let text = format!("{} and {} again", p, p);
        let paths = extract_file_paths(&text);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn extract_skips_urls() {
        let paths = extract_file_paths("see https://example.com/path.html for docs");
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_skips_binary_extensions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("image.png");
        std::fs::write(&file, b"\x89PNG").unwrap();
        let text = format!("open {}", file.display());
        let paths = extract_file_paths(&text);
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_empty_for_no_paths() {
        let paths = extract_file_paths("hello world no files here");
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_skips_nonexistent_files() {
        let paths = extract_file_paths("look at /nonexistent/foo.rs");
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_no_extension_skipped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("Makefile");
        std::fs::write(&file, "all:").unwrap();
        let text = format!("check {}", file.display());
        let paths = extract_file_paths(&text);
        assert!(
            paths.is_empty(),
            "files without extensions should be skipped"
        );
    }

    // ── file path completion tests ──────────────────────────────────────

    #[test]
    fn complete_file_path_finds_src_dir() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let orig = std::env::current_dir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("main.rs"), "").unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let matches = complete_file_path("src");
        std::env::set_current_dir(&orig).unwrap();
        assert!(
            !matches.is_empty(),
            "should list src/ contents: {:?}",
            matches
        );
        assert!(
            matches.iter().any(|m| m.starts_with("src/")),
            "entries should have src/ prefix: {:?}",
            matches
        );
    }

    #[test]
    fn complete_file_path_lists_dir_contents() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let orig = std::env::current_dir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("main.rs"), "").unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let matches = complete_file_path("src/");
        std::env::set_current_dir(&orig).unwrap();
        assert!(!matches.is_empty(), "src/ should have contents");
        assert!(
            matches.iter().any(|m| m.contains("main")),
            "should find main.rs in src/: {:?}",
            matches
        );
    }

    #[test]
    fn complete_file_path_filters_by_prefix() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let orig = std::env::current_dir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir(&src).unwrap();
        std::fs::write(src.join("main.rs"), "").unwrap();
        std::fs::write(src.join("mod.rs"), "").unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let matches = complete_file_path("src/ma");
        std::env::set_current_dir(&orig).unwrap();
        assert!(
            matches.iter().any(|m| m.contains("main")),
            "should match main.rs: {:?}",
            matches
        );
        assert!(matches.len() < 10, "should be filtered: {:?}", matches);
    }

    #[test]
    fn complete_file_path_nonexistent_returns_empty() {
        let matches = complete_file_path("/nonexistent_xyz_123/");
        assert!(matches.is_empty());
    }

    #[test]
    fn longest_common_prefix_basic() {
        let strings = vec!["src/main.rs".to_string(), "src/mod.rs".to_string()];
        assert_eq!(longest_common_prefix(&strings), "src/m");
    }

    #[test]
    fn longest_common_prefix_single() {
        let strings = vec!["hello".to_string()];
        assert_eq!(longest_common_prefix(&strings), "hello");
    }

    #[test]
    fn longest_common_prefix_empty() {
        let strings: Vec<String> = Vec::new();
        assert_eq!(longest_common_prefix(&strings), "");
    }

    #[test]
    fn scroll_position_percentage_calculation() {
        let max_scroll: u16 = 200;
        let scroll: u16 = 100;
        let pct = (scroll as u32 * 100 / max_scroll as u32).min(100);
        assert_eq!(pct, 50);

        let scroll_top: u16 = 0;
        let pct = (scroll_top as u32 * 100 / max_scroll as u32).min(100);
        assert_eq!(pct, 0);

        let scroll_bottom: u16 = 200;
        let pct = (scroll_bottom as u32 * 100 / max_scroll as u32).min(100);
        assert_eq!(pct, 100);
    }

    #[test]
    fn scroll_position_zero_max() {
        let max_scroll: u16 = 0;
        let scroll_bottom: u32 = 0;
        let pct: u32 = if max_scroll > 0 {
            (scroll_bottom * 100 / max_scroll as u32).min(100)
        } else {
            100
        };
        assert_eq!(pct, 100);
    }
}
