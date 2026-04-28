use crate::mcp::client::McpClient;
use crate::ollama::types::ToolCall;
use crate::permissions::engine::PermissionEngine;
use crate::permissions::{Behavior, Decision, Rule};
use crate::session::{storage as session_storage, Session};
use crate::tools::registry::ToolRegistry;
use crate::tui::{BackendEvent, PermissionDecision};
use futures_util::future::join_all;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// Handle one round of tool calls: permission checks, concurrent execution,
/// result collection, event emission, and message append.
///
/// Returns `true` if the turn was cancelled (Cancelled event already sent),
/// `false` on normal completion.
#[allow(clippy::too_many_arguments)]
pub async fn execute_tool_round(
    tool_calls: &[ToolCall],
    registry: &ToolRegistry,
    mcp_clients: &HashMap<String, Arc<Mutex<McpClient>>>,
    hooks_config: &crate::tools::hooks::HooksConfig,
    engine: &mut PermissionEngine,
    staging: bool,
    changeset: &mut Vec<crate::changeset::PendingChange>,
    event_tx: &mpsc::Sender<BackendEvent>,
    verbose: bool,
    active_model_name: &str,
    messages: &mut Vec<Value>,
    cancel_rx: &tokio::sync::watch::Receiver<bool>,
    config_dir: &Path,
    settings_dir: &Path,
    session: &mut Session,
    plan_mode: bool,
) -> bool {
    // Cap tool calls per turn to prevent runaway loops.
    let effective_calls: &[ToolCall] = if tool_calls.len() > crate::conversation::MAX_TOOLS_PER_TURN
    {
        let _ = event_tx
            .send(BackendEvent::StreamToken(format!(
                "\n[warning] Model requested {} tool calls, capping at {}\n",
                tool_calls.len(),
                crate::conversation::MAX_TOOLS_PER_TURN
            )))
            .await;
        &tool_calls[..crate::conversation::MAX_TOOLS_PER_TURN]
    } else {
        tool_calls
    };

    // 1. Handle ask_user_question calls first (must be sequential)
    let mut remaining_calls: Vec<&ToolCall> = Vec::new();
    for tool_call in effective_calls {
        let name = &tool_call.function.name;
        let args = &tool_call.function.arguments;

        if name == "ask_user_question" {
            let decision = engine.check(name, args);
            if matches!(decision, Decision::Deny) {
                messages.push(
                    json!({"role":"tool","name":name,"content":"Denied by rule.","is_error":true}),
                );
                continue;
            }
            let question = args["question"].as_str().unwrap_or("").to_string();
            let options: Vec<String> = args["options"]
                .as_array()
                .map(|v| {
                    v.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
            event_tx
                .send(BackendEvent::AskUserQuestion {
                    question,
                    options,
                    reply: reply_tx,
                })
                .await
                .ok();
            let answer =
                match tokio::time::timeout(std::time::Duration::from_secs(120), reply_rx).await {
                    Ok(Ok(answer)) => answer,
                    Ok(Err(_)) => "(user interaction cancelled)".to_string(),
                    Err(_) => "(user prompt timed out after 120s)".to_string(),
                };
            messages.push(json!({"role":"tool","name":name,"content":answer}));
        } else {
            remaining_calls.push(tool_call);
        }
    }

    // 2. Permission checks (sequential — may need user interaction)
    struct ApprovedCall<'a> {
        tool_call: &'a ToolCall,
        allowed: bool,
    }

    let mut approved: Vec<ApprovedCall> = Vec::new();
    for tool_call in &remaining_calls {
        let name = &tool_call.function.name;
        let args = &tool_call.function.arguments;

        let engine_decision = engine.check(name, args);
        // Risky-command floor: if the bash classifier flags this command as
        // dangerous, upgrade Allow → Ask and thread the reason to the UI so
        // the user sees *why* the prompt fired. Explicit Deny still denies;
        // bypass mode still bypasses. Non-bash tools are unaffected.
        let (decision, risk_reason) =
            crate::tools::bash::decision_with_risk(name, args, engine_decision, engine.is_bypass());
        let allowed = match decision {
            Decision::Allow => true,
            Decision::Deny => {
                messages.push(
                    json!({"role":"tool","name":name,"content":"Denied by rule.","is_error":true}),
                );
                false
            }
            Decision::Ask => {
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                event_tx
                    .send(BackendEvent::PermissionRequired {
                        tool_name: name.clone(),
                        args: args.clone(),
                        reason: risk_reason,
                        reply: reply_tx,
                    })
                    .await
                    .ok();

                match reply_rx.await {
                    Ok(PermissionDecision::AllowOnce) => true,
                    Ok(PermissionDecision::AlwaysAllow) => {
                        engine.add_settings_rule(Rule::tool_wide(name, Behavior::Allow));
                        engine.save_settings(settings_dir).ok();
                        true
                    }
                    Ok(PermissionDecision::DenyOnce) => {
                        messages
                            .push(json!({"role":"tool","name":name,"content":"User denied.","is_error":true}));
                        false
                    }
                    Ok(PermissionDecision::AlwaysDeny) => {
                        engine.add_settings_rule(Rule::tool_wide(name, Behavior::Deny));
                        engine.save_settings(settings_dir).ok();
                        messages.push(
                            json!({"role":"tool","name":name,"content":"User permanently denied.","is_error":true}),
                        );
                        false
                    }
                    Err(_) => false,
                }
            }
        };
        approved.push(ApprovedCall { tool_call, allowed });
    }

    // Plan mode: block non-read-only tools
    if plan_mode {
        for ac in &mut approved {
            if ac.allowed && !registry.is_read_only(&ac.tool_call.function.name) {
                let name = &ac.tool_call.function.name;
                messages.push(json!({
                    "role": "tool",
                    "name": name,
                    "content": "Blocked: plan mode is active. Use /plan to exit plan mode before making changes. Focus on reading code and designing your approach.",
                    "is_error": true,
                }));
                ac.allowed = false;
            }
        }
    }

    // Emit ToolStarted for all approved calls
    for ac in &approved {
        if ac.allowed {
            let name = &ac.tool_call.function.name;
            let args = &ac.tool_call.function.arguments;
            event_tx
                .send(BackendEvent::ToolStarted {
                    name: name.clone(),
                    args: args.clone(),
                    active_model: active_model_name.to_string(),
                })
                .await
                .ok();
        }
    }

    // 3. Execute approved tools concurrently
    // Each future returns (name, args, result, elapsed_ms, pre_out, staged_change)
    let execute_futures: Vec<_> = approved
        .iter()
        .filter(|ac| ac.allowed)
        .map(|ac| {
            let name = ac.tool_call.function.name.clone();
            let args = ac.tool_call.function.arguments.clone();
            let registry_ref = registry;
            let mcp_ref = mcp_clients;
            let hooks_ref = hooks_config;
            let stream_tx = event_tx.clone();
            async move {
                // ── Staging intercept ─────────────────────────────────────────
                if staging && (name == "write_file" || name == "edit_file") {
                    let staged = stage_file_change(&name, &args).await;
                    match staged {
                        Ok(change) => {
                            let (added, removed) = change.lines_changed();
                            let result = crate::tools::ToolResult {
                                content: format!(
                                    "Staged: {} (+{} / -{})",
                                    change.path.display(),
                                    added,
                                    removed
                                ),
                                is_error: false,
                            };
                            return (name, args, result, 0u128, None, Some(change));
                        }
                        Err(e) => {
                            return (
                                name,
                                args,
                                crate::tools::ToolResult {
                                    content: format!("Staging error: {}", e),
                                    is_error: true,
                                },
                                0u128,
                                None,
                                None,
                            );
                        }
                    }
                }

                // ── Pre-hook ──────────────────────────────────────────────────
                let pre_out = if let Some(hook) = hooks_ref.hook_for(&name) {
                    if let Some(pre_cmd) = &hook.pre {
                        crate::tools::hooks::run_hook(pre_cmd, &name, &args, None)
                            .await
                            .ok()
                            .filter(|s| !s.is_empty())
                    } else {
                        None
                    }
                } else {
                    None
                };

                let t0 = std::time::Instant::now();

                // ── Bash: streaming execution ─────────────────────────────────
                let result = if name == "bash" {
                    execute_bash_streaming(&name, &args, stream_tx).await
                } else if let Some(server_name) = registry_ref.mcp_server_for(&name) {
                    if let Some(mc) = mcp_ref.get(server_name) {
                        let mut locked = mc.lock().await;
                        match locked.call_tool(&name, args.clone()).await {
                            Ok(content) => crate::tools::ToolResult {
                                content,
                                is_error: false,
                            },
                            Err(e) => crate::tools::ToolResult {
                                content: format!("MCP tool error: {}", e),
                                is_error: true,
                            },
                        }
                    } else {
                        crate::tools::ToolResult {
                            content: format!("MCP server '{}' not connected", server_name),
                            is_error: true,
                        }
                    }
                } else {
                    registry_ref
                        .call(&name, args.clone())
                        .await
                        .unwrap_or_else(|e| crate::tools::ToolResult {
                            content: format!("Tool error: {}", e),
                            is_error: true,
                        })
                };

                let elapsed_ms = t0.elapsed().as_millis();

                if let Some(hook) = hooks_ref.hook_for(&name) {
                    if let Some(post_cmd) = &hook.post {
                        let _ = crate::tools::hooks::run_hook(
                            post_cmd,
                            &name,
                            &args,
                            Some(&result.content),
                        )
                        .await;
                    }
                }

                (
                    name,
                    args,
                    result,
                    elapsed_ms,
                    pre_out,
                    None::<crate::changeset::PendingChange>,
                )
            }
        })
        .collect();

    let results = join_all(execute_futures).await;

    // 4. Collect results; emit events; push to messages
    for (name, _args, result, elapsed_ms, pre_out, staged_change) in results {
        if let Some(change) = staged_change {
            changeset.push(change);
        }

        if let Some(out) = pre_out {
            event_tx
                .send(BackendEvent::StreamToken(format!("[hook:pre] {}\n", out)))
                .await
                .ok();
        }

        event_tx
            .send(BackendEvent::ToolFinished {
                name: name.clone(),
                output: result.content.clone(),
                is_error: result.is_error,
            })
            .await
            .ok();

        if verbose {
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[verbose] Tool '{}' completed in {}ms\n",
                    name, elapsed_ms
                )))
                .await
                .ok();
        }

        // For file_edit (non-staged), extract and forward the diff
        if name == "edit_file" && !result.is_error && !staging {
            if let Some((path, diff)) = extract_diff(&result.content) {
                event_tx
                    .send(BackendEvent::FileDiff { path, diff })
                    .await
                    .ok();
            }
        }

        let tool_content = crate::util::truncate_tool_output(&result.content);
        let was_truncated = tool_content.len() != result.content.len();
        messages.push(json!({
            "role": "tool",
            "name": name,
            "content": tool_content,
            "is_error": result.is_error,
        }));
        if was_truncated {
            event_tx
                .send(BackendEvent::ContextPruned {
                    chars_removed: result.content.len() - crate::util::MAX_TOOL_OUTPUT_CHARS,
                    messages_affected: 1,
                })
                .await
                .ok();
        }

        if *cancel_rx.borrow() {
            event_tx.send(BackendEvent::Cancelled).await.ok();
            session.messages.clone_from(messages);
            session_storage::save(config_dir, session).ok();
            return true; // cancelled
        }
    }

    if tool_calls.len() > crate::conversation::MAX_TOOLS_PER_TURN {
        let notice = format!(
            "Tool call limit reached ({} requested, {} allowed). \
             Remaining calls were skipped. Please make fewer tool calls per turn.",
            tool_calls.len(),
            crate::conversation::MAX_TOOLS_PER_TURN
        );
        messages.push(json!({"role": "tool", "name": "system", "content": notice}));
    }

    session.messages.clone_from(messages);
    session_storage::save(config_dir, session).ok();
    false // not cancelled
}

/// Execute `bash` with line-by-line stdout/stderr streaming via `ToolOutput` events.
/// Delegates to the shared `run_bash()` with a streaming channel, then maps output
/// lines to `BackendEvent::ToolOutput` for the TUI.
async fn execute_bash_streaming(
    name: &str,
    args: &Value,
    event_tx: mpsc::Sender<BackendEvent>,
) -> crate::tools::ToolResult {
    let command = match args["command"].as_str() {
        Some(c) => c.to_string(),
        None => {
            return crate::tools::ToolResult {
                content: "Missing required parameter: command".to_string(),
                is_error: true,
            }
        }
    };
    let cwd_owned = args["cwd"].as_str().map(|s| s.to_string());
    let timeout_ms = args["timeout_ms"].as_u64().unwrap_or(30_000);

    let (line_tx, mut line_rx) = mpsc::channel::<crate::tools::bash::BashOutputLine>(64);
    let tool_name = name.to_string();

    let handle = tokio::spawn(async move {
        crate::tools::bash::run_bash(&command, cwd_owned.as_deref(), timeout_ms, Some(&line_tx))
            .await
    });

    // Forward streaming lines to the TUI event channel
    while let Some(output_line) = line_rx.recv().await {
        let display = if output_line.is_stderr {
            format!("STDERR: {}", output_line.text)
        } else {
            output_line.text
        };
        event_tx
            .send(BackendEvent::ToolOutput {
                name: tool_name.clone(),
                line: display,
            })
            .await
            .ok();
    }

    match handle.await {
        Ok(result) => result,
        Err(e) => crate::tools::ToolResult {
            content: format!("Bash task panicked: {}", e),
            is_error: true,
        },
    }
}

/// Compute a staged change for `write_file` or `edit_file` without touching disk.
pub async fn stage_file_change(
    tool_name: &str,
    args: &Value,
) -> anyhow::Result<crate::changeset::PendingChange> {
    if tool_name == "write_file" {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let proposed = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing content"))?;
        let original = tokio::fs::read_to_string(path_str)
            .await
            .unwrap_or_default();
        Ok(crate::changeset::make_change(
            std::path::PathBuf::from(path_str),
            &original,
            proposed,
        ))
    } else {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing path"))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing old_string"))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing new_string"))?;
        let original = tokio::fs::read_to_string(path_str)
            .await
            .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path_str, e))?;
        let occurrences = original.matches(old_string).count();
        if occurrences == 0 {
            anyhow::bail!("old_string not found in '{}'", path_str);
        }
        if occurrences > 1 {
            anyhow::bail!(
                "old_string found {} times in '{}' — need exactly 1",
                occurrences,
                path_str
            );
        }
        let proposed = original.replacen(old_string, new_string, 1);
        Ok(crate::changeset::make_change(
            std::path::PathBuf::from(path_str),
            &original,
            &proposed,
        ))
    }
}

/// Extract the (path, `diff_text`) from a `file_edit` result string.
/// Result format: "Applied edit to {path}\n\n{diff}"
///
/// Tolerates two optional decorations emitted by `file_edit.rs`:
/// - `scope_note` prefix: `"Note: {path} is outside the project directory.\n"`
/// - `fmt_note` path suffix: `" (formatted)"` when auto-format ran
pub fn extract_diff(content: &str) -> Option<(String, String)> {
    let body = strip_scope_note(content);

    let rest = body.strip_prefix("Applied edit to ")?;
    let nl2 = rest.find("\n\n")?;
    let raw_path = &rest[..nl2];
    let path = raw_path
        .strip_suffix(" (formatted)")
        .unwrap_or(raw_path)
        .to_string();
    let diff = rest[nl2 + 2..].to_string();
    if diff.is_empty() {
        return None;
    }
    Some((path, diff))
}

/// Strip the `"Note: {path} is outside the project directory.\n"` prefix
/// emitted by `file_edit` and `file_write` for out-of-project paths. Returns
/// the original slice when the prefix is absent or the tail anchor doesn't
/// match — so unrelated output starting with `"Note: "` is never swallowed.
pub(crate) fn strip_scope_note(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("Note: ") else {
        return content;
    };
    let Some(nl) = rest.find('\n') else {
        return content;
    };
    if rest[..nl].ends_with(" is outside the project directory.") {
        &rest[nl + 1..]
    } else {
        content
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── stage_file_change tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn stage_write_file_new_file_produces_change() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("new.rs");

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "fn main() {}\n"
        });
        let change = stage_file_change("write_file", &args).await.unwrap();
        assert_eq!(change.path, path);
        assert_eq!(change.proposed, "fn main() {}\n");
        let (added, removed) = change.lines_changed();
        assert!(added > 0, "should have added lines");
        assert_eq!(removed, 0, "no original content to remove");
    }

    #[tokio::test]
    async fn stage_write_file_existing_file_shows_diff() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("existing.rs");
        std::fs::write(&path, "fn old() {}\n").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "fn new() {}\n"
        });
        let change = stage_file_change("write_file", &args).await.unwrap();
        let (added, removed) = change.lines_changed();
        assert!(added > 0);
        assert!(removed > 0);
    }

    #[tokio::test]
    async fn stage_edit_file_replaces_old_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("src.rs");
        std::fs::write(&path, "let x = 1;\nlet y = 2;\n").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "let x = 1;",
            "new_string": "let x = 42;"
        });
        let change = stage_file_change("edit_file", &args).await.unwrap();
        assert!(change.proposed.contains("let x = 42;"));
        assert!(!change.proposed.contains("let x = 1;"));
    }

    #[tokio::test]
    async fn stage_edit_file_old_string_not_found_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("src.rs");
        std::fs::write(&path, "let x = 1;\n").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "let z = 99;",
            "new_string": "let z = 0;"
        });
        let err = stage_file_change("edit_file", &args).await.unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "expected 'not found' error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn stage_edit_file_multiple_matches_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("src.rs");
        std::fs::write(&path, "foo\nfoo\n").unwrap();

        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "foo",
            "new_string": "bar"
        });
        let err = stage_file_change("edit_file", &args).await.unwrap_err();
        assert!(
            err.to_string().contains("found 2 times"),
            "expected ambiguous match error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn stage_write_file_missing_path_errors() {
        let args = serde_json::json!({"content": "hello"});
        let err = stage_file_change("write_file", &args).await.unwrap_err();
        assert!(
            err.to_string().contains("Missing path"),
            "expected 'Missing path', got: {}",
            err
        );
    }

    // ── extract_diff tests ────────────────────────────────────────────────────

    #[test]
    fn extract_diff_parses_valid_content() {
        let content = "Applied edit to src/main.rs\n\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-old\n+new";
        let result = extract_diff(content);
        assert!(result.is_some());
        let (path, diff) = result.unwrap();
        assert_eq!(path, "src/main.rs");
        assert!(diff.contains("--- a/src/main.rs"));
    }

    #[test]
    fn extract_diff_returns_none_for_wrong_prefix() {
        let content = "Modified file src/main.rs\n\ndiff content";
        assert!(extract_diff(content).is_none());
    }

    #[test]
    fn extract_diff_returns_none_when_no_double_newline() {
        let content = "Applied edit to src/main.rs\ndiff content";
        assert!(extract_diff(content).is_none());
    }

    #[test]
    fn extract_diff_returns_none_when_diff_is_empty() {
        let content = "Applied edit to src/main.rs\n\n";
        assert!(extract_diff(content).is_none());
    }

    #[test]
    fn extract_diff_path_with_spaces_preserved() {
        let content = "Applied edit to src/my file.rs\n\n+added line";
        let result = extract_diff(content);
        assert!(result.is_some());
        let (path, _) = result.unwrap();
        assert_eq!(path, "src/my file.rs");
    }

    #[test]
    fn extract_diff_empty_string_returns_none() {
        assert!(extract_diff("").is_none());
    }

    #[test]
    fn extract_diff_only_prefix_returns_none() {
        // No path, no double newline
        assert!(extract_diff("Applied edit to ").is_none());
    }

    #[test]
    fn extract_diff_strips_fmt_note_suffix() {
        // file_edit.rs:144 appends " (formatted)" to the path line when
        // auto-format runs — must not leak into the extracted path.
        let content =
            "Applied edit to src/main.rs (formatted)\n\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let (path, _) = extract_diff(content).unwrap();
        assert_eq!(path, "src/main.rs");
    }

    #[test]
    fn extract_diff_strips_scope_note_prefix() {
        let content =
            "Note: /tmp/x.rs is outside the project directory.\nApplied edit to /tmp/x.rs\n\n-a\n+b\n";
        let (path, diff) = extract_diff(content).unwrap();
        assert_eq!(path, "/tmp/x.rs");
        assert!(diff.contains("+b"));
    }

    #[test]
    fn extract_diff_strips_both_scope_note_and_fmt_note() {
        let content =
            "Note: /tmp/x.rs is outside the project directory.\nApplied edit to /tmp/x.rs (formatted)\n\n-a\n+b\n";
        let (path, _) = extract_diff(content).unwrap();
        assert_eq!(path, "/tmp/x.rs");
    }

    #[test]
    fn extract_diff_malformed_note_line_not_stripped() {
        // Anchors the tail-match discipline: a line beginning with "Note: "
        // that isn't the scope_note must NOT be swallowed — the parser
        // should reject it as a missing "Applied edit to" prefix.
        let content = "Note: something unrelated\nApplied edit to x\n\n+y\n";
        assert!(extract_diff(content).is_none());
    }

    #[test]
    fn tool_round_cap_constant_accessible() {
        const { assert!(crate::conversation::MAX_TOOLS_PER_TURN > 0) };
        const { assert!(crate::conversation::MAX_TOOLS_PER_TURN <= 50) };
    }

    fn test_registry() -> crate::tools::registry::ToolRegistry {
        let mut r = crate::tools::registry::ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        r.register(crate::tools::grep::GrepTool);
        r.register(crate::tools::glob::GlobTool);
        r.register(crate::tools::ls::LsTool);
        r.register(crate::tools::file_edit::FileEditTool);
        r.register(crate::tools::file_write::FileWriteTool);
        r.register(crate::tools::bash::BashTool);
        r.register(crate::tools::multi_edit::MultiEditTool);
        r.register(crate::tools::apply_diff::ApplyDiffTool);
        r
    }

    #[test]
    fn plan_mode_read_tools_are_read_only() {
        let registry = test_registry();
        assert!(
            registry.is_read_only("read_file"),
            "read_file should be read-only"
        );
        assert!(registry.is_read_only("grep"), "grep should be read-only");
        assert!(registry.is_read_only("glob"), "glob should be read-only");
        assert!(registry.is_read_only("ls"), "ls should be read-only");
    }

    #[test]
    fn plan_mode_write_tools_are_not_read_only() {
        let registry = test_registry();
        assert!(
            !registry.is_read_only("file_edit"),
            "file_edit should NOT be read-only"
        );
        assert!(
            !registry.is_read_only("write_file"),
            "write_file should NOT be read-only"
        );
        assert!(
            !registry.is_read_only("bash"),
            "bash should NOT be read-only"
        );
        assert!(
            !registry.is_read_only("multi_edit"),
            "multi_edit should NOT be read-only"
        );
        assert!(
            !registry.is_read_only("apply_diff"),
            "apply_diff should NOT be read-only"
        );
    }

    #[tokio::test]
    async fn ask_user_question_cancelled_returns_error_message() {
        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        drop(tx);
        let answer = match tokio::time::timeout(std::time::Duration::from_secs(1), rx).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => "(user interaction cancelled)".to_string(),
            Err(_) => "(user prompt timed out after 120s)".to_string(),
        };
        assert_eq!(answer, "(user interaction cancelled)");
    }

    #[tokio::test]
    async fn ask_user_question_timeout_returns_error_message() {
        let (_tx, rx) = tokio::sync::oneshot::channel::<String>();
        let answer = match tokio::time::timeout(std::time::Duration::from_millis(50), rx).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => "(user interaction cancelled)".to_string(),
            Err(_) => "(user prompt timed out after 120s)".to_string(),
        };
        assert_eq!(answer, "(user prompt timed out after 120s)");
    }

    #[tokio::test]
    async fn ask_user_question_normal_response() {
        let (tx, rx) = tokio::sync::oneshot::channel::<String>();
        tx.send("yes".to_string()).unwrap();
        let answer = match tokio::time::timeout(std::time::Duration::from_secs(1), rx).await {
            Ok(Ok(answer)) => answer,
            Ok(Err(_)) => "(user interaction cancelled)".to_string(),
            Err(_) => "(user prompt timed out after 120s)".to_string(),
        };
        assert_eq!(answer, "yes");
    }

    #[test]
    fn tool_output_truncation_short_content_unchanged() {
        let content = "short output";
        let result = crate::util::truncate_tool_output(content);
        assert_eq!(result, content);
    }

    #[test]
    fn tool_output_truncation_large_content_capped() {
        let content = "x".repeat(crate::util::MAX_TOOL_OUTPUT_CHARS + 10_000);
        let result = crate::util::truncate_tool_output(&content);
        assert!(result.len() < content.len());
        assert!(result.contains("[output truncated:"));
        assert!(result.contains("50000/60000"));
    }

    #[test]
    fn tool_output_truncation_exact_boundary_unchanged() {
        let content = "a".repeat(crate::util::MAX_TOOL_OUTPUT_CHARS);
        let result = crate::util::truncate_tool_output(&content);
        assert_eq!(result, content);
    }

    #[test]
    fn tool_output_truncation_preserves_utf8() {
        let content = "🔥".repeat(20_000); // 4 bytes each = 80_000 bytes
        let result = crate::util::truncate_tool_output(&content);
        assert!(result.contains("[output truncated:"));
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }
}
