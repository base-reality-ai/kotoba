//! TUI backend-event handler.
//!
//! Translates `BackendEvent`s emitted by the Ollama stream / daemon /
//! tool dispatch into App state mutations: append thinking tokens,
//! finalize messages, record tool calls, surface errors, update perf
//! stats. Mirrors `tui::input` (which handles user-driven events) on
//! the backend side of the loop.

use crate::tui::{
    app::{App, EntryKind, Mode, PerfStats},
    BackendEvent,
};

pub fn handle_backend(app: &mut App, event: BackendEvent) {
    match event {
        BackendEvent::StreamThinking(tok) => {
            if let Some(partial) = app.streaming_partial.as_mut() {
                partial.push_str(&tok);
            }
        }
        BackendEvent::StreamToken(tok) => {
            if let Some(partial) = app.streaming_partial.as_mut() {
                partial.push_str(&tok);
            }
        }
        BackendEvent::StreamDone {
            content,
            tool_calls,
        } => {
            if tool_calls.is_empty() {
                app.finalize_streaming();
            } else {
                app.streaming_partial = None;
                if !content.is_empty() {
                    app.push_entry(EntryKind::AssistantMessage, content);
                }
            }
        }
        BackendEvent::ToolStarted {
            name,
            args,
            active_model,
        } => {
            app.current_tool = Some(name.clone());
            app.current_tool_model = Some(active_model);
            let repr = match name.as_str() {
                "bash" => args["command"].as_str().unwrap_or("").to_string(),
                "read_file" | "write_file" | "edit_file" => {
                    args["path"].as_str().unwrap_or("").to_string()
                }
                "glob" => args["pattern"].as_str().unwrap_or("").to_string(),
                "grep" => args["pattern"].as_str().unwrap_or("").to_string(),
                _ => String::new(),
            };
            let label = if repr.is_empty() {
                name.clone()
            } else if repr.len() > 40 {
                let mut end = 40usize.min(repr.len());
                while end > 0 && !repr.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}: {}…", name, &repr[..end])
            } else {
                format!("{}: {}", name, repr)
            };
            let activity_kind = match name.as_str() {
                "bash" => crate::tui::app::ActivityKind::BashCommand,
                "edit_file" | "write_file" | "multi_edit" | "apply_diff" => {
                    crate::tui::app::ActivityKind::FileEdit
                }
                _ => crate::tui::app::ActivityKind::ToolCall,
            };
            app.activity_log.push(crate::tui::app::ActivityEntry {
                timestamp: std::time::Instant::now(),
                kind: activity_kind,
                detail: format!("{}: {}", name, repr),
            });
            app.push_entry(EntryKind::ToolCall, label);
        }
        BackendEvent::ToolOutput { name: _, line } => {
            // Accumulate live bash output — rendered in the tool-output panel
            app.current_tool_output.push_back(line);
            // Keep buffer bounded to last 200 lines (O(1) pop vs O(n) remove(0))
            if app.current_tool_output.len() > 200 {
                app.current_tool_output.pop_front();
            }
        }
        BackendEvent::ToolFinished {
            name,
            output,
            is_error,
        } => {
            app.current_tool = None;
            app.current_tool_model = None;
            app.current_tool_output.clear();
            if let Some(rec) =
                crate::tui::agent::edit_summary::parse_tool_output(&name, &output, is_error)
            {
                app.turn_file_edits.push(rec);
            }
            let preview: String = output.lines().take(3).collect::<Vec<_>>().join("\n");
            let label = if preview.is_empty() {
                format!("[{}] {}", name, if is_error { "error" } else { "done" })
            } else {
                format!("[{}] {}", name, preview)
            };
            let kind = if is_error {
                EntryKind::ToolError
            } else {
                EntryKind::ToolResult
            };
            app.push_entry(kind, label);
            app.streaming_partial = Some(String::new());
        }
        BackendEvent::FileDiff { path: _, diff } => {
            app.push_entry(EntryKind::FileDiff, diff);
            app.streaming_partial = Some(String::new());
        }
        BackendEvent::PermissionRequired {
            tool_name,
            args,
            reason,
            reply,
        } => {
            app.pending_permission = Some(crate::tui::app::PendingPermission {
                tool_name,
                args,
                reason,
                reply,
            });
            app.mode = Mode::PermissionDialog;
        }
        BackendEvent::AskUserQuestion {
            question,
            options,
            reply,
        } => {
            app.pending_question = Some(crate::tui::app::PendingQuestion {
                question,
                options,
                reply,
            });
            app.mode = Mode::AskUserQuestion;
        }
        BackendEvent::Error(e) => {
            app.agent_busy = false;
            app.turn_start = None;
            app.streaming_partial = None;
            app.push_entry(EntryKind::SystemInfo, format!("Error: {}", e));
        }
        BackendEvent::Notice(message) => {
            app.push_entry(EntryKind::Notice, message);
        }
        BackendEvent::CompactionStarted(stage) => {
            // No transcript entry — the status bar is the signal. Post-facto
            // compaction summaries already land via ContextPruned / StreamToken.
            app.is_compacting = Some(stage);
        }
        BackendEvent::CompactionCompleted => {
            app.is_compacting = None;
        }
        BackendEvent::TurnComplete {
            prompt_tokens,
            completion_tokens,
        } => {
            if let Some(summary) =
                crate::tui::agent::edit_summary::format_summary(&app.turn_file_edits)
            {
                app.push_entry(EntryKind::SystemInfo, summary);
            }
            app.turn_file_edits.clear();
            app.agent_busy = false;
            app.turn_start = None;
            app.streaming_partial = None;
            app.total_tokens = (prompt_tokens + completion_tokens) as usize;
            app.token_usage
                .record_turn(prompt_tokens, completion_tokens);
            app.activity_log.push(crate::tui::app::ActivityEntry {
                timestamp: std::time::Instant::now(),
                kind: crate::tui::app::ActivityKind::TurnComplete,
                detail: format!("{}+{} tokens", prompt_tokens, completion_tokens),
            });
            if app.bell_on_complete {
                print!("\x07");
            }
        }
        BackendEvent::Cancelled => {
            app.agent_busy = false;
            app.turn_start = None;
            app.streaming_partial = None;
            app.push_entry(EntryKind::SystemInfo, "Turn cancelled.".to_string());
        }
        BackendEvent::PermissionsReport(report) => {
            app.push_entry(EntryKind::SystemInfo, report);
        }
        BackendEvent::TitleGenerated(title) => {
            app.session_title = Some(title);
        }
        BackendEvent::ContextWarning(msg) => {
            app.push_entry(EntryKind::SystemInfo, format!("⚠ {}", msg));
        }
        BackendEvent::GpuUpdate { .. } => {
            // GPU stats are updated via the watch channel in the select! loop; no-op here.
        }
        BackendEvent::ContextUsage { used, limit } => {
            app.ctx_usage = Some((used, limit));
        }
        BackendEvent::ContextPruned {
            chars_removed,
            messages_affected,
        } => {
            app.push_entry(
                EntryKind::SystemInfo,
                format!(
                    "[pruned {} chars from {} tool results]",
                    chars_removed, messages_affected
                ),
            );
        }
        BackendEvent::AgentSpawned {
            prompt_preview,
            depth,
        } => {
            let indent = "  ".repeat(depth as usize);
            app.push_entry(
                EntryKind::SystemInfo,
                format!(
                    "{}⟳ [agent] spawning sub-agent (depth {}): \"{}\"",
                    indent, depth, prompt_preview
                ),
            );
        }
        BackendEvent::AgentFinished { depth, elapsed_ms } => {
            let indent = "  ".repeat(depth as usize);
            app.push_entry(
                EntryKind::SystemInfo,
                format!(
                    "{}✓ [agent] sub-agent done (depth {}, {}ms)",
                    indent, depth, elapsed_ms
                ),
            );
        }
        BackendEvent::PerfUpdate {
            tok_per_sec,
            ttft_ms,
            total_tokens,
        } => {
            app.perf = Some(PerfStats {
                tok_per_sec,
                ttft_ms,
                total_tokens,
            });
        }
        BackendEvent::TurnStarted => {
            app.undo_entries_snapshot = Some(app.entries.clone());
        }
        BackendEvent::UndoComplete => {
            if let Some(snapshot) = app.undo_entries_snapshot.take() {
                app.entries = snapshot;
            }
            app.push_entry(
                EntryKind::SystemInfo,
                "[undo] restored to previous turn".to_string(),
            );
        }
        BackendEvent::NothingToUndo => {
            app.push_entry(EntryKind::SystemInfo, "[undo] nothing to undo".to_string());
        }
        BackendEvent::SessionSwitched {
            old_id,
            new_id,
            new_full_id,
            title,
            message_count,
        } => {
            app.entries.clear();
            app.session_id = new_full_id;
            app.session_title = if title == "(untitled)" {
                None
            } else {
                Some(title.clone())
            };
            app.pending_context = None;
            app.ctx_usage = None;
            app.undo_entries_snapshot = None;
            app.token_usage = crate::tui::app::TokenUsage::default();
            app.push_entry(
                EntryKind::SystemInfo,
                format!(
                    "Switched from {} → {} ({}) — {} messages in context",
                    old_id, new_id, title, message_count
                ),
            );
        }
        BackendEvent::ChangesetApplied(changes) => {
            app.file_undo_history.push(changes);
            if app.file_undo_history.len() > 10 {
                app.file_undo_history.remove(0);
            }
        }
        BackendEvent::StagedChangeset(changes) => {
            let n = changes.len();
            app.diff_file_decisions = vec![None; n];
            app.staged_changes = changes;
            app.diff_review_idx = 0;
            app.diff_scroll = 0;
            app.mode = Mode::DiffReview;
            app.streaming_partial = None;
            app.push_entry(
                EntryKind::SystemInfo,
                format!(
                    "{} staged change{} — [a]pply / [r]eject",
                    n,
                    if n == 1 { "" } else { "s" }
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::BackendEvent;

    fn make_app() -> App {
        App::new(
            "gemma4:26b".into(),
            "localhost:11434".into(),
            "test-session".into(),
            std::path::PathBuf::from("/tmp/dm-events-test"),
            vec![],
        )
    }

    #[test]
    fn stream_token_appends_to_partial() {
        let mut app = make_app();
        app.streaming_partial = Some("hello".into());
        handle_backend(&mut app, BackendEvent::StreamToken(" world".into()));
        assert_eq!(app.streaming_partial.as_deref(), Some("hello world"));
    }

    #[test]
    fn stream_token_noop_when_partial_is_none() {
        let mut app = make_app();
        app.streaming_partial = None;
        handle_backend(&mut app, BackendEvent::StreamToken("tok".into()));
        // partial stays None — token is discarded
        assert!(app.streaming_partial.is_none());
    }

    #[test]
    fn stream_done_no_tools_finalizes_streaming() {
        let mut app = make_app();
        app.streaming_partial = Some("response text".into());
        handle_backend(
            &mut app,
            BackendEvent::StreamDone {
                content: String::new(),
                tool_calls: vec![],
            },
        );
        // finalize_streaming moves partial into entries and clears it
        assert!(app.streaming_partial.is_none());
        assert!(
            app.entries
                .iter()
                .any(|e| e.content.contains("response text")),
            "finalized content should appear in entries"
        );
    }

    #[test]
    fn stream_done_with_tools_pushes_content_entry() {
        let mut app = make_app();
        app.streaming_partial = Some(String::new());
        // tool_calls is non-empty — takes the other branch
        let fake_tool = crate::ollama::types::ToolCall {
            function: crate::ollama::types::FunctionCall {
                name: "bash".into(),
                arguments: serde_json::json!({}),
            },
        };
        handle_backend(
            &mut app,
            BackendEvent::StreamDone {
                content: "pre-tool text".into(),
                tool_calls: vec![fake_tool],
            },
        );
        assert!(app.streaming_partial.is_none(), "partial should be cleared");
        assert!(
            app.entries
                .iter()
                .any(|e| e.kind == EntryKind::AssistantMessage && e.content == "pre-tool text"),
            "non-empty content should be pushed as AssistantMessage"
        );
    }

    #[test]
    fn tool_started_sets_current_tool_and_model() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ToolStarted {
                name: "glob".into(),
                args: serde_json::json!({ "pattern": "*.rs" }),
                active_model: "gemma4:26b".into(),
            },
        );
        // current_tool stores the raw tool name; the label with args goes into entries
        assert_eq!(app.current_tool.as_deref(), Some("glob"));
        assert_eq!(app.current_tool_model.as_deref(), Some("gemma4:26b"));
        // entries should contain the formatted label
        assert!(
            app.entries
                .iter()
                .any(|e| e.kind == EntryKind::ToolCall && e.content.contains("*.rs")),
            "tool call entry should include pattern arg"
        );
    }

    #[test]
    fn tool_started_bash_uses_command_field() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ToolStarted {
                name: "bash".into(),
                args: serde_json::json!({ "command": "cargo build" }),
                active_model: "m".into(),
            },
        );
        // current_tool = raw name; label in entries includes the command
        assert_eq!(app.current_tool.as_deref(), Some("bash"));
        assert!(
            app.entries
                .iter()
                .any(|e| e.kind == EntryKind::ToolCall && e.content.contains("cargo build")),
            "bash tool call entry should include command"
        );
    }

    #[test]
    fn tool_output_accumulates_lines() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ToolOutput {
                name: "bash".into(),
                line: "line1".into(),
            },
        );
        handle_backend(
            &mut app,
            BackendEvent::ToolOutput {
                name: "bash".into(),
                line: "line2".into(),
            },
        );
        assert_eq!(app.current_tool_output.len(), 2);
        assert_eq!(app.current_tool_output[0], "line1");
        assert_eq!(app.current_tool_output[1], "line2");
    }

    #[test]
    fn tool_output_capped_at_200_lines() {
        let mut app = make_app();
        for i in 0..250 {
            handle_backend(
                &mut app,
                BackendEvent::ToolOutput {
                    name: "bash".into(),
                    line: format!("line {}", i),
                },
            );
        }
        assert_eq!(
            app.current_tool_output.len(),
            200,
            "buffer must not exceed 200 lines"
        );
    }

    #[test]
    fn tool_finished_clears_current_tool() {
        let mut app = make_app();
        app.current_tool = Some("bash".into());
        app.current_tool_model = Some("model".into());
        handle_backend(
            &mut app,
            BackendEvent::ToolFinished {
                name: "bash".into(),
                output: "ok".into(),
                is_error: false,
            },
        );
        assert!(app.current_tool.is_none());
        assert!(app.current_tool_model.is_none());
        assert!(app.current_tool_output.is_empty());
    }

    #[test]
    fn tool_finished_error_pushes_tool_error_entry() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ToolFinished {
                name: "bash".into(),
                output: String::new(),
                is_error: true,
            },
        );
        let last = app.entries.last().expect("should have an entry");
        assert_eq!(last.kind, EntryKind::ToolError);
    }

    #[test]
    fn error_clears_agent_busy() {
        let mut app = make_app();
        app.agent_busy = true;
        handle_backend(&mut app, BackendEvent::Error("something went wrong".into()));
        assert!(!app.agent_busy);
        assert!(app
            .entries
            .iter()
            .any(|e| e.content.contains("something went wrong")));
    }

    #[test]
    fn turn_complete_updates_total_tokens_and_clears_busy() {
        let mut app = make_app();
        app.agent_busy = true;
        app.total_tokens = 0;
        handle_backend(
            &mut app,
            BackendEvent::TurnComplete {
                prompt_tokens: 1000,
                completion_tokens: 500,
            },
        );
        assert!(!app.agent_busy);
        assert_eq!(app.total_tokens, 1500);
    }

    #[test]
    fn cancelled_clears_agent_busy() {
        let mut app = make_app();
        app.agent_busy = true;
        handle_backend(&mut app, BackendEvent::Cancelled);
        assert!(!app.agent_busy);
        assert!(app.entries.iter().any(|e| e.content.contains("cancelled")));
    }

    #[test]
    fn context_usage_stored() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ContextUsage {
                used: 2048,
                limit: 8192,
            },
        );
        assert_eq!(app.ctx_usage, Some((2048, 8192)));
    }

    #[test]
    fn title_generated_sets_session_title() {
        let mut app = make_app();
        assert!(app.session_title.is_none());
        handle_backend(&mut app, BackendEvent::TitleGenerated("My Session".into()));
        assert_eq!(app.session_title.as_deref(), Some("My Session"));
    }

    #[test]
    fn perf_update_sets_perf_stats() {
        let mut app = make_app();
        assert!(app.perf.is_none());
        handle_backend(
            &mut app,
            BackendEvent::PerfUpdate {
                tok_per_sec: 42.5,
                ttft_ms: 120,
                total_tokens: 300,
            },
        );
        let p = app.perf.as_ref().expect("perf should be set");
        assert!((p.tok_per_sec - 42.5).abs() < 0.001);
        assert_eq!(p.ttft_ms, 120);
    }

    #[test]
    fn turn_started_captures_entries_snapshot() {
        let mut app = make_app();
        app.push_entry(EntryKind::UserMessage, "hello".into());
        handle_backend(&mut app, BackendEvent::TurnStarted);
        let snap = app
            .undo_entries_snapshot
            .as_ref()
            .expect("snapshot should exist");
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].content, "hello");
    }

    #[test]
    fn undo_complete_restores_entries_from_snapshot() {
        let mut app = make_app();
        app.push_entry(EntryKind::UserMessage, "before".into());
        handle_backend(&mut app, BackendEvent::TurnStarted);
        // Simulate new entries added during the turn
        app.push_entry(EntryKind::AssistantMessage, "new response".into());
        // Now undo
        handle_backend(&mut app, BackendEvent::UndoComplete);
        // entries should be restored to just "before" + the "[undo]" info line
        assert!(
            app.undo_entries_snapshot.is_none(),
            "snapshot consumed after undo"
        );
        assert!(
            app.entries.iter().any(|e| e.content.contains("undo")),
            "undo confirmation message should be pushed"
        );
    }

    #[test]
    fn context_pruned_pushes_info_entry() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::ContextPruned {
                chars_removed: 5000,
                messages_affected: 3,
            },
        );
        let last = app.entries.last().expect("entry should exist");
        assert!(
            last.content.contains("5000"),
            "pruned chars should appear in message"
        );
        assert!(
            last.content.contains('3'),
            "affected messages count should appear"
        );
    }

    #[test]
    fn agent_spawned_pushes_info_entry() {
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::AgentSpawned {
                prompt_preview: "do something".into(),
                depth: 1,
            },
        );
        let last = app.entries.last().expect("entry should exist");
        assert!(last.content.contains("do something"));
        assert!(last.content.contains("depth 1") || last.content.contains('1'));
    }

    #[test]
    fn notice_pushes_notice_entry_without_clearing_busy() {
        // Notice events carry progress text (e.g. retry backoff) while the
        // agent is still working — they must not flip `agent_busy` off.
        let mut app = make_app();
        app.agent_busy = true;
        handle_backend(
            &mut app,
            BackendEvent::Notice("retrying in 1.0s (attempt 1/3)".into()),
        );
        assert!(app.agent_busy, "Notice should not clear agent_busy");
        let last = app.entries.last().expect("entry should exist");
        assert_eq!(last.kind, EntryKind::Notice);
        assert!(last.content.contains("retrying"));
        assert!(last.content.contains("attempt 1/3"));
    }

    #[test]
    fn notice_reconnect_success_text_renders_as_notice_kind() {
        // The retry loop emits a courtesy "✓ Reconnected to Ollama" on
        // success — it must route to EntryKind::Notice so the UI paints
        // it dim/cyan, not red like Error.
        let mut app = make_app();
        handle_backend(
            &mut app,
            BackendEvent::Notice("✓ Reconnected to Ollama".into()),
        );
        let last = app.entries.last().expect("entry should exist");
        assert_eq!(last.kind, EntryKind::Notice);
        assert!(last.content.contains("Reconnected"));
    }

    #[test]
    fn compaction_started_sets_flag_and_completed_clears_it() {
        // Contract: between CompactionStarted and CompactionCompleted the
        // status-bar flag is set; the Completed handler must clear it so the
        // banner doesn't stick around after the pipeline returns. No
        // transcript entry is pushed for either event — the status bar is
        // the sole UI signal.
        let mut app = make_app();
        let before_entries = app.entries.len();
        handle_backend(
            &mut app,
            BackendEvent::CompactionStarted(crate::compaction::CompactionStage::Microcompact {
                chars_removed: 42,
                messages_affected: 3,
            }),
        );
        assert!(
            app.is_compacting.is_some(),
            "CompactionStarted should set is_compacting"
        );
        assert_eq!(
            app.entries.len(),
            before_entries,
            "no transcript entry for CompactionStarted"
        );
        handle_backend(&mut app, BackendEvent::CompactionCompleted);
        assert!(
            app.is_compacting.is_none(),
            "CompactionCompleted should clear is_compacting"
        );
        assert_eq!(
            app.entries.len(),
            before_entries,
            "no transcript entry for CompactionCompleted"
        );
    }

    #[test]
    fn error_clears_turn_start() {
        let mut app = make_app();
        app.agent_busy = true;
        app.turn_start = Some(std::time::Instant::now());
        handle_backend(&mut app, BackendEvent::Error("oops".into()));
        assert!(app.turn_start.is_none());
    }

    #[test]
    fn turn_complete_clears_turn_start() {
        let mut app = make_app();
        app.agent_busy = true;
        app.turn_start = Some(std::time::Instant::now());
        handle_backend(
            &mut app,
            BackendEvent::TurnComplete {
                prompt_tokens: 100,
                completion_tokens: 50,
            },
        );
        assert!(app.turn_start.is_none());
    }

    #[test]
    fn cancelled_clears_turn_start() {
        let mut app = make_app();
        app.agent_busy = true;
        app.turn_start = Some(std::time::Instant::now());
        handle_backend(&mut app, BackendEvent::Cancelled);
        assert!(app.turn_start.is_none());
    }
}
