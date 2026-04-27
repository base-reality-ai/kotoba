use crate::tui::BackendEvent;
use serde_json::Value;

/// Convert a `BackendEvent` to a JSON `Value` for transmission to web clients.
///
/// `PermissionRequired` and `AskUserQuestion` carry `oneshot::Sender` fields
/// and cannot be cloned or serialized. In `--serve` mode the permission engine
/// runs with `bypass_all = true`, so these variants are never emitted; they are
/// included here purely as a safety net.
pub fn event_to_json(event: &BackendEvent) -> Value {
    use BackendEvent::*;
    match event {
        StreamThinking(tok) => serde_json::json!({"type": "thinking", "content": tok}),
        StreamToken(tok) => serde_json::json!({"type": "token", "content": tok}),
        StreamDone { content, .. } => serde_json::json!({"type": "done", "content": content}),
        ToolStarted { name, args, .. } => {
            serde_json::json!({"type": "tool_start", "name": name, "args": args})
        }
        ToolFinished {
            name,
            output,
            is_error,
        } => {
            serde_json::json!({"type": "tool_done", "name": name, "output": output, "is_error": is_error})
        }
        FileDiff { path, diff } => {
            serde_json::json!({"type": "diff", "path": path, "diff": diff})
        }
        PermissionRequired { tool_name, .. } => {
            // Should not occur with bypass_all=true; emit a no-op notice.
            serde_json::json!({"type": "permission_required", "tool": tool_name})
        }
        AskUserQuestion { question, .. } => {
            serde_json::json!({"type": "question", "question": question})
        }
        Error(e) => serde_json::json!({"type": "error", "message": e}),
        Notice(m) => serde_json::json!({"type": "notice", "message": m}),
        PermissionsReport(r) => serde_json::json!({"type": "permissions_report", "report": r}),
        TurnComplete {
            prompt_tokens,
            completion_tokens,
        } => {
            serde_json::json!({
                "type": "turn_complete",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })
        }
        Cancelled => serde_json::json!({"type": "cancelled"}),
        TitleGenerated(t) => serde_json::json!({"type": "title", "title": t}),
        ContextWarning(w) => serde_json::json!({"type": "context_warning", "message": w}),
        GpuUpdate {
            util_pct,
            vram_used_mb,
            vram_total_mb,
            temp_c,
        } => {
            serde_json::json!({
                "type": "gpu_update",
                "util_pct": util_pct,
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "temp_c": temp_c,
            })
        }
        ContextUsage { used, limit } => {
            let pct = if *limit > 0 { used * 100 / limit } else { 0 };
            serde_json::json!({"type": "ctx_usage", "used": used, "limit": limit, "pct": pct})
        }
        ContextPruned {
            chars_removed,
            messages_affected,
        } => {
            serde_json::json!({
                "type": "context_pruned",
                "chars_removed": chars_removed,
                "messages_affected": messages_affected,
            })
        }
        // Staged changesets are TUI-only; the web UI doesn't render diffs interactively.
        StagedChangeset(_) => serde_json::json!({"type": "staged_changeset"}),
        ToolOutput { name, line } => {
            serde_json::json!({"type": "tool_output", "name": name, "line": line})
        }
        AgentSpawned {
            prompt_preview,
            depth,
        } => {
            serde_json::json!({"type": "agent_spawned", "prompt_preview": prompt_preview, "depth": depth})
        }
        AgentFinished { depth, elapsed_ms } => {
            serde_json::json!({"type": "agent_finished", "depth": depth, "elapsed_ms": elapsed_ms})
        }
        PerfUpdate {
            tok_per_sec,
            ttft_ms,
            total_tokens,
        } => {
            serde_json::json!({"type": "perf_update", "tok_per_sec": tok_per_sec, "ttft_ms": ttft_ms, "total_tokens": total_tokens})
        }
        TurnStarted => serde_json::json!({"type": "turn_started"}),
        UndoComplete => serde_json::json!({"type": "undo_complete"}),
        NothingToUndo => serde_json::json!({"type": "nothing_to_undo"}),
        ChangesetApplied(changes) => {
            serde_json::json!({"type": "changeset_applied", "files": changes.len()})
        }
        CompactionStarted(_) => serde_json::json!({"type": "compaction_started"}),
        CompactionCompleted => serde_json::json!({"type": "compaction_completed"}),
        SessionSwitched {
            new_id,
            title,
            message_count,
            ..
        } => {
            serde_json::json!({"type": "session_switched", "session_id": new_id, "title": title, "messages": message_count})
        }
    }
}

/// Convert a chain `DaemonEvent` to a JSON `Value` for web bus relay.
/// Returns `None` for non-chain events.
pub fn chain_event_to_json(event: &crate::daemon::protocol::DaemonEvent) -> Option<Value> {
    use crate::daemon::protocol::DaemonEvent;
    match event {
        DaemonEvent::ChainStarted {
            chain_id,
            name,
            node_count,
        } => Some(serde_json::json!({
            "type": "chain_started",
            "chain_id": chain_id,
            "name": name,
            "node_count": node_count,
        })),
        DaemonEvent::ChainNodeTransition {
            chain_id,
            cycle,
            node_name,
            status,
        } => Some(serde_json::json!({
            "type": "chain_node_transition",
            "chain_id": chain_id,
            "cycle": cycle,
            "node_name": node_name,
            "status": status,
        })),
        DaemonEvent::ChainCycleComplete { chain_id, cycle } => Some(serde_json::json!({
            "type": "chain_cycle_complete",
            "chain_id": chain_id,
            "cycle": cycle,
        })),
        DaemonEvent::ChainFinished {
            chain_id,
            success,
            reason,
        } => Some(serde_json::json!({
            "type": "chain_finished",
            "chain_id": chain_id,
            "success": success,
            "reason": reason,
        })),
        DaemonEvent::ChainLog {
            chain_id,
            level,
            message,
        } => Some(serde_json::json!({
            "type": "chain_log",
            "chain_id": chain_id,
            "level": level,
            "message": message,
        })),
        _ => None,
    }
}

/// Check whether a TCP port on 127.0.0.1 is currently available.
pub fn port_available(port: u16) -> bool {
    std::net::TcpListener::bind(std::net::SocketAddr::from(([127, 0, 0, 1], port))).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_to_json_stream_token() {
        let ev = BackendEvent::StreamToken("hello".to_string());
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "token");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_event_to_json_tool_finished_error() {
        let ev = BackendEvent::ToolFinished {
            name: "bash".to_string(),
            output: "oops".to_string(),
            is_error: true,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "tool_done");
        assert_eq!(json["is_error"], true);
        assert_eq!(json["name"], "bash");
    }

    #[test]
    fn test_event_to_json_file_diff() {
        let ev = BackendEvent::FileDiff {
            path: "src/main.rs".to_string(),
            diff: "+added line".to_string(),
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "diff");
        assert_eq!(json["path"], "src/main.rs");
        assert_eq!(json["diff"], "+added line");
    }

    #[test]
    fn test_port_available_occupied() {
        // Bind a listener to occupy the port, then verify port_available returns false.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_available(port), "port should be in use");
        drop(listener);
    }

    #[test]
    fn test_port_available_free() {
        // Bind-then-release and verify port is free. Retry a few times since
        // parallel tests can race for ephemeral ports.
        let mut found_free = false;
        for _ in 0..5 {
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap().port()
            };
            if port_available(port) {
                found_free = true;
                break;
            }
        }
        assert!(found_free, "port should be free after drop (tried 5 times)");
    }

    #[test]
    fn test_event_to_json_turn_complete() {
        let ev = BackendEvent::TurnComplete {
            prompt_tokens: 100,
            completion_tokens: 50,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "turn_complete");
        assert_eq!(json["prompt_tokens"], 100);
        assert_eq!(json["completion_tokens"], 50);
    }

    #[test]
    fn test_event_to_json_ctx_usage_pct() {
        let ev = BackendEvent::ContextUsage {
            used: 50,
            limit: 100,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "ctx_usage");
        assert_eq!(json["pct"], 50);
    }

    #[test]
    fn test_event_to_json_ctx_usage_zero_limit() {
        let ev = BackendEvent::ContextUsage { used: 10, limit: 0 };
        let json = event_to_json(&ev);
        assert_eq!(json["pct"], 0, "zero limit should not divide by zero");
    }

    #[test]
    fn test_event_to_json_error() {
        let ev = BackendEvent::Error("something went wrong".to_string());
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "error");
        assert_eq!(json["message"], "something went wrong");
    }

    #[test]
    fn test_event_to_json_agent_spawned() {
        let ev = BackendEvent::AgentSpawned {
            prompt_preview: "do a thing".to_string(),
            depth: 1,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "agent_spawned");
        assert_eq!(json["depth"], 1);
        assert_eq!(json["prompt_preview"], "do a thing");
    }

    #[test]
    fn test_event_to_json_stream_done() {
        let ev = BackendEvent::StreamDone {
            content: "final text".to_string(),
            tool_calls: vec![],
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "done");
        assert_eq!(json["content"], "final text");
    }

    #[test]
    fn test_event_to_json_tool_started() {
        let ev = BackendEvent::ToolStarted {
            name: "grep".to_string(),
            args: serde_json::json!({"pattern": "fn main"}),
            active_model: "gemma4:26b".to_string(),
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "tool_start");
        assert_eq!(json["name"], "grep");
        assert_eq!(json["args"]["pattern"], "fn main");
    }

    #[test]
    fn test_event_to_json_cancelled() {
        let json = event_to_json(&BackendEvent::Cancelled);
        assert_eq!(json["type"], "cancelled");
    }

    #[test]
    fn test_event_to_json_title_generated() {
        let ev = BackendEvent::TitleGenerated("My Session Title".to_string());
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "title");
        assert_eq!(json["title"], "My Session Title");
    }

    #[test]
    fn test_event_to_json_context_warning() {
        let ev = BackendEvent::ContextWarning("context is getting full".to_string());
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "context_warning");
        assert_eq!(json["message"], "context is getting full");
    }

    #[test]
    fn test_event_to_json_context_pruned() {
        let ev = BackendEvent::ContextPruned {
            chars_removed: 5000,
            messages_affected: 3,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "context_pruned");
        assert_eq!(json["chars_removed"], 5000);
        assert_eq!(json["messages_affected"], 3);
    }

    #[test]
    fn test_event_to_json_agent_finished() {
        let ev = BackendEvent::AgentFinished {
            depth: 2,
            elapsed_ms: 1234,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "agent_finished");
        assert_eq!(json["depth"], 2);
        assert_eq!(json["elapsed_ms"], 1234);
    }

    #[test]
    fn test_event_to_json_perf_update() {
        let ev = BackendEvent::PerfUpdate {
            tok_per_sec: 42.5_f32,
            ttft_ms: 150_u64,
            total_tokens: 512,
        };
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "perf_update");
        assert_eq!(json["total_tokens"], 512);
    }

    #[test]
    fn test_event_to_json_turn_started() {
        let json = event_to_json(&BackendEvent::TurnStarted);
        assert_eq!(json["type"], "turn_started");
    }

    #[test]
    fn test_event_to_json_stream_thinking() {
        let ev = BackendEvent::StreamThinking("reasoning about X".to_string());
        let json = event_to_json(&ev);
        assert_eq!(json["type"], "thinking");
        assert_eq!(json["content"], "reasoning about X");
    }

    // ── chain_event_to_json tests ──────────────────────────────────────────

    #[test]
    fn test_chain_event_to_json_started() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::ChainStarted {
            chain_id: "chain-abc".to_string(),
            name: "build-chain".to_string(),
            node_count: 3,
        };
        let json = chain_event_to_json(&ev).expect("should produce JSON");
        assert_eq!(json["type"], "chain_started");
        assert_eq!(json["chain_id"], "chain-abc");
        assert_eq!(json["name"], "build-chain");
        assert_eq!(json["node_count"], 3);
    }

    #[test]
    fn test_chain_event_to_json_node_transition() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::ChainNodeTransition {
            chain_id: "chain-1".to_string(),
            cycle: 5,
            node_name: "builder".to_string(),
            status: "running".to_string(),
        };
        let json = chain_event_to_json(&ev).expect("should produce JSON");
        assert_eq!(json["type"], "chain_node_transition");
        assert_eq!(json["cycle"], 5);
        assert_eq!(json["node_name"], "builder");
        assert_eq!(json["status"], "running");
    }

    #[test]
    fn test_chain_event_to_json_finished() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::ChainFinished {
            chain_id: "chain-x".to_string(),
            success: true,
            reason: "all cycles complete".to_string(),
        };
        let json = chain_event_to_json(&ev).expect("should produce JSON");
        assert_eq!(json["type"], "chain_finished");
        assert_eq!(json["success"], true);
        assert_eq!(json["reason"], "all cycles complete");
    }

    #[test]
    fn test_chain_event_to_json_non_chain_returns_none() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::Pong;
        assert!(chain_event_to_json(&ev).is_none());
    }

    #[test]
    fn test_chain_event_to_json_log() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::ChainLog {
            chain_id: "chain-log".to_string(),
            level: "info".to_string(),
            message: "node builder started turn 3".to_string(),
        };
        let json = chain_event_to_json(&ev).expect("should produce JSON");
        assert_eq!(json["type"], "chain_log");
        assert_eq!(json["level"], "info");
        assert_eq!(json["message"], "node builder started turn 3");
    }

    #[test]
    fn test_chain_event_to_json_cycle_complete() {
        use crate::daemon::protocol::DaemonEvent;
        let ev = DaemonEvent::ChainCycleComplete {
            chain_id: "chain-cyc".to_string(),
            cycle: 10,
        };
        let json = chain_event_to_json(&ev).expect("should produce JSON");
        assert_eq!(json["type"], "chain_cycle_complete");
        assert_eq!(json["chain_id"], "chain-cyc");
        assert_eq!(json["cycle"], 10);
    }
}
