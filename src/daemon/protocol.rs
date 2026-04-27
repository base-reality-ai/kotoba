use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRequest {
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DaemonEvent {
    StreamToken {
        session_id: String,
        content: String,
    },
    StreamDone {
        session_id: String,
    },
    ToolStarted {
        session_id: String,
        name: String,
        args: serde_json::Value,
        active_model: String,
    },
    ToolFinished {
        session_id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    FileDiff {
        session_id: String,
        path: String,
        diff: String,
    },
    ToolOutput {
        session_id: String,
        name: String,
        line: String,
    },
    AgentSpawned {
        session_id: String,
        prompt_preview: String,
        depth: u8,
    },
    AgentFinished {
        session_id: String,
        depth: u8,
        elapsed_ms: u64,
    },
    TurnComplete {
        session_id: String,
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    Cancelled {
        session_id: String,
    },
    ContextWarning {
        session_id: String,
        message: String,
    },
    ContextUsage {
        session_id: String,
        used: usize,
        limit: usize,
    },
    ContextPruned {
        session_id: String,
        chars_removed: usize,
        messages_affected: usize,
    },
    TitleGenerated {
        session_id: String,
        title: String,
    },
    Error {
        session_id: String,
        message: String,
    },
    PermissionRequired {
        session_id: String,
        tool_name: String,
        args: serde_json::Value,
        request_id: String,
    },
    StagedChangeset {
        session_id: String,
        changes: Vec<serde_json::Value>,
    },
    SessionCreated {
        session_id: String,
    },
    SessionEnded {
        session_id: String,
    },
    SessionCancelled {
        session_id: String,
    },
    SessionList {
        sessions: Vec<SessionInfo>,
    },
    ChainStarted {
        chain_id: String,
        name: String,
        node_count: usize,
    },
    ChainNodeTransition {
        chain_id: String,
        cycle: usize,
        node_name: String,
        status: String,
    },
    ChainCycleComplete {
        chain_id: String,
        cycle: usize,
    },
    ChainFinished {
        chain_id: String,
        success: bool,
        reason: String,
    },
    ChainLog {
        chain_id: String,
        level: String,
        message: String,
    },
    ChainStatus {
        chain_id: String,
        state: serde_json::Value,
    },
    ChainList {
        chains: Vec<ChainInfo>,
    },
    Health {
        uptime_secs: u64,
        session_count: usize,
        pid: u32,
    },
    Pong,
}

impl DaemonEvent {
    /// Convert a [`crate::tui::BackendEvent`] to a [`DaemonEvent`], given a `session_id`.
    ///
    /// Returns `None` for variants that have no useful daemon representation
    /// (e.g. `PermissionRequired` which carries a oneshot channel, `AskUserQuestion`,
    /// `GpuUpdate` which is daemon-local).
    pub fn from_backend(event: &crate::tui::BackendEvent, session_id: &str) -> Option<Self> {
        let sid = session_id.to_string();
        match event {
            crate::tui::BackendEvent::StreamThinking(content) => Some(DaemonEvent::StreamToken {
                session_id: sid,
                content: content.clone(),
            }),
            crate::tui::BackendEvent::StreamToken(content) => Some(DaemonEvent::StreamToken {
                session_id: sid,
                content: content.clone(),
            }),
            crate::tui::BackendEvent::StreamDone { .. } => {
                Some(DaemonEvent::StreamDone { session_id: sid })
            }
            crate::tui::BackendEvent::ToolStarted {
                name,
                args,
                active_model,
            } => Some(DaemonEvent::ToolStarted {
                session_id: sid,
                name: name.clone(),
                args: args.clone(),
                active_model: active_model.clone(),
            }),
            crate::tui::BackendEvent::ToolFinished {
                name,
                output,
                is_error,
            } => Some(DaemonEvent::ToolFinished {
                session_id: sid,
                name: name.clone(),
                output: output.clone(),
                is_error: *is_error,
            }),
            crate::tui::BackendEvent::FileDiff { path, diff } => Some(DaemonEvent::FileDiff {
                session_id: sid,
                path: path.clone(),
                diff: diff.clone(),
            }),
            crate::tui::BackendEvent::TurnComplete {
                prompt_tokens,
                completion_tokens,
            } => Some(DaemonEvent::TurnComplete {
                session_id: sid,
                prompt_tokens: *prompt_tokens,
                completion_tokens: *completion_tokens,
            }),
            crate::tui::BackendEvent::Cancelled => Some(DaemonEvent::Cancelled { session_id: sid }),
            crate::tui::BackendEvent::Error(message) => Some(DaemonEvent::Error {
                session_id: sid,
                message: message.clone(),
            }),
            crate::tui::BackendEvent::Notice(_) => None,
            crate::tui::BackendEvent::ContextWarning(message) => {
                Some(DaemonEvent::ContextWarning {
                    session_id: sid,
                    message: message.clone(),
                })
            }
            crate::tui::BackendEvent::ContextUsage { used, limit } => {
                Some(DaemonEvent::ContextUsage {
                    session_id: sid,
                    used: *used,
                    limit: *limit,
                })
            }
            crate::tui::BackendEvent::ContextPruned {
                chars_removed,
                messages_affected,
            } => Some(DaemonEvent::ContextPruned {
                session_id: sid,
                chars_removed: *chars_removed,
                messages_affected: *messages_affected,
            }),
            crate::tui::BackendEvent::TitleGenerated(title) => Some(DaemonEvent::TitleGenerated {
                session_id: sid,
                title: title.clone(),
            }),
            crate::tui::BackendEvent::StagedChangeset(changes) => {
                let serialized = changes
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "path": c.path.to_string_lossy(),
                            "diff": c.diff,
                        })
                    })
                    .collect();
                Some(DaemonEvent::StagedChangeset {
                    session_id: sid,
                    changes: serialized,
                })
            }
            crate::tui::BackendEvent::ToolOutput { name, line } => Some(DaemonEvent::ToolOutput {
                session_id: session_id.to_string(),
                name: name.clone(),
                line: line.clone(),
            }),
            crate::tui::BackendEvent::AgentSpawned {
                prompt_preview,
                depth,
            } => Some(DaemonEvent::AgentSpawned {
                session_id: session_id.to_string(),
                prompt_preview: prompt_preview.clone(),
                depth: *depth,
            }),
            crate::tui::BackendEvent::AgentFinished { depth, elapsed_ms } => {
                Some(DaemonEvent::AgentFinished {
                    session_id: session_id.to_string(),
                    depth: *depth,
                    elapsed_ms: *elapsed_ms,
                })
            }
            // These carry live oneshot channels or are daemon-local — not serialisable
            crate::tui::BackendEvent::PermissionRequired { .. } => None,
            crate::tui::BackendEvent::AskUserQuestion { .. } => None,
            crate::tui::BackendEvent::GpuUpdate { .. } => None,
            crate::tui::BackendEvent::PermissionsReport(_) => None,
            crate::tui::BackendEvent::PerfUpdate { .. } => None,
            crate::tui::BackendEvent::TurnStarted => None,
            crate::tui::BackendEvent::UndoComplete => None,
            crate::tui::BackendEvent::NothingToUndo => None,
            crate::tui::BackendEvent::ChangesetApplied(_) => None,
            crate::tui::BackendEvent::CompactionStarted(_) => None,
            crate::tui::BackendEvent::CompactionCompleted => None,
            crate::tui::BackendEvent::SessionSwitched { .. } => None,
        }
    }

    /// Convert a [`DaemonEvent`] back to a [`crate::tui::BackendEvent`] for the TUI client.
    ///
    /// Returns `None` for daemon-only events like `SessionCreated`, `SessionList`, `Pong`.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_backend_event(self) -> Option<crate::tui::BackendEvent> {
        match self {
            DaemonEvent::StreamToken { content, .. } => {
                Some(crate::tui::BackendEvent::StreamToken(content))
            }
            DaemonEvent::StreamDone { .. } => Some(crate::tui::BackendEvent::StreamDone {
                content: String::new(),
                tool_calls: vec![],
            }),
            DaemonEvent::ToolStarted {
                name,
                args,
                active_model,
                ..
            } => Some(crate::tui::BackendEvent::ToolStarted {
                name,
                args,
                active_model,
            }),
            DaemonEvent::ToolFinished {
                name,
                output,
                is_error,
                ..
            } => Some(crate::tui::BackendEvent::ToolFinished {
                name,
                output,
                is_error,
            }),
            DaemonEvent::FileDiff { path, diff, .. } => {
                Some(crate::tui::BackendEvent::FileDiff { path, diff })
            }
            DaemonEvent::ToolOutput { name, line, .. } => {
                Some(crate::tui::BackendEvent::ToolOutput { name, line })
            }
            DaemonEvent::AgentSpawned {
                prompt_preview,
                depth,
                ..
            } => Some(crate::tui::BackendEvent::AgentSpawned {
                prompt_preview,
                depth,
            }),
            DaemonEvent::AgentFinished {
                depth, elapsed_ms, ..
            } => Some(crate::tui::BackendEvent::AgentFinished { depth, elapsed_ms }),
            DaemonEvent::TurnComplete {
                prompt_tokens,
                completion_tokens,
                ..
            } => Some(crate::tui::BackendEvent::TurnComplete {
                prompt_tokens,
                completion_tokens,
            }),
            DaemonEvent::Cancelled { .. } => Some(crate::tui::BackendEvent::Cancelled),
            DaemonEvent::Error { message, .. } => Some(crate::tui::BackendEvent::Error(message)),
            DaemonEvent::ContextWarning { message, .. } => {
                Some(crate::tui::BackendEvent::ContextWarning(message))
            }
            DaemonEvent::ContextUsage { used, limit, .. } => {
                Some(crate::tui::BackendEvent::ContextUsage { used, limit })
            }
            DaemonEvent::ContextPruned {
                chars_removed,
                messages_affected,
                ..
            } => Some(crate::tui::BackendEvent::ContextPruned {
                chars_removed,
                messages_affected,
            }),
            DaemonEvent::TitleGenerated { title, .. } => {
                Some(crate::tui::BackendEvent::TitleGenerated(title))
            }
            // StagedChangeset carries raw JSON values; we cannot reconstruct PendingChange
            // without a full deserializer — emit an informational token instead.
            DaemonEvent::StagedChangeset { changes, .. } => {
                Some(crate::tui::BackendEvent::StreamToken(format!(
                    "[daemon] {} staged change(s) pending\n",
                    changes.len()
                )))
            }
            // Daemon-only events have no TUI representation
            DaemonEvent::PermissionRequired { .. } => None,
            DaemonEvent::SessionCreated { .. } => None,
            DaemonEvent::SessionEnded { .. } => None,
            DaemonEvent::SessionCancelled { .. } => None,
            DaemonEvent::SessionList { .. } => None,
            DaemonEvent::ChainStarted { .. } => None,
            DaemonEvent::ChainNodeTransition { .. } => None,
            DaemonEvent::ChainCycleComplete { .. } => None,
            DaemonEvent::ChainFinished { .. } => None,
            DaemonEvent::ChainLog { .. } => None,
            DaemonEvent::ChainStatus { .. } => None,
            DaemonEvent::ChainList { .. } => None,
            DaemonEvent::Health { .. } => None,
            DaemonEvent::Pong => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainInfo {
    pub chain_id: String,
    pub name: String,
    pub current_cycle: usize,
    pub node_count: usize,
    pub status: String, // "running", "paused", "finished"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub created_at: String,
    pub last_active: String,
    pub client_count: usize,
    pub status: String, // "idle", "thinking", "waiting", "staged"
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn backend_event_to_daemon_event_stream_token() {
        let be = crate::tui::BackendEvent::StreamToken("hi".into());
        let de = DaemonEvent::from_backend(&be, "sess-1").expect("should map");
        match de {
            DaemonEvent::StreamToken {
                session_id,
                content,
            } => {
                assert_eq!(session_id, "sess-1");
                assert_eq!(content, "hi");
            }
            other => panic!("expected StreamToken, got {:?}", other),
        }
    }

    #[test]
    fn daemon_event_to_backend_event_roundtrip() {
        let be = crate::tui::BackendEvent::StreamToken("hello".into());
        let de = DaemonEvent::from_backend(&be, "sess-rt").expect("should map");
        let be2 = de.to_backend_event().expect("should roundtrip");
        match be2 {
            crate::tui::BackendEvent::StreamToken(text) => {
                assert_eq!(text, "hello");
            }
            other => panic!("expected StreamToken, got {:?}", other),
        }
    }

    #[test]
    fn request_serializes_session_create() {
        let req = DaemonRequest {
            id: 1,
            method: "session.create".to_string(),
            params: json!({}),
        };
        let serialized = serde_json::to_string(&req).expect("serialize");
        let deserialized: DaemonRequest = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(deserialized.method, "session.create");
        assert_eq!(deserialized.id, 1);
    }

    #[test]
    fn event_deserializes_stream_token() {
        let raw = r#"{"type":"stream_token","session_id":"x","content":"hi"}"#;
        let event: DaemonEvent = serde_json::from_str(raw).expect("deserialize");
        match event {
            DaemonEvent::StreamToken {
                session_id,
                content,
            } => {
                assert_eq!(session_id, "x");
                assert_eq!(content, "hi");
            }
            other => panic!("expected StreamToken, got {:?}", other),
        }
    }

    #[test]
    fn event_deserializes_permission_required() {
        let raw = r#"{
            "type": "permission_required",
            "session_id": "abc",
            "tool_name": "bash",
            "args": {"cmd": "ls"},
            "request_id": "req-42"
        }"#;
        let event: DaemonEvent = serde_json::from_str(raw).expect("deserialize");
        match event {
            DaemonEvent::PermissionRequired {
                request_id,
                tool_name,
                ..
            } => {
                assert_eq!(request_id, "req-42");
                assert_eq!(tool_name, "bash");
            }
            other => panic!("expected PermissionRequired, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_pong() {
        let event = DaemonEvent::Pong;
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        assert!(matches!(back, DaemonEvent::Pong));
    }

    #[test]
    fn event_serializes_turn_complete() {
        let event = DaemonEvent::TurnComplete {
            session_id: "s1".into(),
            prompt_tokens: 42,
            completion_tokens: 100,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::TurnComplete {
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                assert_eq!(prompt_tokens, 42);
                assert_eq!(completion_tokens, 100);
            }
            other => panic!("expected TurnComplete, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_error() {
        let event = DaemonEvent::Error {
            session_id: "s1".into(),
            message: "something broke".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::Error { message, .. } => {
                assert_eq!(message, "something broke");
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_title_generated() {
        let event = DaemonEvent::TitleGenerated {
            session_id: "s1".into(),
            title: "My Great Session".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::TitleGenerated { title, .. } => {
                assert_eq!(title, "My Great Session");
            }
            other => panic!("expected TitleGenerated, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_stream_done() {
        let event = DaemonEvent::StreamDone {
            session_id: "s1".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::StreamDone { session_id } => assert_eq!(session_id, "s1"),
            other => panic!("expected StreamDone, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_cancelled() {
        let event = DaemonEvent::Cancelled {
            session_id: "sess-cancel".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::Cancelled { session_id } => assert_eq!(session_id, "sess-cancel"),
            other => panic!("expected Cancelled, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_tool_started() {
        let event = DaemonEvent::ToolStarted {
            session_id: "s".into(),
            name: "bash".into(),
            args: json!({"command": "ls"}),
            active_model: "gemma4:26b".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ToolStarted {
                name, active_model, ..
            } => {
                assert_eq!(name, "bash");
                assert_eq!(active_model, "gemma4:26b");
            }
            other => panic!("expected ToolStarted, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_tool_finished() {
        let event = DaemonEvent::ToolFinished {
            session_id: "s".into(),
            name: "read_file".into(),
            output: "file contents".into(),
            is_error: false,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ToolFinished {
                name,
                output,
                is_error,
                ..
            } => {
                assert_eq!(name, "read_file");
                assert_eq!(output, "file contents");
                assert!(!is_error);
            }
            other => panic!("expected ToolFinished, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_context_warning() {
        let event = DaemonEvent::ContextWarning {
            session_id: "s".into(),
            message: "context at 80%".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ContextWarning { message, .. } => {
                assert_eq!(message, "context at 80%");
            }
            other => panic!("expected ContextWarning, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_session_created() {
        let event = DaemonEvent::SessionCreated {
            session_id: "new-sess".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::SessionCreated { session_id } => assert_eq!(session_id, "new-sess"),
            other => panic!("expected SessionCreated, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_agent_spawned() {
        let event = DaemonEvent::AgentSpawned {
            session_id: "s".into(),
            prompt_preview: "implement feature".into(),
            depth: 2,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::AgentSpawned {
                prompt_preview,
                depth,
                ..
            } => {
                assert_eq!(prompt_preview, "implement feature");
                assert_eq!(depth, 2);
            }
            other => panic!("expected AgentSpawned, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_agent_finished() {
        let event = DaemonEvent::AgentFinished {
            session_id: "s".into(),
            depth: 1,
            elapsed_ms: 4200,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::AgentFinished {
                depth, elapsed_ms, ..
            } => {
                assert_eq!(depth, 1);
                assert_eq!(elapsed_ms, 4200);
            }
            other => panic!("expected AgentFinished, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_context_pruned() {
        let event = DaemonEvent::ContextPruned {
            session_id: "s".into(),
            chars_removed: 12_000,
            messages_affected: 3,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ContextPruned {
                chars_removed,
                messages_affected,
                ..
            } => {
                assert_eq!(chars_removed, 12_000);
                assert_eq!(messages_affected, 3);
            }
            other => panic!("expected ContextPruned, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_started() {
        let event = DaemonEvent::ChainStarted {
            chain_id: "chain-abc".into(),
            name: "ci-chain".into(),
            node_count: 3,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainStarted {
                chain_id,
                name,
                node_count,
            } => {
                assert_eq!(chain_id, "chain-abc");
                assert_eq!(name, "ci-chain");
                assert_eq!(node_count, 3);
            }
            other => panic!("expected ChainStarted, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_node_transition() {
        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 2,
            node_name: "builder".into(),
            status: "running".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainNodeTransition {
                chain_id,
                cycle,
                node_name,
                status,
            } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(cycle, 2);
                assert_eq!(node_name, "builder");
                assert_eq!(status, "running");
            }
            other => panic!("expected ChainNodeTransition, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_cycle_complete() {
        let event = DaemonEvent::ChainCycleComplete {
            chain_id: "c1".into(),
            cycle: 5,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainCycleComplete { chain_id, cycle } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(cycle, 5);
            }
            other => panic!("expected ChainCycleComplete, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_finished() {
        let event = DaemonEvent::ChainFinished {
            chain_id: "c1".into(),
            success: true,
            reason: "validation passed".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainFinished {
                chain_id,
                success,
                reason,
            } => {
                assert_eq!(chain_id, "c1");
                assert!(success);
                assert_eq!(reason, "validation passed");
            }
            other => panic!("expected ChainFinished, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_status() {
        let state = json!({ "current_cycle": 3, "turns_used": 12 });
        let event = DaemonEvent::ChainStatus {
            chain_id: "c1".into(),
            state: state.clone(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainStatus { chain_id, state: s } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(s["current_cycle"], 3);
            }
            other => panic!("expected ChainStatus, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_list() {
        let event = DaemonEvent::ChainList {
            chains: vec![super::ChainInfo {
                chain_id: "c1".into(),
                name: "build-chain".into(),
                current_cycle: 2,
                node_count: 3,
                status: "running".into(),
            }],
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainList { chains } => {
                assert_eq!(chains.len(), 1);
                assert_eq!(chains[0].chain_id, "c1");
                assert_eq!(chains[0].name, "build-chain");
                assert_eq!(chains[0].status, "running");
            }
            other => panic!("expected ChainList, got {:?}", other),
        }
    }

    #[test]
    fn event_deserializes_chain_started_from_json() {
        let raw = r#"{"type":"chain_started","chain_id":"x","name":"my-chain","node_count":2}"#;
        let event: DaemonEvent = serde_json::from_str(raw).expect("deserialize");
        match event {
            DaemonEvent::ChainStarted {
                chain_id,
                name,
                node_count,
            } => {
                assert_eq!(chain_id, "x");
                assert_eq!(name, "my-chain");
                assert_eq!(node_count, 2);
            }
            other => panic!("expected ChainStarted, got {:?}", other),
        }
    }

    #[test]
    fn chain_info_serializes() {
        let info = super::ChainInfo {
            chain_id: "c1".into(),
            name: "test".into(),
            current_cycle: 1,
            node_count: 2,
            status: "paused".into(),
        };
        let raw = serde_json::to_string(&info).expect("serialize");
        let back: super::ChainInfo = serde_json::from_str(&raw).expect("deserialize");
        assert_eq!(back.chain_id, "c1");
        assert_eq!(back.status, "paused");
    }

    #[test]
    fn event_serializes_session_ended() {
        let event = DaemonEvent::SessionEnded {
            session_id: "sess-end".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::SessionEnded { session_id } => assert_eq!(session_id, "sess-end"),
            other => panic!("expected SessionEnded, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_session_cancelled() {
        let event = DaemonEvent::SessionCancelled {
            session_id: "sess-cancel".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::SessionCancelled { session_id } => assert_eq!(session_id, "sess-cancel"),
            other => panic!("expected SessionCancelled, got {:?}", other),
        }
    }

    #[test]
    fn session_ended_to_backend_returns_none() {
        let event = DaemonEvent::SessionEnded {
            session_id: "s".into(),
        };
        assert!(event.to_backend_event().is_none());
    }

    #[test]
    fn session_cancelled_to_backend_returns_none() {
        let event = DaemonEvent::SessionCancelled {
            session_id: "s".into(),
        };
        assert!(event.to_backend_event().is_none());
    }

    #[test]
    fn event_deserializes_session_ended_from_json() {
        let raw = r#"{"type":"session_ended","session_id":"x"}"#;
        let event: DaemonEvent = serde_json::from_str(raw).expect("deserialize");
        assert!(matches!(event, DaemonEvent::SessionEnded { session_id } if session_id == "x"));
    }

    #[test]
    fn event_deserializes_session_cancelled_from_json() {
        let raw = r#"{"type":"session_cancelled","session_id":"y"}"#;
        let event: DaemonEvent = serde_json::from_str(raw).expect("deserialize");
        assert!(matches!(event, DaemonEvent::SessionCancelled { session_id } if session_id == "y"));
    }

    #[test]
    fn event_serializes_health() {
        let event = DaemonEvent::Health {
            uptime_secs: 3600,
            session_count: 5,
            pid: 12345,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::Health {
                uptime_secs,
                session_count,
                pid,
            } => {
                assert_eq!(uptime_secs, 3600);
                assert_eq!(session_count, 5);
                assert_eq!(pid, 12345);
            }
            other => panic!("expected Health, got {:?}", other),
        }
    }

    #[test]
    fn health_to_backend_returns_none() {
        let event = DaemonEvent::Health {
            uptime_secs: 0,
            session_count: 0,
            pid: 1,
        };
        assert!(event.to_backend_event().is_none());
    }

    #[test]
    fn event_serializes_tool_output() {
        let event = DaemonEvent::ToolOutput {
            session_id: "s1".into(),
            name: "bash".into(),
            line: "output line".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ToolOutput {
                session_id,
                name,
                line,
            } => {
                assert_eq!(session_id, "s1");
                assert_eq!(name, "bash");
                assert_eq!(line, "output line");
            }
            other => panic!("expected ToolOutput, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_file_diff() {
        let event = DaemonEvent::FileDiff {
            session_id: "s1".into(),
            path: "src/main.rs".into(),
            diff: "+new line".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::FileDiff {
                session_id,
                path,
                diff,
            } => {
                assert_eq!(session_id, "s1");
                assert_eq!(path, "src/main.rs");
                assert_eq!(diff, "+new line");
            }
            other => panic!("expected FileDiff, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_context_usage() {
        let event = DaemonEvent::ContextUsage {
            session_id: "s1".into(),
            used: 5000,
            limit: 32000,
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ContextUsage {
                session_id,
                used,
                limit,
            } => {
                assert_eq!(session_id, "s1");
                assert_eq!(used, 5000);
                assert_eq!(limit, 32000);
            }
            other => panic!("expected ContextUsage, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_staged_changeset() {
        let event = DaemonEvent::StagedChangeset {
            session_id: "s1".into(),
            changes: vec![json!({"file": "test.rs"})],
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::StagedChangeset {
                session_id,
                changes,
            } => {
                assert_eq!(session_id, "s1");
                assert_eq!(changes.len(), 1);
                assert_eq!(changes[0]["file"], "test.rs");
            }
            other => panic!("expected StagedChangeset, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_session_list() {
        let event = DaemonEvent::SessionList {
            sessions: vec![SessionInfo {
                session_id: "s1".into(),
                created_at: "2026-01-01T00:00:00Z".into(),
                last_active: "2026-01-01T00:01:00Z".into(),
                client_count: 1,
                status: "idle".into(),
            }],
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::SessionList { sessions } => {
                assert_eq!(sessions.len(), 1);
                assert_eq!(sessions[0].session_id, "s1");
                assert_eq!(sessions[0].status, "idle");
            }
            other => panic!("expected SessionList, got {:?}", other),
        }
    }

    #[test]
    fn event_serializes_chain_log() {
        let event = DaemonEvent::ChainLog {
            chain_id: "c1".into(),
            level: "warn".into(),
            message: "test message".into(),
        };
        let raw = serde_json::to_string(&event).expect("serialize");
        let back: DaemonEvent = serde_json::from_str(&raw).expect("deserialize");
        match back {
            DaemonEvent::ChainLog {
                chain_id,
                level,
                message,
            } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(level, "warn");
                assert_eq!(message, "test message");
            }
            other => panic!("expected ChainLog, got {:?}", other),
        }
    }

    #[test]
    fn chain_events_to_backend_returns_none() {
        // All chain events are daemon-only and should return None from to_backend_event
        let events = vec![
            DaemonEvent::ChainStarted {
                chain_id: "c".into(),
                name: "n".into(),
                node_count: 1,
            },
            DaemonEvent::ChainNodeTransition {
                chain_id: "c".into(),
                cycle: 1,
                node_name: "n".into(),
                status: "ok".into(),
            },
            DaemonEvent::ChainCycleComplete {
                chain_id: "c".into(),
                cycle: 1,
            },
            DaemonEvent::ChainFinished {
                chain_id: "c".into(),
                success: true,
                reason: "done".into(),
            },
            DaemonEvent::ChainLog {
                chain_id: "c".into(),
                level: "warn".into(),
                message: "test".into(),
            },
            DaemonEvent::ChainStatus {
                chain_id: "c".into(),
                state: json!({}),
            },
            DaemonEvent::ChainList { chains: vec![] },
        ];
        for event in events {
            assert!(
                event.to_backend_event().is_none(),
                "chain events should not map to backend events"
            );
        }
    }
}
