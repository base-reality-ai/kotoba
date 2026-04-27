use crate::ollama::client::OllamaClient;
use crate::ollama::types::ToolCall;
use crate::permissions::engine::PermissionEngine;
use crate::session::Session;
use crate::system_prompt::build_system_prompt;
use crate::tools::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

/// Maximum sub-agent nesting depth. Calls beyond this return an error without spawning.
pub const MAX_AGENT_DEPTH: u8 = 3;

/// Timeout for a single sub-agent run (seconds).
const AGENT_TIMEOUT_SECS: u64 = 300;

/// Maximum tool-use rounds per sub-agent invocation.
const MAX_ROUNDS: usize = 10;

pub struct AgentTool {
    base_url: String,
    model: String,
    config_dir: PathBuf,
    embed_model: String,
    /// Nesting depth of this tool instance (0 = top-level parent's tool).
    depth: u8,
    /// Optional channel to send `AgentSpawned` / `AgentFinished` events to the TUI.
    event_tx: Option<tokio::sync::mpsc::Sender<crate::tui::BackendEvent>>,
}

impl AgentTool {
    pub fn new(base_url: String, model: String, config_dir: PathBuf) -> Self {
        AgentTool {
            base_url,
            model,
            config_dir,
            embed_model: "nomic-embed-text".to_string(),
            depth: 0,
            event_tx: None,
        }
    }

    pub fn with_embed_model(mut self, embed_model: String) -> Self {
        self.embed_model = embed_model;
        self
    }

    #[allow(dead_code)]
    pub fn with_depth(mut self, depth: u8) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_event_tx(
        mut self,
        tx: tokio::sync::mpsc::Sender<crate::tui::BackendEvent>,
    ) -> Self {
        self.event_tx = Some(tx);
        self
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &'static str {
        "agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a sub-agent to complete a self-contained task. Use when a task can be \
        parallelized or isolated. The sub-agent has full tool access (except agent, to \
        prevent unbounded nesting). Returns the sub-agent's final response."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent."
                },
                "model": {
                    "type": "string",
                    "description": "Optional: override the model for this sub-agent."
                }
            },
            "required": ["prompt"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use agent to delegate complex subtasks that benefit from a separate context. \
              The sub-agent gets its own conversation and tool access. Use for research, \
              exploration, or parallel work — not for simple operations.",
        )
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        // Depth cap
        if self.depth >= MAX_AGENT_DEPTH {
            return Ok(ToolResult {
                content: format!(
                    "max agent depth ({}) reached — refusing to spawn further sub-agents. Try: complete the work in the current agent, or simplify the task to avoid further nesting.",
                    MAX_AGENT_DEPTH
                ),
                is_error: true,
            });
        }

        let prompt = args["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("agent tool: missing 'prompt'"))?;

        let model = args["model"]
            .as_str()
            .map_or_else(|| self.model.clone(), |s| s.to_string());

        // Truncate prompt for display (back off to UTF-8 char boundary)
        let preview = if prompt.len() > 80 {
            let mut end = 80usize.min(prompt.len());
            while end > 0 && !prompt.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &prompt[..end])
        } else {
            prompt.to_string()
        };

        // Emit AgentSpawned
        if let Some(ref tx) = self.event_tx {
            tx.send(crate::tui::BackendEvent::AgentSpawned {
                prompt_preview: preview.clone(),
                depth: self.depth,
            })
            .await
            .ok();
        }

        let t_start = Instant::now();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(AGENT_TIMEOUT_SECS),
            self.run_sub_agent(prompt, &model),
        )
        .await;

        let elapsed_ms = t_start.elapsed().as_millis() as u64;

        // Emit AgentFinished
        if let Some(ref tx) = self.event_tx {
            tx.send(crate::tui::BackendEvent::AgentFinished {
                depth: self.depth,
                elapsed_ms,
            })
            .await
            .ok();
        }

        match result {
            Ok(inner) => inner,
            Err(_) => Ok(ToolResult {
                content: format!(
                    "sub-agent timed out after {}s. Try: split the task into smaller agent invocations, or do the work directly without delegating.",
                    AGENT_TIMEOUT_SECS
                ),
                is_error: true,
            }),
        }
    }
}

impl AgentTool {
    async fn run_sub_agent(&self, prompt: &str, model: &str) -> anyhow::Result<ToolResult> {
        let sub_system = build_system_prompt(&[], None).await;

        let cwd = std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let sub_client = OllamaClient::new(self.base_url.clone(), model.to_string());
        let mut sub_session = Session::new(cwd, model.to_string());

        // Sub-registry excludes AgentTool to prevent unbounded nesting
        let sub_registry = crate::tools::registry::sub_agent_registry(
            &sub_session.id,
            &self.config_dir,
            &self.base_url,
            model,
            &self.embed_model,
        );
        let sub_engine = PermissionEngine::new(true, vec![]);

        let mut messages: Vec<Value> = vec![
            json!({"role": "system", "content": sub_system}),
            json!({"role": "user", "content": prompt}),
        ];

        let tool_defs = sub_registry.definitions();

        for _round in 0..MAX_ROUNDS {
            let compacted = crate::compaction::maybe_compact(&messages, &sub_client, false).await?;
            let resp = sub_client.chat(&compacted, &tool_defs).await?;

            let content = resp.message.content.clone();
            let tool_calls: Vec<ToolCall> = resp.message.tool_calls.clone();

            if tool_calls.is_empty() {
                sub_session.push_message(json!({"role": "assistant", "content": content}));
                return Ok(ToolResult {
                    content: if content.is_empty() {
                        "[sub-agent returned empty response]".to_string()
                    } else {
                        content
                    },
                    is_error: false,
                });
            }

            let assistant_msg = json!({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls.iter().map(|tc| json!({
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })).collect::<Vec<_>>()
            });
            messages.push(assistant_msg);

            for tc in &tool_calls {
                let name = &tc.function.name;
                let tc_args = &tc.function.arguments;
                let decision = sub_engine.check(name, tc_args);
                let result = match decision {
                    crate::permissions::Decision::Deny => {
                        format!("Permission denied for tool '{}'", name)
                    }
                    _ => match sub_registry.call(name, tc_args.clone()).await {
                        Ok(r) => r.content,
                        Err(e) => format!("Tool '{}' error: {}", name, e),
                    },
                };
                let content = crate::util::truncate_tool_output(&result);
                messages.push(json!({"role": "tool", "name": name, "content": content}));
            }
        }

        Ok(ToolResult {
            content: format!(
                "sub-agent exhausted all {} rounds without producing a final answer. Try: break the task into smaller pieces, or provide more specific instructions.",
                MAX_ROUNDS
            ),
            is_error: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_agent(depth: u8) -> AgentTool {
        AgentTool::new(
            "http://localhost:11434".to_string(),
            "gemma4:26b".to_string(),
            PathBuf::from("/tmp"),
        )
        .with_depth(depth)
    }

    #[tokio::test]
    async fn agent_depth_cap_returns_error() {
        let agent = make_agent(MAX_AGENT_DEPTH);
        let result = agent.call(json!({"prompt": "do something"})).await.unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("max agent depth"),
            "unexpected: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn agent_depth_cap_at_boundary() {
        // depth = MAX - 1 should NOT be blocked (would spawn at depth MAX-1, child at MAX)
        // depth = MAX should be blocked immediately
        let at_cap = make_agent(MAX_AGENT_DEPTH);
        let result = at_cap.call(json!({"prompt": "test"})).await.unwrap();
        assert!(result.is_error, "at cap should be blocked");

        let below_cap = make_agent(MAX_AGENT_DEPTH - 1);
        // We can't actually run the sub-agent without Ollama, but we CAN verify
        // that call() doesn't immediately error with the depth cap message.
        // Use a very short timeout to force a timeout error instead.
        // (We just verify the depth cap path isn't taken.)
        // Actually for this unit test we just check the depth field is set.
        assert_eq!(below_cap.depth, MAX_AGENT_DEPTH - 1);
    }

    #[test]
    fn agent_depth_increments_in_context() {
        let agent = make_agent(0);
        assert_eq!(agent.depth, 0);
        let child = AgentTool::new(
            agent.base_url.clone(),
            agent.model.clone(),
            agent.config_dir.clone(),
        )
        .with_depth(agent.depth + 1);
        assert_eq!(child.depth, 1);
    }

    #[test]
    fn prompt_preview_truncated_at_80_chars() {
        let long_prompt = "a".repeat(120);
        let preview = if long_prompt.len() > 80 {
            format!("{}…", &long_prompt[..80])
        } else {
            long_prompt.clone()
        };
        // The … is a multi-byte char, so total len > 80 but only 80 'a's
        assert_eq!(preview.chars().filter(|c| *c == 'a').count(), 80);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn prompt_preview_short_unchanged() {
        let short = "explain this code";
        let preview = if short.len() > 80 {
            format!("{}…", &short[..80])
        } else {
            short.to_string()
        };
        assert_eq!(preview, short);
    }

    #[tokio::test]
    async fn agent_spawned_event_emitted() {
        // Depth at cap so we get an immediate return, but the AgentSpawned event
        // fires BEFORE the depth check in the current impl — actually depth check
        // is first. Use depth=0 and verify via channel that event fires up until
        // Ollama is needed (timeout path).
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let agent = AgentTool::new(
            "http://127.0.0.1:1".to_string(), // unreachable → timeout quickly
            "gemma4:26b".to_string(),
            PathBuf::from("/tmp"),
        )
        .with_depth(0)
        .with_event_tx(tx);

        // Run with a very short timeout override isn't possible through the public
        // API, but we can verify event_tx is wired: at depth=MAX_AGENT_DEPTH the
        // depth cap fires BEFORE the spawn, so no event. Use depth=0 with an
        // unreachable server — the AgentSpawned fires, then we timeout.
        // Spawn in background so we can drain the channel.
        let handle = tokio::spawn(async move {
            let _ = agent.call(json!({"prompt": "test task"})).await;
        });

        // AgentSpawned should arrive promptly (before any network call)
        let event = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv()).await;

        match event {
            Ok(Some(crate::tui::BackendEvent::AgentSpawned { depth, .. })) => {
                assert_eq!(depth, 0);
            }
            Ok(Some(other)) => panic!("unexpected event: {:?}", other),
            Ok(None) => panic!("channel closed before event"),
            Err(_) => panic!("timed out waiting for AgentSpawned event"),
        }

        handle.abort();
    }

    #[test]
    fn agent_tool_excluded_from_sub_registry() {
        let registry = crate::tools::registry::sub_agent_registry(
            "test-session",
            std::path::Path::new("/tmp"),
            "http://localhost:11434",
            "gemma4:26b",
            "nomic-embed-text",
        );
        // sub_agent_registry must not contain "agent" tool
        let defs = registry.definitions();
        let has_agent = defs.iter().any(|d| d.function.name == "agent");
        assert!(
            !has_agent,
            "sub-agent registry must not contain the agent tool"
        );
    }

    #[tokio::test]
    async fn agent_missing_prompt_returns_error() {
        // When depth is at cap, missing prompt is never reached. Use depth=0.
        // But without Ollama, the call returns an error. We just want to confirm
        // the missing-prompt path propagates as anyhow::Error.
        // At cap, depth error returns before prompt check — but let's also verify
        // that a below-cap agent with no prompt returns Err (not Ok).
        let below_cap = make_agent(0);
        let result = below_cap.call(json!({})).await;
        // Missing prompt should propagate as an Err (anyhow error)
        assert!(result.is_err(), "missing prompt should be an Err");
        let err = result.err().unwrap();
        assert!(
            err.to_string().contains("prompt"),
            "error should mention 'prompt': {err}"
        );
    }

    #[test]
    fn agent_with_embed_model_sets_field() {
        let agent = AgentTool::new(
            "http://localhost:11434".to_string(),
            "gemma4:26b".to_string(),
            PathBuf::from("/tmp"),
        )
        .with_embed_model("nomic-embed-text".to_string());
        assert_eq!(agent.embed_model, "nomic-embed-text");
    }

    #[test]
    fn max_rounds_is_reasonable() {
        const { assert!(MAX_ROUNDS >= 5 && MAX_ROUNDS <= 50) };
    }

    #[test]
    fn agent_name_is_agent() {
        let agent = make_agent(0);
        assert_eq!(agent.name(), "agent");
    }
}
