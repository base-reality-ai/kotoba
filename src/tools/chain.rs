use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct ChainControlTool;

#[async_trait]
impl Tool for ChainControlTool {
    fn name(&self) -> &'static str {
        "chain_control"
    }

    fn description(&self) -> &'static str {
        "Manage running agent chains: check status, stop, pause, resume, inject messages, \
         or list chain artifacts. Use this to interact with orchestration chains."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["status", "stop", "pause", "resume", "talk", "list"],
                    "description": "The chain operation to perform. 'status' shows current chain state, \
                                    'stop' halts the chain, 'pause'/'resume' toggle execution, \
                                    'talk' injects a message into a node, 'list' shows chain artifacts."
                },
                "node": {
                    "type": "string",
                    "description": "Target node name (required for 'talk' command)."
                },
                "message": {
                    "type": "string",
                    "description": "Message content to inject (required for 'talk' command)."
                },
                "cycle": {
                    "type": "integer",
                    "description": "Filter artifacts to a specific cycle number (optional for 'list' command)."
                }
            },
            "required": ["command"]
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use chain to run multi-agent orchestration pipelines defined in YAML. \
              Chains connect agents in sequence, passing output between steps.",
        )
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let Some(command) = args["command"].as_str() else {
            return Ok(ToolResult {
                content: "'command' parameter is required. Try: include 'command' (one of: status, stop, pause, resume, talk, list).".to_string(),
                is_error: true,
            });
        };

        match command {
            "status" => call_status(),
            "stop" => call_stop(),
            "pause" => call_pause(),
            "resume" => call_resume(),
            "talk" => call_talk(&args),
            "list" => call_list(&args),
            other => Ok(ToolResult {
                content: format!(
                    "unknown command '{}'. Try: use one of status, stop, pause, resume, talk, list.",
                    other
                ),
                is_error: true,
            }),
        }
    }
}

fn call_status() -> anyhow::Result<ToolResult> {
    match crate::orchestrate::chain_status() {
        Some(state) => {
            let json = serde_json::to_string_pretty(&state)?;
            Ok(ToolResult {
                content: json,
                is_error: false,
            })
        }
        None => Ok(ToolResult {
            content: "No chain is currently running.".to_string(),
            is_error: false,
        }),
    }
}

fn call_stop() -> anyhow::Result<ToolResult> {
    match crate::orchestrate::stop_chain() {
        Ok(()) => Ok(ToolResult {
            content: "Chain stop signal sent.".to_string(),
            is_error: false,
        }),
        Err(e) => Ok(ToolResult {
            content: format!(
                "{}. Try: chain status (command='status') to confirm a chain is active.",
                e
            ),
            is_error: true,
        }),
    }
}

fn call_pause() -> anyhow::Result<ToolResult> {
    match crate::orchestrate::pause_chain() {
        Ok(()) => Ok(ToolResult {
            content: "Chain paused.".to_string(),
            is_error: false,
        }),
        Err(e) => Ok(ToolResult {
            content: format!(
                "{}. Try: chain status to confirm a chain is active and not already paused.",
                e
            ),
            is_error: true,
        }),
    }
}

fn call_resume() -> anyhow::Result<ToolResult> {
    match crate::orchestrate::resume_chain() {
        Ok(()) => Ok(ToolResult {
            content: "Chain resumed.".to_string(),
            is_error: false,
        }),
        Err(e) => Ok(ToolResult {
            content: format!(
                "{}. Try: chain status to confirm a chain exists and is paused.",
                e
            ),
            is_error: true,
        }),
    }
}

fn call_talk(args: &Value) -> anyhow::Result<ToolResult> {
    let Some(node) = args["node"].as_str() else {
        return Ok(ToolResult {
            content: "'node' parameter is required for 'talk' command. Try: include 'node' (target node ID).".to_string(),
            is_error: true,
        });
    };
    let Some(message) = args["message"].as_str() else {
        return Ok(ToolResult {
            content: "'message' parameter is required for 'talk' command. Try: include 'message' (text to deliver to the node).".to_string(),
            is_error: true,
        });
    };

    match crate::orchestrate::chain_talk(node, message) {
        Ok(()) => Ok(ToolResult {
            content: format!("Message injected into node '{}'.", node),
            is_error: false,
        }),
        Err(e) => Ok(ToolResult {
            content: format!(
                "{}. Try: confirm 'node' matches an active chain node (use command='status' to list nodes).",
                e
            ),
            is_error: true,
        }),
    }
}

fn call_list(args: &Value) -> anyhow::Result<ToolResult> {
    let cycle = args["cycle"].as_u64().map(|c| c as usize);

    match crate::orchestrate::chain_log(cycle) {
        Ok(entries) => {
            if entries.is_empty() {
                return Ok(ToolResult {
                    content: "No chain artifacts found.".to_string(),
                    is_error: false,
                });
            }
            let json = serde_json::to_string_pretty(&entries)?;
            Ok(ToolResult {
                content: json,
                is_error: false,
            })
        }
        Err(e) => Ok(ToolResult {
            content: format!(
                "{}. Try: confirm a chain has run; pass an explicit 'cycle' to narrow the result.",
                e
            ),
            is_error: true,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn status_with_no_chain_running() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "status"})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No chain"));
    }

    #[tokio::test]
    async fn stop_with_no_chain_running() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "stop"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No chain"));
    }

    #[tokio::test]
    async fn pause_with_no_chain_running() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "pause"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No chain"));
        assert!(
            result.content.contains("Try: chain status"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn resume_with_no_chain_running() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "resume"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No chain"));
        assert!(
            result.content.contains("Try: chain status"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn list_with_no_chain_running() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "list"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No chain"));
        assert!(
            result.content.contains("Try: confirm a chain has run"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn talk_missing_node() {
        let tool = ChainControlTool;
        let result = tool
            .call(json!({"command": "talk", "message": "hello"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("'node' parameter is required"));
    }

    #[tokio::test]
    async fn talk_missing_message() {
        let tool = ChainControlTool;
        let result = tool
            .call(json!({"command": "talk", "node": "builder"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("'message' parameter is required"));
    }

    /// Pin the canonical `Try:` next-step hint on the `stop` no-chain branch.
    /// Mirrors the pause/resume/list pins (same chain-control commands).
    #[tokio::test]
    async fn chain_control_stop_no_chain_pins_canonical_try_hint() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "stop"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("No chain"));
        assert!(
            result.content.contains("Try: chain status"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    /// Pin the canonical `Try:` hint on the `talk` missing-node guard. The
    /// hint must name the param (`'node'`) so the model can self-correct.
    #[tokio::test]
    async fn chain_control_talk_missing_node_pins_canonical_try_hint() {
        let tool = ChainControlTool;
        let result = tool
            .call(json!({"command": "talk", "message": "hello"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("'node' parameter is required"));
        assert!(
            result.content.contains("Try: include 'node'"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    /// Pin the canonical `Try:` hint on the `talk` missing-message guard.
    /// The hint must name the param (`'message'`).
    #[tokio::test]
    async fn chain_control_talk_missing_message_pins_canonical_try_hint() {
        let tool = ChainControlTool;
        let result = tool
            .call(json!({"command": "talk", "node": "builder"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("'message' parameter is required"));
        assert!(
            result.content.contains("Try: include 'message'"),
            "missing canonical Try: hint: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn unknown_command() {
        let tool = ChainControlTool;
        let result = tool.call(json!({"command": "restart"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("unknown command"));
    }

    #[tokio::test]
    async fn missing_command_param() {
        let tool = ChainControlTool;
        let result = tool.call(json!({})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("'command' parameter is required"));
    }

    #[test]
    fn tool_metadata() {
        let tool = ChainControlTool;
        assert_eq!(tool.name(), "chain_control");
        assert!(!tool.is_read_only());
        let params = tool.parameters();
        assert!(params["properties"]["command"].is_object());
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&json!("command")));
    }
}
