use crate::logging;
use crate::tools::registry::ToolRegistry;
use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub async fn run_mcp_server(registry: ToolRegistry) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin).lines();

    logging::log(&format!(
        "[dm-mcp] Server ready. Tools: {}",
        registry.definitions().len()
    ));

    while let Some(line) = reader.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err = json_rpc_error(Value::Null, -32700, format!("Parse error: {}", e));
                write_response(&mut stdout, err).await?;
                continue;
            }
        };

        if is_notification(&request) {
            continue;
        }

        if let Some(response) = process_request(request, &registry).await {
            write_response(&mut stdout, response).await?;
        }
    }

    Ok(())
}

/// Process a single JSON-RPC request. Returns `None` for notifications that
/// need no response.
pub async fn process_request(request: Value, registry: &ToolRegistry) -> Option<Value> {
    let id = request["id"].clone();
    let method = request["method"].as_str().unwrap_or("").to_string();
    let params = request["params"].clone();

    if is_notification(&request) {
        return None;
    }

    let response = match method.as_str() {
        "initialize" => handle_initialize(id),
        "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
        "tools/list" => handle_tools_list(id, registry),
        "tools/call" => handle_tools_call(id, params, registry).await,
        "resources/list" => handle_resources_list(id),
        "resources/read" => handle_resources_read(id, params).await,
        "prompts/list" => handle_prompts_list(id),
        "prompts/get" => handle_prompts_get(id, params),
        other => json_rpc_error(id, -32601, format!("Method not found: {}", other)),
    };

    Some(response)
}

pub fn handle_initialize(id: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {}, "resources": {}, "prompts": {} },
            "serverInfo": {
                "name": "dm",
                "version": env!("CARGO_PKG_VERSION"),
            }
        }
    })
}

pub fn handle_tools_list(id: Value, registry: &ToolRegistry) -> Value {
    let tools: Vec<Value> = registry
        .definitions()
        .iter()
        .map(|def| {
            json!({
                "name": def.function.name,
                "description": def.function.description,
                "inputSchema": def.function.parameters,
            })
        })
        .collect();

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}

pub async fn handle_tools_call(id: Value, params: Value, registry: &ToolRegistry) -> Value {
    let name = match params["name"].as_str() {
        Some(n) => n.to_string(),
        None => return json_rpc_error(id, -32602, "Missing tool name".to_string()),
    };
    let args = params["arguments"].clone();

    match registry.call(&name, args).await {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "content": [{"type": "text", "text": result.content}],
                "isError": result.is_error,
            }
        }),
        Err(e) => json_rpc_error(id, -32603, format!("Tool execution error: {}", e)),
    }
}

pub fn handle_resources_list(id: Value) -> Value {
    let mut resources = Vec::new();

    // DM.md project config
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("DM.md").exists() {
        resources.push(json!({
            "uri": "dm://config/dm.md",
            "name": "DM.md",
            "description": "Project configuration for Dark Matter",
            "mimeType": "text/markdown",
        }));
    }

    // Chain state (if running)
    if crate::orchestrate::chain_status().is_some() {
        resources.push(json!({
            "uri": "dm://chain/state",
            "name": "Chain State",
            "description": "Current orchestration chain status and metrics",
            "mimeType": "application/json",
        }));
    }

    // Git status (always available in a git repo)
    resources.push(json!({
        "uri": "dm://git/status",
        "name": "Git Status",
        "description": "Current git repository status",
        "mimeType": "text/plain",
    }));

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "resources": resources }
    })
}

pub async fn handle_resources_read(id: Value, params: Value) -> Value {
    let Some(uri) = params["uri"].as_str() else {
        return json_rpc_error(id, -32602, "Missing resource URI".into());
    };

    match uri {
        "dm://config/dm.md" => {
            let cwd = std::env::current_dir().unwrap_or_default();
            match std::fs::read_to_string(cwd.join("DM.md")) {
                Ok(content) => {
                    resource_response(id, uri, "text/markdown", &bounded_content(&content))
                }
                Err(e) => json_rpc_error(id, -32603, format!("Cannot read DM.md: {}", e)),
            }
        }
        "dm://chain/state" => match crate::orchestrate::chain_status() {
            Some(state) => {
                let json_str = serde_json::to_string_pretty(&state).unwrap_or_default();
                resource_response(id, uri, "application/json", &bounded_content(&json_str))
            }
            None => json_rpc_error(id, -32603, "No chain is running".into()),
        },
        "dm://git/status" => {
            match tokio::process::Command::new("git")
                .args(["status", "--porcelain=v2", "--branch"])
                .output()
                .await
            {
                Ok(output) => {
                    let text = String::from_utf8_lossy(&output.stdout).to_string();
                    resource_response(id, uri, "text/plain", &bounded_content(&text))
                }
                Err(e) => json_rpc_error(id, -32603, format!("git status failed: {}", e)),
            }
        }
        _ => json_rpc_error(id, -32602, format!("Unknown resource: {}", uri)),
    }
}

pub fn handle_prompts_list(id: Value) -> Value {
    let prompts = vec![
        json!({
            "name": "commit-message",
            "description": "Generate a conventional commits message from a git diff",
            "arguments": [{ "name": "diff", "description": "The git diff to generate a commit message for", "required": true }]
        }),
        json!({
            "name": "code-review",
            "description": "Review code changes for bugs, style, and improvements",
            "arguments": [
                { "name": "diff", "description": "The code diff to review", "required": true },
                { "name": "focus", "description": "Review focus: security, performance, or style", "required": false }
            ]
        }),
        json!({
            "name": "explain-code",
            "description": "Explain what a code snippet does in plain language",
            "arguments": [
                { "name": "code", "description": "The code snippet to explain", "required": true },
                { "name": "language", "description": "Programming language (for syntax context)", "required": false }
            ]
        }),
        json!({
            "name": "summarize",
            "description": "Summarize content to a target length",
            "arguments": [
                { "name": "content", "description": "The text to summarize", "required": true },
                { "name": "length", "description": "Target length: short (~1 paragraph), medium (~3 paragraphs), long (~1 page)", "required": false }
            ]
        }),
    ];
    json!({ "jsonrpc": "2.0", "id": id, "result": { "prompts": prompts } })
}

pub fn handle_prompts_get(id: Value, params: Value) -> Value {
    let Some(name) = params["name"].as_str() else {
        return json_rpc_error(id, -32602, "Missing prompt name".into());
    };
    let args = &params["arguments"];

    let messages = match name {
        "commit-message" => {
            let diff = bounded_arg(args, "diff");
            vec![json!({
                "role": "user",
                "content": { "type": "text", "text": format!(
                    "Generate a concise git commit message for the following staged diff.\n\
                     Follow conventional commits format (type: description) where appropriate.\n\
                     Respond with ONLY the commit message — no explanation, no quotes, no markdown.\n\n\
                     ```diff\n{}\n```", diff) }
            })]
        }
        "code-review" => {
            let diff = bounded_arg(args, "diff");
            let focus = args
                .get("focus")
                .and_then(|v| v.as_str())
                .unwrap_or("general");
            let focus_instruction = match focus {
                "security" => {
                    "Focus on security vulnerabilities: injection, XSS, auth bypass, data exposure."
                }
                "performance" => {
                    "Focus on performance: unnecessary allocations, O(n²) loops, missing caches."
                }
                "style" => {
                    "Focus on code style: naming, consistency, readability, idiomatic patterns."
                }
                _ => "Review for correctness, style, and potential improvements.",
            };
            vec![json!({
                "role": "user",
                "content": { "type": "text", "text": format!(
                    "Review the following code changes. {}\n\n\
                     Report:\n\
                     1. **Issues** — bugs, security problems, or logic errors\n\
                     2. **Suggestions** — improvements, better patterns, or cleaner approaches\n\
                     3. **Summary** — overall assessment (approve / request changes)\n\n\
                     ```diff\n{}\n```", focus_instruction, diff) }
            })]
        }
        "explain-code" => {
            let code = bounded_arg(args, "code");
            let lang = args.get("language").and_then(|v| v.as_str()).unwrap_or("");
            let lang_note = if lang.is_empty() {
                String::new()
            } else {
                format!(" ({})", lang)
            };
            vec![json!({
                "role": "user",
                "content": { "type": "text", "text": format!(
                    "Explain what this code{} does in plain language. \
                     Be concise but thorough. Mention the key logic, \
                     data structures, and any notable patterns or gotchas.\n\n\
                     ```{}\n{}\n```", lang_note, lang, code) }
            })]
        }
        "summarize" => {
            let content = bounded_arg(args, "content");
            let length = args
                .get("length")
                .and_then(|v| v.as_str())
                .unwrap_or("medium");
            let length_instruction = match length {
                "short" => "Summarize in 1-2 sentences.",
                "long" => "Summarize in about 1 page (~500 words).",
                _ => "Summarize in 2-3 paragraphs.",
            };
            vec![json!({
                "role": "user",
                "content": { "type": "text", "text": format!("{}\n\n{}", length_instruction, content) }
            })]
        }
        _ => return json_rpc_error(id, -32602, format!("Unknown prompt: {}", name)),
    };

    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "description": match name {
                "commit-message" => "Generate a commit message from a diff",
                "code-review" => "Review code changes",
                "explain-code" => "Explain code in plain language",
                "summarize" => "Summarize content",
                _ => "",
            },
            "messages": messages,
        }
    })
}

fn is_notification(request: &Value) -> bool {
    request["id"].is_null()
        && request["method"]
            .as_str()
            .unwrap_or("")
            .starts_with("notifications/")
}

const MAX_PROMPT_ARG_BYTES: usize = 500_000;
const MAX_RESOURCE_BYTES: usize = 1_000_000;

fn bound_string(s: &str, max: usize) -> String {
    if s.len() > max {
        format!(
            "{}\n[... truncated at {} bytes ...]",
            crate::util::safe_truncate(s, max),
            max
        )
    } else {
        s.to_string()
    }
}

fn bounded_arg(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| bound_string(s, MAX_PROMPT_ARG_BYTES))
        .unwrap_or_default()
}

fn bounded_content(content: &str) -> String {
    bound_string(content, MAX_RESOURCE_BYTES)
}

fn resource_response(id: Value, uri: &str, mime_type: &str, content: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "contents": [{
                "uri": uri,
                "mimeType": mime_type,
                "text": content,
            }]
        }
    })
}

fn json_rpc_error(id: Value, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    })
}

async fn write_response(stdout: &mut tokio::io::Stdout, response: Value) -> Result<()> {
    let mut line = serde_json::to_string(&response)?;
    line.push('\n');
    stdout.write_all(line.as_bytes()).await?;
    stdout.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::bash::BashTool;
    use crate::tools::registry::ToolRegistry;

    #[test]
    fn test_handle_initialize_protocol_version() {
        let resp = handle_initialize(json!(1));
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(resp["result"]["serverInfo"]["name"], "dm");
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn test_handle_tools_list_empty_registry() {
        let registry = ToolRegistry::new();
        let resp = handle_tools_list(json!(1), &registry);
        let tools = resp["result"]["tools"]
            .as_array()
            .expect("tools should be array");
        assert!(tools.is_empty(), "expected empty tools list");
    }

    #[test]
    fn test_handle_tools_list_includes_registered_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(BashTool);
        let resp = handle_tools_list(json!(1), &registry);
        let tools = resp["result"]["tools"]
            .as_array()
            .expect("tools should be array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "bash");
    }

    #[tokio::test]
    async fn test_process_request_initialize() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}});
        let resp = process_request(req, &registry)
            .await
            .expect("should return Some");
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
    }

    #[tokio::test]
    async fn test_process_request_tools_list() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}});
        let resp = process_request(req, &registry)
            .await
            .expect("should return Some");
        assert!(
            resp["result"]["tools"].is_array(),
            "result.tools should be array"
        );
    }

    #[tokio::test]
    async fn test_process_request_unknown_method_returns_error() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": 3, "method": "nonexistent/method", "params": {}});
        let resp = process_request(req, &registry)
            .await
            .expect("should return Some");
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn test_process_request_notification_returns_none() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": null, "method": "notifications/cancelled", "params": {}});
        let resp = process_request(req, &registry).await;
        assert!(resp.is_none(), "notifications should return None");
    }

    #[test]
    fn json_rpc_error_has_correct_structure() {
        let err = json_rpc_error(json!(42), -32601, "Method not found".to_string());
        assert_eq!(err["jsonrpc"], "2.0");
        assert_eq!(err["id"], 42);
        assert_eq!(err["error"]["code"], -32601);
        assert_eq!(err["error"]["message"], "Method not found");
    }

    #[test]
    fn handle_initialize_id_is_echoed_back() {
        let resp = handle_initialize(json!("req-99"));
        assert_eq!(resp["id"], "req-99");
        assert_eq!(resp["jsonrpc"], "2.0");
    }

    #[test]
    fn handle_tools_list_includes_tool_description() {
        let mut registry = ToolRegistry::new();
        registry.register(BashTool);
        let resp = handle_tools_list(json!(5), &registry);
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert!(
            !tools[0]["description"].as_str().unwrap_or("").is_empty(),
            "tool description should be non-empty"
        );
    }

    #[tokio::test]
    async fn test_ping_returns_empty_result() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}});
        let resp = process_request(req, &registry)
            .await
            .expect("should return Some");
        assert_eq!(resp["id"], 10);
        assert_eq!(resp["result"], json!({}));
    }

    #[test]
    fn test_resources_list_includes_git_status() {
        let resp = handle_resources_list(json!(1));
        let resources = resp["result"]["resources"]
            .as_array()
            .expect("resources array");
        let git = resources.iter().find(|r| r["uri"] == "dm://git/status");
        assert!(git.is_some(), "should always include git status resource");
        assert_eq!(git.unwrap()["mimeType"], "text/plain");
    }

    #[tokio::test]
    async fn test_resources_read_unknown_uri_returns_error() {
        let resp = handle_resources_read(json!(1), json!({"uri": "dm://nonexistent"})).await;
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unknown resource"));
    }

    #[tokio::test]
    async fn test_resources_read_missing_uri_returns_error() {
        let resp = handle_resources_read(json!(1), json!({})).await;
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Missing"));
    }

    #[tokio::test]
    async fn test_resources_read_chain_state_no_chain() {
        let resp = handle_resources_read(json!(1), json!({"uri": "dm://chain/state"})).await;
        assert_eq!(resp["error"]["code"], -32603);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No chain"));
    }

    #[tokio::test]
    async fn test_resources_read_git_status() {
        let resp = handle_resources_read(json!(1), json!({"uri": "dm://git/status"})).await;
        // Should succeed (we're in a git repo)
        let contents = resp["result"]["contents"].as_array();
        assert!(contents.is_some(), "should return contents: {}", resp);
        let first = &contents.unwrap()[0];
        assert_eq!(first["uri"], "dm://git/status");
        assert_eq!(first["mimeType"], "text/plain");
    }

    #[tokio::test]
    async fn test_prompts_list_has_four_prompts() {
        let registry = ToolRegistry::new();
        let req = json!({"jsonrpc": "2.0", "id": 7, "method": "prompts/list", "params": {}});
        let resp = process_request(req, &registry)
            .await
            .expect("should return Some");
        let prompts = resp["result"]["prompts"].as_array().expect("prompts array");
        assert_eq!(prompts.len(), 4);
        let names: Vec<&str> = prompts.iter().filter_map(|p| p["name"].as_str()).collect();
        assert!(names.contains(&"commit-message"));
        assert!(names.contains(&"code-review"));
        assert!(names.contains(&"explain-code"));
        assert!(names.contains(&"summarize"));
    }

    #[test]
    fn test_prompts_list_each_has_name_and_description() {
        let resp = handle_prompts_list(json!(1));
        let prompts = resp["result"]["prompts"].as_array().unwrap();
        for p in prompts {
            assert!(p["name"].is_string(), "prompt missing name: {:?}", p);
            assert!(
                p["description"].is_string(),
                "prompt missing description: {:?}",
                p
            );
            assert!(
                p["arguments"].is_array(),
                "prompt missing arguments: {:?}",
                p
            );
        }
    }

    #[test]
    fn test_initialize_advertises_resources_capability() {
        let resp = handle_initialize(json!(1));
        let caps = &resp["result"]["capabilities"];
        assert!(caps["tools"].is_object(), "should have tools capability");
        assert!(
            caps["resources"].is_object(),
            "should have resources capability"
        );
        assert!(
            caps["prompts"].is_object(),
            "should have prompts capability"
        );
    }

    #[test]
    fn test_resource_response_structure() {
        let resp = resource_response(json!(1), "dm://test", "text/plain", "hello");
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        let contents = resp["result"]["contents"]
            .as_array()
            .expect("contents array");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["uri"], "dm://test");
        assert_eq!(contents[0]["mimeType"], "text/plain");
        assert_eq!(contents[0]["text"], "hello");
    }

    #[test]
    fn test_prompts_get_commit_message() {
        let resp = handle_prompts_get(
            json!(1),
            json!({"name": "commit-message", "arguments": {"diff": "+added line"}}),
        );
        let messages = resp["result"]["messages"]
            .as_array()
            .expect("messages array");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        let text = messages[0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("+added line"), "should include the diff");
        assert!(
            text.contains("conventional commits"),
            "should mention format"
        );
    }

    #[test]
    fn test_prompts_get_code_review_default_focus() {
        let resp = handle_prompts_get(
            json!(1),
            json!({"name": "code-review", "arguments": {"diff": "-old\n+new"}}),
        );
        let text = resp["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(
            text.contains("correctness"),
            "default focus should mention correctness"
        );
        assert!(text.contains("-old\n+new"));
    }

    #[test]
    fn test_prompts_get_code_review_security_focus() {
        let resp = handle_prompts_get(
            json!(1),
            json!({"name": "code-review", "arguments": {"diff": "x", "focus": "security"}}),
        );
        let text = resp["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(
            text.contains("security vulnerabilities"),
            "should include security focus"
        );
    }

    #[test]
    fn test_prompts_get_explain_code_with_language() {
        let resp = handle_prompts_get(
            json!(1),
            json!({"name": "explain-code", "arguments": {"code": "fn main() {}", "language": "rust"}}),
        );
        let text = resp["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("(rust)"), "should note the language");
        assert!(text.contains("fn main() {}"));
    }

    #[test]
    fn test_prompts_get_summarize_short() {
        let resp = handle_prompts_get(
            json!(1),
            json!({"name": "summarize", "arguments": {"content": "long text here", "length": "short"}}),
        );
        let text = resp["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(
            text.contains("1-2 sentences"),
            "short should request 1-2 sentences"
        );
        assert!(text.contains("long text here"));
    }

    #[test]
    fn test_prompts_get_unknown_returns_error() {
        let resp = handle_prompts_get(json!(1), json!({"name": "nonexistent"}));
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Unknown prompt"));
    }

    #[test]
    fn test_prompts_get_missing_name_returns_error() {
        let resp = handle_prompts_get(json!(1), json!({}));
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Missing"));
    }

    // ── is_notification ─────────────────────────────────────────────────────

    #[test]
    fn is_notification_true_for_null_id_and_notifications_method() {
        assert!(is_notification(
            &json!({"id": null, "method": "notifications/cancelled"})
        ));
    }

    #[test]
    fn is_notification_true_for_missing_id() {
        assert!(is_notification(
            &json!({"method": "notifications/progress"})
        ));
    }

    #[test]
    fn is_notification_false_for_non_null_id() {
        assert!(!is_notification(
            &json!({"id": 1, "method": "notifications/cancelled"})
        ));
    }

    #[test]
    fn is_notification_false_for_non_notification_method() {
        assert!(!is_notification(
            &json!({"id": null, "method": "tools/list"})
        ));
    }

    // ── bounded_arg / bounded_content ───────────────────────────────────────

    #[test]
    fn bounded_arg_short_input_unchanged() {
        let args = json!({"diff": "small diff"});
        let result = bounded_arg(&args, "diff");
        assert_eq!(result, "small diff");
    }

    #[test]
    fn bounded_arg_truncates_large_input() {
        let big = "x".repeat(600_000);
        let args = json!({"diff": big});
        let result = bounded_arg(&args, "diff");
        assert!(result.len() < 600_000, "should be truncated");
        assert!(
            result.contains("truncated at 500000 bytes"),
            "result: {}...",
            &result[result.len() - 60..]
        );
    }

    #[test]
    fn bounded_arg_missing_key_returns_empty() {
        let args = json!({"other": "value"});
        assert_eq!(bounded_arg(&args, "diff"), "");
    }

    #[test]
    fn bounded_content_short_unchanged() {
        assert_eq!(bounded_content("hello"), "hello");
    }

    #[test]
    fn bounded_content_truncates_large() {
        let big = "y".repeat(2_000_000);
        let result = bounded_content(&big);
        assert!(result.len() < 2_000_000);
        assert!(result.contains("truncated at 1000000 bytes"));
    }

    #[test]
    fn test_prompts_get_messages_have_user_role() {
        for name in &["commit-message", "code-review", "explain-code", "summarize"] {
            let resp = handle_prompts_get(
                json!(1),
                json!({"name": name, "arguments": {"diff": "x", "code": "x", "content": "x"}}),
            );
            let messages = resp["result"]["messages"]
                .as_array()
                .unwrap_or_else(|| panic!("no messages for {}", name));
            for msg in messages {
                assert_eq!(msg["role"], "user", "prompt {} should have user role", name);
            }
        }
    }
}
