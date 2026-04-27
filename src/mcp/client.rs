use super::types::{JsonRpcRequest, JsonRpcResponse, McpTool};
use anyhow::{Context, Result};
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

#[cfg(not(test))]
const TOOL_CALL_TIMEOUT_SECS: u64 = 30;
#[cfg(test)]
const TOOL_CALL_TIMEOUT_SECS: u64 = 1; // fast in tests

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

pub struct McpClient {
    #[allow(dead_code)]
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    /// Display name used in error messages (derived from the command path).
    pub name_hint: String,
}

impl McpClient {
    /// Spawn an MCP server by command + args and perform the initialize handshake.
    pub async fn spawn(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true) // kill child when McpClient is dropped
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server: {}", command))?;

        let stdin = child
            .stdin
            .take()
            .context("Failed to get stdin of MCP server")?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .context("Failed to get stdout of MCP server")?,
        );

        // Derive a readable name from the command path
        let name_hint = std::path::Path::new(command)
            .file_name()
            .map_or_else(|| command.to_string(), |n| n.to_string_lossy().to_string());

        let mut client = McpClient {
            child,
            stdin,
            stdout,
            next_id: 1,
            name_hint: name_hint.clone(),
        };

        // Perform the initialize handshake
        client
            .send_request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "dm",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;

        let init_resp = tokio::time::timeout(
            std::time::Duration::from_secs(TOOL_CALL_TIMEOUT_SECS),
            client.recv_response(),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "MCP server '{}' did not respond to initialize within {}s",
                name_hint,
                TOOL_CALL_TIMEOUT_SECS
            )
        })?
        .context("MCP initialize failed")?;

        // Warn on protocol version mismatch
        if let Some(result) = &init_resp.result {
            let server_version = result["protocolVersion"].as_str().unwrap_or("unknown");
            if server_version != MCP_PROTOCOL_VERSION {
                crate::warnings::push_warning(format!(
                    "MCP plugin '{}': protocol version '{}', expected '{}'",
                    name_hint, server_version, MCP_PROTOCOL_VERSION
                ));
            }
        }

        client
            .send_notification("notifications/initialized", serde_json::json!({}))
            .await?;

        Ok(client)
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(id)
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        #[derive(serde::Serialize)]
        struct Notification {
            jsonrpc: &'static str,
            method: String,
            params: Value,
        }
        let n = Notification {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
        };
        let mut line = serde_json::to_string(&n)?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn recv_response(&mut self) -> Result<JsonRpcResponse> {
        const MAX_MCP_RESPONSE_BYTES: usize = 2_000_000;
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .await
            .with_context(|| format!("Failed to read from plugin '{}'", self.name_hint))?;
        if line.len() > MAX_MCP_RESPONSE_BYTES {
            anyhow::bail!(
                "Plugin '{}' response too large ({} bytes, limit {})",
                self.name_hint,
                line.len(),
                MAX_MCP_RESPONSE_BYTES,
            );
        }
        serde_json::from_str(&line).map_err(|e| {
            let mut preview_end = line.len().min(120);
            while preview_end > 0 && !line.is_char_boundary(preview_end) {
                preview_end -= 1;
            }
            let preview = &line[..preview_end];
            anyhow::anyhow!(
                "Plugin '{}' sent malformed JSON: {} (line: {:?})",
                self.name_hint,
                e,
                preview
            )
        })
    }

    /// Call `tools/list` and return the available tools.
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        self.send_request("tools/list", serde_json::json!({}))
            .await?;
        let resp = tokio::time::timeout(
            std::time::Duration::from_secs(TOOL_CALL_TIMEOUT_SECS),
            self.recv_response(),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "MCP server did not respond to tools/list within {}s",
                TOOL_CALL_TIMEOUT_SECS
            )
        })?
        .context("MCP tools/list failed")?;
        if let Some(err) = resp.error {
            anyhow::bail!("MCP tools/list error {}: {}", err.code, err.message);
        }
        let tools_val = resp
            .result
            .and_then(|v| v.get("tools").cloned())
            .unwrap_or(serde_json::json!([]));
        serde_json::from_value(tools_val).context("Failed to parse MCP tool list")
    }

    /// Invoke a tool by name and return its text content.
    /// Returns `Err` if the call takes longer than `TOOL_CALL_TIMEOUT_SECS`.
    pub async fn call_tool(&mut self, name: &str, args: Value) -> Result<String> {
        let name_owned = name.to_string();
        let hint = self.name_hint.clone();
        let fut = self.call_tool_inner(name, args);
        tokio::time::timeout(std::time::Duration::from_secs(TOOL_CALL_TIMEOUT_SECS), fut)
            .await
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "Tool '{}' on plugin '{}' timed out after {}s",
                    name_owned,
                    hint,
                    TOOL_CALL_TIMEOUT_SECS
                ))
            })
    }

    // visible for tests
    pub(crate) async fn call_tool_inner(&mut self, name: &str, args: Value) -> Result<String> {
        self.send_request(
            "tools/call",
            serde_json::json!({
                "name": name,
                "arguments": args
            }),
        )
        .await?;
        let resp = self.recv_response().await?;
        if let Some(err) = resp.error {
            anyhow::bail!("MCP tool '{}' error {}: {}", name, err.code, err.message);
        }
        // Content is an array of { type, text } blocks
        let content = resp
            .result
            .and_then(|v| v.get("content").cloned())
            .and_then(|c| {
                c.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|b| b.get("text").and_then(|t| t.as_str()).map(String::from))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
            })
            .unwrap_or_default();
        const MAX_CONTENT: usize = 2_000_000;
        if content.len() > MAX_CONTENT {
            let truncated = crate::util::safe_truncate(&content, MAX_CONTENT);
            return Ok(format!(
                "{}…[truncated: {} bytes]",
                truncated,
                content.len()
            ));
        }
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MCP server script that responds to initialize then hangs — used to trigger timeout.
    fn hanging_server_script() -> &'static str {
        "import sys, json, time\n\
         line = sys.stdin.readline()\n\
         req = json.loads(line)\n\
         resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': '2024-11-05', 'capabilities': {}}}\n\
         sys.stdout.write(json.dumps(resp) + '\\n')\n\
         sys.stdout.flush()\n\
         sys.stdin.readline()  # read initialized notification\n\
         time.sleep(100)\n"
    }

    /// MCP server script that responds to initialize then sends malformed JSON for any tool call.
    fn malformed_json_server_script() -> &'static str {
        "import sys, json\n\
         line = sys.stdin.readline()\n\
         req = json.loads(line)\n\
         resp = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'protocolVersion': '2024-11-05', 'capabilities': {}}}\n\
         sys.stdout.write(json.dumps(resp) + '\\n')\n\
         sys.stdout.flush()\n\
         sys.stdin.readline()  # read initialized notification\n\
         sys.stdin.readline()  # read tool call\n\
         sys.stdout.write('this is not valid json\\n')\n\
         sys.stdout.flush()\n"
    }

    /// Full MCP server (flat, no indented blocks): handles initialize, tools/list, tools/call.
    fn full_server_script() -> &'static str {
        // Flat Python: no function defs or loops so Rust \n\ line-continuation (which strips
        // leading whitespace) does not destroy indentation.
        "import sys\nimport json\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}})\
         +'\\n');sys.stdout.flush()\n\
         sys.stdin.readline()\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'tools':[{'name':'echo','description':'echoes input','input_schema':{'type':'object'}}]}})\
         +'\\n');sys.stdout.flush()\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'content':[{'type':'text','text':'block1'},{'type':'text','text':'block2'}]}})\
         +'\\n');sys.stdout.flush()\n"
    }

    /// MCP server that returns an error response to tools/call.
    fn error_response_server_script() -> &'static str {
        "import sys\nimport json\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}})\
         +'\\n');sys.stdout.flush()\n\
         sys.stdin.readline()\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'error':{'code':-1,'message':'tool failed with error'}})\
         +'\\n');sys.stdout.flush()\n"
    }

    #[tokio::test]
    async fn test_call_tool_timeout() {
        let mut client = McpClient::spawn("python3", &["-c", hanging_server_script()])
            .await
            .expect("Failed to spawn mock server (is python3 available?)");
        let err = client
            .call_tool("any_tool", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("timed out"),
            "expected 'timed out' in error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_call_tool_malformed_json() {
        let mut client = McpClient::spawn("python3", &["-c", malformed_json_server_script()])
            .await
            .expect("Failed to spawn mock server (is python3 available?)");
        let err = client
            .call_tool("any_tool", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("malformed JSON"),
            "expected 'malformed JSON' in error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_list_tools_returns_tools() {
        let mut client = McpClient::spawn("python3", &["-c", full_server_script()])
            .await
            .expect("Failed to spawn full mock server");
        let tools = client.list_tools().await.expect("list_tools failed");
        assert_eq!(tools.len(), 1, "expected 1 tool");
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description.as_deref(), Some("echoes input"));
    }

    #[tokio::test]
    async fn test_call_tool_joins_multiple_text_blocks() {
        let mut client = McpClient::spawn("python3", &["-c", full_server_script()])
            .await
            .expect("Failed to spawn full mock server");
        // Drain the tools/list so the next request is tools/call
        client.list_tools().await.expect("list_tools failed");
        let result = client
            .call_tool("echo", serde_json::json!({"msg": "hi"}))
            .await
            .expect("call_tool failed");
        // The mock returns two text blocks; they should be joined with "\n"
        assert_eq!(result, "block1\nblock2");
    }

    #[tokio::test]
    async fn test_call_tool_error_response_propagates() {
        let mut client = McpClient::spawn("python3", &["-c", error_response_server_script()])
            .await
            .expect("Failed to spawn error mock server");
        let err = client
            .call_tool("any", serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("tool failed with error"),
            "error message not propagated: {}",
            err
        );
    }

    #[test]
    fn name_hint_derived_from_command_path() {
        // name_hint is set from Path::new(command).file_name() inside spawn()
        // We test the derivation logic directly
        let cmd = "/usr/local/bin/my-mcp-server";
        let name_hint = std::path::Path::new(cmd)
            .file_name()
            .map_or_else(|| cmd.to_string(), |n| n.to_string_lossy().to_string());
        assert_eq!(name_hint, "my-mcp-server");
    }

    /// MCP server that returns an empty tools list.
    fn empty_tools_server_script() -> &'static str {
        "import sys\nimport json\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}})\
         +'\\n');sys.stdout.flush()\n\
         sys.stdin.readline()\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'tools':[]}})\
         +'\\n');sys.stdout.flush()\n"
    }

    /// MCP server that returns an error for tools/list.
    fn tools_list_error_server_script() -> &'static str {
        "import sys\nimport json\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}})\
         +'\\n');sys.stdout.flush()\n\
         sys.stdin.readline()\n\
         req=json.loads(sys.stdin.readline())\n\
         sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'error':{'code':-32601,'message':'method not found'}})\
         +'\\n');sys.stdout.flush()\n"
    }

    #[tokio::test]
    async fn test_list_tools_empty_result() {
        let mut client = McpClient::spawn("python3", &["-c", empty_tools_server_script()])
            .await
            .expect("Failed to spawn empty tools mock server");
        let tools = client
            .list_tools()
            .await
            .expect("list_tools should succeed");
        assert!(tools.is_empty(), "expected 0 tools, got {}", tools.len());
    }

    #[tokio::test]
    async fn test_list_tools_error_response() {
        let mut client = McpClient::spawn("python3", &["-c", tools_list_error_server_script()])
            .await
            .expect("Failed to spawn error mock server");
        let err = client.list_tools().await.unwrap_err();
        assert!(
            err.to_string().contains("method not found"),
            "expected 'method not found' in error, got: {}",
            err
        );
    }

    #[test]
    fn mcp_content_truncation_at_boundary() {
        let content = "a".repeat(2_100_000);
        let max = 2_000_000;
        let truncated = crate::util::safe_truncate(&content, max);
        assert!(truncated.len() <= max);
        assert!(truncated.len() >= max - 4);
    }

    #[test]
    fn name_hint_falls_back_to_full_command_when_no_filename() {
        // Edge case: command with no path component (just a name)
        let cmd = "python3";
        let name_hint = std::path::Path::new(cmd)
            .file_name()
            .map_or_else(|| cmd.to_string(), |n| n.to_string_lossy().to_string());
        assert_eq!(name_hint, "python3");
    }
}
