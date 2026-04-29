//! Tier-3 regression tests for the kotoba v0.3 daemon-protocol host
//! extension gap (canonical-side fix).
//!
//! Exercises `dark_matter::host::invoke_tool` — the dispatch primitive
//! the daemon's `host.invoke` RPC arm calls. Single-binary because
//! `install_host_capabilities` writes to a process-global `OnceLock`
//! and must not leak into other test binaries.

use async_trait::async_trait;
use dark_matter::host::{install_host_capabilities, invoke_tool, HostCapabilities, InvokeError};
use dark_matter::tools::{Tool, ToolResult};
use serde_json::{json, Value};

struct InvokeProbeHost;

impl HostCapabilities for InvokeProbeHost {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostInvokeProbeTool), Box::new(HostFailingTool)]
    }
}

struct HostInvokeProbeTool;

#[async_trait]
impl Tool for HostInvokeProbeTool {
    fn name(&self) -> &'static str {
        "host_invoke_probe"
    }
    fn description(&self) -> &'static str {
        "Probe tool used by the host.invoke regression suite."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "echo": { "type": "string" } },
            "required": ["echo"]
        })
    }
    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let echo = args.get("echo").and_then(Value::as_str).unwrap_or("");
        Ok(ToolResult {
            content: format!("invoke:{echo}"),
            is_error: false,
        })
    }
}

struct HostFailingTool;

#[async_trait]
impl Tool for HostFailingTool {
    fn name(&self) -> &'static str {
        "host_invoke_failing"
    }
    fn description(&self) -> &'static str {
        "Always returns an error from `Tool::call`."
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    async fn call(&self, _args: Value) -> anyhow::Result<ToolResult> {
        anyhow::bail!("intentional failure for regression coverage")
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn host_invoke_dispatch_covers_all_paths() {
    // (A) Pre-install: every dispatch must fail with NoHostInstalled.
    let pre = invoke_tool("host_invoke_probe", json!({"echo": "x"}))
        .await
        .expect_err("must fail before install");
    assert!(matches!(pre, InvokeError::NoHostInstalled), "{pre:?}");

    install_host_capabilities(Box::new(InvokeProbeHost)).expect("install host caps");

    // (B) Happy path: the registered tool runs and returns the
    // expected ToolResult — this is the kotoba "fix flipped" case.
    let result = invoke_tool("host_invoke_probe", json!({"echo": "wired"}))
        .await
        .expect("dispatch");
    assert!(!result.is_error, "probe should succeed: {}", result.content);
    assert_eq!(result.content, "invoke:wired");

    // (C) Missing prefix: any non-`host_` name is rejected before
    // touching the registry.
    let no_prefix = invoke_tool("bash", json!({}))
        .await
        .expect_err("non-host_ name must be rejected");
    match no_prefix {
        InvokeError::MissingPrefix(name) => assert_eq!(name, "bash"),
        other => panic!("expected MissingPrefix, got {other:?}"),
    }

    // (D) Unknown host_ tool: name passes the prefix gate but the host
    // does not register it.
    let unknown = invoke_tool("host_does_not_exist", json!({}))
        .await
        .expect_err("unknown tool must be rejected");
    match unknown {
        InvokeError::UnknownTool(name) => assert_eq!(name, "host_does_not_exist"),
        other => panic!("expected UnknownTool, got {other:?}"),
    }

    // (E) Tool-side anyhow error: bubbles up as InvokeError::Tool with
    // the original message preserved (callers can decide whether to
    // surface the inner display verbatim).
    let tool_err = invoke_tool("host_invoke_failing", json!({}))
        .await
        .expect_err("tool error must surface as InvokeError::Tool");
    match tool_err {
        InvokeError::Tool(inner) => {
            let msg = inner.to_string();
            assert!(
                msg.contains("intentional failure"),
                "inner err must preserve message: {msg}"
            );
        }
        other => panic!("expected Tool, got {other:?}"),
    }
}
