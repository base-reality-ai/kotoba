//! Integration test: the `HostCapabilities` install hook is the canonical
//! entry point for host projects to add tools. This test asserts that an
//! installed host capability is automatically merged into every registry
//! produced by [`dark_matter::tools::registry::default_registry`] — the same
//! constructor the TUI, daemon, web, chain, and sub-agent paths all use.
//!
//! Lives in its own integration-test binary because
//! `install_host_capabilities` writes to a process-global `OnceLock`. Running
//! in a separate binary keeps it isolated from the lib's unit tests, which
//! must not see installed host caps.

use async_trait::async_trait;
use dark_matter::host::{install_host_capabilities, HostCapabilities};
use dark_matter::tools::registry::{
    default_registry, default_registry_with_events, sub_agent_registry, ToolRegistry,
};
use dark_matter::tools::{Tool, ToolResult};
use dark_matter::tui::BackendEvent;
use serde_json::{json, Value};

struct ProbeHost;

impl HostCapabilities for ProbeHost {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostProbeTool)]
    }
}

struct HostProbeTool;

#[async_trait]
impl Tool for HostProbeTool {
    fn name(&self) -> &'static str {
        "host_probe"
    }
    fn description(&self) -> &'static str {
        "Probe tool used by the integration test for installed host capabilities."
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
            content: format!("probe:{echo}"),
            is_error: false,
        })
    }
}

/// Single test, single binary: install once, then assert the install
/// reaches every constructor path. A second install attempt errors so a
/// misconfigured host crate cannot accidentally swap caps mid-process.
#[tokio::test]
async fn install_host_capabilities_reaches_default_registry() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // Pre-condition: a registry built before any install holds NO host tool.
    // This guards against accidentally promoting a leaked global from a prior
    // test binary or a stale OnceLock import.
    let pre = ToolRegistry::new();
    assert!(
        !pre.tool_names().contains(&"host_probe"),
        "fresh registry should not see host_probe before install"
    );

    install_host_capabilities(Box::new(ProbeHost)).expect("first install");

    // Second install must error — process-global state is one-shot.
    let err = install_host_capabilities(Box::new(ProbeHost)).expect_err("second install");
    let msg = err.to_string();
    assert!(msg.contains("already installed"), "msg: {msg}");
    assert!(msg.contains("Try:"), "missing actionable hint: {msg}");

    // Default registry merges installed host tool.
    let r = default_registry(
        "test-session",
        tmp.path(),
        "http://localhost:11434",
        "gemma:1b",
        "mxbai-embed-large",
    );
    assert!(
        r.tool_names().contains(&"host_probe"),
        "default_registry must merge installed host tools; have: {:?}",
        r.tool_names()
    );

    // Same merge applies to sub-agent registries — sub-agents in a host
    // project should see host tools too.
    let sub = sub_agent_registry(
        "test-session",
        tmp.path(),
        "http://localhost:11434",
        "gemma:1b",
        "mxbai-embed-large",
    );
    assert!(
        sub.tool_names().contains(&"host_probe"),
        "sub_agent_registry must merge installed host tools; have: {:?}",
        sub.tool_names()
    );

    // The TUI's interactive run path uses `default_registry_with_events` to
    // wire `AgentTool` to the BackendEvent channel. The install hook must
    // reach that constructor too — otherwise the running TUI would silently
    // miss host capabilities while headless / sub-agent paths see them.
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel::<BackendEvent>(1);
    let with_events = default_registry_with_events(
        "test-session",
        tmp.path(),
        "http://localhost:11434",
        "gemma:1b",
        "mxbai-embed-large",
        event_tx,
    );
    assert!(
        with_events.tool_names().contains(&"host_probe"),
        "default_registry_with_events must merge installed host tools; have: {:?}",
        with_events.tool_names()
    );

    // End-to-end: the host tool dispatches through the same `call(...)` path
    // every kernel tool uses. This is the directive's "callable from the
    // running TUI" guarantee — the TUI uses this exact constructor.
    let res = r
        .call("host_probe", json!({ "echo": "wired" }))
        .await
        .expect("dispatch");
    assert!(!res.is_error, "host_probe should succeed: {}", res.content);
    assert_eq!(res.content, "probe:wired");

    // Negative coverage: the registry continues to expose kernel tools too —
    // host caps augment, never replace.
    assert!(
        r.tool_names().contains(&"bash"),
        "kernel tools must remain available alongside host tools; have: {:?}",
        r.tool_names()
    );
}
