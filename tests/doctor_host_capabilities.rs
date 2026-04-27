//! Integration test for `dm doctor` host-capability visibility.
//!
//! This lives in its own integration-test binary because installing host
//! capabilities writes to a process-global `OnceLock`.

use async_trait::async_trait;
use dark_matter::config::Config;
use dark_matter::doctor::run_doctor_capture;
use dark_matter::host::{install_host_capabilities, HostCapabilities};
use dark_matter::identity::{Identity, Mode};
use dark_matter::ollama::client::OllamaClient;
use dark_matter::tools::{Tool, ToolResult};
use serde_json::{json, Value};
use std::collections::HashMap;

struct DoctorHost;

impl HostCapabilities for DoctorHost {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostVisibleTool), Box::new(MissingPrefixTool)]
    }
}

struct HostVisibleTool;

#[async_trait]
impl Tool for HostVisibleTool {
    fn name(&self) -> &'static str {
        "host_visible"
    }
    fn description(&self) -> &'static str {
        "Visible host tool for doctor output."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            content: "visible".into(),
            is_error: false,
        })
    }
}

struct MissingPrefixTool;

#[async_trait]
impl Tool for MissingPrefixTool {
    fn name(&self) -> &'static str {
        "missing_prefix"
    }
    fn description(&self) -> &'static str {
        "Deliberately invalid host tool name for doctor warning coverage."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult {
            content: "invalid".into(),
            is_error: false,
        })
    }
}

fn test_config(config_dir: std::path::PathBuf) -> Config {
    Config {
        host: "127.0.0.1:1".to_string(),
        host_is_default: false,
        model: "test-model".to_string(),
        model_is_default: false,
        tool_model: None,
        embed_model: "nomic-embed-text".to_string(),
        config_dir,
        routing: None,
        aliases: HashMap::new(),
        max_retries: 3,
        retry_delay_ms: 1000,
        max_retry_delay_ms: 30_000,
        fallback_model: None,
        snapshot_interval_secs: 300,
        idle_timeout_secs: 7200,
    }
}

#[tokio::test]
async fn doctor_capture_shows_installed_host_capabilities() {
    let tmp = tempfile::tempdir().expect("tempdir");
    install_host_capabilities(Box::new(DoctorHost)).expect("install host caps");

    let config = test_config(tmp.path().to_path_buf());
    let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
    let identity = Identity {
        mode: Mode::Host,
        host_project: Some("doctor-host".to_string()),
        canonical_dm_revision: None,
        canonical_dm_repo: None,
        source: None,
    };

    let out = run_doctor_capture(&client, &config, &identity).await;

    assert!(
        out.contains("Host capabilities:"),
        "doctor missing host capabilities section:\n{out}"
    );
    assert!(
        out.contains("installed:    2 tool(s)"),
        "doctor missing installed host tool count:\n{out}"
    );
    assert!(
        out.contains("host_visible") && out.contains("missing_prefix"),
        "doctor missing installed host tool names:\n{out}"
    );
    assert!(
        out.contains("missing `host_` prefix")
            && out.contains("Try: rename host tools to start with `host_`"),
        "doctor missing invalid-prefix hint:\n{out}"
    );
}
