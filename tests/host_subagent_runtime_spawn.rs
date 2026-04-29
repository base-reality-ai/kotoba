//! Acceptance test for Tier 6 — actual sub-agent runtime spawn in host mode.
//!
//! `host_subagent_kotoba_caps.rs` pins the structural registry contract. This
//! test drives the real `agent` tool through `default_registry.call(...)`,
//! then has the spawned sub-agent call a kotoba host capability. That proves
//! runtime sub-agent execution inherits the host capability install, not just
//! that `sub_agent_registry` can be constructed in isolation.

use dark_matter::host::install_host_capabilities;
use dark_matter::tools::registry::default_registry;
use dark_matter::tools::Tool;
use mockito::Matcher;
use serde_json::json;
use std::sync::Mutex;
use tempfile::TempDir;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

static CWD_LOCK: Mutex<()> = Mutex::new(());

struct CwdGuard {
    prior: std::path::PathBuf,
}

impl CwdGuard {
    fn chdir(path: &std::path::Path) -> Self {
        let prior = std::env::current_dir().expect("current dir");
        std::env::set_current_dir(path).expect("chdir project");
        Self { prior }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.prior);
    }
}

fn write_host_identity(project_root: &std::path::Path) {
    let dm = project_root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"kotoba\"\n",
    )
    .expect("write identity.toml");
}

#[tokio::test]
async fn agent_tool_spawned_sub_agent_can_call_kotoba_host_capability() {
    let _cwd = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path());
    let _cwd_guard = CwdGuard::chdir(project.path());

    install_host_capabilities(Box::new(host_caps::KotobaCapabilities))
        .expect("install kotoba caps");

    host_caps::LogVocabularyTool
        .call(json!({
            "kanji": "学校",
            "kana": "がっこう",
            "meaning": "school",
            "example_japanese": "学校に行きます。",
            "example_english": "I go to school."
        }))
        .await
        .expect("seed vocabulary");

    let mut server = mockito::Server::new_async().await;
    let base_url = format!("http://{}/api", server.host_with_port());

    let tool_call_response = server
        .mock("POST", "/api/chat")
        .match_body(Matcher::Regex("host_quiz_me".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "host_quiz_me",
                                "arguments": { "count": 1 }
                            }
                        }
                    ]
                },
                "prompt_eval_count": 17,
                "eval_count": 5,
                "eval_duration": 0
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let final_response = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {
                    "role": "assistant",
                    "content": "runtime-subagent-saw-kotoba-host-cap: 学校"
                },
                "prompt_eval_count": 11,
                "eval_count": 7,
                "eval_duration": 0
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let registry = default_registry(
        "kotoba-parent-session",
        &project.path().join(".dm"),
        &base_url,
        "mock-sub-agent",
        "mxbai-embed-large",
    );

    let result = registry
        .call(
            "agent",
            json!({
                "prompt": "Use kotoba's host quiz tool once, then report the word.",
                "model": "mock-sub-agent"
            }),
        )
        .await
        .expect("agent tool call");

    assert!(!result.is_error, "agent call errored: {}", result.content);
    assert!(
        result
            .content
            .contains("runtime-subagent-saw-kotoba-host-cap: 学校"),
        "expected final sub-agent response, got: {}",
        result.content
    );

    tool_call_response.assert_async().await;
    final_response.assert_async().await;
}
