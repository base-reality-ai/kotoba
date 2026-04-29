//! Acceptance tests for Tier 7 partial — opt-in LLM planner wiring.
//!
//! Default kotoba remains rule-based so operators without API keys or a local
//! model still get a working session. When `KOTOBA_PLANNER_USE_LLM=1`, the
//! planner prompt is sent through dm's conversation capture API using
//! `KOTOBA_PLANNER_MODEL`.

use mockito::Matcher;
use serde_json::json;
use std::sync::Mutex;
use tempfile::TempDir;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn set(vars: &[(&'static str, Option<String>)]) -> Self {
        let saved = vars
            .iter()
            .map(|(key, _)| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for (key, value) in vars {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

fn today() -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(2026, 4, 29).unwrap()
}

#[tokio::test]
async fn planner_uses_rule_based_path_when_llm_env_unset() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _env = EnvGuard::set(&[
        ("KOTOBA_PLANNER_USE_LLM", None),
        (
            "KOTOBA_PLANNER_MODEL",
            Some("should-not-be-called".to_string()),
        ),
    ]);
    let project = TempDir::new().expect("tempdir");

    let brief = host_caps::plan_session_in_with_optional_llm(
        project.path(),
        "Yuki",
        3,
        today(),
        "should-not-be-called",
    )
    .await
    .expect("rule planner");

    assert!(brief.contains("# Session brief — 2026-04-29"), "{brief}");
    assert!(
        brief.contains("Wiki has no vocabulary yet"),
        "expected existing rule-based brief shape, got: {brief}"
    );
}

#[tokio::test]
async fn planner_uses_configured_model_when_llm_env_set() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let mut server = mockito::Server::new_async().await;
    let host = server.host_with_port();
    let _env = EnvGuard::set(&[
        ("KOTOBA_PLANNER_USE_LLM", Some("1".to_string())),
        ("KOTOBA_PLANNER_MODEL", Some("mock-planner".to_string())),
        ("OLLAMA_HOST", Some(host)),
    ]);

    let mock = server
        .mock("POST", "/api/chat")
        .match_body(Matcher::PartialJson(json!({
            "model": "mock-planner"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {
                    "role": "assistant",
                    "content": "# Session brief - LLM sentinel\n\nUse 学校 today."
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

    let brief = host_caps::plan_session_in_with_optional_llm(
        project.path(),
        "Yuki",
        3,
        today(),
        "mock-planner",
    )
    .await
    .expect("llm planner");

    assert!(
        brief.contains("LLM sentinel"),
        "expected mock LLM response, got: {brief}"
    );
    assert!(
        brief.contains("学校"),
        "expected response body to be returned verbatim, got: {brief}"
    );
    mock.assert_async().await;
}
