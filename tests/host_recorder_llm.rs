//! Acceptance tests for Tier 7 recorder LLM opt-in wiring.
//!
//! The recorder remains rule-based by default. When
//! `KOTOBA_RECORDER_USE_LLM=1`, kotoba sends the saved transcript through dm's
//! conversation capture API, then feeds the model's extraction text back into
//! the existing indexed recorder path.

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

fn fixed_now() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-04-29T12:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc)
}

#[tokio::test]
async fn recorder_uses_rule_based_path_when_llm_env_unset() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _env = EnvGuard::set(&[
        ("KOTOBA_RECORDER_USE_LLM", None),
        (
            "KOTOBA_RECORDER_MODEL",
            Some("should-not-be-called".to_string()),
        ),
    ]);
    let project = TempDir::new().expect("tempdir");

    let summary = host_caps::record_session_in_with_optional_llm(
        project.path(),
        "Yuki: New word: 猫 (ねこ) means cat.",
        "Yuki",
        fixed_now(),
        "should-not-be-called",
    )
    .await
    .expect("rule recorder");

    assert_eq!(summary.vocabulary_count, 1);
    assert_eq!(summary.struggle_count, 0);
    assert!(project
        .path()
        .join(".dm/wiki/entities/Vocabulary/猫.md")
        .exists());
}

#[tokio::test]
async fn recorder_uses_configured_model_when_llm_env_set() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let mut server = mockito::Server::new_async().await;
    let host = server.host_with_port();
    let _env = EnvGuard::set(&[
        ("KOTOBA_RECORDER_USE_LLM", Some("1".to_string())),
        ("KOTOBA_RECORDER_MODEL", Some("mock-recorder".to_string())),
        ("OLLAMA_HOST", Some(host)),
    ]);

    let mock = server
        .mock("POST", "/api/chat")
        .match_body(Matcher::PartialJson(json!({
            "model": "mock-recorder"
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {
                    "role": "assistant",
                    "content": "Recorder: New word: 学校 (がっこう) means school.\nLearner: I don't know particle confusion."
                },
                "prompt_eval_count": 13,
                "eval_count": 9,
                "eval_duration": 0
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let summary = host_caps::record_session_in_with_optional_llm(
        project.path(),
        "Learner and Yuki discussed old homework without extraction syntax.",
        "Yuki",
        fixed_now(),
        "mock-recorder",
    )
    .await
    .expect("llm recorder");

    assert_eq!(summary.vocabulary_count, 1);
    assert_eq!(summary.struggle_count, 1);
    assert!(project
        .path()
        .join(".dm/wiki/entities/Vocabulary/学校.md")
        .exists());

    let wiki = dark_matter::wiki::Wiki::open(project.path()).expect("open wiki");
    let results = wiki.search("school").expect("wiki search");
    assert!(
        results
            .iter()
            .any(|hit| hit.path == "entities/Vocabulary/学校.md"),
        "expected LLM-extracted vocab to be indexed, got: {results:?}"
    );

    mock.assert_async().await;
}
