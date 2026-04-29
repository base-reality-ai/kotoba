//! Acceptance test for Tier 4 — compaction in host mode.
//!
//! The host session path threads the session's project root into canonical
//! `compact_pipeline_with_failures`. This test drives that public pipeline
//! with a tiny context window and a mock Ollama summary response, then asserts
//! the compact synthesis lands under the host project's `.dm/wiki`, is indexed,
//! and is findable through the same search path `wiki_search` uses.

use dark_matter::compaction::{
    compact_pipeline_with_failures, CompactionStage, CompactionThresholds,
};
use dark_matter::ollama::client::OllamaClient;
use dark_matter::wiki::{Layer, Wiki};
use serde_json::json;
use std::sync::Mutex;
use tempfile::TempDir;

static CWD_LOCK: Mutex<()> = Mutex::new(());
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    auto_ingest: Option<String>,
}

impl EnvGuard {
    fn enable_wiki_auto_ingest() -> Self {
        let auto_ingest = std::env::var("DM_WIKI_AUTO_INGEST").ok();
        std::env::set_var("DM_WIKI_AUTO_INGEST", "1");
        Self { auto_ingest }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.auto_ingest {
            Some(v) => std::env::set_var("DM_WIKI_AUTO_INGEST", v),
            None => std::env::remove_var("DM_WIKI_AUTO_INGEST"),
        }
    }
}

fn large_message(role: &str, label: &str) -> serde_json::Value {
    json!({
        "role": role,
        "content": format!(
            "{label}: host compaction routing sentinel. src/lesson.rs\n{}",
            "日本語の学習履歴と second brain context. ".repeat(260)
        )
    })
}

fn compact_trigger_messages() -> Vec<serde_json::Value> {
    let mut messages = vec![json!({
        "role": "system",
        "content": "host-mode compaction system prompt"
    })];

    for i in 0..8 {
        messages.push(large_message("user", &format!("middle-{i}")));
    }
    for i in 0..3 {
        messages.push(large_message("assistant", &format!("tail-{i}")));
    }

    messages
}

#[tokio::test]
async fn host_mode_compaction_lands_under_project_dm_and_is_searchable() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _env_guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let _env = EnvGuard::enable_wiki_auto_ingest();

    let project = TempDir::new().expect("project tempdir");
    let wrong_cwd = TempDir::new().expect("wrong cwd tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(wrong_cwd.path()).expect("chdir wrong cwd");

    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"message":{"role":"assistant","content":"host compact indexed sentinel summary"},"prompt_eval_count":0,"eval_count":0,"eval_duration":0}"#,
        )
        .expect(1)
        .create_async()
        .await;

    let client = OllamaClient::new(server.url(), "host-compact-model".to_string());
    let thresholds = CompactionThresholds::from_context_window(2000).with_keep_tail(3);
    let mut messages = compact_trigger_messages();
    let mut failures = 0usize;

    let result = compact_pipeline_with_failures(
        &mut messages,
        &client,
        &thresholds,
        false,
        &mut failures,
        Some(project.path()),
    )
    .await;

    assert!(
        matches!(result.stage, CompactionStage::FullSummary { .. }),
        "expected full-summary compaction, got {:?}",
        result.stage
    );
    assert_eq!(failures, 0);

    let project_synth = project.path().join(".dm/wiki/synthesis");
    let compact_pages: Vec<_> = std::fs::read_dir(&project_synth)
        .expect("project synthesis dir")
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("compact-"))
        .collect();
    assert_eq!(
        compact_pages.len(),
        1,
        "expected exactly one host project compact page under {:?}",
        project_synth
    );

    let wrong_synth = wrong_cwd.path().join(".dm/wiki/synthesis");
    assert!(
        !wrong_synth.exists(),
        "compaction should use the threaded host project root, not process cwd"
    );

    let wiki = Wiki::open(project.path()).expect("open project wiki");
    let idx = wiki.load_index().expect("load index");
    let compact_entry = idx
        .entries
        .iter()
        .find(|entry| entry.path.starts_with("synthesis/compact-"))
        .expect("compact synthesis should be indexed");
    let page = wiki
        .read_page(&compact_entry.path)
        .expect("read compact synthesis page");
    assert_eq!(
        page.layer,
        Layer::Kernel,
        "documented Gap #8: compaction synthesis is host-session state but \
         canonical write_compact_synthesis hardcodes Layer::Kernel"
    );

    let hits = wiki
        .search("host compact indexed sentinel")
        .expect("wiki search");
    assert!(
        hits.iter()
            .any(|hit| hit.path.starts_with("synthesis/compact-")),
        "subsequent wiki_search should find the compact summary, got: {:?}",
        hits.iter().map(|hit| &hit.path).collect::<Vec<_>>()
    );

    mock.assert_async().await;

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}
