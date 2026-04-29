//! Acceptance test for kotoba's host_caps migration to
//! `Wiki::register_host_page` (the second-brain promise).
//!
//! Pre-canonical-9259acb, kotoba's host capabilities wrote vocabulary
//! and kanji pages via raw `std::fs::write`. Those writes landed on
//! disk but never made it into the wiki index — `wiki_search` and
//! `wiki_lookup` couldn't see them. The whole "your second brain
//! accumulates everything you encounter" promise was structurally
//! broken because every host-authored page was invisible.
//!
//! Canonical 9259acb shipped `Wiki::register_host_page`. Kotoba then
//! migrated `log_vocabulary_in` and the kanji write site to use it.
//! This test pins the migration: after `host_log_vocabulary` runs,
//! `wiki_search` must find the word.
//!
//! If this test breaks, the second-brain promise is regressing.

use dark_matter::tools::Tool;
use dark_matter::wiki::Wiki;
use serde_json::json;
use std::sync::Mutex;
use tempfile::TempDir;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

/// Process-shared mutex around cwd mutation. Tests that change
/// `std::env::current_dir()` must hold this lock, otherwise parallel
/// test execution races and one test's cwd leaks into another's
/// `project_root()` lookup. Mirrors the kernel's `CWD_LOCK` pattern.
static CWD_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn host_log_vocabulary_results_in_wiki_searchable_page() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    // Initialize the wiki layout so register_host_page has somewhere
    // to write the index against.
    Wiki::open(project.path()).expect("open wiki");

    let tool = host_caps::LogVocabularyTool;
    let result = tool
        .call(json!({
            "kanji": "学校",
            "kana": "がっこう",
            "romaji": "gakkou",
            "meaning": "school",
            "pos": "noun",
            "jlpt": 5,
            "example_japanese": "学校に行きます。",
            "example_english": "I go to school.",
        }))
        .await
        .expect("host_log_vocabulary");

    assert!(
        !result.is_error,
        "host_log_vocabulary returned an error: {}",
        result.content
    );

    let wiki = Wiki::open(project.path()).expect("re-open wiki");

    // Search by English meaning — proves the body indexed and is
    // searchable, not just file-system reachable.
    let hits = wiki.search("school").expect("wiki search");
    assert!(
        !hits.is_empty(),
        "wiki_search('school') returned no hits — host_log_vocabulary \
         did not flow through register_host_page or the body did not \
         contain the meaning"
    );
    assert!(
        hits.iter().any(|h| h.path.contains("学校")),
        "expected hit on entities/Vocabulary/学校.md, got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    // Search by kana — proves the body indexed multiple terms.
    let hits = wiki.search("がっこう").expect("wiki search by kana");
    assert!(!hits.is_empty(), "wiki_search('がっこう') returned no hits");

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}

#[tokio::test]
async fn host_log_vocabulary_twice_upserts_index() {
    // Logging the same word twice (e.g. mastery promotion) must not
    // create a duplicate `IndexEntry`. `register_host_page` upserts by
    // relative path; the two writes resolve to the same slug, so the
    // index keeps exactly one entry.
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    Wiki::open(project.path()).expect("open wiki");

    let tool = host_caps::LogVocabularyTool;
    let first_args = json!({
        "kanji": "学校",
        "kana": "がっこう",
        "romaji": "gakkou",
        "meaning": "school",
        "pos": "noun",
    });
    tool.call(first_args.clone()).await.expect("first log");
    tool.call(first_args).await.expect("second log");

    let wiki = Wiki::open(project.path()).expect("re-open wiki");
    let idx = wiki.load_index().expect("load index");
    let vocab_hits: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.path.contains("学校"))
        .collect();
    assert_eq!(
        vocab_hits.len(),
        1,
        "expected exactly 1 index entry for 学校 after two writes \
         (upsert), got: {:?}",
        vocab_hits.iter().map(|e| &e.path).collect::<Vec<_>>()
    );

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}

#[tokio::test]
async fn host_log_kanji_results_in_wiki_searchable_page() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    Wiki::open(project.path()).expect("open wiki");

    let tool = host_caps::LogKanjiTool;
    let result = tool
        .call(json!({
            "character": "学",
            "meaning": "study, learning",
            "onyomi": ["ガク"],
            "kunyomi": ["まな(ぶ)"],
            "radicals": ["⺍", "冖", "子"],
            "jlpt": 5,
            "mnemonic": "Three children studying under a roof.",
        }))
        .await
        .expect("host_log_kanji");

    assert!(
        !result.is_error,
        "host_log_kanji returned an error: {}",
        result.content
    );

    let wiki = Wiki::open(project.path()).expect("re-open wiki");

    let hits = wiki.search("learning").expect("wiki search");
    assert!(
        hits.iter().any(|h| h.path.contains("学")),
        "expected wiki_search('learning') to find entities/Kanji/学.md, \
         got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}
