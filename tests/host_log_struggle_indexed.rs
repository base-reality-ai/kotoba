//! Acceptance test for the Phase 2a host_caps migration: struggles
//! synthesis pages now route through `Wiki::register_host_page`.
//!
//! Pre-migration, `record_struggle_in` wrote `.dm/wiki/synthesis/struggles-
//! <date>.md` via raw `std::fs::write`. The kernel's `ingest_file_internal`
//! skips anything inside `.dm/wiki/`, so struggles never made it into the
//! wiki index — `wiki_search("particle confusion")` returned no hits even
//! though the page existed on disk.
//!
//! Post-migration: read existing as `WikiPage`, append to body, upsert via
//! `register_host_page`. This test pins three behaviors:
//!
//! 1. First call today creates the synthesis page **and** indexes it.
//! 2. A second call appends to the same page and does NOT create a
//!    duplicate index entry (the upsert contract on `register_host_page`).
//! 3. `wiki_search` finds the struggle by topic.
//!
//! If this test breaks, the second-brain promise for struggles is
//! regressing.

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

static CWD_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn host_record_struggle_indexes_synthesis_page() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    Wiki::open(project.path()).expect("open wiki");

    let tool = host_caps::RecordStruggleTool;
    tool.call(json!({
        "topic": "particle confusion",
        "what_got_confused": "used wa instead of ga",
    }))
    .await
    .expect("first record_struggle")
    .is_error
    .then(|| panic!("first record_struggle returned is_error=true"));

    let wiki = Wiki::open(project.path()).expect("re-open wiki");

    let hits = wiki
        .search("particle confusion")
        .expect("wiki_search particle");
    assert!(
        hits.iter()
            .any(|h| h.path.starts_with("synthesis/struggles-")),
        "wiki_search('particle confusion') should hit a synthesis/struggles- \
         page; got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}

#[tokio::test]
async fn host_record_struggle_second_call_upserts_index() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    Wiki::open(project.path()).expect("open wiki");

    let tool = host_caps::RecordStruggleTool;
    tool.call(json!({
        "topic": "particle confusion",
        "what_got_confused": "used wa instead of ga",
    }))
    .await
    .expect("first record_struggle");

    tool.call(json!({
        "topic": "verb conjugation",
        "what_got_confused": "mixed up godan with ichidan",
    }))
    .await
    .expect("second record_struggle (same date)");

    let wiki = Wiki::open(project.path()).expect("re-open wiki");

    // The two struggles share a single synthesis page (struggles-<date>.md),
    // and `register_host_page` upserts on path equality. The index must
    // therefore contain exactly one entry pointing at the synthesis page.
    let idx = wiki.load_index().expect("load index");
    let synth_entries: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.path.starts_with("synthesis/struggles-"))
        .collect();
    assert_eq!(
        synth_entries.len(),
        1,
        "expected exactly 1 index entry for today's struggles synthesis \
         page (upsert), got: {:?}",
        synth_entries.iter().map(|e| &e.path).collect::<Vec<_>>()
    );

    // Both struggles must be searchable on the same page.
    let particle_hits = wiki.search("particle confusion").expect("search");
    let verb_hits = wiki.search("verb conjugation").expect("search");
    assert!(!particle_hits.is_empty(), "first struggle not searchable");
    assert!(!verb_hits.is_empty(), "second struggle not searchable");
    assert_eq!(
        particle_hits[0].path, verb_hits[0].path,
        "both struggles should hit the same synthesis page"
    );

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}
