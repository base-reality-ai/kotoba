//! Acceptance test for the Phase 2b host_caps migration: persona session
//! entries now route through `Wiki::register_host_page`.
//!
//! Persona pages need host-defined metadata (`sessions_count`). Canonical
//! `WikiPage` does not preserve arbitrary frontmatter, so kotoba stores the
//! counter in the markdown body as `- **Sessions:** N` and registers the page
//! after every recorder update. This keeps persona continuity searchable while
//! documenting the canonical frontmatter-extension gap.

use dark_matter::wiki::Wiki;
use std::sync::Mutex;
use tempfile::TempDir;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

static CWD_LOCK: Mutex<()> = Mutex::new(());

fn fixed_now() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-04-27T12:30:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc)
}

#[test]
fn first_session_creates_indexed_persona_page() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");

    Wiki::open(project.path()).expect("open wiki");

    let summary = host_caps::record_session_in(
        project.path(),
        "Yuki: New word: 学校 (がっこう) means school.",
        "Yuki",
        fixed_now(),
    )
    .expect("record session");
    assert_eq!(summary.sessions_count, 1);

    let persona_path = project.path().join(".dm/wiki/entities/Persona/Yuki.md");
    let persona = std::fs::read_to_string(&persona_path).expect("persona page");
    assert!(persona.contains("- **Sessions:** 1"));
    assert!(persona.contains("## Sessions log"));
    assert!(persona.contains("words introduced: 学校 / がっこう (school)"));

    let wiki = Wiki::open(project.path()).expect("re-open wiki");
    let hits = wiki.search("words introduced").expect("wiki search");
    assert!(
        hits.iter().any(|h| h.path == "entities/Persona/Yuki.md"),
        "expected persona session log to be searchable, got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    let idx = wiki.load_index().expect("load index");
    assert!(
        idx.entries
            .iter()
            .any(|e| e.path == "entities/Persona/Yuki.md"),
        "persona page should be indexed"
    );
}

#[test]
fn subsequent_sessions_increment_and_upsert_persona_index() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");

    Wiki::open(project.path()).expect("open wiki");
    host_caps::record_session_in(
        project.path(),
        "Yuki: New word: 猫 (ねこ) means cat.",
        "Yuki",
        fixed_now(),
    )
    .expect("first session");
    let second = host_caps::record_session_in(
        project.path(),
        "Learner: I don't know は vs が?",
        "Yuki",
        fixed_now(),
    )
    .expect("second session");
    assert_eq!(second.sessions_count, 2);

    let persona = std::fs::read_to_string(project.path().join(".dm/wiki/entities/Persona/Yuki.md"))
        .expect("persona page");
    assert!(persona.contains("- **Sessions:** 2"));
    assert!(persona.contains("words introduced: 猫 / ねこ (cat)"));
    assert!(persona.contains("struggles flagged: は vs が"));

    let wiki = Wiki::open(project.path()).expect("re-open wiki");
    let idx = wiki.load_index().expect("load index");
    let persona_entries: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.path == "entities/Persona/Yuki.md")
        .collect();
    assert_eq!(
        persona_entries.len(),
        1,
        "expected persona page index entry to upsert, got: {:?}",
        persona_entries.iter().map(|e| &e.path).collect::<Vec<_>>()
    );

    let hits = wiki.search("は vs が").expect("wiki search");
    assert!(
        hits.iter().any(|h| h.path == "entities/Persona/Yuki.md"),
        "persona session log should be searchable for the struggle, got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );
}

#[test]
fn multi_persona_session_updates_correct_personas() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");

    Wiki::open(project.path()).expect("open wiki");
    let transcript = format!(
        "\
Yuki: New word: 学校 (がっこう) means school.
tool: {}Hiro'{}
Hiro: New word: 猫 (ねこ) means cat.
",
        host_caps::PERSONA_LOADED_PREFIX,
        host_caps::PERSONA_LOADED_AFTER_NAME
    );

    let summary = host_caps::record_session_in(project.path(), &transcript, "Yuki", fixed_now())
        .expect("record multi-persona session");
    assert_eq!(summary.persona, "Yuki, Hiro");

    let yuki = std::fs::read_to_string(project.path().join(".dm/wiki/entities/Persona/Yuki.md"))
        .expect("Yuki page");
    let hiro = std::fs::read_to_string(project.path().join(".dm/wiki/entities/Persona/Hiro.md"))
        .expect("Hiro page");
    assert!(yuki.contains("- **Sessions:** 1"));
    assert!(hiro.contains("- **Sessions:** 1"));
    assert!(yuki.contains("学校 / がっこう (school)"));
    assert!(!yuki.contains("猫 / ねこ (cat)"));
    assert!(hiro.contains("猫 / ねこ (cat)"));
    assert!(!hiro.contains("学校 / がっこう (school)"));

    let wiki = Wiki::open(project.path()).expect("re-open wiki");
    let idx = wiki.load_index().expect("load index");
    assert!(idx
        .entries
        .iter()
        .any(|e| e.path == "entities/Persona/Yuki.md"));
    assert!(idx
        .entries
        .iter()
        .any(|e| e.path == "entities/Persona/Hiro.md"));
}
