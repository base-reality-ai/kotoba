//! Tier 8 — end-to-end session smoke for the kotoba binary.
//!
//! The directive's v0.4 acceptance shape: spawn the actual `kotoba` binary
//! and assert the second-brain compounds across runs. This test stops short
//! of full-conversation TUI exercise (which would require either a `--print`
//! / non-interactive dm path or a pty harness — neither exists yet) and
//! covers the strongest CI-tractable subset:
//!
//! 1. **Real binary launch.** `kotoba session --persona Yuki --yes
//!    --plan-only` is spawned via `std::process::Command` against a tempdir
//!    project root. This exercises the kotoba CLI parser, host-mode
//!    identity routing, the `dm_config_dir` lookup, and the planner code
//!    path through canonical's identity-aware `Config`.
//!
//! 2. **Empty wiki produces a "first session" brief.** Pre-condition: no
//!    vocabulary entries. Brief output reflects that.
//!
//! 3. **Second-brain compounding.** Pre-seed the wiki via canonical
//!    `Wiki::register_host_page` (the same path host capabilities use post-
//!    Run-32). Run the binary again. Brief now references the seeded vocab.
//!
//! When this passes, kotoba's `<project>/.dm/wiki/` second-brain promise is
//! genuinely lived through the binary boundary — not just unit-test mocks.
//! The full TUI conversation surface remains separate (Tier 8 #1 sub-bullet
//! "synthesized stdin"); that needs a kotoba-side `--print` flag or a pty
//! harness, both deferred.

use dark_matter::wiki::{Layer, PageType, Wiki, WikiPage};
use std::process::Command;
use std::sync::Mutex;
use tempfile::TempDir;

// `cargo test` builds the kotoba binary alongside the integration tests and
// exposes the path through this env var. Resolved at compile time.
const KOTOBA_BIN: &str = env!("CARGO_BIN_EXE_kotoba");

// Some host functions read identity / config that touch process-global
// state under HOME. Each test runs in its own subprocess, but we still
// serialize CWD / env mutation in the test harness for safety.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct ProjectFixture {
    _temp: TempDir,
    project: std::path::PathBuf,
    home: std::path::PathBuf,
}

impl ProjectFixture {
    fn new() -> Self {
        let temp = TempDir::new().expect("tempdir");
        let project = temp.path().join("project");
        let home = temp.path().join("home");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::create_dir_all(&home).expect("create home");
        std::fs::create_dir_all(project.join(".dm")).expect("create .dm");
        std::fs::write(
            project.join(".dm").join("identity.toml"),
            "mode = \"host\"\nhost_project = \"kotoba-e2e\"\n",
        )
        .expect("write identity.toml");
        // Initialize the wiki layout so register_host_page has a place to
        // write the index against.
        Wiki::open(&project).expect("open wiki");
        Self {
            _temp: temp,
            project,
            home,
        }
    }

    fn run_plan_only(&self) -> std::process::Output {
        Command::new(KOTOBA_BIN)
            .args(["session", "--persona", "Yuki", "--yes", "--plan-only"])
            .current_dir(&self.project)
            .env("HOME", &self.home)
            // Ensure the spawn doesn't try to reach a real Ollama for the
            // rule-based planner path. The opt-in LLM planner stays off
            // unless KOTOBA_PLANNER_USE_LLM is set, so this is just defense.
            .env_remove("KOTOBA_PLANNER_USE_LLM")
            .env_remove("KOTOBA_RECORDER_USE_LLM")
            .output()
            .expect("spawn kotoba binary")
    }

    fn seed_vocab(&self, kanji: &str, kana: &str, meaning: &str) {
        let wiki = Wiki::open(&self.project).expect("open wiki");
        let body = format!(
            "# {kanji}\n\n- **Kanji:** {kanji}\n- **Kana:** {kana}\n- **Meaning:** {meaning}\n- **Mastery:** Introduced\n"
        );
        let page = WikiPage {
            title: kanji.to_string(),
            page_type: PageType::Entity,
            layer: Layer::Host,
            sources: vec![],
            last_updated: "2026-04-29 00:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],

            extras: std::collections::BTreeMap::new(),
            body,
        };
        let one_liner = format!("{kanji} ({kana}) — {meaning}");
        wiki.register_host_page(&format!("entities/Vocabulary/{kanji}.md"), &page, &one_liner)
            .expect("register seeded vocab");
    }
}

#[test]
fn empty_wiki_session_brief_announces_first_session() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let fix = ProjectFixture::new();

    let output = fix.run_plan_only();
    assert!(
        output.status.success(),
        "kotoba session --plan-only failed: status={}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Sentinels stable in `host_caps::build_session_brief` — present and load-bearing
    // for the planner UX.
    assert!(
        stdout.contains("# Session brief"),
        "expected planner brief header in stdout; got:\n{stdout}"
    );
    assert!(
        stdout.contains("Wiki has no vocabulary yet"),
        "empty-wiki brief should announce first-session state; got:\n{stdout}"
    );
}

#[test]
fn second_session_brief_surfaces_prior_session_vocab() {
    // The v0.4 second-brain compounding assertion: vocab written to the
    // wiki by a prior session shows up in the next session's planner
    // brief. Without this, the second brain doesn't actually compound —
    // every session would start cold.
    let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let fix = ProjectFixture::new();

    // Seed three vocab pages as if a prior session had logged them.
    fix.seed_vocab("学校", "がっこう", "school");
    fix.seed_vocab("猫", "ねこ", "cat");
    fix.seed_vocab("食べる", "たべる", "to eat");

    let output = fix.run_plan_only();
    assert!(
        output.status.success(),
        "kotoba session --plan-only failed: status={}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("# Session brief"),
        "expected planner brief header in stdout; got:\n{stdout}"
    );

    // The directive's "second-session shows prior knowledge" requirement.
    // The brief's vocab section sorts by mastery and caps at 5; all three
    // seeded words are at the same Introduced mastery level, so all three
    // should surface.
    for kanji in ["学校", "猫", "食べる"] {
        assert!(
            stdout.contains(kanji),
            "expected seeded vocab '{kanji}' in second-session brief; got:\n{stdout}"
        );
    }

    // The empty-wiki sentinel must NOT appear once we have vocab — a
    // regression here would mean the planner is reading from the wrong
    // root or the index isn't being consulted across the binary boundary.
    assert!(
        !stdout.contains("Wiki has no vocabulary yet"),
        "seeded wiki should not produce empty-wiki brief; got:\n{stdout}"
    );
}
