//! Tier-3 end-to-end integration test (kotoba v0.3).
//!
//! Exercises the documented daemon-mode workaround end-to-end through the
//! real `kotoba` binary: the worker subcommand
//! `kotoba record-session --session-id <id> --persona <name>` reads a
//! pre-saved dm session transcript, runs the recorder against the wiki,
//! and writes a status file to `<project>/.dm/recorder-status.json`.
//!
//! This is the test the prior handoff asked for. It builds the kotoba
//! binary (cargo handles via `CARGO_BIN_EXE_kotoba`), spawns it as a
//! child process so the integration covers the real CLI surface — not
//! just the helper functions exercised by the bin's unit tests — and
//! confirms three things:
//!
//! 1. The recorder worker exits successfully when given a valid
//!    project + session id.
//! 2. `recorder-status.json` lands in the project-local `.dm/` and
//!    transitions to `state: "done"`.
//! 3. The wiki actually gains the vocabulary page and the persona log
//!    entry (proving the recorder didn't just claim success — it ran).
//!
//! The test pins the *workaround* shape, not the canonical gap. The
//! gap itself (`paradigm-gap-daemon-protocol-host-extension.md`) stays
//! pinned by `tests/host_daemon_protocol_closed.rs`.

#![cfg(unix)]

use std::path::Path;
use std::process::{Command, Stdio};
use tempfile::TempDir;

fn write_host_identity(root: &Path) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"kotoba\"\n",
    )
    .expect("write identity");
}

fn seed_yuki_persona(root: &Path) {
    let persona_dir = root.join(".dm/wiki/entities/Persona");
    std::fs::create_dir_all(&persona_dir).expect("persona dir");
    std::fs::write(
        persona_dir.join("Yuki.md"),
        "---\n\
title: Yuki\n\
type: entity\n\
entity_kind: persona\n\
layer: host\n\
sessions_count: 0\n\
---\n\n\
# Yuki\n\n\
## Signature topics\n\n\
- Self-introduction\n\n\
## Sessions log\n\n",
    )
    .expect("write Yuki seed");
}

fn save_dm_session(config_dir: &Path, project_root: &Path, id: &str) {
    use chrono::Utc;
    let now = Utc::now();
    let mut session =
        dark_matter::session::Session::new(project_root.display().to_string(), "mock-model".into());
    session.id = id.into();
    session.created_at = now;
    session.updated_at = now;
    session.push_message(serde_json::json!({
        "role": "system",
        "content": "system prompt — recorder must skip this",
    }));
    session.push_message(serde_json::json!({
        "role": "assistant",
        "content": "New word: 学校 (がっこう) means school.",
    }));
    session.push_message(serde_json::json!({
        "role": "user",
        "content": "what is は vs が?",
    }));
    dark_matter::session::storage::save(config_dir, &session).expect("save session");
}

#[test]
fn record_session_worker_runs_end_to_end_through_real_binary() {
    let bin = env!("CARGO_BIN_EXE_kotoba");
    let project = TempDir::new().expect("project tempdir");
    let home = TempDir::new().expect("home tempdir");

    // Host-mode identity → Run-31 routes Config to <project>/.dm
    write_host_identity(project.path());
    seed_yuki_persona(project.path());
    let config_dir = project.path().join(".dm");
    let session_id = "kotoba-recorder-e2e";
    save_dm_session(&config_dir, project.path(), session_id);

    // Run the worker subcommand directly. The recorder is synchronous
    // here; the fork-and-detach happens only on `session --daemon`,
    // which we skip because that path also needs a TUI/dm subprocess.
    // The worker subcommand is the part of the workaround that
    // actually does the wiki work, so it's the test target.
    let output = Command::new(bin)
        .current_dir(project.path())
        .env("HOME", home.path())
        .args([
            "record-session",
            "--session-id",
            session_id,
            "--persona",
            "Yuki",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn kotoba record-session");

    assert!(
        output.status.success(),
        "kotoba record-session failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    // 1. Status file exists and reports done.
    let status_path = config_dir.join("recorder-status.json");
    let status_raw = std::fs::read_to_string(&status_path).expect("read recorder-status.json");
    let status: serde_json::Value =
        serde_json::from_str(&status_raw).expect("parse recorder-status.json");
    assert_eq!(
        status["state"].as_str(),
        Some("done"),
        "recorder-status.json should be done; got {}",
        status_raw
    );
    assert_eq!(status["session_id"].as_str(), Some(session_id));
    assert_eq!(status["persona"].as_str(), Some("Yuki"));

    // 2. Wiki vocab page landed under the host-layer entities dir.
    let vocab_page = project.path().join(".dm/wiki/entities/Vocabulary/学校.md");
    let vocab_body = std::fs::read_to_string(&vocab_page).expect("read 学校.md");
    assert!(
        vocab_body.contains("- **Kana:** がっこう"),
        "{}",
        vocab_body
    );
    assert!(
        vocab_body.contains("- **Meaning:** school"),
        "{}",
        vocab_body
    );

    // 3. Persona page bumped to session 1 with the struggle
    //    captured in the session log.
    let persona_body =
        std::fs::read_to_string(project.path().join(".dm/wiki/entities/Persona/Yuki.md"))
            .expect("read Yuki.md");
    assert!(
        persona_body.contains("- **Sessions:** 1"),
        "{}",
        persona_body
    );
    assert!(
        persona_body.contains("struggles flagged: は vs が"),
        "{}",
        persona_body
    );

    // 4. Today's struggles synthesis page also picked up the same item
    //    (this ties the recorder's struggle-detection to its synthesis
    //    output — a sanity check that the worker ran the full pipeline,
    //    not just the persona append).
    let synthesis_dir = project.path().join(".dm/wiki/synthesis");
    let mut found_struggle = false;
    for entry in std::fs::read_dir(&synthesis_dir).expect("read synthesis dir") {
        let entry = entry.unwrap();
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("struggles-")
        {
            continue;
        }
        let body = std::fs::read_to_string(entry.path()).unwrap();
        if body.contains("**は vs が**") {
            found_struggle = true;
            break;
        }
    }
    assert!(
        found_struggle,
        "expected today's struggles file to contain 'は vs が'"
    );
}
