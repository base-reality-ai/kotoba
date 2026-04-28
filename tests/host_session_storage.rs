//! Host-mode session storage routing.
//!
//! These tests spawn the real `dm` binary so `Config::load()` resolves
//! identity from cwd and the conversation path persists through the same
//! boundary users exercise.

mod common;

use common::mock_ollama::MockOllama;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Stdio;
use tempfile::TempDir;
use tokio::process::Command;

fn write_host_identity(root: &Path, host_project: &str) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        format!("mode = \"host\"\nhost_project = \"{}\"\n", host_project),
    )
    .expect("write identity.toml");
}

fn dir_entries(path: &Path) -> BTreeSet<String> {
    match std::fs::read_dir(path) {
        Ok(entries) => entries
            .map(|entry| {
                entry
                    .expect("read dir entry")
                    .file_name()
                    .to_string_lossy()
                    .to_string()
            })
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeSet::new(),
        Err(e) => panic!("read_dir {} failed: {}", path.display(), e),
    }
}

#[tokio::test]
async fn host_mode_sessions_land_in_project_dir() {
    let project = TempDir::new().expect("project tempdir");
    let home = TempDir::new().expect("home tempdir");
    write_host_identity(project.path(), "session-routing-host");

    let global_sessions = home.path().join(".dm/sessions");
    std::fs::create_dir_all(&global_sessions).expect("create global sessions dir");
    std::fs::write(global_sessions.join("sentinel.txt"), "do not touch").expect("write sentinel");
    let global_before = dir_entries(&global_sessions);

    let mock = MockOllama::new()
        .emit_token("hello from host")
        .emit_done(4, 3)
        .spawn()
        .await;

    let output = Command::new(env!("CARGO_BIN_EXE_dm"))
        .current_dir(project.path())
        .args([
            "--no-tools",
            "--no-memory",
            "--no-workspace-context",
            "--no-claude-md",
            "--quiet",
            "-p",
            "hi",
        ])
        .env("HOME", home.path())
        .env("OLLAMA_HOST", mock.host())
        .env("DM_WIKI_AUTO_INGEST", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn dm");

    assert!(
        output.status.success(),
        "dm -p failed: status={:?}\nstdout={}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let project_sessions = project.path().join(".dm/sessions");
    let session_files: Vec<_> = std::fs::read_dir(&project_sessions)
        .expect("project sessions dir exists")
        .map(|entry| entry.expect("session entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    assert_eq!(
        session_files.len(),
        1,
        "expected exactly one project-local session in {}",
        project_sessions.display()
    );

    let session_text =
        std::fs::read_to_string(&session_files[0]).expect("read project session json");
    let session_json: serde_json::Value =
        serde_json::from_str(&session_text).expect("parse session json");
    assert_eq!(
        session_json["host_project"].as_str(),
        Some("session-routing-host")
    );
    assert!(
        session_text.contains("hello from host"),
        "session should contain mock assistant response: {}",
        session_text
    );

    assert_eq!(
        dir_entries(&global_sessions),
        global_before,
        "host-mode dm must not write sessions under isolated HOME/.dm/sessions"
    );
}
