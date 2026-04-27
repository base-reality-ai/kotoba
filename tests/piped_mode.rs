//! Phase 2.6 — piped mode robustness integration tests.
//!
//! These spawn the `dm` binary end-to-end to verify the exit-code contract
//! for the three user-visible headless entry paths: empty stdin, unreachable
//! Ollama, and `--help` documenting all four exit codes. Kept in
//! `tests/` (separate binary) so they exercise the real process boundary.

use std::process::{Command, Stdio};

// TODO: context-overflow auto-continue requires a mock Ollama server;
// tracked for the Phase 2.5 / server-harness cycle.

#[test]
fn empty_stdin_exits_config_error() {
    // Spawn with stdin piped, then drop the handle so the child reads EOF
    // immediately — reproduces `cmd | dm` where upstream emitted nothing.
    let mut child = Command::new(env!("CARGO_BIN_EXE_dm"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dm");
    // Explicitly drop stdin to close the pipe immediately.
    drop(child.stdin.take());

    let out = child.wait_with_output().expect("wait");
    assert_eq!(
        out.status.code(),
        Some(2),
        "empty piped stdin must exit 2 (ConfigError); stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no input received"),
        "stderr should mention 'no input received': {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.is_empty(),
        "stderr must not leak to stdout; stdout={:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn bogus_ollama_host_exits_model_unreachable() {
    // Port 1 is reserved and effectively never serves — the connect fails
    // fast with a reqwest connect error, which `classify()` maps to exit 3.
    let out = Command::new(env!("CARGO_BIN_EXE_dm"))
        .args(["-p", "hi"])
        .env("OLLAMA_HOST", "127.0.0.1:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn dm");
    assert_eq!(
        out.status.code(),
        Some(3),
        "unreachable Ollama must exit 3 (ModelUnreachable); stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn scheme_prefixed_ollama_host_does_not_double_scheme() {
    // Regression: `OLLAMA_HOST=http://...` must not produce `http://http://`
    // in the resulting URL. Cycle 18 normalized at Config::load; clap's
    // #[arg(env = "OLLAMA_HOST")] reads the raw value independently and the
    // CLI-override write sites (main.rs) were bypassing the normalizer.
    let out = Command::new(env!("CARGO_BIN_EXE_dm"))
        .args(["-p", "hi"])
        .env("OLLAMA_HOST", "http://127.0.0.1:1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn dm");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("http://http://"),
        "double scheme in error: {}",
        err
    );
    assert_eq!(out.status.code(), Some(3));
}

#[test]
fn help_after_help_lists_all_four_exit_codes() {
    let out = Command::new(env!("CARGO_BIN_EXE_dm"))
        .arg("--help")
        .output()
        .expect("spawn dm");
    assert_eq!(out.status.code(), Some(0), "--help must exit 0");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("EXIT CODES"),
        "--help footer must include 'EXIT CODES' section"
    );
    for code in ["0", "1", "2", "3"] {
        assert!(
            s.contains(&format!("  {}  ", code)),
            "--help should list exit code {} in footer",
            code
        );
    }
}
