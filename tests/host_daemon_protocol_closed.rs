//! Tier-3 paradigm-gap regression test (kotoba v0.3).
//!
//! The dm daemon's request dispatcher is a closed match on a fixed set of
//! kernel-defined method strings (`session.*`, `turn.*`, `chain.*`,
//! `daemon.*`, `ping`). There is no `host.*` family and no extension hook
//! for host projects to register their own method handlers — every
//! unknown method is rejected with `DaemonEvent::Error { message: "unknown
//! method: ..." }`.
//!
//! Concrete consequence for kotoba's Tier 3 ambition: even though the
//! daemon process correctly inherits kotoba's host capabilities (they are
//! installed at startup in `src/main.rs` via `install_host_capabilities`),
//! there is no daemon-protocol path to *invoke* a host capability as a
//! deterministic function call. The only way to drive a host tool through
//! the daemon today is `session.create` + `turn.send` with a model that
//! decides to call the tool — which requires a real LLM and yields
//! nondeterministic timing. There is no `host.invoke` shortcut.
//!
//! This test pins the "closed registry" half of the gap: a `host.invoke`
//! request is rejected with the unknown-method error. When the canonical
//! fix lands (some form of host-tool invocation primitive on the
//! daemon), the assertion flips from "rejected" to "dispatched", and the
//! gap doc moves to RESOLVED.
//!
//! See `.dm/wiki/concepts/paradigm-gap-daemon-protocol-host-extension.md`
//! for the full diagnosis + recommended fix.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
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

fn run_dm(bin: &str, cwd: &Path, home: &Path, args: &[&str]) {
    let status = Command::new(bin)
        .current_dir(cwd)
        .args(args)
        .env("HOME", home)
        .env("DM_DAEMON_WATCHDOG", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn dm");
    assert!(status.success(), "dm {:?} failed: {:?}", args, status);
}

fn wait_for_connectable_socket(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if UnixStream::connect(path).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting for socket {}", path.display());
}

#[test]
fn daemon_rejects_host_tool_invocation_method() {
    let bin = env!("CARGO_BIN_EXE_dm");
    let home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path());

    run_dm(bin, project.path(), home.path(), &["--daemon-start"]);
    let socket = project.path().join(".dm/daemon.sock");
    wait_for_connectable_socket(&socket);

    // Try to invoke a host capability through the daemon. There is no
    // `host.invoke` (or `host.record_session`, or `tool.call`, or any
    // other host extension hook) in the daemon's match arm — see
    // `src/daemon/server.rs::handle_connection`, where the closed string
    // match falls through to "unknown method: …" on anything not in the
    // kernel's whitelist.
    let stream = UnixStream::connect(&socket).expect("connect daemon");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut writer = stream;
    let request = r#"{"id":1,"method":"host.invoke","params":{"name":"host_record_session","args":{"transcript":"placeholder","persona":"Yuki"}}}"#;
    writeln!(writer, "{}", request).expect("send request");
    writer.flush().expect("flush request");

    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .expect("read response line");

    // Always stop the daemon before asserting so a failed test still
    // releases the socket cleanly.
    run_dm(bin, project.path(), home.path(), &["--daemon-stop"]);

    let parsed: serde_json::Value =
        serde_json::from_str(response_line.trim()).expect("parse response JSON");
    let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let message = parsed.get("message").and_then(|m| m.as_str()).unwrap_or("");
    assert_eq!(
        event_type, "error",
        "expected error event, got {}: {}",
        event_type, response_line
    );

    assert!(
        message.contains("unknown method"),
        "GAP: expected daemon to reject host.invoke as unknown method. \
         Got response: {}. If this assert fails because the daemon \
         dispatched the call, the canonical fix has landed — flip this \
         test to assert successful dispatch and update the gap doc.",
        response_line
    );
    assert!(
        message.contains("host.invoke"),
        "GAP: error message should echo the rejected method name. \
         Got response: {}.",
        response_line
    );
}
