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
fn daemon_dispatches_host_invoke_post_9259acb() {
    let bin = env!("CARGO_BIN_EXE_dm");
    let home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path());

    run_dm(bin, project.path(), home.path(), &["--daemon-start"]);
    let socket = project.path().join(".dm/daemon.sock");
    wait_for_connectable_socket(&socket);

    // Post-9259acb: invoke a host capability through the daemon. The
    // `host.invoke` method now dispatches against the in-process tool
    // registry, enforces the host_ prefix, and either returns
    // HostInvokeResult on success or an error event with the tool's
    // own message. With kotoba's KotobaCapabilities pre-installed,
    // host_record_session resolves; we send minimal-but-valid args
    // so the dispatch lands without exercising recorder logic.
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

    // Post-9259acb: the daemon must NOT respond with "unknown method"
    // for host.invoke. Either the call succeeds (dispatch + tool ran)
    // or it fails for tool-specific reasons (missing args, host tool
    // returned is_error). What we explicitly verify is that the
    // dispatch primitive is wired — host.invoke is a known method.
    assert!(
        !message.contains("unknown method"),
        "post-9259acb: daemon must NOT reject host.invoke as unknown \
         method. Got response: {}. The canonical fix wires the dispatch \
         primitive — if this assert fires, it has regressed.",
        response_line
    );

    // The response should be either a successful HostInvokeResult or
    // an error that names host.invoke / host_record_session in its
    // message. Both shapes confirm the daemon is now extension-aware.
    let recognized = matches!(event_type, "host_invoke_result")
        || (event_type == "error"
            && (message.contains("host.invoke")
                || message.contains("host_record_session")
                || message.contains("host tool")));
    assert!(
        recognized,
        "post-9259acb: expected HostInvokeResult or a host.invoke-shaped \
         error event. Got: {}",
        response_line
    );
}
