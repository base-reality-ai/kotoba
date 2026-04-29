//! Tier-3 daemon-protocol regression test (canonical-side).
//!
//! Mirrors `kotoba/tests/host_daemon_protocol_closed.rs` but with the
//! kotoba "unknown method" assert flipped: post-fix, the daemon MUST
//! recognize `host.invoke` and route it through the host-tool dispatch
//! primitive. Canonical `dm` itself installs no host capabilities, so
//! the dispatch returns an `Error` event with the new "no host
//! capabilities installed" message — distinct from the pre-fix
//! "unknown method" rejection.
//!
//! Single-binary because it forks `dm --daemon-start` and listens on
//! the project-local socket.

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
fn daemon_recognizes_host_invoke_method() {
    let bin = env!("CARGO_BIN_EXE_dm");
    let home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path());

    run_dm(bin, project.path(), home.path(), &["--daemon-start"]);
    let socket = project.path().join(".dm/daemon.sock");
    wait_for_connectable_socket(&socket);

    let stream = UnixStream::connect(&socket).expect("connect daemon");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut writer = stream;
    let request =
        r#"{"id":1,"method":"host.invoke","params":{"name":"host_record_session","args":{}}}"#;
    writeln!(writer, "{}", request).expect("send request");
    writer.flush().expect("flush request");

    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .expect("read response line");

    run_dm(bin, project.path(), home.path(), &["--daemon-stop"]);

    let parsed: serde_json::Value =
        serde_json::from_str(response_line.trim()).expect("parse response JSON");
    let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let message = parsed.get("message").and_then(|m| m.as_str()).unwrap_or("");

    // Canonical dm installs no host caps, so dispatch lands on
    // NoHostInstalled — but the *protocol* now accepts the method
    // (no more "unknown method" rejection).
    assert_eq!(
        event_type, "error",
        "canonical dm with no host caps should still surface an error: {response_line}"
    );
    assert!(
        !message.contains("unknown method"),
        "post-fix: daemon must NOT reject host.invoke as unknown method. \
         got: {response_line}"
    );
    assert!(
        message.contains("host.invoke"),
        "error must echo the method context: {response_line}"
    );
    // Canonical dm has no host caps installed → "no host capabilities
    // installed" message. Kotoba binary boots with KotobaCapabilities
    // pre-installed → the dispatch finds host_record_session and only
    // fails because args are empty. Either error path is correct: both
    // confirm the primitive is wired and the daemon stopped rejecting
    // host.invoke as an unknown method.
    assert!(
        message.contains("no host capabilities installed")
            || message.contains("host tool failed")
            || message.contains("missing required"),
        "expected either NoHostInstalled (canonical) or a tool-side \
         dispatch error (host project with caps installed). \
         got: {response_line}"
    );
}
