//! Daemon socket routing across host identities.
//!
//! Spawns real daemon worker processes through the public CLI so the test
//! covers the startup path, not just the socket path helper.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn write_host_identity(root: &Path, host_project: &str) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        format!("mode = \"host\"\nhost_project = \"{}\"\n", host_project),
    )
    .expect("write identity.toml");
}

fn run_dm(bin: &str, cwd: &Path, home: &Path, args: &[&str]) -> Output {
    Command::new(bin)
        .current_dir(cwd)
        .args(args)
        .env("HOME", home)
        .env("DM_DAEMON_WATCHDOG", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn dm")
}

fn run_dm_status(bin: &str, cwd: &Path, home: &Path, args: &[&str]) -> ExitStatus {
    Command::new(bin)
        .current_dir(cwd)
        .args(args)
        .env("HOME", home)
        .env("DM_DAEMON_WATCHDOG", "0")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn dm")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{} failed: status={:?}\nstdout={}\nstderr={}",
        context,
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn wait_for_connectable_socket(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for connectable socket at {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(25));
    }
}

struct DaemonGuard {
    bin: &'static str,
    cwd: PathBuf,
    home: PathBuf,
}

impl DaemonGuard {
    fn start(bin: &'static str, cwd: &Path, home: &Path) -> Self {
        let status = run_dm_status(bin, cwd, home, &["--daemon-start"]);
        assert!(status.success(), "--daemon-start failed: status={status:?}");
        Self {
            bin,
            cwd: cwd.to_path_buf(),
            home: home.to_path_buf(),
        }
    }

    fn stop(&self) {
        let _ = run_dm(self.bin, &self.cwd, &self.home, &["--daemon-stop"]);
    }
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

#[test]
fn host_daemons_start_with_project_local_sockets() {
    let bin = env!("CARGO_BIN_EXE_dm");
    let home = TempDir::new().expect("home tempdir");
    let alpha = TempDir::new().expect("alpha tempdir");
    let beta = TempDir::new().expect("beta tempdir");
    write_host_identity(alpha.path(), "daemon-alpha");
    write_host_identity(beta.path(), "daemon-beta");

    let alpha_socket = alpha.path().join(".dm/daemon.sock");
    let beta_socket = beta.path().join(".dm/daemon.sock");
    let global_socket = home.path().join(".dm/daemon.sock");

    let alpha_daemon = DaemonGuard::start(bin, alpha.path(), home.path());
    wait_for_connectable_socket(&alpha_socket);

    let beta_daemon = DaemonGuard::start(bin, beta.path(), home.path());
    wait_for_connectable_socket(&beta_socket);

    assert_ne!(alpha_socket, beta_socket);
    assert!(alpha_socket.exists(), "alpha socket missing");
    assert!(beta_socket.exists(), "beta socket missing");
    assert!(
        !global_socket.exists(),
        "host daemon startup must not create global kernel socket at {}",
        global_socket.display()
    );

    let alpha_status = run_dm(bin, alpha.path(), home.path(), &["--daemon-status"]);
    assert_success(&alpha_status, "alpha --daemon-status");
    let alpha_stdout = String::from_utf8_lossy(&alpha_status.stdout);
    assert!(
        alpha_stdout.contains(&alpha_socket.to_string_lossy().to_string()),
        "alpha status should report project socket; stdout={}",
        alpha_stdout
    );

    let beta_status = run_dm(bin, beta.path(), home.path(), &["--daemon-status"]);
    assert_success(&beta_status, "beta --daemon-status");
    let beta_stdout = String::from_utf8_lossy(&beta_status.stdout);
    assert!(
        beta_stdout.contains(&beta_socket.to_string_lossy().to_string()),
        "beta status should report project socket; stdout={}",
        beta_stdout
    );

    alpha_daemon.stop();
    beta_daemon.stop();
}
