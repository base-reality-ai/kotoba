//! Consolidated host-mode artifact routing smoke.
//!
//! Exercises the public binary across sessions, daemon, daemon log, and web API
//! startup with one host fixture and one isolated HOME.

#![cfg(unix)]

mod common;

use common::mock_ollama::MockOllama;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
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

fn snapshot_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries {
            let entry = entry.expect("read dir entry");
            let path = entry.path();
            let rel = path
                .strip_prefix(root)
                .expect("strip prefix")
                .to_string_lossy()
                .to_string();
            if path.is_dir() {
                out.insert(format!("{rel}/"), Vec::new());
                walk(root, &path, out);
            } else {
                out.insert(rel, std::fs::read(&path).expect("read snapshot file"));
            }
        }
    }

    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

async fn run_dm(
    bin: &str,
    cwd: &Path,
    home: &Path,
    ollama_host: &str,
    args: &[&str],
) -> std::process::Output {
    Command::new(bin)
        .current_dir(cwd)
        .args(args)
        .env("HOME", home)
        .env("OLLAMA_HOST", ollama_host)
        .env("DM_DAEMON_WATCHDOG", "0")
        .env("DM_WIKI_AUTO_INGEST", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("spawn dm")
}

fn assert_success(output: &std::process::Output, context: &str) {
    assert!(
        output.status.success(),
        "{} failed: status={:?}\nstdout={}\nstderr={}",
        context,
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

async fn wait_for_path(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if path.exists() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_connectable_socket(path: &Path) {
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
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

struct DaemonGuard {
    bin: &'static str,
    cwd: PathBuf,
    home: PathBuf,
}

impl DaemonGuard {
    async fn start(bin: &'static str, cwd: &Path, home: &Path) -> Self {
        let status = Command::new(bin)
            .current_dir(cwd)
            .arg("--daemon-start")
            .env("HOME", home)
            .env("DM_DAEMON_WATCHDOG", "0")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .expect("spawn dm --daemon-start");
        assert!(status.success(), "--daemon-start failed: {status:?}");
        Self {
            bin,
            cwd: cwd.to_path_buf(),
            home: home.to_path_buf(),
        }
    }

    fn stop(&self) {
        let _ = std::process::Command::new(self.bin)
            .current_dir(&self.cwd)
            .arg("--daemon-stop")
            .env("HOME", &self.home)
            .env("DM_DAEMON_WATCHDOG", "0")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

#[tokio::test]
async fn host_mode_artifacts_land_under_project_dm() {
    let bin = env!("CARGO_BIN_EXE_dm");
    let project = TempDir::new().expect("project tempdir");
    let home = TempDir::new().expect("home tempdir");
    write_host_identity(project.path(), "host-e2e");

    let global_dm = home.path().join(".dm");
    std::fs::create_dir_all(&global_dm).expect("create global .dm");
    std::fs::write(global_dm.join("sentinel.txt"), "unchanged").expect("write sentinel");
    let global_before = snapshot_tree(&global_dm);

    let mock = MockOllama::new()
        .emit_token("host e2e response")
        .emit_done(4, 3)
        .spawn()
        .await;

    let daemon = DaemonGuard::start(bin, project.path(), home.path()).await;
    let project_dm = project.path().join(".dm");
    let daemon_sock = project_dm.join("daemon.sock");
    let daemon_pid = project_dm.join("daemon.pid");
    let daemon_log = project_dm.join("daemon.log");
    wait_for_connectable_socket(&daemon_sock).await;
    wait_for_path(&daemon_pid).await;
    wait_for_path(&daemon_log).await;

    let prompt_output = run_dm(
        bin,
        project.path(),
        home.path(),
        &mock.host(),
        &[
            "--no-tools",
            "--no-memory",
            "--no-workspace-context",
            "--no-claude-md",
            "--quiet",
            "-p",
            "hi",
        ],
    )
    .await;
    assert_success(&prompt_output, "dm -p");

    let mut web_child = Command::new(bin)
        .current_dir(project.path())
        .args(["--web", "--web-port", "0"])
        .env("HOME", home.path())
        .env("OLLAMA_HOST", mock.host())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn dm --web");
    wait_for_path(&project_dm.join("web.token")).await;
    wait_for_path(&project_dm.join("web.last_port")).await;
    let _ = web_child.kill().await;
    let _ = web_child.wait().await;

    assert!(
        project_dm.join("sessions").exists(),
        "session dir should be project-local"
    );
    let session_count = std::fs::read_dir(project_dm.join("sessions"))
        .expect("read project sessions")
        .filter(|entry| {
            entry
                .as_ref()
                .ok()
                .and_then(|e| e.path().extension().map(|ext| ext == "json"))
                .unwrap_or(false)
        })
        .count();
    assert_eq!(session_count, 1, "expected one project-local session");
    assert!(
        daemon_sock.exists(),
        "daemon socket should be project-local"
    );
    assert!(daemon_pid.exists(), "daemon pid should be project-local");
    assert!(daemon_log.exists(), "daemon log should be project-local");
    assert!(project_dm.join("web.token").exists());
    assert!(
        project_dm.join("logs").exists(),
        "runtime logs should be project-local"
    );
    assert_eq!(
        std::fs::read_to_string(project_dm.join("web.last_port"))
            .expect("read web.last_port")
            .trim(),
        "0"
    );

    assert!(!global_dm.join("sessions").exists());
    assert!(!global_dm.join("daemon.sock").exists());
    assert!(!global_dm.join("daemon.pid").exists());
    assert!(!global_dm.join("daemon.log").exists());
    assert!(!global_dm.join("web.token").exists());
    assert!(!global_dm.join("web.last_port").exists());
    assert!(!global_dm.join("logs").exists());
    assert_eq!(
        snapshot_tree(&global_dm),
        global_before,
        "host-mode artifacts must not mutate isolated HOME/.dm"
    );

    daemon.stop();
}
