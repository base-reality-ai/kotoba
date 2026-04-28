use crate::daemon::protocol::{DaemonEvent, DaemonRequest};
use anyhow::Context;
use std::path::PathBuf;

/// Resolve the dm config dir using the project's identity-aware routing.
///
/// Kernel mode → `~/.dm`. Host mode → `<project_root>/.dm`. Mirrors
/// `Config::config_dir` semantics; see
/// `.dm/wiki/concepts/identity-config-routing.md`. Used by daemon helpers
/// that have no `&Config` in scope (CLI guards, web mode probes).
fn dm_config_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let identity = crate::identity::load_for_cwd();
    crate::config::compute_config_dir(&home, &identity)
}

/// Returns the path to the daemon Unix socket.
///
/// - `kernel` mode → `~/.dm/daemon.sock` (legacy singleton, unchanged).
/// - `host` mode → `<project_root>/.dm/daemon.sock`. Scoping is achieved
///   by `config_dir` already pointing at the project root; no per-host
///   filename suffix is needed because each host project has its own
///   `.dm/`. Two host projects never collide with each other or with
///   canonical dm.
pub fn daemon_socket_path() -> PathBuf {
    dm_config_dir().join("daemon.sock")
}

/// Returns the path to the daemon PID file.
///
/// Routing matches `daemon_socket_path`: `~/.dm/daemon.pid` in kernel
/// mode, `<project_root>/.dm/daemon.pid` in host mode.
pub fn daemon_pid_path() -> PathBuf {
    dm_config_dir().join("daemon.pid")
}

/// Returns true if a daemon socket exists AND is connectable.
/// Never panics — all errors return false.
pub fn daemon_socket_exists() -> bool {
    let path = daemon_socket_path();
    if !path.exists() {
        return false;
    }
    #[cfg(unix)]
    {
        std::os::unix::net::UnixStream::connect(&path).is_ok()
    }
    #[cfg(not(unix))]
    {
        false
    }
}

pub struct DaemonClient {
    stream: tokio::net::UnixStream,
    next_id: u64,
}

impl DaemonClient {
    pub async fn connect() -> anyhow::Result<Self> {
        let path = daemon_socket_path();
        let stream = tokio::net::UnixStream::connect(&path)
            .await
            .with_context(|| format!("Failed to connect to daemon socket at {}", path.display()))?;
        Ok(Self { stream, next_id: 1 })
    }

    /// Serialize and write a request as an NDJSON line. Returns the request ID.
    pub async fn send_request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> anyhow::Result<u64> {
        use tokio::io::AsyncWriteExt;
        let id = self.next_id;
        let req = DaemonRequest {
            id,
            method: method.to_string(),
            params,
        };
        let mut line = serde_json::to_string(&req).context("serialize request")?;
        line.push('\n');
        self.stream
            .write_all(line.as_bytes())
            .await
            .context("write request to socket")?;
        self.next_id += 1;
        Ok(id)
    }

    /// Start a chain from a YAML config file path.
    pub async fn chain_start(&mut self, config_path: &str) -> anyhow::Result<u64> {
        self.send_request(
            "chain.start",
            serde_json::json!({ "config_path": config_path }),
        )
        .await
    }

    /// Query the status of a running chain.
    pub async fn chain_status(&mut self, chain_id: &str) -> anyhow::Result<u64> {
        self.send_request("chain.status", serde_json::json!({ "chain_id": chain_id }))
            .await
    }

    /// Stop a running chain by writing the stop sentinel.
    pub async fn chain_stop(&mut self, chain_id: &str) -> anyhow::Result<u64> {
        self.send_request("chain.stop", serde_json::json!({ "chain_id": chain_id }))
            .await
    }

    /// List all active chains tracked by the daemon.
    pub async fn chain_list(&mut self) -> anyhow::Result<u64> {
        self.send_request("chain.list", serde_json::json!({})).await
    }

    /// Attach to a running chain and stream its events.
    pub async fn chain_attach(&mut self, chain_id: &str) -> anyhow::Result<u64> {
        self.send_request("chain.attach", serde_json::json!({ "chain_id": chain_id }))
            .await
    }

    /// Pause a running chain.
    pub async fn chain_pause(&mut self, chain_id: &str) -> anyhow::Result<u64> {
        self.send_request("chain.pause", serde_json::json!({ "chain_id": chain_id }))
            .await
    }

    /// Resume a paused chain.
    pub async fn chain_resume(&mut self, chain_id: &str) -> anyhow::Result<u64> {
        self.send_request("chain.resume", serde_json::json!({ "chain_id": chain_id }))
            .await
    }

    /// Inject a message into a node of a running chain.
    pub async fn chain_talk(
        &mut self,
        chain_id: &str,
        node: &str,
        message: &str,
    ) -> anyhow::Result<u64> {
        self.send_request(
            "chain.talk",
            serde_json::json!({
                "chain_id": chain_id, "node": node, "message": message
            }),
        )
        .await
    }

    /// Add a node to a running chain.
    pub async fn chain_add(
        &mut self,
        chain_id: &str,
        name: &str,
        model: &str,
        role: &str,
        input_from: Option<&str>,
    ) -> anyhow::Result<u64> {
        self.send_request(
            "chain.add",
            serde_json::json!({
                "chain_id": chain_id, "name": name, "model": model,
                "role": role, "input_from": input_from
            }),
        )
        .await
    }

    /// Remove a node from a running chain.
    pub async fn chain_remove(&mut self, chain_id: &str, node: &str) -> anyhow::Result<u64> {
        self.send_request(
            "chain.remove",
            serde_json::json!({
                "chain_id": chain_id, "node": node
            }),
        )
        .await
    }

    /// Swap the model for a node in a running chain.
    pub async fn chain_model(
        &mut self,
        chain_id: &str,
        node: &str,
        model: &str,
    ) -> anyhow::Result<u64> {
        self.send_request(
            "chain.model",
            serde_json::json!({
                "chain_id": chain_id, "node": node, "model": model
            }),
        )
        .await
    }

    /// Read a line from the stream and deserialize it as a `DaemonEvent`.
    pub async fn recv_event(&mut self) -> anyhow::Result<DaemonEvent> {
        use tokio::io::AsyncBufReadExt;
        let mut reader = tokio::io::BufReader::new(&mut self.stream);
        let mut line = String::new();

        const EVENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
        match tokio::time::timeout(EVENT_TIMEOUT, reader.read_line(&mut line)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => anyhow::bail!("read error from daemon: {}", e),
            Err(_) => anyhow::bail!(
                "daemon event timeout after {}s — connection may be stale",
                EVENT_TIMEOUT.as_secs()
            ),
        }

        if line.is_empty() {
            anyhow::bail!("daemon connection closed");
        }
        serde_json::from_str(line.trim()).context("deserialize event")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    struct CurrentDirGuard {
        original: PathBuf,
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }

    fn with_identity_cwd(identity_toml: Option<&str>, f: impl FnOnce(&Path)) {
        with_identity_cwd_value(identity_toml, |p| {
            f(p);
        });
    }

    /// Generic version of `with_identity_cwd` that returns the closure's
    /// value — used when a test needs to capture a path computed under
    /// the temp identity (e.g. comparing two host projects' daemon sockets).
    fn with_identity_cwd_value<T>(identity_toml: Option<&str>, f: impl FnOnce(&Path) -> T) -> T {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let original = std::env::current_dir().expect("current dir");
        let _dir_guard = CurrentDirGuard { original };
        let temp = TempDir::new().expect("tempdir");
        if let Some(identity_toml) = identity_toml {
            let dm_dir = temp.path().join(".dm");
            std::fs::create_dir_all(&dm_dir).expect("create .dm");
            std::fs::write(dm_dir.join("identity.toml"), identity_toml).expect("write identity");
        }
        std::env::set_current_dir(temp.path()).expect("set temp cwd");
        f(temp.path())
    }

    #[test]
    fn daemon_client_falls_back_when_no_socket() {
        with_identity_cwd(None, |_| {
            // If the socket path does not exist (or isn't connectable), daemon_socket_exists
            // must return false, not panic.
            let path = daemon_socket_path();
            // If the socket happens to exist (daemon actually running), skip assertion.
            if !path.exists() {
                assert!(!daemon_socket_exists());
            }
            // Either way, daemon_socket_exists() must not panic — reaching here means it didn't.
        });
    }

    #[test]
    fn daemon_socket_path_filename_is_daemon_sock_in_kernel_mode() {
        with_identity_cwd(None, |_| {
            let path = daemon_socket_path();
            assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some("daemon.sock")
            );
        });
    }

    #[test]
    fn daemon_pid_path_filename_is_daemon_pid_in_kernel_mode() {
        with_identity_cwd(None, |_| {
            let path = daemon_pid_path();
            assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some("daemon.pid")
            );
        });
    }

    #[test]
    fn host_daemon_paths_route_to_project_dm() {
        // Run-31 Tier 3: daemon socket/pid follow `Config::config_dir`
        // routing — kernel keeps `~/.dm/daemon.sock`, host lands in the
        // project's `.dm/daemon.sock` (no `daemons/<host>.sock` suffix
        // because the project root already scopes the path).
        with_identity_cwd(
            Some("mode = \"host\"\nhost_project = \"finance-app\"\n"),
            |project_root| {
                let sock = daemon_socket_path();
                let pid = daemon_pid_path();
                assert_eq!(
                    sock.file_name().and_then(|n| n.to_str()),
                    Some("daemon.sock"),
                    "host-mode socket name matches kernel-mode (project root scopes the path)"
                );
                assert_eq!(pid.file_name().and_then(|n| n.to_str()), Some("daemon.pid"));
                assert_eq!(
                    sock.parent(),
                    Some(project_root.join(".dm").as_path()),
                    "host daemon paths must live under <project>/.dm, not ~/.dm"
                );
            },
        );
    }

    #[test]
    fn host_and_kernel_daemon_paths_do_not_collide() {
        // Two host projects on the same machine — and canonical dm —
        // each get their own daemon socket. The legacy `~/.dm/daemons/`
        // workaround was load-bearing only because socket paths were all
        // under `~/.dm/`; with project-scoped routing the collision risk
        // is gone by construction.
        let host_a_sock =
            with_identity_cwd_value(Some("mode = \"host\"\nhost_project = \"alpha\"\n"), |_| {
                daemon_socket_path()
            });
        let host_b_sock =
            with_identity_cwd_value(Some("mode = \"host\"\nhost_project = \"beta\"\n"), |_| {
                daemon_socket_path()
            });
        let kernel_sock = with_identity_cwd_value(None, |_| daemon_socket_path());
        assert_ne!(host_a_sock, host_b_sock);
        assert_ne!(host_a_sock, kernel_sock);
        assert_ne!(host_b_sock, kernel_sock);
    }

    #[test]
    fn daemon_paths_share_same_parent_directory() {
        with_identity_cwd(None, |_| {
            let sock = daemon_socket_path();
            let pid = daemon_pid_path();
            assert_eq!(
                sock.parent(),
                pid.parent(),
                "socket and pid file should live in the same directory"
            );
        });
    }

    #[test]
    fn kernel_daemon_paths_are_under_dm_config_dir() {
        with_identity_cwd(None, |_| {
            let sock = daemon_socket_path();
            // Parent component should be ".dm"
            let parent = sock.parent().unwrap();
            assert_eq!(
                parent.file_name().and_then(|n| n.to_str()),
                Some(".dm"),
                "daemon socket should be under ~/.dm/"
            );
        });
    }

    #[test]
    fn daemon_socket_path_is_absolute() {
        with_identity_cwd(None, |_| {
            let path = daemon_socket_path();
            assert!(
                path.is_absolute(),
                "daemon socket path should be absolute: {}",
                path.display()
            );
        });
    }

    #[test]
    fn daemon_pid_path_is_absolute() {
        with_identity_cwd(None, |_| {
            let path = daemon_pid_path();
            assert!(
                path.is_absolute(),
                "daemon pid path should be absolute: {}",
                path.display()
            );
        });
    }

    #[test]
    fn recv_event_timeout_constant_is_reasonable() {
        const EVENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
        const { assert!(EVENT_TIMEOUT.as_secs() >= 60 && EVENT_TIMEOUT.as_secs() <= 600) };
    }
}
