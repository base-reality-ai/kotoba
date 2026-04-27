//! Daemon session persistence: snapshot serialization and on-disk round trip.
//!
//! The snapshot type, atomic writes to `~/.dm/daemon/sessions/<id>.json`,
//! tolerant reload from disk, and directory helpers. Wiring into `run_daemon`
//! (startup reload + shutdown save) lives in `daemon/server.rs`. The 5-minute
//! periodic timer is a later cycle.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Serializable projection of a live daemon session. Built by the agent task
/// on demand (via `AgentCommand::Snapshot`), so the live channels and
/// non-serializable `Instant` in `SessionHandle` stay out of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub session_id: String,
    /// Seconds since `UNIX_EPOCH` at snapshot time. `Instant` is monotonic and
    /// non-serializable, so we capture wall-clock on the way out.
    pub created_at_unix: u64,
    pub messages: Vec<serde_json::Value>,
    pub session: crate::session::Session,
}

/// Directory where daemon session snapshots are persisted for this config.
/// `<config_dir>/daemon/sessions/` — matches the directive spec
/// `~/.dm/daemon/sessions/`.
pub fn sessions_persist_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("daemon").join("sessions")
}

/// Delete every `*.json` file directly under `dir`. Preserves `.tmp` files
/// (those are prior-crash forensics, not ours to clean) and subdirectories.
/// Returns `Ok(())` if `dir` does not exist — shutdown on a fresh install
/// should not error just because the directory has never been created.
pub fn clear_sessions_dir(dir: &Path) -> io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Err(e) = fs::remove_file(&path) {
                crate::logging::log_err(&format!(
                    "[daemon] clear_sessions_dir: failed to remove {}: {}",
                    path.display(),
                    e
                ));
            }
        }
    }
    Ok(())
}

impl SessionSnapshot {
    pub fn now(
        session_id: String,
        messages: Vec<serde_json::Value>,
        session: crate::session::Session,
    ) -> Self {
        let created_at_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            session_id,
            created_at_unix,
            messages,
            session,
        }
    }
}

/// Write each snapshot to `<dir>/<session_id>.json` via a `.tmp` + rename so a
/// crash mid-serialization leaves the prior file intact. Individual failures
/// are logged and skipped; we never abort the batch.
pub fn persist_sessions_to_dir(dir: &Path, snapshots: &[SessionSnapshot]) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    for snap in snapshots {
        let final_path = dir.join(format!("{}.json", snap.session_id));
        let tmp_path = dir.join(format!("{}.json.tmp", snap.session_id));
        match write_one_atomically(&tmp_path, &final_path, snap) {
            Ok(()) => {}
            Err(e) => {
                crate::logging::log_err(&format!(
                    "[daemon] persist failed for session {}: {}",
                    snap.session_id, e
                ));
                let _ = fs::remove_file(&tmp_path);
            }
        }
    }
    Ok(())
}

fn write_one_atomically(
    tmp_path: &Path,
    final_path: &Path,
    snap: &SessionSnapshot,
) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(snap)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(tmp_path, &bytes)?;
    fs::rename(tmp_path, final_path)?;
    Ok(())
}

/// Snapshot every live session and atomically mirror the set to `persist_dir`.
/// Clears the dir first so it reflects exactly the live set (no stale files
/// from prior runs). Logs outcomes via `write_log_entry` with `trigger`
/// embedded in the extra JSON for forensic filtering (`"shutdown"`,
/// `"periodic"`). Returns the number of sessions that produced a snapshot.
pub async fn snapshot_and_persist_all(
    session_manager: &tokio::sync::Mutex<crate::daemon::session_manager::SessionManager>,
    persist_dir: &Path,
    config_dir: &Path,
    timeout: std::time::Duration,
    trigger: &str,
) -> usize {
    let snapshots = {
        let sm = session_manager.lock().await;
        sm.snapshot_all(timeout).await
    };
    let count = snapshots.len();
    if let Err(e) = clear_sessions_dir(persist_dir) {
        crate::daemon::server::write_log_entry(
            config_dir,
            "warn",
            &format!("{}: clear persist dir failed: {}", trigger, e),
            Some(serde_json::json!({"trigger": trigger})),
        );
    }
    match persist_sessions_to_dir(persist_dir, &snapshots) {
        Ok(()) => crate::daemon::server::write_log_entry(
            config_dir,
            "info",
            &format!("{}: persisted {} session(s)", trigger, count),
            Some(serde_json::json!({
                "sessions_persisted": count,
                "trigger": trigger,
            })),
        ),
        Err(e) => crate::daemon::server::write_log_entry(
            config_dir,
            "error",
            &format!("{}: persist failed: {}", trigger, e),
            Some(serde_json::json!({"trigger": trigger})),
        ),
    }
    count
}

/// Load every `*.json` in `dir` as a `SessionSnapshot`. A corrupt file is
/// logged and skipped — we do not delete it, so an operator can inspect it.
/// Stray `*.json.tmp` leftovers from a crashed write are ignored.
pub fn load_persisted_sessions(dir: &Path) -> Vec<SessionSnapshot> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match fs::read_to_string(&path) {
            Ok(body) => match serde_json::from_str::<SessionSnapshot>(&body) {
                Ok(snap) => out.push(snap),
                Err(e) => crate::logging::log_err(&format!(
                    "[daemon] skipping corrupt snapshot {}: {}",
                    path.display(),
                    e
                )),
            },
            Err(e) => crate::logging::log_err(&format!(
                "[daemon] skipping unreadable snapshot {}: {}",
                path.display(),
                e
            )),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_snapshot(id: &str, msg_count: usize) -> SessionSnapshot {
        let mut session =
            crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        session.id = id.to_string();
        let messages: Vec<serde_json::Value> = (0..msg_count)
            .map(|i| serde_json::json!({"role": "user", "content": format!("msg {}", i)}))
            .collect();
        session.messages = messages.clone();
        SessionSnapshot::now(id.to_string(), messages, session)
    }

    #[test]
    fn snapshot_roundtrip_serde() {
        let snap = make_snapshot("abc123", 3);
        let encoded = serde_json::to_string(&snap).expect("encode");
        let decoded: SessionSnapshot = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded.session_id, "abc123");
        assert_eq!(decoded.messages.len(), 3);
        assert_eq!(decoded.session.id, "abc123");
        assert_eq!(decoded.messages[1]["content"].as_str().unwrap(), "msg 1");
    }

    #[test]
    fn persist_sessions_to_dir_creates_files() {
        let dir = TempDir::new().unwrap();
        let snaps = vec![
            make_snapshot("s1", 1),
            make_snapshot("s2", 2),
            make_snapshot("s3", 5),
        ];
        persist_sessions_to_dir(dir.path(), &snaps).expect("persist");
        for id in ["s1", "s2", "s3"] {
            let p = dir.path().join(format!("{}.json", id));
            assert!(p.exists(), "{} should be written", p.display());
            let body = fs::read_to_string(&p).unwrap();
            let decoded: SessionSnapshot = serde_json::from_str(&body).unwrap();
            assert_eq!(decoded.session_id, id);
        }
    }

    #[test]
    fn load_persisted_sessions_roundtrip() {
        let dir = TempDir::new().unwrap();
        let snaps = vec![
            make_snapshot("a", 2),
            make_snapshot("b", 4),
            make_snapshot("c", 0),
        ];
        persist_sessions_to_dir(dir.path(), &snaps).expect("persist");
        let mut loaded = load_persisted_sessions(dir.path());
        loaded.sort_by(|l, r| l.session_id.cmp(&r.session_id));
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].session_id, "a");
        assert_eq!(loaded[1].session_id, "b");
        assert_eq!(loaded[2].session_id, "c");
        assert_eq!(loaded[1].messages.len(), 4);
    }

    #[test]
    fn load_persisted_sessions_tolerates_corrupt_files() {
        let dir = TempDir::new().unwrap();
        let snaps = vec![make_snapshot("good", 1)];
        persist_sessions_to_dir(dir.path(), &snaps).expect("persist");
        fs::write(dir.path().join("bogus.json"), "{").unwrap();
        fs::write(dir.path().join("not_json.json"), "not json at all").unwrap();
        let loaded = load_persisted_sessions(dir.path());
        assert_eq!(loaded.len(), 1, "corrupt files must be skipped");
        assert_eq!(loaded[0].session_id, "good");
        assert!(
            dir.path().join("bogus.json").exists(),
            "corrupt file must be preserved for forensics"
        );
    }

    #[test]
    fn atomic_write_leaves_no_tmp_after_success() {
        let dir = TempDir::new().unwrap();
        // Simulate a prior-crash leftover; persist must overwrite cleanly.
        fs::write(dir.path().join("sX.json.tmp"), "stale garbage").unwrap();
        let snaps = vec![make_snapshot("sX", 2)];
        persist_sessions_to_dir(dir.path(), &snaps).expect("persist");

        let final_path = dir.path().join("sX.json");
        assert!(final_path.exists());
        // After a successful rename, the final .tmp from this write is gone.
        // (A stale .tmp with a different name could remain from a prior crash;
        // this test only asserts the happy-path write does not leave its own.)
        let body = fs::read_to_string(&final_path).unwrap();
        let decoded: SessionSnapshot = serde_json::from_str(&body).unwrap();
        assert_eq!(decoded.session_id, "sX");

        // Verify no *sX*.json.tmp lingers from this write.
        let tmp_for_this = dir.path().join("sX.json.tmp");
        // The stale tmp was renamed-over, so it should not exist.
        assert!(
            !tmp_for_this.exists(),
            "successful write must not leave sX.json.tmp behind"
        );
    }

    #[test]
    fn load_from_missing_dir_returns_empty() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist");
        let loaded = load_persisted_sessions(&missing);
        assert!(
            loaded.is_empty(),
            "missing dir yields empty vec, not a panic"
        );
    }

    #[test]
    fn sessions_persist_dir_returns_expected_path() {
        let got = sessions_persist_dir(Path::new("/tmp/fakedm"));
        assert_eq!(got, PathBuf::from("/tmp/fakedm/daemon/sessions"));
    }

    #[test]
    fn clear_sessions_dir_removes_json_preserves_tmp() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.json"), "{}").unwrap();
        fs::write(dir.path().join("b.json"), "{}").unwrap();
        fs::write(dir.path().join("c.json.tmp"), "crashed mid-write").unwrap();
        fs::write(dir.path().join("notes.txt"), "readme").unwrap();

        clear_sessions_dir(dir.path()).expect("clear");

        assert!(!dir.path().join("a.json").exists());
        assert!(!dir.path().join("b.json").exists());
        assert!(
            dir.path().join("c.json.tmp").exists(),
            ".tmp files are operator forensics, must survive"
        );
        assert!(
            dir.path().join("notes.txt").exists(),
            "non-json files must survive"
        );
    }

    #[test]
    fn clear_sessions_dir_tolerates_missing_dir() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert!(clear_sessions_dir(&missing).is_ok());
    }

    fn test_config(config_dir: std::path::PathBuf) -> std::sync::Arc<crate::config::Config> {
        std::sync::Arc::new(crate::config::Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "gemma4:26b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir,
            routing: None,
            aliases: std::collections::HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        })
    }

    fn seed_session(manager: &mut crate::daemon::session_manager::SessionManager, id: &str) {
        let mut session =
            crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        session.id = id.to_string();
        let msgs = vec![serde_json::json!({"role": "user", "content": format!("hi {}", id)})];
        session.messages = msgs.clone();
        let snap = SessionSnapshot::now(id.to_string(), msgs, session);
        assert_eq!(manager.reload_from_snapshots(vec![snap]), 1);
    }

    fn count_json_files(dir: &Path) -> usize {
        fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
                    .count()
            })
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn snapshot_and_persist_all_writes_expected_count() {
        use crate::daemon::session_manager::SessionManager;
        use std::time::Duration;

        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path().join("cfg"));
        std::fs::create_dir_all(&cfg.config_dir).unwrap();
        let persist_dir = sessions_persist_dir(&cfg.config_dir);

        let mut manager = SessionManager::new(std::sync::Arc::clone(&cfg));
        seed_session(&mut manager, "alpha");
        seed_session(&mut manager, "beta");
        let mutex = tokio::sync::Mutex::new(manager);

        let count = snapshot_and_persist_all(
            &mutex,
            &persist_dir,
            &cfg.config_dir,
            Duration::from_secs(10),
            "periodic",
        )
        .await;

        assert_eq!(count, 2);
        assert_eq!(count_json_files(&persist_dir), 2);
    }

    #[tokio::test]
    async fn snapshot_and_persist_all_mirrors_live_set() {
        use crate::daemon::session_manager::SessionManager;
        use std::time::Duration;

        let tmp = TempDir::new().unwrap();
        let cfg = test_config(tmp.path().join("cfg"));
        std::fs::create_dir_all(&cfg.config_dir).unwrap();
        let persist_dir = sessions_persist_dir(&cfg.config_dir);

        let mut manager = SessionManager::new(std::sync::Arc::clone(&cfg));
        seed_session(&mut manager, "keep");
        seed_session(&mut manager, "evict");
        let mutex = tokio::sync::Mutex::new(manager);

        // First pass: both persisted.
        let c1 = snapshot_and_persist_all(
            &mutex,
            &persist_dir,
            &cfg.config_dir,
            Duration::from_secs(10),
            "periodic",
        )
        .await;
        assert_eq!(c1, 2);
        assert_eq!(count_json_files(&persist_dir), 2);

        // Drop one live session, persist again — disk must mirror the live set.
        mutex.lock().await.remove("evict");
        let c2 = snapshot_and_persist_all(
            &mutex,
            &persist_dir,
            &cfg.config_dir,
            Duration::from_secs(10),
            "periodic",
        )
        .await;
        assert_eq!(c2, 1);
        assert_eq!(count_json_files(&persist_dir), 1);
        assert!(persist_dir.join("keep.json").exists());
        assert!(!persist_dir.join("evict.json").exists());
    }

    #[tokio::test]
    async fn end_to_end_roundtrip_via_persist_dir() {
        use crate::daemon::session_manager::{AgentCommand, SessionManager};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::sync::oneshot;

        // Reuse the same test_config shape as session_manager tests.
        let cfg = Arc::new(crate::config::Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "gemma4:26b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::env::temp_dir().join("dm_persist_e2e"),
            routing: None,
            aliases: std::collections::HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        });

        let tmp = TempDir::new().unwrap();
        let persist_dir = sessions_persist_dir(tmp.path());

        // Manager A: seed a session with a known-marker message via reload.
        let mut manager_a = SessionManager::new(Arc::clone(&cfg));
        let mut session =
            crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        session.id = "e2e-one".to_string();
        let seeded = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "E2E_MARKER_42"}),
        ];
        session.messages = seeded.clone();
        let snap_seed = SessionSnapshot::now("e2e-one".to_string(), seeded, session);
        assert_eq!(manager_a.reload_from_snapshots(vec![snap_seed]), 1);

        // A → disk.
        let snaps_a = manager_a.snapshot_all(Duration::from_secs(10)).await;
        assert_eq!(snaps_a.len(), 1);
        clear_sessions_dir(&persist_dir).expect("clear");
        persist_sessions_to_dir(&persist_dir, &snaps_a).expect("persist");

        // Tear down A entirely; no shared state survives.
        drop(manager_a);

        // Disk → Manager B.
        let loaded = load_persisted_sessions(&persist_dir);
        assert_eq!(loaded.len(), 1);
        let mut manager_b = SessionManager::new(Arc::clone(&cfg));
        assert_eq!(manager_b.reload_from_snapshots(loaded), 1);

        // Round-trip via Snapshot to confirm the marker survived.
        let handle = manager_b.get("e2e-one").expect("reloaded");
        let (tx, rx) = oneshot::channel();
        handle
            .agent_tx
            .try_send(AgentCommand::Snapshot(tx))
            .expect("send snapshot");
        let back = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .expect("timeout")
            .expect("recv");
        let marker_seen = back
            .messages
            .iter()
            .any(|m| m["content"].as_str() == Some("E2E_MARKER_42"));
        assert!(marker_seen, "marker survived A → disk → B round trip");
    }

    /// C91 canary for the 10s `snapshot_all` budget: a single reloaded
    /// session must return exactly one snapshot within the widened
    /// budget. Mirrors the happy path of `e2e_round_trip_across_managers`
    /// at `:499` without the disk round trip — guards against any
    /// future change that turns `snapshot_all` itself into a slow path
    /// whose happy case approaches the 10s envelope. Correctness-only:
    /// no `elapsed < X` assertion (see planner guardrail).
    ///
    /// Cycles 12–18 originally inherited a sibling-grouping `#[ignore]`
    /// alongside `agent_tx_closed_resolves_within_nine_seconds_under_moderate_session_load`,
    /// but unlike that sibling — which has a hard `elapsed < 9s`
    /// assertion under 10 concurrent sessions — this test's only
    /// assertion is `snaps.len() == 1`, invariant under load. Cycle 19
    /// un-ignored after verifying clean default + stress runs across
    /// the chain.
    #[tokio::test]
    async fn snapshot_all_with_ten_second_budget_completes_for_single_reloaded_session() {
        use crate::daemon::session_manager::SessionManager;
        use std::sync::Arc;
        use std::time::Duration;

        let cfg = Arc::new(crate::config::Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "gemma4:26b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::env::temp_dir().join("dm_persist_canary_c91"),
            routing: None,
            aliases: std::collections::HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        });

        let mut manager = SessionManager::new(cfg);
        let mut session =
            crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        session.id = "canary-one".to_string();
        let seeded = vec![serde_json::json!({"role": "system", "content": "sys"})];
        let snap = SessionSnapshot::now("canary-one".to_string(), seeded, session);
        assert_eq!(manager.reload_from_snapshots(vec![snap]), 1);

        let snaps = manager.snapshot_all(Duration::from_secs(10)).await;
        assert_eq!(
            snaps.len(),
            1,
            "single reloaded session must produce exactly one snapshot under 10s budget"
        );
    }
}
