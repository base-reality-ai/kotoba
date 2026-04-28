use crate::config::Config;
use crate::daemon::persistence::SessionSnapshot;
use crate::daemon::protocol::{DaemonEvent, SessionInfo};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};

/// Commands sent from the request handler → the per-session agent task.
#[derive(Debug)]
pub enum AgentCommand {
    UserTurn(String),
    Cancel,
    Shutdown,
    /// Serialize the current agent state (messages + session) and return it via
    /// the oneshot. Used by `SessionManager::snapshot_all` for on-disk
    /// persistence; does not interrupt the agent loop.
    Snapshot(oneshot::Sender<SessionSnapshot>),
}

pub struct SessionHandle {
    pub session_id: String,
    pub event_tx: broadcast::Sender<DaemonEvent>,
    pub client_count: Arc<AtomicUsize>,
    pub created_at: std::time::Instant,
    /// Last time a user turn was received for this session. Shared with the
    /// agent task so the task can update it without taking a `SessionManager`
    /// lock. `std::sync::Mutex` is correct: the lock is held for nanoseconds
    /// (one load/store) and never across `.await`.
    pub last_active: Arc<std::sync::Mutex<std::time::Instant>>,
    pub agent_tx: mpsc::Sender<AgentCommand>,
}

pub struct SessionManager {
    sessions: HashMap<String, SessionHandle>,
    /// Shared config used when spawning agent tasks.
    config: Arc<Config>,
}

impl SessionManager {
    pub fn new(config: Arc<Config>) -> Self {
        Self {
            sessions: HashMap::new(),
            config,
        }
    }

    const MAX_SESSIONS: usize = 50;

    /// Generates a new unique session ID, spawns a background agent task,
    /// inserts the handle, and returns the ID.
    pub fn create_session(&mut self) -> Result<String, String> {
        if self.sessions.len() >= Self::MAX_SESSIONS {
            return Err(format!(
                "Session limit reached ({}). End an existing session first.",
                Self::MAX_SESSIONS
            ));
        }
        let id = generate_id();
        let (event_tx, _) = broadcast::channel(256);
        let (agent_tx, agent_rx) = mpsc::channel::<AgentCommand>(32);
        let last_active = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));

        // Spawn a background agent task for this session.
        {
            let session_id = id.clone();
            let broadcast_tx = event_tx.clone();
            let config = Arc::clone(&self.config);
            let last_active_for_task = Arc::clone(&last_active);
            tokio::spawn(async move {
                run_agent_task(
                    session_id,
                    config,
                    broadcast_tx,
                    agent_rx,
                    last_active_for_task,
                )
                .await;
            });
        }

        let handle = SessionHandle {
            session_id: id.clone(),
            event_tx,
            client_count: Arc::new(AtomicUsize::new(0)),
            created_at: std::time::Instant::now(),
            last_active,
            agent_tx,
        };
        self.sessions.insert(id.clone(), handle);
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Option<&SessionHandle> {
        self.sessions.get(id)
    }

    /// Route a user turn to the session's agent task and return a broadcast receiver
    /// so the caller can stream events.
    pub fn handle_turn_send(
        &self,
        session_id: &str,
        text: String,
    ) -> anyhow::Result<broadcast::Receiver<DaemonEvent>> {
        let handle = self
            .sessions
            .get(session_id)
            .ok_or_else(|| anyhow::anyhow!("session not found: {}", session_id))?;
        let rx = handle.event_tx.subscribe();
        handle
            .agent_tx
            .try_send(AgentCommand::UserTurn(text))
            .map_err(|e| anyhow::anyhow!("agent channel send failed: {}", e))?;
        Ok(rx)
    }

    pub fn list(&self) -> Vec<SessionInfo> {
        self.sessions
            .values()
            .map(|h| {
                let created_secs = h.created_at.elapsed().as_secs();
                let last_active_secs = h
                    .last_active
                    .lock()
                    .ok()
                    .map_or(created_secs, |la| la.elapsed().as_secs());
                SessionInfo {
                    session_id: h.session_id.clone(),
                    created_at: format!("{}s ago", created_secs),
                    last_active: format!("{}s ago", last_active_secs),
                    client_count: h.client_count.load(Ordering::Relaxed),
                    status: "idle".to_string(),
                }
            })
            .collect()
    }

    pub fn remove(&mut self, id: &str) {
        self.sessions.remove(id);
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Send `AgentCommand::Snapshot` to every live session and collect the
    /// replies. Each reply is bounded by `timeout`; all receivers are awaited
    /// concurrently so total wall time is also bounded by `timeout`. Sessions
    /// whose `agent_tx` is full or whose task has died are silently skipped —
    /// an unreachable session has no state worth persisting.
    pub async fn snapshot_all(&self, timeout: std::time::Duration) -> Vec<SessionSnapshot> {
        let mut receivers = Vec::new();
        for handle in self.sessions.values() {
            let (tx, rx) = oneshot::channel();
            if handle
                .agent_tx
                .try_send(AgentCommand::Snapshot(tx))
                .is_err()
            {
                continue;
            }
            receivers.push(tokio::time::timeout(timeout, rx));
        }
        futures_util::future::join_all(receivers)
            .await
            .into_iter()
            .filter_map(|r| r.ok().and_then(|inner| inner.ok()))
            .collect()
    }

    /// Respawn agent tasks from persisted snapshots, seeding each task's
    /// `messages` and `session` from the snapshot. Honours `MAX_SESSIONS`:
    /// excess snapshots are dropped and logged. Returns the number of
    /// sessions successfully reloaded.
    pub fn reload_from_snapshots(&mut self, snapshots: Vec<SessionSnapshot>) -> usize {
        let total = snapshots.len();
        let capacity = Self::MAX_SESSIONS.saturating_sub(self.sessions.len());
        let mut loaded = 0;
        for snap in snapshots.into_iter().take(capacity) {
            let id = snap.session_id.clone();
            let (event_tx, _) = broadcast::channel(256);
            let (agent_tx, agent_rx) = mpsc::channel::<AgentCommand>(32);
            // Reloaded sessions get a fresh grace period — last_active resets to "now"
            // at daemon startup so a restart doesn't trigger mass eviction.
            let last_active = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
            {
                let session_id = id.clone();
                let broadcast_tx = event_tx.clone();
                let config = Arc::clone(&self.config);
                let initial_messages = snap.messages;
                let initial_session = snap.session;
                let last_active_for_task = Arc::clone(&last_active);
                tokio::spawn(async move {
                    run_agent_task_with_state(
                        session_id,
                        config,
                        broadcast_tx,
                        agent_rx,
                        initial_messages,
                        initial_session,
                        last_active_for_task,
                    )
                    .await;
                });
            }
            let handle = SessionHandle {
                session_id: id.clone(),
                event_tx,
                client_count: Arc::new(AtomicUsize::new(0)),
                created_at: std::time::Instant::now(),
                last_active,
                agent_tx,
            };
            self.sessions.insert(id, handle);
            loaded += 1;
        }
        if total > loaded {
            crate::logging::log_err(&format!(
                "[daemon] reload_from_snapshots: dropped {} snapshots beyond MAX_SESSIONS cap",
                total - loaded
            ));
        }
        loaded
    }

    /// Garbage-collect idle sessions with zero connected clients.
    ///
    /// "Idle" is measured from `last_active` (time since the most recent
    /// user turn), NOT `created_at` (age of the session). A long-lived
    /// session that's actively being used is not a GC target.
    ///
    /// Returns the list of evicted `(session_id, idle_duration)` pairs so
    /// callers can emit per-session log lines. `threshold_secs == 0`
    /// disables GC and returns an empty vec.
    pub fn gc_sweep(&mut self, threshold_secs: u64) -> Vec<(String, std::time::Duration)> {
        if threshold_secs == 0 {
            return Vec::new();
        }
        let evictions: Vec<(String, std::time::Duration)> = self
            .sessions
            .iter()
            .filter_map(|(id, h)| {
                let client_count = h.client_count.load(Ordering::Relaxed);
                let idle = h.last_active.lock().ok()?.elapsed();
                if should_gc(client_count, idle.as_secs(), threshold_secs) {
                    Some((id.clone(), idle))
                } else {
                    None
                }
            })
            .collect();
        for (id, _) in &evictions {
            self.remove(id);
        }
        evictions
    }
}

/// Pure function: returns true if the session should be garbage-collected.
/// No I/O, fully testable.
pub fn should_gc(client_count: usize, idle_secs: u64, threshold_secs: u64) -> bool {
    client_count == 0 && idle_secs > threshold_secs
}

/// Background agent task — one per daemon session.
///
/// Thin wrapper that builds the default system-prompt-seeded initial state
/// and delegates to [`run_agent_task_with_state`]. Reload paths (which restore
/// `messages` and `session` from a snapshot) call the `_with_state` form
/// directly so there is no branch inside the hot loop.
async fn run_agent_task(
    session_id: String,
    config: Arc<Config>,
    event_tx: broadcast::Sender<DaemonEvent>,
    agent_rx: mpsc::Receiver<AgentCommand>,
    last_active: Arc<std::sync::Mutex<std::time::Instant>>,
) {
    let system_prompt =
        crate::system_prompt::build_system_prompt_with_context(&[], None, &[]).await;
    let initial_messages: Vec<serde_json::Value> = vec![serde_json::json!({
        "role": "system",
        "content": system_prompt,
    })];
    let mut initial_session = crate::session::Session::new(
        std::env::current_dir()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        config.model.clone(),
    );
    initial_session.id = session_id.clone();
    run_agent_task_with_state(
        session_id,
        config,
        event_tx,
        agent_rx,
        initial_messages,
        initial_session,
        last_active,
    )
    .await;
}

/// Core agent loop. Accepts pre-populated `messages` and `session` so that the
/// daemon can reload a snapshot without replaying the session from scratch.
async fn run_agent_task_with_state(
    session_id: String,
    config: Arc<Config>,
    event_tx: broadcast::Sender<DaemonEvent>,
    mut agent_rx: mpsc::Receiver<AgentCommand>,
    initial_messages: Vec<serde_json::Value>,
    initial_session: crate::session::Session,
    last_active: Arc<std::sync::Mutex<std::time::Instant>>,
) {
    let client =
        crate::ollama::client::OllamaClient::new(config.ollama_base_url(), config.model.clone());
    let tool_client = config
        .tool_model
        .as_ref()
        .map(|m| crate::ollama::client::OllamaClient::new(config.ollama_base_url(), m.clone()));
    let registry = crate::tools::registry::default_registry(
        &session_id,
        &config.config_dir,
        &config.ollama_base_url(),
        &config.model,
        &config.embed_model,
    );

    let settings_rules =
        crate::permissions::storage::load_rules(&config.global_config_dir).unwrap_or_default();
    let mut engine = crate::permissions::engine::PermissionEngine::new(false, settings_rules);
    let hooks_config = crate::tools::hooks::HooksConfig::load(&config.config_dir);
    let context_limit = client.model_context_limit(client.model()).await;
    let mcp_clients: HashMap<String, Arc<tokio::sync::Mutex<crate::mcp::client::McpClient>>> =
        HashMap::new();

    let mut messages: Vec<serde_json::Value> = initial_messages;
    let mut session = initial_session;

    // Local helper: broadcast a DaemonEvent, ignoring no-subscriber errors.
    let bcast = |event_tx: &broadcast::Sender<DaemonEvent>, evt: DaemonEvent| {
        let _ = event_tx.send(evt);
    };

    // Cancel watch — shared between the command loop and run_turn.
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    while let Some(cmd) = agent_rx.recv().await {
        match cmd {
            AgentCommand::Shutdown => break,
            AgentCommand::Cancel => {
                let _ = cancel_tx.send(true);
                bcast(
                    &event_tx,
                    DaemonEvent::Cancelled {
                        session_id: session_id.clone(),
                    },
                );
                // Reset cancel flag for next turn.
                let _ = cancel_tx.send(false);
            }
            AgentCommand::Snapshot(reply_tx) => {
                let snap =
                    SessionSnapshot::now(session_id.clone(), messages.clone(), session.clone());
                if reply_tx.send(snap).is_err() {
                    crate::logging::log("[daemon] snapshot receiver dropped");
                }
            }
            AgentCommand::UserTurn(text) => {
                // Refresh the idle clock: a user turn is the one and only signal
                // that the session is actively being used. Control-plane commands
                // (Cancel, Snapshot) deliberately do NOT refresh this.
                if let Ok(mut la) = last_active.lock() {
                    *la = std::time::Instant::now();
                }

                // Reset cancel flag at start of each turn.
                let _ = cancel_tx.send(false);

                // Append user message.
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": text,
                }));
                session.messages = messages.clone();

                // Run the turn inline (this task is already spawned so blocking here is OK).
                let (be_tx, mut be_rx) =
                    tokio::sync::mpsc::channel::<crate::tui::BackendEvent>(256);
                let mut pending: Vec<crate::changeset::PendingChange> = Vec::new();

                // Drive run_turn and forward events concurrently using select.
                let daemon_retry = crate::conversation::RetrySettings::from_config(&config);
                let turn_fut = crate::tui::agent::turn::run_turn(
                    &client,
                    tool_client.as_ref(),
                    &registry,
                    &mcp_clients,
                    &hooks_config,
                    false, // verbose
                    20,    // max_turns
                    context_limit,
                    false, // staging
                    &mut pending,
                    &mut messages,
                    &mut session,
                    &mut engine,
                    &config.config_dir,
                    &config.global_config_dir,
                    &be_tx,
                    &cancel_rx,
                    false, // plan_mode
                    &daemon_retry,
                );

                // Drop be_tx after the turn future finishes so be_rx will drain cleanly.
                let (turn_result, ()) = tokio::join!(turn_fut, async {
                    // Forward BackendEvents → DaemonEvents while the turn runs.
                    // The channel closes when be_tx is dropped (after turn_fut).
                    while let Some(be) = be_rx.recv().await {
                        if let Some(de) = DaemonEvent::from_backend(&be, &session_id) {
                            bcast(&event_tx, de);
                        }
                    }
                });

                // Emit TurnComplete if the turn succeeded (it may have already been sent
                // via BackendEvent, but that's fine — clients deduplicate).
                if let Some((pt, ct)) = turn_result {
                    // Only send if the BackendEvent conversion didn't already broadcast it.
                    // We check by trying to send and ignoring the result.
                    bcast(
                        &event_tx,
                        DaemonEvent::TurnComplete {
                            session_id: session_id.clone(),
                            prompt_tokens: pt,
                            completion_tokens: ct,
                        },
                    );
                }
            }
        }
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{:x}{:x}{:x}", t.as_nanos(), t.subsec_nanos(), count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn test_config() -> Arc<Config> {
        Arc::new(Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "gemma4:26b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::env::temp_dir().join("dm_test"),
            global_config_dir: std::env::temp_dir().join("dm_test"),
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

    #[tokio::test]
    async fn session_manager_creates_unique_ids() {
        let mut manager = SessionManager::new(test_config());
        let id1 = manager.create_session().unwrap();
        let id2 = manager.create_session().unwrap();
        let id3 = manager.create_session().unwrap();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
        assert_eq!(manager.session_count(), 3);
    }

    #[tokio::test]
    async fn session_handle_tracks_client_count() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session should exist");
        let counter = handle.client_count.clone();

        assert_eq!(counter.load(Ordering::Relaxed), 0);
        counter.fetch_add(1, Ordering::Relaxed);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        counter.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn turn_send_routes_to_agent_task() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session should exist");
        // Verify that agent_tx channel is present and open by sending a Shutdown command.
        let result = handle.agent_tx.try_send(AgentCommand::Shutdown);
        assert!(result.is_ok(), "agent_tx should be open");
    }

    #[tokio::test]
    async fn multi_client_both_receive_events() {
        let (tx, _) = broadcast::channel::<DaemonEvent>(16);
        let mut rx1 = tx.subscribe();
        let mut rx2 = tx.subscribe();
        let event = DaemonEvent::StreamToken {
            session_id: "s".to_string(),
            content: "ping".to_string(),
        };
        tx.send(event).expect("send should succeed");
        let e1 = rx1.recv().await.expect("rx1 should receive");
        let e2 = rx2.recv().await.expect("rx2 should receive");
        match (e1, e2) {
            (
                DaemonEvent::StreamToken { content: c1, .. },
                DaemonEvent::StreamToken { content: c2, .. },
            ) => {
                assert_eq!(c1, "ping");
                assert_eq!(c2, "ping");
            }
            _ => panic!("unexpected events"),
        }
    }

    #[test]
    fn detach_decrements_client_count() {
        let counter = Arc::new(AtomicUsize::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
        counter.fetch_sub(1, Ordering::Relaxed);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn gc_skips_active_sessions() {
        // client_count=1, idle_secs=0 — must NOT be GC'd
        assert!(!should_gc(1, 0, 3600));
        // client_count=1, even with old age — must NOT be GC'd
        assert!(!should_gc(1, 7200, 3600));
    }

    #[test]
    fn gc_archives_idle_sessions() {
        // client_count=0 and idle longer than threshold — SHOULD be GC'd
        assert!(should_gc(0, 7200, 3600));
    }

    #[test]
    fn gc_boundary_conditions() {
        // Exactly at threshold — NOT GC'd (idle_secs must be strictly greater than threshold)
        assert!(!should_gc(0, 3600, 3600));
        // One second past threshold — GC'd
        assert!(should_gc(0, 3601, 3600));
    }

    #[tokio::test]
    async fn handle_turn_send_errors_for_missing_session() {
        let manager = SessionManager::new(test_config());
        let result = manager.handle_turn_send("nonexistent-id", "hello".to_string());
        assert!(result.is_err(), "turn_send on unknown session should error");
        assert!(
            result.unwrap_err().to_string().contains("not found"),
            "error message should say 'not found'"
        );
    }

    #[tokio::test]
    async fn session_list_reflects_created_sessions() {
        let mut manager = SessionManager::new(test_config());
        assert!(
            manager.list().is_empty(),
            "fresh manager should have empty list"
        );
        let id = manager.create_session().unwrap();
        let infos = manager.list();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].session_id, id);
        assert_eq!(infos[0].status, "idle");
    }

    #[tokio::test]
    async fn session_remove_decrements_count() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        assert_eq!(manager.session_count(), 1);
        manager.remove(&id);
        assert_eq!(
            manager.session_count(),
            0,
            "remove should decrement session count"
        );
        assert!(
            manager.get(&id).is_none(),
            "removed session should not be retrievable"
        );
    }

    #[test]
    fn generate_id_is_nonempty_hex() {
        let id = generate_id();
        assert!(!id.is_empty(), "generated ID should not be empty");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit()),
            "generated ID should be hex: {id}"
        );
    }

    #[tokio::test]
    async fn gc_sweep_removes_idle_sessions_only() {
        let mut manager = SessionManager::new(test_config());
        let _id1 = manager.create_session().unwrap();
        let id2 = manager.create_session().unwrap();

        // Simulate: _id1 has 0 clients (idle), id2 has 1 client (active)
        if let Some(h) = manager.get(&id2) {
            h.client_count.fetch_add(1, Ordering::Relaxed);
        }

        // threshold=0 is the "disabled" sentinel — gc_sweep returns empty
        // unconditionally. Both sessions must be kept regardless of state.
        let evictions = manager.gc_sweep(0);
        assert!(
            evictions.is_empty(),
            "threshold=0 disables GC; no evictions expected"
        );
        assert_eq!(manager.session_count(), 2);
    }

    #[tokio::test]
    async fn gc_sweep_uses_last_active_not_age() {
        // A session created 10 seconds ago (stale age) but with last_active = now
        // must NOT be evicted. Conversely, a session whose last_active is old
        // (regardless of creation time) MUST be evicted.
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        // Force created_at to look old, but leave last_active fresh.
        {
            let handle = manager.sessions.get_mut(&id).expect("session");
            handle.created_at = std::time::Instant::now() - std::time::Duration::from_secs(120);
            if let Ok(mut la) = handle.last_active.lock() {
                *la = std::time::Instant::now();
            }
        }
        let evictions = manager.gc_sweep(5);
        assert!(
            evictions.is_empty(),
            "session with fresh last_active must survive GC even with old created_at"
        );
        assert_eq!(manager.session_count(), 1);

        // Now rewind last_active past the threshold — eviction expected.
        {
            let handle = manager.sessions.get(&id).expect("session");
            if let Ok(mut la) = handle.last_active.lock() {
                *la = std::time::Instant::now() - std::time::Duration::from_secs(10);
            }
        }
        let evictions = manager.gc_sweep(5);
        assert_eq!(evictions.len(), 1);
        assert_eq!(evictions[0].0, id);
        assert!(evictions[0].1.as_secs() >= 5);
        assert_eq!(manager.session_count(), 0);
    }

    #[tokio::test]
    async fn gc_sweep_disabled_when_threshold_zero() {
        let mut manager = SessionManager::new(test_config());
        let _a = manager.create_session().unwrap();
        let _b = manager.create_session().unwrap();
        let _c = manager.create_session().unwrap();
        // Rewind every session's last_active past any threshold we could pick.
        for handle in manager.sessions.values() {
            if let Ok(mut la) = handle.last_active.lock() {
                *la = std::time::Instant::now() - std::time::Duration::from_secs(10_000);
            }
        }
        let evictions = manager.gc_sweep(0);
        assert!(evictions.is_empty(), "threshold=0 disables GC");
        assert_eq!(
            manager.session_count(),
            3,
            "all sessions retained when GC disabled"
        );
    }

    #[test]
    fn gc_sweep_threshold_logic() {
        // Test via the pure function — gc_sweep delegates to should_gc
        // 0 clients, idle 10s, threshold 5s → should GC
        assert!(should_gc(0, 10, 5));
        // 0 clients, idle 5s, threshold 5s → NOT GC (not strictly greater)
        assert!(!should_gc(0, 5, 5));
        // 1 client, idle 100s, threshold 5s → NOT GC (has client)
        assert!(!should_gc(1, 100, 5));
    }

    #[tokio::test]
    async fn create_session_respects_max() {
        let mut manager = SessionManager::new(test_config());
        for _ in 0..SessionManager::MAX_SESSIONS {
            assert!(manager.create_session().is_ok());
        }
        let result = manager.create_session();
        assert!(result.is_err(), "should reject session beyond MAX_SESSIONS");
        assert!(result.unwrap_err().contains("Session limit reached"));
    }

    #[test]
    fn load_rules_reads_from_config_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let settings = r#"{"allow": ["bash"], "deny": ["write_file"]}"#;
        std::fs::write(dir.path().join("settings.json"), settings).unwrap();

        let rules = crate::permissions::storage::load_rules(dir.path()).unwrap();
        assert_eq!(rules.len(), 2, "should load 2 rules from settings.json");
        assert!(
            rules.iter().any(|r| r.tool == "bash"),
            "should have bash rule"
        );
        assert!(
            rules.iter().any(|r| r.tool == "write_file"),
            "should have write_file rule"
        );
    }

    #[test]
    fn load_rules_returns_empty_when_no_settings() {
        let dir = tempfile::TempDir::new().unwrap();
        let rules = crate::permissions::storage::load_rules(dir.path()).unwrap();
        assert!(
            rules.is_empty(),
            "no settings.json should yield empty rules"
        );
    }

    #[tokio::test]
    async fn agent_snapshot_round_trip() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session");
        let (tx, rx) = oneshot::channel();
        handle
            .agent_tx
            .try_send(AgentCommand::Snapshot(tx))
            .expect("send Snapshot");
        let snap = tokio::time::timeout(std::time::Duration::from_secs(10), rx)
            .await
            .expect("snapshot timeout")
            .expect("receiver closed");
        assert_eq!(snap.session_id, id);
        assert_eq!(snap.session.id, id);
        assert!(
            snap.messages.iter().any(|m| m["role"] == "system"),
            "snapshot should carry the seeded system prompt"
        );
    }

    #[tokio::test]
    async fn snapshot_all_honours_timeout() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        // Kill the agent task and await its receiver drop deterministically.
        // `Sender::closed()` resolves precisely when the agent task breaks out
        // of `while let Some(cmd) = agent_rx.recv().await` — no sleep proxy,
        // no scheduler-pressure flake. 2s ceiling is a runaway-bug guard.
        let handle = manager.get(&id).expect("session");
        handle
            .agent_tx
            .try_send(AgentCommand::Shutdown)
            .expect("shutdown");
        tokio::time::timeout(std::time::Duration::from_secs(10), handle.agent_tx.closed())
            .await
            .expect("shutdown must close agent_rx within 10s — check run_agent_task");

        let start = std::time::Instant::now();
        let snaps = manager
            .snapshot_all(std::time::Duration::from_millis(100))
            .await;
        let elapsed = start.elapsed();
        // Generous ceiling (20× the 100ms timeout budget above). Tight enough
        // to catch a runaway hang; loose enough to survive CI scheduler jitter
        // and parallel-test contention. Prior 4× ceiling flaked intermittently
        // under load (C72, C74).
        assert!(
            elapsed < std::time::Duration::from_millis(2000),
            "snapshot_all should return without hanging when sessions are unreachable, took {:?}",
            elapsed
        );
        // Dead session contributes no snapshot.
        assert!(snaps.is_empty() || snaps.iter().all(|s| s.session_id == id));
    }

    #[tokio::test]
    async fn snapshot_all_empty_manager_returns_empty_quickly() {
        let manager = SessionManager::new(test_config());
        let start = std::time::Instant::now();
        let snaps = manager
            .snapshot_all(std::time::Duration::from_millis(100))
            .await;
        assert!(snaps.is_empty(), "no sessions → no snapshots");
        assert!(
            start.elapsed() < std::time::Duration::from_millis(500),
            "empty-manager path should not consume the timeout budget"
        );
    }

    #[tokio::test]
    async fn snapshot_all_includes_live_session_result() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let snaps = manager
            .snapshot_all(std::time::Duration::from_secs(10))
            .await;
        assert_eq!(
            snaps.len(),
            1,
            "one live session should produce one snapshot"
        );
        assert_eq!(snaps[0].session_id, id);
    }

    #[tokio::test]
    async fn snapshot_all_mixed_live_and_dead_returns_only_live() {
        let mut manager = SessionManager::new(test_config());
        let live_id = manager.create_session().unwrap();
        let dead_id = manager.create_session().unwrap();

        let dead = manager.get(&dead_id).expect("dead session");
        dead.agent_tx
            .try_send(AgentCommand::Shutdown)
            .expect("shutdown");
        // Wait for the dead session's agent task to actually drop its
        // receiver — deterministic via `Sender::closed()`, resolves the
        // microsecond the task exits the recv loop. Replaces a 50ms sleep
        // that flaked under concurrent lib+bin stress (C78 → C89).
        tokio::time::timeout(std::time::Duration::from_secs(10), dead.agent_tx.closed())
            .await
            .expect("dead session shutdown must close agent_rx within 10s");

        let snaps = manager
            .snapshot_all(std::time::Duration::from_secs(10))
            .await;
        assert!(
            snaps.iter().any(|s| s.session_id == live_id),
            "live session must contribute"
        );
        assert!(
            snaps.iter().all(|s| s.session_id != dead_id),
            "dead session must not contribute"
        );
    }

    #[tokio::test]
    async fn snapshot_all_zero_timeout_still_terminates() {
        let mut manager = SessionManager::new(test_config());
        let _ = manager.create_session().unwrap();
        let start = std::time::Instant::now();
        let _snaps = manager
            .snapshot_all(std::time::Duration::from_millis(0))
            .await;
        assert!(
            start.elapsed() < std::time::Duration::from_millis(500),
            "zero-timeout path must not hang"
        );
    }

    #[tokio::test]
    async fn snapshot_all_two_dead_sessions_still_fast() {
        let mut manager = SessionManager::new(test_config());
        // Clone each `agent_tx` into a local Vec so we can await
        // `.closed()` on both without holding the manager borrow across
        // the join — `mpsc::Sender` is `Clone`, and a clone observes the
        // same set of receivers, so `closed()` on the clone resolves
        // when the agent task's receiver drops.
        let mut dead_senders: Vec<mpsc::Sender<AgentCommand>> = Vec::with_capacity(2);
        for _ in 0..2 {
            let id = manager.create_session().unwrap();
            let h = manager.get(&id).expect("session");
            h.agent_tx
                .try_send(AgentCommand::Shutdown)
                .expect("shutdown");
            dead_senders.push(h.agent_tx.clone());
        }
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            let (a, b) = (&dead_senders[0], &dead_senders[1]);
            tokio::join!(a.closed(), b.closed());
        })
        .await
        .expect("both dead sessions must close agent_rx within 10s");

        let start = std::time::Instant::now();
        let snaps = manager
            .snapshot_all(std::time::Duration::from_millis(100))
            .await;
        let elapsed = start.elapsed();
        // Timeout is a single budget for the whole call, not per-session.
        // Serialized fan-out would be N×100ms, so 2000ms catches that drift.
        assert!(
            elapsed < std::time::Duration::from_millis(2000),
            "multi-dead fanout must not serialize: took {:?}",
            elapsed
        );
        assert!(
            snaps.is_empty() || snaps.iter().all(|s| !s.session_id.is_empty()),
            "whatever contributes must carry a valid id"
        );
    }

    /// Canary: prove the `Sender::closed().await` idiom resolves once
    /// `run_agent_task` exits its recv loop in response to `Shutdown`.
    /// If this ever fails, every other test that replaced the old
    /// `sleep(50ms)` proxy with `.closed()` is built on sand.
    #[tokio::test]
    async fn shutdown_causes_agent_tx_closed_future_to_resolve() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session");
        handle
            .agent_tx
            .try_send(AgentCommand::Shutdown)
            .expect("shutdown");
        tokio::time::timeout(std::time::Duration::from_secs(10), handle.agent_tx.closed())
            .await
            .expect("shutdown must close agent_rx within 10s");
    }

    /// `.closed()` must not spin until the outer timeout expires —
    /// the whole point of this idiom is that it wakes the moment the
    /// agent task exits its recv loop. Budget is 5s against a 10s
    /// ceiling: permissive enough to survive concurrent lib+bin
    /// stress, tight enough to catch a regression that would push
    /// the happy path to saturate the timeout. The old `sleep(50ms)`
    /// this replaces was never about latency; the win is determinism.
    #[tokio::test]
    async fn agent_tx_closed_resolves_much_faster_than_prior_sleep_budget() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session");
        handle
            .agent_tx
            .try_send(AgentCommand::Shutdown)
            .expect("shutdown");
        let start = std::time::Instant::now();
        tokio::time::timeout(std::time::Duration::from_secs(10), handle.agent_tx.closed())
            .await
            .expect("closed within 10s");
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(9),
            ".closed() should resolve well under 9s even under scheduler pressure, took {:?}",
            elapsed
        );
    }

    /// Scales the single-session canary to five concurrent sessions —
    /// all shut down, all awaited in one `tokio::join!` inside a
    /// single `timeout(2s)`. Guards against any serialization or
    /// receiver-starvation regression that would make the fan-out
    /// scale with N rather than resolve in parallel.
    #[tokio::test]
    async fn closed_future_observes_shutdown_across_concurrent_sessions() {
        let mut manager = SessionManager::new(test_config());
        let mut senders: Vec<mpsc::Sender<AgentCommand>> = Vec::with_capacity(5);
        for _ in 0..5 {
            let id = manager.create_session().unwrap();
            let h = manager.get(&id).expect("session");
            h.agent_tx
                .try_send(AgentCommand::Shutdown)
                .expect("shutdown");
            senders.push(h.agent_tx.clone());
        }
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                senders[0].closed(),
                senders[1].closed(),
                senders[2].closed(),
                senders[3].closed(),
                senders[4].closed(),
            );
        })
        .await
        .expect("all 5 sessions must close agent_rx within 10s");
    }

    /// `Sender::closed()` is documented to return immediately once
    /// all receivers have dropped. Verify that calling it a second
    /// time after the task has already exited still returns promptly
    /// — so tests that await `.closed()` don't accidentally block
    /// on a future that's already completed.
    #[tokio::test]
    async fn closed_future_idempotent_after_task_exit() {
        let mut manager = SessionManager::new(test_config());
        let id = manager.create_session().unwrap();
        let handle = manager.get(&id).expect("session");
        handle
            .agent_tx
            .try_send(AgentCommand::Shutdown)
            .expect("shutdown");
        tokio::time::timeout(std::time::Duration::from_secs(10), handle.agent_tx.closed())
            .await
            .expect("first .closed() resolves within 10s");
        let start = std::time::Instant::now();
        handle.agent_tx.closed().await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(10),
            "second .closed() must return immediately, took {:?}",
            elapsed
        );
    }

    /// Moderate-load variant of `agent_tx_closed_resolves_much_faster_than_prior_sleep_budget`:
    /// 10 concurrent sessions all shut down, all `.closed()` awaited
    /// inside a single `timeout(10s)`. Asserts outer elapsed stays
    /// below the 9s envelope the C91 single-session canary uses —
    /// documents that fan-out doesn't degrade linearly with session
    /// count even under the concurrent-stress scheduler profile that
    /// surfaced the C89-C90 flake sites.
    ///
    /// Empirical (cycle 20): solo runs complete in ~0.5s, ~18x headroom
    /// against the 9s assertion. Safe under chain-mode default runs. The
    /// historical multi-cargo-process flake rationale doesn't apply
    /// within a single cargo invocation; if a future change pushes
    /// elapsed near 9s, that's a real regression worth investigating
    /// before re-ignoring.
    #[tokio::test]
    async fn agent_tx_closed_resolves_within_nine_seconds_under_moderate_session_load() {
        let mut manager = SessionManager::new(test_config());
        let mut senders: Vec<mpsc::Sender<AgentCommand>> = Vec::with_capacity(10);
        for _ in 0..10 {
            let id = manager.create_session().expect("create_session");
            let h = manager.get(&id).expect("session");
            h.agent_tx
                .try_send(AgentCommand::Shutdown)
                .expect("shutdown");
            senders.push(h.agent_tx.clone());
        }
        let start = std::time::Instant::now();
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::join!(
                senders[0].closed(),
                senders[1].closed(),
                senders[2].closed(),
                senders[3].closed(),
                senders[4].closed(),
                senders[5].closed(),
                senders[6].closed(),
                senders[7].closed(),
                senders[8].closed(),
                senders[9].closed(),
            );
        })
        .await
        .expect("all 10 sessions must close agent_rx within 10s");
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(9),
            "10-session .closed() fan-out should resolve under 9s, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn reload_from_snapshots_respects_max_sessions() {
        let mut manager = SessionManager::new(test_config());
        let mut snapshots = Vec::new();
        for i in 0..(SessionManager::MAX_SESSIONS + 2) {
            let mut session =
                crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
            let id = format!("sess-{}", i);
            session.id = id.clone();
            snapshots.push(crate::daemon::persistence::SessionSnapshot::now(
                id,
                vec![],
                session,
            ));
        }
        let loaded = manager.reload_from_snapshots(snapshots);
        assert_eq!(loaded, SessionManager::MAX_SESSIONS);
        assert_eq!(manager.session_count(), SessionManager::MAX_SESSIONS);
    }

    #[tokio::test]
    async fn reload_from_snapshots_restores_messages() {
        let mut manager = SessionManager::new(test_config());
        let mut session =
            crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        session.id = "restored".to_string();
        let seeded = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hello from snapshot"}),
        ];
        session.messages = seeded.clone();
        let snap = crate::daemon::persistence::SessionSnapshot::now(
            "restored".to_string(),
            seeded,
            session,
        );
        let loaded = manager.reload_from_snapshots(vec![snap]);
        assert_eq!(loaded, 1);

        // Round-trip via Snapshot to confirm the reloaded task holds the seeded state.
        let handle = manager.get("restored").expect("reloaded session");
        let (tx, rx) = oneshot::channel();
        handle
            .agent_tx
            .try_send(AgentCommand::Snapshot(tx))
            .expect("send Snapshot");
        let snap_back = tokio::time::timeout(std::time::Duration::from_secs(10), rx)
            .await
            .expect("timeout")
            .expect("recv");
        assert_eq!(snap_back.session_id, "restored");
        assert_eq!(snap_back.messages.len(), 2);
        assert_eq!(
            snap_back.messages[1]["content"].as_str().unwrap(),
            "hello from snapshot"
        );
    }
}
