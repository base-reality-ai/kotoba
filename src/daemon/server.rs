use crate::config::Config;
use crate::daemon::client::{daemon_pid_path, daemon_socket_path};
use crate::daemon::protocol::{ChainInfo, DaemonEvent, DaemonRequest};
use crate::daemon::session_manager::SessionManager;
use anyhow::Context;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Tracks a chain running inside the daemon.
struct DaemonChainHandle {
    chain_id: String,
    name: String,
    node_count: usize,
    workspace: std::path::PathBuf,
    handle: JoinHandle<()>,
    event_tx: tokio::sync::broadcast::Sender<DaemonEvent>,
}

/// Thread-safe map of active chains tracked by the daemon.
type ChainRegistry = Arc<Mutex<HashMap<String, DaemonChainHandle>>>;

// ── Daemon log helpers ────────────────────────────────────────────────────────

/// Returns the path to the daemon log file: `<config_dir>/daemon.log`.
pub fn daemon_log_path(config_dir: &std::path::Path) -> std::path::PathBuf {
    config_dir.join("daemon.log")
}

/// Appends one JSON-line log entry to the daemon log. Silently ignores all I/O errors.
pub fn write_log_entry(
    config_dir: &std::path::Path,
    level: &str,
    msg: &str,
    extra: Option<serde_json::Value>,
) {
    let ts = chrono::Utc::now().to_rfc3339();
    let pid = std::process::id();
    let mut obj = serde_json::json!({
        "ts": ts,
        "level": level,
        "msg": msg,
        "pid": pid,
    });
    if let Some(ext) = extra {
        if let (Some(map), Some(ext_map)) = (obj.as_object_mut(), ext.as_object()) {
            for (k, v) in ext_map {
                map.insert(k.clone(), v.clone());
            }
        }
    }
    let Ok(mut line) = serde_json::to_string(&obj) else {
        return;
    };
    line.push('\n');
    let path = daemon_log_path(config_dir);
    // Ensure the parent directory exists; ignore errors silently.
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Reads the daemon log file and returns the last `n` lines.
/// Returns `Ok(vec![])` if the file does not exist.
pub fn tail_log(config_dir: &std::path::Path, n: usize) -> anyhow::Result<Vec<String>> {
    let path = daemon_log_path(config_dir);
    if !path.exists() {
        return Ok(vec![]);
    }
    let content = std::fs::read_to_string(&path).context("read daemon log")?;
    let lines: Vec<String> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect();
    let skip = lines.len().saturating_sub(n);
    Ok(lines[skip..].to_vec())
}

// ── Daemon server ─────────────────────────────────────────────────────────────

pub async fn run_daemon(config: Arc<Config>) -> anyhow::Result<()> {
    let socket_path = daemon_socket_path();
    let pid_path = daemon_pid_path();

    // Ensure parent directory exists
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent).context("create socket parent dir")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
        }
    }

    // Remove stale socket if it exists
    if socket_path.exists() {
        std::fs::remove_file(&socket_path).context("remove stale socket")?;
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind Unix socket at {}", socket_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600));
    }

    // Write PID file
    std::fs::write(&pid_path, std::process::id().to_string()).context("write PID file")?;

    write_log_entry(
        &config.config_dir,
        "info",
        "daemon started",
        Some(
            serde_json::json!({ "socket": socket_path.to_string_lossy(), "pid": std::process::id() }),
        ),
    );

    let session_manager = Arc::new(Mutex::new(SessionManager::new(Arc::clone(&config))));
    let chain_registry: ChainRegistry = Arc::new(Mutex::new(HashMap::new()));
    let daemon_start = std::time::Instant::now();
    let config = Arc::clone(&config);

    // Shutdown channel
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let shutdown_tx = Arc::new(shutdown_tx);

    // Signal bridge: SIGTERM/SIGINT → shutdown_tx, so `kill -TERM $pid` and
    // Ctrl+C both converge on the same graceful-exit arm as the daemon.shutdown
    // IPC. Without this, `dm daemon stop` (which sends SIGTERM) would kill the
    // daemon mid-select and skip session persistence.
    {
        let sd_tx = Arc::clone(&shutdown_tx);
        let cfg_for_sig = Arc::clone(&config);
        tokio::spawn(async move {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                let mut term = match signal(SignalKind::terminate()) {
                    Ok(s) => s,
                    Err(e) => {
                        write_log_entry(
                            &cfg_for_sig.config_dir,
                            "error",
                            &format!("install SIGTERM failed: {}", e),
                            None,
                        );
                        return;
                    }
                };
                let mut int = match signal(SignalKind::interrupt()) {
                    Ok(s) => s,
                    Err(e) => {
                        write_log_entry(
                            &cfg_for_sig.config_dir,
                            "error",
                            &format!("install SIGINT failed: {}", e),
                            None,
                        );
                        return;
                    }
                };
                tokio::select! {
                    _ = term.recv() => {
                        write_log_entry(&cfg_for_sig.config_dir, "info", "received SIGTERM", None);
                    }
                    _ = int.recv() => {
                        write_log_entry(&cfg_for_sig.config_dir, "info", "received SIGINT", None);
                    }
                }
            }
            #[cfg(not(unix))]
            {
                let _ = tokio::signal::ctrl_c().await;
                write_log_entry(&cfg_for_sig.config_dir, "info", "received Ctrl+C", None);
            }
            let _ = sd_tx.send(true);
        });
    }

    // Startup reload: restore any sessions persisted by a prior shutdown. The
    // listener is already bound, so clients connecting during this block queue
    // at the kernel level; we reload before the first `accept()`.
    {
        let persist_dir = crate::daemon::persistence::sessions_persist_dir(&config.config_dir);
        let snapshots = crate::daemon::persistence::load_persisted_sessions(&persist_dir);
        let on_disk = snapshots.len();
        let loaded = {
            let mut sm = session_manager.lock().await;
            sm.reload_from_snapshots(snapshots)
        };
        if on_disk > 0 || loaded > 0 {
            write_log_entry(
                &config.config_dir,
                "info",
                &format!(
                    "reloaded {} session(s) from {} snapshot(s)",
                    loaded, on_disk
                ),
                Some(serde_json::json!({"loaded": loaded, "on_disk": on_disk})),
            );
        }
    }

    // Scheduler tick: check for due scheduled tasks every 60 seconds
    let mut scheduler_tick = tokio::time::interval(std::time::Duration::from_secs(60));
    // Don't fire immediately on startup — wait one full interval first
    scheduler_tick.tick().await;

    // Periodic session snapshot timer. 0 disables entirely; the select arm
    // uses `std::future::pending()` in that case so it never fires.
    // `MissedTickBehavior::Delay` keeps a paused daemon from firing a burst of
    // catch-up snapshots when it wakes.
    let snapshot_interval_secs = config.snapshot_interval_secs;
    let mut snapshot_tick: Option<tokio::time::Interval> = if snapshot_interval_secs > 0 {
        let mut iv = tokio::time::interval(std::time::Duration::from_secs(snapshot_interval_secs));
        iv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        iv.tick().await; // skip immediate fire
        Some(iv)
    } else {
        write_log_entry(
            &config.config_dir,
            "info",
            "periodic session snapshot disabled (snapshot_interval_secs=0)",
            None,
        );
        None
    };

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((stream, _)) => {
                        let sm = Arc::clone(&session_manager);
                        let cr = Arc::clone(&chain_registry);
                        let cfg = Arc::clone(&config);
                        let sd_tx = Arc::clone(&shutdown_tx);
                        let start = daemon_start;
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, sm, cr, Arc::clone(&cfg), sd_tx, start).await {
                                write_log_entry(
                                    &cfg.config_dir,
                                    "error",
                                    &format!("connection error: {}", e),
                                    None,
                                );
                            }
                        });
                    }
                    Err(e) => {
                        write_log_entry(
                            &config.config_dir,
                            "error",
                            &format!("accept error: {}", e),
                            None,
                        );
                    }
                }
            }
            _ = scheduler_tick.tick() => {
                let cfg = Arc::clone(&config);
                tokio::spawn(async move {
                    run_scheduler_tick(&cfg).await;
                });
                // Session GC: sweep sessions idle past `idle_timeout_secs`
                // (time since last user turn, not session age). 0 disables
                // eviction. Per-session detail is logged first so operators
                // can correlate evictions with specific session IDs; the
                // aggregate summary line preserves legacy log-scraping shape.
                let sm = Arc::clone(&session_manager);
                let mut sm_lock = sm.lock().await;
                let evictions = sm_lock.gc_sweep(config.idle_timeout_secs);
                let remaining = sm_lock.session_count();
                drop(sm_lock);
                for (id, idle) in &evictions {
                    write_log_entry(
                        &config.config_dir,
                        "info",
                        &format!("gc: evicted session {} idle {}s", id, idle.as_secs()),
                        Some(serde_json::json!({
                            "session_id": id,
                            "idle_secs": idle.as_secs(),
                            "event": "session_evicted",
                        })),
                    );
                }
                if !evictions.is_empty() {
                    write_log_entry(
                        &config.config_dir,
                        "info",
                        &format!(
                            "gc: removed {} idle session(s), {} remaining",
                            evictions.len(),
                            remaining
                        ),
                        None,
                    );
                }
            }
            _ = async {
                match snapshot_tick.as_mut() {
                    Some(iv) => { iv.tick().await; }
                    None => { std::future::pending::<()>().await; }
                }
            } => {
                let _ = crate::daemon::persistence::snapshot_and_persist_all(
                    &session_manager,
                    &crate::daemon::persistence::sessions_persist_dir(&config.config_dir),
                    &config.config_dir,
                    std::time::Duration::from_secs(5),
                    "periodic",
                )
                .await;
            }
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    write_log_entry(&config.config_dir, "info", "daemon shutting down", None);
                    break;
                }
            }
        }
    }

    // Graceful shutdown: delegate to the shared helper so the shutdown path and
    // the periodic timer path use identical snapshot/clear/persist semantics.
    let _ = crate::daemon::persistence::snapshot_and_persist_all(
        &session_manager,
        &crate::daemon::persistence::sessions_persist_dir(&config.config_dir),
        &config.config_dir,
        std::time::Duration::from_secs(5),
        "shutdown",
    )
    .await;

    // Cleanup
    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_file(&pid_path);

    Ok(())
}

/// Check for due scheduled tasks, execute them, and advance their `next_run`.
async fn run_scheduler_tick(config: &Config) {
    use crate::daemon::scheduler::{advance_next_run, load_schedules, next_due, save_schedules};

    let mut tasks = match load_schedules(&config.config_dir) {
        Ok(t) => t,
        Err(e) => {
            write_log_entry(
                &config.config_dir,
                "warn",
                &format!("scheduler: failed to load schedules: {}", e),
                None,
            );
            return;
        }
    };

    if tasks.is_empty() {
        return;
    }

    // Find indices of due tasks (need indices to mutate in-place later)
    let due_indices: Vec<usize> = {
        let due_set: std::collections::HashSet<&str> =
            next_due(&tasks).iter().map(|t| t.id.as_str()).collect();
        tasks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if due_set.contains(t.id.as_str()) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    };

    if due_indices.is_empty() {
        return;
    }

    write_log_entry(
        &config.config_dir,
        "info",
        &format!("scheduler: {} task(s) due", due_indices.len()),
        None,
    );

    for &idx in &due_indices {
        let task = &tasks[idx];
        let prompt = task.prompt.clone();
        let task_id = task.id.clone();
        let model = task.model.clone().unwrap_or_else(|| config.model.clone());
        let base_url = config.ollama_base_url();
        let config_dir = config.config_dir.clone();

        // Spawn each task execution as an independent background job
        tokio::spawn(async move {
            let client = crate::ollama::client::OllamaClient::new(base_url.clone(), model);
            let session_id = uuid::Uuid::new_v4().to_string();
            let registry = crate::tools::registry::default_registry(
                &session_id,
                &config_dir,
                &base_url,
                client.model(),
                "",
            );

            write_log_entry(
                &config_dir,
                "info",
                &format!("scheduler: running task '{}'", task_id),
                Some(serde_json::json!({ "task_id": &task_id, "prompt": &prompt })),
            );

            match crate::conversation::run_conversation_capture(
                &prompt, "daemon", &client, &registry,
            )
            .await
            {
                Ok(result) => {
                    write_log_entry(
                        &config_dir,
                        "info",
                        &format!(
                            "scheduler: task '{}' completed ({} chars output)",
                            task_id,
                            result.text.len()
                        ),
                        Some(serde_json::json!({ "task_id": &task_id })),
                    );
                    crate::notify::notify(
                        "Scheduled task complete",
                        &format!("Task '{}': {} chars output", task_id, result.text.len()),
                    );
                }
                Err(e) => {
                    write_log_entry(
                        &config_dir,
                        "error",
                        &format!("scheduler: task '{}' failed: {}", task_id, e),
                        Some(serde_json::json!({ "task_id": &task_id })),
                    );
                    crate::notify::notify(
                        "Scheduled task failed",
                        &format!("Task '{}': {}", task_id, e),
                    );
                }
            }
        });

        // Advance the task's schedule
        advance_next_run(&mut tasks[idx]);
    }

    // Persist updated schedules
    if let Err(e) = save_schedules(&config.config_dir, &tasks) {
        write_log_entry(
            &config.config_dir,
            "error",
            &format!("scheduler: failed to save schedules: {}", e),
            None,
        );
    }
}

async fn handle_connection(
    stream: tokio::net::UnixStream,
    session_manager: Arc<Mutex<SessionManager>>,
    chain_registry: ChainRegistry,
    config: Arc<Config>,
    shutdown_tx: Arc<tokio::sync::watch::Sender<bool>>,
    daemon_start: std::time::Instant,
) -> anyhow::Result<()> {
    let (read_half, write_half) = stream.into_split();
    let write_half = Arc::new(Mutex::new(write_half));
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await.context("read line")?;
        if n == 0 {
            write_log_entry(&config.config_dir, "info", "client disconnected", None);
            break; // Client disconnected
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: DaemonRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let event = DaemonEvent::Error {
                    session_id: String::new(),
                    message: format!(
                        "invalid request: {}. Try: confirm the request body is valid JSON conforming to DaemonRequest.",
                        e
                    ),
                };
                send_event(&write_half, &event).await?;
                continue;
            }
        };

        match req.method.as_str() {
            "session.create" => {
                let result = {
                    let mut sm = session_manager.lock().await;
                    sm.create_session()
                };
                match result {
                    Ok(session_id) => {
                        write_log_entry(
                            &config.config_dir,
                            "info",
                            "session created",
                            Some(serde_json::json!({ "session_id": session_id })),
                        );
                        let event = DaemonEvent::SessionCreated { session_id };
                        send_event(&write_half, &event).await?;
                    }
                    Err(msg) => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: msg,
                        };
                        send_event(&write_half, &event).await?;
                    }
                }
            }

            "session.list" => {
                let sessions = {
                    let sm = session_manager.lock().await;
                    sm.list()
                };
                let event = DaemonEvent::SessionList { sessions };
                send_event(&write_half, &event).await?;
            }

            "session.attach" => {
                let session_id = req
                    .params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let rx = {
                    let sm = session_manager.lock().await;
                    sm.get(&session_id).map(|h| {
                        h.client_count
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        h.event_tx.subscribe()
                    })
                };
                match rx {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id,
                            message: "session not found. Try: session.list to see active sessions, or session.create to start a new one.".to_string(),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(mut rx) => {
                        let wh = Arc::clone(&write_half);
                        let sm = Arc::clone(&session_manager);
                        let sid = session_id.clone();
                        let cfg = Arc::clone(&config);
                        let mut cancel = cancel_rx.clone();
                        tokio::spawn(async move {
                            loop {
                                tokio::select! {
                                    event = rx.recv() => {
                                        match event {
                                            Ok(event) => {
                                                if send_event(&wh, &event).await.is_err() {
                                                    break;
                                                }
                                            }
                                            Err(_) => break,
                                        }
                                    }
                                    _ = cancel.changed() => break,
                                }
                            }
                            let sm = sm.lock().await;
                            if let Some(h) = sm.get(&sid) {
                                h.client_count
                                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            write_log_entry(
                                &cfg.config_dir,
                                "info",
                                "client detached from session",
                                Some(serde_json::json!({ "session_id": sid })),
                            );
                        });
                    }
                }
            }

            "session.end" => {
                let session_id = req
                    .params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut sm = session_manager.lock().await;
                sm.remove(&session_id);
                write_log_entry(
                    &config.config_dir,
                    "info",
                    "session ended",
                    Some(serde_json::json!({ "session_id": session_id })),
                );
                let event = DaemonEvent::SessionEnded { session_id };
                send_event(&write_half, &event).await?;
            }

            "session.cancel" => {
                let session_id = req
                    .params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let sm = session_manager.lock().await;
                match sm.get(&session_id) {
                    Some(handle) => {
                        let _ = handle
                            .agent_tx
                            .try_send(crate::daemon::session_manager::AgentCommand::Cancel);
                        write_log_entry(
                            &config.config_dir,
                            "info",
                            "session cancel requested",
                            Some(serde_json::json!({ "session_id": session_id })),
                        );
                        let event = DaemonEvent::SessionCancelled { session_id };
                        send_event(&write_half, &event).await?;
                    }
                    None => {
                        let event = DaemonEvent::Error {
                            session_id,
                            message: "session not found. Try: session.list to see active sessions, or session.create to start a new one.".to_string(),
                        };
                        send_event(&write_half, &event).await?;
                    }
                }
            }

            "session.shutdown" => {
                // Gracefully shut down a session's agent task then remove it.
                let session_id = req
                    .params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                {
                    let sm = session_manager.lock().await;
                    if let Some(handle) = sm.get(&session_id) {
                        let _ = handle
                            .agent_tx
                            .try_send(crate::daemon::session_manager::AgentCommand::Shutdown);
                    }
                }
                // Give the agent task a moment to clean up, then remove.
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                {
                    let mut sm = session_manager.lock().await;
                    sm.remove(&session_id);
                }
                write_log_entry(
                    &config.config_dir,
                    "info",
                    "session shutdown",
                    Some(serde_json::json!({ "session_id": session_id })),
                );
                let event = DaemonEvent::SessionEnded { session_id };
                send_event(&write_half, &event).await?;
            }

            "turn.send" => {
                let session_id = req
                    .params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = req
                    .params
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let rx_result = {
                    let sm = session_manager.lock().await;
                    sm.handle_turn_send(&session_id, text)
                };
                match rx_result {
                    Err(e) => {
                        let event = DaemonEvent::Error {
                            session_id,
                            message: format!(
                                "turn.send failed: {}. Try: confirm the session is alive with session.list, and the request payload is correct.",
                                e
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Ok(mut rx) => {
                        let wh = Arc::clone(&write_half);
                        let mut cancel = cancel_rx.clone();
                        tokio::spawn(async move {
                            loop {
                                tokio::select! {
                                    event = rx.recv() => {
                                        match event {
                                            Ok(event) => {
                                                if send_event(&wh, &event).await.is_err() {
                                                    break;
                                                }
                                                match &event {
                                                    DaemonEvent::TurnComplete { .. }
                                                    | DaemonEvent::Cancelled { .. }
                                                    | DaemonEvent::Error { .. } => break,
                                                    _ => {}
                                                }
                                            }
                                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                                            Err(_) => break,
                                        }
                                    }
                                    _ = cancel.changed() => break,
                                }
                            }
                        });
                    }
                }
            }

            "daemon.shutdown" => {
                write_log_entry(&config.config_dir, "info", "shutdown requested", None);
                let _ = shutdown_tx.send(true);
                break;
            }

            "chain.start" => {
                let config_path = req
                    .params
                    .get("config_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let path = std::path::Path::new(&config_path);
                match crate::orchestrate::load_chain_config(path) {
                    Err(e) => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain.start: failed to load config: {}. Try: confirm 'config_path' points to a valid YAML file.",
                                e
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Ok(mut chain_config) => {
                        chain_config.resolve_aliases(&config.aliases);
                        if let Err(e) = crate::orchestrate::validate_chain_config(&chain_config) {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "chain.start: invalid config: {}. Try: chain.validate with the same 'path' to see structural errors before starting.",
                                    e
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        } else {
                            let chain_id = format!("chain-{}", uuid::Uuid::new_v4().as_simple());
                            let name = chain_config.name.clone();
                            let node_count = chain_config.nodes.len();
                            let workspace = chain_config.workspace.clone();

                            let orch_config = crate::orchestrate::OrchestrationConfig {
                                chain: chain_config,
                                chain_id: chain_id.clone(),
                                retry: crate::conversation::RetrySettings::from_config(&config),
                                resume_state: None,
                            };
                            let base_url = config.ollama_base_url();
                            let client = crate::ollama::client::OllamaClient::new(
                                base_url.clone(),
                                config.model.clone(),
                            );
                            let session_id = uuid::Uuid::new_v4().to_string();
                            let registry = crate::tools::registry::default_registry(
                                &session_id,
                                &config.config_dir,
                                &base_url,
                                &config.model,
                                "",
                            );

                            let (chain_event_tx, _) = tokio::sync::broadcast::channel(64);
                            let runner_tx = chain_event_tx.clone();
                            let cid = chain_id.clone();
                            let cr = Arc::clone(&chain_registry);
                            let log_dir = config.config_dir.clone();
                            let handle = tokio::spawn(async move {
                                if let Err(e) = crate::orchestrate::runner::run_orchestration(
                                    orch_config,
                                    client,
                                    registry,
                                    Some(runner_tx),
                                )
                                .await
                                {
                                    write_log_entry(
                                        &log_dir,
                                        "error",
                                        &format!("chain '{}' error: {}", cid, e),
                                        None,
                                    );
                                }
                                // Remove from registry when done
                                let mut reg = cr.lock().await;
                                reg.remove(&cid);
                            });

                            {
                                let mut reg = chain_registry.lock().await;
                                reg.insert(
                                    chain_id.clone(),
                                    DaemonChainHandle {
                                        chain_id: chain_id.clone(),
                                        name: name.clone(),
                                        node_count,
                                        workspace,
                                        handle,
                                        event_tx: chain_event_tx,
                                    },
                                );
                            }

                            write_log_entry(
                                &config.config_dir,
                                "info",
                                "chain started",
                                Some(serde_json::json!({
                                    "chain_id": chain_id,
                                    "name": name,
                                })),
                            );

                            let event = DaemonEvent::ChainStarted {
                                chain_id,
                                name,
                                node_count,
                            };
                            send_event(&write_half, &event).await?;
                        }
                    }
                }
            }

            "chain.status" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let reg = chain_registry.lock().await;
                match reg.get(&chain_id) {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(ch) => {
                        let state_path = ch.workspace.join("chain_state.json");
                        let state_json = match std::fs::read_to_string(&state_path) {
                            Ok(content) => {
                                serde_json::from_str(&content).unwrap_or(serde_json::Value::Null)
                            }
                            Err(_) => serde_json::Value::Null,
                        };
                        let event = DaemonEvent::ChainStatus {
                            chain_id,
                            state: state_json,
                        };
                        send_event(&write_half, &event).await?;
                    }
                }
            }

            "chain.stop" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let found = {
                    let reg = chain_registry.lock().await;
                    match reg.get(&chain_id) {
                        None => false,
                        Some(ch) => {
                            let stop_file = ch.workspace.join(".dm-stop");
                            let _ = std::fs::write(&stop_file, b"");
                            ch.handle.abort();
                            true
                        }
                    }
                };
                if found {
                    let mut reg = chain_registry.lock().await;
                    reg.remove(&chain_id);
                    write_log_entry(
                        &config.config_dir,
                        "info",
                        "chain stop requested",
                        Some(serde_json::json!({ "chain_id": chain_id })),
                    );
                    let event = DaemonEvent::ChainFinished {
                        chain_id,
                        success: false,
                        reason: "stop requested via daemon".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: format!(
                            "chain '{}' not found. Try: chain.list to see active chains.",
                            chain_id
                        ),
                    };
                    send_event(&write_half, &event).await?;
                }
            }

            "chain.list" => {
                let reg = chain_registry.lock().await;
                let chains: Vec<ChainInfo> = reg
                    .values()
                    .map(|ch| {
                        let status = if ch.handle.is_finished() {
                            "finished"
                        } else {
                            let pause_file = ch.workspace.join(".dm-pause");
                            if pause_file.exists() {
                                "paused"
                            } else {
                                "running"
                            }
                        };
                        // Read current cycle from state file if available
                        let state_path = ch.workspace.join("chain_state.json");
                        let current_cycle = std::fs::read_to_string(&state_path)
                            .ok()
                            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                            .and_then(|v| v.get("current_cycle")?.as_u64())
                            .unwrap_or(0) as usize;
                        ChainInfo {
                            chain_id: ch.chain_id.clone(),
                            name: ch.name.clone(),
                            current_cycle,
                            node_count: ch.node_count,
                            status: status.to_string(),
                        }
                    })
                    .collect();
                let event = DaemonEvent::ChainList { chains };
                send_event(&write_half, &event).await?;
            }

            "chain.attach" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let rx = {
                    let reg = chain_registry.lock().await;
                    reg.get(&chain_id).map(|ch| ch.event_tx.subscribe())
                };
                match rx {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(mut rx) => {
                        let wh = Arc::clone(&write_half);
                        let cfg = Arc::clone(&config);
                        let cid = chain_id.clone();
                        let mut cancel = cancel_rx.clone();
                        tokio::spawn(async move {
                            loop {
                                tokio::select! {
                                    event = rx.recv() => {
                                        match event {
                                            Ok(event) => {
                                                if send_event(&wh, &event).await.is_err() {
                                                    break;
                                                }
                                            }
                                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                                            Err(_) => break,
                                        }
                                    }
                                    _ = cancel.changed() => break,
                                }
                            }
                            write_log_entry(
                                &cfg.config_dir,
                                "info",
                                "client detached from chain",
                                Some(serde_json::json!({ "chain_id": cid })),
                            );
                        });
                    }
                }
            }

            "chain.pause" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let reg = chain_registry.lock().await;
                match reg.get(&chain_id) {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(ch) => {
                        let _ = std::fs::write(ch.workspace.join(".dm-pause"), b"");
                        write_log_entry(
                            &config.config_dir,
                            "info",
                            "chain paused",
                            Some(serde_json::json!({"chain_id": chain_id})),
                        );
                        let event = DaemonEvent::ChainStatus {
                            chain_id,
                            state: serde_json::json!({"paused": true}),
                        };
                        send_event(&write_half, &event).await?;
                    }
                }
            }

            "chain.resume" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let reg = chain_registry.lock().await;
                match reg.get(&chain_id) {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(ch) => {
                        let pause_file = ch.workspace.join(".dm-pause");
                        if pause_file.exists() {
                            let _ = std::fs::remove_file(&pause_file);
                        }
                        write_log_entry(
                            &config.config_dir,
                            "info",
                            "chain resumed",
                            Some(serde_json::json!({"chain_id": chain_id})),
                        );
                        let event = DaemonEvent::ChainStatus {
                            chain_id,
                            state: serde_json::json!({"paused": false}),
                        };
                        send_event(&write_half, &event).await?;
                    }
                }
            }

            "chain.talk" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let node = req
                    .params
                    .get("node")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let message = req
                    .params
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if node.is_empty() || message.is_empty() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: "node and message are required. Try: include both 'node' (target node name) and 'message' (text to deliver) in params.".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let reg = chain_registry.lock().await;
                    match reg.get(&chain_id) {
                        None => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "chain '{}' not found. Try: chain.list to see active chains.",
                                    chain_id
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Some(ch) => {
                            let talk_file = ch.workspace.join(format!("talk-{}.md", node));
                            let _ = std::fs::write(&talk_file, message.as_bytes());
                            write_log_entry(
                                &config.config_dir,
                                "info",
                                "chain talk injected",
                                Some(serde_json::json!({"chain_id": chain_id, "node": node})),
                            );
                            let event = DaemonEvent::ChainStatus {
                                chain_id,
                                state: serde_json::json!({"talk_injected": node}),
                            };
                            send_event(&write_half, &event).await?;
                        }
                    }
                }
            }

            "chain.add" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = req
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let model = req
                    .params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let role = req
                    .params
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&name)
                    .to_string();
                let input_from = req
                    .params
                    .get("input_from")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                if name.is_empty() || model.is_empty() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: "name and model are required. Try: include both 'name' (new node ID) and 'model' (model name) in params.".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let reg = chain_registry.lock().await;
                    match reg.get(&chain_id) {
                        None => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "chain '{}' not found. Try: chain.list to see active chains.",
                                    chain_id
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Some(ch) => {
                            let state_path = ch.workspace.join("chain_state.json");
                            match crate::orchestrate::types::ChainState::load(&state_path) {
                                Err(e) => {
                                    let event = DaemonEvent::Error {
                                        session_id: String::new(),
                                        message: format!("failed to load chain state: {}. Try: chain.start to begin a new run, or remove the corrupted state file.", e),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                                Ok(mut state) => {
                                    let node = crate::orchestrate::types::ChainNodeConfig {
                                        id: name.clone(),
                                        name: name.clone(),
                                        role,
                                        model,
                                        description: None,
                                        system_prompt_override: None,
                                        system_prompt_file: None,
                                        input_from,
                                        max_retries: 1,
                                        timeout_secs: 3600,
                                        max_tool_turns: 200,
                                    };
                                    state.pending_additions.push(node);
                                    let _ = state.save(&ch.workspace);
                                    write_log_entry(
                                        &config.config_dir,
                                        "info",
                                        "chain node add queued",
                                        Some(
                                            serde_json::json!({"chain_id": chain_id, "node": name}),
                                        ),
                                    );
                                    let event = DaemonEvent::ChainStatus {
                                        chain_id,
                                        state: serde_json::json!({"node_added": name}),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                            }
                        }
                    }
                }
            }

            "chain.remove" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let node = req
                    .params
                    .get("node")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if node.is_empty() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: "node name is required. Try: include 'node' (the node ID to remove) in params.".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let reg = chain_registry.lock().await;
                    match reg.get(&chain_id) {
                        None => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "chain '{}' not found. Try: chain.list to see active chains.",
                                    chain_id
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Some(ch) => {
                            let state_path = ch.workspace.join("chain_state.json");
                            match crate::orchestrate::types::ChainState::load(&state_path) {
                                Err(e) => {
                                    let event = DaemonEvent::Error {
                                        session_id: String::new(),
                                        message: format!("failed to load chain state: {}. Try: chain.start to begin a new run, or remove the corrupted state file.", e),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                                Ok(mut state) => {
                                    state.pending_removals.push(node.clone());
                                    let _ = state.save(&ch.workspace);
                                    write_log_entry(
                                        &config.config_dir,
                                        "info",
                                        "chain node remove queued",
                                        Some(
                                            serde_json::json!({"chain_id": chain_id, "node": node}),
                                        ),
                                    );
                                    let event = DaemonEvent::ChainStatus {
                                        chain_id,
                                        state: serde_json::json!({"node_removed": node}),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                            }
                        }
                    }
                }
            }

            "chain.model" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let node = req
                    .params
                    .get("node")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let model = req
                    .params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if node.is_empty() || model.is_empty() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: "node and model are required. Try: include both 'node' (target node) and 'model' (new model) in params.".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let reg = chain_registry.lock().await;
                    match reg.get(&chain_id) {
                        None => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "chain '{}' not found. Try: chain.list to see active chains.",
                                    chain_id
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Some(ch) => {
                            let state_path = ch.workspace.join("chain_state.json");
                            match crate::orchestrate::types::ChainState::load(&state_path) {
                                Err(e) => {
                                    let event = DaemonEvent::Error {
                                        session_id: String::new(),
                                        message: format!("failed to load chain state: {}. Try: chain.start to begin a new run, or remove the corrupted state file.", e),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                                Ok(mut state) => {
                                    state
                                        .pending_model_swaps
                                        .insert(node.clone(), model.clone());
                                    let _ = state.save(&ch.workspace);
                                    write_log_entry(
                                        &config.config_dir,
                                        "info",
                                        "chain model swap queued",
                                        Some(
                                            serde_json::json!({"chain_id": chain_id, "node": node, "model": model}),
                                        ),
                                    );
                                    let event = DaemonEvent::ChainStatus {
                                        chain_id,
                                        state: serde_json::json!({"model_swapped": {"node": node, "model": model}}),
                                    };
                                    send_event(&write_half, &event).await?;
                                }
                            }
                        }
                    }
                }
            }

            "chain.log" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let cycle = req
                    .params
                    .get("cycle")
                    .and_then(|v| v.as_u64())
                    .map(|c| c as usize);
                let reg = chain_registry.lock().await;
                match reg.get(&chain_id) {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(ch) => {
                        match crate::orchestrate::chain_log_from_workspace(&ch.workspace, cycle) {
                            Ok(entries) => {
                                let mut output = String::new();
                                if entries.is_empty() {
                                    output.push_str("No chain artifacts found.");
                                } else {
                                    for entry in &entries {
                                        write!(
                                            output,
                                            "\n--- {} ---\n{}\n",
                                            entry.filename, entry.content
                                        )
                                        .expect("write to String never fails");
                                    }
                                }
                                let event = DaemonEvent::ChainLog {
                                    chain_id,
                                    level: "info".into(),
                                    message: output,
                                };
                                send_event(&write_half, &event).await?;
                            }
                            Err(e) => {
                                let event = DaemonEvent::Error {
                                    session_id: String::new(),
                                    message: format!(
                                        "chain.log failed: {}. Try: confirm the chain has produced cycle artifacts; pass an explicit 'cycle' to narrow.",
                                        e
                                    ),
                                };
                                send_event(&write_half, &event).await?;
                            }
                        }
                    }
                }
            }

            "chain.validate" => {
                let path_str = req
                    .params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if path_str.is_empty() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: "path is required. Try: include 'path' (chain config file path) in params.".to_string(),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let path = std::path::Path::new(&path_str);
                    match crate::orchestrate::load_chain_config(path) {
                        Err(e) => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                    "Failed to load config: {}. Try: confirm the YAML is well-formed and the file path is correct.",
                                    e
                                ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Ok(chain_config) => {
                            let base_url = config.ollama_base_url();
                            let client = crate::ollama::client::OllamaClient::new(
                                base_url,
                                config.model.clone(),
                            );
                            let models: Vec<String> = client
                                .list_models()
                                .await
                                .map(|ms| ms.into_iter().map(|m| m.name).collect())
                                .unwrap_or_default();
                            let validation = crate::orchestrate::validate_chain_config_detailed(
                                &chain_config,
                                &models,
                            );
                            let event = DaemonEvent::ChainLog {
                                chain_id: String::new(),
                                level: if validation.errors.is_empty() {
                                    "info".into()
                                } else {
                                    "error".into()
                                },
                                message: validation.report,
                            };
                            send_event(&write_half, &event).await?;
                        }
                    }
                }
            }

            "chain.metrics" => {
                let chain_id = req
                    .params
                    .get("chain_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let reg = chain_registry.lock().await;
                match reg.get(&chain_id) {
                    None => {
                        let event = DaemonEvent::Error {
                            session_id: String::new(),
                            message: format!(
                                "chain '{}' not found. Try: chain.list to see active chains.",
                                chain_id
                            ),
                        };
                        send_event(&write_half, &event).await?;
                    }
                    Some(ch) => {
                        let state_path = ch.workspace.join("chain_state.json");
                        match crate::orchestrate::types::ChainState::load(&state_path) {
                            Ok(state) => {
                                let metrics = crate::orchestrate::format_chain_metrics(&state);
                                let event = DaemonEvent::ChainLog {
                                    chain_id,
                                    level: "info".into(),
                                    message: metrics,
                                };
                                send_event(&write_half, &event).await?;
                            }
                            Err(e) => {
                                let event = DaemonEvent::Error {
                                    session_id: String::new(),
                                    message: format!(
                                        "Failed to load chain state: {}. Try: chain.start to begin a new run, or remove the corrupted state file.",
                                        e
                                    ),
                                };
                                send_event(&write_half, &event).await?;
                            }
                        }
                    }
                }
            }

            "chain.init" => {
                let name = req
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("my-chain")
                    .to_string();
                let chains_dir = config.config_dir.join("chains");
                let _ = std::fs::create_dir_all(&chains_dir);
                let filename = format!("{}.chain.yaml", name);
                let path = chains_dir.join(&filename);
                if path.exists() {
                    let event = DaemonEvent::Error {
                        session_id: String::new(),
                        message: format!(
                            "File '{}' already exists. Try: pick a different 'name' param, or remove the existing file first.",
                            path.display()
                        ),
                    };
                    send_event(&write_half, &event).await?;
                } else {
                    let default_model = config.model.clone();
                    let yaml = crate::orchestrate::generate_chain_template(&name, &default_model);
                    match std::fs::write(&path, &yaml) {
                        Ok(()) => {
                            write_log_entry(
                                &config.config_dir,
                                "info",
                                "chain config initialized",
                                Some(
                                    serde_json::json!({"name": name, "path": path.display().to_string()}),
                                ),
                            );
                            let event = DaemonEvent::ChainLog {
                                chain_id: String::new(),
                                level: "info".into(),
                                message: format!("Created chain config: {}\n\nEdit it, then start with chain.start", path.display()),
                            };
                            send_event(&write_half, &event).await?;
                        }
                        Err(e) => {
                            let event = DaemonEvent::Error {
                                session_id: String::new(),
                                message: format!(
                                "Failed to write {}: {}. Try: confirm ~/.dm/chains/ is writable.",
                                path.display(),
                                e
                            ),
                            };
                            send_event(&write_half, &event).await?;
                        }
                    }
                }
            }

            "ping" => {
                send_event(&write_half, &DaemonEvent::Pong).await?;
            }

            "daemon.health" => {
                let uptime_secs = daemon_start.elapsed().as_secs();
                let sm = session_manager.lock().await;
                let session_count = sm.session_count();
                drop(sm);
                let event = DaemonEvent::Health {
                    uptime_secs,
                    session_count,
                    pid: std::process::id(),
                };
                send_event(&write_half, &event).await?;
            }

            unknown => {
                let event = DaemonEvent::Error {
                    session_id: String::new(),
                    message: format!(
                        "unknown method: {}. Try: ping or daemon.health for health-checks; see daemon protocol for the full method list.",
                        unknown
                    ),
                };
                send_event(&write_half, &event).await?;
            }
        }
    }

    let _ = cancel_tx.send(true);
    Ok(())
}

async fn send_event(
    write_half: &Arc<Mutex<tokio::net::unix::OwnedWriteHalf>>,
    event: &DaemonEvent,
) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(event).context("serialize event")?;
    line.push('\n');
    let mut wh = write_half.lock().await;
    wh.write_all(line.as_bytes())
        .await
        .context("write event to socket")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn daemon_log_entry_is_valid_json() {
        let dir = TempDir::new().expect("create tempdir");
        write_log_entry(dir.path(), "info", "test message", None);
        let path = daemon_log_path(dir.path());
        let content = std::fs::read_to_string(&path).expect("read log file");
        let line = content.lines().next().expect("at least one line");
        let obj: serde_json::Value = serde_json::from_str(line).expect("parse JSON");
        assert_eq!(obj["level"], "info");
        assert_eq!(obj["msg"], "test message");
        assert!(obj["ts"].is_string());
        assert!(obj["pid"].is_number());
    }

    #[test]
    fn daemon_log_tail_returns_last_n() {
        let dir = TempDir::new().expect("create tempdir");
        let path = daemon_log_path(dir.path());
        // Write 10 lines manually.
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .expect("open file");
            for i in 0..10u32 {
                writeln!(f, r#"{{"n":{}}}"#, i).expect("write line");
            }
        }
        let lines = tail_log(dir.path(), 3).expect("tail_log");
        assert_eq!(lines.len(), 3, "should return exactly 3 lines");
        // The last 3 lines should be n=7, n=8, n=9
        for (i, line) in lines.iter().enumerate() {
            let obj: serde_json::Value = serde_json::from_str(line).expect("parse JSON");
            assert_eq!(obj["n"], 7 + i as u64);
        }
    }

    #[test]
    fn tail_log_returns_empty_when_file_missing() {
        let dir = TempDir::new().expect("create tempdir");
        let lines = tail_log(dir.path(), 10).expect("tail_log should not error");
        assert!(lines.is_empty());
    }

    #[test]
    fn write_log_entry_never_panics_on_bad_path() {
        // Writing to a path that cannot exist should not panic.
        write_log_entry(
            std::path::Path::new("/dev/null/this/cannot/exist"),
            "warn",
            "test",
            None,
        );
        // If we reach here, no panic occurred — test passes.
    }

    #[test]
    fn write_log_entry_merges_extra_fields() {
        let dir = TempDir::new().expect("create tempdir");
        let extra = serde_json::json!({"session_id": "abc123", "model": "gemma4:26b"});
        write_log_entry(dir.path(), "debug", "agent started", Some(extra));

        let path = daemon_log_path(dir.path());
        let content = std::fs::read_to_string(&path).expect("read log file");
        let line = content.lines().next().expect("at least one line");
        let obj: serde_json::Value = serde_json::from_str(line).expect("parse JSON");
        // Core fields
        assert_eq!(obj["level"], "debug");
        assert_eq!(obj["msg"], "agent started");
        // Merged extra fields
        assert_eq!(obj["session_id"], "abc123");
        assert_eq!(obj["model"], "gemma4:26b");
    }

    #[test]
    fn tail_log_n_larger_than_file_returns_all_lines() {
        let dir = TempDir::new().expect("create tempdir");
        write_log_entry(dir.path(), "info", "line1", None);
        write_log_entry(dir.path(), "info", "line2", None);
        // Requesting more lines than exist
        let lines = tail_log(dir.path(), 100).expect("tail_log");
        assert_eq!(
            lines.len(),
            2,
            "should return all 2 lines when n > file length"
        );
    }

    #[test]
    fn tail_log_skips_blank_lines() {
        let dir = TempDir::new().expect("create tempdir");
        let path = daemon_log_path(dir.path());
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .expect("open");
            writeln!(f, r#"{{"n":1}}"#).unwrap();
            writeln!(f).unwrap(); // blank line
            writeln!(f, r#"{{"n":2}}"#).unwrap();
            writeln!(f).unwrap(); // trailing blank
        }
        let lines = tail_log(dir.path(), 10).expect("tail_log");
        // Blank lines must be filtered out
        assert_eq!(
            lines.len(),
            2,
            "blank lines should not be counted: {:?}",
            lines
        );
    }

    #[test]
    fn tail_log_n_zero_returns_empty() {
        let dir = TempDir::new().expect("create tempdir");
        write_log_entry(dir.path(), "info", "some entry", None);
        let lines = tail_log(dir.path(), 0).expect("tail_log");
        assert!(lines.is_empty(), "n=0 should return empty slice");
    }

    #[test]
    fn daemon_log_path_structure() {
        let dir = std::path::Path::new("/tmp/dm_test_daemon");
        let p = daemon_log_path(dir);
        assert_eq!(p, dir.join("daemon.log"));
    }

    #[test]
    fn write_log_entry_non_object_extra_does_not_panic() {
        let dir = TempDir::new().expect("create tempdir");
        // Pass a non-object JSON value as extra; the merge branch should be skipped gracefully.
        write_log_entry(
            dir.path(),
            "warn",
            "extra is not an object",
            Some(serde_json::json!([1, 2, 3])),
        );
        let path = daemon_log_path(dir.path());
        let content = std::fs::read_to_string(&path).expect("read log");
        let obj: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("one line")).expect("valid JSON");
        // Core fields still present; array extra is silently ignored
        assert_eq!(obj["msg"], "extra is not an object");
    }

    fn test_config(dir: &std::path::Path) -> Config {
        Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "test".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: dir.to_path_buf(),
            routing: None,
            aliases: std::collections::HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        }
    }

    #[tokio::test]
    async fn scheduler_tick_no_schedules_is_noop() {
        let dir = TempDir::new().expect("create tempdir");
        let config = test_config(dir.path());
        // No schedules.json exists — tick should return without error
        run_scheduler_tick(&config).await;
        // No log entry about task execution expected
        let lines = tail_log(dir.path(), 100).expect("tail_log");
        assert!(
            !lines.iter().any(|l| l.contains("running task")),
            "no tasks should have been dispatched"
        );
    }

    #[tokio::test]
    async fn scheduler_tick_skips_future_tasks() {
        use crate::daemon::scheduler::{save_schedules, ScheduledTask};
        let dir = TempDir::new().expect("create tempdir");
        let config = test_config(dir.path());
        let tasks = vec![ScheduledTask {
            id: "future1".to_string(),
            cron: "0 9 * * *".to_string(),
            prompt: "hello".to_string(),
            model: None,
            last_run: None,
            next_run: Some("3000-01-01T00:00:00Z".to_string()),
        }];
        save_schedules(dir.path(), &tasks).expect("save");

        run_scheduler_tick(&config).await;

        // Task should not have been advanced (next_run unchanged)
        let loaded = crate::daemon::scheduler::load_schedules(dir.path()).expect("load");
        assert_eq!(loaded[0].next_run.as_deref(), Some("3000-01-01T00:00:00Z"));
        assert!(
            loaded[0].last_run.is_none(),
            "future task should not have run"
        );
    }

    #[tokio::test]
    async fn scheduler_tick_advances_due_task() {
        use crate::daemon::scheduler::{save_schedules, ScheduledTask};
        let dir = TempDir::new().expect("create tempdir");
        let config = test_config(dir.path());
        // Task due in the past
        let tasks = vec![ScheduledTask {
            id: "due1".to_string(),
            cron: "0 9 * * *".to_string(),
            prompt: "test prompt".to_string(),
            model: None,
            last_run: None,
            next_run: Some("2000-01-01T00:00:00Z".to_string()),
        }];
        save_schedules(dir.path(), &tasks).expect("save");

        run_scheduler_tick(&config).await;

        // Task should have been advanced — next_run should be ~1 day from now
        let loaded = crate::daemon::scheduler::load_schedules(dir.path()).expect("load");
        assert!(
            loaded[0].last_run.is_some(),
            "last_run should be set after execution"
        );
        let next = loaded[0]
            .next_run
            .as_ref()
            .expect("next_run should still be set");
        let next_dt = chrono::DateTime::parse_from_rfc3339(next).expect("valid RFC3339");
        assert!(
            next_dt > chrono::Utc::now(),
            "next_run should be in the future after advance"
        );
    }

    #[test]
    fn write_log_entry_connection_error_format() {
        let dir = TempDir::new().expect("create tempdir");
        write_log_entry(dir.path(), "error", "connection error: broken pipe", None);
        let path = daemon_log_path(dir.path());
        let content = std::fs::read_to_string(&path).expect("read log");
        let obj: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("one line")).expect("valid JSON");
        assert_eq!(obj["level"], "error");
        assert_eq!(obj["msg"], "connection error: broken pipe");
    }

    #[test]
    fn write_log_entry_shutdown_format() {
        let dir = TempDir::new().expect("create tempdir");
        write_log_entry(dir.path(), "info", "daemon shutting down", None);
        let path = daemon_log_path(dir.path());
        let content = std::fs::read_to_string(&path).expect("read log");
        let obj: serde_json::Value =
            serde_json::from_str(content.lines().next().expect("one line")).expect("valid JSON");
        assert_eq!(obj["level"], "info");
        assert_eq!(obj["msg"], "daemon shutting down");
        assert!(obj["pid"].is_u64());
    }

    #[test]
    fn chain_log_from_workspace_returns_artifacts() {
        let dir = TempDir::new().expect("create tempdir");
        std::fs::write(dir.path().join("cycle-01-planner.md"), "plan output").unwrap();
        std::fs::write(dir.path().join("cycle-01-builder.md"), "build output").unwrap();
        let entries = crate::orchestrate::chain_log_from_workspace(dir.path(), None).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].node, "builder");
        assert_eq!(entries[1].node, "planner");
    }

    #[test]
    fn chain_log_from_workspace_filters_by_cycle() {
        let dir = TempDir::new().expect("create tempdir");
        std::fs::write(dir.path().join("cycle-01-planner.md"), "c1").unwrap();
        std::fs::write(dir.path().join("cycle-02-planner.md"), "c2").unwrap();
        let entries = crate::orchestrate::chain_log_from_workspace(dir.path(), Some(2)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].cycle, 2);
    }

    #[test]
    fn chain_validate_detailed_catches_errors() {
        let config = crate::orchestrate::types::ChainConfig {
            name: String::new(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 60,
            workspace: std::path::PathBuf::from("/tmp/test"),
            directive: None,
            loop_forever: false,
            skip_permissions_warning: false,
        };
        let report = crate::orchestrate::validate_chain_config_detailed(&config, &[]);
        assert!(
            !report.errors.is_empty(),
            "empty name should produce errors"
        );
        assert!(report.report.contains("FAIL"));
    }

    #[test]
    fn chain_validate_detailed_passes_good_config() {
        let config = crate::orchestrate::types::ChainConfig {
            name: "test".to_string(),
            description: Some("a test chain".to_string()),
            nodes: vec![crate::orchestrate::types::ChainNodeConfig {
                id: "n1".to_string(),
                name: "n1".to_string(),
                role: "worker".to_string(),
                model: "llama3".to_string(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 3,
            max_total_turns: 60,
            workspace: std::path::PathBuf::from("/tmp/test"),
            directive: None,
            loop_forever: false,
            skip_permissions_warning: false,
        };
        let report = crate::orchestrate::validate_chain_config_detailed(&config, &[]);
        assert!(report.errors.is_empty());
        assert!(report.report.contains("PASS"));
    }

    fn make_test_chain_state() -> crate::orchestrate::types::ChainState {
        use chrono::Utc;
        crate::orchestrate::types::ChainState {
            chain_id: "test-1".to_string(),
            config: crate::orchestrate::types::ChainConfig {
                name: "test-chain".to_string(),
                description: None,
                nodes: vec![crate::orchestrate::types::ChainNodeConfig {
                    id: "n1".to_string(),
                    name: "worker".to_string(),
                    role: "worker".to_string(),
                    model: "llama3".to_string(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                }],
                max_cycles: 3,
                max_total_turns: 60,
                workspace: std::path::PathBuf::from("/tmp/test"),
                directive: None,
                loop_forever: false,
                skip_permissions_warning: false,
            },
            active_node_index: Some(0),
            node_statuses: std::collections::HashMap::new(),
            current_cycle: 2,
            turns_used: 10,
            node_outputs: std::collections::HashMap::new(),
            last_signal: None,
            last_updated: Utc::now(),
            last_abort_reason: None,
            total_duration_secs: 45.5,
            node_durations: {
                let mut m = std::collections::HashMap::new();
                m.insert("worker".to_string(), vec![10.0, 15.0, 20.5]);
                m
            },
            node_failures: std::collections::HashMap::new(),
            node_prompt_tokens: std::collections::HashMap::new(),
            node_completion_tokens: std::collections::HashMap::new(),
            pending_additions: vec![],
            pending_removals: vec![],
            pending_model_swaps: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn format_chain_metrics_produces_output() {
        let state = make_test_chain_state();
        let output = crate::orchestrate::format_chain_metrics(&state);
        assert!(output.contains("test-chain"));
        assert!(output.contains("worker"));
        assert!(output.contains("45.5"));
    }

    #[test]
    fn generate_chain_template_contains_name_and_model() {
        let yaml = crate::orchestrate::generate_chain_template("my-test", "llama3");
        assert!(yaml.contains("name: my-test"));
        assert!(yaml.contains("model: llama3"));
        assert!(yaml.contains("Planner"));
        assert!(yaml.contains("Builder"));
    }

    #[test]
    fn chain_init_writes_template_to_disk() {
        let dir = TempDir::new().expect("create tempdir");
        let chains_dir = dir.path().join("chains");
        std::fs::create_dir_all(&chains_dir).unwrap();
        let path = chains_dir.join("test.chain.yaml");
        let yaml = crate::orchestrate::generate_chain_template("test", "llama3");
        std::fs::write(&path, &yaml).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("name: test"));
        let parsed: serde_yaml::Value = serde_yaml::from_str(&content).unwrap();
        assert!(parsed["nodes"].is_sequence());
    }

    #[test]
    fn chain_log_event_serializes() {
        let event = DaemonEvent::ChainLog {
            chain_id: "c1".into(),
            level: "info".into(),
            message: "test log".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("chain_log"));
        assert!(json.contains("test log"));
        let deser: DaemonEvent = serde_json::from_str(&json).unwrap();
        match deser {
            DaemonEvent::ChainLog {
                chain_id,
                level,
                message,
            } => {
                assert_eq!(chain_id, "c1");
                assert_eq!(level, "info");
                assert_eq!(message, "test log");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn daemon_socket_has_restricted_permissions() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let socket_path = tmp.path().join("daemon.sock");
        let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let mode = std::fs::metadata(&socket_path).unwrap().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "daemon socket should be owner-only rw: {:o}",
            mode
        );
        drop(listener);
    }
}
