use crate::logging;
use anyhow::Context;

use super::protocol::DaemonEvent;
use super::{daemon_pid_path, daemon_socket_exists, daemon_socket_path, DaemonClient};

/// Next-step hints for the repeated daemon-operation failure modes.
/// Each matches the `Try:` convention enforced by `format_dm_error` so
/// the final stderr line reads `dm: error: <msg>. Try: <hint>`.
const HINT_DAEMON_NOT_RUNNING: &str = "Try: dm --daemon-start (or check ~/.dm/daemon.pid)";
const HINT_SESSION_NOT_FOUND: &str = "Try: dm daemon session list — then re-run with a valid id";
const HINT_PROTOCOL_MISMATCH: &str =
    "Try: dm --daemon-stop && dm --daemon-start (server/client version drift)";
#[cfg(test)]
const HINT_DAEMON_ALREADY_RUNNING: &str = "Try: dm --daemon-stop, then dm --daemon-start";

/// Pure formatter — returns the string the `eprintln!` site would emit.
/// Splitting this out of `report_daemon_error` keeps the hint logic
/// unit-testable without capturing stderr.
///
/// Semantics:
/// - `DaemonEvent::Error { message }` → caller's `hint` (may be None).
/// - Any other variant arriving at an error site is protocol drift;
///   we always attach `HINT_PROTOCOL_MISMATCH` for those, ignoring
///   the caller-supplied hint (it would be the wrong hint for the
///   wrong condition).
fn format_daemon_error(event: &DaemonEvent, hint: Option<&str>) -> String {
    match event {
        DaemonEvent::Error { message, .. } => {
            crate::error_hints::format_dm_error(&format!("error: {}", message), hint)
        }
        other => crate::error_hints::format_dm_error(
            &format!("unexpected response: {:?}", other),
            Some(HINT_PROTOCOL_MISMATCH),
        ),
    }
}

/// Thin `eprintln!` wrapper around `format_daemon_error` used by the
/// daemon subcommand call sites.
fn report_daemon_error(event: &DaemonEvent, hint: Option<&str>) {
    logging::log_err(&format_daemon_error(event, hint));
}

pub async fn handle_daemon_start(use_watchdog: bool) -> anyhow::Result<()> {
    if daemon_socket_exists() {
        println!("Daemon already running.");
        return Ok(());
    }

    let exe = std::env::current_exe().context("cannot determine executable path")?;

    if use_watchdog {
        let child = std::process::Command::new(&exe)
            .arg("--_daemon-watchdog")
            .env("DM_DAEMON_WATCHDOG", "1")
            .spawn()
            .context("failed to spawn daemon watchdog")?;
        let pid = child.id();
        println!("Daemon started with watchdog (pid: {}).", pid);
    } else {
        let child = std::process::Command::new(&exe)
            .arg("--_daemon-worker")
            .spawn()
            .context("failed to spawn daemon worker")?;
        let pid = child.id();
        println!("Daemon started (pid: {}).", pid);
    }
    Ok(())
}

pub fn handle_daemon_stop() -> anyhow::Result<()> {
    let pid_path = daemon_pid_path();
    let socket_path = daemon_socket_path();

    if !pid_path.exists() {
        println!("Daemon is not running (no PID file found).");
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_path).context("read PID file")?;
    let pid: i32 = pid_str.trim().parse().context("parse PID")?;

    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        match kill(Pid::from_raw(pid), Signal::SIGTERM) {
            Ok(()) => {}
            Err(nix::errno::Errno::ESRCH) => {}
            Err(e) => {
                logging::log_err(&crate::error_hints::format_dm_error(
                    &format!("failed to send SIGTERM to pid {}: {}", pid, e),
                    Some("Try: kill -TERM <pid> manually, or rm ~/.dm/daemon.pid if stale"),
                ));
            }
        }
    }
    #[cfg(not(unix))]
    {
        logging::log_err(&format!(
            "dm: daemon stop not supported on non-Unix platforms (pid: {})",
            pid
        ));
    }

    let _ = std::fs::remove_file(&pid_path);
    let _ = std::fs::remove_file(&socket_path);
    println!("Daemon stopped.");
    Ok(())
}

pub async fn handle_daemon_status() -> anyhow::Result<()> {
    let pid_path = daemon_pid_path();

    if !daemon_socket_exists() {
        if pid_path.exists() {
            println!("Daemon: not running (stale PID file present, socket unreachable).");
        } else {
            println!("Daemon: not running.");
        }
        return Ok(());
    }

    let pid_str = std::fs::read_to_string(&pid_path).unwrap_or_else(|_| "unknown".to_string());
    let pid = pid_str.trim().to_string();

    let uptime = std::fs::metadata(&pid_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|mtime| std::time::SystemTime::now().duration_since(mtime).ok())
        .map_or_else(
            || "unknown".to_string(),
            |d| {
                let secs = d.as_secs();
                if secs < 60 {
                    format!("{}s", secs)
                } else if secs < 3600 {
                    format!("{}m {}s", secs / 60, secs % 60)
                } else {
                    format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
                }
            },
        );

    let session_info = match DaemonClient::connect().await {
        Ok(mut client) => {
            if client
                .send_request("session.list", serde_json::json!({}))
                .await
                .is_ok()
            {
                match client.recv_event().await {
                    Ok(DaemonEvent::SessionList { sessions }) => {
                        format!("{} active session(s)", sessions.len())
                    }
                    _ => "session info unavailable".to_string(),
                }
            } else {
                "session info unavailable".to_string()
            }
        }
        Err(_) => "session info unavailable".to_string(),
    };

    let watchdog_enabled = std::env::var("DM_DAEMON_WATCHDOG").as_deref() == Ok("1");
    println!("Daemon: running");
    println!("  PID:      {}", pid);
    println!("  Uptime:   {}", uptime);
    println!("  Socket:   {}", daemon_socket_path().display());
    println!("  Status:   {}", session_info);
    println!(
        "  Watchdog: {}",
        if watchdog_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    Ok(())
}

pub async fn handle_daemon_chain_start(config_path: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_start(config_path).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStarted {
            chain_id,
            name,
            node_count,
        } => {
            println!("Chain started: {} (id: {})", name, chain_id);
            println!("  Nodes: {}", node_count);
        }
        other => report_daemon_error(&other, Some(HINT_DAEMON_NOT_RUNNING)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_status(chain_id: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_status(chain_id).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, state } => {
            let name = state["config"]["name"].as_str().unwrap_or("unknown");
            let cycle = state["current_cycle"].as_u64().unwrap_or(0);
            let turns = state["turns_used"].as_u64().unwrap_or(0);
            let max_turns = state["config"]["max_total_turns"].as_u64().unwrap_or(0);
            let total_secs = state["total_duration_secs"].as_f64().unwrap_or(0.0);
            println!("Chain: {} (id: {})", name, chain_id);
            println!("  Cycle:    {}", cycle);
            println!("  Turns:    {}/{}", turns, max_turns);
            println!("  Duration: {:.1}s", total_secs);
            if let Some(nodes) = state["config"]["nodes"].as_array() {
                println!("  Nodes:");
                for node in nodes {
                    let nname = node["name"].as_str().unwrap_or("?");
                    let status = state["node_statuses"]
                        .get(nname)
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    println!("    {:<16} {}", nname, status);
                }
            }
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_stop(chain_id: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_stop(chain_id).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainFinished {
            chain_id, reason, ..
        } => {
            println!("Chain '{}' stopping: {}", chain_id, reason);
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_list() -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_list().await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainList { chains } => {
            if chains.is_empty() {
                println!("No chains running.");
            } else {
                println!(
                    "{:<32} {:<20} {:>6} {:>6} {:<10}",
                    "CHAIN_ID", "NAME", "CYCLE", "NODES", "STATUS"
                );
                for ch in &chains {
                    println!(
                        "{:<32} {:<20} {:>6} {:>6} {:<10}",
                        ch.chain_id, ch.name, ch.current_cycle, ch.node_count, ch.status
                    );
                }
            }
        }
        other => report_daemon_error(&other, Some(HINT_DAEMON_NOT_RUNNING)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_attach(chain_id: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_attach(chain_id).await?;

    let first = dc.recv_event().await?;
    match &first {
        DaemonEvent::Error { .. } => {
            report_daemon_error(&first, Some(HINT_SESSION_NOT_FOUND));
            return Ok(());
        }
        _ => {
            print_chain_event(&first);
        }
    }

    loop {
        match dc.recv_event().await {
            Ok(event) => {
                let is_terminal = matches!(
                    &event,
                    DaemonEvent::ChainFinished { .. } | DaemonEvent::Error { .. }
                );
                print_chain_event(&event);
                if is_terminal {
                    break;
                }
            }
            Err(_) => {
                logging::log("[detached]");
                break;
            }
        }
    }
    Ok(())
}

pub fn print_chain_event(event: &DaemonEvent) {
    match event {
        DaemonEvent::ChainNodeTransition {
            cycle,
            node_name,
            status,
            ..
        } => {
            println!("[cycle {}] {} → {}", cycle, node_name, status);
        }
        DaemonEvent::ChainCycleComplete { cycle, .. } => {
            println!("[cycle {}] complete", cycle);
        }
        DaemonEvent::ChainFinished {
            success, reason, ..
        } => {
            let icon = if *success { "✓" } else { "✗" };
            println!("{} Chain finished: {}", icon, reason);
        }
        DaemonEvent::ChainStarted {
            name, node_count, ..
        } => {
            println!("Chain '{}' started ({} nodes)", name, node_count);
        }
        other => {
            println!("{:?}", other);
        }
    }
}

pub async fn handle_daemon_chain_pause(chain_id: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_pause(chain_id).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!("Chain '{}' paused.", chain_id);
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_resume(chain_id: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_resume(chain_id).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!("Chain '{}' resumed.", chain_id);
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_talk(arg: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let parts: Vec<&str> = arg.splitn(3, ':').collect();
    if parts.len() < 3 {
        anyhow::bail!("Expected format: CHAIN_ID:NODE:MESSAGE");
    }
    let (chain_id, node, message) = (parts[0], parts[1], parts[2]);
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_talk(chain_id, node, message).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!(
                "Message injected into node '{}' of chain '{}'.",
                node, chain_id
            );
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_add(arg: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let parts: Vec<&str> = arg.splitn(3, ':').collect();
    if parts.len() < 3 {
        anyhow::bail!("Expected format: CHAIN_ID:NAME:MODEL");
    }
    let (chain_id, name, model) = (parts[0], parts[1], parts[2]);
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_add(chain_id, name, model, name, None).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!("Node '{}' added to chain '{}'.", name, chain_id);
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_remove(arg: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let parts: Vec<&str> = arg.splitn(2, ':').collect();
    if parts.len() < 2 {
        anyhow::bail!("Expected format: CHAIN_ID:NODE");
    }
    let (chain_id, node) = (parts[0], parts[1]);
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_remove(chain_id, node).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!("Node '{}' removal queued for chain '{}'.", node, chain_id);
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

pub async fn handle_daemon_chain_model(arg: &str) -> anyhow::Result<()> {
    if !daemon_socket_exists() {
        anyhow::bail!("Daemon is not running. Start it with: dm --daemon-start");
    }
    let parts: Vec<&str> = arg.splitn(3, ':').collect();
    if parts.len() < 3 {
        anyhow::bail!("Expected format: CHAIN_ID:NODE:MODEL");
    }
    let (chain_id, node, model) = (parts[0], parts[1], parts[2]);
    let mut dc = DaemonClient::connect().await.context("connect to daemon")?;
    dc.chain_model(chain_id, node, model).await?;
    match dc.recv_event().await? {
        DaemonEvent::ChainStatus { chain_id, .. } => {
            println!(
                "Model swap to '{}' queued for node '{}' in chain '{}'.",
                model, node, chain_id
            );
        }
        evt @ DaemonEvent::Error { .. } => {
            report_daemon_error(&evt, Some(HINT_SESSION_NOT_FOUND));
            std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
        }
        other => report_daemon_error(&other, Some(HINT_SESSION_NOT_FOUND)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_event(message: &str) -> DaemonEvent {
        DaemonEvent::Error {
            session_id: "test-session".into(),
            message: message.into(),
        }
    }

    fn chain_finished_event() -> DaemonEvent {
        DaemonEvent::ChainFinished {
            chain_id: "c1".into(),
            success: true,
            reason: "done".into(),
        }
    }

    #[test]
    fn report_error_with_hint_renders_dm_prefix_and_try() {
        let out = format_daemon_error(&err_event("boom"), Some(HINT_DAEMON_NOT_RUNNING));
        assert!(out.starts_with("dm: "), "dm prefix present: {out}");
        assert!(out.contains("error: boom"), "message body: {out}");
        assert!(out.contains("Try:"), "Try tail: {out}");
        assert!(out.contains("dm --daemon-start"), "hint content: {out}");
    }

    #[test]
    fn report_error_without_hint_omits_try() {
        let out = format_daemon_error(&err_event("boom"), None);
        assert!(out.starts_with("dm: "), "dm prefix still present: {out}");
        assert!(
            !out.contains("Try:"),
            "no Try tail when hint is None: {out}"
        );
        assert!(out.contains("error: boom"), "message body: {out}");
    }

    #[test]
    fn report_unexpected_response_always_hints_protocol_mismatch() {
        // Caller's hint is intentionally ignored for non-Error variants —
        // the condition is protocol drift, not whatever the caller expected.
        let out = format_daemon_error(&chain_finished_event(), Some(HINT_SESSION_NOT_FOUND));
        assert!(
            out.contains("unexpected response"),
            "message marks drift: {out}"
        );
        assert!(
            out.contains("version drift") || out.contains("server/client"),
            "HINT_PROTOCOL_MISMATCH wording: {out}"
        );
        assert!(
            !out.contains("session list"),
            "caller's SESSION_NOT_FOUND hint must not leak for drift: {out}"
        );
    }

    #[test]
    fn hint_constants_all_start_with_try() {
        for hint in [
            HINT_DAEMON_NOT_RUNNING,
            HINT_SESSION_NOT_FOUND,
            HINT_PROTOCOL_MISMATCH,
            HINT_DAEMON_ALREADY_RUNNING,
        ] {
            assert!(
                hint.starts_with("Try:"),
                "daemon hint should start with `Try:` to chain after `format_dm_error`: {hint}"
            );
            assert!(!hint.is_empty(), "hint non-empty: {hint}");
        }
    }

    /// Pillar 1 Headless #1: `src/daemon/server.rs` must stay
    /// `eprintln!`-free. Route any new output through `logging::log_err`
    /// / `log_info` / `log_warn`. A future cycle that regresses this
    /// will fail this canary before it ships.
    #[test]
    fn daemon_server_rs_has_zero_eprintlns() {
        const SRC: &str = include_str!("server.rs");
        let mut offenders: Vec<(usize, &str)> = Vec::new();
        for (i, line) in SRC.lines().enumerate() {
            if line.contains("eprintln!(") && !line.trim_start().starts_with("//") {
                offenders.push((i + 1, line));
            }
        }
        assert!(
            offenders.is_empty(),
            "src/daemon/server.rs must remain eprintln-free — route output through \
             `crate::logging::log_err` / `log_info` / `log_warn`. Offenders:\n{}",
            offenders
                .iter()
                .map(|(ln, l)| format!("  :{} {}", ln, l.trim()))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Same canary for `src/api/mod.rs` — the REST/web API path is
    /// headless by definition; eprintln there corrupts clients that
    /// capture stderr as log output.
    #[test]
    fn api_mod_rs_has_zero_eprintlns() {
        const SRC: &str = include_str!("../api/mod.rs");
        let mut offenders: Vec<(usize, &str)> = Vec::new();
        for (i, line) in SRC.lines().enumerate() {
            if line.contains("eprintln!(") && !line.trim_start().starts_with("//") {
                offenders.push((i + 1, line));
            }
        }
        assert!(
            offenders.is_empty(),
            "src/api/mod.rs must remain eprintln-free — route output through \
             `crate::logging::log_err` / `log_info` / `log_warn`. Offenders:\n{}",
            offenders
                .iter()
                .map(|(ln, l)| format!("  :{} {}", ln, l.trim()))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn daemon_stop_hint_mentions_daemon_start() {
        let out = format_daemon_error(
            &err_event("pid 42 not found"),
            Some(HINT_DAEMON_NOT_RUNNING),
        );
        assert!(
            out.contains("dm --daemon-start"),
            "daemon-not-running hint points at start command: {out}"
        );
    }

    #[test]
    fn daemon_send_hint_mentions_session_list() {
        let out = format_daemon_error(
            &err_event("session 'abc' not found"),
            Some(HINT_SESSION_NOT_FOUND),
        );
        assert!(
            out.contains("session list"),
            "session-not-found hint points at list command: {out}"
        );
    }
}
