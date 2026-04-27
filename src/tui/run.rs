//! TUI main event loop and layout execution.
//!
//! Handles terminal initialization, frame rendering, and polling for
//! both UI inputs and backend agent events.

use crate::daemon::client::DaemonClient;
use crate::daemon::protocol::DaemonEvent;
use crate::ollama::client::OllamaClient;
use crate::orchestrate::types::ChainNodeStatus;
use crate::tui::{
    app::{App, EntryKind},
    events::handle_backend,
    input::handle_key,
    BackendEvent,
};
use crossterm::{
    event::{DisableBracketedPaste, EnableBracketedPaste, Event, EventStream},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::stdout;
use tokio::sync::mpsc;

/// Restore the terminal to a usable state: leave raw mode, disable
/// xterm alternate-scroll, disable bracketed paste, leave the alternate
/// screen. Extracted so the panic hook can run it *before* writing to
/// stderr — otherwise `LeaveAlternateScreen` (via `TerminalGuard::drop`
/// during unwind) discards the crash message on most terminals.
pub(crate) fn restore_terminal() {
    let _ = disable_raw_mode();
    // Disable xterm alternate-scroll mode
    let _ = std::io::Write::write_all(&mut std::io::stdout(), b"\x1b[?1007l");
    let _ = execute!(
        std::io::stdout(),
        DisableBracketedPaste,
        LeaveAlternateScreen
    );
}

/// RAII guard that restores the terminal even if the future is dropped (SIGTERM, panic, etc.)
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Run the TUI event loop.
///
/// `daemon_client` is `Some` when connected to a running daemon; `None` for
/// the default local-agent mode. When `Some`, user input is forwarded to the
/// daemon via `turn.send` and daemon events are converted back to
/// [`BackendEvent`] via [`DaemonEvent::to_backend_event`].
pub async fn run_tui(
    app: &mut App,
    event_rx: &mut mpsc::Receiver<BackendEvent>,
    user_tx: &mpsc::Sender<String>,
    client: &OllamaClient,
    cancel_tx: tokio::sync::watch::Sender<bool>,
    gpu_rx: tokio::sync::watch::Receiver<Option<crate::gpu::GpuStats>>,
    daemon_client: Option<DaemonClient>,
) -> anyhow::Result<()> {
    // Pre-fetch model list for /model Tab autocomplete
    if let Ok(models) = client.list_models().await {
        app.available_models = models.into_iter().map(|m| m.name).collect();
    }

    if daemon_client.is_some() {
        app.daemon_mode = true;
    }

    // Install the panic hook before entering raw mode. The hook fires FIRST
    // (before unwind, before TerminalGuard's Drop), so it restores the
    // terminal itself via the injected callback — otherwise stderr writes
    // land inside the alt-screen and get discarded when LeaveAlternateScreen
    // runs during unwind.
    crate::panic_hook::install(
        crate::panic_hook::CrashContext::for_session(
            app.config_dir.clone(),
            app.session_id.clone(),
        ),
        restore_terminal,
    );

    // Surface unrecovered crash markers from prior sessions as a one-line
    // SystemInfo nudge. The full list is available via `dm --recovery`.
    if let Some(banner) = crate::panic_hook::build_recovery_banner(
        &crate::panic_hook::list_panic_markers(&app.config_dir),
    ) {
        app.push_entry(EntryKind::SystemInfo, banner);
    }

    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableBracketedPaste)?;
    // Enable xterm alternate-scroll mode: scroll wheel sends Up/Down arrow keys
    // instead of mouse events, so native text selection and paste still work.
    std::io::Write::write_all(&mut out, b"\x1b[?1007h")?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;

    // _guard restores the terminal on drop — handles SIGTERM, panic, and normal exit.
    let _guard = TerminalGuard;
    run_loop(
        &mut terminal,
        app,
        event_rx,
        user_tx,
        client,
        &cancel_tx,
        gpu_rx,
        daemon_client,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    event_rx: &mut mpsc::Receiver<BackendEvent>,
    user_tx: &mpsc::Sender<String>,
    client: &OllamaClient,
    cancel_tx: &tokio::sync::watch::Sender<bool>,
    mut gpu_rx: tokio::sync::watch::Receiver<Option<crate::gpu::GpuStats>>,
    mut daemon_client: Option<DaemonClient>,
) -> anyhow::Result<()> {
    let event_stream = EventStream::new();
    tokio::pin!(event_stream);

    loop {
        terminal.draw(|f| crate::tui::ui::render(f, app))?;

        tokio::select! {
            maybe_key = event_stream.next() => {
                match maybe_key {
                    Some(Ok(Event::Key(key))) => {
                        handle_key(app, key, user_tx, client, cancel_tx, &mut daemon_client).await?;
                    }
                    Some(Ok(Event::Paste(text))) => {
                        if app.mode == crate::tui::app::Mode::Input {
                            let cleaned = text.replace('\r', "");
                            for ch in cleaned.chars() {
                                app.input_insert(ch);
                            }
                        }
                    }
                    Some(Ok(Event::Mouse(_))) => {
                        // Mouse capture is disabled — scroll wheel arrives as
                        // arrow-key events via xterm alternate-scroll mode.
                    }
                    Some(Err(e)) => return Err(e.into()),
                    _ => {}
                }
            }
            maybe_backend = event_rx.recv() => {
                match maybe_backend {
                    Some(event) => handle_backend(app, event),
                    None => break,
                }
            }
            // Daemon mode: receive events from the daemon and forward them to the TUI.
            maybe_daemon = recv_daemon_event(&mut daemon_client) => {
                match maybe_daemon {
                    Some(Ok(daemon_event)) => {
                        if let Some(be) = daemon_event_to_backend(daemon_event) {
                            handle_backend(app, be);
                        }
                    }
                    Some(Err(_)) => {
                        // Daemon disconnected — fall back gracefully.
                        app.push_entry(
                            EntryKind::SystemInfo,
                            "Daemon connection lost — continuing in local mode.".to_string(),
                        );
                        app.daemon_mode = false;
                        daemon_client = None;
                    }
                    None => {} // No daemon client — normal local-agent path.
                }
            }
            _ = gpu_rx.changed() => {
                app.gpu_stats.clone_from(&gpu_rx.borrow());
            }
            maybe_chain = recv_chain_event(&mut app.chain_event_rx) => {
                if let Some(ref event) = maybe_chain {
                    if let Some(msg) = format_chain_event(event) {
                        app.push_chain_log(msg.clone());
                        app.push_entry(EntryKind::SystemInfo, msg);
                    }
                    apply_chain_event(app, event);
                }
            }
            // Tick every 100ms to drive spinner animation; poll chain status every second.
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                app.tick = app.tick.wrapping_add(1);
                if app.tick.is_multiple_of(10) {
                    app.chain_state = crate::orchestrate::chain_status();
                }
                for w in crate::warnings::drain_warnings() {
                    app.push_entry(EntryKind::SystemInfo, format!("⚠ {}", w));
                }
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Attempt to receive one event from the daemon client.
/// Returns `None` immediately (polling-free) when there is no daemon client.
/// Returns `Some(Err(_))` on connection close or parse failure.
async fn recv_daemon_event(
    daemon_client: &mut Option<DaemonClient>,
) -> Option<anyhow::Result<DaemonEvent>> {
    match daemon_client {
        None => {
            // No daemon — park this future forever so it never fires.
            std::future::pending::<Option<anyhow::Result<DaemonEvent>>>().await
        }
        Some(dc) => Some(dc.recv_event().await),
    }
}

/// Convert a [`DaemonEvent`] to a [`BackendEvent`], returning `None` for daemon-only events.
fn daemon_event_to_backend(de: DaemonEvent) -> Option<BackendEvent> {
    de.to_backend_event()
}

async fn recv_chain_event(
    rx: &mut Option<tokio::sync::broadcast::Receiver<DaemonEvent>>,
) -> Option<DaemonEvent> {
    match rx {
        Some(rx) => match rx.recv().await {
            Ok(event) => Some(event),
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                Some(DaemonEvent::ChainFinished {
                    chain_id: String::new(),
                    success: false,
                    reason: "chain task exited unexpectedly".into(),
                })
            }
            Err(_) => None,
        },
        None => {
            std::future::pending::<()>().await;
            None
        }
    }
}

fn apply_chain_event(app: &mut App, event: &DaemonEvent) {
    match event {
        DaemonEvent::ChainStarted { .. } => {
            app.chain_state = crate::orchestrate::chain_status();
        }
        DaemonEvent::ChainNodeTransition {
            node_name,
            status,
            cycle,
            ..
        } => {
            if let Some(ref mut cs) = app.chain_state {
                let Some(node_status) = ChainNodeStatus::from_status_str(status) else {
                    return;
                };
                if matches!(node_status, ChainNodeStatus::Running) {
                    cs.active_node_index =
                        cs.config.nodes.iter().position(|n| n.name == *node_name);
                }
                cs.node_statuses.insert(node_name.clone(), node_status);
                cs.current_cycle = *cycle;
            }
        }
        DaemonEvent::ChainCycleComplete { cycle, .. } => {
            if let Some(ref mut cs) = app.chain_state {
                cs.current_cycle = *cycle;
                cs.active_node_index = None;
            }
        }
        DaemonEvent::ChainFinished { .. } => {
            app.chain_state = None;
            app.chain_event_rx = None;
            crate::orchestrate::clear_chain();
        }
        _ => {}
    }
}

fn format_chain_event(event: &DaemonEvent) -> Option<String> {
    match event {
        DaemonEvent::ChainStarted {
            name, node_count, ..
        } => Some(format!(
            "[chain] started '{}' with {} nodes",
            name, node_count
        )),
        DaemonEvent::ChainNodeTransition {
            node_name,
            status,
            cycle,
            ..
        } => Some(format!(
            "[chain] cycle {} · {} → {}",
            cycle, node_name, status
        )),
        DaemonEvent::ChainCycleComplete { cycle, .. } => {
            Some(format!("[chain] cycle {} complete", cycle))
        }
        DaemonEvent::ChainFinished {
            success, reason, ..
        } => {
            let icon = if *success { "✓" } else { "✗" };
            Some(format!("[chain] {} finished: {}", icon, reason))
        }
        DaemonEvent::ChainLog { level, message, .. } => {
            let prefix = if level == "error" { "✗" } else { "⚠" };
            Some(format!("[chain] {} {}", prefix, message))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_chain_event_node_transition() {
        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 1,
            node_name: "builder".into(),
            status: "running".into(),
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainNodeTransition formats to Some"),
            "[chain] cycle 1 · builder → running"
        );
    }

    #[test]
    fn format_chain_event_cycle_complete() {
        let event = DaemonEvent::ChainCycleComplete {
            chain_id: "c1".into(),
            cycle: 2,
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainCycleComplete formats to Some"),
            "[chain] cycle 2 complete"
        );
    }

    #[test]
    fn format_chain_event_finished_success() {
        let event = DaemonEvent::ChainFinished {
            chain_id: "c1".into(),
            success: true,
            reason: "all cycles done".into(),
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainFinished (success) formats to Some"),
            "[chain] ✓ finished: all cycles done"
        );
    }

    #[test]
    fn format_chain_event_finished_failure() {
        let event = DaemonEvent::ChainFinished {
            chain_id: "c1".into(),
            success: false,
            reason: "node crashed".into(),
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainFinished (failure) formats to Some"),
            "[chain] ✗ finished: node crashed"
        );
    }

    #[test]
    fn format_chain_event_started() {
        let event = DaemonEvent::ChainStarted {
            chain_id: "c1".into(),
            name: "build-chain".into(),
            node_count: 3,
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainStarted formats to Some"),
            "[chain] started 'build-chain' with 3 nodes"
        );
    }

    #[test]
    fn format_chain_event_non_chain_returns_none() {
        let event = DaemonEvent::Pong;
        assert!(format_chain_event(&event).is_none());
    }

    #[test]
    fn format_chain_event_log_warn() {
        let event = DaemonEvent::ChainLog {
            chain_id: "c1".into(),
            level: "warn".into(),
            message: "node 'builder' expects input from 'planner' but no output available yet"
                .into(),
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainLog warn formats to Some"),
            "[chain] ⚠ node 'builder' expects input from 'planner' but no output available yet"
        );
    }

    #[test]
    fn format_chain_event_log_error() {
        let event = DaemonEvent::ChainLog {
            chain_id: "c1".into(),
            level: "error".into(),
            message: "node 'builder' failed after 3 attempt(s): timeout. Aborting cycle.".into(),
        };
        assert_eq!(
            format_chain_event(&event).expect("ChainLog error formats to Some"),
            "[chain] ✗ node 'builder' failed after 3 attempt(s): timeout. Aborting cycle."
        );
    }

    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};
    use crate::tui::app::App;
    use std::collections::HashMap;

    fn make_app() -> App {
        App::new(
            "test-model".into(),
            "localhost:11434".into(),
            "test-session".into(),
            std::path::PathBuf::from("/tmp/dm-test"),
            vec![],
        )
    }

    fn make_chain_state() -> ChainState {
        ChainState {
            chain_id: "c1".into(),
            config: ChainConfig {
                name: "test-chain".into(),
                description: None,
                nodes: vec![
                    ChainNodeConfig {
                        id: "n1".into(),
                        name: "planner".into(),
                        role: "planner".into(),
                        model: "m1".into(),
                        description: None,
                        system_prompt_override: None,
                        system_prompt_file: None,
                        input_from: None,
                        max_retries: 1,
                        timeout_secs: 3600,
                        max_tool_turns: 200,
                    },
                    ChainNodeConfig {
                        id: "n2".into(),
                        name: "builder".into(),
                        role: "builder".into(),
                        model: "m1".into(),
                        description: None,
                        system_prompt_override: None,
                        system_prompt_file: None,
                        input_from: Some("n1".into()),
                        max_retries: 1,
                        timeout_secs: 3600,
                        max_tool_turns: 200,
                    },
                ],
                max_cycles: 5,
                max_total_turns: 100,
                workspace: std::path::PathBuf::from("/tmp"),
                skip_permissions_warning: true,
                loop_forever: false,
                directive: None,
            },
            active_node_index: None,
            node_statuses: HashMap::new(),
            current_cycle: 0,
            turns_used: 0,
            node_outputs: HashMap::new(),
            last_signal: None,
            last_updated: chrono::Utc::now(),
            last_abort_reason: None,
            pending_additions: vec![],
            pending_removals: vec![],
            pending_model_swaps: HashMap::new(),
            node_durations: HashMap::new(),
            node_failures: HashMap::new(),
            total_duration_secs: 0.0,
            node_prompt_tokens: HashMap::new(),
            node_completion_tokens: HashMap::new(),
        }
    }

    #[test]
    fn apply_chain_event_node_transition_updates_state() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 2,
            node_name: "builder".into(),
            status: "running".into(),
        };
        apply_chain_event(&mut app, &event);

        let cs = app
            .chain_state
            .as_ref()
            .expect("ChainNodeTransition must populate chain_state");
        assert!(matches!(
            cs.node_statuses.get("builder"),
            Some(ChainNodeStatus::Running)
        ));
        assert_eq!(cs.active_node_index, Some(1));
        assert_eq!(cs.current_cycle, 2);
    }

    #[test]
    fn apply_chain_event_cycle_complete_clears_active_node() {
        let mut app = make_app();
        let mut cs = make_chain_state();
        cs.active_node_index = Some(0);
        app.chain_state = Some(cs);

        let event = DaemonEvent::ChainCycleComplete {
            chain_id: "c1".into(),
            cycle: 3,
        };
        apply_chain_event(&mut app, &event);

        let cs = app
            .chain_state
            .as_ref()
            .expect("ChainCycleComplete preserves chain_state (only clears active_node)");
        assert_eq!(cs.current_cycle, 3);
        assert_eq!(cs.active_node_index, None);
    }

    #[test]
    fn apply_chain_event_finished_clears_everything() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainFinished {
            chain_id: "c1".into(),
            success: true,
            reason: "done".into(),
        };
        apply_chain_event(&mut app, &event);

        assert!(app.chain_state.is_none());
        assert!(app.chain_event_rx.is_none());
    }

    #[test]
    fn apply_chain_event_completed_status() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 1,
            node_name: "planner".into(),
            status: "completed".into(),
        };
        apply_chain_event(&mut app, &event);

        let cs = app
            .chain_state
            .as_ref()
            .expect("ChainNodeTransition with 'completed' preserves chain_state");
        assert!(matches!(
            cs.node_statuses.get("planner"),
            Some(ChainNodeStatus::Completed)
        ));
    }

    #[test]
    fn apply_chain_event_failed_status() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 1,
            node_name: "builder".into(),
            status: "failed: timeout".into(),
        };
        apply_chain_event(&mut app, &event);

        let cs = app
            .chain_state
            .as_ref()
            .expect("ChainNodeTransition with 'failed' preserves chain_state");
        match cs.node_statuses.get("builder") {
            Some(ChainNodeStatus::Failed(reason)) => assert_eq!(reason, "timeout"),
            other => panic!("expected Failed(\"timeout\"), got {:?}", other),
        }
    }

    #[test]
    fn apply_chain_event_unknown_status_ignored() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainNodeTransition {
            chain_id: "c1".into(),
            cycle: 1,
            node_name: "planner".into(),
            status: "bogus_status".into(),
        };
        apply_chain_event(&mut app, &event);

        let cs = app
            .chain_state
            .as_ref()
            .expect("unknown-status event leaves chain_state untouched");
        assert!(!cs.node_statuses.contains_key("planner"));
    }

    #[test]
    fn status_roundtrip_through_display_and_parse() {
        use crate::orchestrate::types::ChainNodeStatus;
        let cases = vec![
            ChainNodeStatus::Pending,
            ChainNodeStatus::Running,
            ChainNodeStatus::Completed,
            ChainNodeStatus::Paused,
            ChainNodeStatus::Failed("connection refused".into()),
        ];
        for original in &cases {
            let display = original.to_string();
            let parsed = ChainNodeStatus::from_status_str(&display)
                .unwrap_or_else(|| panic!("failed to parse '{}'", display));
            assert_eq!(original.to_string(), parsed.to_string());
        }
    }

    #[tokio::test]
    async fn recv_chain_event_sender_dropped_triggers_cleanup() {
        let (tx, rx) = tokio::sync::broadcast::channel::<DaemonEvent>(4);
        let mut opt_rx = Some(rx);
        drop(tx);
        let event = recv_chain_event(&mut opt_rx).await;
        match event {
            Some(DaemonEvent::ChainFinished {
                success, reason, ..
            }) => {
                assert!(!success);
                assert!(reason.contains("unexpectedly"));
            }
            other => panic!("expected ChainFinished, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn recv_chain_event_none_rx_pends_forever() {
        let mut opt_rx: Option<tokio::sync::broadcast::Receiver<DaemonEvent>> = None;
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            recv_chain_event(&mut opt_rx),
        )
        .await;
        assert!(result.is_err(), "None rx should pend indefinitely");
    }

    #[test]
    fn apply_chain_finished_cleans_up_ghost_chain() {
        let mut app = make_app();
        app.chain_state = Some(make_chain_state());

        let event = DaemonEvent::ChainFinished {
            chain_id: String::new(),
            success: false,
            reason: "runner error: turn cap reached".into(),
        };
        apply_chain_event(&mut app, &event);

        assert!(
            app.chain_state.is_none(),
            "chain_state should be cleaned up"
        );
        assert!(
            app.chain_event_rx.is_none(),
            "chain_event_rx should be cleaned up"
        );
    }

    #[test]
    fn format_chain_event_produces_log_lines() {
        let event = DaemonEvent::ChainLog {
            chain_id: "c1".into(),
            level: "info".into(),
            message: "test log message".into(),
        };
        let msg = format_chain_event(&event).expect("ChainLog should produce a message");
        assert!(msg.contains("test log message"));
    }

    #[test]
    fn chain_log_toggle_requires_data() {
        let mut app = make_app();
        assert!(!app.show_chain_pane);
        // Toggle without chain data — should remain hidden
        if app.chain_state.is_some() || !app.chain_log.is_empty() {
            app.show_chain_pane = !app.show_chain_pane;
        }
        assert!(!app.show_chain_pane, "should not toggle without chain data");

        // Add log data and toggle
        app.push_chain_log("test".into());
        if app.chain_state.is_some() || !app.chain_log.is_empty() {
            app.show_chain_pane = !app.show_chain_pane;
        }
        assert!(app.show_chain_pane, "should toggle with chain log data");
    }
}
