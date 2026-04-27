use crate::ollama::client::OllamaClient;
use std::path::Path;
use std::time::Instant;

/// Returns true if enough time has passed since the last trigger (debounce logic).
/// Updates `last_trigger` to now when returning true.
pub fn should_trigger(last_trigger: &mut Option<Instant>, debounce_ms: u64) -> bool {
    let now = Instant::now();
    match *last_trigger {
        Some(t) if now.duration_since(t).as_millis() < debounce_ms as u128 => false,
        _ => {
            *last_trigger = Some(now);
            true
        }
    }
}

/// Watch files matching `pattern` and re-run `testfix::run_test_fix` on change.
/// Uses notify for filesystem events and debounces to 500ms.
pub async fn run_watch_fix(
    pattern: &str,
    cmd: &str,
    max_rounds: usize,
    client: &OllamaClient,
    config_dir: &Path,
) -> anyhow::Result<()> {
    use notify::{RecommendedWatcher, RecursiveMode, Watcher};

    // Determine directories to watch from the glob pattern.
    // If pattern starts with "src/", watch "src/". Otherwise expand the glob and collect parent dirs.
    let watch_dirs: Vec<std::path::PathBuf> = if pattern.starts_with("src/") {
        let src = std::env::current_dir()?.join("src");
        if src.exists() {
            vec![src]
        } else {
            vec![std::env::current_dir()?]
        }
    } else {
        let mut dirs: std::collections::HashSet<std::path::PathBuf> =
            std::collections::HashSet::new();
        if let Ok(paths) = glob::glob(pattern) {
            for path_result in paths.flatten() {
                if let Some(parent) = path_result.parent() {
                    dirs.insert(parent.to_path_buf());
                }
            }
        }
        if dirs.is_empty() {
            vec![std::env::current_dir()?]
        } else {
            dirs.into_iter().collect()
        }
    };

    crate::logging::log(&format!(
        "[dm] Watching {} director{} for changes matching '{}' (Ctrl+C to stop)…",
        watch_dirs.len(),
        if watch_dirs.len() == 1 { "y" } else { "ies" },
        pattern
    ));

    // Bridge sync notify channel to async tokio channel.
    let (sync_tx, sync_rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();
    let (async_tx, mut async_rx) = tokio::sync::mpsc::channel::<()>(16);

    // Spawn a blocking thread to drain the sync channel and forward to the async side.
    let _notify_thread = std::thread::spawn(move || {
        // Create watcher inside the thread so it's owned here.
        let mut watcher: RecommendedWatcher = {
            let tx = sync_tx.clone();
            match Watcher::new(
                tx,
                notify::Config::default().with_poll_interval(std::time::Duration::from_millis(500)),
            ) {
                Ok(w) => w,
                Err(e) => {
                    crate::logging::log_err(&format!("[dm] Failed to create watcher: {}", e));
                    return;
                }
            }
        };

        for dir in &watch_dirs {
            if let Err(e) = watcher.watch(dir, RecursiveMode::Recursive) {
                crate::logging::log_err(&format!("[dm] Failed to watch {}: {}", dir.display(), e));
            }
        }

        let mut last_trigger: Option<Instant> = None;

        loop {
            match sync_rx.recv() {
                Ok(Ok(_event)) => {
                    if should_trigger(&mut last_trigger, 500) {
                        // Forward to async side; ignore send errors (means receiver dropped)
                        if async_tx.blocking_send(()).is_err() {
                            break;
                        }
                    }
                }
                Ok(Err(e)) => {
                    crate::logging::log_err(&format!("[dm] Watch error: {}", e));
                }
                Err(_) => break, // sync channel disconnected
            }
        }
    });

    // Run an initial fix pass immediately.
    crate::logging::log("[dm] Running initial test pass…");
    run_and_report(cmd, max_rounds, client, config_dir).await;

    loop {
        tokio::select! {
            Some(()) = async_rx.recv() => {
                crate::logging::log("[dm] Change detected — running tests…");
                run_and_report(cmd, max_rounds, client, config_dir).await;
            }
            _ = tokio::signal::ctrl_c() => {
                crate::logging::log("[dm] Stopping watch-fix.");
                break;
            }
        }
    }

    Ok(())
}

async fn run_and_report(cmd: &str, max_rounds: usize, client: &OllamaClient, config_dir: &Path) {
    match super::run_test_fix(cmd, max_rounds, client, config_dir).await {
        Ok(()) => {
            println!("✓ All tests pass");
        }
        Err(e) => {
            println!("✗ Still failing after {} rounds: {}", max_rounds, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debounce_deduplicates_rapid_events() {
        let mut last = None;
        assert!(should_trigger(&mut last, 500));
        assert!(!should_trigger(&mut last, 500)); // immediate second call
        std::thread::sleep(std::time::Duration::from_millis(600));
        assert!(should_trigger(&mut last, 500)); // after debounce window
    }

    #[test]
    fn debounce_allows_first_call_with_none() {
        let mut last: Option<Instant> = None;
        assert!(
            should_trigger(&mut last, 500),
            "first call should always trigger"
        );
        assert!(last.is_some(), "last_trigger should be set after trigger");
    }

    #[test]
    fn debounce_zero_ms_always_triggers() {
        let mut last: Option<Instant> = None;
        assert!(
            should_trigger(&mut last, 0),
            "zero debounce: first call triggers"
        );
        // With 0ms debounce, any elapsed time is >= 0, so subsequent calls also trigger.
        assert!(
            should_trigger(&mut last, 0),
            "zero debounce: second call also triggers"
        );
    }

    #[test]
    fn debounce_updates_timestamp_on_trigger() {
        let mut last: Option<Instant> = None;
        let before = Instant::now();
        should_trigger(&mut last, 500);
        let ts = last.expect("last_trigger must be set after triggering");
        assert!(ts >= before, "timestamp should be set to now or after");
    }

    #[test]
    fn debounce_does_not_update_timestamp_when_suppressed() {
        let mut last: Option<Instant> = None;
        should_trigger(&mut last, 500); // first: sets timestamp
        let ts1 = last.expect("must be set");
        std::thread::sleep(std::time::Duration::from_millis(1));
        should_trigger(&mut last, 500); // second: suppressed, should NOT update
        let ts2 = last.expect("still set");
        assert_eq!(ts1, ts2, "suppressed call must not update last_trigger");
    }

    #[test]
    fn debounce_very_large_threshold_suppresses() {
        // 60-second debounce window: second call within milliseconds should be suppressed
        let mut last: Option<Instant> = None;
        assert!(should_trigger(&mut last, 60_000), "first call triggers");
        assert!(
            !should_trigger(&mut last, 60_000),
            "second call suppressed within debounce window"
        );
    }
}
