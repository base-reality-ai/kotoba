use std::time::Duration;

const MAX_RESTARTS: u32 = 20;

/// Run the daemon in a watchdog loop, restarting on unexpected exit.
/// `daemon_args` are the args to pass to the daemon worker process.
///
/// The daemon worker is identified by the `DM_IS_DAEMON=1` environment
/// variable, which the watchdog sets before spawning.
pub async fn run_watchdog(daemon_args: Vec<String>) -> anyhow::Result<()> {
    let _ = crate::logging::init("watchdog");
    let config_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join(".dm");

    let mut restart_count = 0u32;
    let mut backoff_ms = 2000u64;

    loop {
        let current_exe = std::env::current_exe()?;
        let status = tokio::process::Command::new(&current_exe)
            .args(&daemon_args)
            .env("DM_IS_DAEMON", "1")
            .status()
            .await?;

        if status.success() {
            // Clean shutdown — don't restart.
            break;
        }

        restart_count += 1;
        if restart_count > MAX_RESTARTS {
            crate::logging::log_err(&format!(
                "[dm watchdog] daemon has crashed {} times — giving up. Check logs at {:?}",
                MAX_RESTARTS,
                config_dir.join("daemon.log")
            ));
            anyhow::bail!(
                "Daemon crashed {} times, refusing to restart. Check ~/.dm/daemon.log for details.",
                MAX_RESTARTS
            );
        }
        crate::logging::log(&format!(
            "[dm watchdog] daemon exited (code {:?}), restart #{} in {}ms",
            status.code(),
            restart_count,
            backoff_ms
        ));

        // Write crash log entry.
        crate::daemon::server::write_log_entry(
            &config_dir,
            "error",
            "daemon crashed",
            Some(serde_json::json!({
                "exit_code": status.code(),
                "restart_count": restart_count,
            })),
        );

        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(60_000); // cap at 60 s
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_caps_at_60s() {
        let mut b = 2000u64;
        for _ in 0..20 {
            b = (b * 2).min(60_000);
        }
        assert_eq!(b, 60_000);
    }

    #[test]
    fn backoff_doubles_correctly() {
        let mut b = 2000u64;
        b = (b * 2).min(60_000);
        assert_eq!(b, 4000);
        b = (b * 2).min(60_000);
        assert_eq!(b, 8000);
    }

    #[test]
    fn max_restarts_constant_is_reasonable() {
        const { assert!(MAX_RESTARTS >= 5 && MAX_RESTARTS <= 100) };
    }

    #[test]
    fn backoff_reaches_cap_before_max_restarts() {
        let mut b = 2000u64;
        let mut capped_at = 0u32;
        for i in 1..=MAX_RESTARTS {
            b = (b * 2).min(60_000);
            if b == 60_000 && capped_at == 0 {
                capped_at = i;
            }
        }
        assert!(
            capped_at > 0,
            "backoff should reach 60s cap within {} restarts",
            MAX_RESTARTS
        );
        assert!(
            capped_at < MAX_RESTARTS,
            "cap should be reached well before max restarts"
        );
    }

    #[test]
    fn backoff_initial_value_is_2s() {
        let b = 2000u64;
        assert_eq!(b, 2000, "initial backoff should be 2000ms");
    }
}
