//! Headless log sink for long-running dm modes (print / daemon / web / chain).
//!
//! Writes to `~/.dm/logs/<mode>-<YYYY-MM-DD>.log`. Rolls over on date change
//! (for processes that cross midnight) and at 50 MB per file via
//! `<mode>-<date>.<N>.log` numbered suffixes. Files older than 7 days are
//! pruned at init time.
//!
//! `DM_LOG_DIR` overrides the log directory (used by tests; also available
//! for sandboxed deployments).
//!
//! If `init` is never called, `log` is a silent noop — the TUI path uses
//! `BackendEvent` and does not need a log sink, so no spurious files are
//! created when dm runs interactively.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime};

pub const MAX_FILE_SIZE_BYTES: u64 = 50 * 1024 * 1024;
pub const RETENTION_DAYS: i64 = 7;
const LOG_DIR_ENV: &str = "DM_LOG_DIR";
const MAX_ROTATION_SUFFIX: u32 = 999;

struct Sink {
    writer: BufWriter<File>,
    mode: String,
    date: String,
    bytes_written: u64,
    dir: PathBuf,
}

static SINK: OnceLock<Mutex<Option<Sink>>> = OnceLock::new();

fn sink_cell() -> &'static Mutex<Option<Sink>> {
    SINK.get_or_init(|| Mutex::new(None))
}

/// Return the log directory (respects `DM_LOG_DIR` override, else `~/.dm/logs`).
/// Falls back to `./.dm/logs` if the home dir is unknown.
pub fn log_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var(LOG_DIR_ENV) {
        return PathBuf::from(override_dir);
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".dm").join("logs");
    }
    PathBuf::from(".dm").join("logs")
}

fn today_str() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

fn timestamp_str() -> String {
    chrono::Local::now().format("%H:%M:%S%.3f").to_string()
}

/// Compute the file path for a given (dir, mode, date). Starts at
/// `<mode>-<date>.log`; if that file is at or past `MAX_FILE_SIZE_BYTES`,
/// walks `.1.log`, `.2.log`, … up to `.999.log` looking for a slot that
/// is absent or still under cap.
pub fn compute_file_path(dir: &Path, mode: &str, date: &str) -> PathBuf {
    let bare = dir.join(format!("{mode}-{date}.log"));
    if !is_at_cap(&bare) {
        return bare;
    }
    for n in 1..=MAX_ROTATION_SUFFIX {
        let candidate = dir.join(format!("{mode}-{date}.{n}.log"));
        if !is_at_cap(&candidate) {
            return candidate;
        }
    }
    // Exhausted rotation slots — fall back to the last suffix. Caller will
    // keep appending; better than losing logs entirely.
    dir.join(format!("{mode}-{date}.{MAX_ROTATION_SUFFIX}.log"))
}

fn is_at_cap(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(m) => m.len() >= MAX_FILE_SIZE_BYTES,
        Err(_) => false,
    }
}

/// Prune `<mode>-<YYYY-MM-DD>[.N].log` files whose mtime is older than
/// `cutoff`. Returns the number of files removed. Ignores filenames that
/// don't match the dm log pattern so user-dropped files survive.
pub(crate) fn prune_older_than(dir: &Path, cutoff: SystemTime) -> std::io::Result<usize> {
    let mut removed = 0usize;
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    for entry in rd.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !is_dm_log_filename(&name) {
            continue;
        }
        let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if mtime < cutoff {
            if let Err(e) = std::fs::remove_file(&path) {
                crate::warnings::push_warning(format!(
                    "logging: failed to prune {}: {e}",
                    path.display()
                ));
            } else {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// Match `<mode>-<YYYY-MM-DD>.log` and `<mode>-<YYYY-MM-DD>.<N>.log`.
/// Conservative by design — don't delete arbitrary `.log` files.
fn is_dm_log_filename(name: &str) -> bool {
    if !name.ends_with(".log") {
        return false;
    }
    let stem = &name[..name.len() - 4];
    let Some(dash) = stem.find('-') else {
        return false;
    };
    let mode = &stem[..dash];
    if mode.is_empty() || !mode.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let rest = &stem[dash + 1..];
    // rest must start with YYYY-MM-DD (10 chars), optionally followed by .<digits>.
    if rest.len() < 10 {
        return false;
    }
    let date = &rest[..10];
    let bytes = date.as_bytes();
    let date_shape_ok = bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(|b| b.is_ascii_digit())
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[8..10].iter().all(|b| b.is_ascii_digit());
    if !date_shape_ok {
        return false;
    }
    let tail = &rest[10..];
    tail.is_empty()
        || (tail.starts_with('.')
            && tail.len() > 1
            && tail[1..].chars().all(|c| c.is_ascii_digit()))
}

fn open_sink(dir: &Path, mode: &str) -> std::io::Result<Sink> {
    std::fs::create_dir_all(dir)?;
    let date = today_str();
    let path = compute_file_path(dir, mode, &date);
    let file = OpenOptions::new().create(true).append(true).open(&path)?;
    let bytes_written = file.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(Sink {
        writer: BufWriter::new(file),
        mode: mode.to_string(),
        date,
        bytes_written,
        dir: dir.to_path_buf(),
    })
}

/// Initialize the headless log sink for `mode` ("print", "daemon", "web",
/// "chain"). Creates the log dir if missing, prunes files older than
/// `RETENTION_DAYS` by mtime, and opens the day's log file for append.
pub fn init(mode: &str) -> std::io::Result<()> {
    let dir = log_dir();
    std::fs::create_dir_all(&dir)?;
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(RETENTION_DAYS as u64 * 24 * 60 * 60))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    if let Err(e) = prune_older_than(&dir, cutoff) {
        crate::warnings::push_warning(format!("logging: prune failed: {e}"));
    }
    let sink = open_sink(&dir, mode)?;
    let mut guard = sink_cell().lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(sink);
    Ok(())
}

/// Append a timestamped line to the active log. Silent noop if `init` was
/// never called.
pub fn log(line: &str) {
    let mut guard = match sink_cell().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let Some(sink) = guard.as_mut() else {
        return;
    };

    let today = today_str();
    if today != sink.date || sink.bytes_written >= MAX_FILE_SIZE_BYTES {
        let dir = sink.dir.clone();
        let mode = sink.mode.clone();
        match open_sink(&dir, &mode) {
            Ok(new_sink) => *sink = new_sink,
            Err(e) => {
                crate::warnings::push_warning(format!("logging: rotation failed: {e}"));
            }
        }
    }

    let ts = timestamp_str();
    let written = writeln!(sink.writer, "[{ts}] {line}").and_then(|_| sink.writer.flush());
    if let Err(e) = written {
        crate::warnings::push_warning(format!("logging: write failed: {e}"));
        return;
    }
    sink.bytes_written = sink
        .bytes_written
        .saturating_add(ts.len() as u64 + line.len() as u64 + 4);
}

/// Convenience for error-level entries. Prefixes "ERROR " so log scrapers
/// can `grep ^\[.*\] ERROR` for fast filtering.
pub fn log_err(line: &str) {
    log(&format!("ERROR {line}"));
}

/// Flush the active sink. Call before clean shutdown.
#[allow(dead_code)]
pub fn flush() {
    let mut guard = match sink_cell().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if let Some(sink) = guard.as_mut() {
        if let Err(e) = sink.writer.flush() {
            crate::warnings::push_warning(format!("logging: flush failed: {e}"));
        }
    }
}

/// Drop the active sink (closes the file). Idempotent. Used at shutdown and
/// when the Phase 2 daemon hot-reloads a new mode.
#[allow(dead_code)]
pub fn shutdown() {
    let mut guard = match sink_cell().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    *guard = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    struct EnvGuard {
        prev: Option<String>,
    }

    impl EnvGuard {
        fn set(dir: &Path) -> Self {
            let prev = std::env::var(LOG_DIR_ENV).ok();
            std::env::set_var(LOG_DIR_ENV, dir);
            Self { prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(LOG_DIR_ENV, v),
                None => std::env::remove_var(LOG_DIR_ENV),
            }
            shutdown();
        }
    }

    #[test]
    fn init_creates_log_dir_and_opens_file() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        let sub = tmp.path().join("nested");
        let _env = EnvGuard::set(&sub);

        init("print").expect("init");
        shutdown();

        assert!(sub.exists(), "log dir should have been created");
        let today = today_str();
        let expected = sub.join(format!("print-{today}.log"));
        assert!(expected.exists(), "expected file {expected:?}");
    }

    #[test]
    fn log_writes_timestamped_line_matching_pattern() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        let _env = EnvGuard::set(tmp.path());

        init("daemon").expect("init");
        log("hello world");
        flush();
        shutdown();

        let today = today_str();
        let path = tmp.path().join(format!("daemon-{today}.log"));
        let body = std::fs::read_to_string(&path).expect("read log");
        assert!(body.contains("hello world"), "missing payload in {body:?}");
        assert!(body.contains('['), "missing timestamp bracket in {body:?}");
        assert!(body.contains(']'), "missing timestamp close in {body:?}");
        assert!(body.ends_with('\n'), "missing trailing newline in {body:?}");
    }

    #[test]
    fn log_without_init_is_silent_noop() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();
        let _env = EnvGuard::set(tmp.path());
        shutdown();

        log("no sink");
        log_err("also no sink");
        flush();

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(
            entries.is_empty(),
            "noop must not create files, got {entries:?}"
        );
    }

    #[test]
    fn compute_file_path_uses_bare_name_when_absent_or_under_cap() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();

        let absent = compute_file_path(tmp.path(), "daemon", "2026-04-18");
        assert_eq!(absent, tmp.path().join("daemon-2026-04-18.log"));

        let under_cap = tmp.path().join("daemon-2026-04-18.log");
        std::fs::write(&under_cap, vec![0u8; 1000]).unwrap();
        let still_bare = compute_file_path(tmp.path(), "daemon", "2026-04-18");
        assert_eq!(
            still_bare, under_cap,
            "under-cap file should keep bare name"
        );
    }

    #[test]
    fn compute_file_path_rotates_to_numbered_suffix_when_cap_exceeded() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();

        let bare = tmp.path().join("daemon-2026-04-18.log");
        let f = File::create(&bare).unwrap();
        f.set_len(MAX_FILE_SIZE_BYTES + 1).unwrap();
        drop(f);

        let next = compute_file_path(tmp.path(), "daemon", "2026-04-18");
        assert_eq!(next, tmp.path().join("daemon-2026-04-18.1.log"));

        let suffix_one = tmp.path().join("daemon-2026-04-18.1.log");
        let f1 = File::create(&suffix_one).unwrap();
        f1.set_len(MAX_FILE_SIZE_BYTES + 1).unwrap();
        drop(f1);

        let after = compute_file_path(tmp.path(), "daemon", "2026-04-18");
        assert_eq!(after, tmp.path().join("daemon-2026-04-18.2.log"));
    }

    #[test]
    fn prune_older_than_removes_files_past_cutoff_and_spares_others() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = tempfile::TempDir::new().unwrap();

        let log_a = tmp.path().join("print-2026-04-10.log");
        let log_b = tmp.path().join("daemon-2026-04-18.log");
        let notes = tmp.path().join("notes.txt");
        let unrelated_log = tmp.path().join("random.log");
        for p in [&log_a, &log_b, &notes, &unrelated_log] {
            std::fs::write(p, b"x").unwrap();
        }

        let future_cutoff = SystemTime::now() + Duration::from_secs(60);
        let removed = prune_older_than(tmp.path(), future_cutoff).unwrap();

        assert_eq!(removed, 2, "should prune both matching log files");
        assert!(!log_a.exists(), "print-* log should be gone");
        assert!(!log_b.exists(), "daemon-* log should be gone");
        assert!(notes.exists(), "non-log file must be preserved");
        assert!(
            unrelated_log.exists(),
            "unrelated .log without date pattern must be preserved"
        );
    }

    #[test]
    fn is_dm_log_filename_accepts_and_rejects_correct_patterns() {
        assert!(is_dm_log_filename("print-2026-04-18.log"));
        assert!(is_dm_log_filename("daemon-2026-04-18.12.log"));
        assert!(is_dm_log_filename("web-2026-04-18.999.log"));

        assert!(!is_dm_log_filename("print-2026-04-18.txt"));
        assert!(!is_dm_log_filename("random.log"));
        assert!(!is_dm_log_filename("print-2026-04.log"));
        assert!(!is_dm_log_filename("print-2026-04-18.log.bak"));
        assert!(!is_dm_log_filename("-2026-04-18.log"));
        assert!(!is_dm_log_filename("print1-2026-04-18.log"));
    }
}
