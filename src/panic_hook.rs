//! Panic-time diagnostic logger.
//!
//! Installs a global panic hook that writes a structured log record to
//! `${config_dir}/crashes.log`, prints a user-readable recovery hint to
//! stderr, and chains to the previously-installed hook so bug-reporting
//! (backtraces via `RUST_BACKTRACE=1`) still works for developers.
//!
//! Scoped deliberately: does NOT attempt to persist in-memory session
//! state (that requires mid-turn autosave — Part 2). The promise here is
//! diagnostic fidelity, not state recovery.

use arc_swap::ArcSwapOption;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::panic::PanicHookInfo;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

static INSTALLED: OnceLock<()> = OnceLock::new();

/// Process-global snapshot of the currently-streaming model turn. Set by
/// `conversation.rs` right before the token loop starts, updated per token,
/// cleared at every exit path. The panic hook reads this to attach partial
/// text to the marker so a crash mid-turn isn't indistinguishable from a
/// crash at idle. Lock-free so the panic handler never deadlocks on a
/// partially-held lock.
static TURN_IN_FLIGHT: OnceLock<ArcSwapOption<TurnInFlight>> = OnceLock::new();

fn turn_in_flight_cell() -> &'static ArcSwapOption<TurnInFlight> {
    TURN_IN_FLIGHT.get_or_init(ArcSwapOption::empty)
}

/// Install `turn` as the in-flight snapshot, replacing any previous entry.
/// Called by the streaming loop immediately before the first token.
pub fn set_turn_in_flight(turn: TurnInFlight) {
    turn_in_flight_cell().store(Some(Arc::new(turn)));
}

/// Append `tok` to the partial text of the in-flight turn, if one is set.
/// No-op when no turn is currently in flight (e.g. called from a caller
/// that hasn't wired `set_turn_in_flight` yet).
pub fn update_turn_partial(tok: &str) {
    let cell = turn_in_flight_cell();
    if let Some(current) = cell.load_full() {
        let mut next = (*current).clone();
        next.partial_text.push_str(tok);
        cell.store(Some(Arc::new(next)));
    }
}

/// Clear the in-flight snapshot. Called at every stream exit path (done,
/// error, timeout) so a crash at idle produces a marker with `in_flight: None`.
pub fn clear_turn_in_flight() {
    turn_in_flight_cell().store(None);
}

/// Read the current in-flight snapshot. Returns `None` when no turn is
/// active or the cell was never initialized.
pub fn current_turn_in_flight() -> Option<Arc<TurnInFlight>> {
    turn_in_flight_cell().load_full()
}

#[derive(Clone, Debug)]
pub struct CrashContext {
    pub config_dir: PathBuf,
    pub session_id: String,
    pub binary_version: &'static str,
}

impl CrashContext {
    /// Context for entry points with no persistent session (doctor, init,
    /// daemon status pings, etc.). Uses a placeholder `session_id` so the
    /// log line is still parseable; per-invocation attribution comes from
    /// the timestamp.
    pub fn transient(config_dir: PathBuf) -> Self {
        Self {
            config_dir,
            session_id: "transient".to_string(),
            binary_version: env!("CARGO_PKG_VERSION"),
        }
    }

    /// Context for entry points that own a single session (TUI, headless
    /// one-shot). `session_id` is what appears in both `crashes.log` and
    /// the stderr recovery message — must be resumable via `dm --resume`.
    pub fn for_session(config_dir: PathBuf, session_id: String) -> Self {
        Self {
            config_dir,
            session_id,
            binary_version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// Install the panic hook exactly once per process. A second call is a no-op
/// so tests or re-entrant startup paths can call it safely.
///
/// `restore_terminal` runs first inside the hook so the crash message isn't
/// eaten by the alternate-screen buffer. Non-TUI callers can pass `|| {}`.
pub fn install<R>(ctx: CrashContext, restore_terminal: R)
where
    R: Fn() + Send + Sync + 'static,
{
    if INSTALLED.set(()).is_err() {
        return;
    }
    let prior = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = extract_panic_message(info);
        let location = extract_panic_location(info);
        run_panic_hook_actions(
            &msg,
            location.as_deref(),
            &ctx,
            Utc::now(),
            &restore_terminal,
            |buf| std::io::stderr().write_all(buf),
            append_line,
            write_panic_marker,
        );
        prior(info);
    }));
}

/// Ordered side effects of a panic: (1) restore the terminal, (2) append a
/// structured line to `crashes.log`, (3) write a human-readable message to
/// stderr, (4) write a structured marker file for non-transient sessions.
/// Each step is best-effort — a broken pipe on stderr or a permission-denied
/// on the log must not mask the original panic. Extracted as a generic
/// helper so tests can pin the ordering and swallowing without touching
/// real files or raw terminals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_panic_hook_actions<R, W, L, M>(
    message: &str,
    location: Option<&str>,
    ctx: &CrashContext,
    now: DateTime<Utc>,
    restore_terminal: R,
    mut write_stderr: W,
    mut append_log: L,
    mut write_marker: M,
) where
    R: Fn(),
    W: FnMut(&[u8]) -> std::io::Result<()>,
    L: FnMut(&Path, &str) -> std::io::Result<()>,
    M: FnMut(&Path, &PanicMarker) -> std::io::Result<PathBuf>,
{
    restore_terminal();

    let log_path = crash_log_path(&ctx.config_dir);
    let log_line = format_crash_log_line(message, location, ctx, now);
    let _ = append_log(&log_path, &log_line);

    let user_msg = format_user_message(message, location, ctx, &log_path);
    let _ = write_stderr(user_msg.as_bytes());

    // Transient contexts (doctor, daemon pings) have no resumable session,
    // so a marker would be noise. Skip.
    if ctx.session_id != "transient" {
        let marker = PanicMarker {
            session_id: ctx.session_id.clone(),
            binary_version: ctx.binary_version.to_string(),
            panicked_at: now.to_rfc3339(),
            message: message.to_string(),
            location: location.map(str::to_string),
            in_flight: current_turn_in_flight().map(|arc| (*arc).clone()),
        };
        let dir = marker_dir(&ctx.config_dir);
        let _ = write_marker(&dir, &marker);
    }
}

/// Structured record of a panic, written alongside `crashes.log` so
/// watchdogs and future `dm --recovery` UX can enumerate unrecovered
/// sessions without re-parsing the log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PanicMarker {
    pub session_id: String,
    pub binary_version: String,
    /// RFC 3339 timestamp — lexicographic sort doubles as chronological sort.
    pub panicked_at: String,
    pub message: String,
    pub location: Option<String>,
    /// Snapshot of the in-flight model turn at the moment of panic, if any.
    /// `default`+`skip_serializing_if = None` keeps pre-C75 JSON readable and
    /// keeps empty markers terse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight: Option<TurnInFlight>,
}

/// Snapshot of a streaming model turn. Captured into `PanicMarker::in_flight`
/// so a crash mid-stream leaves the partial assistant text recoverable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnInFlight {
    pub session_id: String,
    pub partial_text: String,
    /// RFC 3339 timestamp of when the stream began.
    pub started_at: String,
}

/// Directory beneath `config_dir` where marker files live. Kept separate
/// from `crashes.log` so a consumer can list markers with a single
/// `read_dir` instead of tail-parsing the log.
pub fn marker_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("panic_markers")
}

/// Persist `marker` into `dir` as `{session_id}-{unix_ts}.json`. The unix
/// suffix lets multiple crashes of the same session coexist without
/// clobbering each other. Best-effort: callers should ignore errors so a
/// broken disk never masks the original panic.
pub fn write_panic_marker(dir: &Path, marker: &PanicMarker) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let ts_suffix = DateTime::parse_from_rfc3339(&marker.panicked_at)
        .map(|dt| dt.timestamp())
        .unwrap_or(0);
    let fname = format!("{}-{}.json", marker.session_id, ts_suffix);
    let path = dir.join(fname);
    let json = serde_json::to_string_pretty(marker)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Enumerate marker files under `config_dir`, newest-first by
/// `panicked_at`. Malformed JSON is silently skipped so one bad file does
/// not hide the rest. Missing directory yields an empty vec (first-run
/// friendly, no `Err` noise at startup).
pub fn list_panic_markers(config_dir: &Path) -> Vec<(PathBuf, PanicMarker)> {
    let dir = marker_dir(config_dir);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<(PathBuf, PanicMarker)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(contents) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(marker) = serde_json::from_str::<PanicMarker>(&contents) else {
            continue;
        };
        out.push((path, marker));
    }
    out.sort_by(|a, b| b.1.panicked_at.cmp(&a.1.panicked_at));
    out
}

/// Remove a marker file after its session has been resumed or dismissed.
pub fn clear_panic_marker(path: &Path) -> std::io::Result<()> {
    std::fs::remove_file(path)
}

/// Format a user-readable recovery report from the marker list. Pure — no
/// I/O — so tests and `--recovery` share a single canonical rendering.
///
/// Expects `markers` in the order `list_panic_markers` returns them
/// (newest-first). Ends with a dismiss hint so "Error messages include
/// next steps" still holds when the caller has nothing to resume.
pub fn format_recovery_report(markers: &[(PathBuf, PanicMarker)]) -> String {
    if markers.is_empty() {
        return "No unrecovered crashes found.".to_string();
    }
    let mut lines = Vec::new();
    lines.push(format!("{} unrecovered crash(es):", markers.len()));
    lines.push(String::new());
    for (i, (_path, marker)) in markers.iter().enumerate() {
        lines.push(format!("  {}. session {}", i + 1, marker.session_id));
        lines.push(format!("     crashed at {}", marker.panicked_at));
        lines.push(format!("     message:    {}", marker.message));
        if let Some(loc) = &marker.location {
            lines.push(format!("     location:   {}", loc));
        }
        lines.push(format!(
            "     resume:     dm --resume {}",
            marker.session_id
        ));
        lines.push(String::new());
    }
    lines.push("To dismiss without resuming: rm the file under ~/.dm/panic_markers/".to_string());
    lines.push("Or clear all:                rm -rf ~/.dm/panic_markers/".to_string());
    lines.join("\n")
}

/// Build a one-line banner for the TUI scroll buffer when unrecovered
/// crashes exist. `None` on empty input lets the caller skip pushing.
/// Deliberately terse — details live in `dm --recovery`. Singular and
/// plural wording is kept internal so the call site is a dumb nudge.
///
/// This is a nudge, not a dump: it must NOT leak session ids or panic
/// messages into the always-visible scroll buffer. Tests pin both.
pub fn build_recovery_banner(markers: &[(PathBuf, PanicMarker)]) -> Option<String> {
    match markers.len() {
        0 => None,
        1 => Some(
            "1 unrecovered session from a previous crash. Run `dm --recovery` for details."
                .to_string(),
        ),
        n => Some(format!(
            "{n} unrecovered sessions from previous crashes. Run `dm --recovery` for details."
        )),
    }
}

/// Remove every marker whose `session_id` matches, returning the count
/// cleared. Best-effort: per-marker removal errors are swallowed so a
/// single permission-denied file doesn't strand the rest.
pub fn clear_panic_markers_for_session(config_dir: &Path, session_id: &str) -> usize {
    let markers = list_panic_markers(config_dir);
    let mut cleared = 0usize;
    for (path, marker) in markers {
        if marker.session_id == session_id && clear_panic_marker(&path).is_ok() {
            cleared += 1;
        }
    }
    cleared
}

pub fn crash_log_path(config_dir: &Path) -> PathBuf {
    config_dir.join("crashes.log")
}

/// Extract the panic payload as a string. Payloads can be `&str`, `String`,
/// or arbitrary `Box<dyn Any>` — handle all three so the log never loses a
/// message just because the source used `panic_any!` with a custom type.
pub fn extract_panic_message(info: &PanicHookInfo) -> String {
    let payload = info.payload();
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_string()
}

pub fn extract_panic_location(info: &PanicHookInfo) -> Option<String> {
    info.location()
        .map(|loc| format!("{}:{}", loc.file(), loc.line()))
}

/// Single-line structured log record. Single-line is load-bearing so that
/// `tail -1` / `grep` over `crashes.log` returns complete records even under
/// concurrent writes from multiple dm processes.
pub fn format_crash_log_line(
    message: &str,
    location: Option<&str>,
    ctx: &CrashContext,
    now: DateTime<Utc>,
) -> String {
    let msg_one_line = message.replace('\n', " ⏎ ").replace('\r', "");
    let loc = location.unwrap_or("unknown");
    format!(
        "[{}] v{} session={} at={} msg={}\n",
        now.to_rfc3339(),
        ctx.binary_version,
        ctx.session_id,
        loc,
        msg_one_line,
    )
}

/// Human-readable recovery message written to stderr. Names the cause, the
/// affected session, where to find the log, how to resume, and where to
/// report — "Error messages include next steps" per Prime Directive.
pub fn format_user_message(
    message: &str,
    location: Option<&str>,
    ctx: &CrashContext,
    log_path: &Path,
) -> String {
    let loc = location.unwrap_or("unknown");
    format!(
        "\n\
         dm encountered an internal error. Your session has been preserved up to the last turn.\n\
         \n  \
         Session: {}\n  \
         Reason:  {}\n  \
         Where:   {}\n  \
         Log:     {}\n\
         \n\
         Resume with: dm --resume {}\n\
         Report this at: https://github.com/.../issues (include the log line above).\n\
         \n",
        ctx.session_id,
        message,
        loc,
        log_path.display(),
        ctx.session_id,
    )
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(line.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> CrashContext {
        CrashContext {
            config_dir: PathBuf::from("/tmp/dm-test"),
            session_id: "8e3f1a2b".to_string(),
            binary_version: "4.0.0",
        }
    }

    fn ts() -> DateTime<Utc> {
        "2026-04-18T12:34:56Z".parse().unwrap()
    }

    #[test]
    fn log_line_is_single_line() {
        let s = format_crash_log_line(
            "thread 'main' panicked\nwith newline",
            Some("src/foo.rs:10"),
            &ctx(),
            ts(),
        );
        assert_eq!(s.matches('\n').count(), 1, "trailing newline only: {s}");
        assert!(!s[..s.len() - 1].contains('\n'));
    }

    #[test]
    fn log_line_preserves_newlines_as_marker() {
        // Multi-line panic messages must still be findable via grep on one line.
        let s = format_crash_log_line("line1\nline2", None, &ctx(), ts());
        assert!(s.contains("line1 ⏎ line2"));
    }

    #[test]
    fn log_line_includes_timestamp_version_session_location() {
        let s = format_crash_log_line("boom", Some("src/foo.rs:10"), &ctx(), ts());
        assert!(s.contains("2026-04-18T12:34:56"), "timestamp: {s}");
        assert!(s.contains("v4.0.0"), "version: {s}");
        assert!(s.contains("session=8e3f1a2b"), "session id: {s}");
        assert!(s.contains("at=src/foo.rs:10"), "location: {s}");
        assert!(s.contains("msg=boom"), "message: {s}");
    }

    #[test]
    fn log_line_with_unknown_location() {
        let s = format_crash_log_line("boom", None, &ctx(), ts());
        assert!(s.contains("at=unknown"));
    }

    #[test]
    fn user_message_has_actionable_next_steps() {
        let msg = format_user_message(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            Path::new("/tmp/dm-test/crashes.log"),
        );
        assert!(msg.contains("Session: 8e3f1a2b"), "names session: {msg}");
        assert!(msg.contains("Reason:  boom"), "names reason: {msg}");
        assert!(
            msg.contains("Where:   src/foo.rs:10"),
            "names location: {msg}"
        );
        assert!(
            msg.contains("dm --resume 8e3f1a2b"),
            "shows recover cmd: {msg}"
        );
        assert!(msg.contains("crashes.log"), "points to log: {msg}");
        assert!(
            msg.contains("Report this"),
            "points to issue tracker: {msg}"
        );
    }

    #[test]
    fn user_message_handles_unknown_location() {
        let msg = format_user_message("boom", None, &ctx(), Path::new("/tmp/crashes.log"));
        assert!(msg.contains("Where:   unknown"));
    }

    #[test]
    fn crash_log_path_joins_config_dir() {
        let p = crash_log_path(Path::new("/home/user/.dm"));
        assert_eq!(p, PathBuf::from("/home/user/.dm/crashes.log"));
    }

    #[test]
    fn append_line_creates_parent_and_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nested/dir/crashes.log");
        append_line(&path, "line one\n").unwrap();
        append_line(&path, "line two\n").unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "line one\nline two\n");
    }

    use std::cell::RefCell;

    #[derive(Default)]
    struct ActionLog {
        events: RefCell<Vec<&'static str>>,
        stderr_buf: RefCell<Vec<u8>>,
        log_writes: RefCell<Vec<(PathBuf, String)>>,
    }

    #[test]
    fn actions_order_restore_then_log_then_stderr() {
        let log = ActionLog::default();
        run_panic_hook_actions(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            ts(),
            || log.events.borrow_mut().push("restore"),
            |buf| {
                log.events.borrow_mut().push("stderr");
                log.stderr_buf.borrow_mut().extend_from_slice(buf);
                Ok(())
            },
            |path, line| {
                log.events.borrow_mut().push("log");
                log.log_writes
                    .borrow_mut()
                    .push((path.to_path_buf(), line.to_string()));
                Ok(())
            },
            |_, _| Ok(PathBuf::new()),
        );
        assert_eq!(*log.events.borrow(), vec!["restore", "log", "stderr"]);
    }

    #[test]
    fn actions_threads_crash_log_path_into_log_and_stderr() {
        let log = ActionLog::default();
        run_panic_hook_actions(
            "boom",
            None,
            &ctx(),
            ts(),
            || {},
            |buf| {
                log.stderr_buf.borrow_mut().extend_from_slice(buf);
                Ok(())
            },
            |path, line| {
                log.log_writes
                    .borrow_mut()
                    .push((path.to_path_buf(), line.to_string()));
                Ok(())
            },
            |_, _| Ok(PathBuf::new()),
        );
        let expected_path = PathBuf::from("/tmp/dm-test/crashes.log");
        let writes = log.log_writes.borrow();
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, expected_path);
        let stderr_text = String::from_utf8(log.stderr_buf.borrow().clone()).unwrap();
        assert!(
            stderr_text.contains("/tmp/dm-test/crashes.log"),
            "stderr names log: {stderr_text}"
        );
    }

    #[test]
    fn actions_swallow_stderr_write_error() {
        let log = ActionLog::default();
        run_panic_hook_actions(
            "boom",
            None,
            &ctx(),
            ts(),
            || log.events.borrow_mut().push("restore"),
            |_| Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe)),
            |_, _| {
                log.events.borrow_mut().push("log");
                Ok(())
            },
            |_, _| Ok(PathBuf::new()),
        );
        // Did not panic — and restore + log still ran.
        assert_eq!(*log.events.borrow(), vec!["restore", "log"]);
    }

    #[test]
    fn actions_swallow_log_append_error() {
        let log = ActionLog::default();
        run_panic_hook_actions(
            "boom",
            None,
            &ctx(),
            ts(),
            || log.events.borrow_mut().push("restore"),
            |_| {
                log.events.borrow_mut().push("stderr");
                Ok(())
            },
            |_, _| Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
            |_, _| Ok(PathBuf::new()),
        );
        // Log failure must not stop the stderr write — user still sees recovery text.
        assert_eq!(*log.events.borrow(), vec!["restore", "stderr"]);
    }

    #[test]
    fn crash_context_transient_uses_placeholder_session_id() {
        let ctx = CrashContext::transient(PathBuf::from("/home/u/.dm"));
        assert_eq!(ctx.config_dir, PathBuf::from("/home/u/.dm"));
        assert_eq!(ctx.session_id, "transient");
    }

    #[test]
    fn crash_context_for_session_uses_provided_id() {
        let ctx = CrashContext::for_session(
            PathBuf::from("/home/u/.dm"),
            "8e3f1a2b-1234-4567-89ab-cdef01234567".to_string(),
        );
        assert_eq!(ctx.session_id, "8e3f1a2b-1234-4567-89ab-cdef01234567");
    }

    #[test]
    fn crash_context_constructors_set_binary_version_from_env() {
        let t = CrashContext::transient(PathBuf::from("/tmp"));
        let s = CrashContext::for_session(PathBuf::from("/tmp"), "x".to_string());
        assert_eq!(t.binary_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(s.binary_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(t.binary_version, s.binary_version);
    }

    #[test]
    fn actions_stderr_content_matches_format_user_message() {
        let log = ActionLog::default();
        run_panic_hook_actions(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            ts(),
            || {},
            |buf| {
                log.stderr_buf.borrow_mut().extend_from_slice(buf);
                Ok(())
            },
            |_, _| Ok(()),
            |_, _| Ok(PathBuf::new()),
        );
        let stderr_text = String::from_utf8(log.stderr_buf.borrow().clone()).unwrap();
        let expected = format_user_message(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            &crash_log_path(&ctx().config_dir),
        );
        assert_eq!(stderr_text, expected);
    }

    // ── PanicMarker + recovery API ─────────────────────────────────────────

    fn sample_marker(session: &str, panicked_at: &str) -> PanicMarker {
        PanicMarker {
            session_id: session.to_string(),
            binary_version: "4.0.0".to_string(),
            panicked_at: panicked_at.to_string(),
            message: "boom".to_string(),
            location: Some("src/foo.rs:10".to_string()),
            in_flight: None,
        }
    }

    #[test]
    fn write_panic_marker_creates_dir_and_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("panic_markers");
        let marker = sample_marker("abc123", "2026-04-18T12:34:56+00:00");
        let path = write_panic_marker(&dir, &marker).unwrap();
        assert!(dir.is_dir(), "marker dir created: {}", dir.display());
        assert!(path.is_file(), "marker file created: {}", path.display());
        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: PanicMarker = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed, marker);
    }

    #[test]
    fn write_panic_marker_filename_uses_session_and_timestamp() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("panic_markers");
        let marker = sample_marker("sess42", "2026-04-18T12:34:56+00:00");
        let path = write_panic_marker(&dir, &marker).unwrap();
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(fname.starts_with("sess42-"), "session prefix: {fname}");
        assert!(fname.ends_with(".json"), "json suffix: {fname}");
        let expected_ts = DateTime::parse_from_rfc3339("2026-04-18T12:34:56+00:00")
            .unwrap()
            .timestamp()
            .to_string();
        assert!(fname.contains(&expected_ts), "unix ts in name: {fname}");
    }

    #[test]
    fn list_panic_markers_returns_newest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = marker_dir(tmp.path());
        let cases = [
            ("old", "2026-04-01T00:00:00+00:00"),
            ("new", "2026-04-18T00:00:00+00:00"),
            ("mid", "2026-04-10T00:00:00+00:00"),
        ];
        for (sess, when) in cases {
            write_panic_marker(&dir, &sample_marker(sess, when)).unwrap();
        }
        let listed = list_panic_markers(tmp.path());
        let ids: Vec<&str> = listed.iter().map(|(_, m)| m.session_id.as_str()).collect();
        assert_eq!(ids, vec!["new", "mid", "old"]);
    }

    #[test]
    fn list_panic_markers_empty_when_dir_missing() {
        let tmp = tempfile::tempdir().unwrap();
        // Note: no panic_markers/ subdir created under tmp.
        let listed = list_panic_markers(tmp.path());
        assert!(listed.is_empty());
    }

    #[test]
    fn list_panic_markers_skips_malformed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = marker_dir(tmp.path());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("garbage.json"), "{ not valid json").unwrap();
        std::fs::write(dir.join("empty.json"), "").unwrap();
        write_panic_marker(&dir, &sample_marker("good", "2026-04-18T00:00:00+00:00")).unwrap();
        let listed = list_panic_markers(tmp.path());
        assert_eq!(listed.len(), 1, "only the well-formed marker survives");
        assert_eq!(listed[0].1.session_id, "good");
    }

    #[test]
    fn clear_panic_marker_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = marker_dir(tmp.path());
        let path = write_panic_marker(&dir, &sample_marker("resumed", "2026-04-18T00:00:00+00:00"))
            .unwrap();
        assert!(path.exists());
        clear_panic_marker(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn run_panic_hook_actions_writes_marker_for_session_ctx() {
        let writes: RefCell<Vec<(PathBuf, PanicMarker)>> = RefCell::new(Vec::new());
        run_panic_hook_actions(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            ts(),
            || {},
            |_| Ok(()),
            |_, _| Ok(()),
            |dir, marker| {
                writes
                    .borrow_mut()
                    .push((dir.to_path_buf(), marker.clone()));
                Ok(dir.join("stub.json"))
            },
        );
        let w = writes.borrow();
        assert_eq!(w.len(), 1, "one marker written for real session");
        assert_eq!(w[0].0, PathBuf::from("/tmp/dm-test/panic_markers"));
        assert_eq!(w[0].1.session_id, "8e3f1a2b");
        assert_eq!(w[0].1.binary_version, "4.0.0");
        assert_eq!(w[0].1.message, "boom");
        assert_eq!(w[0].1.location, Some("src/foo.rs:10".to_string()));
        assert!(
            w[0].1.panicked_at.contains("2026-04-18T12:34:56"),
            "panicked_at carries the panic timestamp: {}",
            w[0].1.panicked_at
        );
    }

    #[test]
    fn run_panic_hook_actions_skips_marker_for_transient_ctx() {
        let transient = CrashContext::transient(PathBuf::from("/tmp/dm-test"));
        let writes: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
        run_panic_hook_actions(
            "boom",
            None,
            &transient,
            ts(),
            || {},
            |_| Ok(()),
            |_, _| Ok(()),
            |dir, _| {
                writes.borrow_mut().push(dir.to_path_buf());
                Ok(dir.join("stub.json"))
            },
        );
        assert!(
            writes.borrow().is_empty(),
            "transient ctx must not produce markers"
        );
    }

    // ── Recovery report + session-scoped clear ─────────────────────────────

    #[test]
    fn format_recovery_report_empty_is_no_crashes_message() {
        let report = format_recovery_report(&[]);
        assert!(
            report.contains("No unrecovered crashes"),
            "empty report: {report}"
        );
    }

    #[test]
    fn format_recovery_report_single_marker_has_all_fields() {
        let m = sample_marker("sess42", "2026-04-18T12:34:56+00:00");
        let report = format_recovery_report(&[(PathBuf::from("/tmp/sess42-0.json"), m)]);
        assert!(report.contains("sess42"), "session id: {report}");
        assert!(
            report.contains("2026-04-18T12:34:56"),
            "timestamp: {report}"
        );
        assert!(report.contains("boom"), "message: {report}");
        assert!(report.contains("src/foo.rs:10"), "location: {report}");
        assert!(
            report.contains("dm --resume sess42"),
            "resume hint: {report}"
        );
    }

    #[test]
    fn format_recovery_report_numbers_multiple() {
        let markers: Vec<(PathBuf, PanicMarker)> = vec![
            (
                PathBuf::from("/a"),
                sample_marker("a", "2026-04-18T00:00:03+00:00"),
            ),
            (
                PathBuf::from("/b"),
                sample_marker("b", "2026-04-18T00:00:02+00:00"),
            ),
            (
                PathBuf::from("/c"),
                sample_marker("c", "2026-04-18T00:00:01+00:00"),
            ),
        ];
        let report = format_recovery_report(&markers);
        let idx_1 = report.find("1.").expect("has '1.'");
        let idx_2 = report.find("2.").expect("has '2.'");
        let idx_3 = report.find("3.").expect("has '3.'");
        assert!(
            idx_1 < idx_2 && idx_2 < idx_3,
            "numbered in order: {report}"
        );
        assert!(report.contains("3 unrecovered"), "header count: {report}");
    }

    #[test]
    fn format_recovery_report_omits_location_line_when_none() {
        let mut m = sample_marker("sess", "2026-04-18T00:00:00+00:00");
        m.location = None;
        let report = format_recovery_report(&[(PathBuf::from("/tmp/x.json"), m)]);
        assert!(
            !report.contains("location:"),
            "no 'location:' line when None: {report}"
        );
        assert!(
            report.contains("dm --resume sess"),
            "still has resume hint: {report}"
        );
    }

    #[test]
    fn format_recovery_report_includes_dismiss_hint() {
        let m = sample_marker("sess", "2026-04-18T00:00:00+00:00");
        let report = format_recovery_report(&[(PathBuf::from("/tmp/x.json"), m)]);
        assert!(
            report.contains("rm"),
            "dismiss hint with rm instruction: {report}"
        );
    }

    #[test]
    fn clear_panic_markers_for_session_removes_only_matching() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = marker_dir(tmp.path());
        write_panic_marker(&dir, &sample_marker("a", "2026-04-18T00:00:01+00:00")).unwrap();
        write_panic_marker(&dir, &sample_marker("a", "2026-04-18T00:00:02+00:00")).unwrap();
        write_panic_marker(&dir, &sample_marker("b", "2026-04-18T00:00:03+00:00")).unwrap();

        let n = clear_panic_markers_for_session(tmp.path(), "a");
        assert_eq!(n, 2, "two 'a' markers cleared");

        let remaining = list_panic_markers(tmp.path());
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].1.session_id, "b");
    }

    #[test]
    fn clear_panic_markers_for_session_returns_zero_for_unknown_id() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = marker_dir(tmp.path());
        write_panic_marker(&dir, &sample_marker("kept", "2026-04-18T00:00:00+00:00")).unwrap();
        let n = clear_panic_markers_for_session(tmp.path(), "not-there");
        assert_eq!(n, 0);
        assert_eq!(
            list_panic_markers(tmp.path()).len(),
            1,
            "non-matching id leaves marker intact"
        );
    }

    #[test]
    fn clear_panic_markers_for_session_returns_zero_on_missing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        // No panic_markers/ subdir created at all.
        let n = clear_panic_markers_for_session(tmp.path(), "any");
        assert_eq!(n, 0, "missing dir is not a panic, not an error");
    }

    // ── TUI startup banner (nudge, not dump) ───────────────────────────────

    fn banner_pair(session: &str, message: &str) -> (PathBuf, PanicMarker) {
        (
            PathBuf::from(format!("/tmp/{}.json", session)),
            PanicMarker {
                session_id: session.to_string(),
                binary_version: "4.0.0".to_string(),
                panicked_at: "2026-04-18T00:00:00+00:00".to_string(),
                message: message.to_string(),
                location: Some("src/foo.rs:1".to_string()),
                in_flight: None,
            },
        )
    }

    #[test]
    fn build_recovery_banner_empty_returns_none() {
        assert!(build_recovery_banner(&[]).is_none());
    }

    #[test]
    fn build_recovery_banner_single_uses_singular_form() {
        let s = build_recovery_banner(&[banner_pair("a", "m")]).expect("some");
        assert!(s.contains("1 unrecovered session"), "singular count: {s}");
        assert!(!s.contains("sessions"), "no plural 's': {s}");
    }

    #[test]
    fn build_recovery_banner_multiple_uses_plural_form() {
        let markers = [
            banner_pair("a", "m"),
            banner_pair("b", "m"),
            banner_pair("c", "m"),
        ];
        let s = build_recovery_banner(&markers).expect("some");
        assert!(s.contains("3 unrecovered sessions"), "plural count: {s}");
    }

    #[test]
    fn build_recovery_banner_two_uses_plural_form() {
        let markers = [banner_pair("a", "m"), banner_pair("b", "m")];
        let s = build_recovery_banner(&markers).expect("some");
        assert!(
            s.contains("2 unrecovered sessions"),
            "plural at n=2 (off-by-one guard): {s}"
        );
    }

    #[test]
    fn build_recovery_banner_includes_recovery_command_hint() {
        let s = build_recovery_banner(&[banner_pair("a", "m")]).expect("some");
        assert!(s.contains("dm --recovery"), "next-step command: {s}");
    }

    #[test]
    fn build_recovery_banner_is_single_line() {
        let s = build_recovery_banner(&[banner_pair("a", "m")]).expect("some");
        assert!(
            !s.contains('\n'),
            "banner must be single-line for scroll buffer: {s:?}"
        );
    }

    #[test]
    fn build_recovery_banner_does_not_leak_session_details() {
        let s = build_recovery_banner(&[banner_pair("secret-session-id", "m")]).expect("some");
        assert!(
            !s.contains("secret-session-id"),
            "banner is a nudge, not a dump — session id stays in --recovery: {s}"
        );
    }

    #[test]
    fn build_recovery_banner_does_not_leak_panic_message() {
        let s = build_recovery_banner(&[banner_pair("a", "index out of bounds")]).expect("some");
        assert!(
            !s.contains("index out of bounds"),
            "banner must not surface panic messages: {s}"
        );
    }

    // ── In-flight turn snapshot (C75) ─────────────────────────────────────
    //
    // TURN_IN_FLIGHT is a process-global, so each test clears it at the end
    // to avoid cross-test pollution. The set→read→clear pattern is always
    // fully contained within one test.

    /// Serializes tests that mutate the process-global `TURN_IN_FLIGHT`
    /// `ArcSwapOption` so set/update/clear interleavings from parallel test
    /// threads can't race. Production writers stay lock-free — the panic
    /// hook path must never deadlock — but tests need serialization or
    /// the shared global races once the suite fan-out grows.
    static TURN_IN_FLIGHT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sample_turn(session: &str, partial: &str) -> TurnInFlight {
        TurnInFlight {
            session_id: session.to_string(),
            partial_text: partial.to_string(),
            started_at: "2026-04-18T12:34:56+00:00".to_string(),
        }
    }

    #[test]
    fn turn_in_flight_get_set_clear_roundtrip() {
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear_turn_in_flight();
        assert!(current_turn_in_flight().is_none(), "starts empty");
        set_turn_in_flight(sample_turn("sess1", "hello"));
        let got = current_turn_in_flight().expect("set");
        assert_eq!(got.session_id, "sess1");
        assert_eq!(got.partial_text, "hello");
        clear_turn_in_flight();
        assert!(current_turn_in_flight().is_none(), "cleared");
    }

    #[test]
    fn turn_in_flight_update_partial_appends() {
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear_turn_in_flight();
        set_turn_in_flight(sample_turn("sess2", "he"));
        update_turn_partial("llo");
        update_turn_partial(" world");
        let got = current_turn_in_flight().expect("set");
        assert_eq!(got.partial_text, "hello world");
        assert_eq!(
            got.session_id, "sess2",
            "session id preserved across updates"
        );
        clear_turn_in_flight();
    }

    #[test]
    fn turn_in_flight_update_when_none_is_noop() {
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear_turn_in_flight();
        update_turn_partial("dropped");
        assert!(
            current_turn_in_flight().is_none(),
            "update with no active turn must not create one"
        );
    }

    #[test]
    fn turn_in_flight_set_twice_replaces() {
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear_turn_in_flight();
        set_turn_in_flight(sample_turn("first", "aaa"));
        set_turn_in_flight(sample_turn("second", "bbb"));
        let got = current_turn_in_flight().expect("set");
        assert_eq!(got.session_id, "second", "newer snapshot wins");
        assert_eq!(got.partial_text, "bbb", "no carry-over from first snapshot");
        clear_turn_in_flight();
    }

    #[test]
    fn turn_in_flight_lock_serializes_parallel_set_clear() {
        // Regression guard for the shared-global race that preempted C81:
        // under the lock, 32 set→read→clear roundtrips must be
        // deterministic even when this test is stress-scheduled alongside
        // its siblings. Without the lock, fan-out under cargo's default
        // parallelism would eventually observe a stale `current_turn_in_flight`.
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for i in 0..32 {
            clear_turn_in_flight();
            set_turn_in_flight(sample_turn(&format!("s{i}"), "tok"));
            let got = current_turn_in_flight().expect("set");
            assert_eq!(got.session_id, format!("s{i}"));
            clear_turn_in_flight();
            assert!(current_turn_in_flight().is_none());
        }
    }

    #[test]
    fn panic_marker_serializes_in_flight_when_present() {
        let marker = PanicMarker {
            session_id: "s".to_string(),
            binary_version: "4.0.0".to_string(),
            panicked_at: "2026-04-18T00:00:00+00:00".to_string(),
            message: "boom".to_string(),
            location: None,
            in_flight: Some(sample_turn("s", "partial response")),
        };
        let json = serde_json::to_string(&marker).unwrap();
        assert!(
            json.contains("\"in_flight\""),
            "in_flight field present: {json}"
        );
        assert!(
            json.contains("\"partial_text\":\"partial response\""),
            "partial text serialized: {json}"
        );
    }

    #[test]
    fn panic_marker_omits_in_flight_when_none() {
        let marker = PanicMarker {
            session_id: "s".to_string(),
            binary_version: "4.0.0".to_string(),
            panicked_at: "2026-04-18T00:00:00+00:00".to_string(),
            message: "boom".to_string(),
            location: None,
            in_flight: None,
        };
        let json = serde_json::to_string(&marker).unwrap();
        assert!(
            !json.contains("in_flight"),
            "in_flight omitted when None (keeps pre-C75 readers happy): {json}"
        );
    }

    #[test]
    fn panic_marker_deserializes_pre_c75_json_with_in_flight_none() {
        // JSON as written by pre-C75 binaries (no in_flight field).
        let pre_c75 = r#"{
            "session_id": "old",
            "binary_version": "4.0.0",
            "panicked_at": "2026-04-18T00:00:00+00:00",
            "message": "boom",
            "location": null
        }"#;
        let marker: PanicMarker = serde_json::from_str(pre_c75).unwrap();
        assert_eq!(marker.session_id, "old");
        assert!(
            marker.in_flight.is_none(),
            "missing field deserializes to None"
        );
    }

    #[test]
    fn run_panic_hook_actions_captures_in_flight_into_marker() {
        let _guard = TURN_IN_FLIGHT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        clear_turn_in_flight();
        set_turn_in_flight(sample_turn("8e3f1a2b", "half a sentence"));

        let writes: RefCell<Vec<PanicMarker>> = RefCell::new(Vec::new());
        run_panic_hook_actions(
            "boom",
            Some("src/foo.rs:10"),
            &ctx(),
            ts(),
            || {},
            |_| Ok(()),
            |_, _| Ok(()),
            |_, marker| {
                writes.borrow_mut().push(marker.clone());
                Ok(PathBuf::new())
            },
        );

        let w = writes.borrow();
        assert_eq!(w.len(), 1);
        let snap = w[0].in_flight.as_ref().expect("captures active turn");
        assert_eq!(snap.session_id, "8e3f1a2b");
        assert_eq!(snap.partial_text, "half a sentence");

        clear_turn_in_flight();
    }
}
