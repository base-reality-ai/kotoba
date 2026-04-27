//! `/sessions`-family command support.
//!
//! Phase 1.2 module 4 (seeded C50, closed C54). Houses session-management
//! dispatch handlers (`handle_fork`, `handle_resume`, `handle_sessions`,
//! `handle_sessions_tree`, `handle_export`) extracted from
//! `commands::execute()`. Migrated: `/fork` (C50), `/resume` (C51),
//! `/sessions tree` (C52), `/sessions` (C53), `/export` (C54).
//! Family closed: 5/5.

use super::{format_entries_as_markdown_filtered, format_session_row, SlashResult};
use crate::session::short_id;
use crate::tui::app::{App, EntryKind};

/// `/fork [at_turn]` — duplicate the current session, optionally truncated to `at_turn` turns.
///
/// Loads the active session via `session::storage::load`, calls
/// `Session::fork(at_turn)`, persists the fork via `session::storage::save`.
/// Returns an info message with the new session's id and a `/resume` hint
/// on success; an error string on load or save failure.
pub(super) fn handle_fork(at_turn: Option<usize>, app: &App) -> SlashResult {
    let session = match crate::session::storage::load(&app.config_dir, &app.session_id) {
        Ok(s) => s,
        Err(e) => {
            return SlashResult::Error(format!(
                "Failed to load current session: {}. Try: /sessions to list available sessions, or start a fresh one.",
                e
            ));
        }
    };
    let forked = session.fork(at_turn);
    if let Err(e) = crate::session::storage::save(&app.config_dir, &forked) {
        return SlashResult::Error(format!(
            "Failed to save forked session: {}. Try: confirm ~/.dm/sessions/ is writable.",
            e
        ));
    }
    let short = short_id(&forked.id);
    let turn_info = match at_turn {
        Some(n) => format!(" at turn {}", n),
        None => String::new(),
    };
    SlashResult::Info(format!(
        "Forked session{} → {}\nUse /resume {} to switch.",
        turn_info, forked.id, short
    ))
}

/// `/resume <id>` — switch the TUI to an earlier session by ID prefix.
///
/// Empty input emits a usage hint as a system entry. Otherwise routes the
/// raw `"/resume {id}"` string to the outer event loop via `user_tx` for
/// actual session-swap handling. Always returns `SlashResult::Done`
/// (makes the previous fallthrough-to-`Done` behavior explicit).
pub(super) async fn handle_resume(
    id: String,
    app: &mut App,
    user_tx: &tokio::sync::mpsc::Sender<String>,
) -> SlashResult {
    if id.is_empty() {
        app.push_entry(
            EntryKind::SystemInfo,
            "Usage: /resume <session-id-prefix>".to_string(),
        );
    } else {
        user_tx.send(format!("/resume {}", id)).await.ok();
    }
    SlashResult::Done
}

/// `/sessions tree` — render the full session forest (parent/fork relationships) as a tree.
///
/// Reads all persisted sessions via `session::storage::list`, builds a
/// tree via `session::storage::format_session_tree` using
/// `format_session_row` as the per-row renderer, and pushes the result
/// as a `SystemInfo` entry. Handles empty-list and I/O-error branches
/// with distinct system messages. Falls through to `SlashResult::Done`.
pub(super) fn handle_sessions_tree(app: &mut App) -> SlashResult {
    match crate::session::storage::list(&app.config_dir) {
        Ok(sessions) if sessions.is_empty() => {
            app.push_entry(EntryKind::SystemInfo, "No saved sessions.".to_string());
        }
        Ok(sessions) => {
            let tree = crate::session::storage::format_session_tree(&sessions, format_session_row);
            app.push_entry(EntryKind::SystemInfo, format!("Session tree:\n{}", tree));
        }
        Err(e) => {
            app.push_entry(
                EntryKind::SystemInfo,
                format!("Error listing sessions: {}", e),
            );
        }
    }
    SlashResult::Done
}

/// `/sessions [limit]` — render the most recent `limit` sessions as a flat table.
///
/// Reads all persisted sessions via `session::storage::list`, takes the
/// first `limit`, formats each row via `format_session_row`, and pushes
/// the concatenated list as a `SystemInfo` entry. Handles empty-list
/// and I/O-error branches. Falls through to `SlashResult::Done`.
pub(super) fn handle_sessions(limit: usize, app: &mut App) -> SlashResult {
    match crate::session::storage::list(&app.config_dir) {
        Ok(sessions) if sessions.is_empty() => {
            app.push_entry(EntryKind::SystemInfo, "No saved sessions.".to_string());
        }
        Ok(sessions) => {
            let list = sessions
                .iter()
                .take(limit)
                .map(format_session_row)
                .collect::<Vec<_>>()
                .join("\n");
            app.push_entry(EntryKind::SystemInfo, format!("Recent sessions:\n{}", list));
        }
        Err(e) => {
            app.push_entry(
                EntryKind::SystemInfo,
                format!("Error listing sessions: {}", e),
            );
        }
    }
    SlashResult::Done
}

/// `/export [clean] [path]` — write the current conversation to a markdown file.
///
/// Optional `clean` mode strips tool-call entries before rendering.
/// If `path` is empty, generates `dm-export-{YYYYMMDD-HHMMSS}[-clean].md`
/// in the cwd. Returns `Info("Conversation exported to {path} ({n} lines)")`
/// on success, `Error(format_write_error(...))` on I/O failure.
pub(super) fn handle_export(arg: String, app: &App) -> SlashResult {
    // Accept `/export [path]` and `/export clean [path]`. The
    // guarded strip_prefix ensures a stray literal path like
    // "cleanup.md" is not mis-parsed as clean mode + "up.md":
    // we only treat "clean" as the mode token when it is
    // followed by whitespace or end-of-string.
    let (strip_tools, path_arg) = match arg.strip_prefix("clean") {
        Some(rest) if rest.is_empty() || rest.starts_with(' ') => {
            (true, rest.trim_start().to_string())
        }
        _ => (false, arg.clone()),
    };
    let markdown = format_entries_as_markdown_filtered(&app.entries, strip_tools);
    let path = if path_arg.is_empty() {
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let suffix = if strip_tools { "-clean" } else { "" };
        format!("dm-export-{}{}.md", ts, suffix)
    } else {
        path_arg
    };
    match std::fs::write(&path, &markdown) {
        Ok(()) => SlashResult::Info(format!(
            "Conversation exported to {} ({} lines)",
            path,
            markdown.lines().count()
        )),
        Err(e) => SlashResult::Error(crate::tools::fs_error::format_write_error(
            &e,
            &path,
            "Export failed",
        )),
    }
}
