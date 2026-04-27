//! Pure formatters used by `commands.rs` — session rows, files reports, dir
//! context collection, and markdown export. Free functions only, no App
//! state. Extracted from `commands.rs` to keep the dispatch file focused on
//! parsing and routing.

use std::fmt::Write as _;

use crate::session::short_id;
use crate::session::storage as session_storage;
use crate::tui::app::{DisplayEntry, EntryKind};

/// Format one row of the `/sessions` listing.
///
/// Layout: `{short_id} | {timestamp} | {turns_label} | {tokens_label} | {title} | {project} | {model}`
///
/// Turns are shown from the persisted `turn_count` when available; for older
/// sessions (pre-counter, `turn_count == 0`) we fall back to counting user
/// messages and prefix with `~` to signal the estimate. Tokens are only shown
/// when persisted — we never estimate tokens from message content.
pub fn format_session_row(s: &crate::session::Session) -> String {
    let short = short_id(&s.id);
    let ts = s.updated_at.format("%Y-%m-%d %H:%M").to_string();
    let title = s
        .title
        .as_deref()
        .filter(|t| !t.is_empty())
        .unwrap_or("(untitled)");

    let turns_label = if s.turn_count > 0 {
        let unit = if s.turn_count == 1 { "turn" } else { "turns" };
        format!("{} {}", s.turn_count, unit)
    } else {
        let est = session_storage::count_user_messages(&s.messages);
        if est == 0 {
            "—".to_string()
        } else {
            let unit = if est == 1 { "turn" } else { "turns" };
            format!("~{} {}", est, unit)
        }
    };

    let tokens_label = {
        let total = s.prompt_tokens + s.completion_tokens;
        if total == 0 {
            "—".to_string()
        } else {
            format!("{} tok", session_storage::human_tokens(total))
        }
    };
    let project = s.host_project.as_deref().unwrap_or("kernel");

    format!(
        "  • {} | {} | {} | {} | {} | {} | {}",
        short, ts, turns_label, tokens_label, title, project, s.model
    )
}

/// Build a human-readable report of files in the current conversation context.
///
/// `pinned_files` are files prepended on every message; `has_pending_context` indicates
/// a one-shot context (e.g. from `/add`, `/diff`) is queued for the next message.
/// Pure and testable: no filesystem access.
pub fn format_files_report(pinned_files: &[String], has_pending_context: bool) -> String {
    if pinned_files.is_empty() && !has_pending_context {
        return "No files in context.\n\
                Use /pin <file> to prepend a file to every message, \
                or /add <file> to attach it to the next message."
            .to_string();
    }
    let mut out = String::from("Files in context:\n");
    if !pinned_files.is_empty() {
        out.push_str("\nPinned (injected on every message):\n");
        for f in pinned_files {
            writeln!(out, "  • {}", f).expect("write to String never fails");
        }
    }
    if has_pending_context {
        out.push_str("\nOne-shot context queued for next message (e.g. from /add or /diff).\n");
    }
    out
}

/// Collect the contents of all text files under `dir_path` as a single context string.
///
/// Files are discovered with the indexer's `collect_indexable_files`, so the same
/// `.dmignore` / default ignore rules apply. Each file is formatted as a fenced code
/// block with its relative path as the label.
///
/// `max_bytes` caps the total output so it doesn't blow up the context window.
/// Returns `(context_string, file_count)` on success, or an error message string on failure.
pub fn collect_dir_context(dir_path: &str, max_bytes: usize) -> Result<(String, usize), String> {
    let base = std::path::Path::new(dir_path);
    if !base.exists() {
        return Err(format!(
            "'{}' does not exist. Try: confirm the path is correct and the directory has been created.",
            dir_path
        ));
    }
    if !base.is_dir() {
        return Err(format!(
            "'{}' is not a directory. Try: pass a directory path, or use /add to attach a single file.",
            dir_path
        ));
    }
    let files = crate::index::chunker::collect_indexable_files(base).map_err(|e| {
        format!(
            "Failed to scan '{}': {}. Try: confirm read access to the directory and its contents.",
            dir_path, e
        )
    })?;
    if files.is_empty() {
        return Err(format!(
            "No indexable text files found in '{}'. Try: confirm the directory contains text files (binary and .dmignore-excluded files are skipped).",
            dir_path
        ));
    }
    let mut parts: Vec<String> = Vec::new();
    let mut total_bytes = 0usize;
    let mut included = 0usize;
    for file in &files {
        if total_bytes >= max_bytes {
            break;
        }
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        let rel = file
            .strip_prefix(base)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();
        let lang = file.extension().and_then(|e| e.to_str()).unwrap_or("");
        let remaining = max_bytes.saturating_sub(total_bytes);
        let truncated = crate::git::safe_truncate(&content, remaining);
        let block = format!("File: {}\n```{}\n{}\n```", rel, lang, truncated);
        total_bytes += block.len();
        parts.push(block);
        included += 1;
    }
    if parts.is_empty() {
        return Err(format!(
            "All files in '{}' were unreadable. Try: confirm read access to the files in the directory.",
            dir_path
        ));
    }
    let skipped = files.len().saturating_sub(included);
    let mut result = parts.join("\n\n");
    if skipped > 0 {
        write!(
            result,
            "\n\n({} file(s) omitted — context window limit reached)",
            skipped
        )
        .expect("write to String never fails");
    }
    Ok((result, included))
}

/// Format the TUI display entries as a Markdown document suitable for export.
/// Includes every entry kind, wrapping tool traffic in `<details>` blocks.
///
/// Dead in the `dm` binary (the dispatcher calls `_filtered` directly); kept
/// as a documented API alias for "full export" and exercised by unit tests.
#[allow(dead_code)]
pub fn format_entries_as_markdown(entries: &[DisplayEntry]) -> String {
    format_entries_as_markdown_filtered(entries, false)
}

/// Format display entries as Markdown, optionally stripping tool noise.
///
/// When `strip_tools` is `true`, `ToolCall`, `ToolResult`, `ToolError`,
/// `SystemInfo`, and `Notice` entries are omitted — the output keeps only
/// user messages, assistant responses, file diffs, and image attachments.
/// Intended for `/export clean`, which produces a shareable transcript.
pub fn format_entries_as_markdown_filtered(entries: &[DisplayEntry], strip_tools: bool) -> String {
    use chrono::Local;
    let timestamp = Local::now().format("%Y-%m-%d %H:%M");
    let mut out = format!("# dm conversation export — {}\n\n", timestamp);

    for entry in entries {
        match entry.kind {
            EntryKind::UserMessage => {
                write!(out, "**You:** {}\n\n", entry.content).expect("write to String never fails");
            }
            EntryKind::AssistantMessage => {
                out.push_str(&entry.content);
                out.push_str("\n\n");
            }
            EntryKind::FileDiff => {
                out.push_str("```diff\n");
                out.push_str(&entry.content);
                out.push_str("\n```\n\n");
            }
            EntryKind::ImageAttachment => {
                write!(out, "[image: {}]\n\n", entry.content).expect("write to String never fails");
            }
            EntryKind::ToolCall if !strip_tools => {
                let first_line = entry.content.lines().next().unwrap_or("");
                write!(
                    out,
                    "<details><summary>Tool call: {}</summary>\n\n```\n{}\n```\n\n</details>\n\n",
                    first_line, entry.content
                )
                .expect("write to String never fails");
            }
            EntryKind::ToolResult if !strip_tools => {
                out.push_str("<details><summary>Tool result</summary>\n\n```\n");
                out.push_str(&entry.content);
                out.push_str("\n```\n\n</details>\n\n");
            }
            EntryKind::ToolError if !strip_tools => {
                write!(out, "⚠ {}\n\n", entry.content).expect("write to String never fails");
            }
            EntryKind::SystemInfo if !strip_tools => {
                write!(out, "*{}*\n\n", entry.content).expect("write to String never fails");
            }
            EntryKind::Notice if !strip_tools => {
                write!(out, "_{}_\n\n", entry.content).expect("write to String never fails");
            }
            _ => {}
        }
    }
    out
}
