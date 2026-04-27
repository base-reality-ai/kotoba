//! Turn-level summary of file edits.
//!
//! The TUI accumulates an `EditRecord` per file-mutating tool call
//! (parsed from the tool's result string) and, at turn completion,
//! emits a single "Modified N files: ..." line when ≥2 files changed.
//! Single-file turns skip the summary — the per-file diff already
//! shows what changed, and a summary would be redundant noise.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditRecord {
    pub path: String,
    pub added: u64,
    pub removed: u64,
}

/// Parse a file-mutating tool's output into an `EditRecord`. Returns
/// `None` for tools that don't mutate files (bash, grep, etc.) or when
/// the output doesn't match a recognized shape (error path, unexpected
/// format — absent data is always safer than bad data here).
pub fn parse_tool_output(tool_name: &str, output: &str, is_error: bool) -> Option<EditRecord> {
    if is_error {
        return None;
    }
    match tool_name {
        "edit_file" => parse_edit_file(output),
        "write_file" => parse_write_file(output),
        _ => None,
    }
}

fn parse_edit_file(output: &str) -> Option<EditRecord> {
    let (path, diff) = super::tool_dispatch::extract_diff(output)?;
    let (added, removed) = count_diff_lines(&diff);
    Some(EditRecord {
        path,
        added,
        removed,
    })
}

fn parse_write_file(output: &str) -> Option<EditRecord> {
    // Tolerate the optional scope_note prefix emitted by file_write.rs:123
    // for out-of-project writes.
    let body = super::tool_dispatch::strip_scope_note(output);

    // Shape: "{verb} {path} ({N} lines){optional fmt_note}"
    // verb ∈ {"Created", "Overwrote"} per file_write.rs:129. Explicit match
    // so a future verb fails visibly (None) instead of capturing garbage.
    let rest = body
        .strip_prefix("Created ")
        .or_else(|| body.strip_prefix("Overwrote "))?;
    let paren_start = rest.find(" (")?;
    let path = rest[..paren_start].to_string();
    let after_paren = &rest[paren_start + 2..];
    let lines_idx = after_paren.find(" lines)")?;
    let n: u64 = after_paren[..lines_idx].parse().ok()?;
    Some(EditRecord {
        path,
        added: n,
        removed: 0,
    })
}

fn count_diff_lines(diff: &str) -> (u64, u64) {
    let mut added = 0u64;
    let mut removed = 0u64;
    for line in diff.lines() {
        // Skip unified-diff file headers — they start with "+++" / "---"
        // and must not inflate the edit-line counts.
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if let Some(first) = line.chars().next() {
            match first {
                '+' => added += 1,
                '-' => removed += 1,
                _ => {}
            }
        }
    }
    (added, removed)
}

/// Format a turn summary. Returns `None` for fewer than 2 records —
/// the caller suppresses the summary entirely so single-file turns
/// don't get redundant noise alongside their per-file diff.
pub fn format_summary(records: &[EditRecord]) -> Option<String> {
    if records.len() < 2 {
        return None;
    }
    let n = records.len();
    let per_file: Vec<String> = records.iter().map(format_one).collect();
    Some(format!("Modified {} files: {}", n, per_file.join(", ")))
}

fn format_one(rec: &EditRecord) -> String {
    if rec.removed == 0 {
        format!("{} (+{})", rec.path, rec.added)
    } else {
        format!("{} (+{}/-{})", rec.path, rec.added, rec.removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_edit_file_extracts_path_and_counts() {
        let out = "Applied edit to src/main.rs\n\n\
                   --- a/src/main.rs\n\
                   +++ b/src/main.rs\n\
                   @@ -1,3 +1,4 @@\n\
                   -old line\n\
                   +new line one\n\
                   +new line two\n\
                    unchanged\n";
        let r = parse_tool_output("edit_file", out, false).unwrap();
        assert_eq!(r.path, "src/main.rs");
        assert_eq!(r.added, 2);
        assert_eq!(r.removed, 1);
    }

    #[test]
    fn parse_edit_file_ignores_header_lines() {
        let out = "Applied edit to foo.rs\n\n\
                   --- a/foo.rs\n\
                   +++ b/foo.rs\n\
                   @@ -1 +1 @@\n\
                   -a\n\
                   +b\n";
        let r = parse_tool_output("edit_file", out, false).unwrap();
        assert_eq!(r.added, 1, "+++ header must not count as add");
        assert_eq!(r.removed, 1, "--- header must not count as remove");
    }

    #[test]
    fn parse_write_file_new_file_counts_as_added_only() {
        let out = "Created src/lib.rs (42 lines)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.path, "src/lib.rs");
        assert_eq!(r.added, 42);
        assert_eq!(r.removed, 0);
    }

    #[test]
    fn parse_write_file_created_verb() {
        let out = "Created src/lib.rs (42 lines)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.path, "src/lib.rs");
    }

    #[test]
    fn parse_write_file_overwrote_verb() {
        let out = "Overwrote src/lib.rs (42 lines)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.path, "src/lib.rs");
        assert_eq!(r.added, 42);
    }

    #[test]
    fn parse_write_file_with_format_note() {
        // file_write.rs:128 appends " (formatted)" only — not " (formatted with rustfmt)".
        let out = "Created src/lib.rs (42 lines) (formatted)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.added, 42);
    }

    #[test]
    fn parse_write_file_with_scope_note() {
        let out = "Note: /tmp/x.rs is outside the project directory.\nCreated /tmp/x.rs (5 lines)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.path, "/tmp/x.rs");
        assert_eq!(r.added, 5);
    }

    #[test]
    fn parse_write_file_with_scope_note_and_fmt_note() {
        let out = "Note: /tmp/x.rs is outside the project directory.\nOverwrote /tmp/x.rs (5 lines) (formatted)";
        let r = parse_tool_output("write_file", out, false).unwrap();
        assert_eq!(r.path, "/tmp/x.rs");
        assert_eq!(r.added, 5);
    }

    #[test]
    fn parse_write_file_unknown_verb_returns_none() {
        // Explicit verb match means a future rename in file_write.rs fails
        // visibly (None → no summary entry) instead of silently capturing
        // a bogus path.
        let out = "Wrote src/lib.rs (42 lines)";
        assert!(parse_tool_output("write_file", out, false).is_none());
    }

    #[test]
    fn parse_edit_file_with_fmt_note_strips_suffix() {
        let out = "Applied edit to src/main.rs (formatted)\n\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-a\n+b\n";
        let r = parse_tool_output("edit_file", out, false).unwrap();
        assert_eq!(r.path, "src/main.rs");
    }

    #[test]
    fn parse_edit_file_with_scope_note_prefix_parses() {
        let out = "Note: /tmp/x.rs is outside the project directory.\nApplied edit to /tmp/x.rs\n\n-a\n+b\n";
        let r = parse_tool_output("edit_file", out, false).unwrap();
        assert_eq!(r.path, "/tmp/x.rs");
        assert_eq!(r.added, 1);
        assert_eq!(r.removed, 1);
    }

    #[test]
    fn parse_error_result_returns_none() {
        // Error results must never leak into the summary — they'd confuse
        // the user into thinking a failed tool call modified a file.
        let out = "Created src/lib.rs (42 lines)";
        assert!(parse_tool_output("write_file", out, true).is_none());
    }

    #[test]
    fn parse_unknown_tool_returns_none() {
        assert!(parse_tool_output("bash", "foo", false).is_none());
        assert!(parse_tool_output("grep", "match: x", false).is_none());
    }

    #[test]
    fn parse_malformed_write_output_returns_none() {
        assert!(parse_tool_output("write_file", "oops", false).is_none());
        assert!(parse_tool_output("write_file", "Created nolines", false).is_none());
    }

    #[test]
    fn format_summary_single_file_is_none() {
        let r = vec![EditRecord {
            path: "a.rs".into(),
            added: 1,
            removed: 0,
        }];
        assert!(format_summary(&r).is_none());
    }

    #[test]
    fn format_summary_empty_is_none() {
        assert!(format_summary(&[]).is_none());
    }

    #[test]
    fn format_summary_two_files_mixed_stats() {
        let r = vec![
            EditRecord {
                path: "src/main.rs".into(),
                added: 12,
                removed: 3,
            },
            EditRecord {
                path: "src/lib.rs".into(),
                added: 1,
                removed: 0,
            },
        ];
        let out = format_summary(&r).unwrap();
        assert_eq!(
            out,
            "Modified 2 files: src/main.rs (+12/-3), src/lib.rs (+1)"
        );
    }

    #[test]
    fn format_summary_three_files_preserves_order() {
        let r = vec![
            EditRecord {
                path: "a.rs".into(),
                added: 1,
                removed: 0,
            },
            EditRecord {
                path: "b.rs".into(),
                added: 2,
                removed: 2,
            },
            EditRecord {
                path: "c.rs".into(),
                added: 5,
                removed: 1,
            },
        ];
        let out = format_summary(&r).unwrap();
        assert!(out.starts_with("Modified 3 files: "));
        let after = &out["Modified 3 files: ".len()..];
        assert_eq!(after, "a.rs (+1), b.rs (+2/-2), c.rs (+5/-1)");
    }
}
