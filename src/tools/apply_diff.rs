use std::fmt::Write as _;

use super::{Tool, ToolResult};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

const MAX_FUZZ_OFFSET: usize = 30;

pub struct ApplyDiffTool;

#[async_trait]
impl Tool for ApplyDiffTool {
    fn name(&self) -> &'static str {
        "apply_diff"
    }

    fn description(&self) -> &'static str {
        "Apply a unified diff to a file. More precise than file_edit for multi-hunk changes."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to patch"
                },
                "diff": {
                    "type": "string",
                    "description": "Unified diff string (output of `diff -u` or similar)"
                }
            },
            "required": ["path", "diff"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use apply_diff for precise multi-hunk changes with line-number context. \
              Best for large structural changes where edit_file would need too much context. \
              The diff must be in unified diff format.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let path = args["path"].as_str().ok_or_else(|| {
            anyhow!(
                "Missing required parameter: path\n    \
                 Try: include a \"path\" string in the tool arguments (relative or absolute)."
            )
        })?;
        let diff = args["diff"].as_str().ok_or_else(|| {
            anyhow!(
                "Missing required parameter: diff\n    \
                 Try: include a \"diff\" string in unified-diff format (output of `diff -u` or `git diff`)."
            )
        })?;

        // Path safety: block writes through symlinks
        let project_root = std::env::current_dir().unwrap_or_default();
        let resolved = match super::path_safety::validate_path(path, &project_root) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Error: invalid path '{}': {}", path, e),
                    is_error: true,
                });
            }
        };
        if let Some(msg) = super::path_safety::check_write_blocked(&resolved, path, "apply diff to")
        {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }
        if let Some(msg) =
            super::path_safety::check_sensitive_blocked(&resolved.canonical, path, "apply diff to")
        {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        if let Some(err) = super::path_safety::check_file_editable(path, "apply diff to").await {
            return Ok(err);
        }

        let original = tokio::fs::read_to_string(path).await.map_err(|e| {
            anyhow!(super::fs_error::format_read_error(
                &e,
                path,
                &format!("Cannot read file '{}'", path),
            ))
        })?;

        let (patched, adjusted) = apply_diff(&original, diff).map_err(|e| {
            anyhow!(
                "Failed to apply diff to '{}': {}\n    \
                 Try: regenerate the diff against the current file (lines may have drifted). \
                 Re-read the file with read_file and produce a fresh unified diff.",
                path,
                e
            )
        })?;

        let hunk_count = diff.lines().filter(|l| l.starts_with("@@")).count();

        tokio::fs::write(path, &patched).await.map_err(|e| {
            anyhow!(super::fs_error::format_write_error(
                &e,
                path,
                &format!("Cannot write file '{}'", path),
            ))
        })?;

        let formatted = if crate::config::auto_format_enabled() {
            crate::format::format_file(std::path::Path::new(path))
                .await
                .unwrap_or(false)
        } else {
            false
        };

        let suffix = if adjusted > 0 {
            format!(" ({} adjusted)", adjusted)
        } else {
            String::new()
        };
        let fmt_note = if formatted { " (formatted)" } else { "" };
        let mut content = format!(
            "Applied {} hunk{} to {}{}{}",
            hunk_count,
            if hunk_count == 1 { "" } else { "s" },
            path,
            suffix,
            fmt_note
        );

        let drift_warning = crate::tools::post_write_wiki_sync(&resolved.canonical, &patched).await;
        if let Some(w) = drift_warning {
            content.push_str("\n\n");
            content.push_str(&w);
        }

        Ok(ToolResult {
            content,
            is_error: false,
        })
    }
}

/// Parse and apply a unified diff to `content`.
/// Returns `(patched_content, adjusted_count)` where `adjusted_count` is the number
/// of hunks that required fuzzy position matching.
pub fn apply_diff(content: &str, diff: &str) -> Result<(String, usize)> {
    let mut lines: Vec<&str> = content.lines().collect();
    let trailing_newline = content.ends_with('\n');

    let hunks = split_into_hunks(diff);

    if hunks.is_empty() {
        return Ok((content.to_string(), 0));
    }

    let mut adjusted = 0usize;

    struct ResolvedHunk<'a> {
        actual_start: usize,
        rel_old: usize,
        replacement: Vec<&'a str>,
        was_adjusted: bool,
    }

    // Pass 1: resolve all hunk positions
    let mut resolved: Vec<ResolvedHunk> = Vec::with_capacity(hunks.len());
    for hunk in &hunks {
        let header = hunk[0];
        let (old_start, _old_count) = parse_hunk_header(header).map_err(|e| {
            anyhow!(
                "Bad hunk header '{}': {}\n    \
                 Check: hunk headers must be `@@ -old_start[,old_count] +new_start[,new_count] @@`. \
                 Regenerate the diff with `diff -u` or `git diff`.",
                header,
                e
            )
        })?;

        let mut expected_lines: Vec<&str> = Vec::new();
        let mut rel_old = 0usize;

        for raw in &hunk[1..] {
            if let Some(rest) = raw.strip_prefix('-') {
                expected_lines.push(rest);
                rel_old += 1;
            } else if raw.starts_with('+') {
                // additions don't consume old lines
            } else if let Some(rest) = raw.strip_prefix(' ') {
                expected_lines.push(rest);
                rel_old += 1;
            } else if raw.is_empty() {
                expected_lines.push("");
                rel_old += 1;
            }
        }

        let declared_start = if old_start > 0 { old_start - 1 } else { 0 };
        let actual_start =
            find_hunk_position(&lines, declared_start, &expected_lines, MAX_FUZZ_OFFSET)?;

        let was_adjusted = actual_start != declared_start;

        let mut replacement: Vec<&str> = Vec::new();
        for raw in &hunk[1..] {
            if let Some(rest) = raw.strip_prefix(' ') {
                replacement.push(rest);
            } else if raw.is_empty() {
                replacement.push("");
            } else if let Some(rest) = raw.strip_prefix('+') {
                replacement.push(rest);
            }
        }

        resolved.push(ResolvedHunk {
            actual_start,
            rel_old,
            replacement,
            was_adjusted,
        });
    }

    // Sort by start position descending (apply back-to-front)
    resolved.sort_by(|a, b| b.actual_start.cmp(&a.actual_start));

    // Check for overlaps: each hunk's end must not exceed the next hunk's start
    for i in 0..resolved.len().saturating_sub(1) {
        let earlier = &resolved[i + 1];
        let later = &resolved[i];
        let earlier_end = earlier.actual_start + earlier.rel_old;
        if earlier_end > later.actual_start {
            return Err(anyhow!(
                "Overlapping hunks: range {}..{} overlaps {}..{}",
                earlier.actual_start,
                earlier_end,
                later.actual_start,
                later.actual_start + later.rel_old,
            ));
        }
    }

    // Pass 2: apply splices back-to-front
    for rh in &resolved {
        let end = rh.actual_start + rh.rel_old;
        lines.splice(rh.actual_start..end, rh.replacement.iter().copied());
        if rh.was_adjusted {
            adjusted += 1;
        }
    }

    let mut result = lines.join("\n");
    if trailing_newline {
        result.push('\n');
    }
    Ok((result, adjusted))
}

/// Search for the correct position of a hunk by sliding ±`max_offset` from the declared start.
/// Tries offset 0, +1, -1, +2, -2, ... Returns the 0-indexed start on match.
fn find_hunk_position(
    lines: &[&str],
    declared_start: usize,
    expected: &[&str],
    max_offset: usize,
) -> Result<usize> {
    if expected.is_empty() {
        return Ok(declared_start);
    }

    for delta in 0..=max_offset {
        for &sign in &[1i64, -1i64] {
            if delta == 0 && sign == -1 {
                continue;
            }
            let candidate = declared_start as i64 + (delta as i64 * sign);
            if candidate < 0 {
                continue;
            }
            let start = candidate as usize;
            if start + expected.len() > lines.len() {
                continue;
            }
            if lines[start..start + expected.len()] == *expected {
                return Ok(start);
            }
        }
    }

    let diag = diagnose_hunk_failure(lines, expected, declared_start);
    Err(anyhow!(
        "Hunk context not found within ±{} lines of declared position {}.\n{}",
        max_offset,
        declared_start + 1,
        diag
    ))
}

fn diagnose_hunk_failure(lines: &[&str], expected: &[&str], declared_start: usize) -> String {
    let mut out = String::new();

    // Tier 1: Full-file scan for best overlap position
    let mut best_pos = 0usize;
    let mut best_count = 0usize;
    if !expected.is_empty() && lines.len() >= expected.len() {
        for pos in 0..=lines.len() - expected.len() {
            let matching = expected
                .iter()
                .enumerate()
                .filter(|(j, e)| lines[pos + j] == **e)
                .count();
            if matching > best_count {
                best_count = matching;
                best_pos = pos;
            }
        }
    }

    if best_count >= 2 || (best_count > 0 && best_count * 2 >= expected.len()) {
        writeln!(
            out,
            "Best match at line {} ({} of {} context lines match). File content there:",
            best_pos + 1,
            best_count,
            expected.len()
        )
        .expect("write to String never fails");
        let ctx_start = best_pos;
        let ctx_end = (best_pos + expected.len()).min(lines.len());
        for (j, line) in lines[ctx_start..ctx_end].iter().enumerate() {
            writeln!(out, "  {}: {}", ctx_start + j + 1, line)
                .expect("write to String never fails");
            if out.len() > 500 {
                out.push_str("  ...\n");
                break;
            }
        }
        return out;
    }

    // Tier 2: Find first expected line anywhere in file
    let first_expected = expected.first().copied().unwrap_or("");
    if !first_expected.is_empty() {
        for (idx, line) in lines.iter().enumerate() {
            if *line == first_expected {
                writeln!(
                    out,
                    "First expected line {:?} found at line {}.",
                    first_expected,
                    idx + 1
                )
                .expect("write to String never fails");
                let ctx_start = idx.saturating_sub(1);
                let ctx_end = (idx + 3).min(lines.len());
                for (j, l) in lines[ctx_start..ctx_end].iter().enumerate() {
                    writeln!(out, "  {}: {}", ctx_start + j + 1, l)
                        .expect("write to String never fails");
                }
                return out;
            }
        }
    }

    // Tier 3: Show what's actually at the declared position
    let ctx_start = declared_start.saturating_sub(2);
    let ctx_end = (declared_start + 3).min(lines.len());
    if ctx_start < lines.len() {
        writeln!(
            out,
            "Expected {:?} but file has at line {}:",
            first_expected,
            declared_start + 1
        )
        .expect("write to String never fails");
        for (j, line) in lines[ctx_start..ctx_end].iter().enumerate() {
            writeln!(out, "  {}: {}", ctx_start + j + 1, line)
                .expect("write to String never fails");
        }
    } else {
        writeln!(
            out,
            "Declared position {} is past end of file ({} lines).",
            declared_start + 1,
            lines.len()
        )
        .expect("write to String never fails");
    }
    out
}

/// Parse the @@ -L,N +L,N @@ header, returning (`old_start`, `old_count`).
fn parse_hunk_header(header: &str) -> Result<(usize, usize)> {
    // Format: @@ -old_start[,old_count] +new_start[,new_count] @@[ optional text]
    let inner = header
        .trim_start_matches('@')
        .trim_start()
        .trim_start_matches('-');
    // inner is now like "1,5 +1,6 @@ ..."
    let old_part = inner.split_whitespace().next().unwrap_or("1");
    let mut parts = old_part.splitn(2, ',');
    let start: usize = parts.next().unwrap_or("1").parse().map_err(|_| {
        anyhow!(
            "Cannot parse old_start in header '{}'. \
             Expected a positive integer before the comma (e.g. `@@ -12,3 +12,4 @@`).",
            header
        )
    })?;
    let count: usize = parts.next().unwrap_or("1").parse().unwrap_or(1);
    Ok((start, count))
}

/// Split a multi-file unified diff into per-file sections.
/// Returns `Vec<(path, per_file_diff_text)>`.
/// Path is taken from the `+++ b/path` (or `+++ path`) header line.
pub fn split_diff_by_file(diff: &str) -> Vec<(String, String)> {
    let mut result: Vec<(String, String)> = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_lines: Vec<&str> = Vec::new();

    for line in diff.lines() {
        if let Some(raw) = line.strip_prefix("+++ ") {
            // Flush previous file
            if let Some(path) = current_path.take() {
                result.push((path, current_lines.join("\n")));
                current_lines.clear();
            }
            // Parse path from "+++ b/path" or "+++ path"
            let path = raw.strip_prefix("b/").unwrap_or(raw).to_string();
            current_path = Some(path);
        } else if current_path.is_some() {
            current_lines.push(line);
        }
        // Lines before first "+++ " (index lines, "diff --git", "--- a/") are skipped
    }
    if let Some(path) = current_path {
        result.push((path, current_lines.join("\n")));
    }
    result
}

/// Split diff lines into hunks. Each hunk is a Vec<&str> where `0` is the @@ header.
fn split_into_hunks<'a>(diff: &'a str) -> Vec<Vec<&'a str>> {
    let mut hunks: Vec<Vec<&'a str>> = Vec::new();
    let mut current: Option<Vec<&'a str>> = None;

    for line in diff.lines() {
        if line.starts_with("@@") {
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            current = Some(vec![line]);
        } else if let Some(ref mut h) = current {
            h.push(line);
        }
        // Lines before the first @@ (file headers like --- +++) are skipped
    }
    if let Some(h) = current {
        hunks.push(h);
    }
    hunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_single_hunk() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let diff = "\
@@ -2,1 +2,1 @@
-line2
+line2_changed
";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert_eq!(adjusted, 0);
        assert!(result.contains("line2_changed"), "result: {}", result);
        assert!(
            !result.contains("\nline2\n"),
            "old line2 should be gone: {}",
            result
        );
        assert!(result.contains("line1"), "line1 should remain");
        assert!(result.contains("line3"), "line3 should remain");
    }

    #[test]
    fn apply_multi_hunk() {
        let content = "a\nb\nc\nd\ne\n";
        let diff = "\
@@ -1,1 +1,1 @@
-a
+A
@@ -5,1 +5,1 @@
-e
+E
";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert_eq!(adjusted, 0);
        assert!(result.contains('A'), "result: {}", result);
        assert!(result.contains('E'), "result: {}", result);
        assert!(!result.contains("\na\n"), "old 'a' should be gone");
        assert!(!result.contains("\ne\n"), "old 'e' should be gone");
        assert!(result.contains('b'), "b should remain");
    }

    #[test]
    fn apply_context_mismatch_errors() {
        let content = "line1\nline2\nline3\n";
        let diff = "\
@@ -1,1 +1,1 @@
-WRONG_LINE
+replacement
";
        let result = apply_diff(content, diff);
        assert!(result.is_err(), "should return Err on context mismatch");
        // Original content should be unchanged (we never wrote to file in pure fn)
    }

    #[test]
    fn apply_add_only_hunk() {
        let content = "line1\nline2\n";
        let diff = "\
@@ -1,1 +1,2 @@
 line1
+inserted
";
        let (result, _) = apply_diff(content, diff).unwrap();
        assert!(result.contains("inserted"), "result: {}", result);
        assert!(result.contains("line1"), "line1 should remain");
        assert!(result.contains("line2"), "line2 should remain");
    }

    #[test]
    fn split_diff_single_file() {
        let diff = "\
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -1,1 +1,1 @@
-old
+new
";
        let files = split_diff_by_file(diff);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "src/foo.rs");
        assert!(files[0].1.contains("@@ -1,1 +1,1 @@"));
    }

    #[test]
    fn split_diff_two_files() {
        let diff = "\
--- a/src/a.rs
+++ b/src/a.rs
@@ -1,1 +1,1 @@
-old_a
+new_a
--- a/src/b.rs
+++ b/src/b.rs
@@ -1,1 +1,1 @@
-old_b
+new_b
";
        let files = split_diff_by_file(diff);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].0, "src/a.rs");
        assert_eq!(files[1].0, "src/b.rs");
        assert!(files[0].1.contains("old_a"));
        assert!(files[1].1.contains("old_b"));
    }

    #[test]
    fn split_diff_no_b_prefix() {
        let diff = "\
--- src/foo.rs
+++ src/foo.rs
@@ -1,1 +1,1 @@
-x
+y
";
        let files = split_diff_by_file(diff);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "src/foo.rs");
    }

    #[test]
    fn parse_hunk_header_with_count() {
        let (start, count) = parse_hunk_header("@@ -5,3 +5,4 @@").unwrap();
        assert_eq!(start, 5);
        assert_eq!(count, 3);
    }

    #[test]
    fn parse_hunk_header_without_count() {
        // Single-line hunks omit the count: "@@ -5 +5 @@"
        let (start, count) = parse_hunk_header("@@ -5 +5 @@").unwrap();
        assert_eq!(start, 5);
        assert_eq!(count, 1); // default when count absent
    }

    #[test]
    fn apply_remove_only_hunk() {
        let content = "line1\nline2\nline3\n";
        let diff = "\
@@ -2,1 +2,0 @@
-line2
";
        let (result, _) = apply_diff(content, diff).unwrap();
        assert!(
            !result.contains("line2"),
            "line2 should be removed: {}",
            result
        );
        assert!(result.contains("line1"), "line1 should remain");
        assert!(result.contains("line3"), "line3 should remain");
    }

    // ── split_diff_by_file extra edge cases ──────────────────────────────────

    #[test]
    fn split_diff_empty_input() {
        let files = split_diff_by_file("");
        assert!(files.is_empty(), "empty diff should yield no files");
    }

    #[test]
    fn split_diff_strips_b_prefix() {
        // "+++ b/src/lib.rs" → path should be "src/lib.rs" (not "b/src/lib.rs")
        let diff = "+++ b/src/lib.rs\n@@ -1 +1 @@\n+new line\n";
        let files = split_diff_by_file(diff);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "src/lib.rs", "b/ prefix should be stripped");
    }

    #[test]
    fn apply_diff_no_hunks_returns_original() {
        let content = "line1\nline2\nline3\n";
        let diff = "+++ b/file.txt\n--- a/file.txt\n";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert_eq!(adjusted, 0);
        assert_eq!(
            result, content,
            "diff with no hunks should leave content unchanged"
        );
    }

    #[test]
    fn parse_hunk_header_minimal_at_at() {
        // @@ -1 +1 @@ (no count, no trailing text)
        let (start, count) = parse_hunk_header("@@ -1 +1 @@").unwrap();
        assert_eq!(start, 1, "start should be 1");
        assert_eq!(count, 1, "count should default to 1");
    }

    #[tokio::test]
    async fn apply_diff_blocks_symlink_write() {
        let dir = tempfile::TempDir::new().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, "line1\nline2\nline3\n").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let tool = ApplyDiffTool;
        let result = tool
            .call(json!({
                "path": link.to_str().unwrap(),
                "diff": "@@ -1,1 +1,1 @@\n-line1\n+changed\n"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "should refuse symlink write");
        assert!(
            result.content.contains("symlink"),
            "should mention symlink: {}",
            result.content
        );
        // Verify file was NOT modified
        assert_eq!(
            std::fs::read_to_string(&real).unwrap(),
            "line1\nline2\nline3\n"
        );
    }

    #[test]
    fn split_diff_skips_diff_git_header_lines() {
        let diff = "diff --git a/foo.rs b/foo.rs\nindex abc..def 100644\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1 +1 @@\n+fn bar() {}\n";
        let files = split_diff_by_file(diff);
        assert_eq!(files.len(), 1, "should parse exactly one file");
        assert_eq!(files[0].0, "foo.rs");
    }

    #[test]
    fn apply_diff_fuzzy_offset_finds_shifted_hunk() {
        // 3 extra lines inserted before the target, so hunk header says line 5
        // but actual content is at line 8
        let content = "extra1\nextra2\nextra3\na\nb\ntarget\nc\nd\n";
        let diff = "\
@@ -3,1 +3,1 @@
-target
+replaced
";
        // "target" is actually at line 6, but hunk says line 3
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert_eq!(adjusted, 1, "should report 1 adjusted hunk");
        assert!(result.contains("replaced"), "result: {}", result);
        assert!(
            !result.contains("target"),
            "target should be replaced: {}",
            result
        );
    }

    #[test]
    fn hunk_failure_shows_best_match() {
        let mut file_lines: Vec<String> = (0..80).map(|i| format!("filler_{}", i)).collect();
        file_lines.push("target_a".to_string());
        file_lines.push("target_b".to_string());
        file_lines.push("target_c".to_string());
        let content = file_lines.join("\n") + "\n";
        let diff = "@@ -1,3 +1,3 @@\n-target_a\n-target_b\n-target_c\n+replaced\n";
        let result = apply_diff(&content, diff);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Best match at line 81"),
            "should show best match location: {}",
            msg
        );
        assert!(msg.contains("3 of 3"), "should show match count: {}", msg);
    }

    #[test]
    fn hunk_failure_shows_first_line_location() {
        let mut file_lines: Vec<String> = (0..80).map(|i| format!("filler_{}", i)).collect();
        file_lines.push("unique_marker".to_string());
        let content = file_lines.join("\n") + "\n";
        let diff = "@@ -1,2 +1,1 @@\n-unique_marker\n-nonexistent_line\n+replaced\n";
        let result = apply_diff(&content, diff);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("First expected line") || msg.contains("Best match"),
            "should locate first line: {}",
            msg
        );
        assert!(
            msg.contains("81") || msg.contains("unique_marker"),
            "should show line number or content: {}",
            msg
        );
    }

    #[test]
    fn hunk_failure_fallback_shows_surrounding() {
        let content = "aaa\nbbb\nccc\nddd\neee\n";
        let diff = "@@ -3,1 +3,1 @@\n-zzz_not_here\n+replaced\n";
        let result = apply_diff(content, diff);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("ccc") || msg.contains("bbb") || msg.contains("ddd"),
            "fallback should show surrounding file content: {}",
            msg
        );
    }

    #[test]
    fn apply_diff_fuzzy_no_match_within_window() {
        // Content is shifted far beyond the ±30 window
        let mut lines: Vec<String> = (0..80).map(|i| format!("filler_{}", i)).collect();
        lines.push("target_line".to_string());
        let content = lines.join("\n") + "\n";
        // Hunk says line 1, but target is at line 81 — beyond ±30
        let diff = "\
@@ -1,1 +1,1 @@
-target_line
+replaced
";
        let result = apply_diff(&content, diff);
        assert!(
            result.is_err(),
            "should fail when content is beyond fuzz window"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "error should explain failure: {}",
            err_msg
        );
    }

    #[test]
    fn apply_diff_exact_match_still_works() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        let diff = "\
@@ -3,1 +3,1 @@
-line3
+LINE3
";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert_eq!(adjusted, 0, "exact match should not count as adjusted");
        assert!(result.contains("LINE3"), "result: {}", result);
        assert!(!result.contains("\nline3\n"), "old line3 should be gone");
    }

    #[test]
    fn apply_diff_fuzzy_multi_hunk_different_offsets() {
        // Two hunks, each off by different amounts
        let content = "padding\na\nb\nc\nd\npadding2\ne\nf\ng\n";
        // Hunk 1 says line 1 but "a" is at line 2 (off by 1)
        // Hunk 2 says line 5 but "e" is at line 7 (off by 2)
        let diff = "\
@@ -1,1 +1,1 @@
-a
+A
@@ -5,1 +5,1 @@
-e
+E
";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert!(adjusted >= 1, "at least one hunk should be adjusted");
        assert!(result.contains('A'), "result: {}", result);
        assert!(result.contains('E'), "result: {}", result);
    }

    #[test]
    fn apply_diff_has_system_prompt_hint() {
        let tool = ApplyDiffTool;
        let hint = tool.system_prompt_hint();
        assert!(
            hint.is_some(),
            "apply_diff should have a system_prompt_hint"
        );
        assert!(!hint.unwrap().is_empty(), "hint should be non-empty");
    }

    #[test]
    fn apply_diff_fuzzy_reports_adjustment_count() {
        // Two hunks: first is exact, second is off by 2
        let content = "a\nb\nc\nextra1\nextra2\nd\ne\n";
        let diff = "\
@@ -1,1 +1,1 @@
-a
+A
@@ -4,1 +4,1 @@
-d
+D
";
        let (result, adjusted) = apply_diff(content, diff).unwrap();
        assert!(result.contains('A'));
        assert!(result.contains('D'));
        // "d" is at line 6, hunk says line 4 → 1 adjustment
        assert_eq!(adjusted, 1, "exactly one hunk should need adjustment");
    }

    #[tokio::test]
    async fn apply_diff_binary_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("bin.dat");
        let mut content = b"line one\n".to_vec();
        content.push(0u8);
        content.extend_from_slice(b"line two\n");
        std::fs::write(&p, content).unwrap();
        let tool = ApplyDiffTool;
        let result = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "diff": "@@ -1,1 +1,1 @@\n-line one\n+changed\n"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "binary file should be rejected");
        assert!(
            result.content.contains("binary"),
            "should mention binary: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn apply_diff_large_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("huge.txt");
        let content = "a\n".repeat(6 * 1024 * 1024);
        std::fs::write(&p, &content).unwrap();
        let tool = ApplyDiffTool;
        let result = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "diff": "@@ -1,1 +1,1 @@\n-a\n+b\n"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "large file should be rejected");
        assert!(
            result.content.contains("too large"),
            "should mention too large: {}",
            result.content
        );
    }

    #[test]
    fn max_edit_file_size_is_reasonable() {
        const { assert!(crate::tools::path_safety::MAX_EDIT_FILE_SIZE >= 1_048_576) };
    }

    #[test]
    fn apply_diff_overlapping_hunks_detected() {
        let content = "line1\nline2\nline3\nline4\nline5\n";
        // Two hunks that overlap: first targets lines 2-3, second targets lines 3-4
        let diff = "@@ -2,2 +2,2 @@\n-line2\n-line3\n+LINE2\n+LINE3\n\
                     @@ -3,2 +3,2 @@\n-line3\n-line4\n+LINE3\n+LINE4\n";
        let result = apply_diff(content, diff);
        assert!(result.is_err(), "overlapping hunks should error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Overlapping"),
            "error should mention overlapping: {msg}"
        );
    }

    #[test]
    fn apply_diff_adjacent_hunks_ok() {
        let content = "line1\nline2\nline3\nline4\n";
        // Two adjacent hunks: first targets line 1-2, second targets line 3-4
        let diff = "@@ -1,2 +1,2 @@\n-line1\n-line2\n+LINE1\n+LINE2\n\
                     @@ -3,2 +3,2 @@\n-line3\n-line4\n+LINE3\n+LINE4\n";
        let (result, _) = apply_diff(content, diff).unwrap();
        assert_eq!(result, "LINE1\nLINE2\nLINE3\nLINE4\n");
    }

    #[test]
    fn apply_diff_multiple_hunks_reverse_order() {
        let content = "aaa\nbbb\nccc\nddd\neee\n";
        // Hunks in reverse file order: second hunk first, first hunk second
        let diff = "@@ -4,1 +4,1 @@\n-ddd\n+DDD\n\
                     @@ -2,1 +2,1 @@\n-bbb\n+BBB\n";
        let (result, _) = apply_diff(content, diff).unwrap();
        assert_eq!(result, "aaa\nBBB\nccc\nDDD\neee\n");
    }

    // --- error-hint retrofits (Cycle 69) ---------------------------------

    #[tokio::test]
    async fn missing_path_param_error_names_next_step() {
        // `path` is required — the error message must tell the caller
        // how to fix the call, not just that it's missing.
        let tool = ApplyDiffTool;
        let err = tool
            .call(json!({"diff": "@@ -1,1 +1,1 @@\n-a\n+b\n"}))
            .await
            .expect_err("missing path must error");
        let msg = format!("{}", err);
        assert!(msg.contains("path"), "error names field: {}", msg);
        assert!(
            msg.contains("Try:") || msg.contains("include"),
            "error names next step: {}",
            msg
        );
    }

    #[tokio::test]
    async fn missing_diff_param_error_names_unified_diff() {
        // When `diff` is missing, the hint must point at the unified
        // diff format (the format the tool accepts).
        let tool = ApplyDiffTool;
        let err = tool
            .call(json!({"path": "/tmp/nonexistent_x.txt"}))
            .await
            .expect_err("missing diff must error");
        let msg = format!("{}", err);
        assert!(msg.contains("diff"), "error names field: {}", msg);
        assert!(
            msg.to_lowercase().contains("unified") || msg.contains("diff -u"),
            "error names the format: {}",
            msg
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn read_failure_mentions_next_step() {
        // Reading a file that doesn't exist is caught by
        // `path_safety::check_file_editable`, which returns a
        // ToolResult { is_error: true, ... }. The content must route
        // through fs_error::format_read_error so the NotFound hint
        // ("Check: path exists") is appended.
        //
        // Uses an absolute path — no cwd manipulation needed, so the
        // test doesn't have to contend with the CWD_LOCK.
        let tool = ApplyDiffTool;
        let res = tool
            .call(json!({
                "path": "/tmp/dm_no_such_file_for_apply_diff_c69.txt",
                "diff": "@@ -1,1 +1,1 @@\n-a\n+b\n",
            }))
            .await
            .expect("tool returns Ok(ToolResult { is_error: true })");
        assert!(res.is_error, "missing file must flag is_error: {:?}", res);
        assert!(
            res.content.contains("Cannot read file"),
            "preserves prefix: {}",
            res.content
        );
        assert!(
            res.content.contains("Check:"),
            "fs_error NotFound hint appended: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn apply_failure_mentions_regenerate_diff() {
        // When the diff applies cleanly at the parse level but the
        // context doesn't match, the user needs to be told to regenerate
        // — re-reading the file and producing a fresh diff is the fix.
        // Uses an absolute path so no cwd manipulation is required.
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("target.txt");
        tokio::fs::write(&target, "hello\nworld\n").await.unwrap();

        let tool = ApplyDiffTool;
        // Diff removes a line that isn't in the file — forces apply_diff
        // to fail after the read succeeded.
        let res = tool
            .call(json!({
                "path": target.to_str().unwrap(),
                "diff": "@@ -1,1 +1,1 @@\n-completely-absent-line\n+replacement\n",
            }))
            .await;
        let err = res.expect_err("mismatched diff must error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("Failed to apply diff"),
            "preserves prefix: {}",
            msg
        );
        assert!(
            msg.contains("regenerate") || msg.contains("read_file"),
            "points at regenerate or read_file: {}",
            msg
        );
    }

    #[test]
    fn bad_hunk_header_shows_format() {
        // parse_hunk_header + its map_err caller must surface the
        // expected header format so the user can fix the diff.
        let content = "a\nb\n";
        let err = apply_diff(content, "@@ totally-bogus @@\n-a\n+A\n")
            .expect_err("bad header must error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("hunk header") || msg.contains("old_start"),
            "names the failing piece: {}",
            msg
        );
        assert!(
            msg.contains("@@") || msg.contains("diff -u") || msg.contains("git diff"),
            "shows the format or regen command: {}",
            msg
        );
    }

    #[test]
    fn parse_old_start_error_includes_header_context() {
        // Direct test on parse_hunk_header — when old_start isn't a
        // number, the error must name the header it failed on so the
        // caller can find it in the diff.
        let bad = "@@ -nan,1 +1,1 @@";
        let err = parse_hunk_header(bad).expect_err("non-numeric start must error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("'@@ -nan,1 +1,1 @@'") || msg.contains(bad),
            "echoes the header: {}",
            msg
        );
        assert!(
            msg.contains("integer") || msg.contains("positive"),
            "names the expected type: {}",
            msg
        );
    }

    /// Per-tool drift guard for `apply_diff`. Mirrors the `file_edit` /
    /// `multi_edit` / `file_write` siblings; same setup, different tool
    /// dispatch shape.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn apply_diff_appends_drift_warning_when_auto_ingest_disabled() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let file = proj.join("drift_probe.rs");
        std::fs::write(&file, "fn probe() {}\n").unwrap();
        {
            let wiki = crate::wiki::Wiki::open(&proj).unwrap();
            wiki.ingest_file(&proj, &file, "fn probe() {}\n").unwrap();
        }

        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        let res = ApplyDiffTool
            .call(json!({
                "path": "drift_probe.rs",
                "diff": "@@ -1,1 +1,1 @@\n-fn probe() {}\n+fn probe() { drifted }\n"
            }))
            .await
            .unwrap();

        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!res.is_error, "apply_diff should succeed: {}", res.content);
        assert!(
            res.content.contains("[wiki-drift]"),
            "should carry [wiki-drift] marker when auto-ingest is disabled: {}",
            res.content
        );
    }
}
