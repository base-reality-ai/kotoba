use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use similar::TextDiff;

pub struct FileEditTool;

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn description(&self) -> &'static str {
        "Edit a file by replacing an exact string with new content. The old_string must match exactly (including whitespace and indentation). For new files, use write_file instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to replace (must be unique in the file)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string"
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some("Always read a file before editing it. The old_string must match the file content \
              exactly — including whitespace and indentation. Provide enough surrounding context \
              in old_string to make the match unique. Prefer this over write_file for modifications.")
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let old_string = args["old_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: old_string"))?;
        let new_string = args["new_string"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: new_string"))?;

        if old_string.is_empty() {
            return Ok(ToolResult {
                content: "Error: old_string cannot be empty. Provide the exact text to replace."
                    .to_string(),
                is_error: true,
            });
        }

        if old_string == new_string {
            return Ok(ToolResult {
                content: "Error: old_string and new_string are identical — nothing to change."
                    .to_string(),
                is_error: true,
            });
        }

        let project_root = std::env::current_dir()?;
        let resolved = super::path_safety::validate_path(path_str, &project_root)?;

        if let Some(msg) = super::path_safety::check_write_blocked(&resolved, path_str, "edit") {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        if let Some(msg) =
            super::path_safety::check_sensitive_blocked(&resolved.canonical, path_str, "edit")
        {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        let path = resolved.canonical.to_str().unwrap_or(path_str);

        if let Some(err) = super::path_safety::check_file_editable(path, "edit").await {
            return Ok(err);
        }

        let original = tokio::fs::read_to_string(path).await.map_err(|e| {
            let hint = super::fs_error::hint_for(&e, path, super::fs_error::FsOp::Edit);
            anyhow::anyhow!("Cannot read file '{}': {}{}", path, e, hint)
        })?;

        let occurrences = original.matches(old_string).count();
        if occurrences == 0 {
            return Ok(ToolResult {
                content: diagnose_no_match(&original, old_string, path),
                is_error: true,
            });
        }
        if occurrences > 1 {
            return Ok(ToolResult {
                content: format!(
                    "Error: old_string found {} times in '{}'. Provide more context to make it unique.",
                    occurrences, path
                ),
                is_error: true,
            });
        }

        let new_content = original.replacen(old_string, new_string, 1);
        tokio::fs::write(path, &new_content).await.map_err(|e| {
            let hint = super::fs_error::hint_for(&e, path, super::fs_error::FsOp::Edit);
            anyhow::anyhow!("Cannot write file '{}': {}{}", path, e, hint)
        })?;

        // Build unified diff
        let diff = TextDiff::from_lines(&original, &new_content);
        let diff_text = diff
            .unified_diff()
            .header(&format!("a/{}", path), &format!("b/{}", path))
            .to_string();

        let formatted = if crate::config::auto_format_enabled() {
            crate::format::format_file(std::path::Path::new(path))
                .await
                .unwrap_or(false)
        } else {
            false
        };

        let scope_note = if !resolved.in_project {
            format!("Note: {} is outside the project directory.\n", path)
        } else {
            String::new()
        };
        let fmt_note = if formatted { " (formatted)" } else { "" };
        let mut content = format!(
            "{}Applied edit to {}{}\n\n{}",
            scope_note, path, fmt_note, diff_text
        );

        let drift_warning =
            crate::tools::post_write_wiki_sync(&resolved.canonical, &new_content).await;

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

pub(crate) fn diagnose_no_match(file_content: &str, old_string: &str, path: &str) -> String {
    let file_lines: Vec<&str> = file_content.lines().collect();
    let old_lines: Vec<&str> = old_string.lines().collect();

    // Check 1: whitespace-normalized match
    let norm_file: Vec<String> = file_lines.iter().map(|l| l.trim().to_string()).collect();
    let norm_old: Vec<String> = old_lines.iter().map(|l| l.trim().to_string()).collect();
    if !norm_old.is_empty() && norm_file.len() >= norm_old.len() {
        for i in 0..=norm_file.len() - norm_old.len() {
            if norm_file[i..i + norm_old.len()] == norm_old[..] {
                let end = (i + norm_old.len() + 1).min(file_lines.len());
                let actual: Vec<String> = file_lines[i..end]
                    .iter()
                    .enumerate()
                    .map(|(j, l)| format!("  {}: {}", i + j + 1, l))
                    .collect();
                return format!(
                    "Error: old_string not found in '{}' (but a whitespace-normalized match \
                     was found at line {}). The actual indentation differs. Here's what the file has:\n{}",
                    path,
                    i + 1,
                    actual.join("\n")
                );
            }
        }
    }

    // Check 2: best line overlap
    if old_lines.len() >= 2 {
        let check_count = old_lines.len().min(5);
        let mut best_pos = 0usize;
        let mut best_overlap = 0usize;
        for i in 0..file_lines.len() {
            let mut overlap = 0;
            for j in 0..check_count {
                if i + j < file_lines.len() && file_lines[i + j] == old_lines[j] {
                    overlap += 1;
                }
            }
            if overlap > best_overlap {
                best_overlap = overlap;
                best_pos = i;
            }
        }
        let threshold = if old_lines.len() <= 4 {
            2
        } else {
            check_count / 2
        };
        if best_overlap >= threshold {
            let start = best_pos;
            let end = (start + old_lines.len() + 2).min(file_lines.len());
            let context: Vec<String> = file_lines[start..end]
                .iter()
                .enumerate()
                .map(|(j, l)| format!("  {}: {}", start + j + 1, l))
                .take(10)
                .collect();
            return format!(
                "Error: old_string not found in '{}'.\n\
                 Best partial match at line {} ({} of {} lines match). The file has:\n{}",
                path,
                best_pos + 1,
                best_overlap,
                old_lines.len(),
                context.join("\n")
            );
        }
    }

    // Check 3: find first non-empty line
    if let Some(first_line) = old_lines.iter().find(|l| !l.trim().is_empty()) {
        if let Some(pos) = file_lines.iter().position(|l| l == first_line) {
            let start = pos.saturating_sub(1);
            let end = (pos + 4).min(file_lines.len());
            let context: Vec<String> = file_lines[start..end]
                .iter()
                .enumerate()
                .map(|(j, l)| format!("  {}: {}", start + j + 1, l))
                .collect();
            return format!(
                "Error: old_string not found in '{}'.\n\
                 The first line of your edit appears at line {}. Context:\n{}",
                path,
                pos + 1,
                context.join("\n")
            );
        }
    }

    // Final fallback
    let preview_count = 10.min(file_lines.len());
    let head: Vec<String> = file_lines
        .iter()
        .take(preview_count)
        .enumerate()
        .map(|(i, l)| format!("  {}: {}", i + 1, l))
        .collect();
    format!(
        "Error: old_string not found in '{}'. No similar content found.\n\
         The file has {} lines. First {} lines:\n{}",
        path,
        file_lines.len(),
        preview_count,
        head.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn name_and_schema() {
        let t = FileEditTool;
        assert_eq!(t.name(), "edit_file");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("old_string")));
        assert!(required.contains(&json!("new_string")));
    }

    #[test]
    fn is_not_read_only() {
        assert!(!FileEditTool.is_read_only());
    }

    async fn write_temp(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let p = dir.path().join(name);
        tokio::fs::write(&p, content).await.unwrap();
        p
    }

    #[tokio::test]
    async fn edit_basic_replacement() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "a.txt", "hello world\n").await;
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "goodbye"
            }))
            .await
            .unwrap();
        assert!(!res.is_error, "unexpected error: {}", res.content);
        let on_disk = tokio::fs::read_to_string(&p).await.unwrap();
        assert_eq!(on_disk, "goodbye world\n");
    }

    #[tokio::test]
    async fn edit_diff_included_in_output() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "b.txt", "foo\nbar\nbaz\n").await;
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "bar",
                "new_string": "qux"
            }))
            .await
            .unwrap();
        assert!(!res.is_error);
        // Unified diff markers should be present
        assert!(
            res.content.contains('-') || res.content.contains('+'),
            "no diff in output"
        );
    }

    #[tokio::test]
    async fn edit_old_string_not_found_is_error() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "c.txt", "hello world\n").await;
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "not_there",
                "new_string": "x"
            }))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(res.content.contains("not found"));
        // File must be unchanged
        let on_disk = tokio::fs::read_to_string(&p).await.unwrap();
        assert_eq!(on_disk, "hello world\n");
    }

    #[tokio::test]
    async fn edit_duplicate_old_string_is_error() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "d.txt", "foo foo foo\n").await;
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "bar"
            }))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("3 times") || res.content.contains("times"),
            "expected occurrence count in error: {}",
            res.content
        );
        // File must be unchanged
        let on_disk = tokio::fs::read_to_string(&p).await.unwrap();
        assert_eq!(on_disk, "foo foo foo\n");
    }

    #[tokio::test]
    async fn edit_missing_path_param_errors() {
        let tool = FileEditTool;
        let res = tool
            .call(json!({"old_string": "a", "new_string": "b"}))
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn edit_missing_old_string_param_errors() {
        let tool = FileEditTool;
        let res = tool
            .call(json!({"path": "/tmp/x.txt", "new_string": "b"}))
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn edit_nonexistent_file_is_error() {
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": "/tmp/dm_test_no_such_file_xyz.txt",
                "old_string": "a",
                "new_string": "b"
            }))
            .await
            .unwrap();
        assert!(res.is_error, "nonexistent file should return error result");
    }

    #[tokio::test]
    async fn edit_replaces_only_first_occurrence() {
        let dir = TempDir::new().unwrap();
        // Exactly one occurrence — unique context makes it unique
        let p = write_temp(&dir, "e.txt", "line one unique_marker_A\nline two\n").await;
        let tool = FileEditTool;
        let res = tool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "unique_marker_A",
                "new_string": "REPLACED"
            }))
            .await
            .unwrap();
        assert!(!res.is_error);
        let on_disk = tokio::fs::read_to_string(&p).await.unwrap();
        assert_eq!(on_disk, "line one REPLACED\nline two\n");
    }

    #[test]
    fn edit_not_found_shows_whitespace_hint() {
        let file = "    indented\n    more\n";
        let old = "\tindented\n\tmore\n";
        let msg = diagnose_no_match(file, old, "test.rs");
        assert!(
            msg.contains("whitespace"),
            "should mention whitespace: {}",
            msg
        );
        assert!(
            msg.contains("indented"),
            "should show actual content: {}",
            msg
        );
    }

    #[test]
    fn edit_not_found_shows_partial_match() {
        let file = "line1\nline2\nline3\nline4\nline5\n";
        let old = "line1\nline2\nTYPO\nline4\nline5\n";
        let msg = diagnose_no_match(file, old, "test.rs");
        assert!(
            msg.contains("partial match") || msg.contains("Best partial match"),
            "should show partial match: {}",
            msg
        );
        assert!(msg.contains("line 1"), "should reference line 1: {}", msg);
    }

    #[test]
    fn edit_not_found_shows_first_line_location() {
        let file = "aaa\nbbb\nccc\nddd\n";
        let old = "ccc\nxxx\nyyy\n";
        let msg = diagnose_no_match(file, old, "test.rs");
        assert!(
            msg.contains("line 3"),
            "should find first line at line 3: {}",
            msg
        );
    }

    #[test]
    fn edit_not_found_total_miss_shows_file_head() {
        let file = "alpha\nbeta\ngamma\n";
        let old = "zzz\nyyy\n";
        let msg = diagnose_no_match(file, old, "test.rs");
        assert!(
            msg.contains("No similar content"),
            "should show fallback: {}",
            msg
        );
        assert!(msg.contains("3 lines"), "should show line count: {}", msg);
        assert!(msg.contains("alpha"), "should show file head: {}", msg);
    }

    #[test]
    fn edit_not_found_total_miss_shows_10_line_preview() {
        let file = (1..=20)
            .map(|i| format!("line{}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let old = "zzz\nyyy\n";
        let msg = diagnose_no_match(&file, old, "test.rs");
        assert!(
            msg.contains("First 10 lines"),
            "should preview 10 lines: {}",
            msg
        );
        assert!(msg.contains("line1"), "should include first line: {}", msg);
        assert!(msg.contains("line10"), "should include 10th line: {}", msg);
        assert!(
            !msg.contains("line11"),
            "should not include 11th line: {}",
            msg
        );
    }

    #[test]
    fn edit_not_found_empty_file() {
        let msg = diagnose_no_match("", "something\n", "test.rs");
        assert!(
            msg.contains("0 lines") || msg.contains("No similar"),
            "should handle empty file: {}",
            msg
        );
    }

    #[tokio::test]
    async fn edit_binary_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("bin.dat");
        let mut content = b"header".to_vec();
        content.push(0u8);
        content.extend_from_slice(b"more");
        std::fs::write(&p, content).unwrap();
        let tool = FileEditTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "old_string": "header", "new_string": "changed"}))
            .await
            .unwrap();
        assert!(res.is_error, "binary file should be rejected");
        assert!(
            res.content.contains("binary"),
            "should mention binary: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn edit_large_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("huge.txt");
        let content = "a".repeat(11 * 1024 * 1024);
        std::fs::write(&p, &content).unwrap();
        let tool = FileEditTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "old_string": "a", "new_string": "b"}))
            .await
            .unwrap();
        assert!(res.is_error, "large file should be rejected");
        assert!(
            res.content.contains("too large"),
            "should mention too large: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn edit_normal_file_not_blocked_by_size_check() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("normal.txt");
        std::fs::write(&p, "hello world\n").unwrap();
        let tool = FileEditTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "old_string": "hello", "new_string": "goodbye"}))
            .await
            .unwrap();
        assert!(!res.is_error, "normal file should succeed: {}", res.content);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "goodbye world\n");
    }

    #[tokio::test]
    async fn edit_rejects_empty_old_string() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "empty.txt", "hello world\n").await;
        let res = FileEditTool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "",
                "new_string": "x"
            }))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("cannot be empty"),
            "msg: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn edit_rejects_identical_strings() {
        let dir = TempDir::new().unwrap();
        let p = write_temp(&dir, "same.txt", "hello world\n").await;
        let res = FileEditTool
            .call(json!({
                "path": p.to_str().unwrap(),
                "old_string": "hello",
                "new_string": "hello"
            }))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(res.content.contains("identical"), "msg: {}", res.content);
    }

    #[test]
    fn max_edit_file_size_is_reasonable() {
        const { assert!(crate::tools::path_safety::MAX_EDIT_FILE_SIZE >= 1_048_576) };
    }

    #[tokio::test]
    async fn edit_env_file_blocked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let env_path = tmp.path().join(".env.local");
        std::fs::write(&env_path, "OLD_KEY=val").unwrap();
        let tool = FileEditTool;
        let result = tool
            .call(serde_json::json!({
                "path": env_path.to_str().unwrap(),
                "old_string": "OLD_KEY=val",
                "new_string": "NEW_KEY=newval"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "should block edits to .env files");
        assert!(
            result.content.contains("Blocked"),
            "msg: {}",
            result.content
        );
        let contents = std::fs::read_to_string(&env_path).unwrap();
        assert_eq!(contents, "OLD_KEY=val", "file should be unchanged");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn file_edit_triggers_wiki_ingest() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DM_WIKI_AUTO_INGEST");
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let file = proj.join("probe.txt");
        std::fs::write(&file, "marker-original\n").unwrap();

        let res = FileEditTool
            .call(json!({
                "path": "probe.txt",
                "old_string": "marker-original",
                "new_string": "marker-edited"
            }))
            .await
            .unwrap();

        let _ = crate::wiki::wait_for_ingest_log_marker(&proj, "ingest | probe.txt").await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(!res.is_error, "edit should succeed: {}", res.content);
        let page = proj.join(".dm/wiki/entities/probe_txt.md");
        assert!(
            page.is_file(),
            "hook did not produce entity page: {:?}",
            page
        );
        let log = std::fs::read_to_string(proj.join(".dm/wiki/log.md")).unwrap();
        assert!(
            log.contains("ingest | probe.txt"),
            "log missing ingest line: {}",
            log
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn file_edit_appends_drift_warning_when_auto_ingest_disabled() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        // Create a source file and pre-ingest it so an entity page exists.
        let file = proj.join("drift_probe.rs");
        std::fs::write(&file, "fn probe() {}\n").unwrap();
        {
            let wiki = crate::wiki::Wiki::open(&proj).unwrap();
            wiki.ingest_file(&proj, &file, "fn probe() {}\n").unwrap();
        }

        // Now disable auto-ingest so the edit won't update the wiki.
        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        let res = FileEditTool
            .call(json!({
                "path": "drift_probe.rs",
                "old_string": "fn probe() {}",
                "new_string": "fn probe() { updated }"
            }))
            .await
            .unwrap();

        // Restore process-global state BEFORE assertions.
        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!res.is_error, "edit should succeed: {}", res.content);
        assert!(
            res.content.contains("[wiki-drift]"),
            "should carry [wiki-drift] marker when auto-ingest is disabled: {}",
            res.content
        );
        assert!(
            res.content.contains("may be stale"),
            "should append drift warning when auto-ingest is disabled: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn edit_ssh_path_blocked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ssh_dir = tmp.path().join(".ssh");
        std::fs::create_dir(&ssh_dir).unwrap();
        let config = ssh_dir.join("config");
        std::fs::write(&config, "Host *\n  User git").unwrap();
        let tool = FileEditTool;
        let result = tool
            .call(serde_json::json!({
                "path": config.to_str().unwrap(),
                "old_string": "User git",
                "new_string": "User root"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "should block edits to .ssh/ files");
        assert!(
            result.content.contains("Blocked"),
            "msg: {}",
            result.content
        );
    }
}
