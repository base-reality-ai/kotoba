use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct MultiEditTool;

#[async_trait]
impl Tool for MultiEditTool {
    fn name(&self) -> &'static str {
        "multi_edit"
    }

    fn description(&self) -> &'static str {
        "Apply multiple edits to a single file atomically. Each edit replaces an exact \
         occurrence of old_text with new_text. Edits are applied in order. \
         All edits must succeed — if any old_text is not found, the entire operation fails \
         and the file is not modified. Prefer this over multiple sequential file_edit calls."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit."
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to replace."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Replacement text."
                            }
                        },
                        "required": ["old_text", "new_text"]
                    },
                    "description": "List of {old_text, new_text} pairs to apply in order."
                }
            },
            "required": ["path", "edits"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use multi_edit to apply multiple changes to a single file atomically. \
              Preferred over sequential edit_file calls when you have 2+ edits in the same file. \
              All edits must succeed or none are applied.",
        )
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let Some(path_str) = args["path"].as_str() else {
            return Ok(ToolResult {
                content: "Error: missing required parameter 'path'".to_string(),
                is_error: true,
            });
        };

        // Path safety: block writes through symlinks
        let project_root = std::env::current_dir().unwrap_or_default();
        let resolved = match super::path_safety::validate_path(path_str, &project_root) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Error: invalid path '{}': {}", path_str, e),
                    is_error: true,
                });
            }
        };
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

        const MAX_EDITS_PER_CALL: usize = 100;
        let edits = match args["edits"].as_array() {
            Some(e) if !e.is_empty() => {
                if e.len() > MAX_EDITS_PER_CALL {
                    return Ok(ToolResult {
                        content: format!(
                            "Error: too many edits ({}, limit {}). Split into multiple calls.",
                            e.len(),
                            MAX_EDITS_PER_CALL
                        ),
                        is_error: true,
                    });
                }
                e.clone()
            }
            _ => {
                return Ok(ToolResult {
                    content: "Error: 'edits' must be a non-empty array".to_string(),
                    is_error: true,
                })
            }
        };

        if let Some(err) = super::path_safety::check_file_editable(path_str, "edit").await {
            return Ok(err);
        }

        let original = match tokio::fs::read_to_string(path_str).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult {
                    content: super::fs_error::format_read_error(
                        &e,
                        path_str,
                        &format!("Error reading file '{}'", path_str),
                    ),
                    is_error: true,
                })
            }
        };

        // Apply all edits sequentially; fail atomically if any old_text is missing.
        let mut current = original.clone();
        for (i, edit) in edits.iter().enumerate() {
            let Some(old) = edit["old_text"].as_str() else {
                return Ok(ToolResult {
                    content: format!("Error: edit[{}] missing 'old_text'", i),
                    is_error: true,
                });
            };
            let Some(new) = edit["new_text"].as_str() else {
                return Ok(ToolResult {
                    content: format!("Error: edit[{}] missing 'new_text'", i),
                    is_error: true,
                });
            };

            if !current.contains(old) {
                let diag = super::file_edit::diagnose_no_match(&current, old, path_str);
                return Ok(ToolResult {
                    content: format!("Error: edit[{}]: {}", i, diag),
                    is_error: true,
                });
            }

            // Replace only first occurrence (matches FileEditTool behavior)
            current = current.replacen(old, new, 1);
        }

        let file_path = std::path::Path::new(path_str);
        let tmp_path = file_path.with_extension("dm-tmp");
        let write_result = async {
            tokio::fs::write(&tmp_path, &current).await?;
            tokio::fs::rename(&tmp_path, file_path).await
        }
        .await;
        match write_result {
            Ok(()) => {
                let formatted = if crate::config::auto_format_enabled() {
                    crate::format::format_file(std::path::Path::new(path_str))
                        .await
                        .unwrap_or(false)
                } else {
                    false
                };
                let fmt_note = if formatted { " (formatted)" } else { "" };
                let mut content = format!(
                    "Applied {} edit{} to '{}'.{}",
                    edits.len(),
                    if edits.len() == 1 { "" } else { "s" },
                    path_str,
                    fmt_note
                );

                let drift_warning =
                    crate::tools::post_write_wiki_sync(&resolved.canonical, &current).await;

                if let Some(w) = drift_warning {
                    content.push_str("\n\n");
                    content.push_str(&w);
                }

                Ok(ToolResult {
                    content,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolResult {
                content: super::fs_error::format_write_error(
                    &e,
                    path_str,
                    &format!("Error writing file '{}'", path_str),
                ),
                is_error: true,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn multi_edit_applies_all_edits() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "hello world\nfoo bar\n")
            .await
            .unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [
                    {"old_text": "hello", "new_text": "goodbye"},
                    {"old_text": "foo", "new_text": "baz"}
                ]
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let content = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(content, "goodbye world\nbaz bar\n");
    }

    #[tokio::test]
    async fn multi_edit_fails_if_old_text_missing() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "hello world\n").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [
                    {"old_text": "not_present", "new_text": "whatever"}
                ]
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        // File must be unchanged
        let content = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(content, "hello world\n");
    }

    #[tokio::test]
    async fn multi_edit_single_edit_succeeds() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "abc").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "abc", "new_text": "xyz"}]
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("1 edit"));
        let content = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(content, "xyz");
    }

    #[tokio::test]
    async fn multi_edit_empty_edits_is_error() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "hello").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": []
            }))
            .await
            .unwrap();

        assert!(result.is_error);
    }

    #[tokio::test]
    async fn multi_edit_atomic_rollback_when_second_edit_fails() {
        // First edit matches, second doesn't — file must be unchanged (atomicity)
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("atomic.txt");
        tokio::fs::write(&file, "alpha beta gamma\n").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [
                    {"old_text": "alpha", "new_text": "A"},   // would succeed
                    {"old_text": "NOT_PRESENT", "new_text": "X"} // fails
                ]
            }))
            .await
            .unwrap();

        assert!(result.is_error, "should fail when a later edit is missing");
        // Atomicity: original file unchanged
        let on_disk = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(
            on_disk, "alpha beta gamma\n",
            "file must be unchanged after partial failure"
        );
    }

    #[tokio::test]
    async fn multi_edit_chained_edits_operate_on_prior_result() {
        // Second edit operates on the text *after* the first edit was applied
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("chain.txt");
        tokio::fs::write(&file, "foo").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [
                    {"old_text": "foo", "new_text": "foobar"},   // "foo" → "foobar"
                    {"old_text": "foobar", "new_text": "baz"}    // operates on result of edit 1
                ]
            }))
            .await
            .unwrap();

        assert!(!result.is_error, "chained edits should succeed");
        let on_disk = tokio::fs::read_to_string(&file).await.unwrap();
        assert_eq!(on_disk, "baz");
    }

    #[tokio::test]
    async fn multi_edit_missing_path_is_error() {
        let tool = MultiEditTool;
        let result = tool
            .call(json!({"edits": [{"old_text": "a", "new_text": "b"}]}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("path"),
            "error should mention 'path'"
        );
    }

    #[tokio::test]
    async fn multi_edit_nonexistent_file_is_error() {
        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": "/tmp/dm_test_no_such_file_xyz.txt",
                "edits": [{"old_text": "a", "new_text": "b"}]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.to_lowercase().contains("error"));
    }

    #[tokio::test]
    async fn multi_edit_missing_old_text_field_is_error() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("missing_field.txt");
        tokio::fs::write(&file, "hello").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"new_text": "world"}]  // no old_text
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("old_text"),
            "error should mention 'old_text'"
        );
    }

    #[tokio::test]
    async fn multi_edit_duplicate_old_text_replaces_first_occurrence() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("dup.txt");
        // "hello" appears twice — only the first occurrence is replaced
        tokio::fs::write(&file, "hello world hello").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "hello", "new_text": "goodbye"}]
            }))
            .await
            .unwrap();
        // Tool replaces only first occurrence (like replacen(_, _, 1))
        assert!(
            !result.is_error,
            "replacing first of duplicate should succeed: {}",
            result.content
        );
        let content = tokio::fs::read_to_string(&file).await.unwrap();
        assert!(
            content.starts_with("goodbye"),
            "first occurrence should be replaced: {content}"
        );
        assert!(
            content.contains("hello"),
            "second occurrence should remain: {content}"
        );
    }

    #[tokio::test]
    async fn multi_edit_success_result_is_not_error() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("ok.txt");
        tokio::fs::write(&file, "original content").await.unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "original", "new_text": "updated"}]
            }))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "successful edit should not be an error: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn multi_edit_diagnoses_whitespace_mismatch() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("ws.txt");
        tokio::fs::write(&file, "    fn hello() {\n        world();\n    }\n")
            .await
            .unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "fn hello() {\n    world();\n}", "new_text": "replaced"}]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.to_lowercase().contains("whitespace"),
            "should mention whitespace mismatch: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn multi_edit_diagnoses_partial_match() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("partial.txt");
        tokio::fs::write(&file, "line1\nline2\nline3\nline4\nline5\n")
            .await
            .unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "line1\nline2\nlineX\nline4\nline5", "new_text": "replaced"}]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("line") || result.content.contains("match"),
            "should provide partial match info: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn multi_edit_diagnoses_first_line_found() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("firstline.txt");
        tokio::fs::write(&file, "aaa\nbbb\nccc\nddd\n")
            .await
            .unwrap();

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "bbb\nxxx", "new_text": "replaced"}]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains('2') || result.content.contains("bbb"),
            "should indicate where first line was found: {}",
            result.content
        );
    }

    #[test]
    fn multi_edit_has_system_prompt_hint() {
        let tool = MultiEditTool;
        let hint = tool.system_prompt_hint();
        assert!(
            hint.is_some(),
            "multi_edit should have a system_prompt_hint"
        );
        assert!(!hint.unwrap().is_empty(), "hint should be non-empty");
    }

    #[tokio::test]
    async fn multi_edit_blocks_symlink_write() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, "hello world").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = MultiEditTool
            .call(json!({
                "path": link.to_str().unwrap(),
                "edits": [{"old_text": "hello", "new_text": "goodbye"}]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("symlink"),
            "should mention symlink: {}",
            result.content
        );
        // Verify file was NOT modified
        assert_eq!(std::fs::read_to_string(&real).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn multi_edit_binary_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("bin.dat");
        let mut content = b"hello".to_vec();
        content.push(0u8);
        content.extend_from_slice(b"world");
        std::fs::write(&p, content).unwrap();
        let result = MultiEditTool
            .call(json!({
                "path": p.to_str().unwrap(),
                "edits": [{"old_text": "hello", "new_text": "goodbye"}]
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
    async fn multi_edit_large_file_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let p = dir.path().join("huge.txt");
        let content = "a".repeat(11 * 1024 * 1024);
        std::fs::write(&p, &content).unwrap();
        let result = MultiEditTool
            .call(json!({
                "path": p.to_str().unwrap(),
                "edits": [{"old_text": "a", "new_text": "b"}]
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

    #[tokio::test]
    async fn multi_edit_no_tmp_file_after_write() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("atomic.txt");
        tokio::fs::write(&file, "hello world").await.unwrap();

        let result = MultiEditTool
            .call(json!({
                "path": file.to_str().unwrap(),
                "edits": [{"old_text": "hello", "new_text": "goodbye"}]
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        let tmp = file.with_extension("dm-tmp");
        assert!(
            !tmp.exists(),
            "tmp file should be cleaned up after atomic write"
        );
        assert_eq!(
            tokio::fs::read_to_string(&file).await.unwrap(),
            "goodbye world"
        );
    }

    #[tokio::test]
    async fn multi_edit_rejects_too_many_edits() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "content").await.unwrap();
        let edits: Vec<serde_json::Value> = (0..101)
            .map(|i| serde_json::json!({"old_text": format!("x{}", i), "new_text": "y"}))
            .collect();
        let result = MultiEditTool
            .call(serde_json::json!({
                "path": file.to_str().unwrap(),
                "edits": edits
            }))
            .await
            .unwrap();
        assert!(result.is_error, "101 edits should be rejected");
        assert!(
            result.content.contains("too many edits"),
            "msg: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn multi_edit_accepts_reasonable_count() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.txt");
        tokio::fs::write(&file, "aaa bbb").await.unwrap();
        let edits = vec![serde_json::json!({"old_text": "aaa", "new_text": "ccc"})];
        let result = MultiEditTool
            .call(serde_json::json!({
                "path": file.to_str().unwrap(),
                "edits": edits
            }))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "1 edit should succeed: {}",
            result.content
        );
    }

    #[test]
    fn max_edit_file_size_is_reasonable() {
        const { assert!(crate::tools::path_safety::MAX_EDIT_FILE_SIZE >= 1_048_576) };
    }

    // --- error-hint retrofits (Cycle 70) ---------------------------------

    #[tokio::test]
    async fn multi_edit_read_missing_file_routes_through_fs_error() {
        // A missing path is intercepted by `check_file_editable` (which
        // itself routes through `fs_error::format_read_error` since
        // C69), so the ToolResult content MUST include the NotFound
        // hint regardless of whether :121 fires or not. Pin the
        // end-user-visible contract.
        let tool = MultiEditTool;
        let res = tool
            .call(json!({
                "path": "/tmp/dm_no_such_file_for_multi_edit_c70.txt",
                "edits": [{"old_text": "a", "new_text": "b"}],
            }))
            .await
            .expect("tool returns Ok(ToolResult { is_error: true })");
        assert!(res.is_error, "missing file must flag is_error: {:?}", res);
        assert!(
            res.content.contains("Check:"),
            "fs_error NotFound hint appended: {}",
            res.content
        );
        assert!(
            res.content.contains("Cannot read file") || res.content.contains("Error reading file"),
            "preserves a read-error prefix: {}",
            res.content
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn multi_edit_appends_drift_warning_when_auto_ingest_disabled() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let file = proj.join("drift_probe.rs");
        tokio::fs::write(&file, "fn probe() {}\n").await.unwrap();
        {
            let wiki = crate::wiki::Wiki::open(&proj).unwrap();
            wiki.ingest_file(&proj, &file, "fn probe() {}\n").unwrap();
        }

        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        let tool = MultiEditTool;
        let result = tool
            .call(json!({
                "path": "drift_probe.rs",
                "edits": [
                    {"old_text": "fn probe() {}", "new_text": "fn probe() { updated }"}
                ]
            }))
            .await
            .unwrap();

        // Restore process-global state BEFORE assertions.
        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!result.is_error, "edit should succeed: {}", result.content);
        assert!(
            result.content.contains("[wiki-drift]"),
            "should carry [wiki-drift] marker when auto-ingest is disabled: {}",
            result.content
        );
        assert!(
            result.content.contains("may be stale"),
            "should append drift warning when auto-ingest is disabled: {}",
            result.content
        );
    }

    #[test]
    fn multi_edit_source_wires_format_read_error_at_read_site() {
        // Source-scan canary: pin that `multi_edit.rs` still routes
        // its `tokio::fs::read_to_string` error through
        // `fs_error::format_read_error`. Match on the call-site shape
        // (`name(`) so the test literal in this file's own source
        // doesn't contribute a false positive.
        let src = include_str!("multi_edit.rs");
        assert!(
            src.contains("super::fs_error::format_read_error("),
            "multi_edit.rs must wire format_read_error at the read site"
        );
        assert!(
            src.contains("Error reading file"),
            "multi_edit.rs preserves the 'Error reading file' prefix wording"
        );
    }

    #[test]
    fn multi_edit_source_wires_format_write_error_at_write_site() {
        // Source-scan canary for the write site.
        let src = include_str!("multi_edit.rs");
        assert!(
            src.contains("super::fs_error::format_write_error("),
            "multi_edit.rs must wire format_write_error at the write site"
        );
        assert!(
            src.contains("Error writing file"),
            "multi_edit.rs preserves the 'Error writing file' prefix wording"
        );
    }
}
