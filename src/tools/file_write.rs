use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

const MAX_WRITE_BYTES: usize = 10 * 1024 * 1024; // 10 MB

pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Write content to a file. Creates the file (and parent directories) if they don't exist. Overwrites existing content."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Only use write_file for creating new files. For modifying existing files, \
              prefer edit_file — it sends only the diff and is less error-prone. \
              This tool overwrites the entire file.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: content"))?;

        if content.len() > MAX_WRITE_BYTES {
            return Ok(ToolResult {
                content: format!(
                    "Error: content too large ({} bytes, max {} bytes). \
                     Split into multiple files or reduce content size.",
                    content.len(),
                    MAX_WRITE_BYTES
                ),
                is_error: true,
            });
        }

        let project_root = std::env::current_dir()?;
        let resolved = super::path_safety::validate_path(path_str, &project_root)?;

        if let Some(msg) = super::path_safety::check_write_blocked(&resolved, path_str, "write") {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        if let Some(msg) =
            super::path_safety::check_sensitive_blocked(&resolved.canonical, path_str, "write to")
        {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        let path = resolved.canonical.to_str().unwrap_or(path_str);

        let is_new = !std::path::Path::new(path).exists();

        if !is_new {
            if let Ok(meta) = std::fs::metadata(path) {
                if meta.is_dir() {
                    return Ok(ToolResult {
                        content: format!(
                            "Error: '{}' is a directory, not a file. Provide a file path.",
                            path
                        ),
                        is_error: true,
                    });
                }
            }
        }

        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let parent_str = parent.display().to_string();
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    let hint =
                        super::fs_error::hint_for(&e, &parent_str, super::fs_error::FsOp::Write);
                    anyhow::anyhow!("Cannot create directory '{}': {}{}", parent_str, e, hint)
                })?;
            }
        }

        tokio::fs::write(path, content).await.map_err(|e| {
            let hint = super::fs_error::hint_for(&e, path, super::fs_error::FsOp::Write);
            anyhow::anyhow!("Cannot write file '{}': {}{}", path, e, hint)
        })?;

        let formatted = if crate::config::auto_format_enabled() {
            crate::format::format_file(std::path::Path::new(path))
                .await
                .unwrap_or(false)
        } else {
            false
        };

        let line_count = content.lines().count();
        let scope_note = if !resolved.in_project {
            format!("Note: {} is outside the project directory.\n", path)
        } else {
            String::new()
        };
        let fmt_note = if formatted { " (formatted)" } else { "" };
        let verb = if is_new { "Created" } else { "Overwrote" };

        let drift_warning = crate::tools::post_write_wiki_sync(&resolved.canonical, content).await;

        let mut result_content = format!(
            "{}{} {} ({} lines){}",
            scope_note, verb, path, line_count, fmt_note
        );
        if let Some(w) = drift_warning {
            result_content.push_str("\n\n");
            result_content.push_str(&w);
        }

        Ok(ToolResult {
            content: result_content,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn name_and_schema() {
        let t = FileWriteTool;
        assert_eq!(t.name(), "write_file");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(required.contains(&json!("content")));
    }

    #[test]
    fn is_not_read_only() {
        assert!(!FileWriteTool.is_read_only());
    }

    #[tokio::test]
    async fn write_creates_new_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("out.txt");
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "hello"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a/b/c/file.txt");
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "nested"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(path.exists());
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.txt");
        FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "first"}))
            .await
            .unwrap();
        FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "second"}))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
    }

    #[tokio::test]
    async fn write_missing_path_errors() {
        let err = FileWriteTool.call(json!({})).await.err().unwrap();
        assert!(err.to_string().contains("path"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn write_missing_content_errors() {
        let err = FileWriteTool
            .call(json!({"path": "/tmp/x"}))
            .await
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("content"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn write_reports_line_count() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("lines.txt");
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "a\nb\nc"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("3 lines"),
            "expected '3 lines' in: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn write_empty_content_reports_zero_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.txt");
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": ""}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("0 lines"),
            "empty content should report 0 lines: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn write_new_file_says_created() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("brand_new.txt");
        assert!(!path.exists());
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "hello\n"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Created"),
            "new file should say Created: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn write_existing_file_says_overwrote() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("existing.txt");
        std::fs::write(&path, "original").unwrap();
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": "replaced\n"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Overwrote"),
            "existing file should say Overwrote: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn write_rejects_oversized_content() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("huge.txt");
        let big = "x".repeat(MAX_WRITE_BYTES + 1);
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": big}))
            .await
            .unwrap();
        assert!(result.is_error, "oversized content should be rejected");
        assert!(
            result.content.contains("too large"),
            "error should mention too large: {}",
            result.content
        );
        assert!(!path.exists(), "file should not be written");
    }

    #[tokio::test]
    async fn write_at_limit_succeeds() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("at_limit.txt");
        let content = "x".repeat(MAX_WRITE_BYTES);
        let result = FileWriteTool
            .call(json!({"path": path.to_str().unwrap(), "content": content}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "content at exactly MAX_WRITE_BYTES should succeed: {}",
            result.content
        );
        assert!(path.exists());
    }

    #[tokio::test]
    async fn write_rejects_directory_path() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("mydir");
        std::fs::create_dir(&sub).unwrap();
        let result = FileWriteTool
            .call(json!({"path": sub.to_str().unwrap(), "content": "data"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("is a directory"),
            "should say directory: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn write_allows_new_file_in_directory() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("newfile.txt");
        let result = FileWriteTool
            .call(json!({"path": file.to_str().unwrap(), "content": "hello"}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "new file should succeed: {}",
            result.content
        );
        assert!(file.exists());
    }

    #[tokio::test]
    async fn write_to_unwritable_path_errors() {
        let err = FileWriteTool
            .call(json!({"path": "/proc/dm_test_nonexistent/file.txt", "content": "x"}))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Cannot"),
            "error should mention 'Cannot': {err}"
        );
    }

    #[tokio::test]
    async fn write_to_env_file_blocked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let env_path = tmp.path().join(".env");
        let tool = FileWriteTool;
        let result = tool
            .call(serde_json::json!({
                "path": env_path.to_str().unwrap(),
                "content": "SECRET_KEY=abc123"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "should block writes to .env files");
        assert!(
            result.content.contains("Blocked"),
            "msg: {}",
            result.content
        );
        assert!(
            result.content.contains("secrets"),
            "msg: {}",
            result.content
        );
        assert!(!env_path.exists(), ".env file should not have been created");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn file_write_triggers_wiki_ingest() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DM_WIKI_AUTO_INGEST");
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let res = FileWriteTool
            .call(json!({"path": "probe.txt", "content": "marker-content"}))
            .await
            .unwrap();

        let _ = crate::wiki::wait_for_ingest_log_marker(&proj, "ingest | probe.txt").await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(!res.is_error, "write should succeed: {}", res.content);
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
    async fn write_to_ssh_key_blocked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ssh_dir = tmp.path().join(".ssh");
        std::fs::create_dir(&ssh_dir).unwrap();
        let key_path = ssh_dir.join("id_rsa");
        let tool = FileWriteTool;
        let result = tool
            .call(serde_json::json!({
                "path": key_path.to_str().unwrap(),
                "content": "PRIVATE KEY DATA"
            }))
            .await
            .unwrap();
        assert!(result.is_error, "should block writes to .ssh/ files");
        assert!(
            result.content.contains("Blocked"),
            "msg: {}",
            result.content
        );
    }

    /// Per-tool drift guard: a `file_write` against a file with a
    /// pre-existing entity page must propagate `[wiki-drift]` when
    /// auto-ingest is disabled. Mirrors `file_edit_appends_drift_warning_*`
    /// (`src/tools/file_edit.rs:692`) and `multi_edit_appends_drift_warning_*`
    /// (`src/tools/multi_edit.rs:719`). Closes the gap that the cycle-7
    /// registry-level test exposed for the other two edit tools.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn file_write_appends_drift_warning_when_auto_ingest_disabled() {
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

        let res = FileWriteTool
            .call(json!({
                "path": "drift_probe.rs",
                "content": "fn probe() { drifted }\n"
            }))
            .await
            .unwrap();

        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!res.is_error, "write should succeed: {}", res.content);
        assert!(
            res.content.contains("[wiki-drift]"),
            "should carry [wiki-drift] marker when auto-ingest is disabled: {}",
            res.content
        );
    }
}
