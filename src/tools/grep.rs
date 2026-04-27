use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }

    fn description(&self) -> &'static str {
        "Search file contents using ripgrep (rg) or grep. Supports regex patterns, file type filtering, and context lines. Returns matching lines with file paths and line numbers."
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use grep for searching file contents — don't use bash with grep/rg commands. \
              Supports full regex syntax. Use the glob parameter to filter by file type \
              (e.g. '*.rs'). Use context_lines to see surrounding code.",
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.rs')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive search (default: false)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context to show before and after each match (default: 0)"
                },
                "files_only": {
                    "type": "boolean",
                    "description": "Only return file names, not matching lines (default: false)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;

        let rg_available = which::which("rg").is_ok();

        let mut command = if rg_available {
            let mut cmd = Command::new("rg");
            cmd.args([
                "--no-heading",
                "--line-number",
                "--color=never",
                "--max-columns=500",
                "--glob=!.git",
            ]);
            cmd
        } else {
            let mut cmd = Command::new("grep");
            cmd.args(["-r", "-n", "-I", "--exclude-dir=.git"]);
            cmd
        };

        if args["case_insensitive"].as_bool().unwrap_or(false) {
            command.arg("-i");
        }
        if args["files_only"].as_bool().unwrap_or(false) {
            command.arg("-l");
        }
        if let Some(ctx) = args["context_lines"].as_u64() {
            if ctx > 0 {
                command.arg(format!("-C{}", ctx));
            }
        }
        if let Some(glob_pat) = args["glob"].as_str() {
            if rg_available {
                command.arg(format!("--glob={}", glob_pat));
            }
        }

        command.arg(pattern);
        let search_path = args["path"].as_str().unwrap_or(".");
        command.arg(search_path);

        // Validate search path if explicitly provided (skip for default ".")
        let mut warnings = Vec::new();
        if search_path != "." {
            let project_root = std::env::current_dir().unwrap_or_default();
            if let Ok(resolved) = super::path_safety::validate_path(search_path, &project_root) {
                warnings = super::path_safety::read_warnings(&resolved, search_path);
            }
        }

        let output = tokio::time::timeout(std::time::Duration::from_secs(30), command.output())
            .await
            .map_err(|_| anyhow::anyhow!("Grep timed out after 30s"))??;

        let exit_code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Exit code 2+ means error (invalid regex, bad path, permission denied)
        if exit_code >= 2 || (exit_code != 0 && exit_code != 1) {
            let err_text = stderr.trim();
            return Ok(ToolResult {
                content: format!(
                    "Grep error: {}\n\nPattern: {}, Path: {}\n\nTry: confirm the regex syntax is valid, the path exists, and you have read access.",
                    if err_text.is_empty() {
                        "unknown error"
                    } else {
                        err_text
                    },
                    pattern,
                    search_path
                ),
                is_error: true,
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let content = if stdout.trim().is_empty() {
            let path = std::path::Path::new(search_path);
            if search_path != "." && !path.exists() {
                format!(
                    "No matches — path '{}' does not exist. Check the path and try again.",
                    search_path
                )
            } else {
                format!(
                    "No matches found for pattern '{}' in '{}'.",
                    pattern, search_path
                )
            }
        } else {
            let total = stdout.lines().count();
            let lines: Vec<&str> = stdout.lines().take(250).collect();
            if total > 250 {
                format!(
                    "[Showing 250 of {} lines — refine your pattern or use a narrower path]\n{}",
                    total,
                    lines.join("\n")
                )
            } else {
                lines.join("\n")
            }
        };

        let mut final_content = String::new();
        for w in &warnings {
            final_content.push_str(w);
            final_content.push('\n');
        }
        final_content.push_str(&content);

        Ok(ToolResult {
            content: final_content,
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
        let t = GrepTool;
        assert_eq!(t.name(), "grep");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("pattern")));
        assert!(p["properties"]["path"].is_object());
        assert!(p["properties"]["case_insensitive"].is_object());
        assert!(p["properties"]["files_only"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(GrepTool.is_read_only());
    }

    #[tokio::test]
    async fn grep_finds_pattern_in_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "line with needle inside").unwrap();
        let result = GrepTool
            .call(json!({"pattern": "needle", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("needle"),
            "expected match in output: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_returns_no_matches_when_absent() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "nothing interesting here").unwrap();
        let result = GrepTool
            .call(json!({"pattern": "zzznomatch", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No matches"),
            "should contain 'No matches': {}",
            result.content
        );
        assert!(
            result.content.contains("zzznomatch"),
            "should include pattern searched: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "NEEDLE in a haystack").unwrap();
        let result = GrepTool
            .call(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "case_insensitive": true
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.to_lowercase().contains("needle"),
            "expected case-insensitive match: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_files_only_flag() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("match.txt");
        std::fs::write(&file, "needle is here").unwrap();
        let result = GrepTool
            .call(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "files_only": true
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("match.txt"),
            "expected filename in output: {}",
            result.content
        );
        assert!(
            !result.content.contains("needle is here"),
            "files_only should not include line content: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_missing_pattern_errors() {
        let err = GrepTool.call(json!({})).await.err().unwrap();
        assert!(
            err.to_string().contains("pattern"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn grep_output_truncated_at_250_lines() {
        let dir = TempDir::new().unwrap();
        // Write 300 lines each containing "match"
        let content: String = (0..300).map(|i| format!("match line {}\n", i)).collect();
        std::fs::write(dir.path().join("big.txt"), &content).unwrap();

        let result = GrepTool
            .call(json!({"pattern": "match", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("[Showing 250 of 300 lines"),
            "expected truncation notice in: {}",
            &result.content[..result.content.len().min(200)]
        );
        let notice_pos = result
            .content
            .find("[Showing")
            .expect("truncation notice missing");
        let match_pos = result
            .content
            .find("match line 0")
            .expect("first match missing");
        assert!(
            notice_pos < match_pos,
            "truncation notice should appear before match lines"
        );
    }

    #[tokio::test]
    async fn grep_context_lines_shows_surrounding() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("ctx.txt"), "before\nneedle\nafter\n").unwrap();

        let result = GrepTool
            .call(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "context_lines": 1
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("before") || result.content.contains("after"),
            "context lines should include surrounding content: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_multiple_files_finds_in_each() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("alpha.txt"), "needle in alpha").unwrap();
        std::fs::write(dir.path().join("beta.txt"), "needle in beta").unwrap();
        std::fs::write(dir.path().join("gamma.txt"), "no match here").unwrap();

        let result = GrepTool
            .call(json!({"pattern": "needle", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("alpha"),
            "alpha.txt should appear: {}",
            result.content
        );
        assert!(
            result.content.contains("beta"),
            "beta.txt should appear: {}",
            result.content
        );
        assert!(
            !result.content.contains("gamma"),
            "gamma.txt should not appear: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_line_number_shown_in_output() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("numbered.txt"),
            "first\nsecond needle\nthird\n",
        )
        .unwrap();
        let result = GrepTool
            .call(json!({"pattern": "needle", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // rg and grep -n both include line numbers
        assert!(
            result.content.contains('2'),
            "line 2 number should appear in output: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_warns_outside_project() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "needle").unwrap();
        let result = GrepTool
            .call(json!({"pattern": "needle", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // Temp dir is outside cwd
        assert!(
            result.content.contains("outside project"),
            "should warn about outside project: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_invalid_regex_returns_error() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "some content").unwrap();
        let result = GrepTool
            .call(json!({"pattern": "[invalid", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(
            result.is_error,
            "invalid regex should be an error: {}",
            result.content
        );
        assert!(
            result.content.contains("Grep error") || result.content.contains("error"),
            "should mention error: {}",
            result.content
        );
        assert!(
            result.content.contains("[invalid"),
            "should include the pattern: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_nonexistent_path_returns_error() {
        let result = GrepTool
            .call(json!({"pattern": "test", "path": "/tmp/dm_no_such_dir_xyz_abc_123"}))
            .await
            .unwrap();
        // rg returns exit code 2 for nonexistent paths
        assert!(
            result.is_error || result.content.contains("does not exist"),
            "nonexistent path should be error or mention nonexistence: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_no_matches_includes_pattern() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello world").unwrap();
        let result = GrepTool
            .call(json!({"pattern": "xyzzy_unique_pattern", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("xyzzy_unique_pattern"),
            "no-match message should include pattern: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_no_matches_shows_search_path() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello world").unwrap();
        let path_str = dir.path().to_str().unwrap();
        let result = GrepTool
            .call(json!({"pattern": "xyzzy_no_match", "path": path_str}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains(path_str),
            "no-match message should include search path: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_warns_on_symlink_path() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("file.txt"), "needle").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = GrepTool
            .call(json!({"pattern": "needle", "path": link.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("symlink"),
            "should warn about symlink: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("text.txt"), "needle in text").unwrap();
        std::fs::write(
            dir.path().join("binary.dat"),
            b"needle\x00\x01\x02\x03binary",
        )
        .unwrap();

        let result = GrepTool
            .call(json!({"pattern": "needle", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("text.txt"),
            "text match should appear: {}",
            result.content
        );
        assert!(
            !result.content.contains("binary.dat"),
            "binary file should be skipped: {}",
            result.content
        );
    }
}
