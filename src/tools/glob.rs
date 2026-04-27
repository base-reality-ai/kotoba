use std::fmt::Write as _;

use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &'static str {
        "glob"
    }

    fn description(&self) -> &'static str {
        "Find files matching a glob pattern. Returns file paths sorted by modification time (most recent first). Use for finding files by name or extension."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.toml')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current working directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use glob for finding files by name/extension — don't use bash with find/ls. \
              Common patterns: '**/*.rs' (all Rust files), 'src/**/*.ts' (TypeScript in src).",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: pattern"))?;
        let base = args["path"].as_str().map_or_else(
            || std::env::current_dir().unwrap_or_default(),
            std::path::PathBuf::from,
        );

        // Validate the search root if explicitly provided
        let mut warnings = Vec::new();
        if let Some(path_str) = args["path"].as_str() {
            let project_root = std::env::current_dir().unwrap_or_default();
            if let Ok(resolved) = super::path_safety::validate_path(path_str, &project_root) {
                warnings = super::path_safety::read_warnings(&resolved, path_str);
            }
        }

        let full_pattern = base.join(pattern).to_string_lossy().to_string();

        let mut matches: Vec<std::path::PathBuf> = glob::glob(&full_pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", pattern, e))?
            .filter_map(|r| r.ok())
            .filter(|p| p.is_file())
            .collect();

        // Sort by modification time descending
        matches.sort_by(|a, b| {
            let at = a.metadata().and_then(|m| m.modified()).ok();
            let bt = b.metadata().and_then(|m| m.modified()).ok();
            bt.cmp(&at)
        });

        let truncated = matches.len() > 100;
        let shown = &matches[..matches.len().min(100)];

        let cwd = std::env::current_dir().unwrap_or_default();
        let paths: Vec<String> = shown
            .iter()
            .map(|p| {
                p.strip_prefix(&cwd).map_or_else(
                    |_| p.to_string_lossy().to_string(),
                    |r| r.to_string_lossy().to_string(),
                )
            })
            .collect();

        let mut content = String::new();
        for w in &warnings {
            content.push_str(w);
            content.push('\n');
        }
        writeln!(content, "Found {} file(s)", shown.len()).expect("write to String never fails");
        content.push_str(&paths.join("\n"));
        if truncated {
            content.push_str("\n[Results truncated to 100 files]");
        }

        Ok(ToolResult {
            content,
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
        let t = GlobTool;
        assert_eq!(t.name(), "glob");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("pattern")));
        assert!(p["properties"]["path"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(GlobTool.is_read_only());
    }

    #[tokio::test]
    async fn glob_finds_matching_files() {
        let dir = TempDir::new().unwrap();
        for name in &["a.txt", "b.txt", "c.txt"] {
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let result = GlobTool
            .call(json!({"pattern": "*.txt", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("3 file"),
            "expected 3 files, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_returns_no_files_when_no_match() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        let result = GlobTool
            .call(json!({"pattern": "*.xyz", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("0 file"),
            "expected 0 files, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_respects_path_param() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("target.rs"), "").unwrap();
        // No .rs files in the parent dir itself, only in sub/
        let result = GlobTool
            .call(json!({"pattern": "*.rs", "path": sub.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("1 file"),
            "expected 1 file in subdir, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_invalid_pattern_errors() {
        let err = GlobTool
            .call(json!({"pattern": "[invalid"}))
            .await
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("Invalid glob pattern"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn glob_missing_pattern_errors() {
        let result = GlobTool.call(json!({})).await;
        assert!(result.is_err(), "missing pattern should error");
    }

    #[tokio::test]
    async fn glob_recursive_pattern_finds_nested_files() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("nested").join("deep");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(dir.path().join("top.rs"), "").unwrap();
        std::fs::write(sub.join("deep.rs"), "").unwrap();

        let result = GlobTool
            .call(json!({"pattern": "**/*.rs", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("2 file"),
            "expected 2 .rs files recursively, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_does_not_return_directories() {
        let dir = TempDir::new().unwrap();
        // Create a file and a sub-directory with the same extension (name-wise)
        std::fs::write(dir.path().join("real.txt"), "").unwrap();
        std::fs::create_dir_all(dir.path().join("fake.txt")).unwrap(); // dir named like a file

        let result = GlobTool
            .call(json!({"pattern": "*.txt", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        // Only the real file should be returned, not the directory
        assert!(
            result.content.contains("1 file"),
            "directory should not appear in results, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_result_content_contains_filename() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("needle.txt"), "").unwrap();
        let result = GlobTool
            .call(json!({"pattern": "needle.txt", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("needle.txt"),
            "matched filename should appear: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_multiple_extensions_via_star() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.rs"), "").unwrap();
        std::fs::write(dir.path().join("b.toml"), "").unwrap();
        let result = GlobTool
            .call(json!({"pattern": "*", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("2 file"),
            "both files should be returned: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_warns_on_symlink_path() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("file.txt"), "").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = GlobTool
            .call(json!({"pattern": "*.txt", "path": link.to_str().unwrap()}))
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
    async fn glob_warns_outside_project() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "").unwrap();
        let result = GlobTool
            .call(json!({"pattern": "*.txt", "path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("outside project"),
            "should warn about outside project: {}",
            result.content
        );
    }
}
