use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;

const MAX_ENTRIES: usize = 500;

pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &'static str {
        "ls"
    }

    fn description(&self) -> &'static str {
        "List the contents of a directory. Returns file names, types (file/dir/symlink), \
         and sizes. Defaults to current directory. Use this to explore the project structure \
         before reading specific files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory."
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with '.'). Default false."
                }
            },
            "required": []
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use ls to explore directory structure. Use glob to find files by pattern \
              and grep to search file contents — ls is only for directory listings.",
        )
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let path_str = args["path"].as_str().unwrap_or(".");
        let show_hidden = args["show_hidden"].as_bool().unwrap_or(false);

        // Path safety: warn on symlinks and out-of-project paths
        let mut warnings = Vec::new();
        if path_str != "." {
            let project_root = std::env::current_dir().unwrap_or_default();
            if let Ok(resolved) = super::path_safety::validate_path(path_str, &project_root) {
                warnings = super::path_safety::read_warnings(&resolved, path_str);
            }
        }

        let path = Path::new(path_str);
        if !path.exists() {
            return Ok(ToolResult {
                content: format!(
                    "path does not exist: {}. Try: confirm the path is correct.",
                    path_str
                ),
                is_error: true,
            });
        }
        if !path.is_dir() {
            return Ok(ToolResult {
                content: format!(
                    "not a directory: {}. Try: pass a directory path; for a single file use file_read.",
                    path_str
                ),
                is_error: true,
            });
        }

        let read_dir = match std::fs::read_dir(path) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!(
                        "error reading directory: {}. Try: confirm read access to the directory.",
                        e
                    ),
                    is_error: true,
                })
            }
        };

        let mut entries: Vec<(String, String, String)> = Vec::new(); // (name, kind, size)
        for entry in read_dir.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !show_hidden && name.starts_with('.') {
                continue;
            }
            let meta = entry.metadata();
            let (kind, size) = match meta {
                Ok(m) if m.is_dir() => ("dir".to_string(), "-".to_string()),
                Ok(m) if m.file_type().is_symlink() => ("symlink".to_string(), "-".to_string()),
                Ok(m) => ("file".to_string(), format_size(m.len())),
                Err(_) => ("?".to_string(), "?".to_string()),
            };
            entries.push((name, kind, size));
        }

        // Dirs first, then files; both alphabetical
        entries.sort_by(|a, b| {
            let a_is_dir = a.1 == "dir";
            let b_is_dir = b.1 == "dir";
            b_is_dir.cmp(&a_is_dir).then(a.0.cmp(&b.0))
        });

        let total_count = entries.len();
        let truncated = total_count > MAX_ENTRIES;
        if truncated {
            entries.truncate(MAX_ENTRIES);
        }

        if entries.is_empty() {
            return Ok(ToolResult {
                content: format!("{} (empty)", path_str),
                is_error: false,
            });
        }

        let mut rows = Vec::new();
        for w in &warnings {
            rows.push(w.clone());
        }
        rows.push(format!("Contents of {}:", path_str));
        for (name, kind, size) in &entries {
            let display_name = if kind == "dir" {
                format!("{}/", name)
            } else {
                name.clone()
            };
            rows.push(format!("  {:7}  {:8}  {}", kind, size, display_name));
        }

        if truncated {
            rows.push(format!(
                "\n[Showing {} of {} entries — use grep or glob to find specific files]",
                MAX_ENTRIES, total_count
            ));
        }

        Ok(ToolResult {
            content: rows.join("\n"),
            is_error: false,
        })
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}K", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1}M", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1}G", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn ls_lists_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "hello").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("subdir/"));
        assert!(result.content.contains("file.txt"));
    }

    #[tokio::test]
    async fn ls_nonexistent_path_is_error() {
        let tool = LsTool;
        let result = tool
            .call(json!({"path": "/nonexistent/path/xyz"}))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn ls_hidden_files_excluded_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "x").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "y").unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.content.contains(".hidden"));
        assert!(result.content.contains("visible.txt"));
    }

    #[tokio::test]
    async fn ls_shows_hidden_when_requested() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "x").unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap(), "show_hidden": true}))
            .await
            .unwrap();
        assert!(result.content.contains(".hidden"));
    }

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1023), "1023B");
    }

    #[test]
    fn format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(2048), "2.0K");
    }

    #[test]
    fn format_size_megabytes() {
        assert_eq!(format_size(1024 * 1024), "1.0M");
        assert_eq!(format_size(5 * 1024 * 1024), "5.0M");
    }

    #[test]
    fn format_size_gigabytes() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0G");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.0G");
        assert_eq!(format_size(10 * 1024 * 1024 * 1024), "10.0G");
    }

    #[tokio::test]
    async fn ls_dirs_come_before_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("aaa.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("zzz")).unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        let zzz_pos = result.content.find("zzz/").unwrap();
        let aaa_pos = result.content.find("aaa.txt").unwrap();
        assert!(zzz_pos < aaa_pos, "dirs should appear before files");
    }

    #[tokio::test]
    async fn ls_file_path_is_error() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("regular.txt");
        std::fs::write(&file, "content").unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": file.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(result.is_error, "passing a file path should be an error");
        assert!(result.content.contains("not a directory"));
    }

    #[tokio::test]
    async fn ls_empty_directory_reports_empty() {
        let dir = TempDir::new().unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("(empty)"),
            "empty directory should say '(empty)': {}",
            result.content
        );
    }

    #[test]
    fn format_size_zero_bytes() {
        assert_eq!(format_size(0), "0B");
    }

    #[tokio::test]
    async fn ls_warns_on_symlink_path() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::write(real.join("file.txt"), "content").unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let tool = LsTool;
        let result = tool
            .call(json!({"path": link.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "ls through symlink should succeed (read-only)"
        );
        assert!(
            result.content.contains("symlink"),
            "should warn about symlink: {}",
            result.content
        );
        assert!(
            result.content.contains("file.txt"),
            "should still list contents: {}",
            result.content
        );
    }

    #[test]
    fn format_size_exactly_1023_is_bytes() {
        assert_eq!(format_size(1023), "1023B");
    }

    #[test]
    fn format_size_exactly_1024_is_kilobytes() {
        let s = format_size(1024);
        assert!(s.ends_with('K'), "1024 bytes should be 1.0K: {s}");
    }

    #[tokio::test]
    async fn ls_caps_large_directory() {
        let dir = TempDir::new().unwrap();
        for i in 0..600 {
            std::fs::write(dir.path().join(format!("file_{:04}.txt", i)), "x").unwrap();
        }
        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        let entry_count = result
            .content
            .lines()
            .filter(|l| l.starts_with("  "))
            .count();
        assert_eq!(entry_count, MAX_ENTRIES, "should cap at MAX_ENTRIES");
        assert!(
            result.content.contains("Showing 500 of 600"),
            "should show truncation note: {}",
            &result.content[result.content.len().saturating_sub(100)..]
        );
    }

    #[tokio::test]
    async fn ls_small_directory_not_truncated() {
        let dir = TempDir::new().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{}.txt", i)), "x").unwrap();
        }
        let tool = LsTool;
        let result = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            !result.content.contains("Showing"),
            "should not truncate small dirs"
        );
        let entry_count = result
            .content
            .lines()
            .filter(|l| l.starts_with("  "))
            .count();
        assert_eq!(entry_count, 10);
    }
}
