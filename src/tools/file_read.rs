use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

const HARD_LIMIT: u64 = 10_485_760; // 10 MB
const STREAM_THRESHOLD: u64 = 1_048_576; // 1 MB
const MAX_LINE_CHARS: usize = 5000;

pub struct FileReadTool;

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Read the contents of a file from disk. Returns file content with line numbers. Use offset and limit to read specific sections of large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default 1)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 2000)"
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some("Always read a file before modifying it. For large files, use offset and limit \
              to read only the section you need. Don't re-read a file you just successfully edited.")
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;
        let offset = args["offset"].as_u64().unwrap_or(1).saturating_sub(1) as usize;
        let limit = args["limit"].as_u64().unwrap_or(2000) as usize;

        let project_root = std::env::current_dir()?;
        let resolved = super::path_safety::validate_path(path_str, &project_root)?;
        let path = resolved.canonical.to_str().unwrap_or(path_str);
        let warnings = super::path_safety::read_warnings(&resolved, path_str);

        let meta = tokio::fs::metadata(path).await.map_err(|e| {
            let cwd = std::env::current_dir()
                .map_or_else(|_| "(unknown)".to_string(), |p| p.display().to_string());
            let hint = super::fs_error::hint_for(&e, path, super::fs_error::FsOp::Read);
            anyhow::anyhow!(
                "Cannot read file '{}': {}\nCurrent directory: {}{}",
                path,
                e,
                cwd,
                hint
            )
        })?;
        let file_size = meta.len();

        if !meta.is_file() {
            return Ok(ToolResult {
                content: format!(
                    "Cannot read '{}': not a regular file{}",
                    path_str,
                    if meta.is_dir() {
                        " (is a directory)"
                    } else {
                        " (special file — device, socket, or FIFO)"
                    }
                ),
                is_error: true,
            });
        }

        if file_size > HARD_LIMIT {
            let size_mb = file_size as f64 / 1_048_576.0;
            return Ok(ToolResult {
                content: format!(
                    "File is {:.1}MB — too large to read. Use offset and limit to read specific sections, \
                     or use grep to search for specific content.",
                    size_mb
                ),
                is_error: true,
            });
        }

        if crate::index::chunker::is_binary(std::path::Path::new(path)) {
            return Ok(ToolResult {
                content: format!(
                    "File '{}' appears to be binary ({:.1}KB). \
                     Binary files cannot be meaningfully displayed as text. \
                     Use bash with `file`, `hexdump`, or `strings` for binary inspection.",
                    path_str,
                    file_size as f64 / 1024.0
                ),
                is_error: true,
            });
        }

        let (selected, total) = if file_size > STREAM_THRESHOLD {
            read_lines_streaming(path, offset, limit).await?
        } else {
            let content = match tokio::fs::read_to_string(path).await {
                Ok(c) => c,
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                    return Ok(ToolResult {
                        content: format!(
                            "File '{}' contains invalid UTF-8. It may be a binary file. \
                             Use bash with `file` or `strings` to inspect.",
                            path_str
                        ),
                        is_error: true,
                    });
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(super::fs_error::format_read_error(
                        &e,
                        path,
                        &format!("Cannot read file '{}'", path),
                    )));
                }
            };
            let lines: Vec<String> = content.lines().map(String::from).collect();
            let total = lines.len();
            let end = (offset + limit).min(total);
            let selected = lines[offset.min(total)..end].to_vec();
            (selected, total)
        };

        let end = offset + selected.len();

        let mut long_lines_truncated = 0usize;
        let numbered: String = selected
            .iter()
            .enumerate()
            .map(|(i, line)| {
                if line.len() > MAX_LINE_CHARS {
                    long_lines_truncated += 1;
                    let cut = crate::util::safe_truncate(line, MAX_LINE_CHARS);
                    format!(
                        "{}\t{}... [line truncated — {} chars total]",
                        offset + i + 1,
                        cut,
                        line.len()
                    )
                } else {
                    format!("{}\t{}", offset + i + 1, line)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let header = if total > end {
            format!(
                "[File truncated: showing lines {}-{} of {}]\n",
                offset + 1,
                end,
                total
            )
        } else {
            String::new()
        };

        let long_line_note = if long_lines_truncated > 0 {
            format!(
                "[Note: {} line{} truncated at {} chars — file may contain minified code]\n",
                long_lines_truncated,
                if long_lines_truncated == 1 { "" } else { "s" },
                MAX_LINE_CHARS
            )
        } else {
            String::new()
        };

        let warn_prefix = if warnings.is_empty() {
            String::new()
        } else {
            format!("{}\n", warnings.join("\n"))
        };

        // Wiki auto-ingest — best-effort, non-blocking. Fires only on the
        // success path; the error returns above skip this entirely. Failure
        // here must never surface in the tool response — surface via warnings.
        let ingest_path = resolved.canonical.clone();
        let ingest_root = project_root.clone();
        let ingest_content = selected.join("\n");
        tokio::task::spawn_blocking(move || {
            if let Ok(wiki) = crate::wiki::Wiki::open(&ingest_root) {
                if let Err(e) = wiki.ingest_file(&ingest_root, &ingest_path, &ingest_content) {
                    crate::warnings::push_warning(format!(
                        "wiki ingest failed for {}: {}",
                        ingest_path.display(),
                        e
                    ));
                }
            }
        });

        Ok(ToolResult {
            content: format!("{}{}{}{}", warn_prefix, header, long_line_note, numbered),
            is_error: false,
        })
    }
}

async fn read_lines_streaming(
    path: &str,
    offset: usize,
    limit: usize,
) -> Result<(Vec<String>, usize)> {
    let file = tokio::fs::File::open(path).await.map_err(|e| {
        anyhow::anyhow!(super::fs_error::format_read_error(
            &e,
            path,
            &format!("Cannot read file '{}'", path),
        ))
    })?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut line_num = 0usize;
    let mut selected = Vec::with_capacity(limit);

    while let Some(line) = lines.next_line().await? {
        if line_num >= offset && selected.len() < limit {
            selected.push(line);
        }
        line_num += 1;
        if line_num > offset + limit && selected.len() >= limit {
            // Count remaining lines without storing them
            while lines.next_line().await?.is_some() {
                line_num += 1;
            }
            break;
        }
    }

    Ok((selected, line_num))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn name_and_description() {
        let t = FileReadTool;
        assert_eq!(t.name(), "read_file");
        assert!(!t.description().is_empty());
    }

    #[test]
    fn parameters_schema() {
        let t = FileReadTool;
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(p["properties"]["offset"].is_object());
        assert!(p["properties"]["limit"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(FileReadTool.is_read_only());
    }

    const FIVE_LINES: &str = "alpha\nbeta\ngamma\ndelta\nepsilon\n";

    async fn temp_file(content: &str) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("f.txt");
        tokio::fs::write(&p, content).await.unwrap();
        (dir, p)
    }

    #[tokio::test]
    async fn read_full_file_has_line_numbers() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(res.content.contains("1\talpha"));
        assert!(res.content.contains("5\tepsilon"));
    }

    #[tokio::test]
    async fn read_with_offset_skips_leading_lines() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "offset": 3}))
            .await
            .unwrap();
        assert!(!res.is_error);
        // Line 3 onwards: gamma(3), delta(4), epsilon(5)
        assert!(
            !res.content.contains("1\talpha"),
            "line 1 should be skipped"
        );
        assert!(res.content.contains("3\tgamma"));
        assert!(res.content.contains("5\tepsilon"));
    }

    #[tokio::test]
    async fn read_with_limit_shows_truncation_header() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "limit": 2}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(
            res.content.contains("truncated") || res.content.contains("File truncated"),
            "expected truncation notice, got: {}",
            res.content
        );
        assert!(res.content.contains("1\talpha"));
        assert!(res.content.contains("2\tbeta"));
        assert!(
            !res.content.contains("3\tgamma"),
            "gamma should be truncated"
        );
    }

    #[tokio::test]
    async fn read_offset_beyond_file_returns_empty_content() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "offset": 100}))
            .await
            .unwrap();
        assert!(!res.is_error);
        // No line numbers expected — file has only 5 lines
        assert!(
            !res.content.contains('\t'),
            "no numbered lines expected: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_nonexistent_file_is_error() {
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": "/tmp/dm_no_such_file_xyz_read.txt"}))
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn read_missing_path_param_errors() {
        let tool = FileReadTool;
        let res = tool.call(json!({})).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn read_offset_zero_same_as_offset_one() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res0 = tool
            .call(json!({"path": p.to_str().unwrap(), "offset": 0}))
            .await
            .unwrap();
        let res1 = tool
            .call(json!({"path": p.to_str().unwrap(), "offset": 1}))
            .await
            .unwrap();
        assert_eq!(res0.content, res1.content);
    }

    #[tokio::test]
    async fn read_empty_file_returns_empty_content() {
        let (_dir, p) = temp_file("").await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error);
        // Header is empty when no truncation; numbered content is empty too.
        assert!(res.content.is_empty() || !res.content.contains('\t'));
    }

    #[tokio::test]
    async fn read_limit_one_shows_only_first_line() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "limit": 1}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(res.content.contains("1\talpha"), "first line should appear");
        assert!(
            !res.content.contains("2\tbeta"),
            "second line should not appear"
        );
    }

    #[tokio::test]
    async fn read_truncation_header_shows_correct_range() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "limit": 3}))
            .await
            .unwrap();
        assert!(
            res.content.contains("1-3"),
            "header should show lines 1-3: {}",
            res.content
        );
        assert!(
            res.content.contains('5'),
            "header should reference total of 5: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn large_file_rejected() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("huge.txt");
        // 11 MB file
        let chunk = "x".repeat(1024) + "\n";
        let content = chunk.repeat(11 * 1024);
        tokio::fs::write(&p, &content).await.unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("too large"),
            "expected size guidance: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn medium_file_streams_with_offset() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("medium.txt");
        // ~1.5 MB file: 15000 lines of ~100 chars each
        let lines: Vec<String> = (0..15000)
            .map(|i| format!("line-{:05}-{}", i, "x".repeat(90)))
            .collect();
        let content = lines.join("\n") + "\n";
        tokio::fs::write(&p, &content).await.unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap(), "offset": 100, "limit": 5}))
            .await
            .unwrap();
        assert!(!res.is_error, "medium file should succeed: {}", res.content);
        assert!(
            res.content.contains("100\tline-00099"),
            "should start at line 100: {}",
            res.content
        );
        assert!(
            res.content.contains("104\tline-00103"),
            "should include line 104: {}",
            res.content
        );
        assert!(
            res.content.contains("truncated"),
            "should show truncation: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_binary_file_returns_error() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("image.bin");
        let mut content = b"some header text".to_vec();
        content.push(0u8); // null byte → binary
        content.extend_from_slice(b"more data");
        std::fs::write(&p, &content).unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(res.is_error, "binary file should return error");
        assert!(
            res.content.contains("binary"),
            "should mention binary: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_binary_suggests_alternatives() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("data.bin");
        std::fs::write(&p, [0u8; 64]).unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("hexdump")
                || res.content.contains("strings")
                || res.content.contains("file"),
            "should suggest alternatives: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_utf8_invalid_returns_error() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("bad_utf8.txt");
        // Valid ASCII with invalid UTF-8 sequence in the middle (no null bytes so is_binary passes)
        let mut content = b"hello ".to_vec();
        content.extend_from_slice(&[0xFF, 0xFE]); // invalid UTF-8
        content.extend_from_slice(b" world");
        std::fs::write(&p, &content).unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(res.is_error, "invalid UTF-8 should return error");
        assert!(
            res.content.contains("UTF-8") || res.content.contains("binary"),
            "should mention UTF-8 or binary: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn read_long_lines_are_truncated() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("minified.js");
        let long_line = "x".repeat(10_000);
        let content = format!("short line\n{}\nanother short\n", long_line);
        tokio::fs::write(&p, &content).await.unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(
            res.content.contains("line truncated"),
            "should indicate truncation: {}",
            &res.content[..200.min(res.content.len())]
        );
        assert!(
            res.content.contains("10000 chars total"),
            "should show original length"
        );
        assert!(
            res.content.contains("1 line"),
            "should note 1 line truncated"
        );
        assert!(
            res.content.contains("1\tshort line"),
            "short lines should be unchanged"
        );
    }

    #[tokio::test]
    async fn read_line_at_boundary_not_truncated() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("boundary.txt");
        let exact = "y".repeat(MAX_LINE_CHARS);
        tokio::fs::write(&p, &exact).await.unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(
            !res.content.contains("truncated"),
            "exact limit should not truncate"
        );
    }

    #[tokio::test]
    async fn small_file_unchanged_behavior() {
        let (_dir, p) = temp_file(FIVE_LINES).await;
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": p.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error);
        assert!(res.content.contains("1\talpha"));
        assert!(res.content.contains("5\tepsilon"));
        assert!(!res.content.contains("truncated"));
    }

    #[tokio::test]
    async fn file_read_rejects_directory() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": dir.path().to_str().unwrap()}))
            .await
            .unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("not a regular file"),
            "msg: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn file_read_rejects_dev_null() {
        let tool = FileReadTool;
        let res = tool.call(json!({"path": "/dev/null"})).await.unwrap();
        assert!(
            res.is_error,
            "dev/null is a character device, not a regular file"
        );
        assert!(
            res.content.contains("not a regular file"),
            "msg: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn file_read_accepts_regular_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("regular.txt");
        std::fs::write(&file, "hello world").unwrap();
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": file.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!res.is_error, "regular file should work: {}", res.content);
        assert!(res.content.contains("hello world"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn file_read_triggers_wiki_ingest() {
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("DM_WIKI_AUTO_INGEST");
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let file = proj.join("probe.txt");
        std::fs::write(&file, "marker-content").unwrap();

        let res = FileReadTool
            .call(json!({"path": "probe.txt"}))
            .await
            .unwrap();

        let _ = crate::wiki::wait_for_ingest_log_marker(&proj, "ingest | probe.txt").await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(!res.is_error, "read should succeed: {}", res.content);
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

    // --- error-hint retrofits (Cycle 69) ---------------------------------

    #[tokio::test]
    async fn read_nonexistent_file_error_has_check_hint() {
        // Reading a missing file hits the metadata map_err at the top of
        // `call`. That path already routed through fs_error::hint_for —
        // pin the invariant that the "Check: path exists" hint survives.
        let tool = FileReadTool;
        let res = tool
            .call(json!({"path": "/tmp/dm_no_such_file_xyz_c69.txt"}))
            .await;
        let err = res.expect_err("missing file should error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("Cannot read file"),
            "preserves prefix: {}",
            msg
        );
        assert!(msg.contains("Check:"), "NotFound hint must appear: {}", msg);
    }

    #[tokio::test]
    async fn streaming_read_error_at_open_has_hint() {
        // The streaming path opens the file with `tokio::fs::File::open`.
        // To reach it we need a file above STREAM_THRESHOLD (1 MB) whose
        // metadata succeeds but whose open fails. The only portable way
        // to engineer this is TOCTOU: create a large file, let the
        // metadata call succeed, then verify the streaming open path's
        // error message routes through fs_error::format_read_error.
        //
        // We can't race reliably, so we directly unit-test the streaming
        // helper against a nonexistent path — metadata hasn't been run
        // first, but map_err at the open call will still trigger.
        let res = read_lines_streaming("/tmp/dm_no_such_streaming_xyz_c69.txt", 0, 10).await;
        let err = res.expect_err("missing file should error");
        let msg = format!("{}", err);
        assert!(
            msg.contains("Cannot read file"),
            "preserves prefix: {}",
            msg
        );
        assert!(
            msg.contains("Check:"),
            "NotFound hint must appear on streaming path: {}",
            msg
        );
    }
}
