use super::{Tool, ToolResult};
use async_trait::async_trait;
use serde_json::{json, Value};

// ── NotebookReadTool ──────────────────────────────────────────────────────────

pub struct NotebookReadTool;

#[async_trait]
impl Tool for NotebookReadTool {
    fn name(&self) -> &'static str {
        "notebook_read"
    }

    fn description(&self) -> &'static str {
        "Read a Jupyter notebook (.ipynb) file. Returns all cells with their type \
         (code/markdown), source content, and output text. Use this to understand \
         notebook content before editing it."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the .ipynb file."
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use notebook to read and edit Jupyter notebook cells. \
              Supports code, markdown, and output cells.",
        )
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let Some(path) = args["path"].as_str() else {
            return Ok(ToolResult {
                content: "Error: missing required parameter 'path'".to_string(),
                is_error: true,
            });
        };

        // Path safety: warn on symlinks and out-of-project paths
        let project_root = std::env::current_dir().unwrap_or_default();
        let mut warn_prefix = String::new();
        if let Ok(resolved) = super::path_safety::validate_path(path, &project_root) {
            for w in super::path_safety::read_warnings(&resolved, path) {
                warn_prefix.push_str(&w);
                warn_prefix.push('\n');
            }
        }

        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult {
                    content: super::fs_error::format_read_error(
                        &e,
                        path,
                        &format!("Error reading '{}'", path),
                    ),
                    is_error: true,
                })
            }
        };

        let notebook: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Error parsing notebook '{}': {}", path, e),
                    is_error: true,
                })
            }
        };

        let Some(cells) = notebook["cells"].as_array() else {
            return Ok(ToolResult {
                content: format!(
                    "'{}' does not appear to be a valid notebook (no 'cells' array)",
                    path
                ),
                is_error: true,
            });
        };

        let mut parts = vec![format!("Notebook: {}\n({} cells)\n", path, cells.len())];

        for (i, cell) in cells.iter().enumerate() {
            let cell_type = cell["cell_type"].as_str().unwrap_or("unknown");
            let source = join_source(&cell["source"]);
            parts.push(format!("## Cell {} [{}]\n{}", i, cell_type, source));

            // Include outputs for code cells
            if cell_type == "code" {
                if let Some(outputs) = cell["outputs"].as_array() {
                    let output_text: Vec<String> = outputs
                        .iter()
                        .filter_map(|o| {
                            if o["text"].is_array() {
                                Some(join_source(&o["text"]))
                            } else if o["data"]["text/plain"].is_array() {
                                Some(join_source(&o["data"]["text/plain"]))
                            } else {
                                o["ename"].as_str().map(|ename| {
                                    format!(
                                        "ERROR: {} — {}",
                                        ename,
                                        o["evalue"].as_str().unwrap_or("")
                                    )
                                })
                            }
                        })
                        .collect();

                    if !output_text.is_empty() {
                        parts.push(format!("### Output\n{}", output_text.join("")));
                    }
                }
            }
        }

        Ok(ToolResult {
            content: format!("{}{}", warn_prefix, parts.join("\n")),
            is_error: false,
        })
    }
}

// ── NotebookEditTool ──────────────────────────────────────────────────────────

pub struct NotebookEditTool;

#[async_trait]
impl Tool for NotebookEditTool {
    fn name(&self) -> &'static str {
        "notebook_edit"
    }

    fn description(&self) -> &'static str {
        "Edit a cell in a Jupyter notebook (.ipynb) file. Replaces the source of a \
         specific cell identified by its index. Use notebook_read first to see cell \
         indices and content. Automatically clears stale outputs when editing a code cell."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the .ipynb file."
                },
                "cell_index": {
                    "type": "integer",
                    "description": "Zero-based index of the cell to edit."
                },
                "new_source": {
                    "type": "string",
                    "description": "New source content for the cell."
                },
                "cell_type": {
                    "type": "string",
                    "enum": ["code", "markdown"],
                    "description": "Optionally change the cell type."
                }
            },
            "required": ["path", "cell_index", "new_source"]
        })
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let Some(path) = args["path"].as_str() else {
            return Ok(ToolResult {
                content: "Error: missing required parameter 'path'".to_string(),
                is_error: true,
            });
        };

        // Path safety: block writes through symlinks
        let project_root = std::env::current_dir().unwrap_or_default();
        if let Ok(resolved) = super::path_safety::validate_path(path, &project_root) {
            if let Some(msg) =
                super::path_safety::check_write_blocked(&resolved, path, "edit notebook")
            {
                return Ok(ToolResult {
                    content: msg,
                    is_error: true,
                });
            }
            if let Some(msg) = super::path_safety::check_sensitive_blocked(
                &resolved.canonical,
                path,
                "edit notebook in",
            ) {
                return Ok(ToolResult {
                    content: msg,
                    is_error: true,
                });
            }
        }

        if let Some(err) = super::path_safety::check_file_editable(path, "edit notebook").await {
            return Ok(err);
        }

        let cell_index = match args["cell_index"].as_u64() {
            Some(i) => i as usize,
            None => {
                return Ok(ToolResult {
                    content: "Error: missing required parameter 'cell_index'".to_string(),
                    is_error: true,
                })
            }
        };

        const MAX_NOTEBOOK_SOURCE: usize = 1_000_000;
        let new_source = match args["new_source"].as_str() {
            Some(s) => {
                if s.len() > MAX_NOTEBOOK_SOURCE {
                    return Ok(ToolResult {
                        content: format!(
                            "Error: new_source too large ({} bytes, limit {})",
                            s.len(),
                            MAX_NOTEBOOK_SOURCE
                        ),
                        is_error: true,
                    });
                }
                s
            }
            None => {
                return Ok(ToolResult {
                    content: "Error: missing required parameter 'new_source'".to_string(),
                    is_error: true,
                })
            }
        };

        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult {
                    content: super::fs_error::format_read_error(
                        &e,
                        path,
                        &format!("Error reading '{}'", path),
                    ),
                    is_error: true,
                })
            }
        };

        let mut notebook: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Error parsing notebook: {}", e),
                    is_error: true,
                })
            }
        };

        let Some(cells) = notebook["cells"].as_array_mut() else {
            return Ok(ToolResult {
                content: "Invalid notebook: no 'cells' array".to_string(),
                is_error: true,
            });
        };

        if cell_index >= cells.len() {
            return Ok(ToolResult {
                content: format!(
                    "Cell index {} out of range (notebook has {} cells, indices 0-{})",
                    cell_index,
                    cells.len(),
                    cells.len().saturating_sub(1)
                ),
                is_error: true,
            });
        }

        // Convert new_source string to notebook's "source" array format.
        // Jupyter stores source as an array of lines with \n terminators (except the last).
        let line_count = new_source.lines().count();
        let source_lines: Vec<Value> = new_source
            .lines()
            .enumerate()
            .map(|(i, line)| {
                if i + 1 < line_count {
                    Value::String(format!("{}\n", line))
                } else {
                    Value::String(line.to_string())
                }
            })
            .collect();

        cells[cell_index]["source"] = Value::Array(source_lines);

        // Clear stale outputs when editing a code cell
        if cells[cell_index]["cell_type"].as_str() == Some("code") {
            cells[cell_index]["outputs"] = Value::Array(vec![]);
            cells[cell_index]["execution_count"] = Value::Null;
        }

        // Optionally change the cell type
        if let Some(ct) = args["cell_type"].as_str() {
            cells[cell_index]["cell_type"] = Value::String(ct.to_string());
        }

        let serialized = serde_json::to_string_pretty(&notebook)?;
        match tokio::fs::write(path, serialized).await {
            Ok(()) => Ok(ToolResult {
                content: format!("Updated cell {} in '{}'.", cell_index, path),
                is_error: false,
            }),
            Err(e) => Ok(ToolResult {
                content: super::fs_error::format_write_error(
                    &e,
                    path,
                    &format!("Error writing '{}'", path),
                ),
                is_error: true,
            }),
        }
    }
}

/// Join a notebook "source" field (either a string or array of strings) into a single string.
pub fn join_source(source: &Value) -> String {
    match source {
        Value::String(s) => s.clone(),
        Value::Array(lines) => lines
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn sample_notebook() -> String {
        json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Hello\n", "World"],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": ["print('hello')"],
                    "outputs": [],
                    "execution_count": null,
                    "metadata": {}
                }
            ]
        })
        .to_string()
    }

    #[tokio::test]
    async fn read_returns_all_cells() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.ipynb");
        tokio::fs::write(&path, sample_notebook()).await.unwrap();

        let tool = NotebookReadTool;
        let result = tool
            .call(json!({"path": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.contains("Cell 0 [markdown]"));
        assert!(result.content.contains("Cell 1 [code]"));
        assert!(result.content.contains("Hello"));
        assert!(result.content.contains("print('hello')"));
    }

    #[tokio::test]
    async fn edit_replaces_cell_source() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.ipynb");
        tokio::fs::write(&path, sample_notebook()).await.unwrap();

        let tool = NotebookEditTool;
        let result = tool
            .call(json!({
                "path": path.to_str().unwrap(),
                "cell_index": 1,
                "new_source": "print('goodbye')"
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "{}", result.content);

        let nb: Value =
            serde_json::from_str(&tokio::fs::read_to_string(&path).await.unwrap()).unwrap();
        let source = join_source(&nb["cells"][1]["source"]);
        assert!(
            source.contains("goodbye"),
            "source should be updated: {:?}",
            nb["cells"][1]["source"]
        );
    }

    #[tokio::test]
    async fn edit_clears_outputs_on_code_cell() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.ipynb");
        let nb = json!({
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [{
                "cell_type": "code",
                "source": ["x = 1"],
                "outputs": [{"output_type": "stream", "text": ["1"]}],
                "execution_count": 1,
                "metadata": {}
            }]
        });
        tokio::fs::write(&path, nb.to_string()).await.unwrap();

        let tool = NotebookEditTool;
        tool.call(json!({
            "path": path.to_str().unwrap(),
            "cell_index": 0,
            "new_source": "x = 2"
        }))
        .await
        .unwrap();

        let saved: Value =
            serde_json::from_str(&tokio::fs::read_to_string(&path).await.unwrap()).unwrap();
        assert_eq!(
            saved["cells"][0]["outputs"].as_array().unwrap().len(),
            0,
            "outputs should be cleared after edit"
        );
    }

    #[tokio::test]
    async fn edit_rejects_out_of_bounds_index() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.ipynb");
        tokio::fs::write(&path, sample_notebook()).await.unwrap();

        let tool = NotebookEditTool;
        let result = tool
            .call(json!({
                "path": path.to_str().unwrap(),
                "cell_index": 99,
                "new_source": "x = 1"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("out of range"));
    }

    // ── join_source pure function tests ────────────────────────────────────────

    #[test]
    fn join_source_from_string_value() {
        let src = json!("print('hello')");
        assert_eq!(join_source(&src), "print('hello')");
    }

    #[test]
    fn join_source_from_array_of_strings() {
        let src = json!(["line 1\n", "line 2\n", "line 3"]);
        assert_eq!(join_source(&src), "line 1\nline 2\nline 3");
    }

    #[test]
    fn join_source_from_empty_array() {
        let src = json!([]);
        assert_eq!(join_source(&src), "");
    }

    #[test]
    fn join_source_from_array_filters_non_strings() {
        // Arrays that contain non-string values (e.g. numbers) — they should be skipped
        let src = json!(["hello", 42, "world"]);
        assert_eq!(join_source(&src), "helloworld");
    }

    #[test]
    fn join_source_from_null_returns_empty() {
        let src = json!(null);
        assert_eq!(join_source(&src), "");
    }

    #[test]
    fn join_source_from_number_returns_empty() {
        let src = json!(123);
        assert_eq!(join_source(&src), "");
    }

    #[test]
    fn join_source_from_bool_returns_empty() {
        let src = json!(true);
        assert_eq!(join_source(&src), "");
    }

    #[test]
    fn join_source_from_object_returns_empty() {
        let src = json!({"key": "value"});
        assert_eq!(join_source(&src), "");
    }

    #[tokio::test]
    async fn read_missing_path_param_errors() {
        let tool = NotebookReadTool;
        let result = tool.call(json!({})).await.unwrap();
        assert!(result.is_error, "missing path should be an error");
    }

    #[tokio::test]
    async fn read_nonexistent_file_is_error() {
        let tool = NotebookReadTool;
        let result = tool
            .call(json!({"path": "/tmp/dm_no_such_notebook_xyz.ipynb"}))
            .await
            .unwrap();
        assert!(result.is_error, "nonexistent file should be an error");
    }

    #[tokio::test]
    async fn notebook_read_warns_on_symlink() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.ipynb");
        tokio::fs::write(&real, sample_notebook()).await.unwrap();
        let link = dir.path().join("link.ipynb");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = NotebookReadTool
            .call(json!({"path": link.to_str().unwrap()}))
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
    async fn notebook_edit_blocks_symlink_write() {
        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real.ipynb");
        tokio::fs::write(&real, sample_notebook()).await.unwrap();
        let link = dir.path().join("link.ipynb");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let result = NotebookEditTool
            .call(json!({
                "path": link.to_str().unwrap(),
                "cell_index": 0,
                "new_source": "# Modified"
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
        let on_disk = std::fs::read_to_string(&real).unwrap();
        assert!(
            on_disk.contains("Hello"),
            "original content should be unchanged"
        );
    }

    #[tokio::test]
    async fn edit_missing_path_is_error() {
        let tool = NotebookEditTool;
        let result = tool
            .call(json!({"cell_index": 0, "new_source": "x"}))
            .await
            .unwrap();
        assert!(result.is_error, "missing path should be an error");
    }

    #[tokio::test]
    async fn notebook_edit_rejects_oversized_source() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("big.ipynb");
        tokio::fs::write(&path, sample_notebook()).await.unwrap();
        let big_source = "x".repeat(1_100_000);
        let result = NotebookEditTool
            .call(json!({
                "path": path.to_str().unwrap(),
                "cell_index": 0,
                "new_source": big_source
            }))
            .await
            .unwrap();
        assert!(result.is_error, "oversized source should be rejected");
        assert!(
            result.content.contains("too large"),
            "error: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn notebook_edit_accepts_reasonable_source() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ok.ipynb");
        tokio::fs::write(&path, sample_notebook()).await.unwrap();
        let result = NotebookEditTool
            .call(json!({
                "path": path.to_str().unwrap(),
                "cell_index": 0,
                "new_source": "x = 42\nprint(x)"
            }))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "reasonable source should succeed: {}",
            result.content
        );
    }

    // --- error-hint retrofits (Cycle 70) ---------------------------------

    #[tokio::test]
    async fn notebook_read_missing_file_routes_through_fs_error() {
        // NotebookReadTool has NO `check_file_editable` gate — the
        // `tokio::fs::read_to_string` map_err at the top of `call` is
        // the direct error surface. This test fires the real retrofit
        // at `:70` and verifies the NotFound hint rides along.
        let tool = NotebookReadTool;
        let res = tool
            .call(json!({"path": "/tmp/dm_no_such_notebook_c70.ipynb"}))
            .await
            .expect("tool returns Ok(ToolResult { is_error: true })");
        assert!(
            res.is_error,
            "missing notebook must flag is_error: {:?}",
            res
        );
        assert!(
            res.content.contains("Error reading"),
            "preserves 'Error reading' prefix: {}",
            res.content
        );
        assert!(
            res.content.contains("Check:"),
            "NotFound hint must appear: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn notebook_edit_missing_file_routes_through_fs_error() {
        // NotebookEditTool has `check_file_editable` which routes
        // through `format_read_error` (C69). The `:259` retrofit is
        // defense-in-depth. Pin the contract that the ToolResult
        // includes the hint for a missing file.
        let tool = NotebookEditTool;
        let res = tool
            .call(json!({
                "path": "/tmp/dm_no_such_notebook_edit_c70.ipynb",
                "cell_index": 0,
                "new_source": "x = 1",
            }))
            .await
            .expect("tool returns Ok(ToolResult { is_error: true })");
        assert!(
            res.is_error,
            "missing notebook must flag is_error: {:?}",
            res
        );
        assert!(
            res.content.contains("Check:"),
            "fs_error NotFound hint appended: {}",
            res.content
        );
    }

    #[test]
    fn notebook_source_wires_format_read_error_twice_and_format_write_error_once() {
        // Source-scan canary: `notebook.rs` has TWO distinct
        // cell-read sites (`NotebookReadTool::call` and
        // `NotebookEditTool::call`) and ONE cell-write site. Pin the
        // exact counts so a future edit can't silently drop a
        // retrofit from one of the three sites.
        // Count only in the production source, not inside the #[cfg(test)]
        // module — otherwise this test's own literals inflate the count.
        let full = include_str!("notebook.rs");
        let prod = full
            .split_once("#[cfg(test)]")
            .map_or(full, |(before, _)| before);
        let read_count = prod.matches("super::fs_error::format_read_error(").count();
        let write_count = prod.matches("super::fs_error::format_write_error(").count();
        assert_eq!(
            read_count, 2,
            "notebook.rs must wire format_read_error at both read sites (saw {})",
            read_count
        );
        assert_eq!(
            write_count, 1,
            "notebook.rs must wire format_write_error at the write site (saw {})",
            write_count
        );
        // And the wording must survive — read sites use 'Error reading',
        // the write site uses 'Error writing'.
        assert!(prod.contains("Error reading"), "read prefix preserved");
        assert!(prod.contains("Error writing"), "write prefix preserved");
    }
}
