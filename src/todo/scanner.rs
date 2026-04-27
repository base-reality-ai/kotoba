/// A single TODO-style comment found in a source file.
#[derive(Debug)]
pub struct TodoItem {
    pub file: String,
    pub line: usize,
    pub tag: String,  // TODO, FIXME, HACK, OPTIMIZE, XXX
    pub text: String, // the rest of the line after the tag
}

const TAGS: &[&str] = &["TODO", "FIXME", "HACK", "OPTIMIZE", "XXX"];

/// Scan a set of paths for TODO-style comments.
/// Binary files (detected by a null byte in the first 512 bytes) are skipped.
pub fn scan_todos(paths: &[std::path::PathBuf]) -> Vec<TodoItem> {
    let mut items = Vec::new();
    for path in paths {
        if let Ok(bytes) = std::fs::read(path) {
            // Skip binary files: check for null byte in first 512 bytes
            let probe = &bytes[..bytes.len().min(512)];
            if probe.contains(&0u8) {
                continue;
            }
            let Ok(content) = String::from_utf8(bytes) else {
                continue;
            };
            let file = path.to_string_lossy().to_string();
            for (idx, line) in content.lines().enumerate() {
                let upper = line.to_uppercase();
                // Find the first matching tag
                let found = TAGS.iter().find_map(|&tag| {
                    upper.find(tag).map(|pos| {
                        // Extract the text after the tag in the original line
                        let after = &line[pos + tag.len()..];
                        // Strip leading colon/space
                        let text = after.trim_start_matches(':').trim_start().to_string();
                        (tag.to_string(), text)
                    })
                });
                if let Some((tag, text)) = found {
                    items.push(TodoItem {
                        file: file.clone(),
                        line: idx + 1, // 1-indexed
                        tag,
                        text,
                    });
                }
            }
        }
    }
    items
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    #[test]
    fn scan_finds_todo_comments() {
        let f = write_temp_file("// TODO: foo\n// FIXME: bar\nclean line\n");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert_eq!(
            items.len(),
            2,
            "expected 2 items, got: {:?}",
            items.iter().map(|i| &i.tag).collect::<Vec<_>>()
        );
        assert_eq!(items[0].tag, "TODO");
        assert_eq!(items[0].line, 1);
        assert!(items[0].text.contains("foo"), "text: {}", items[0].text);
        assert_eq!(items[1].tag, "FIXME");
        assert_eq!(items[1].line, 2);
        assert!(items[1].text.contains("bar"), "text: {}", items[1].text);
    }

    #[test]
    fn scan_case_insensitive() {
        let f = write_temp_file("// todo: lower\n");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tag, "TODO");
        assert!(items[0].text.contains("lower"), "text: {}", items[0].text);
    }

    #[test]
    fn scan_ignores_lines_without_tags() {
        let f = write_temp_file("fn main() {}\nlet x = 1;\n// regular comment\n");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert!(items.is_empty(), "expected empty, got: {:?}", items.len());
    }

    #[test]
    fn scan_skips_binary_file() {
        // Write a file with a null byte in the first 512 bytes
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        let mut content = b"// TODO: this looks like code\n".to_vec();
        content.push(0u8); // null byte → binary
        f.write_all(&content).expect("write");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert!(items.is_empty(), "binary file should be skipped");
    }

    #[test]
    fn scan_finds_all_tags() {
        let src = "// TODO: a\n// FIXME: b\n// HACK: c\n// OPTIMIZE: d\n// XXX: e\n";
        let f = write_temp_file(src);
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert_eq!(items.len(), 5, "should find all 5 tag types");
        let tags: Vec<&str> = items.iter().map(|i| i.tag.as_str()).collect();
        assert!(tags.contains(&"TODO"));
        assert!(tags.contains(&"FIXME"));
        assert!(tags.contains(&"HACK"));
        assert!(tags.contains(&"OPTIMIZE"));
        assert!(tags.contains(&"XXX"));
    }

    #[test]
    fn scan_strips_colon_and_whitespace_from_text() {
        let f = write_temp_file("// TODO:   text after spaces\n");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "text after spaces");
    }

    #[test]
    fn scan_line_numbers_are_one_indexed() {
        let f = write_temp_file("line one\nline two\n// TODO: on line 3\n");
        let paths = vec![f.path().to_path_buf()];
        let items = scan_todos(&paths);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].line, 3);
    }

    #[test]
    fn scan_empty_paths_returns_empty() {
        let items = scan_todos(&[]);
        assert!(items.is_empty());
    }

    #[test]
    fn scan_nonexistent_file_is_ignored() {
        let paths = vec![std::path::PathBuf::from("/nonexistent/path/file.rs")];
        let items = scan_todos(&paths);
        assert!(items.is_empty(), "missing file should be silently skipped");
    }

    /// Discipline guard: `src/` must remain free of real `// TODO:` /
    /// `// FIXME:` / `// HACK:` / `// XXX:` line comments. The
    /// `src/todo/` module is excluded — it legitimately mentions these
    /// tags as keyword strings in `TAGS` and as test fixtures. Verified
    /// empty across cycles 1, 5, 9, 12, 29 by manual grep; this
    /// structural test pins the discipline so future drift fails at
    /// test time. See memory `project_no_real_todos_in_src` for the
    /// reasoning ("no stubs" rule from the directive).
    #[test]
    fn src_has_no_real_todo_line_comments() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let project_root = match std::env::current_dir() {
            Ok(p) => p,
            Err(_) => return,
        };
        let src_dir = project_root.join("src");
        if !src_dir.is_dir() {
            return;
        }

        let mut rs_paths: Vec<std::path::PathBuf> = Vec::new();
        collect_rs_files(&src_dir, &mut rs_paths);

        let pattern = regex::Regex::new(r"^\s*//\s*(TODO|FIXME|HACK|XXX)\b").expect("regex");

        let mut violations: Vec<String> = Vec::new();
        for path in &rs_paths {
            let rel = path
                .strip_prefix(&project_root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            if rel.starts_with("src/todo/") {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(path) else {
                continue;
            };
            for (idx, line) in content.lines().enumerate() {
                if pattern.is_match(line) {
                    violations.push(format!("{}:{}: {}", rel, idx + 1, line.trim()));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "src/ must stay TODO-free per the directive's no-stubs rule \
             (see memory `project_no_real_todos_in_src`). Found {} \
             violation(s):\n  {}",
            violations.len(),
            violations.join("\n  ")
        );
    }

    fn collect_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for ent in rd.flatten() {
            let Ok(ft) = ent.file_type() else {
                continue;
            };
            let path = ent.path();
            if ft.is_dir() {
                collect_rs_files(&path, out);
            } else if ft.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}
