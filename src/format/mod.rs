//! Ecosystem code formatter integration.
//!
//! Provides auto-formatting support (rustfmt, black, prettier, etc.)
//! applied after automated code modifications.

use std::path::Path;

/// Returns the formatter command and arguments for a given file path, or None if not applicable.
/// Returns `(program, args_before_path)`.
pub fn formatter_for_path(path: &Path) -> Option<(&'static str, Vec<&'static str>)> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => Some(("rustfmt", vec![])),
        Some("py") => Some(("black", vec![])),
        Some("js") | Some("ts") | Some("jsx") | Some("tsx") | Some("json") | Some("css")
        | Some("scss") | Some("less") | Some("html") | Some("htm") | Some("md") | Some("mdx")
        | Some("yaml") | Some("yml") | Some("graphql") | Some("gql") => {
            Some(("prettier", vec!["--write"]))
        }
        Some("go") => Some(("gofmt", vec!["-w"])),
        Some("sh") | Some("bash") => Some(("shfmt", vec!["-w"])),
        Some("c") | Some("h") | Some("cpp") | Some("hpp") | Some("cc") | Some("cxx") => {
            Some(("clang-format", vec!["-i"]))
        }
        Some("lua") => Some(("stylua", vec![])),
        Some("zig") => Some(("zig", vec!["fmt"])),
        _ => None,
    }
}

/// Run the appropriate formatter for path.
/// Returns Ok(true) if ran successfully, Ok(false) if no formatter or formatter not found.
/// Returns Err if the formatter ran but exited nonzero.
pub async fn format_file(path: &Path) -> anyhow::Result<bool> {
    let Some((prog, args)) = formatter_for_path(path) else {
        return Ok(false);
    };

    let result = tokio::process::Command::new(prog)
        .args(&args)
        .arg(path)
        .stdout(std::process::Stdio::null())
        .output()
        .await;

    match result {
        Ok(output) => {
            if output.status.success() {
                Ok(true)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = if stderr.trim().is_empty() {
                    String::new()
                } else {
                    format!(": {}", crate::util::safe_truncate(stderr.trim(), 500))
                };
                anyhow::bail!(
                    "Formatter '{}' failed for '{}'{}",
                    prog,
                    path.display(),
                    detail
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => {
            anyhow::bail!(
                "Failed to run formatter '{}' for '{}': {}",
                prog,
                path.display(),
                e
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_skips_unknown_extension() {
        let result = formatter_for_path(Path::new("foo.xyz"));
        assert!(result.is_none());
    }

    #[test]
    fn format_detects_rust_formatter() {
        let result = formatter_for_path(Path::new("foo.rs"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "rustfmt");
    }

    #[test]
    fn format_detects_python_formatter() {
        let result = formatter_for_path(Path::new("foo.py"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "black");
    }

    #[test]
    fn format_detects_js_formatter() {
        let result = formatter_for_path(Path::new("foo.js"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "prettier");
    }

    #[test]
    fn format_detects_go_formatter() {
        let result = formatter_for_path(Path::new("foo.go"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "gofmt");
    }

    #[test]
    fn format_detects_json_formatter() {
        let result = formatter_for_path(Path::new("foo.json"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "prettier");
    }

    #[test]
    fn format_detects_ts_formatter() {
        let result = formatter_for_path(Path::new("app.ts"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "prettier");
    }

    #[test]
    fn format_detects_tsx_formatter() {
        let result = formatter_for_path(Path::new("app.tsx"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "prettier");
    }

    #[test]
    fn format_no_extension_returns_none() {
        let result = formatter_for_path(Path::new("Makefile"));
        assert!(result.is_none());
    }

    #[test]
    fn format_detects_jsx_formatter() {
        let result = formatter_for_path(Path::new("component.jsx"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "prettier");
    }

    #[test]
    fn format_rust_formatter_has_no_extra_args() {
        let (_, args) = formatter_for_path(Path::new("lib.rs")).unwrap();
        assert!(
            args.is_empty(),
            "rustfmt should have no extra args before the file path"
        );
    }

    #[test]
    fn format_go_formatter_has_write_flag() {
        let (_, args) = formatter_for_path(Path::new("main.go")).unwrap();
        assert!(
            args.contains(&"-w"),
            "gofmt should use -w to write in place"
        );
    }

    #[test]
    fn format_detects_css_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("style.css")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_scss_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("style.scss")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_html_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("index.html")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_htm_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("page.htm")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_markdown_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("README.md")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_yaml_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("config.yaml")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_yml_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("ci.yml")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_graphql_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("schema.graphql")).unwrap().0,
            "prettier"
        );
    }

    #[test]
    fn format_detects_shell_formatter() {
        assert_eq!(formatter_for_path(Path::new("run.sh")).unwrap().0, "shfmt");
    }

    #[test]
    fn format_detects_bash_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("build.bash")).unwrap().0,
            "shfmt"
        );
    }

    #[test]
    fn format_detects_c_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("main.c")).unwrap().0,
            "clang-format"
        );
    }

    #[test]
    fn format_detects_cpp_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("app.cpp")).unwrap().0,
            "clang-format"
        );
    }

    #[test]
    fn format_detects_hpp_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("lib.hpp")).unwrap().0,
            "clang-format"
        );
    }

    #[test]
    fn format_detects_lua_formatter() {
        assert_eq!(
            formatter_for_path(Path::new("init.lua")).unwrap().0,
            "stylua"
        );
    }

    #[test]
    fn format_detects_zig_formatter() {
        let (prog, args) = formatter_for_path(Path::new("main.zig")).unwrap();
        assert_eq!(prog, "zig");
        assert!(args.contains(&"fmt"));
    }

    #[test]
    fn format_shell_formatter_has_write_flag() {
        let (_, args) = formatter_for_path(Path::new("run.sh")).unwrap();
        assert!(args.contains(&"-w"), "shfmt should use -w");
    }

    #[test]
    fn format_c_formatter_has_inplace_flag() {
        let (_, args) = formatter_for_path(Path::new("main.c")).unwrap();
        assert!(args.contains(&"-i"), "clang-format should use -i");
    }

    #[tokio::test]
    async fn format_file_unknown_extension_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.xyz");
        std::fs::write(&path, "some content").unwrap();
        let result = format_file(&path).await.unwrap();
        assert!(!result, "unknown extension should return Ok(false)");
    }

    #[tokio::test]
    async fn format_file_missing_formatter_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.lua");
        std::fs::write(&path, "print('hello')").unwrap();
        // stylua is unlikely to be installed in CI — should return Ok(false)
        if which::which("stylua").is_err() {
            let result = format_file(&path).await.unwrap();
            assert!(!result, "missing formatter binary should return Ok(false)");
        }
    }
}
