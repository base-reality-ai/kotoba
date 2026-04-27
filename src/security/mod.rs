//! Security-audit command support.
//!
//! Collects best-effort dependency audit output, samples files from an
//! operator-supplied glob, streams an Ollama review, and writes the resulting
//! markdown report under the configured dm security directory.

use crate::ollama::client::OllamaClient;
use anyhow::Result;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Returns the path for a security report file in `config_dir/security`/\<timestamp\>.md.
pub fn security_report_path(config_dir: &Path) -> PathBuf {
    let now = chrono::Local::now();
    let timestamp = now.format("%Y-%m-%d_%H%M%S").to_string();
    config_dir
        .join("security")
        .join(format!("{}.md", timestamp))
}

/// Run a security audit against files matching `glob_pattern`.
pub async fn run_security(
    glob_pattern: &str,
    client: &OllamaClient,
    config_dir: &Path,
) -> Result<()> {
    // Step 1: Try running audit tools (best-effort)
    let audit_findings = collect_audit_findings().await;

    // Step 2: Expand glob_pattern, read up to 10 files, cap total at 50,000 chars
    let mut file_contents = String::new();
    let mut file_count = 0usize;
    const MAX_FILES: usize = 10;
    const MAX_CHARS: usize = 50_000;

    match glob::glob(glob_pattern) {
        Ok(paths) => {
            for entry in paths {
                if file_count >= MAX_FILES {
                    break;
                }
                if file_contents.len() >= MAX_CHARS {
                    break;
                }
                let Ok(path) = entry else {
                    continue;
                };
                if !path.is_file() {
                    continue;
                }
                let Ok(content) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let remaining = MAX_CHARS.saturating_sub(file_contents.len());
                let snippet = if content.len() > remaining {
                    crate::util::safe_truncate(&content, remaining)
                } else {
                    &content
                };
                write!(
                    file_contents,
                    "\n\n// --- {} ---\n{}",
                    path.display(),
                    snippet
                )
                .expect("write to String never fails");
                file_count += 1;
            }
        }
        Err(e) => {
            crate::warnings::push_warning(format!(
                "security: invalid glob pattern '{}': {}",
                glob_pattern, e
            ));
        }
    }

    // Step 3: Build prompt and call client
    let audit_section = if audit_findings.is_empty() {
        String::new()
    } else {
        format!("\n\nAudit tool findings:\n{}", audit_findings)
    };

    let prompt = format!(
        "Perform a security audit of the following source code. \
         Look for: SQL injection, XSS, buffer overflows, insecure deserialization, \
         hardcoded secrets, path traversal, unsafe dependencies, and other vulnerabilities. \
         For each issue found, describe: severity (critical/high/medium/low), location, \
         description, and recommended fix. Format as markdown.\
         {audit_section}\n\nCode to audit:\n{file_contents}"
    );

    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];

    // Stream the response to stdout and collect it
    use crate::ollama::types::StreamEvent;
    use futures_util::StreamExt;

    let mut stream = client.chat_stream_with_tools(&messages, &[]).await?;
    let mut report_content = String::new();

    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Token(tok) => {
                print!("{}", tok);
                report_content.push_str(&tok);
            }
            StreamEvent::Done { .. } => break,
            StreamEvent::Error(e) => anyhow::bail!("Stream error: {}", e),
            StreamEvent::Thinking(_) | StreamEvent::ToolCalls(_) => {}
        }
    }
    println!();

    // Step 4: Save report
    let report_path = security_report_path(config_dir);
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("Cannot create security dir: {}", e))?;
    }
    std::fs::write(&report_path, &report_content)
        .map_err(|e| anyhow::anyhow!("Cannot write report '{}': {}", report_path.display(), e))?;

    println!("Report saved to {}", report_path.display());
    Ok(())
}

/// Parse cargo audit JSON output into human-readable findings.
fn parse_cargo_audit_json(json: &serde_json::Value) -> String {
    let mut findings = String::new();
    if let Some(vulns) = json["vulnerabilities"]["list"].as_array() {
        if !vulns.is_empty() {
            writeln!(
                findings,
                "cargo audit: {} vulnerabilities found",
                vulns.len()
            )
            .expect("write to String never fails");
            for v in vulns.iter().take(10) {
                let id = v["advisory"]["id"].as_str().unwrap_or("?");
                let title = v["advisory"]["title"].as_str().unwrap_or("?");
                writeln!(findings, "  - {} {}", id, title).expect("write to String never fails");
            }
        }
    }
    findings
}

/// Parse npm audit JSON output into human-readable findings.
fn parse_npm_audit_json(json: &serde_json::Value) -> String {
    if let Some(total) = json["metadata"]["vulnerabilities"]["total"].as_u64() {
        if total > 0 {
            return format!("npm audit: {} vulnerabilities found\n", total);
        }
    }
    String::new()
}

/// Run audit tools silently and return any findings as text.
async fn collect_audit_findings() -> String {
    let mut findings = String::new();

    // Try cargo audit if Cargo.toml exists
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("Cargo.toml").exists() {
        let result = tokio::process::Command::new("cargo")
            .args(["audit", "--json"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .await;

        if let Ok(output) = result {
            if let Ok(text) = std::str::from_utf8(&output.stdout) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                    findings.push_str(&parse_cargo_audit_json(&json));
                }
            }
        }
    }

    // Try npm audit if package.json exists
    if cwd.join("package.json").exists() {
        let result = tokio::process::Command::new("npm")
            .args(["audit", "--json"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .await;

        if let Ok(output) = result {
            if let Ok(text) = std::str::from_utf8(&output.stdout) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
                    findings.push_str(&parse_npm_audit_json(&json));
                }
            }
        }
    }

    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn security_report_path_has_timestamp() {
        let dir = std::path::Path::new("/tmp/dm_test");
        let path = security_report_path(dir);
        assert!(path.starts_with(dir.join("security")));
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.ends_with(".md"));
        assert!(name.len() > 5); // has some timestamp content
    }

    #[test]
    fn security_report_path_is_under_security_subdir() {
        let dir = std::path::Path::new("/home/user/.dm");
        let path = security_report_path(dir);
        let parent = path.parent().unwrap();
        assert_eq!(parent, dir.join("security"));
    }

    #[test]
    fn security_report_path_filename_is_md() {
        let dir = std::path::Path::new("/tmp/dm_test");
        let path = security_report_path(dir);
        assert_eq!(path.extension().unwrap(), "md");
    }

    #[test]
    fn security_report_path_different_dirs_stay_separate() {
        let dir1 = std::path::Path::new("/tmp/dm_a");
        let dir2 = std::path::Path::new("/tmp/dm_b");
        let p1 = security_report_path(dir1);
        let p2 = security_report_path(dir2);
        // Both should be under their respective config dirs
        assert!(p1.starts_with(dir1));
        assert!(p2.starts_with(dir2));
        assert!(!p1.starts_with(dir2));
    }

    #[test]
    fn security_report_path_filename_contains_date() {
        let dir = std::path::Path::new("/tmp/dm_sec_test");
        let path = security_report_path(dir);
        let name = path.file_name().unwrap().to_str().unwrap();
        // Format: YYYY-MM-DD_HHMMSS.md — at least 18 chars before extension
        let stem = path.file_stem().unwrap().to_str().unwrap();
        assert!(stem.len() >= 17, "timestamp stem too short: {stem}");
        // Should contain a dash (date separator)
        assert!(stem.contains('-'), "stem should have date dashes: {stem}");
        // Should contain an underscore separating date from time
        assert!(
            stem.contains('_'),
            "stem should have date/time separator: {name}"
        );
    }

    #[test]
    fn security_report_path_is_under_correct_subdir_name() {
        let dir = std::path::Path::new("/tmp/dm_sec2");
        let path = security_report_path(dir);
        let components: Vec<_> = path.components().collect();
        // The second-to-last component is "security"
        let parent_name = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(parent_name, "security");
        let _ = components; // silence unused warning
    }

    // ── parse_cargo_audit_json ────────────────────────────────────────────────

    #[test]
    fn security_truncation_utf8_safe() {
        let content = "🦀".repeat(200);
        let remaining = 50;
        let snippet = crate::util::safe_truncate(&content, remaining);
        assert!(snippet.len() <= remaining);
        assert!(std::str::from_utf8(snippet.as_bytes()).is_ok());
    }

    #[test]
    fn parse_cargo_audit_json_with_vulnerabilities() {
        let json = serde_json::json!({
            "vulnerabilities": {
                "list": [
                    {"advisory": {"id": "RUSTSEC-2023-0001", "title": "Buffer overflow in foo"}},
                    {"advisory": {"id": "RUSTSEC-2023-0002", "title": "Use after free in bar"}},
                ]
            }
        });
        let result = parse_cargo_audit_json(&json);
        assert!(result.contains("2 vulnerabilities found"));
        assert!(result.contains("RUSTSEC-2023-0001"));
        assert!(result.contains("Buffer overflow in foo"));
        assert!(result.contains("RUSTSEC-2023-0002"));
    }

    #[test]
    fn parse_cargo_audit_json_empty_list() {
        let json = serde_json::json!({
            "vulnerabilities": {"list": []}
        });
        let result = parse_cargo_audit_json(&json);
        assert!(result.is_empty(), "no vulns should produce empty string");
    }

    #[test]
    fn parse_cargo_audit_json_missing_fields() {
        // Completely wrong structure should not panic
        let json = serde_json::json!({"unrelated": "data"});
        let result = parse_cargo_audit_json(&json);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_cargo_audit_json_caps_at_ten_entries() {
        let vulns: Vec<serde_json::Value> = (0..15)
            .map(|i| serde_json::json!({"advisory": {"id": format!("RUSTSEC-{}", i), "title": format!("vuln {}", i)}}))
            .collect();
        let json = serde_json::json!({"vulnerabilities": {"list": vulns}});
        let result = parse_cargo_audit_json(&json);
        assert!(
            result.contains("15 vulnerabilities found"),
            "header should show total: {result}"
        );
        // Only first 10 entries listed
        assert!(result.contains("RUSTSEC-9"));
        assert!(
            !result.contains("RUSTSEC-10"),
            "should cap at 10 entries: {result}"
        );
    }

    #[test]
    fn parse_cargo_audit_json_missing_advisory_fields() {
        let json = serde_json::json!({
            "vulnerabilities": {
                "list": [{"advisory": {}}]
            }
        });
        let result = parse_cargo_audit_json(&json);
        assert!(result.contains("1 vulnerabilities"));
        assert!(
            result.contains('?'),
            "missing fields should show ? placeholder"
        );
    }

    // ── parse_npm_audit_json ──────────────────────────────────────────────────

    #[test]
    fn parse_npm_audit_json_with_vulnerabilities() {
        let json = serde_json::json!({
            "metadata": {"vulnerabilities": {"total": 5}}
        });
        let result = parse_npm_audit_json(&json);
        assert!(result.contains("5 vulnerabilities found"));
    }

    #[test]
    fn parse_npm_audit_json_zero_vulnerabilities() {
        let json = serde_json::json!({
            "metadata": {"vulnerabilities": {"total": 0}}
        });
        let result = parse_npm_audit_json(&json);
        assert!(result.is_empty(), "zero vulns should produce empty string");
    }

    #[test]
    fn parse_npm_audit_json_missing_fields() {
        let json = serde_json::json!({"audit": {}});
        let result = parse_npm_audit_json(&json);
        assert!(result.is_empty());
    }
}
