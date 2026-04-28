//! Automated test failure detection and remediation.
//!
//! Implements a continuous watcher and AI-driven repair loop that triggers
//! on test failures to propose and apply code fixes.

pub mod detect;
pub mod watcher;

use crate::logging;
use crate::ollama::client::OllamaClient;
use crate::permissions;
use crate::session::Session;
use crate::tools;
use anyhow::anyhow;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

type McpMap = HashMap<String, std::sync::Arc<tokio::sync::Mutex<crate::mcp::client::McpClient>>>;

// Typed diagnostic-level vocabulary. Current parsers (rustc, tsc, python
// traceback) only construct Error and Warning; Note is matched in sort_key
// and Display but has no constructor today. Retained for API completeness.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum DiagLevel {
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagLevel,
    pub code: Option<String>,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub context_lines: Vec<String>,
}

impl Diagnostic {
    pub fn summary_line(&self) -> String {
        let code_part = self
            .code
            .as_ref()
            .map(|c| format!("[{}] ", c))
            .unwrap_or_default();
        let location = match (&self.file, self.line) {
            (Some(f), Some(l)) => format!(" → {}:{}", f, l),
            (Some(f), None) => format!(" → {}", f),
            _ => String::new(),
        };
        format!("{}{}{}", code_part, self.message, location)
    }
}

pub fn parse_diagnostics(output: &str) -> Vec<Diagnostic> {
    let lines: Vec<&str> = output.lines().collect();
    let mut diags: Vec<Diagnostic> = Vec::new();
    let mut i = 0;

    while i < lines.len() && diags.len() < 20 {
        let line = lines[i];

        // Detect test failure blocks: ---- test_name stdout ----
        if line.starts_with("---- ") && line.ends_with(" ----") {
            let test_name = line.trim_start_matches("---- ").trim_end_matches(" ----");
            let mut ctx = vec![line.to_string()];
            i += 1;
            while i < lines.len() {
                if lines[i].starts_with("---- ") && lines[i].ends_with(" ----") {
                    break;
                }
                ctx.push(lines[i].to_string());
                i += 1;
            }
            diags.push(Diagnostic {
                level: DiagLevel::Error,
                code: None,
                message: format!("test failure: {}", test_name),
                file: None,
                line: None,
                context_lines: ctx,
            });
            continue;
        }

        // Detect Rust compiler diagnostics
        if let Some(diag) = try_parse_rust_diagnostic(line) {
            let mut ctx = vec![line.to_string()];
            let mut file = None;
            let mut diag_line = None;
            i += 1;

            // Capture location line: "  --> file:line:col"
            if i < lines.len() {
                let next = lines[i].trim();
                if next.starts_with("-->") {
                    ctx.push(lines[i].to_string());
                    let loc = next.trim_start_matches("-->").trim();
                    let parts: Vec<&str> = loc.splitn(3, ':').collect();
                    if !parts.is_empty() {
                        file = Some(parts[0].to_string());
                    }
                    if parts.len() >= 2 {
                        diag_line = parts[1].parse().ok();
                    }
                    i += 1;
                }
            }

            // Capture indented context lines (|, spaces, blank within block)
            while i < lines.len() {
                let next = lines[i];
                if next.is_empty()
                    || next.starts_with(' ')
                    || next.starts_with('\t')
                    || next.contains('|')
                {
                    // Check if this is actually a new diagnostic
                    if try_parse_rust_diagnostic(next.trim_start()).is_some() {
                        break;
                    }
                    ctx.push(next.to_string());
                    i += 1;
                } else {
                    break;
                }
            }

            diags.push(Diagnostic {
                level: diag.0,
                code: diag.1,
                message: diag.2,
                file,
                line: diag_line,
                context_lines: ctx,
            });
            continue;
        }

        // Generic file:line:col: error/warning: message (Go, GCC, ESLint, etc.)
        if let Some(diag) = try_parse_generic_diagnostic(line) {
            diags.push(diag);
            i += 1;
            continue;
        }

        // TypeScript: file.ts(line,col): error TS1234: message
        if let Some(diag) = try_parse_tsc_diagnostic(line) {
            diags.push(diag);
            i += 1;
            continue;
        }

        // Python traceback: File "path", line N
        if let Some(diag) = try_parse_python_traceback(line, &lines, &mut i) {
            diags.push(diag);
            continue;
        }

        i += 1;
    }

    // Prioritize errors over warnings
    diags.sort_by_key(|d| match d.level {
        DiagLevel::Error => 0,
        DiagLevel::Warning => 1,
        DiagLevel::Note => 2,
    });
    diags.truncate(20);
    diags
}

fn try_parse_rust_diagnostic(line: &str) -> Option<(DiagLevel, Option<String>, String)> {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("error[") {
        let (code, msg) = parse_bracketed_diagnostic(rest);
        Some((DiagLevel::Error, Some(code), msg))
    } else if let Some(rest) = trimmed.strip_prefix("warning[") {
        let (code, msg) = parse_bracketed_diagnostic(rest);
        Some((DiagLevel::Warning, Some(code), msg))
    } else if let Some(rest) = trimmed.strip_prefix("error: ") {
        Some((DiagLevel::Error, None, rest.to_string()))
    } else {
        trimmed
            .strip_prefix("warning: ")
            .map(|rest| (DiagLevel::Warning, None, rest.to_string()))
    }
}

fn parse_bracketed_diagnostic(rest: &str) -> (String, String) {
    if let Some(bracket_end) = rest.find(']') {
        let code = rest[..bracket_end].to_string();
        let msg = rest[bracket_end + 1..]
            .trim_start_matches(':')
            .trim()
            .to_string();
        (code, msg)
    } else {
        (String::new(), rest.to_string())
    }
}

/// Generic `file:line:col: error: message` or `file:line: error: message`.
/// Covers Go, GCC, `ESLint` default formatter, and many other tools.
fn try_parse_generic_diagnostic(line: &str) -> Option<Diagnostic> {
    let trimmed = line.trim();
    // Match: path:line:col: level: message  OR  path:line: level: message
    // Require the path to contain a dot (file extension) or slash to avoid false positives
    let colon1 = trimmed.find(':')?;
    let path = &trimmed[..colon1];
    if !path.contains('.') && !path.contains('/') && !path.contains('\\') {
        return None;
    }
    let rest = &trimmed[colon1 + 1..];
    // Parse line number
    let colon2 = rest.find(':')?;
    let line_str = &rest[..colon2];
    let line_num: usize = line_str.trim().parse().ok()?;
    // After line:col or line, look for ": error:" or ": warning:"
    let after_line = &rest[colon2 + 1..];
    // Skip optional col number
    let msg_start = if let Some(c3) = after_line.find(':') {
        let maybe_col = &after_line[..c3];
        if maybe_col.trim().chars().all(|c| c.is_ascii_digit()) {
            &after_line[c3 + 1..]
        } else {
            after_line
        }
    } else {
        return None;
    };
    let msg_trimmed = msg_start.trim();
    let (level, message) = if let Some(m) = msg_trimmed.strip_prefix("error:") {
        (DiagLevel::Error, m.trim().to_string())
    } else if let Some(m) = msg_trimmed.strip_prefix("warning:") {
        (DiagLevel::Warning, m.trim().to_string())
    } else if let Some(m) = msg_trimmed.strip_prefix("fatal error:") {
        (DiagLevel::Error, m.trim().to_string())
    } else {
        return None;
    };
    Some(Diagnostic {
        level,
        code: None,
        message,
        file: Some(path.to_string()),
        line: Some(line_num),
        context_lines: vec![trimmed.to_string()],
    })
}

/// TypeScript: `file.ts(line,col): error TS1234: message`
fn try_parse_tsc_diagnostic(line: &str) -> Option<Diagnostic> {
    let trimmed = line.trim();
    let paren = trimmed.find('(')?;
    let path = &trimmed[..paren];
    if !path.ends_with(".ts") && !path.ends_with(".tsx") {
        return None;
    }
    let close = trimmed[paren..].find(')')?;
    let coords = &trimmed[paren + 1..paren + close];
    let line_num: usize = coords.split(',').next()?.trim().parse().ok()?;
    let after_paren = &trimmed[paren + close + 1..];
    let after_colon = after_paren.strip_prefix(':')?;
    let msg_trimmed = after_colon.trim();
    let (level, rest) = if let Some(r) = msg_trimmed.strip_prefix("error") {
        (DiagLevel::Error, r)
    } else if let Some(r) = msg_trimmed.strip_prefix("warning") {
        (DiagLevel::Warning, r)
    } else {
        return None;
    };
    // Extract TS code: " TS1234: message"
    let rest = rest.trim_start();
    let (code, message) = if let Some(r) = rest.strip_prefix("TS") {
        if let Some(colon) = r.find(':') {
            let code = format!("TS{}", &r[..colon]);
            let msg = r[colon + 1..].trim().to_string();
            (Some(code), msg)
        } else {
            (None, rest.to_string())
        }
    } else {
        let message = rest.strip_prefix(':').unwrap_or(rest).trim().to_string();
        (None, message)
    };
    Some(Diagnostic {
        level,
        code,
        message,
        file: Some(path.to_string()),
        line: Some(line_num),
        context_lines: vec![trimmed.to_string()],
    })
}

/// Python traceback: `File "path", line N` followed by the error line.
fn try_parse_python_traceback(line: &str, lines: &[&str], i: &mut usize) -> Option<Diagnostic> {
    let trimmed = line.trim();
    if !trimmed.starts_with("File \"") {
        return None;
    }
    let quote_end = trimmed[6..].find('"')?;
    let path = trimmed[6..6 + quote_end].to_string();
    let after_path = &trimmed[6 + quote_end + 1..];
    let line_num: usize = after_path.find("line ").and_then(|pos| {
        let num_start = pos + 5;
        let num_end = after_path[num_start..]
            .find(|c: char| !c.is_ascii_digit())
            .map_or(after_path.len(), |e| num_start + e);
        after_path[num_start..num_end].parse().ok()
    })?;
    let mut ctx = vec![line.to_string()];
    *i += 1;
    // Capture the code line (indented)
    if *i < lines.len() && (lines[*i].starts_with(' ') || lines[*i].starts_with('\t')) {
        ctx.push(lines[*i].to_string());
        *i += 1;
    }
    // Capture the error line (e.g., "TypeError: ...")
    let message = if *i < lines.len() {
        let err_line = lines[*i].trim();
        if err_line.contains("Error") || err_line.contains("Exception") {
            ctx.push(lines[*i].to_string());
            *i += 1;
            err_line.to_string()
        } else {
            format!("error at {}:{}", path, line_num)
        }
    } else {
        format!("error at {}:{}", path, line_num)
    };
    Some(Diagnostic {
        level: DiagLevel::Error,
        code: None,
        message,
        file: Some(path),
        line: Some(line_num),
        context_lines: ctx,
    })
}

pub fn format_diagnostics(diags: &[Diagnostic]) -> String {
    let mut out = String::new();
    for (i, d) in diags.iter().enumerate() {
        let level_str = match d.level {
            DiagLevel::Error => "Error",
            DiagLevel::Warning => "Warning",
            DiagLevel::Note => "Note",
        };
        let code_str = d
            .code
            .as_ref()
            .map(|c| format!(" [{}]", c))
            .unwrap_or_default();
        writeln!(out, "## {} {}: {}{}", level_str, i + 1, d.message, code_str)
            .expect("write to String never fails");
        if let Some(ref f) = d.file {
            if let Some(l) = d.line {
                writeln!(out, "File: {}:{}", f, l).expect("write to String never fails");
            } else {
                writeln!(out, "File: {}", f).expect("write to String never fails");
            }
        }
        if !d.context_lines.is_empty() {
            out.push_str("```\n");
            for cl in &d.context_lines {
                out.push_str(cl);
                out.push('\n');
            }
            out.push_str("```\n");
        }
        out.push_str("---\n");

        if out.len() > 8000 {
            crate::util::safe_string_truncate(&mut out, 8000);
            out.push_str("\n[truncated]");
            break;
        }
    }
    out
}

/// Build a fresh session, tool registry, permission engine, empty MCP map, and
/// system prompt — the standard set needed to call `run_conversation`.
async fn fresh_fix_context(
    client: &OllamaClient,
    config_dir: &Path,
) -> (
    Session,
    crate::tools::registry::ToolRegistry,
    permissions::engine::PermissionEngine,
    McpMap,
    String,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let sess = Session::new(cwd, client.model().to_string());
    let session_id = sess.id.clone();
    let registry = tools::registry::default_registry(
        &session_id,
        config_dir,
        client.base_url(),
        client.model(),
        "nomic-embed-text",
    );
    let engine = permissions::engine::PermissionEngine::new(true, vec![]);
    let mcp_clients: McpMap = HashMap::new();
    let system_prompt = crate::system_prompt::build_system_prompt(&[], None).await;
    (sess, registry, engine, mcp_clients, system_prompt)
}

/// Extract lines from test/compiler output that indicate failures.
/// Keeps lines containing: FAILED, error[, thread ', panicked at, assertion
/// Caps result at 8000 chars.
pub fn extract_failures(output: &str) -> String {
    extract_failures_inner(output, false)
}

/// Like `extract_failures`, but also captures `warning[` lines when `lint_mode` is true.
pub fn extract_failures_with_lint(output: &str) -> String {
    extract_failures_inner(output, true)
}

fn extract_failures_inner(output: &str, lint_mode: bool) -> String {
    // Try structured diagnostic parsing first
    let diags = parse_diagnostics(output);
    let filtered: Vec<&Diagnostic> = diags
        .iter()
        .filter(|d| d.level == DiagLevel::Error || (lint_mode && d.level == DiagLevel::Warning))
        .collect();

    if !filtered.is_empty() {
        let owned: Vec<Diagnostic> = filtered.into_iter().cloned().collect();
        return format_diagnostics(&owned);
    }

    // Fall back to keyword-based extraction for non-Rust output
    let keywords = ["FAILED", "error[", "thread '", "panicked at", "assertion"];
    let lines: Vec<&str> = output
        .lines()
        .filter(|line| {
            keywords.iter().any(|kw| line.contains(kw)) || (lint_mode && line.contains("warning["))
        })
        .collect();

    if lines.is_empty() {
        return String::new();
    }

    let joined = lines.join("\n");
    if joined.len() <= 8000 {
        joined
    } else {
        let mut end = 8000usize.min(joined.len());
        while end > 0 && !joined.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = joined[..end].to_string();
        truncated.push_str("\n[truncated]");
        truncated
    }
}

/// Run CMD, fix failures with AI, repeat up to `max_rounds`.
pub async fn run_test_fix(
    cmd: &str,
    max_rounds: usize,
    client: &OllamaClient,
    config_dir: &Path,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        anyhow::bail!("Empty command for --test-fix");
    }
    let (prog, args) = (parts[0], &parts[1..]);

    for round in 1..=max_rounds {
        // Run the command
        let output = tokio::process::Command::new(prog)
            .args(args)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = format!("{}\n{}", stdout, stderr);

        if output.status.success() {
            println!("All tests pass.");
            return Ok(());
        }

        let failures = extract_failures(&combined);
        let context = if failures.is_empty() {
            let raw_len = combined.len();
            let start = raw_len.saturating_sub(500);
            combined[start..].to_string()
        } else {
            failures.clone()
        };

        let failure_count = if failures.is_empty() {
            0
        } else {
            failures.lines().count()
        };
        println!("Round {}: {} failures found", round, failure_count);

        let prompt = format!(
            "The following command failed: `{}`\n\n\
            Here are the structured diagnostics:\n\
            {}\n\n\
            For each error, read the file at the indicated line, understand the issue, \
            and apply a targeted fix. Do not refactor unrelated code.",
            cmd, context
        );

        let (mut sess, registry, mut engine, mcp_clients, system_prompt) =
            fresh_fix_context(client, config_dir).await;

        crate::conversation::run_conversation(
            &prompt,
            "testfix",
            client,
            None,
            &registry,
            &mcp_clients,
            system_prompt,
            &mut engine,
            &mut sess,
            config_dir,
            config_dir,
            false,
            "text",
            10,
            false,
            None,
        )
        .await?;
    }

    // Final check after all rounds
    let output = tokio::process::Command::new(prog)
        .args(args)
        .output()
        .await?;

    if output.status.success() {
        println!("All tests pass.");
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{}\n{}", stdout, stderr);
    let failures = extract_failures(&combined);
    logging::log_err(&format!("Final failures:\n{}", failures));
    Err(anyhow!("Tests still failing after {} rounds", max_rounds))
}

/// Run CMD (linter), fix lint warnings with AI, repeat up to `max_rounds`.
/// Captures both test failures AND `warning[` lines.
pub async fn run_lint_fix(
    cmd: &str,
    max_rounds: usize,
    client: &OllamaClient,
    config_dir: &Path,
) -> anyhow::Result<()> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        anyhow::bail!("Empty command for --lint-fix");
    }
    let (prog, args) = (parts[0], &parts[1..]);

    for round in 1..=max_rounds {
        let output = tokio::process::Command::new(prog)
            .args(args)
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = format!("{}\n{}", stdout, stderr);

        // For linters, success exit code AND no warnings means done
        let failures = extract_failures_with_lint(&combined);
        if output.status.success() && failures.is_empty() {
            println!("No lint issues.");
            return Ok(());
        }

        let context = if failures.is_empty() {
            let raw_len = combined.len();
            let start = raw_len.saturating_sub(500);
            combined[start..].to_string()
        } else {
            failures.clone()
        };

        let issue_count = if failures.is_empty() {
            0
        } else {
            failures.lines().count()
        };
        println!("Round {}: {} lint issue(s) found", round, issue_count);

        let prompt = format!(
            "The following lint command reported issues: `{}`\n\n\
            Here are the relevant lines from the output:\n\
            ```\n{}\n```\n\n\
            Please fix these lint warnings/errors. Use the available file tools \
            to read and edit the necessary source files. Focus on the actual \
            issues shown — do not refactor code unnecessarily.",
            cmd, context
        );

        let (mut sess, registry, mut engine, mcp_clients, system_prompt) =
            fresh_fix_context(client, config_dir).await;

        crate::conversation::run_conversation(
            &prompt,
            "testfix",
            client,
            None,
            &registry,
            &mcp_clients,
            system_prompt,
            &mut engine,
            &mut sess,
            config_dir,
            config_dir,
            false,
            "text",
            10,
            false,
            None,
        )
        .await?;
    }

    // Final check
    let output = tokio::process::Command::new(prog)
        .args(args)
        .output()
        .await?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{}\n{}", stdout, stderr);
    let failures = extract_failures_with_lint(&combined);

    if output.status.success() && failures.is_empty() {
        println!("No lint issues.");
        return Ok(());
    }

    logging::log_err(&format!("Final lint issues:\n{}", failures));
    Err(anyhow!(
        "Lint issues still present after {} rounds",
        max_rounds
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_failures_finds_failed_lines() {
        let input =
            "test foo ... ok\ntest bar ... FAILED\nerror[E0308]: type mismatch\nsome other line";
        let result = extract_failures(input);
        assert!(
            result.contains("E0308"),
            "should contain error code: {}",
            result
        );
        assert!(
            result.contains("type mismatch"),
            "should contain error message: {}",
            result
        );
        assert!(
            !result.contains("some other line"),
            "should not contain irrelevant lines"
        );
    }

    #[test]
    fn extract_failures_caps_at_8000_chars() {
        let long_line = "FAILED ".repeat(2000); // ~14000 chars
        let result = extract_failures(&long_line);
        assert!(result.len() <= 8100, "should be capped near 8000 chars");
    }

    #[test]
    fn extract_failures_empty_on_clean_output() {
        let input = "test foo ... ok\ntest bar ... ok\ntest result: ok. 3 passed";
        let result = extract_failures(input);
        assert!(
            result.is_empty() || result.len() < 50,
            "clean output should yield minimal failures"
        );
    }

    #[test]
    fn extract_failures_with_lint_includes_warnings() {
        let input = "test ok\nwarning[unused_vars]: x is unused\nerror[E0308]: type mismatch";
        let result = extract_failures_with_lint(input);
        assert!(
            result.contains("warning["),
            "lint mode should include warning lines"
        );
        assert!(
            result.contains("error["),
            "lint mode should include error lines"
        );
    }

    #[test]
    fn extract_failures_no_lint_excludes_warnings() {
        let input = "warning[unused_vars]: x is unused\nsome clean line\n";
        let result = extract_failures(input);
        assert!(
            result.is_empty(),
            "non-lint mode should not capture warning[ lines"
        );
    }

    #[test]
    fn extract_failures_finds_panic_lines() {
        let input = "thread 'main' panicked at 'assertion failed', src/lib.rs:10\n";
        let result = extract_failures(input);
        assert!(result.contains("panicked at"), "should capture panicked at");
        assert!(
            result.contains("thread '"),
            "should capture thread panic line"
        );
    }

    #[test]
    fn extract_failures_truncation_is_char_safe() {
        // Build lines containing "FAILED" with emoji so the joined result exceeds 8000 bytes
        // Each line: "FAILED 🦀🦀🦀🦀🦀" = 7 + 5*4 = 27 bytes; 400 lines → ~10800 bytes
        let line = "FAILED 🦀🦀🦀🦀🦀";
        let input: String = vec![line; 400].join("\n");
        // Should not panic even though the join may cut mid-emoji at byte 8000
        let result = extract_failures(&input);
        assert!(
            result.ends_with("\n[truncated]"),
            "truncated result should end with marker"
        );
    }

    #[test]
    fn extract_failures_assertion_keyword_captured() {
        let input = "assertion `left == right` failed\ntest bar ... ok\n";
        let result = extract_failures(input);
        assert!(
            result.contains("assertion"),
            "should capture assertion lines"
        );
        assert!(
            !result.contains("test bar"),
            "should not capture passing lines"
        );
    }

    #[test]
    fn extract_failures_empty_input_returns_empty() {
        let result = extract_failures("");
        assert!(result.is_empty(), "empty input should produce empty output");
    }

    #[test]
    fn extract_failures_with_lint_does_not_include_clean_lines() {
        let input = "warning[unused]: x\ntest ok\nsome info line\n";
        let result = extract_failures_with_lint(input);
        assert!(result.contains("warning"), "should include warning line");
        assert!(
            !result.contains("test ok"),
            "should not include clean lines"
        );
        assert!(
            !result.contains("some info line"),
            "should not include irrelevant lines"
        );
    }

    #[test]
    fn parse_diagnostics_rust_error_block() {
        let input = "\
error[E0308]: mismatched types
  --> src/foo.rs:10:5
   |
10 |     let x: u32 = \"hello\";
   |                  ^^^^^^^ expected `u32`, found `&str`
";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Error);
        assert_eq!(diags[0].code.as_deref(), Some("E0308"));
        assert_eq!(diags[0].message, "mismatched types");
        assert_eq!(diags[0].file.as_deref(), Some("src/foo.rs"));
        assert_eq!(diags[0].line, Some(10));
        assert!(diags[0].context_lines.len() >= 3);
    }

    #[test]
    fn parse_diagnostics_multiple_errors() {
        let input = "\
error[E0308]: mismatched types
  --> src/foo.rs:10:5
   |
error[E0425]: cannot find value
  --> src/bar.rs:20:9
   |
";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 2);
        assert_eq!(diags[0].file.as_deref(), Some("src/foo.rs"));
        assert_eq!(diags[1].file.as_deref(), Some("src/bar.rs"));
    }

    #[test]
    fn parse_diagnostics_warning_captured() {
        let input = "warning[dead_code]: unused function\n  --> src/lib.rs:5:1\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Warning);
        assert_eq!(diags[0].code.as_deref(), Some("dead_code"));
    }

    #[test]
    fn parse_diagnostics_test_failure_block() {
        let input = "\
---- tests::my_test stdout ----
thread 'tests::my_test' panicked at 'assertion failed'
note: run with `RUST_BACKTRACE=1`

";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("my_test"));
        assert!(diags[0].context_lines.len() >= 2);
    }

    #[test]
    fn parse_diagnostics_no_diagnostics() {
        let input = "Compiling foo v0.1.0\nFinished dev profile\ntest result: ok. 5 passed\n";
        let diags = parse_diagnostics(input);
        assert!(diags.is_empty());
    }

    #[test]
    fn format_diagnostics_structured_output() {
        let diags = vec![Diagnostic {
            level: DiagLevel::Error,
            code: Some("E0308".to_string()),
            message: "mismatched types".to_string(),
            file: Some("src/foo.rs".to_string()),
            line: Some(10),
            context_lines: vec!["error[E0308]: mismatched types".to_string()],
        }];
        let out = format_diagnostics(&diags);
        assert!(out.contains("Error 1"), "should have error number");
        assert!(out.contains("[E0308]"), "should have error code");
        assert!(out.contains("src/foo.rs:10"), "should have file:line");
    }

    #[test]
    fn format_diagnostics_caps_at_8000() {
        let diags: Vec<Diagnostic> = (0..50)
            .map(|i| Diagnostic {
                level: DiagLevel::Error,
                code: Some(format!("E{:04}", i)),
                message: "x".repeat(200),
                file: Some("src/long.rs".to_string()),
                line: Some(i),
                context_lines: vec!["context line".repeat(20)],
            })
            .collect();
        let out = format_diagnostics(&diags);
        assert!(
            out.len() <= 8200,
            "should be capped near 8000: {}",
            out.len()
        );
    }

    #[test]
    fn format_diagnostics_multibyte_no_panic() {
        let diags: Vec<Diagnostic> = (0..50)
            .map(|i| Diagnostic {
                level: DiagLevel::Error,
                code: Some(format!("E{:04}", i)),
                message: "型が一致しません".repeat(30),
                file: Some("src/日本語.rs".to_string()),
                line: Some(i),
                context_lines: vec!["コンテキスト行".repeat(20)],
            })
            .collect();
        let out = format_diagnostics(&diags);
        assert!(
            std::str::from_utf8(out.as_bytes()).is_ok(),
            "must be valid UTF-8"
        );
        assert!(out.contains("[truncated]"), "should be truncated");
    }

    #[test]
    fn extract_failures_uses_structured_when_available() {
        let input = "\
error[E0308]: mismatched types
  --> src/foo.rs:10:5
   |
10 |     let x: u32 = \"hello\";
   |                  ^^^^^^^ expected `u32`, found `&str`
";
        let result = extract_failures(input);
        assert!(
            result.contains("src/foo.rs:10"),
            "structured output should include file:line: {}",
            result
        );
        assert!(result.contains("E0308"), "should include error code");
    }

    #[test]
    fn parse_diagnostics_bare_warning_captured() {
        let input = "warning: trait objects without an explicit `dyn` are deprecated\n  --> src/lib.rs:15:10\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Warning);
        assert!(diags[0].message.contains("trait objects"));
    }

    #[test]
    fn parse_diagnostics_all_bare_warnings_captured() {
        let input = "warning: field `foo` is never read\nwarning: associated function `bar` is never used\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 2);
        assert!(diags.iter().all(|d| d.level == DiagLevel::Warning));
    }

    #[test]
    fn diagnostic_summary_line_with_all_fields() {
        let d = Diagnostic {
            level: DiagLevel::Error,
            code: Some("E0308".to_string()),
            message: "mismatched types".to_string(),
            file: Some("src/foo.rs".to_string()),
            line: Some(10),
            context_lines: vec![],
        };
        assert_eq!(d.summary_line(), "[E0308] mismatched types → src/foo.rs:10");
    }

    #[test]
    fn diagnostic_summary_line_no_code() {
        let d = Diagnostic {
            level: DiagLevel::Error,
            code: None,
            message: "aborting due to errors".to_string(),
            file: None,
            line: None,
            context_lines: vec![],
        };
        assert_eq!(d.summary_line(), "aborting due to errors");
    }

    #[test]
    fn diagnostic_summary_line_no_location() {
        let d = Diagnostic {
            level: DiagLevel::Error,
            code: Some("E0308".to_string()),
            message: "mismatched types".to_string(),
            file: None,
            line: None,
            context_lines: vec![],
        };
        assert_eq!(d.summary_line(), "[E0308] mismatched types");
    }

    // ── Multi-language diagnostic parsing ──────────────────────────────────

    #[test]
    fn parse_diagnostics_go_error() {
        let input = "main.go:15:2: error: undefined: fmt.Printl\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Error);
        assert_eq!(diags[0].file.as_deref(), Some("main.go"));
        assert_eq!(diags[0].line, Some(15));
        assert!(diags[0].message.contains("undefined"));
    }

    #[test]
    fn parse_diagnostics_gcc_error() {
        let input = "src/main.c:42:10: error: expected ';' after expression\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].file.as_deref(), Some("src/main.c"));
        assert_eq!(diags[0].line, Some(42));
    }

    #[test]
    fn parse_diagnostics_gcc_warning() {
        let input = "lib/util.c:10:5: warning: unused variable 'x'\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Warning);
        assert_eq!(diags[0].file.as_deref(), Some("lib/util.c"));
    }

    #[test]
    fn parse_diagnostics_typescript_error() {
        let input = "src/app.ts(15,3): error TS2304: Cannot find name 'foo'\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].level, DiagLevel::Error);
        assert_eq!(diags[0].code.as_deref(), Some("TS2304"));
        assert_eq!(diags[0].file.as_deref(), Some("src/app.ts"));
        assert_eq!(diags[0].line, Some(15));
        assert!(diags[0].message.contains("Cannot find name"));
    }

    #[test]
    fn parse_diagnostics_tsx_error() {
        let input = "components/App.tsx(8,1): error TS1005: '}' expected.\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].file.as_deref(), Some("components/App.tsx"));
    }

    #[test]
    fn parse_diagnostics_python_traceback() {
        let input = "\
  File \"app/main.py\", line 42, in <module>
    result = process(data)
TypeError: 'NoneType' object is not subscriptable
";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1, "should parse python traceback: {:?}", diags);
        assert_eq!(diags[0].level, DiagLevel::Error);
        assert_eq!(diags[0].file.as_deref(), Some("app/main.py"));
        assert_eq!(diags[0].line, Some(42));
        assert!(diags[0].message.contains("TypeError"));
    }

    #[test]
    fn parse_diagnostics_python_traceback_no_error_line() {
        let input = "  File \"test.py\", line 10, in test_fn\n    x = broken()\n";
        let diags = parse_diagnostics(input);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].file.as_deref(), Some("test.py"));
        assert_eq!(diags[0].line, Some(10));
    }

    #[test]
    fn parse_diagnostics_mixed_languages() {
        let input = "\
error[E0308]: mismatched types
  --> src/foo.rs:10:5
   |
main.go:20:5: error: undefined variable
src/app.ts(5,1): error TS2304: Cannot find name 'bar'
";
        let diags = parse_diagnostics(input);
        assert!(
            diags.len() >= 3,
            "should parse all 3 languages: got {}",
            diags.len()
        );
    }

    #[test]
    fn parse_diagnostics_generic_no_false_positive_on_plain_text() {
        let input = "Building project...\nDone in 3.2s\nAll tests passed\n";
        let diags = parse_diagnostics(input);
        assert!(
            diags.is_empty(),
            "should not produce false positives on clean output"
        );
    }

    #[test]
    fn summarize_build_errors_finds_go_errors() {
        let input = "main.go:10:5: error: undefined: foo\nutil.go:20:3: error: cannot convert\n";
        let summary = crate::tools::bash::summarize_build_errors_for_test(input);
        assert!(summary.is_some(), "should detect Go errors");
        let s = summary.unwrap();
        assert!(s.contains("2 error(s)"), "should count errors: {}", s);
    }

    #[test]
    fn summarize_build_errors_finds_tsc_errors() {
        let input = "src/index.ts(3,1): error TS2304: Cannot find name 'x'\n";
        let summary = crate::tools::bash::summarize_build_errors_for_test(input);
        assert!(summary.is_some(), "should detect TypeScript errors");
    }

    #[test]
    fn extract_failures_falls_back_for_unknown_format() {
        let input = "FAILED some/test.py::test_thing\nAssertionError: expected 5 got 3\n";
        let result = extract_failures(input);
        assert!(
            result.contains("FAILED"),
            "should fall back to keyword matching"
        );
    }
}
