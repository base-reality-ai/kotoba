//! `dm run <spec.md>` — execute a markdown run-spec sequentially.
//!
//! Format:
//!   ## Turn label (optional — used only for display / transcript headers)
//!   Prompt text for this turn.
//!
//!   ## Another turn
//!   More prompt text.
//!
//! Rules:
//! - `##` headings delimit turns.
//! - Body text (trimmed) under each heading is the prompt.
//! - Turns with an empty prompt after trimming are skipped.
//! - Each turn runs with bypass-all permissions.
//! - `--run-output <file>` writes a markdown transcript.

use crate::logging;
use anyhow::{Context, Result};
use std::fmt::Write as _;
use std::path::Path;

/// A single parsed turn from the run-spec.
#[derive(Debug, Clone, PartialEq)]
pub struct RunTurn {
    /// Optional label from the `##` heading.
    pub label: Option<String>,
    /// Prompt text (trimmed body under the heading).
    pub prompt: String,
}

/// Parse a markdown run-spec into an ordered list of turns.
/// Only `##` headings are treated as turn delimiters; `#`, `###`, etc. are
/// included as part of the prompt body.
pub fn parse_spec(markdown: &str) -> Vec<RunTurn> {
    let mut turns: Vec<RunTurn> = Vec::new();
    let mut current_label: Option<String> = None;
    let mut body_lines: Vec<&str> = Vec::new();

    for line in markdown.lines() {
        if let Some(rest) = line.strip_prefix("## ") {
            // Flush previous turn (if any body)
            flush(&mut turns, &mut current_label, &mut body_lines);
            current_label = Some(rest.trim().to_string());
        } else if line.trim_start().starts_with("## ") {
            // Indented ## — treat as new turn
            let rest = line.trim_start().strip_prefix("## ").unwrap_or("").trim();
            flush(&mut turns, &mut current_label, &mut body_lines);
            current_label = Some(rest.to_string());
        } else {
            body_lines.push(line);
        }
    }
    // Flush the last pending turn
    flush(&mut turns, &mut current_label, &mut body_lines);

    // Also handle a spec with no ## headings at all — the whole file is one turn
    if turns.is_empty() {
        let body = markdown.trim().to_string();
        if !body.is_empty() {
            turns.push(RunTurn {
                label: None,
                prompt: body,
            });
        }
    }

    turns
}

fn flush(turns: &mut Vec<RunTurn>, label: &mut Option<String>, body: &mut Vec<&str>) {
    if label.is_some() {
        let prompt = body.join("\n").trim().to_string();
        if !prompt.is_empty() {
            turns.push(RunTurn {
                label: label.take(),
                prompt,
            });
        } else {
            label.take(); // discard empty turns
        }
        body.clear();
    } else if !body.is_empty() {
        body.clear(); // pre-heading preamble: discard
    }
}

/// Substitute template variables in a prompt string.
///
/// Supported patterns:
/// - `{{model}}` — model name from the active client
/// - `{{date}}` — today's date in `YYYY-MM-DD` format
/// - `{{env:VAR}}` — value of environment variable `VAR` (empty string if unset)
/// - `{{routing.KEY}}` — value from routing config under key `KEY`
/// - Unknown patterns are left as-is (passthrough).
pub fn substitute_vars(
    text: &str,
    model: &str,
    routing: &Option<crate::config::RoutingConfig>,
) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("{{") {
        result.push_str(&rest[..start]);
        rest = &rest[start + 2..];
        if let Some(end) = rest.find("}}") {
            let key = &rest[..end];
            rest = &rest[end + 2..];
            if key == "model" {
                result.push_str(model);
            } else if key == "date" {
                // Use chrono if available; otherwise fall back to a simple approach via SystemTime.
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days = now / 86400;
                // Compute year/month/day from days since epoch (1970-01-01).
                let (y, m, d) = days_to_ymd(days);
                write!(result, "{:04}-{:02}-{:02}", y, m, d).expect("write to String never fails");
            } else if let Some(var) = key.strip_prefix("env:") {
                result.push_str(&std::env::var(var).unwrap_or_default());
            } else if let Some(routing_key) = key.strip_prefix("routing.") {
                if let Some(rc) = routing {
                    if routing_key == "default" {
                        result.push_str(&rc.default);
                    } else if let Some(model_name) = rc.rules.get(routing_key) {
                        result.push_str(model_name);
                    } else {
                        // Unknown routing key — passthrough
                        result.push_str("{{");
                        result.push_str(key);
                        result.push_str("}}");
                    }
                } else {
                    result.push_str("{{");
                    result.push_str(key);
                    result.push_str("}}");
                }
            } else {
                // Unknown variable — passthrough
                result.push_str("{{");
                result.push_str(key);
                result.push_str("}}");
            }
        } else {
            // Unclosed `{{` — emit literally and stop scanning
            result.push_str("{{");
            result.push_str(rest);
            rest = "";
        }
    }
    result.push_str(rest);
    result
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
    days += 719468;
    let era = days / 146097;
    let doe = days % 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m as u32, d as u32)
}

/// Execute a run-spec file, printing responses to stdout.
/// `output_path`: if Some, also write a markdown transcript.
/// `dry_run`: if true, print prompts but do not call the LLM.
#[allow(clippy::too_many_arguments)]
pub async fn run_spec(
    spec_path: &Path,
    output_path: Option<&Path>,
    client: &crate::ollama::client::OllamaClient,
    config_dir: &Path,
    embed_model: &str,
    verbose: bool,
    dry_run: bool,
    routing: &Option<crate::config::RoutingConfig>,
) -> Result<()> {
    let content = tokio::fs::read_to_string(spec_path)
        .await
        .with_context(|| format!("Cannot read run-spec: {}", spec_path.display()))?;

    let turns = parse_spec(&content);
    if turns.is_empty() {
        logging::log("[dm run] No turns found in spec — nothing to do.");
        return Ok(());
    }

    logging::log(&format!(
        "[dm run] {} turn{} from {}",
        turns.len(),
        if turns.len() == 1 { "" } else { "s" },
        spec_path.display()
    ));

    let mut transcript = String::new();
    if output_path.is_some() {
        write!(
            transcript,
            "# dm run transcript — {}\n\n",
            spec_path.display()
        )
        .expect("write to String never fails");
    }

    for (i, turn) in turns.iter().enumerate() {
        let default_label = format!("Turn {}", i + 1);
        let label = turn.label.as_deref().unwrap_or(&default_label);

        let prompt = substitute_vars(&turn.prompt, client.model(), routing);

        logging::log(&format!("\n[dm run] ── {} ──", label));
        if verbose {
            let mut prompt_preview_end = prompt.len().min(120);
            while prompt_preview_end > 0 && !prompt.is_char_boundary(prompt_preview_end) {
                prompt_preview_end -= 1;
            }
            logging::log(&format!(
                "[dm run] prompt: {}",
                &prompt[..prompt_preview_end]
            ));
        }

        if dry_run {
            println!("\n## {} [dry-run]\n\n{}", label, prompt);
            if output_path.is_some() {
                write!(transcript, "## {} [dry-run]\n\n", label)
                    .expect("write to String never fails");
                transcript.push_str("**Prompt:**\n\n");
                transcript.push_str(&prompt);
                transcript.push_str("\n\n---\n\n");
            }
            continue;
        }

        let registry = crate::tools::registry::default_registry(
            &format!("run-{}", i),
            config_dir,
            client.base_url(),
            client.model(),
            embed_model,
        );

        let capture =
            crate::conversation::run_conversation_capture(&prompt, "print", client, &registry)
                .await
                .with_context(|| format!("Turn '{}' failed", label))?;
        let response = &capture.text;

        println!("\n## {}\n\n{}", label, response);

        if output_path.is_some() {
            write!(transcript, "## {}\n\n", label).expect("write to String never fails");
            transcript.push_str("**Prompt:**\n\n");
            transcript.push_str(&prompt);
            transcript.push_str("\n\n**Response:**\n\n");
            transcript.push_str(response);
            transcript.push_str("\n\n---\n\n");
        }
    }

    if let Some(out) = output_path {
        tokio::fs::write(out, &transcript)
            .await
            .with_context(|| format!("Cannot write transcript to {}", out.display()))?;
        logging::log(&format!("\n[dm run] transcript saved to {}", out.display()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_headed_turns() {
        let md = "## First\nHello world.\n\n## Second\nGoodbye.";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].label.as_deref(), Some("First"));
        assert_eq!(turns[0].prompt, "Hello world.");
        assert_eq!(turns[1].label.as_deref(), Some("Second"));
        assert_eq!(turns[1].prompt, "Goodbye.");
    }

    #[test]
    fn parse_no_headings_whole_file_is_one_turn() {
        let md = "explain this code";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].label, None);
        assert_eq!(turns[0].prompt, "explain this code");
    }

    #[test]
    fn parse_empty_body_turns_are_skipped() {
        let md = "## Empty\n\n## WithBody\nActual prompt.";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].label.as_deref(), Some("WithBody"));
    }

    #[test]
    fn parse_multiline_body() {
        let md = "## Multi\nLine one.\nLine two.\n\nLine three after blank.";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 1);
        let body = &turns[0].prompt;
        assert!(body.contains("Line one."), "should contain line one");
        assert!(body.contains("Line two."), "should contain line two");
        assert!(body.contains("Line three"), "should contain line three");
    }

    #[test]
    fn parse_empty_spec_returns_empty() {
        let turns = parse_spec("  \n  \n");
        assert!(turns.is_empty());
    }

    #[test]
    fn parse_h1_and_h3_not_treated_as_delimiters() {
        let md = "# Title\n## Turn\nThe # title and ### sub are part of body.\n### sub\nmore.";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].label.as_deref(), Some("Turn"));
        assert!(turns[0].prompt.contains("# title"));
        assert!(turns[0].prompt.contains("### sub"));
    }

    #[test]
    fn parse_preamble_before_first_heading_discarded() {
        let md = "Some preamble text.\n\n## Real Turn\nPrompt here.";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].prompt, "Prompt here.");
    }

    #[test]
    fn parse_label_whitespace_trimmed() {
        let md = "##   Spaced Label   \nBody.";
        let turns = parse_spec(md);
        assert_eq!(turns[0].label.as_deref(), Some("Spaced Label"));
    }

    #[test]
    fn parse_three_turns_order_preserved() {
        let md = "## A\na\n## B\nb\n## C\nc";
        let turns = parse_spec(md);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].label.as_deref(), Some("A"));
        assert_eq!(turns[1].label.as_deref(), Some("B"));
        assert_eq!(turns[2].label.as_deref(), Some("C"));
    }

    // ── substitute_vars tests ──────────────────────────────────────────────────

    #[test]
    fn var_substitution_model() {
        let result = substitute_vars("Use {{model}} for this task.", "gemma4:26b", &None);
        assert_eq!(result, "Use gemma4:26b for this task.");
    }

    #[test]
    fn var_substitution_date_format() {
        let result = substitute_vars("Today is {{date}}.", "m", &None);
        // Just check it looks like YYYY-MM-DD (10 chars, dashes at positions 4 and 7)
        let date_part = result
            .strip_prefix("Today is ")
            .unwrap()
            .strip_suffix(".")
            .unwrap();
        assert_eq!(date_part.len(), 10, "date should be YYYY-MM-DD");
        assert_eq!(&date_part[4..5], "-");
        assert_eq!(&date_part[7..8], "-");
        let year: u32 = date_part[..4].parse().unwrap();
        assert!(year >= 2024, "year should be plausible");
    }

    #[test]
    fn var_substitution_env() {
        std::env::set_var("DM_TEST_VAR", "hello_env");
        let result = substitute_vars("val={{env:DM_TEST_VAR}}", "m", &None);
        assert_eq!(result, "val=hello_env");
        std::env::remove_var("DM_TEST_VAR");
    }

    #[test]
    fn var_substitution_env_missing_is_empty() {
        std::env::remove_var("DM_DEFINITELY_NOT_SET");
        let result = substitute_vars("val={{env:DM_DEFINITELY_NOT_SET}}", "m", &None);
        assert_eq!(result, "val=");
    }

    #[test]
    fn var_substitution_routing_key() {
        let mut routes = std::collections::HashMap::new();
        routes.insert("code".to_string(), "codellama:34b".to_string());
        let rc = crate::config::RoutingConfig {
            rules: routes,
            default: "gemma4:26b".to_string(),
        };
        let result = substitute_vars("code model: {{routing.code}}", "m", &Some(rc));
        assert_eq!(result, "code model: codellama:34b");
    }

    #[test]
    fn var_substitution_routing_default() {
        let rc = crate::config::RoutingConfig {
            rules: std::collections::HashMap::new(),
            default: "llama3:8b".to_string(),
        };
        let result = substitute_vars("default: {{routing.default}}", "m", &Some(rc));
        assert_eq!(result, "default: llama3:8b");
    }

    #[test]
    fn var_substitution_unknown_passthrough() {
        let result = substitute_vars("keep {{unknown_var}} as is", "m", &None);
        assert_eq!(result, "keep {{unknown_var}} as is");
    }

    #[test]
    fn var_substitution_no_vars_returns_unchanged() {
        let s = "just a plain string with no braces";
        assert_eq!(substitute_vars(s, "model", &None), s);
    }

    #[test]
    fn var_substitution_unclosed_brace_emits_literally() {
        // Unclosed {{ should be emitted as-is and not panic
        let result = substitute_vars("prefix {{unclosed", "m", &None);
        assert!(
            result.starts_with("prefix {{"),
            "should emit {{ literally: {result}"
        );
        assert!(
            result.contains("unclosed"),
            "rest of string preserved: {result}"
        );
    }

    #[test]
    fn var_substitution_multiple_vars_in_sequence() {
        let result = substitute_vars("{{model}} {{model}}", "gemma", &None);
        assert_eq!(result, "gemma gemma");
    }

    #[test]
    fn var_substitution_routing_key_missing_no_config_passthrough() {
        // routing.something when routing config is None → passthrough
        let result = substitute_vars("{{routing.fast}}", "m", &None);
        assert_eq!(
            result, "{{routing.fast}}",
            "should passthrough when no routing config"
        );
    }

    #[test]
    fn var_substitution_routing_unknown_key_passthrough() {
        // routing.unknownkey when config exists but key not in rules → passthrough
        let rc = crate::config::RoutingConfig {
            rules: std::collections::HashMap::new(),
            default: "llama3:8b".to_string(),
        };
        let result = substitute_vars("{{routing.nonexistent}}", "m", &Some(rc));
        assert_eq!(result, "{{routing.nonexistent}}");
    }

    #[test]
    fn days_to_ymd_epoch_is_1970_01_01() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date_2024_02_29() {
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }

    #[test]
    fn days_to_ymd_known_date_2000_03_01() {
        assert_eq!(days_to_ymd(11017), (2000, 3, 1));
    }

    #[test]
    fn days_to_ymd_end_of_year() {
        assert_eq!(days_to_ymd(19722), (2023, 12, 31));
    }

    #[test]
    fn days_to_ymd_start_of_year() {
        assert_eq!(days_to_ymd(19723), (2024, 1, 1));
    }
}
