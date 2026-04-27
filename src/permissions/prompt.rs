use crate::logging;
use std::io::{self, BufRead};

pub enum UserChoice {
    AllowOnce,
    DenyOnce,
    /// Always allow — persist to settings
    AlwaysAllow,
    /// Always deny — session only
    AlwaysDeny,
}

/// Parse a user's permission-prompt response into a `UserChoice`.
///
/// When `risk_present` is true, a dangerous bash command is in play and the
/// prompt demands typed confirmation:
/// - empty Enter denies (no accidental confirmation via muscle-memory)
/// - bare "y" denies (typo-proof — require the full word "yes")
/// - "yes" allows once; "always"/"a" are rejected (no blanket persistence for
///   risky commands)
///
/// In the safe path (`risk_present == false`) the legacy permissive mapping
/// applies: empty Enter or "y"/"yes" allow once, "a"/"always" persist Allow,
/// "N"/"never" persist Deny, anything else denies once.
pub fn parse_permission_input(line: &str, risk_present: bool) -> UserChoice {
    let trimmed = line.trim();
    if risk_present {
        // Typed-yes path: only the full word "yes" confirms.
        match trimmed {
            "yes" => UserChoice::AllowOnce,
            "N" | "never" => UserChoice::AlwaysDeny,
            _ => UserChoice::DenyOnce,
        }
    } else {
        // Legacy permissive mapping — keep empty-Enter = AllowOnce for
        // non-risky prompts where typing "y" every time is cognitive overhead.
        match trimmed {
            "y" | "yes" | "" => UserChoice::AllowOnce,
            "a" | "always" => UserChoice::AlwaysAllow,
            "N" | "never" => UserChoice::AlwaysDeny,
            _ => UserChoice::DenyOnce,
        }
    }
}

pub fn ask_permission(
    tool_name: &str,
    args: &serde_json::Value,
    risk_reason: Option<&str>,
) -> UserChoice {
    let args_display = format_args_for_display(tool_name, args);

    logging::log("");
    if let Some(reason) = risk_reason {
        logging::log("╭─ ⚠ Dangerous Command ─────────────────────────────────────");
        logging::log(&format!("│  Tool: {}", tool_name));
        if !args_display.is_empty() {
            logging::log(&format!("│  Args: {}", args_display));
        }
        logging::log(&format!("│  Risk: {}", reason));
        logging::log("╰───────────────────────────────────────────────────────────");
        logging::log("  Type 'yes' to allow once, anything else denies  > ");
    } else {
        logging::log("╭─ Permission Required ─────────────────────────────────────");
        logging::log(&format!("│  Tool: {}", tool_name));
        if !args_display.is_empty() {
            logging::log(&format!("│  Args: {}", args_display));
        }
        logging::log("╰───────────────────────────────────────────────────────────");
        logging::log("  [y] Allow once  [n] Deny  [a] Always allow  [N] Deny always  > ");
    }

    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line).ok();
    parse_permission_input(&line, risk_reason.is_some())
}

fn format_args_for_display(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "bash" => args["command"].as_str().unwrap_or("").to_string(),
        "write_file" | "read_file" | "edit_file" | "multi_edit" | "apply_diff"
        | "notebook_edit" => args["path"].as_str().unwrap_or("").to_string(),
        "glob" => args["pattern"].as_str().unwrap_or("").to_string(),
        "grep" | "semantic_search" => {
            let pattern = args["pattern"]
                .as_str()
                .or_else(|| args["query"].as_str())
                .unwrap_or("");
            let path = args["path"].as_str().unwrap_or(".");
            format!("{} in {}", pattern, path)
        }
        "ls" => args["path"].as_str().unwrap_or(".").to_string(),
        "web_fetch" => args["url"].as_str().unwrap_or("").to_string(),
        "web_search" => args["query"].as_str().unwrap_or("").to_string(),
        "agent" => {
            let prompt = args["prompt"].as_str().unwrap_or("");
            if prompt.len() > 80 {
                format!("{}…", crate::util::safe_truncate(prompt, 80))
            } else {
                prompt.to_string()
            }
        }
        _ => {
            let s = args.to_string();
            if s.len() > 80 {
                format!("{}…", crate::util::safe_truncate(&s, 80))
            } else {
                s
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn format_bash_shows_command() {
        let args = json!({"command": "ls -la"});
        assert_eq!(format_args_for_display("bash", &args), "ls -la");
    }

    #[test]
    fn format_read_file_shows_path() {
        let args = json!({"path": "/home/user/file.rs"});
        assert_eq!(
            format_args_for_display("read_file", &args),
            "/home/user/file.rs"
        );
    }

    #[test]
    fn format_write_file_shows_path() {
        let args = json!({"path": "/tmp/out.txt", "content": "hello"});
        assert_eq!(format_args_for_display("write_file", &args), "/tmp/out.txt");
    }

    #[test]
    fn format_edit_file_shows_path() {
        let args = json!({"path": "src/main.rs", "old_string": "x", "new_string": "y"});
        assert_eq!(format_args_for_display("edit_file", &args), "src/main.rs");
    }

    #[test]
    fn format_glob_shows_pattern() {
        let args = json!({"pattern": "**/*.rs"});
        assert_eq!(format_args_for_display("glob", &args), "**/*.rs");
    }

    #[test]
    fn format_grep_shows_pattern_and_path() {
        let args = json!({"pattern": "fn main", "path": "src/"});
        assert_eq!(format_args_for_display("grep", &args), "fn main in src/");
    }

    #[test]
    fn format_unknown_tool_truncates_at_80_chars() {
        // JSON of 100+ chars should be truncated
        let long_val = "x".repeat(100);
        let args = json!({"key": long_val});
        let result = format_args_for_display("unknown_tool", &args);
        // Should end with the ellipsis character
        assert!(
            result.ends_with('…'),
            "should be truncated with …: {}",
            result
        );
        // The truncated part should be ≤ 80 chars (before the …)
        let without_ellipsis: &str = result.trim_end_matches('…');
        assert!(
            without_ellipsis.len() <= 80,
            "truncated text should be ≤ 80 chars"
        );
    }

    #[test]
    fn format_unknown_tool_short_args_not_truncated() {
        let args = json!({"key": "short"});
        let result = format_args_for_display("unknown_tool", &args);
        assert!(!result.ends_with('…'), "short args should not be truncated");
    }

    #[test]
    fn format_bash_missing_command_returns_empty() {
        let args = json!({});
        assert_eq!(format_args_for_display("bash", &args), "");
    }

    #[test]
    fn format_multi_edit_shows_path() {
        let args = json!({"path": "src/lib.rs", "edits": [{"old_text": "a", "new_text": "b"}]});
        assert_eq!(format_args_for_display("multi_edit", &args), "src/lib.rs");
    }

    #[test]
    fn format_apply_diff_shows_path() {
        let args = json!({"path": "src/main.rs", "diff": "@@ -1 +1 @@\n-old\n+new"});
        assert_eq!(format_args_for_display("apply_diff", &args), "src/main.rs");
    }

    #[test]
    fn format_web_fetch_shows_url() {
        let args = json!({"url": "https://example.com/api"});
        assert_eq!(
            format_args_for_display("web_fetch", &args),
            "https://example.com/api"
        );
    }

    #[test]
    fn format_agent_shows_prompt_truncated() {
        let long_prompt = "x".repeat(120);
        let args = json!({"prompt": long_prompt});
        let result = format_args_for_display("agent", &args);
        assert!(
            result.ends_with('…'),
            "long agent prompt should be truncated: {result}"
        );
        assert!(result.len() <= 84); // 80 + "…" (3 bytes)
    }

    // ── parse_permission_input tests ────────────────────────────────────

    #[test]
    fn parse_safe_empty_enter_allows_once() {
        assert!(matches!(
            parse_permission_input("\n", false),
            UserChoice::AllowOnce
        ));
    }

    #[test]
    fn parse_safe_y_allows_once() {
        assert!(matches!(
            parse_permission_input("y\n", false),
            UserChoice::AllowOnce
        ));
    }

    #[test]
    fn parse_safe_always_persists_allow() {
        assert!(matches!(
            parse_permission_input("always\n", false),
            UserChoice::AlwaysAllow
        ));
        assert!(matches!(
            parse_permission_input("a\n", false),
            UserChoice::AlwaysAllow
        ));
    }

    #[test]
    fn parse_risk_empty_enter_denies() {
        // Muscle-memory "just hit Enter" must not allow a dangerous command.
        assert!(matches!(
            parse_permission_input("\n", true),
            UserChoice::DenyOnce
        ));
    }

    #[test]
    fn parse_risk_y_denies() {
        // Typo-proofing: single-letter "y" is not enough under risk.
        assert!(matches!(
            parse_permission_input("y\n", true),
            UserChoice::DenyOnce
        ));
    }

    #[test]
    fn parse_risk_typed_yes_allows_once() {
        assert!(matches!(
            parse_permission_input("yes\n", true),
            UserChoice::AllowOnce
        ));
    }

    #[test]
    fn parse_risk_always_rejected() {
        // No blanket persistence for risky commands.
        assert!(matches!(
            parse_permission_input("always\n", true),
            UserChoice::DenyOnce
        ));
        assert!(matches!(
            parse_permission_input("a\n", true),
            UserChoice::DenyOnce
        ));
    }
}
