use super::{Behavior, Rule};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Settings {
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
}

fn settings_path(config_dir: &Path) -> std::path::PathBuf {
    config_dir.join("settings.json")
}

pub fn load_rules(config_dir: &Path) -> Result<Vec<Rule>> {
    let path = settings_path(config_dir);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&path).context("Failed to read settings.json")?;
    let settings: Settings =
        serde_json::from_str(&content).context("Failed to parse settings.json")?;

    let mut rules = Vec::new();
    for entry in &settings.allow {
        rules.push(parse_rule_string(entry, Behavior::Allow));
    }
    for entry in &settings.deny {
        rules.push(parse_rule_string(entry, Behavior::Deny));
    }
    Ok(rules)
}

pub fn save_rules(config_dir: &Path, rules: &[Rule]) -> Result<()> {
    let mut settings = Settings::default();
    for rule in rules {
        let s = format_rule_string(rule);
        match rule.behavior {
            Behavior::Allow => settings.allow.push(s),
            Behavior::Deny => settings.deny.push(s),
            Behavior::Ask => {} // Ask rules are not persisted
        }
    }

    let content = serde_json::to_string_pretty(&settings)?;
    std::fs::write(settings_path(config_dir), content).context("Failed to write settings.json")?;
    Ok(())
}

fn parse_rule_string(s: &str, behavior: Behavior) -> Rule {
    // Format: "toolname" or "toolname(pattern)"
    if let Some(open) = s.find('(') {
        if s.ends_with(')') {
            let tool = &s[..open];
            let pattern = &s[open + 1..s.len() - 1];
            return Rule::with_pattern(tool, pattern, behavior);
        }
    }
    Rule::tool_wide(s, behavior)
}

fn format_rule_string(rule: &Rule) -> String {
    match &rule.pattern {
        None => rule.tool.clone(),
        Some(p) => format!("{}({})", rule.tool, p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::Behavior;
    use tempfile::TempDir;

    // ── parse_rule_string ──────────────────────────────────────────────────────

    #[test]
    fn parse_rule_tool_wide_no_parens() {
        let rule = parse_rule_string("bash", Behavior::Allow);
        assert_eq!(rule.tool, "bash");
        assert!(rule.pattern.is_none());
        assert_eq!(rule.behavior, Behavior::Allow);
    }

    #[test]
    fn parse_rule_with_pattern() {
        let rule = parse_rule_string("bash(rm -rf)", Behavior::Deny);
        assert_eq!(rule.tool, "bash");
        assert_eq!(rule.pattern.as_deref(), Some("rm -rf"));
        assert_eq!(rule.behavior, Behavior::Deny);
    }

    #[test]
    fn parse_rule_empty_pattern() {
        let rule = parse_rule_string("bash()", Behavior::Allow);
        assert_eq!(rule.tool, "bash");
        assert_eq!(rule.pattern.as_deref(), Some(""));
    }

    #[test]
    fn parse_rule_unclosed_paren_treated_as_tool_wide() {
        // No closing ')' — falls through to tool_wide
        let rule = parse_rule_string("bash(no_close", Behavior::Allow);
        assert_eq!(rule.tool, "bash(no_close");
        assert!(rule.pattern.is_none());
    }

    // ── format_rule_string ─────────────────────────────────────────────────────

    #[test]
    fn format_tool_wide_rule() {
        let rule = Rule::tool_wide("read_file", Behavior::Allow);
        assert_eq!(format_rule_string(&rule), "read_file");
    }

    #[test]
    fn format_rule_with_pattern() {
        let rule = Rule::with_pattern("bash", "/etc", Behavior::Deny);
        assert_eq!(format_rule_string(&rule), "bash(/etc)");
    }

    // ── round-trip via save/load ───────────────────────────────────────────────

    #[test]
    fn save_and_load_round_trips_rules() {
        let dir = TempDir::new().unwrap();
        let rules = vec![
            Rule::tool_wide("bash", Behavior::Allow),
            Rule::with_pattern("bash", "rm -rf", Behavior::Deny),
            Rule::tool_wide("read_file", Behavior::Allow),
        ];
        save_rules(dir.path(), &rules).unwrap();
        let loaded = load_rules(dir.path()).unwrap();
        // save_rules groups by behavior: all allow first, then deny
        assert_eq!(loaded.len(), 3);
        // allow rules: bash, read_file
        assert_eq!(loaded[0].tool, "bash");
        assert!(loaded[0].pattern.is_none());
        assert_eq!(loaded[0].behavior, Behavior::Allow);
        assert_eq!(loaded[1].tool, "read_file");
        assert_eq!(loaded[1].behavior, Behavior::Allow);
        // deny rule: bash(rm -rf)
        assert_eq!(loaded[2].tool, "bash");
        assert_eq!(loaded[2].pattern.as_deref(), Some("rm -rf"));
        assert_eq!(loaded[2].behavior, Behavior::Deny);
    }

    #[test]
    fn ask_rules_are_not_persisted() {
        let dir = TempDir::new().unwrap();
        let rules = vec![Rule::tool_wide("bash", Behavior::Ask)];
        save_rules(dir.path(), &rules).unwrap();
        let loaded = load_rules(dir.path()).unwrap();
        assert!(loaded.is_empty(), "Ask rules should not be saved");
    }

    #[test]
    fn load_returns_empty_when_no_file() {
        let dir = TempDir::new().unwrap();
        let rules = load_rules(dir.path()).unwrap();
        assert!(rules.is_empty());
    }
}
