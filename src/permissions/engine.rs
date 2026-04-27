use super::{Behavior, Decision, Rule};
use std::path::Path;

#[derive(Clone)]
pub struct PermissionEngine {
    pub bypass_all: bool,
    settings_rules: Vec<Rule>,
    session_rules: Vec<Rule>,
}

impl PermissionEngine {
    pub fn new(bypass_all: bool, settings_rules: Vec<Rule>) -> Self {
        PermissionEngine {
            bypass_all,
            settings_rules,
            session_rules: Vec::new(),
        }
    }

    /// Whether the engine is in `--dangerously-skip-permissions` / bypass mode.
    /// Callers that apply additional safety gates (e.g. the bash risk floor)
    /// must honor this so `--bypass-permissions` still means what it says.
    pub fn is_bypass(&self) -> bool {
        self.bypass_all
    }

    pub fn add_session_rule(&mut self, rule: Rule) {
        self.session_rules.push(rule);
    }

    pub fn add_settings_rule(&mut self, rule: Rule) {
        self.settings_rules.push(rule);
    }

    /// Save current settings rules to disk
    pub fn save_settings(&self, config_dir: &Path) -> anyhow::Result<()> {
        super::storage::save_rules(config_dir, &self.settings_rules)
    }

    /// Describe current rules as a human-readable string for the /permissions command.
    pub fn describe_rules(&self) -> String {
        if self.bypass_all {
            return "⚠ All permissions bypassed (--dangerously-skip-permissions)\n".to_string();
        }

        let mut lines = vec!["Current permission rules:".to_string()];

        if self.settings_rules.is_empty() && self.session_rules.is_empty() {
            lines.push("  (no rules — all tool calls will prompt for permission)".to_string());
        }

        if !self.settings_rules.is_empty() {
            lines.push("  Persistent (settings):".to_string());
            for rule in &self.settings_rules {
                let pat = rule
                    .pattern
                    .as_deref()
                    .map(|p| format!(" (pattern: {})", p))
                    .unwrap_or_default();
                lines.push(format!("    {:?} — {}{}", rule.behavior, rule.tool, pat));
            }
        }

        if !self.session_rules.is_empty() {
            lines.push("  Session (until restart):".to_string());
            for rule in &self.session_rules {
                let pat = rule
                    .pattern
                    .as_deref()
                    .map(|p| format!(" (pattern: {})", p))
                    .unwrap_or_default();
                lines.push(format!("    {:?} — {}{}", rule.behavior, rule.tool, pat));
            }
        }

        lines.join("\n")
    }

    /// Priority: `bypass_all` > `session_rules` (last-match wins) > `settings_rules` (last-match wins) > Ask
    pub fn check(&self, tool_name: &str, args: &serde_json::Value) -> Decision {
        if self.bypass_all {
            return Decision::Allow;
        }

        let args_str = args.to_string();

        for rule in self.session_rules.iter().rev() {
            if rule.matches(tool_name, &args_str) {
                return match rule.behavior {
                    Behavior::Allow => Decision::Allow,
                    Behavior::Deny => Decision::Deny,
                    Behavior::Ask => Decision::Ask,
                };
            }
        }

        for rule in self.settings_rules.iter().rev() {
            if rule.matches(tool_name, &args_str) {
                return match rule.behavior {
                    Behavior::Allow => Decision::Allow,
                    Behavior::Deny => Decision::Deny,
                    Behavior::Ask => Decision::Ask,
                };
            }
        }

        Decision::Ask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bypass_all_always_allows() {
        let engine = PermissionEngine::new(true, vec![]);
        let result = engine.check("bash", &serde_json::json!({"command": "rm -rf /"}));
        assert_eq!(result, Decision::Allow);
    }

    #[test]
    fn no_rules_returns_ask() {
        let engine = PermissionEngine::new(false, vec![]);
        let result = engine.check("bash", &serde_json::json!({}));
        assert_eq!(result, Decision::Ask);
    }

    #[test]
    fn deny_rule_blocks_tool() {
        let rule = Rule::tool_wide("bash", Behavior::Deny);
        let engine = PermissionEngine::new(false, vec![rule]);
        assert_eq!(engine.check("bash", &serde_json::json!({})), Decision::Deny);
    }

    #[test]
    fn allow_rule_permits_tool() {
        let rule = Rule::tool_wide("bash", Behavior::Allow);
        let engine = PermissionEngine::new(false, vec![rule]);
        assert_eq!(
            engine.check("bash", &serde_json::json!({})),
            Decision::Allow
        );
    }

    #[test]
    fn pattern_rule_matches_args() {
        let rule = Rule::with_pattern("bash", "rm -rf", Behavior::Deny);
        let engine = PermissionEngine::new(false, vec![rule]);
        assert_eq!(
            engine.check("bash", &serde_json::json!({"command": "rm -rf /tmp"})),
            Decision::Deny
        );
        // Different command should still Ask (no matching rule)
        assert_eq!(
            engine.check("bash", &serde_json::json!({"command": "echo hello"})),
            Decision::Ask
        );
    }

    #[test]
    fn describe_rules_shows_allow_rule() {
        let engine = PermissionEngine::new(false, vec![Rule::tool_wide("bash", Behavior::Allow)]);
        let desc = engine.describe_rules();
        assert!(desc.contains("bash"));
        assert!(desc.to_lowercase().contains("allow"));
    }

    #[test]
    fn describe_rules_bypass_all() {
        let engine = PermissionEngine::new(true, vec![]);
        let desc = engine.describe_rules();
        assert!(desc.contains("bypassed"));
    }

    #[test]
    fn describe_rules_no_rules_message() {
        let engine = PermissionEngine::new(false, vec![]);
        let desc = engine.describe_rules();
        assert!(desc.contains("no rules"));
    }

    #[test]
    fn session_rule_takes_priority_over_settings() {
        let settings_rule = Rule::tool_wide("bash", Behavior::Allow);
        let mut engine = PermissionEngine::new(false, vec![settings_rule]);
        engine.add_session_rule(Rule::tool_wide("bash", Behavior::Deny));
        // Session deny should win over settings allow
        assert_eq!(engine.check("bash", &serde_json::json!({})), Decision::Deny);
    }

    #[test]
    fn last_settings_rule_wins_when_multiple_match() {
        // Two settings rules for bash: first Allow, then Deny → last one wins
        let engine = PermissionEngine::new(
            false,
            vec![
                Rule::tool_wide("bash", Behavior::Allow),
                Rule::tool_wide("bash", Behavior::Deny),
            ],
        );
        assert_eq!(engine.check("bash", &serde_json::json!({})), Decision::Deny);
    }

    #[test]
    fn rule_for_one_tool_does_not_affect_another() {
        let engine = PermissionEngine::new(false, vec![Rule::tool_wide("bash", Behavior::Allow)]);
        // "read_file" has no rule → should Ask
        assert_eq!(
            engine.check("read_file", &serde_json::json!({"path": "/tmp/x"})),
            Decision::Ask
        );
    }
}
