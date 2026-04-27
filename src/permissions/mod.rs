//! Tool permission engine — Allow / Deny / Ask rules per tool + arg pattern.
//!
//! Every `Tool::call` is gated through `engine` against rules from
//! `storage` (`~/.dm/permissions.json` and project-local overrides).
//! `prompt` handles interactive Ask resolution in TUI mode; headless
//! mode uses defaults from operator policy.

pub mod engine;
pub mod prompt;
pub mod storage;

use serde::{Deserialize, Serialize};

/// What a permission rule does when it matches
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Behavior {
    Allow,
    Deny,
    Ask,
}

/// A single permission rule: tool name + optional content pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub tool: String,
    /// If Some, only matches when args JSON contains this string
    pub pattern: Option<String>,
    pub behavior: Behavior,
}

impl Rule {
    pub fn tool_wide(tool: &str, behavior: Behavior) -> Self {
        Rule {
            tool: tool.to_string(),
            pattern: None,
            behavior,
        }
    }

    pub fn with_pattern(tool: &str, pattern: &str, behavior: Behavior) -> Self {
        Rule {
            tool: tool.to_string(),
            pattern: Some(pattern.to_string()),
            behavior,
        }
    }

    pub fn matches(&self, tool_name: &str, args_json: &str) -> bool {
        if self.tool != tool_name {
            return false;
        }
        match &self.pattern {
            None => true,
            Some(pat) => args_json.contains(pat.as_str()),
        }
    }
}

/// The outcome of a permission check
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Allow,
    Deny,
    Ask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_matches_same_tool_no_pattern() {
        let rule = Rule::tool_wide("bash", Behavior::Allow);
        assert!(rule.matches("bash", "{}"));
    }

    #[test]
    fn rule_does_not_match_different_tool() {
        let rule = Rule::tool_wide("bash", Behavior::Allow);
        assert!(!rule.matches("read_file", "{}"));
    }

    #[test]
    fn rule_with_pattern_matches_when_args_contain_pattern() {
        let rule = Rule::with_pattern("bash", "rm -rf", Behavior::Deny);
        assert!(rule.matches("bash", r#"{"command":"rm -rf /tmp"}"#));
    }

    #[test]
    fn rule_with_pattern_does_not_match_when_args_lack_pattern() {
        let rule = Rule::with_pattern("bash", "rm -rf", Behavior::Deny);
        assert!(!rule.matches("bash", r#"{"command":"echo hello"}"#));
    }

    #[test]
    fn rule_with_pattern_wrong_tool_returns_false() {
        let rule = Rule::with_pattern("bash", "rm", Behavior::Deny);
        assert!(!rule.matches("read_file", r#"{"path":"rm"}"#));
    }

    #[test]
    fn rule_tool_wide_any_args_match() {
        // A tool_wide rule with no pattern should match regardless of args content
        let rule = Rule::tool_wide("bash", Behavior::Allow);
        assert!(rule.matches("bash", r#"{"command":"anything at all"}"#));
        assert!(rule.matches("bash", "{}"));
        assert!(rule.matches("bash", ""));
    }

    #[test]
    fn behavior_equality() {
        assert_eq!(Behavior::Allow, Behavior::Allow);
        assert_ne!(Behavior::Allow, Behavior::Deny);
        assert_ne!(Behavior::Deny, Behavior::Ask);
    }

    #[test]
    fn rule_pattern_partial_match_in_args() {
        // Pattern is a substring of the args — any substring triggers a match
        let rule = Rule::with_pattern("bash", "secret", Behavior::Deny);
        assert!(rule.matches("bash", r#"{"command":"echo secret password"}"#));
        assert!(!rule.matches("bash", r#"{"command":"echo safe message"}"#));
    }

    /// Pin the substring (NOT prefix, NOT whole-word) match semantics on
    /// `Rule::matches`. The implementation uses `args_json.contains(pat)`,
    /// which means:
    /// - matches at the start (prefix-style — any candidate refactor preserves this)
    /// - matches at the end (suffix — would fail under "prefix-only" refactor)
    /// - matches mid-string with separators (broad — would fail under "whole-word" refactor)
    /// - matches mid-string with no separators (most aggressive — strongest signal that
    ///   the contract is plain substring, not regex/word-boundary)
    ///
    /// And the negative case: a pattern that never appears anywhere as a
    /// substring must not match. Together these pin the security-broad
    /// contract that `bash` tool rules apply to `bash X foo Y`-style
    /// invocations even when the trigger token is buried mid-args.
    #[test]
    fn rule_pattern_substring_semantics_not_prefix_or_whole_word() {
        let rule = Rule::with_pattern("bash", "foo", Behavior::Deny);

        // Prefix position — should match (would also pass under prefix-only refactor).
        assert!(
            rule.matches("bash", r#"{"command":"foo bar"}"#),
            "substring at start must match"
        );
        // Suffix position — must match. Fails if a refactor switches to prefix-only.
        assert!(
            rule.matches("bash", r#"{"command":"baz foo"}"#),
            "substring at end must match (catches prefix-only regression)"
        );
        // Mid-string with whitespace separators — must match. Fails if a
        // refactor switches to whole-word matching that's anchored on
        // non-alphanumeric boundaries (which would still pass here, so we
        // also need the next assertion).
        assert!(
            rule.matches("bash", r#"{"command":"baz foo bar"}"#),
            "substring in middle (whitespace-separated) must match"
        );
        // Mid-string with NO separators — must match. Fails if a refactor
        // switches to word-boundary-aware matching.
        assert!(
            rule.matches("bash", r#"{"command":"prefoobar"}"#),
            "substring buried with no boundaries must match (catches whole-word regression)"
        );

        // Negative: pattern absent entirely → no match.
        assert!(
            !rule.matches("bash", r#"{"command":"bar baz"}"#),
            "non-substring args must not match"
        );
    }

    #[test]
    fn rule_with_empty_pattern_matches_any_args() {
        // `parse_rule_string("bash()")` produces `pattern: Some("")` (pinned
        // in storage::tests::parse_rule_empty_pattern). Once that empty
        // pattern lands in `matches`, `args_json.contains("")` is always
        // true — so an empty-pattern rule behaves identically to a tool-wide
        // rule. A future refactor that special-cases `Some("")` to "match
        // nothing" or "literal empty args" would change auth semantics
        // silently; pin the current behavior so the change has to be
        // intentional.
        let rule = Rule::with_pattern("bash", "", Behavior::Deny);
        assert!(rule.matches("bash", r#"{"command":"anything"}"#));
        assert!(rule.matches("bash", "{}"));
        assert!(rule.matches("bash", ""));
        // Different tool still doesn't match (tool gate runs first).
        assert!(!rule.matches("read_file", r#"{"path":"x"}"#));
    }
}
