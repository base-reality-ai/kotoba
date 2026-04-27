//! Prompt-based model routing.
//!
//! Classifies a prompt with simple keyword/length heuristics (no LLM call),
//! then maps the resulting key to a model name via `~/.dm/config.toml [routing]`.

use crate::config::RoutingConfig;

/// Symbolic bucket a prompt is classified into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteKey {
    Code,
    Explain,
    Quick,
    Default,
}

impl RouteKey {
    pub fn as_str(&self) -> &'static str {
        match self {
            RouteKey::Code => "code",
            RouteKey::Explain => "explain",
            RouteKey::Quick => "quick",
            RouteKey::Default => "default",
        }
    }
}

/// Classify a prompt into a `RouteKey` using keyword and length heuristics.
/// No LLM call — pure string matching, O(n) in prompt length.
///
/// Priority order (first match wins):
/// 1. Code keywords  → `Code`
/// 2. Explain keyword → `Explain`
/// 3. Short (< 50 chars) or question words → `Quick`
/// 4. Fallthrough → `Default`
pub fn classify_prompt(prompt: &str) -> RouteKey {
    let lower = prompt.to_lowercase();

    // Code: coding task keywords.
    // Short keywords use word-boundary matching to avoid false positives
    // (e.g. "fix" in "prefix", "build" in "building").
    let exact_keywords = [
        "diff", "patch", "review", "commit", "debug", "fix", "build", "lint", "test", "pr",
    ];
    let substring_keywords = ["refactor", "implement", "compile", "deploy"];
    if exact_keywords.iter().any(|kw| {
        lower
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| w == *kw)
    }) || substring_keywords.iter().any(|kw| lower.contains(kw))
    {
        return RouteKey::Code;
    }

    // Explain: explicit explanation request
    if lower.contains("explain") {
        return RouteKey::Explain;
    }

    // Quick: short prompt or common question openers
    let quick_keywords = ["what", "how", "why", "when", "where", "who"];
    if prompt.len() < 50 || quick_keywords.iter().any(|kw| lower.contains(kw)) {
        return RouteKey::Quick;
    }

    RouteKey::Default
}

/// Resolve the model to use for a given prompt.
///
/// Resolution order:
/// 1. Routing config present + rule for the classified key → use that model
/// 2. Routing config present + `default` field set → use that
/// 3. Fall back to `cli_model` (whatever `--model` / settings.json resolved to)
pub fn resolve_model(
    prompt: &str,
    routing: &Option<RoutingConfig>,
    cli_model: &str,
) -> (String, RouteKey) {
    let key = classify_prompt(prompt);
    let model = match routing {
        None => cli_model.to_string(),
        Some(r) => {
            if let Some(m) = r.rules.get(key.as_str()) {
                m.clone()
            } else if !r.default.is_empty() {
                r.default.clone()
            } else {
                cli_model.to_string()
            }
        }
    };
    (model, key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routing(rules: &[(&str, &str)], default: &str) -> Option<RoutingConfig> {
        Some(RoutingConfig {
            rules: rules
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            default: default.to_string(),
        })
    }

    #[test]
    fn classify_short_prompt_is_quick() {
        assert_eq!(classify_prompt("fix it"), RouteKey::Code);
        assert_eq!(classify_prompt("help"), RouteKey::Quick);
    }

    #[test]
    fn classify_explain_keyword() {
        assert_eq!(
            classify_prompt("explain how closures work in Rust"),
            RouteKey::Explain
        );
        assert_eq!(
            classify_prompt("Can you explain this function?"),
            RouteKey::Explain
        );
    }

    #[test]
    fn classify_code_keywords() {
        assert_eq!(
            classify_prompt("review this PR before merge"),
            RouteKey::Code
        );
        assert_eq!(
            classify_prompt("write a commit message for these changes"),
            RouteKey::Code
        );
        assert_eq!(
            classify_prompt("apply the patch to fix the bug"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_default_for_long_non_matching_prompt() {
        let long =
            "Refactor the entire authentication subsystem to use JWT tokens instead of sessions";
        assert_eq!(classify_prompt(long), RouteKey::Code);
    }

    #[test]
    fn resolve_uses_routing_rule_when_present() {
        let r = routing(
            &[("code", "gemma4:26b"), ("quick", "llama3.2:3b")],
            "gemma4:26b",
        );
        let (model, key) = resolve_model("review this diff carefully", &r, "fallback:7b");
        assert_eq!(key, RouteKey::Code);
        assert_eq!(model, "gemma4:26b");
    }

    #[test]
    fn resolve_falls_back_to_cli_model_when_no_rule() {
        // No routing config → always use cli_model
        let (model, _) = resolve_model("review this diff", &None, "mymodel:7b");
        assert_eq!(model, "mymodel:7b");
    }

    #[test]
    fn resolve_falls_back_to_default_rule() {
        // No "explain" rule, but default is set
        let r = routing(&[("code", "gemma4:26b")], "gemma4:26b");
        let (model, key) = resolve_model("explain closures in Rust", &r, "fallback:7b");
        assert_eq!(key, RouteKey::Explain);
        assert_eq!(model, "gemma4:26b"); // default rule
    }

    #[test]
    fn classify_code_takes_priority_over_explain() {
        // "explain" is present but so is a code keyword — Code wins (checked first)
        assert_eq!(
            classify_prompt("explain this diff line by line"),
            RouteKey::Code
        );
    }

    #[test]
    fn route_key_as_str_values() {
        assert_eq!(RouteKey::Code.as_str(), "code");
        assert_eq!(RouteKey::Explain.as_str(), "explain");
        assert_eq!(RouteKey::Quick.as_str(), "quick");
        assert_eq!(RouteKey::Default.as_str(), "default");
    }

    #[test]
    fn classify_question_word_is_quick() {
        assert_eq!(
            classify_prompt("What is the meaning of life and everything else in the universe?"),
            RouteKey::Quick
        );
        // "code" is not a Code keyword — it's a noun here, not a code-review task
        assert_eq!(
            classify_prompt("Why does this code use a HashMap instead of a BTreeMap?"),
            RouteKey::Quick
        );
        // "programs" should NOT trigger Code classification (no longer a false positive)
        assert_eq!(
            classify_prompt("How does the memory allocator work in Rust?"),
            RouteKey::Quick
        );
    }

    #[test]
    fn classify_pr_word_boundary_matches_pull_request_context() {
        // "pr" as a standalone word should classify as Code
        assert_eq!(
            classify_prompt("review this pr before merge"),
            RouteKey::Code
        );
        assert_eq!(classify_prompt("open a pr for this branch"), RouteKey::Code);
    }

    #[test]
    fn classify_pr_substring_does_not_match() {
        // "pr" appearing inside words like "programs" should NOT classify as Code
        let prompt = "How do I write programs in Rust that allocate on the heap efficiently?";
        assert_ne!(
            classify_prompt(prompt),
            RouteKey::Code,
            "substring 'pr' in 'programs' should not classify as Code"
        );
    }

    #[test]
    fn classify_prompt_refactor_is_code() {
        assert_eq!(
            classify_prompt("refactor this module to use iterators"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_prompt_debug_is_code() {
        assert_eq!(
            classify_prompt("debug the authentication bug"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_prompt_fix_is_code() {
        assert_eq!(
            classify_prompt("fix the broken build pipeline"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_prompt_implement_is_code() {
        assert_eq!(
            classify_prompt("implement retry logic for the HTTP client"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_fix_word_boundary() {
        assert_eq!(classify_prompt("fix the bug"), RouteKey::Code);
        assert_ne!(
            classify_prompt("prefix the label with a namespace identifier for clarity"),
            RouteKey::Code
        );
        assert_ne!(
            classify_prompt("the suffix needs adjustment in the configuration template"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_build_word_boundary() {
        assert_eq!(classify_prompt("build the project"), RouteKey::Code);
        assert_ne!(
            classify_prompt("building a house requires careful planning and material selection"),
            RouteKey::Code
        );
    }

    #[test]
    fn classify_test_word_boundary() {
        assert_eq!(classify_prompt("test this function"), RouteKey::Code);
        assert_ne!(
            classify_prompt("the testimonial page needs a redesign for the marketing campaign"),
            RouteKey::Code
        );
    }

    #[test]
    fn resolve_uses_routing_default_when_no_matching_rule() {
        // No "quick" rule, but default is set
        let r = routing(&[("code", "big-model:70b")], "default-model:13b");
        let (model, key) = resolve_model("what is this?", &r, "cli-fallback:7b");
        assert_eq!(key, RouteKey::Quick);
        assert_eq!(model, "default-model:13b"); // falls to routing default
    }

    #[test]
    fn resolve_falls_back_to_cli_when_routing_default_empty() {
        // Routing config exists but default is empty and no matching rule
        let r = routing(&[("code", "big-model:70b")], ""); // empty default
        let (model, _) = resolve_model("what is this?", &r, "cli-fallback:7b");
        assert_eq!(model, "cli-fallback:7b"); // empty default → use cli_model
    }
}
