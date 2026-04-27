/// Returns model-specific system prompt guidance, or None for unknown models.
/// Hints are short behavioral nudges that improve output quality for each model family.
pub fn model_hint(model_name: &str) -> Option<&'static str> {
    let base = model_name.split(':').next().unwrap_or(model_name);
    match base {
        "deepseek-coder" | "deepseek-coder-v2" | "deepseek-v2.5" | "deepseek-r1" => Some(
            "Use <think> tags for reasoning before answering. \
             Show complete code blocks rather than partial snippets.",
        ),
        "llama3" | "llama3.1" | "llama3.2" | "llama3.3" | "llama4" => Some(
            "Be concise and direct. Prefer numbered steps for multi-step tasks. \
             Avoid over-explaining — show code, not prose.",
        ),
        "gemma" | "gemma2" | "gemma3" | "gemma4" => Some(
            "You handle long context well. When editing files, show only the changed \
             sections with enough surrounding context. Be precise with tool arguments.",
        ),
        "qwen" | "qwen2" | "qwen2.5" | "qwen3" => Some(
            "Use structured thinking before tool calls. For complex tasks, \
             plan your approach before executing. Be explicit about file paths.",
        ),
        "codestral" | "mistral" => Some(
            "Focus on clean, idiomatic code. Prefer small, targeted edits over \
             large rewrites. Use grep/glob to understand context before editing.",
        ),
        "phi3" | "phi4" => Some(
            "Keep responses focused. For code tasks, prioritize correctness over \
             completeness — it's better to make one correct edit than attempt many.",
        ),
        "command-r" | "command-r-plus" => Some(
            "Ground answers in the provided context. When using tools, explain \
             your reasoning briefly before each call.",
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_with_tag_returns_hint() {
        let hint = model_hint("llama3:70b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("concise"));
    }

    #[test]
    fn deepseek_coder_v2_returns_hint() {
        let hint = model_hint("deepseek-coder-v2:16b-q4_0");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("think"));
    }

    #[test]
    fn gemma4_returns_hint() {
        let hint = model_hint("gemma4:26b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("context"));
    }

    #[test]
    fn qwen3_returns_hint() {
        let hint = model_hint("qwen3:32b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("structured"));
    }

    #[test]
    fn codestral_returns_hint() {
        let hint = model_hint("codestral:22b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("idiomatic"));
    }

    #[test]
    fn phi4_returns_hint() {
        let hint = model_hint("phi4:14b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("focused"));
    }

    #[test]
    fn unknown_model_returns_none() {
        assert!(model_hint("unknown-model").is_none());
        assert!(model_hint("some-random:7b").is_none());
    }

    #[test]
    fn bare_name_no_tag() {
        assert!(model_hint("llama3").is_some());
        assert!(model_hint("gemma4").is_some());
        assert!(model_hint("qwen2.5").is_some());
    }

    #[test]
    fn deepseek_r1_returns_hint() {
        let hint = model_hint("deepseek-r1:70b");
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("think"));
    }
}
