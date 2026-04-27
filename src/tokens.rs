//! Token-budget estimation helpers for conversation compaction.
//!
//! Uses dm's coarse 4-chars-per-token heuristic plus explicit overhead for
//! message framing, tool calls, and images. The estimates decide when long
//! sessions should compact before exceeding model context limits.

/// Sum the estimated token count across a slice of Ollama message JSON objects.
/// Counts `content`, `tool_calls` (function name + arguments), `images`, and
/// per-message overhead (4 tokens each for JSON/role framing).
pub fn conversation_tokens(messages: &[serde_json::Value]) -> usize {
    let mut content_chars: usize = 0;
    let mut image_tokens: usize = 0;

    for m in messages {
        if let Some(s) = m["content"].as_str() {
            content_chars += s.len();
        }
        if let Some(tcs) = m["tool_calls"].as_array() {
            for tc in tcs {
                if let Some(name) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                {
                    content_chars += name.len();
                }
                if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")) {
                    // Arguments arrive in Object form via the conversation-message
                    // builder (see `ollama::types::deserialize_args` — Ollama's
                    // String form is always normalised to Object before it lands
                    // here, so the prior `.as_str()`-only path counted zero bytes
                    // for every tool call). Serialize the value to JSON and count
                    // its bytes so both forms are represented faithfully.
                    if let Some(s) = args.as_str() {
                        content_chars += s.len();
                    } else {
                        content_chars += serde_json::to_string(args).map(|s| s.len()).unwrap_or(0);
                    }
                }
            }
        }
        if let Some(imgs) = m["images"].as_array() {
            image_tokens += imgs.len() * 256;
        }
    }

    let overhead = messages.len() * 4;
    content_chars / 4 + image_tokens + overhead
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }

    #[test]
    fn estimate_empty_string_returns_zero() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_short_string() {
        // "hello" = 5 chars → 5/4 = 1
        assert_eq!(estimate_tokens("hello"), 1);
    }

    #[test]
    fn estimate_longer_string() {
        // 40 chars → 10 tokens
        let s = "a".repeat(40);
        assert_eq!(estimate_tokens(&s), 10);
    }

    #[test]
    fn conversation_tokens_empty() {
        assert_eq!(conversation_tokens(&[]), 0);
    }

    #[test]
    fn conversation_tokens_sums_content() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": "a".repeat(40)}),
            serde_json::json!({"role": "assistant", "content": "b".repeat(40)}),
        ];
        // Each message → 10 content tokens, plus 2*4=8 overhead → 28
        assert_eq!(conversation_tokens(&msgs), 28);
    }

    #[test]
    fn conversation_tokens_counts_images() {
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": "",
            "images": ["data1", "data2"]
        })];
        // empty content = 0 + 2 * 256 = 512 + 4 overhead = 516
        assert_eq!(conversation_tokens(&msgs), 516);
    }

    #[test]
    fn conversation_tokens_counts_tool_calls() {
        let msgs = vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"/tmp/test.rs\"}"
                }
            }]
        })];
        // "read_file" = 9 chars, arguments = 24 chars → 33 chars → 33/4 = 8, + 4 overhead = 12
        let result = conversation_tokens(&msgs);
        assert_eq!(result, 12);
    }

    #[test]
    fn conversation_tokens_counts_tool_call_arguments_in_object_form() {
        // Per `ollama::types::deserialize_args`, tool-call arguments are
        // always normalised to Object form before reaching the conversation
        // message builder (`conversation.rs:498` embeds the Value directly).
        // Production messages therefore carry arguments as `{"k":"v"}`, NOT
        // as a string. The previous `.as_str()`-only path counted ZERO
        // bytes for every real tool call — silently underestimating tokens
        // and causing compaction to fire too late.
        let msgs = vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/tmp/test.rs"}
                }
            }]
        })];
        // Serialised arguments: `{"path":"/tmp/test.rs"}` = 23 chars.
        // "read_file" = 9 chars. Total = 32 → 32/4 = 8, + 4 overhead = 12.
        let result = conversation_tokens(&msgs);
        assert!(
            result >= 10,
            "Object-form args must contribute non-trivial bytes, got {} (would be 5 if args were dropped)",
            result
        );
        assert_eq!(result, 12);
    }

    #[test]
    fn conversation_tokens_includes_overhead() {
        let msgs = vec![
            serde_json::json!({"role": "user", "content": ""}),
            serde_json::json!({"role": "assistant", "content": ""}),
            serde_json::json!({"role": "user", "content": ""}),
        ];
        // No content, just 3 * 4 = 12 overhead
        assert_eq!(conversation_tokens(&msgs), 12);
    }

    #[test]
    fn conversation_tokens_mixed_content_tools_images() {
        let msgs = vec![
            serde_json::json!({
                "role": "user",
                "content": "a".repeat(40),
                "images": ["img1"]
            }),
            serde_json::json!({
                "role": "assistant",
                "content": "b".repeat(20),
                "tool_calls": [{
                    "function": {
                        "name": "bash",
                        "arguments": "{\"cmd\":\"ls\"}"
                    }
                }]
            }),
        ];
        // msg1: 40 content chars + 256 image tokens
        // msg2: 20 content chars + "bash"(4) + "{\"cmd\":\"ls\"}"(12) = 36 tool chars
        // total chars = 40 + 20 + 4 + 12 = 76 → 76/4 = 19
        // images = 256, overhead = 2*4 = 8
        // total = 19 + 256 + 8 = 283
        assert_eq!(conversation_tokens(&msgs), 283);
    }

    #[test]
    fn threshold_80_pct() {
        let limit = 1000usize;
        let used = 820usize;
        let pct = used * 100 / limit;
        assert!(pct >= 80);
        assert!(pct < 95);
    }

    #[test]
    fn threshold_95_pct() {
        let limit = 1000usize;
        let used = 960usize;
        let pct = used * 100 / limit;
        assert!(pct >= 95);
    }
}
