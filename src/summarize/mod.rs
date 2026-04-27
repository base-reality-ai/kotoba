//! File and text summarization utilities.
//!
//! Provides the `dm summarize` CLI functionality to read target files
//! and stream concise summaries back to the user.

use crate::ollama::client::OllamaClient;
use crate::ollama::types::StreamEvent;
use anyhow::Result;
use futures_util::StreamExt;

const MAX_CONTENT_CHARS: usize = 32_000;
const MAX_FILE_BYTES: usize = 500 * 1024; // 500 KB

pub async fn run_summarize(
    target: &str,
    length: usize,
    style: &str,
    client: &OllamaClient,
) -> Result<()> {
    let raw_content = load_content(target).await?;
    let content = truncate_content(&raw_content);
    let prompt = build_prompt(&content, length, style);

    let messages = build_messages(&prompt);

    let mut stream = client.chat_stream_with_tools(&messages, &[]).await?;
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::Token(tok) => print!("{}", tok),
            StreamEvent::Done { .. } => break,
            StreamEvent::Error(e) => anyhow::bail!("Stream error: {}", e),
            StreamEvent::Thinking(_) | StreamEvent::ToolCalls(_) => {}
        }
    }
    println!();

    Ok(())
}

async fn load_content(target: &str) -> Result<String> {
    if target == "-" || target.is_empty() {
        // Read from stdin
        use std::io::Read as _;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        Ok(buf)
    } else if target.starts_with("http://") || target.starts_with("https://") {
        // Fetch URL
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        let response = http_client.get(target).send().await?;
        let body = response.text().await?;
        // Convert HTML to plain text
        let text = html2text::from_read(body.as_bytes(), 100);
        Ok(text)
    } else {
        // Read file
        let meta = std::fs::metadata(target)
            .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", target, e))?;
        if meta.len() as usize > MAX_FILE_BYTES {
            anyhow::bail!(
                "File '{}' is too large ({} bytes, max 500 KB)",
                target,
                meta.len()
            );
        }
        let content = std::fs::read_to_string(target)
            .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", target, e))?;
        Ok(content)
    }
}

fn truncate_content(content: &str) -> String {
    if content.len() > MAX_CONTENT_CHARS {
        let cut = crate::util::safe_truncate(content, MAX_CONTENT_CHARS);
        format!("{}\n[content truncated]", cut)
    } else {
        content.to_string()
    }
}

fn build_prompt(content: &str, length: usize, style: &str) -> String {
    format!(
        "Summarize the following in {style} format, approximately {length} words.\n\n---\n{content}\n---"
    )
}

fn build_messages(prompt: &str) -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "role": "system",
            "content": "You are a concise technical summarizer. Output only the summary — \
                         no preamble, no sign-off, no meta-commentary. Preserve code identifiers, \
                         file paths, and technical terms exactly as written."
        }),
        serde_json::json!({
            "role": "user",
            "content": prompt
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_truncates_long_input() {
        let long_input = "x".repeat(100_000);
        let truncated = truncate_content(&long_input);
        // Should be at most MAX_CONTENT_CHARS + len("[content truncated]") + newline
        assert!(
            truncated.len() <= MAX_CONTENT_CHARS + 20,
            "truncated content should be at most ~32020 chars, got {}",
            truncated.len()
        );
        assert!(
            truncated.contains("[content truncated]"),
            "should include truncation notice"
        );
    }

    #[test]
    fn summarize_style_appears_in_prompt() {
        let prompt = build_prompt("some text here", 150, "tldr");
        assert!(
            prompt.contains("tldr"),
            "prompt should contain the style 'tldr'"
        );
    }

    #[test]
    fn summarize_length_appears_in_prompt() {
        let prompt = build_prompt("some text here", 200, "bullets");
        assert!(
            prompt.contains("200"),
            "prompt should contain the target word count"
        );
    }

    #[test]
    fn truncate_content_short_passthrough() {
        let short = "hello world";
        assert_eq!(truncate_content(short), short);
    }

    #[test]
    fn truncate_content_multibyte_safe() {
        // Build content that exceeds MAX_CONTENT_CHARS with multibyte chars
        // so a naïve slice at 32000 would panic
        let line = "🦀".repeat(10); // 40 bytes per line
        let content: String = vec![line.as_str(); 1000].join("\n");
        // Should not panic
        let result = truncate_content(&content);
        assert!(result.contains("[content truncated]"));
    }

    #[test]
    fn build_prompt_content_appears() {
        let prompt = build_prompt("my important content", 100, "markdown");
        assert!(
            prompt.contains("my important content"),
            "content missing from prompt: {prompt}"
        );
    }

    #[test]
    fn truncate_content_exactly_at_limit_passthrough() {
        let at_limit = "x".repeat(MAX_CONTENT_CHARS);
        let result = truncate_content(&at_limit);
        assert_eq!(
            result, at_limit,
            "exactly MAX_CONTENT_CHARS should pass through unchanged"
        );
        assert!(!result.contains("[content truncated]"));
    }

    #[test]
    fn build_prompt_contains_delimiters() {
        let prompt = build_prompt("content", 50, "tldr");
        assert!(
            prompt.contains("---"),
            "prompt should contain --- delimiters"
        );
    }

    #[test]
    fn truncate_content_one_over_limit_truncates() {
        let content = "x".repeat(MAX_CONTENT_CHARS + 1);
        let result = truncate_content(&content);
        assert!(result.contains("[content truncated]"));
        assert!(result.len() <= MAX_CONTENT_CHARS + 20);
    }

    #[test]
    fn build_messages_includes_system_message() {
        let msgs = build_messages("summarize this");
        assert_eq!(msgs.len(), 2, "should have system + user messages");
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[0]["content"].as_str().unwrap().contains("concise"));
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "summarize this");
    }
}
