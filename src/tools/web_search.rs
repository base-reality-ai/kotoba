use super::{Tool, ToolResult};
use crate::util::safe_truncate;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Duration;

pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }

    fn description(&self) -> &'static str {
        "Search the web and return a list of results with titles, URLs, and snippets. \
         Use this to find current information, documentation, or research topics. \
         For fetching a specific URL's content, use web_fetch instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10)."
                }
            },
            "required": ["query"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use web_search for finding information online. Returns titles, URLs, and snippets. \
              Follow up with web_fetch to read specific pages.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let query = match args["query"].as_str() {
            Some(q) if !q.trim().is_empty() => q.to_string(),
            _ => {
                return Ok(ToolResult {
                    content: "'query' parameter is required and cannot be empty. Try: include 'query' (search term)."
                        .to_string(),
                    is_error: true,
                })
            }
        };
        let max_results = args["max_results"].as_u64().unwrap_or(10) as usize;

        let encoded = urlencoding::encode(&query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
            .build()?;

        let response = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: classify_search_error(&e),
                    is_error: true,
                })
            }
        };

        if !response.status().is_success() {
            return Ok(ToolResult {
                content: classify_search_http_error(response.status()),
                is_error: true,
            });
        }

        let html = response.text().await.unwrap_or_default();

        let results = parse_ddg_html(&html, max_results);

        if results.is_empty() {
            return Ok(ToolResult {
                content: format!(
                    "No results found for: \"{}\". Try: rephrase the query, broaden it, or check spelling.",
                    query
                ),
                is_error: false,
            });
        }

        let mut parts = vec![format!("Search results for: \"{}\"\n", query)];
        for (i, (title, result_url, snippet)) in results.iter().enumerate() {
            parts.push(format!(
                "{}. {}\n   URL: {}\n   {}",
                i + 1,
                title,
                result_url,
                snippet
            ));
        }

        Ok(ToolResult {
            content: parts.join("\n\n"),
            is_error: false,
        })
    }
}

/// Parse DDG HTML results. Returns vec of (title, url, snippet).
fn parse_ddg_html(html: &str, max: usize) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut pos = 0;

    while results.len() < max {
        // Find next result__a link
        let Some(marker_offset) = html[pos..].find("class=\"result__a\"") else {
            break;
        };
        let marker_abs = pos + marker_offset;

        // Walk back to find the opening `<a` tag
        let tag_start = html[..marker_abs].rfind('<').unwrap_or(marker_abs);

        // Extract href from the tag
        let tag_slice = &html[tag_start..marker_abs + 20];
        let url = if let Some(h) = tag_slice.find("href=\"") {
            let url_start = tag_start + h + 6;
            let url_end = html[url_start..]
                .find('"')
                .map_or(url_start, |e| url_start + e);
            let raw = &html[url_start..url_end];
            decode_ddg_url(raw)
        } else {
            String::new()
        };

        // Find title text (between `>` and `</a>`)
        let Some(gt_offset) = html[marker_abs..].find('>') else {
            pos = marker_abs + 1;
            continue;
        };
        let title_start = marker_abs + gt_offset + 1;
        let title_end = html[title_start..]
            .find("</a>")
            .map_or(title_start, |e| title_start + e);
        let title = strip_tags(&html[title_start..title_end]).trim().to_string();

        // Find snippet after this result
        let snippet = if let Some(snip_offset) = html[title_end..].find("result__snippet") {
            let snip_abs = title_end + snip_offset;
            if let Some(open_offset) = html[snip_abs..].find('>') {
                let text_start = snip_abs + open_offset + 1;
                let raw_end = html[text_start..]
                    .find("</a>")
                    .map_or(html.len(), |e| text_start + e);
                let raw_slice = &html[text_start..raw_end];
                strip_tags(safe_truncate(raw_slice, 300)).trim().to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !title.is_empty() && !url.is_empty() {
            results.push((title, url, snippet));
        }

        pos = title_end + 1;
    }

    results
}

/// Decode a DDG redirect URL (extracts the real URL from `uddg=` param).
fn decode_ddg_url(raw: &str) -> String {
    if let Some(uddg) = raw.split("uddg=").nth(1) {
        let encoded_part = uddg.split('&').next().unwrap_or(uddg);
        urlencoding::decode(encoded_part)
            .unwrap_or_default()
            .into_owned()
    } else {
        raw.to_string()
    }
}

fn classify_search_error(error: &reqwest::Error) -> String {
    let msg = error.to_string().to_lowercase();
    if error.is_timeout() || msg.contains("timed out") {
        "Search timed out after 15s. Try: retry; DuckDuckGo may be temporarily unreachable."
            .to_string()
    } else if msg.contains("dns") || msg.contains("resolve") {
        "Could not resolve search provider. Try: check network connection and DNS.".to_string()
    } else if msg.contains("connection refused") {
        "Connection refused to search provider. Try: retry; the provider may be temporarily unreachable.".to_string()
    } else {
        format!("Search request failed: {}. Try: retry the search.", error)
    }
}

fn classify_search_http_error(status: reqwest::StatusCode) -> String {
    match status.as_u16() {
        403 => "Search blocked: HTTP 403. Try: wait and retry; DuckDuckGo may be rate-limiting.".to_string(),
        429 => "Rate limited by search provider. Try: wait a few minutes and retry.".to_string(),
        code if code >= 500 => format!("Search provider error: HTTP {}. Try: wait and retry; the provider may be experiencing an outage.", code),
        code => format!("Search failed: HTTP {}. Try: retry the search, or rephrase the query.", code),
    }
}

/// Strip HTML tags and decode common entities.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── strip_tags ────────────────────────────────────────────────────────────

    #[test]
    fn strip_tags_plain_text_unchanged() {
        assert_eq!(strip_tags("hello world"), "hello world");
    }

    #[test]
    fn strip_tags_removes_html_tags() {
        assert_eq!(strip_tags("<b>bold</b>"), "bold");
    }

    #[test]
    fn strip_tags_decodes_entities() {
        assert_eq!(strip_tags("a &amp; b"), "a & b");
        assert_eq!(strip_tags("&lt;tag&gt;"), "<tag>");
        assert_eq!(strip_tags("&quot;hi&quot;"), "\"hi\"");
        assert_eq!(strip_tags("&nbsp;"), " ");
        assert_eq!(strip_tags("&#x27;"), "'");
    }

    #[test]
    fn strip_tags_mixed() {
        assert_eq!(
            strip_tags("<a href=\"x\">Link &amp; more</a>"),
            "Link & more"
        );
    }

    // ── decode_ddg_url ────────────────────────────────────────────────────────

    #[test]
    fn decode_ddg_url_plain_url_passthrough() {
        assert_eq!(decode_ddg_url("https://example.com"), "https://example.com");
    }

    #[test]
    fn decode_ddg_url_extracts_uddg_param() {
        let raw = "/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc";
        assert_eq!(decode_ddg_url(raw), "https://example.com/page");
    }

    #[test]
    fn decode_ddg_url_stops_at_ampersand() {
        let raw = "/l/?uddg=https%3A%2F%2Fa.com&other=x";
        assert_eq!(decode_ddg_url(raw), "https://a.com");
    }

    // ── Tool trait basics ────────────────────────────────────────────────────

    #[test]
    fn name_and_schema() {
        let t = WebSearchTool;
        assert_eq!(t.name(), "web_search");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("query")));
        assert!(p["properties"]["max_results"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(WebSearchTool.is_read_only());
    }

    #[test]
    fn has_system_prompt_hint() {
        let hint = WebSearchTool.system_prompt_hint();
        assert!(hint.is_some());
        assert!(!hint.unwrap().is_empty());
    }

    // ── Missing query parameter ─────────────────────────────────────────────

    #[tokio::test]
    async fn missing_query_returns_error() {
        let result = WebSearchTool.call(json!({})).await.unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("query"),
            "should mention query: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn missing_query_with_max_results_returns_error() {
        let result = WebSearchTool.call(json!({"max_results": 5})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("query"));
    }

    // ── classify_search_http_error ──────────────────────────────────────────

    #[test]
    fn classify_search_http_403() {
        let msg = classify_search_http_error(reqwest::StatusCode::FORBIDDEN);
        assert!(msg.contains("blocked"), "msg: {}", msg);
        assert!(msg.contains("403"));
    }

    #[test]
    fn classify_search_http_429() {
        let msg = classify_search_http_error(reqwest::StatusCode::TOO_MANY_REQUESTS);
        assert!(msg.contains("Rate limited"), "msg: {}", msg);
    }

    #[test]
    fn classify_search_http_500() {
        let msg = classify_search_http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(msg.contains("provider error"), "msg: {}", msg);
        assert!(msg.contains("500"));
    }

    #[test]
    fn classify_search_http_other() {
        let msg = classify_search_http_error(reqwest::StatusCode::IM_A_TEAPOT);
        assert!(msg.contains("418"), "msg: {}", msg);
    }

    // ── parse_ddg_html ────────────────────────────────────────────────────────

    fn fake_ddg_result(title: &str, url: &str, snippet: &str) -> String {
        // The parser reads href from html[tag_start..marker_abs+20], so href must
        // precede class in the tag (matching real DDG HTML structure).
        format!(
            r#"<a href="/l/?uddg={encoded_url}" class="result__a">{title}</a> <a class="result__snippet">{snippet}</a>"#,
            encoded_url = urlencoding::encode(url),
            title = title,
            snippet = snippet,
        )
    }

    #[test]
    fn parse_ddg_html_empty_returns_empty() {
        assert!(parse_ddg_html("", 10).is_empty());
    }

    #[tokio::test]
    async fn web_search_rejects_empty_query() {
        let result = WebSearchTool.call(json!({"query": ""})).await.unwrap();
        assert!(result.is_error, "empty query should be an error");
        assert!(
            result.content.contains("cannot be empty"),
            "msg: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn web_search_rejects_whitespace_query() {
        let result = WebSearchTool.call(json!({"query": "   "})).await.unwrap();
        assert!(result.is_error, "whitespace query should be an error");
    }

    #[tokio::test]
    async fn web_search_rejects_missing_query() {
        let result = WebSearchTool.call(json!({})).await.unwrap();
        assert!(result.is_error, "missing query should be an error");
    }

    #[test]
    fn parse_ddg_html_no_markers_returns_empty() {
        assert!(parse_ddg_html("<html><body>nothing here</body></html>", 10).is_empty());
    }

    #[test]
    fn parse_ddg_html_extracts_result() {
        let html = fake_ddg_result("Rust Lang", "https://rust-lang.org", "Systems language");
        let results = parse_ddg_html(&html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Rust Lang");
        assert_eq!(results[0].1, "https://rust-lang.org");
        assert!(results[0].2.contains("Systems language"));
    }

    #[test]
    fn parse_ddg_html_max_results_respected() {
        let mut html = String::new();
        for i in 0..5 {
            html.push_str(&fake_ddg_result(
                &format!("Result {}", i),
                &format!("https://example.com/{}", i),
                "snippet",
            ));
        }
        let results = parse_ddg_html(&html, 3);
        assert_eq!(results.len(), 3);
    }
}
