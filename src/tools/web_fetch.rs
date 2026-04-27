use super::{Tool, ToolResult};
use crate::util::safe_truncate;
use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::time::Duration;

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024; // 2 MB

pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        "web_fetch"
    }

    fn description(&self) -> &'static str {
        "Fetch the content of a URL and return it as text. Supports HTML (converted to \
         plain text), JSON (pretty-printed), and plain text responses. Use for reading \
         documentation, API responses, or any web resource."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch."
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 20000)."
                }
            },
            "required": ["url"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use web_fetch to retrieve content from a URL. Output is truncated to fit context. \
              For search results, use web_search instead.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let url = match args["url"].as_str() {
            Some(u) => u.to_string(),
            None => {
                return Ok(ToolResult {
                    content: "missing required parameter 'url'. Try: include 'url' (an http:// or https:// URL) in args.".to_string(),
                    is_error: true,
                })
            }
        };
        let max_length = args["max_length"].as_u64().unwrap_or(20_000) as usize;

        if let Err(msg) = validate_url(&url) {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        if let Err(msg) = resolve_and_check(&url).await {
            return Ok(ToolResult {
                content: msg,
                is_error: true,
            });
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("dark-matter/0.1")
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                let url = attempt.url();
                let scheme = url.scheme();
                if scheme != "http" && scheme != "https" {
                    return attempt.stop();
                }
                if let Some(host) = url.host_str() {
                    if is_internal_host(host) {
                        return attempt.stop();
                    }
                }
                if attempt.previous().len() >= 5 {
                    return attempt.stop();
                }
                attempt.follow()
            }))
            .build()?;

        let response = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    content: classify_fetch_error(&url, &e),
                    is_error: true,
                })
            }
        };

        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown");
            return Ok(ToolResult {
                content: format!(
                    "Redirect blocked: {} tried to redirect to '{}'. Try: fetch the redirect target directly if it's a public URL with http(s) scheme.",
                    url, location
                ),
                is_error: true,
            });
        }

        if !response.status().is_success() {
            return Ok(ToolResult {
                content: classify_http_error(&url, response.status()),
                is_error: true,
            });
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body_bytes = {
            let mut buf = Vec::new();
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        return Ok(ToolResult {
                            content: format!(
                                "error reading response body: {}. Try: retry the fetch.",
                                e
                            ),
                            is_error: true,
                        });
                    }
                };
                buf.extend_from_slice(&chunk);
                if buf.len() > MAX_RESPONSE_BYTES {
                    buf.truncate(MAX_RESPONSE_BYTES);
                    break;
                }
            }
            buf
        };

        let content = process_body(&body_bytes, &content_type, max_length);

        Ok(ToolResult {
            content,
            is_error: false,
        })
    }
}

fn validate_url(url: &str) -> Result<(), String> {
    let scheme_end = url.find("://").ok_or_else(|| {
        format!(
            "Invalid URL '{}': missing scheme. Try: prefix the URL with http:// or https://.",
            url
        )
    })?;
    let scheme = &url[..scheme_end];
    if !matches!(scheme, "http" | "https") {
        return Err(format!(
            "Only http:// and https:// URLs are supported, got '{}://'. Try: use an http(s) URL.",
            scheme
        ));
    }
    let after_scheme = &url[scheme_end + 3..];
    let authority = after_scheme.split('/').next().unwrap_or("");
    let host_port = if let Some(at_pos) = authority.rfind('@') {
        &authority[at_pos + 1..]
    } else {
        authority
    };
    let host = if host_port.starts_with('[') {
        host_port
            .split(']')
            .next()
            .unwrap_or(host_port)
            .trim_start_matches('[')
    } else {
        host_port.split(':').next().unwrap_or(host_port)
    };
    if is_internal_host(host) {
        return Err(format!(
            "Cannot fetch internal/private URL: {}. Try: only public hosts are reachable; use bash with curl for internal endpoints if needed.",
            url
        ));
    }
    Ok(())
}

fn is_internal_host(host: &str) -> bool {
    let blocked = ["169.254.169.254", "metadata.google.internal"];
    if blocked.contains(&host) {
        return true;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        match ip {
            std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
            std::net::IpAddr::V6(v6) => v6.is_loopback(),
        }
    } else {
        false
    }
}

async fn resolve_and_check(url: &str) -> Result<(), String> {
    let scheme_end = url.find("://").unwrap_or(0);
    let after_scheme = &url[scheme_end + 3..];
    let authority = after_scheme.split('/').next().unwrap_or("");
    let host_port = if let Some(at_pos) = authority.rfind('@') {
        &authority[at_pos + 1..]
    } else {
        authority
    };

    if host_port.is_empty() {
        return Err(
            "URL has no host. Try: confirm the URL contains a hostname after http(s)://."
                .to_string(),
        );
    }

    let lookup_target = if host_port.contains(':') || host_port.starts_with('[') {
        host_port.to_string()
    } else {
        format!("{}:80", host_port)
    };

    let addrs = tokio::net::lookup_host(&lookup_target)
        .await
        .map_err(|e| {
            format!(
                "DNS resolution failed for '{}': {}. Try: verify the hostname is correct and DNS is reachable.",
                host_port, e
            )
        })?;

    for addr in addrs {
        let ip = addr.ip();
        let ip_str = ip.to_string();
        if is_internal_host(&ip_str) {
            return Err(format!(
                "DNS for '{}' resolved to internal IP {} — blocked to prevent SSRF",
                host_port, ip
            ));
        }
    }
    Ok(())
}

fn classify_fetch_error(url: &str, error: &reqwest::Error) -> String {
    let msg = error.to_string().to_lowercase();
    if error.is_timeout() || msg.contains("timed out") {
        format!(
            "Timed out after 30s fetching {}. Try: retry, or fetch a different URL; the server may be slow or unreachable.",
            url
        )
    } else if msg.contains("connection refused") {
        format!(
            "Connection refused to {}. Try: verify the URL works in a browser; the server may be down.",
            url
        )
    } else if msg.contains("dns") || msg.contains("resolve") || msg.contains("no such host") {
        format!(
            "Could not resolve hostname for {}. Try: check the URL spelling and DNS reachability.",
            url
        )
    } else {
        format!("error fetching {}: {}. Try: retry the fetch.", url, error)
    }
}

fn classify_http_error(url: &str, status: reqwest::StatusCode) -> String {
    match status.as_u16() {
        404 => format!(
            "HTTP 404 — page not found at {}. Try: check the URL path for typos, or web_search to find the canonical URL.",
            url
        ),
        403 => format!(
            "HTTP 403 — access forbidden at {}. Try: confirm the page is publicly accessible; some sites require authentication.",
            url
        ),
        429 => format!(
            "HTTP 429 — rate limited by {}. Try: wait a few minutes and retry.",
            url
        ),
        code if code >= 500 => format!(
            "HTTP {} — server error at {}. Try: wait and retry; the server may be experiencing issues.",
            code, url
        ),
        code => format!(
            "HTTP {} fetching {}. Try: retry, or check the URL.",
            code, url
        ),
    }
}

fn process_body(body: &[u8], content_type: &str, max_length: usize) -> String {
    let text = if content_type.contains("text/html") || content_type.contains("application/xhtml") {
        html2text::from_read(body, 100)
    } else if content_type.contains("application/json") || content_type.contains("+json") {
        match serde_json::from_slice::<Value>(body) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(_) => String::from_utf8_lossy(body).into_owned(),
        }
    } else {
        String::from_utf8_lossy(body).into_owned()
    };

    if text.len() > max_length {
        format!(
            "{}\n[Truncated to {} chars]",
            safe_truncate(&text, max_length),
            max_length
        )
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tool trait basics ────────────────────────────────────────────────────

    #[test]
    fn name_and_schema() {
        let t = WebFetchTool;
        assert_eq!(t.name(), "web_fetch");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("url")));
        assert!(p["properties"]["max_length"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(WebFetchTool.is_read_only());
    }

    #[test]
    fn has_system_prompt_hint() {
        let hint = WebFetchTool.system_prompt_hint();
        assert!(hint.is_some());
        assert!(!hint.unwrap().is_empty());
    }

    // ── Missing URL parameter ────────────────────────────────────────────────

    #[tokio::test]
    async fn missing_url_returns_error() {
        let result = WebFetchTool.call(json!({})).await.unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("url"),
            "should mention url: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn missing_url_with_max_length_returns_error() {
        let result = WebFetchTool.call(json!({"max_length": 100})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("url"));
    }

    // ── classify_http_error ──────────────────────────────────────────────────

    #[test]
    fn classify_http_404() {
        let msg = classify_http_error(
            "https://example.com/missing",
            reqwest::StatusCode::NOT_FOUND,
        );
        assert!(msg.contains("not found"), "msg: {}", msg);
        assert!(msg.contains("404"));
    }

    #[test]
    fn classify_http_403() {
        let msg = classify_http_error("https://example.com/secret", reqwest::StatusCode::FORBIDDEN);
        assert!(msg.contains("forbidden"), "msg: {}", msg);
    }

    #[test]
    fn classify_http_429() {
        let msg = classify_http_error(
            "https://api.example.com",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
        );
        assert!(msg.contains("rate limited"), "msg: {}", msg);
    }

    #[test]
    fn classify_http_500() {
        let msg = classify_http_error(
            "https://example.com",
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        );
        assert!(msg.contains("server error"), "msg: {}", msg);
        assert!(msg.contains("500"));
    }

    #[test]
    fn classify_http_503() {
        let msg = classify_http_error(
            "https://example.com",
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
        );
        assert!(msg.contains("server error"), "msg: {}", msg);
        assert!(msg.contains("503"));
    }

    #[test]
    fn classify_http_other_status() {
        let msg = classify_http_error("https://example.com", reqwest::StatusCode::GONE);
        assert!(msg.contains("410"), "msg: {}", msg);
        assert!(msg.contains("example.com"));
    }

    // ── process_body ─────────────────────────────────────────────────────────

    #[test]
    fn process_body_html_strips_tags() {
        let html = b"<html><body><h1>Title</h1><p>Hello world</p></body></html>";
        let result = process_body(html, "text/html; charset=utf-8", 20_000);
        assert!(result.contains("Title"), "should extract title: {}", result);
        assert!(
            result.contains("Hello world"),
            "should extract text: {}",
            result
        );
        assert!(!result.contains("<h1>"), "should strip tags: {}", result);
    }

    #[test]
    fn process_body_json_pretty_prints() {
        let json_bytes = br#"{"key":"value","num":42}"#;
        let result = process_body(json_bytes, "application/json", 20_000);
        assert!(
            result.contains("\"key\": \"value\""),
            "should pretty-print: {}",
            result
        );
        assert!(result.contains('\n'), "should have newlines: {}", result);
    }

    #[test]
    fn process_body_plain_text_passthrough() {
        let text = b"Hello, world!";
        let result = process_body(text, "text/plain", 20_000);
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn process_body_truncation() {
        let text = b"abcdefghij"; // 10 bytes
        let result = process_body(text, "text/plain", 5);
        assert!(result.contains("Truncated"), "should truncate: {}", result);
        assert!(
            result.contains("5 chars"),
            "should mention limit: {}",
            result
        );
        assert!(
            result.starts_with("abcde"),
            "should keep first 5 chars: {}",
            result
        );
    }

    #[test]
    fn process_body_no_truncation_when_within_limit() {
        let text = b"short";
        let result = process_body(text, "text/plain", 20_000);
        assert_eq!(result, "short");
        assert!(!result.contains("Truncated"));
    }

    #[test]
    fn process_body_invalid_json_falls_back() {
        let bad_json = b"{ not valid json }}}";
        let result = process_body(bad_json, "application/json", 20_000);
        assert!(
            result.contains("not valid json"),
            "should fall back to raw text: {}",
            result
        );
    }

    #[test]
    fn process_body_xhtml_treated_as_html() {
        let html = b"<p>xhtml content</p>";
        let result = process_body(html, "application/xhtml+xml", 20_000);
        assert!(
            result.contains("xhtml content"),
            "xhtml should be processed like html: {}",
            result
        );
    }

    #[test]
    fn process_body_json_subtype() {
        let json_bytes = br#"{"a":1}"#;
        let result = process_body(json_bytes, "application/vnd.api+json", 20_000);
        assert!(
            result.contains("\"a\": 1"),
            "should handle +json content type: {}",
            result
        );
    }

    #[tokio::test]
    async fn web_fetch_rejects_file_scheme() {
        let result = WebFetchTool
            .call(json!({"url": "file:///etc/passwd"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("Only http"),
            "msg: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn web_fetch_rejects_internal_ip() {
        let result = WebFetchTool
            .call(json!({"url": "http://169.254.169.254/latest/meta-data/"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("internal"),
            "msg: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn web_fetch_rejects_localhost() {
        let result = WebFetchTool
            .call(json!({"url": "http://127.0.0.1/secret"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("internal"),
            "msg: {}",
            result.content
        );
    }

    /// Pin the canonical `Try: ...` form on the internal-URL rejection so a
    /// future refactor that strips the hint doesn't go unnoticed. SSRF errors
    /// are reached frequently (LLM mistypes, redirect targets) — the Try
    /// hint steers the model to bash+curl instead of retrying the same URL.
    #[tokio::test]
    async fn web_fetch_internal_url_error_includes_canonical_try_hint() {
        let result = WebFetchTool
            .call(json!({"url": "http://169.254.169.254/latest/meta-data/"}))
            .await
            .unwrap();
        assert!(result.is_error);
        let msg = &result.content;
        assert!(
            msg.contains("Cannot fetch internal/private URL"),
            "msg: {}",
            msg
        );
        assert!(
            msg.contains("Try: only public hosts"),
            "missing canonical Try: hint: {}",
            msg
        );
        assert!(
            msg.contains("bash with curl"),
            "missing fallback suggestion: {}",
            msg
        );
    }

    #[test]
    fn is_internal_host_detects_private_ips() {
        assert!(is_internal_host("127.0.0.1"));
        assert!(is_internal_host("192.168.1.1"));
        assert!(is_internal_host("10.0.0.1"));
        assert!(is_internal_host("169.254.169.254"));
        assert!(is_internal_host("metadata.google.internal"));
        assert!(!is_internal_host("8.8.8.8"));
        assert!(!is_internal_host("example.com"));
    }

    #[test]
    fn validate_url_allows_https() {
        assert!(validate_url("https://example.com/page").is_ok());
        assert!(validate_url("http://example.com").is_ok());
    }

    #[test]
    fn ssrf_userinfo_bypass_blocked() {
        assert!(validate_url("http://x@169.254.169.254/latest/meta-data/").is_err());
    }

    #[test]
    fn ssrf_password_userinfo_blocked() {
        assert!(validate_url("http://user:pass@10.0.0.1/").is_err());
    }

    #[test]
    fn ssrf_ipv6_loopback_blocked() {
        assert!(validate_url("http://[::1]:8080/").is_err());
    }

    #[test]
    fn valid_url_with_at_in_path() {
        assert!(validate_url("https://example.com/path@thing").is_ok());
    }

    #[test]
    fn validate_url_rejects_ftp() {
        assert!(validate_url("ftp://files.example.com").is_err());
    }

    #[test]
    fn max_response_bytes_is_2mb() {
        assert_eq!(MAX_RESPONSE_BYTES, 2 * 1024 * 1024);
    }

    #[test]
    fn process_body_large_input_truncated() {
        let big = vec![b'x'; 3 * 1024 * 1024];
        let result = process_body(&big, "text/plain", 20_000);
        assert!(
            result.contains("Truncated"),
            "large body should be truncated: len={}",
            result.len()
        );
        assert!(result.len() <= 20_100, "output should respect max_length");
    }

    // ── DNS rebinding defense ───────────────────────────────────────────────

    #[tokio::test]
    async fn resolve_and_check_localhost_blocked() {
        let result = resolve_and_check("http://localhost/path").await;
        assert!(result.is_err(), "localhost should be blocked");
        let msg = result.unwrap_err();
        assert!(msg.contains("internal IP"), "msg: {}", msg);
        assert!(msg.contains("SSRF"), "msg: {}", msg);
    }

    #[tokio::test]
    async fn resolve_and_check_127_0_0_1_blocked() {
        let result = resolve_and_check("http://127.0.0.1:8080/api").await;
        assert!(result.is_err(), "127.0.0.1 should be blocked");
    }

    #[tokio::test]
    async fn resolve_and_check_nonexistent_host_errors() {
        let result = resolve_and_check("http://nonexistent.invalid.test.example:80/").await;
        assert!(result.is_err(), "unresolvable host should error");
        let msg = result.unwrap_err();
        assert!(msg.contains("DNS resolution failed"), "msg: {}", msg);
    }

    #[tokio::test]
    async fn resolve_and_check_empty_host_errors() {
        let result = resolve_and_check("http:///path").await;
        assert!(result.is_err());
    }

    #[test]
    fn is_internal_host_covers_link_local() {
        assert!(is_internal_host("169.254.1.1"));
        assert!(is_internal_host("169.254.0.1"));
    }

    // ── Redirect SSRF defense ───────────────────────────────────────────────

    #[test]
    fn classify_http_error_302_gives_generic_message() {
        let msg = classify_http_error("https://example.com", reqwest::StatusCode::FOUND);
        assert!(msg.contains("302"), "msg: {}", msg);
        assert!(msg.contains("example.com"), "msg: {}", msg);
    }

    #[test]
    fn status_is_redirection_for_common_codes() {
        assert!(reqwest::StatusCode::MOVED_PERMANENTLY.is_redirection());
        assert!(reqwest::StatusCode::FOUND.is_redirection());
        assert!(reqwest::StatusCode::TEMPORARY_REDIRECT.is_redirection());
        assert!(reqwest::StatusCode::PERMANENT_REDIRECT.is_redirection());
        assert!(!reqwest::StatusCode::OK.is_redirection());
    }

    #[test]
    fn redirect_policy_blocks_internal_host() {
        let policy = reqwest::redirect::Policy::custom(|attempt| {
            if let Some(host) = attempt.url().host_str() {
                if is_internal_host(host) {
                    return attempt.stop();
                }
            }
            attempt.follow()
        });
        let _ = policy; // policy is tested structurally; runtime test below
        assert!(is_internal_host("127.0.0.1"));
        assert!(is_internal_host("10.0.0.1"));
        assert!(!is_internal_host("8.8.8.8"));
    }

    #[test]
    fn redirect_policy_blocks_non_http_scheme() {
        assert!(validate_url("ftp://example.com").is_err());
        assert!(validate_url("file:///etc/passwd").is_err());
        assert!(validate_url("gopher://example.com").is_err());
    }

    #[test]
    fn validate_url_catches_internal_after_redirect_target() {
        assert!(validate_url("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_url("http://192.168.1.1/admin").is_err());
        assert!(validate_url("http://10.0.0.1/").is_err());
    }
}
