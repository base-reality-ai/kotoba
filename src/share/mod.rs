//! Session transcript sharing and export.
//!
//! Formats agent conversations into HTML or Markdown files for sharing,
//! ensuring tool calls and reasoning phases are rendered legibly.

use std::fmt::Write as _;

use crate::session::{short_id, Session};

/// Render a tool call's `arguments` field for display.
///
/// Per `ollama::types::deserialize_args`, `arguments` is normalised to
/// Object form before reaching the conversation message builder
/// (`conversation.rs:498`). The legacy/test path keeps it as a JSON
/// string. The previous `.as_str().unwrap_or("{}")` form silently rendered
/// every real (Object-form) tool call as an empty `{}` in shared sessions
/// — the user lost all information about what tools were invoked with
/// what arguments. This helper handles both shapes faithfully.
fn render_tool_args(args: &serde_json::Value) -> String {
    if let Some(s) = args.as_str() {
        s.to_string()
    } else {
        serde_json::to_string(args).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Render a session as Markdown.
pub fn render_session_markdown(sess: &Session) -> String {
    let title = sess
        .title
        .as_deref()
        .filter(|t| !t.is_empty())
        .unwrap_or("Untitled Session");

    let date = sess.created_at.format("%Y-%m-%d").to_string();
    let short = short_id(&sess.id);

    let mut out = format!("# {}\n\n", title);
    write!(
        out,
        "> model: {} | session: {} | {}\n\n",
        sess.model, short, date
    )
    .expect("write to String never fails");
    out.push_str("---\n\n");

    for msg in &sess.messages {
        let role = msg["role"].as_str().unwrap_or("");
        let content = msg["content"].as_str().unwrap_or("");
        match role {
            "system" => {}
            "user" => {
                write!(out, "**You:** {}\n\n", content).expect("write to String never fails");
            }
            "assistant" => {
                if content.is_empty() {
                    // Check for tool_calls
                    if let Some(tool_calls) = msg["tool_calls"].as_array() {
                        for tc in tool_calls {
                            let name = tc["function"]["name"].as_str().unwrap_or("tool");
                            let args = render_tool_args(&tc["function"]["arguments"]);
                            write!(out, "```tool:{}\n{}\n```\n\n", name, args)
                                .expect("write to String never fails");
                        }
                    }
                } else {
                    out.push_str(content);
                    out.push_str("\n\n");
                    // Also render any tool_calls after content
                    if let Some(tool_calls) = msg["tool_calls"].as_array() {
                        for tc in tool_calls {
                            let name = tc["function"]["name"].as_str().unwrap_or("tool");
                            let args = render_tool_args(&tc["function"]["arguments"]);
                            write!(out, "```tool:{}\n{}\n```\n\n", name, args)
                                .expect("write to String never fails");
                        }
                    }
                }
            }
            "tool" => {
                out.push_str("<details><summary>Tool Result</summary>\n\n");
                out.push_str(content);
                out.push_str("\n\n</details>\n\n");
            }
            _ => {}
        }
    }

    out
}

pub fn render_session(sess: &Session) -> anyhow::Result<String> {
    let title = sess
        .title
        .as_deref()
        .filter(|t| !t.is_empty())
        .unwrap_or("dark matter session");

    let body_html = render_messages(&sess.messages);

    let created = sess.created_at.format("%Y-%m-%d %H:%M UTC").to_string();
    let meta = format!(
        "Model: {} · Session: {} · Started: {}",
        sess.model,
        short_id(&sess.id),
        created
    );

    Ok(TEMPLATE
        .replace("{{TITLE}}", &html_escape(title))
        .replace("{{META}}", &html_escape(&meta))
        .replace("{{BODY}}", &body_html))
}

fn render_messages(messages: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");
        let content = msg["content"].as_str().unwrap_or("");
        match role {
            "system" => {} // skip system prompt
            "user" => {
                writeln!(
                    out,
                    "<div class=\"msg user\"><span class=\"label\">You</span>\
                     <div class=\"content\">{}</div></div>",
                    render_content(content)
                )
                .expect("write to String never fails");
            }
            "assistant" => {
                if content.is_empty() {
                    if let Some(tool_calls) = msg["tool_calls"].as_array() {
                        for tc in tool_calls {
                            let name = tc["function"]["name"].as_str().unwrap_or("tool");
                            let args = render_tool_args(&tc["function"]["arguments"]);
                            let id = rand_id();
                            writeln!(
                                out,
                                "<details class=\"tool-result\" id=\"t{}\"><summary>Tool: {}</summary>\
                                 <pre>{}</pre></details>",
                                id,
                                html_escape(name),
                                html_escape(&args)
                            )
                            .expect("write to String never fails");
                        }
                    }
                    continue;
                }
                writeln!(
                    out,
                    "<div class=\"msg assistant\"><span class=\"label\">Dark Matter</span>\
                     <div class=\"content\">{}</div></div>",
                    render_markdown(content)
                )
                .expect("write to String never fails");
                if let Some(tool_calls) = msg["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let name = tc["function"]["name"].as_str().unwrap_or("tool");
                        let args = render_tool_args(&tc["function"]["arguments"]);
                        let id = rand_id();
                        writeln!(
                            out,
                            "<details class=\"tool-result\" id=\"t{}\"><summary>Tool: {}</summary>\
                             <pre>{}</pre></details>",
                            id,
                            html_escape(name),
                            html_escape(&args)
                        )
                        .expect("write to String never fails");
                    }
                }
            }
            "tool" => {
                let name = msg["name"].as_str().unwrap_or("tool");
                let first_line = content.lines().next().unwrap_or("");
                let preview = crate::util::safe_truncate(first_line, 120);
                let id = rand_id();
                writeln!(
                    out,
                    "<details class=\"tool-result\" id=\"t{}\"><summary>[{}] {}</summary>\
                     <pre>{}</pre></details>",
                    id,
                    html_escape(name),
                    html_escape(preview),
                    html_escape(content)
                )
                .expect("write to String never fails");
            }
            _ => {}
        }
    }
    out
}

/// Minimal Markdown → HTML: code fences, inline code, bold, headers, lists.
pub fn render_markdown(text: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    let mut fence_lang = String::new();
    let mut fence_buf = String::new();

    for line in text.lines() {
        if line.starts_with("```") {
            if in_fence {
                writeln!(
                    out,
                    "<pre><code class=\"lang-{}\">{}</code></pre>",
                    html_escape(&fence_lang),
                    html_escape(fence_buf.trim_end())
                )
                .expect("write to String never fails");
                fence_buf.clear();
                fence_lang.clear();
                in_fence = false;
            } else {
                fence_lang = line.trim_start_matches('`').to_string();
                in_fence = true;
            }
        } else if in_fence {
            fence_buf.push_str(line);
            fence_buf.push('\n');
        } else if let Some(rest) = line.strip_prefix("### ") {
            writeln!(out, "<h3>{}</h3>", render_inline(rest)).expect("write to String never fails");
        } else if let Some(rest) = line.strip_prefix("## ") {
            writeln!(out, "<h2>{}</h2>", render_inline(rest)).expect("write to String never fails");
        } else if let Some(rest) = line.strip_prefix("# ") {
            writeln!(out, "<h1>{}</h1>", render_inline(rest)).expect("write to String never fails");
        } else if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            writeln!(out, "<li>{}</li>", render_inline(rest)).expect("write to String never fails");
        } else if line.is_empty() {
            out.push_str("<br>\n");
        } else {
            writeln!(out, "<p>{}</p>", render_inline(line)).expect("write to String never fails");
        }
    }

    if in_fence && !fence_buf.is_empty() {
        writeln!(
            out,
            "<pre><code class=\"lang-{}\">{}</code></pre>",
            html_escape(&fence_lang),
            html_escape(fence_buf.trim_end())
        )
        .expect("write to String never fails");
    }

    out
}

fn render_inline(s: &str) -> String {
    let escaped = html_escape(s);
    let s = replace_bold(&escaped);
    replace_inline_code(&s)
}

/// Replace `**text**` with `<strong>text</strong>` using a simple state machine.
/// Assumes input is already HTML-escaped.
fn replace_bold(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("**") {
        out.push_str(&rest[..i]);
        rest = &rest[i + 2..];
        if let Some(j) = rest.find("**") {
            out.push_str("<strong>");
            out.push_str(&rest[..j]);
            out.push_str("</strong>");
            rest = &rest[j + 2..];
        } else {
            out.push_str("**");
        }
    }
    out.push_str(rest);
    out
}

/// Replace `` `code` `` with `<code>code</code>`.
/// Assumes input is already HTML-escaped.
fn replace_inline_code(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('`') {
        out.push_str(&rest[..i]);
        rest = &rest[i + 1..];
        if let Some(j) = rest.find('`') {
            out.push_str("<code>");
            out.push_str(&rest[..j]);
            out.push_str("</code>");
            rest = &rest[j + 1..];
        } else {
            out.push('`');
        }
    }
    out.push_str(rest);
    out
}

fn render_content(s: &str) -> String {
    html_escape(s).replace('\n', "<br>")
}

pub fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn rand_id() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0)
}

const TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,monospace;background:#1a1a1a;color:#e0e0e0;padding:24px;max-width:900px;margin:0 auto;line-height:1.6}
h1{font-size:1.4em;margin-bottom:4px;color:#7eb8f7}
.meta{color:#888;font-size:0.85em;margin-bottom:24px;font-family:monospace}
.msg{margin:16px 0;padding:12px 16px;border-radius:6px}
.msg .label{font-size:0.75em;text-transform:uppercase;letter-spacing:1px;opacity:0.6;margin-bottom:6px;display:block}
.user{background:#1e2e1e;border-left:3px solid #4a8a4a}
.user .label{color:#7ec87e}
.assistant{background:#1a1e2a;border-left:3px solid #4a6a9a}
.assistant .label{color:#7eb8f7}
.content p{margin:6px 0}
.content h1,.content h2,.content h3{margin:12px 0 6px;color:#aac8f0}
.content pre{background:#111;padding:12px;border-radius:4px;overflow-x:auto;margin:8px 0;font-size:0.9em}
.content code{background:#252525;padding:1px 5px;border-radius:3px;font-size:0.9em}
.content li{margin-left:20px;margin-bottom:2px}
.content strong{color:#fff}
details.tool-result{margin:8px 0;background:#252520;border-radius:4px;font-family:monospace;font-size:0.85em}
details summary{padding:6px 10px;cursor:pointer;color:#999;user-select:none}
details summary:hover{color:#ccc}
details pre{padding:8px 12px;color:#aaa;overflow-x:auto;max-height:300px;overflow-y:auto}
</style>
</head>
<body>
<h1>{{TITLE}}</h1>
<p class="meta">{{META}}</p>
{{BODY}}
</body>
</html>"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;

    fn make_session() -> Session {
        Session::new("/tmp".to_string(), "gemma4:26b".to_string())
    }

    fn make_session_with_messages(msgs: Vec<serde_json::Value>) -> Session {
        let mut s = make_session();
        for m in msgs {
            s.push_message(m);
        }
        s
    }

    /// Pin Object-form tool-call arguments rendering. Per
    /// `ollama::types::deserialize_args`, args are normalised to Object
    /// before reaching the conversation-message builder, so production
    /// `tool_calls[].function.arguments` is `{...}`, not a string. The
    /// previous `.as_str().unwrap_or("{}")` form silently rendered every
    /// real tool call as an empty `{}` in shared sessions, losing the
    /// "what tool was called with what args" information for the user.
    /// Pin both shapes (legacy String and current Object) on both
    /// renderers (Markdown + HTML).
    #[test]
    fn render_tool_args_handles_object_form_in_both_renderers() {
        // Object-form (production shape) — should serialize back to JSON.
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "bash",
                    "arguments": {"command": "ls /tmp"}
                }
            }]
        })]);
        let md = render_session_markdown(&sess);
        let html = render_session(&sess).unwrap();
        for (label, out) in [("markdown", &md), ("html", &html)] {
            assert!(
                out.contains("ls /tmp"),
                "{} must surface Object-form arg value, not '{{}}': {}",
                label,
                &out[..out.len().min(800)]
            );
            assert!(
                out.contains("command"),
                "{} must surface Object-form arg key: {}",
                label,
                &out[..out.len().min(800)]
            );
        }

        // Legacy String-form — same renderers must still work.
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\":\"echo legacy\"}"
                }
            }]
        })]);
        let md = render_session_markdown(&sess);
        let html = render_session(&sess).unwrap();
        for (label, out) in [("markdown", &md), ("html", &html)] {
            assert!(
                out.contains("echo legacy"),
                "{} must still handle legacy String-form args: {}",
                label,
                &out[..out.len().min(800)]
            );
        }
    }

    #[test]
    fn render_markdown_contains_title() {
        let mut sess = make_session();
        sess.title = Some("Test Session".to_string());
        let md = render_session_markdown(&sess);
        assert!(
            md.contains("# Test Session"),
            "md: {}",
            &md[..md.len().min(200)]
        );
    }

    #[test]
    fn render_session_includes_title() {
        let mut sess = make_session();
        sess.title = Some("Fix the auth bug".to_string());
        let html = render_session(&sess).unwrap();
        assert!(html.contains("Fix the auth bug"), "html: {}", &html[..200]);
    }

    #[test]
    fn render_session_skips_system_messages() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "system",
            "content": "You are a helpful assistant"
        })]);
        let html = render_session(&sess).unwrap();
        assert!(!html.contains("You are a helpful assistant"));
    }

    #[test]
    fn render_markdown_code_fence() {
        let md = "```rust\nfn main() {}\n```";
        let html = render_markdown(md);
        assert!(html.contains("<pre><code"), "html: {}", html);
        assert!(html.contains("fn main()"), "html: {}", html);
        assert!(
            !html.contains("```"),
            "should not contain raw backticks: {}",
            html
        );
    }

    #[test]
    fn render_markdown_bold() {
        let html = render_markdown("**important**");
        assert!(
            html.contains("<strong>important</strong>"),
            "html: {}",
            html
        );
    }

    #[test]
    fn render_markdown_inline_code() {
        let html = render_markdown("`cargo build`");
        assert!(html.contains("<code>cargo build</code>"), "html: {}", html);
    }

    #[test]
    fn html_escape_handles_lt_gt() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn render_session_tool_results_in_details() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "tool",
            "name": "bash",
            "content": "hello from bash"
        })]);
        let html = render_session(&sess).unwrap();
        assert!(html.contains("<details"), "html: {}", html);
        assert!(html.contains("<summary>"), "html: {}", html);
    }

    #[test]
    fn html_escape_handles_ampersand() {
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }

    #[test]
    fn html_escape_handles_quotes() {
        assert_eq!(html_escape("\"hello\""), "&quot;hello&quot;");
    }

    #[test]
    fn render_markdown_unmatched_bold_passthrough() {
        // A single ** without closing pair should not break
        let html = render_markdown("foo **bar baz");
        assert!(
            html.contains("bar baz"),
            "unmatched bold should not eat content: {html}"
        );
    }

    #[test]
    fn render_markdown_nested_inline_code_in_paragraph() {
        let html = render_markdown("Call `fn()` now.");
        assert!(
            html.contains("<code>fn()</code>"),
            "inline code missing: {html}"
        );
        assert!(
            html.contains("Call") && html.contains("now"),
            "surrounding text missing: {html}"
        );
    }

    #[test]
    fn render_markdown_headers() {
        let html = render_markdown("# H1\n## H2\n### H3");
        assert!(html.contains("<h1>"), "h1 missing: {html}");
        assert!(html.contains("<h2>"), "h2 missing: {html}");
        assert!(html.contains("<h3>"), "h3 missing: {html}");
        assert!(html.contains("H1") && html.contains("H2") && html.contains("H3"));
    }

    #[test]
    fn render_markdown_list_items() {
        let html = render_markdown("- item one\n- item two\n* item three");
        assert!(html.contains("<li>"), "li missing: {html}");
        assert!(
            html.contains("item one") && html.contains("item two") && html.contains("item three")
        );
    }

    #[test]
    fn render_markdown_unclosed_fence_still_renders() {
        // An unclosed ``` fence should still emit its buffered content
        let md = "```rust\nfn foo() {}";
        let html = render_markdown(md);
        assert!(
            html.contains("fn foo()"),
            "buffered content should appear: {html}"
        );
    }

    #[test]
    fn render_markdown_session_skips_no_title() {
        // render_session_markdown falls back to "Untitled Session" when title is None
        let sess = make_session();
        let md = render_session_markdown(&sess);
        assert!(
            md.contains("Untitled Session"),
            "should use fallback title: {md}"
        );
    }

    #[test]
    fn render_markdown_assistant_tool_calls_no_content() {
        // Assistant message with no content but tool_calls → should emit tool code block
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]
        })]);
        let md = render_session_markdown(&sess);
        assert!(
            md.contains("```tool:bash"),
            "tool block should appear: {md}"
        );
        assert!(md.contains("ls"), "args should appear: {md}");
    }

    #[test]
    fn render_session_user_message_appears() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "user",
            "content": "What is the answer?"
        })]);
        let html = render_session(&sess).unwrap();
        assert!(
            html.contains("What is the answer?"),
            "user msg missing: {}",
            &html[..html.len().min(400)]
        );
    }

    #[test]
    fn render_content_newline_becomes_br() {
        let out = render_content("line1\nline2");
        assert!(out.contains("<br>"), "newline should become <br>: {out}");
        assert!(out.contains("line1") && out.contains("line2"));
    }

    #[test]
    fn render_inline_escapes_html_in_bold() {
        let out = render_inline("**<em>**");
        assert!(
            out.contains("&lt;em&gt;"),
            "html inside bold should be escaped: {out}"
        );
        assert!(
            out.contains("<strong>"),
            "bold wrapper should be present: {out}"
        );
    }

    #[test]
    fn replace_inline_code_unclosed_backtick_passthrough() {
        let out = replace_inline_code("start `unclosed");
        assert!(
            out.contains('`'),
            "unclosed backtick should remain in output: {out}"
        );
        assert!(
            out.contains("unclosed"),
            "remaining text should appear: {out}"
        );
    }

    #[test]
    fn html_escape_newline_and_plain_text_unchanged() {
        let out = html_escape("plain text\nwith newline");
        assert_eq!(
            out, "plain text\nwith newline",
            "plain text and newlines should be unchanged"
        );
    }

    #[test]
    fn share_render_html_short_id_no_panic() {
        let mut sess = make_session();
        sess.id = "abc".to_string(); // shorter than 8 chars
        let html = render_session(&sess).unwrap();
        assert!(
            html.contains("abc"),
            "short ID should appear in output: {}",
            &html[..html.len().min(300)]
        );
    }

    #[test]
    fn render_markdown_empty_string_returns_something() {
        // Should not panic on empty input
        let html = render_markdown("");
        let _ = html; // just verify no panic
    }

    #[test]
    fn inline_code_escapes_surrounding_html() {
        let result = render_inline("<script>alert(1)</script> `safe`");
        assert!(
            result.contains("&lt;script&gt;"),
            "leading HTML should be escaped: {result}"
        );
        assert!(result.contains("<code>safe</code>"));
        assert!(!result.contains("<script>"));
    }

    #[test]
    fn render_html_assistant_tool_calls_no_content() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}]
        })]);
        let html = render_session(&sess).unwrap();
        assert!(
            html.contains("Tool: bash"),
            "tool name should appear: {}",
            &html[..html.len().min(500)]
        );
        assert!(
            html.contains("&quot;cmd&quot;"),
            "args should be escaped: {}",
            &html[..html.len().min(500)]
        );
    }

    #[test]
    fn render_html_assistant_tool_calls_with_content() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/x\"}"}}]
        })]);
        let html = render_session(&sess).unwrap();
        assert!(html.contains("Let me check that"), "content should appear");
        assert!(
            html.contains("Tool: read_file"),
            "tool_calls should also appear"
        );
    }

    #[test]
    fn render_html_assistant_empty_no_tool_calls() {
        let sess = make_session_with_messages(vec![serde_json::json!({
            "role": "assistant",
            "content": ""
        })]);
        let html = render_session(&sess).unwrap();
        assert!(
            !html.contains("Dark Matter"),
            "empty assistant msg should not render"
        );
    }

    #[test]
    fn render_messages_tool_preview_truncates_long_line() {
        let long_line = "x".repeat(200);
        let msgs = vec![serde_json::json!({
            "role": "tool",
            "name": "bash",
            "content": long_line
        })];
        let html = render_messages(&msgs);
        let summary_start = html.find("<summary>").unwrap();
        let summary_end = html.find("</summary>").unwrap();
        let summary = &html[summary_start..summary_end];
        assert!(
            summary.contains("[bash]"),
            "tool name should appear in summary"
        );
        let x_count = summary.matches('x').count();
        assert!(
            x_count <= 120,
            "summary should truncate to ~120 chars, got {} x's",
            x_count
        );
    }

    #[test]
    fn inline_code_escapes_trailing_html() {
        let result = render_inline("`code` <img onerror=x>");
        assert!(result.contains("<code>code</code>"));
        assert!(
            result.contains("&lt;img"),
            "trailing HTML should be escaped: {result}"
        );
        assert!(!result.contains("<img"));
    }
}
