//! Desktop notifications.
//!
//! Sends OS-level completion alerts (e.g. `osascript` or `notify-send`)
//! when long-running tasks or chains finish.

/// Truncate body to 80 chars with "..." ellipsis appended if truncated.
pub fn truncate_body(s: &str) -> String {
    match s.char_indices().nth(80) {
        Some((idx, _)) => format!("{}...", &s[..idx]),
        None => s.to_string(),
    }
}

/// Escape single quotes for embedding in shell strings.
/// Used on macOS for `osascript` commands; always compiled for test coverage.
#[cfg(any(target_os = "macos", test))]
fn escape_sq(s: &str) -> String {
    s.replace('\'', "'\\''")
}

/// Send a desktop notification.  Best-effort — never returns an error.
pub fn notify(title: &str, raw_body: &str) {
    let body = truncate_body(raw_body);

    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("notify-send")
            .arg(title)
            .arg(&body)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    #[cfg(target_os = "macos")]
    {
        let safe_body = escape_sq(&body);
        let safe_title = escape_sq(title);
        let script = format!(
            "display notification '{}' with title '{}'",
            safe_body, safe_title
        );
        let _ = std::process::Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        // no-op on other platforms
        let _ = (title, body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notify_body_truncates_at_80_chars() {
        let long = "x".repeat(200);
        let result = truncate_body(&long);
        assert!(
            result.len() <= 83,
            "truncated body should be at most 83 chars (80 + '...'), got {}",
            result.len()
        );
        assert!(
            result.ends_with("..."),
            "truncated body must end with '...'"
        );
    }

    #[test]
    fn notify_body_short_unchanged() {
        let short = "hello world";
        let result = truncate_body(short);
        assert_eq!(result, short);
    }

    #[test]
    fn notify_body_exact_80_chars_unchanged() {
        let s = "a".repeat(80);
        let result = truncate_body(&s);
        assert_eq!(result.len(), 80);
        assert!(!result.ends_with("..."));
    }

    #[test]
    fn notify_body_81_chars_truncated() {
        let s = "a".repeat(81);
        let result = truncate_body(&s);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 83); // 80 + 3
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn notify_command_is_notify_send_on_linux() {
        // Best-effort — may fail silently if notify-send is not installed.
        notify("", ""); // must not panic
    }

    #[cfg(any(target_os = "macos", test))]
    #[test]
    fn escape_sq_handles_single_quotes() {
        let input = "it's a test";
        let escaped = escape_sq(input);
        // The replacement should expand ' into '\'' — resulting string is longer
        assert!(
            escaped.len() > input.len(),
            "escaped string should be longer due to substitution"
        );
        // The original unescaped sequence 't'\''s' should not appear
        assert!(
            !escaped.contains("it's"),
            "original unescaped form should be replaced"
        );
    }

    #[cfg(any(target_os = "macos", test))]
    #[test]
    fn escape_sq_no_quotes_unchanged() {
        let input = "no quotes here";
        assert_eq!(escape_sq(input), input);
    }

    #[test]
    fn truncate_body_multibyte_chars_counted_by_char() {
        // Each 🦀 is 4 bytes but 1 char. 80 crabs should pass through unchanged.
        let s: String = "🦀".repeat(80);
        let result = truncate_body(&s);
        assert!(!result.ends_with("..."), "80 chars should not be truncated");
        // 81 crabs → truncated
        let s81: String = "🦀".repeat(81);
        let result81 = truncate_body(&s81);
        assert!(result81.ends_with("..."), "81 chars should be truncated");
    }
}
