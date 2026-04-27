//! OSC 8 hyperlink support for clickable file paths in tool output.
//!
//! Terminals that support OSC 8 (iTerm2, Kitty, `WezTerm`, foot, …) render these
//! as clickable links. Terminals that don't understand OSC 8 render the plain text
//! unchanged — zero regression.
//!
//! Gated behind `app.hyperlinks` (`--hyperlinks` flag). Off by default.

use regex::Regex;
use std::sync::OnceLock;

/// Matches file paths in tool output text:
///   - `src/foo/bar.rs:42`     (relative, src/ prefix)
///   - `tests/integration.rs`  (tests/, benches/, examples/, build/ prefixes)
///   - `./relative/path.rs`    (explicit ./ prefix)
///   - `/absolute/path.rs`     (starts with /)
static PATH_RE: OnceLock<Regex> = OnceLock::new();

pub fn path_regex() -> &'static Regex {
    PATH_RE.get_or_init(|| {
        // Group 1: the path (with optional :line)
        Regex::new(r"(?:(?:src|tests|benches|examples|build)/|\.{1,2}/|/)[\w./\-]+(:\d+)?")
            .expect("PATH_RE regex is valid")
    })
}

/// Wrap a single path string in an OSC 8 hyperlink escape sequence.
/// `uri` is the full `file:///…` URI; `display` is the text to show.
pub fn osc8_hyperlink(uri: &str, display: &str) -> String {
    format!("\x1b]8;;{}\x1b\\{}\x1b]8;;\x1b\\", uri, display)
}

/// Convert a relative/absolute path token to a `file://` URI.
pub fn path_to_file_uri(path: &str) -> String {
    // Strip :line suffix for the URI (keep it in the display text)
    let bare = path.split(':').next().unwrap_or(path);
    if bare.starts_with('/') {
        format!("file://{}", bare)
    } else {
        // Resolve relative paths against cwd
        let cwd = std::env::current_dir().unwrap_or_default();
        format!("file://{}/{}", cwd.display(), bare)
    }
}

/// Process a single line of tool output: replace path tokens with OSC 8 hyperlinks.
/// Returns the original string unchanged when `enable` is false or no paths are found.
pub fn maybe_wrap_paths(line: &str, enable: bool) -> String {
    if !enable {
        return line.to_string();
    }
    let re = path_regex();
    let mut result = String::with_capacity(line.len());
    let mut last = 0;
    for m in re.find_iter(line) {
        result.push_str(&line[last..m.start()]);
        let token = m.as_str();
        let uri = path_to_file_uri(token);
        result.push_str(&osc8_hyperlink(&uri, token));
        last = m.end();
    }
    result.push_str(&line[last..]);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_detection_regex_src_prefix() {
        let re = path_regex();
        assert!(re.is_match("src/foo.rs:42"));
        assert!(re.is_match("src/lib/bar.rs"));
    }

    #[test]
    fn path_detection_regex_tests_benches_examples_build() {
        let re = path_regex();
        assert!(re.is_match("tests/integration.rs:10"), "tests/ prefix");
        assert!(re.is_match("benches/bench_main.rs"), "benches/ prefix");
        assert!(re.is_match("examples/demo.rs:1"), "examples/ prefix");
        assert!(re.is_match("build/output.rs"), "build/ prefix");
    }

    #[test]
    fn hyperlink_applied_to_tests_path() {
        let line = "error in tests/integration.rs:42";
        let result = maybe_wrap_paths(line, true);
        assert!(
            result.contains('\x1b'),
            "should contain escape sequences for tests/ path"
        );
        assert!(
            result.contains("tests/integration.rs:42"),
            "display text preserved"
        );
    }

    #[test]
    fn path_detection_regex_dot_slash() {
        let re = path_regex();
        assert!(re.is_match("./bar.rs"));
        assert!(re.is_match("../parent/file.rs"));
    }

    #[test]
    fn path_detection_regex_absolute() {
        let re = path_regex();
        assert!(re.is_match("/abs/path.rs"));
        assert!(re.is_match("/home/user/project/src/main.rs"));
    }

    #[test]
    fn hyperlink_escape_format_correct() {
        let result = osc8_hyperlink("file:///foo/bar.rs", "src/bar.rs");
        assert!(result.starts_with("\x1b]8;;file:///foo/bar.rs\x1b\\"));
        assert!(result.contains("src/bar.rs"));
        assert!(result.ends_with("\x1b]8;;\x1b\\"));
    }

    #[test]
    fn hyperlink_skipped_when_flag_off() {
        let line = "error in src/main.rs:42";
        let result = maybe_wrap_paths(line, false);
        assert_eq!(result, line);
        assert!(!result.contains('\x1b'));
    }

    #[test]
    fn hyperlink_applied_when_flag_on() {
        let line = "error in src/main.rs:42";
        let result = maybe_wrap_paths(line, true);
        assert!(result.contains('\x1b'), "should contain escape sequences");
        assert!(result.contains("src/main.rs:42"), "display text preserved");
    }

    #[test]
    fn path_to_file_uri_absolute_returns_file_scheme() {
        let uri = path_to_file_uri("/home/user/project/src/main.rs");
        assert!(
            uri.starts_with("file:///home/user/project"),
            "should be file URI: {uri}"
        );
    }

    #[test]
    fn path_to_file_uri_strips_line_suffix() {
        // "src/foo.rs:42" → URI should reference "src/foo.rs", not "src/foo.rs:42"
        let uri = path_to_file_uri("src/foo.rs:42");
        assert!(
            !uri.contains(":42"),
            "URI should not contain :line suffix: {uri}"
        );
    }

    #[test]
    fn maybe_wrap_paths_no_paths_returns_original() {
        let line = "some plain text without any file paths";
        let result = maybe_wrap_paths(line, true);
        assert_eq!(result, line, "plain text without paths should be unchanged");
    }
}
