//! Uniform error formatting for headless entry points.
//!
//! Pillar 1 "Error messages include next steps": every user-facing error
//! should state what went wrong *and* an actionable `Try: <hint>`. This
//! module centralises the wording so call sites stay terse and the style
//! stays greppable.
//!
//! Pure — no I/O, no `eprintln!` — so the rendering is unit-testable and
//! the same function can feed stderr today and structured logging later.

/// Normalise `msg`, ensure it ends with a period, and append `Try: {hint}`.
///
/// - Interior line breaks (`\n`, `\r\n`, bare `\r`) are collapsed to a
///   single space in **both** `msg` and `hint` so the output stays on one
///   logical line (stderr readers and test assertions expect that). The
///   `\r\n` case is collapsed *first* so a single CRLF becomes one space,
///   not two.
/// - Leading and trailing whitespace is trimmed from both `msg` and `hint`
///   *after* line-break collapse, so `"foo   "` → `"foo. ..."` and
///   `"   "` is treated as empty (no separator artifact).
/// - An empty (or whitespace-only) `msg` skips the leading
///   period-and-space, so `("", "fix it")` returns `"Try: fix it"` rather
///   than `". Try: fix it"`.
/// - An empty (or whitespace-only) `hint` returns just the period-
///   normalised message with no `Try:` tail.
/// - When both are empty the function returns the empty string.
pub fn format_with_hint(msg: &str, hint: &str) -> String {
    let collapsed_msg = collapse_line_breaks(msg);
    let trimmed_msg = collapsed_msg.trim();
    let collapsed_hint = collapse_line_breaks(hint);
    let trimmed_hint = collapsed_hint.trim();

    let base = if trimmed_msg.is_empty() {
        String::new()
    } else if trimmed_msg.ends_with('.') {
        trimmed_msg.to_string()
    } else {
        format!("{trimmed_msg}.")
    };

    match (base.is_empty(), trimmed_hint.is_empty()) {
        (true, true) => String::new(),
        (true, false) => format!("Try: {trimmed_hint}"),
        (false, true) => base,
        (false, false) => format!("{base} Try: {trimmed_hint}"),
    }
}

/// Collapse all common line-break forms (`\r\n`, `\r`, `\n`) to single
/// spaces. CRLF is handled first so a single Windows line ending becomes
/// one space rather than two.
fn collapse_line_breaks(s: &str) -> String {
    s.replace("\r\n", " ").replace(['\r', '\n'], " ")
}

/// Convenience wrapper for the `dm: ` prefix used by every headless
/// error path. Pass `Some(hint)` when the error has an obvious next
/// step; pass `None` for errors where no single hint fits (and consider
/// adding one before shipping).
///
/// Both branches normalise line breaks so the "single logical line"
/// contract holds whether or not a hint is supplied — a multi-line `msg`
/// with `None` hint must not leak newlines through to stderr/log readers.
pub fn format_dm_error(msg: &str, hint: Option<&str>) -> String {
    let prefixed = format!("dm: {msg}");
    match hint {
        Some(h) => format_with_hint(&prefixed, h),
        None => collapse_line_breaks(&prefixed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_with_hint_appends_period_when_missing() {
        let out = format_with_hint("config error", "dm --doctor");
        assert_eq!(out, "config error. Try: dm --doctor");
    }

    #[test]
    fn format_with_hint_preserves_existing_period() {
        let out = format_with_hint("config error.", "dm --doctor");
        assert_eq!(out, "config error. Try: dm --doctor");
        assert!(
            !out.contains(".."),
            "no double period from duplicate normalisation"
        );
    }

    #[test]
    fn format_with_hint_trims_trailing_whitespace_before_period() {
        let out = format_with_hint("config error   ", "dm --doctor");
        assert_eq!(out, "config error. Try: dm --doctor");
    }

    #[test]
    fn format_dm_error_prefixes_dm() {
        let out = format_dm_error("boom", Some("dm --doctor"));
        assert!(out.starts_with("dm: "), "output: {out}");
    }

    #[test]
    fn format_dm_error_without_hint_omits_try() {
        let out = format_dm_error("boom", None);
        assert!(!out.contains("Try:"), "no hint tail when None: {out}");
        assert!(out.starts_with("dm: boom"), "still prefixed: {out}");
    }

    #[test]
    fn format_dm_error_with_hint_includes_try() {
        let out = format_dm_error("boom", Some("dm --doctor"));
        assert!(out.contains("Try: dm --doctor"), "hint present: {out}");
    }

    #[test]
    fn format_with_hint_empty_hint_returns_no_try() {
        let out = format_with_hint("boom", "");
        assert_eq!(out, "boom.");
        assert!(!out.contains("Try:"), "empty hint = no tail");
    }

    #[test]
    fn format_dm_error_multiline_message_single_line_output() {
        let out = format_dm_error("line1\nline2", Some("dm --doctor"));
        assert!(!out.contains('\n'), "single-line output: {out:?}");
        assert!(out.contains("line1 line2"), "newline collapsed: {out}");
        assert!(out.contains("Try: dm --doctor"), "hint preserved: {out}");
    }

    /// Pin the empty/whitespace-only contract: a `msg` that's empty or
    /// all whitespace (after line-break collapse) must NOT produce a
    /// stray `". "` artifact. Likewise an empty/whitespace `hint` must
    /// NOT produce a trailing `"Try: "` with no payload. Both empty
    /// returns the empty string. Catches a future "always emit period"
    /// or "always emit Try:" refactor that would silently inject the
    /// separator artifacts callers grep against.
    #[test]
    fn format_with_hint_empty_and_whitespace_branches_have_no_separator_artifacts() {
        // Empty msg + non-empty hint → just `Try: hint`, no leading dot.
        let out = format_with_hint("", "fix it");
        assert_eq!(
            out, "Try: fix it",
            "empty msg must not leave leading '. ' artifact"
        );
        assert!(!out.starts_with('.'), "no leading period: {out:?}");

        // Whitespace-only msg + non-empty hint — same.
        let out = format_with_hint("   ", "fix it");
        assert_eq!(out, "Try: fix it", "whitespace msg = empty msg");

        // Non-empty msg + empty hint → just `msg.`, no trailing `Try:`.
        let out = format_with_hint("config error", "");
        assert_eq!(out, "config error.");
        assert!(
            !out.contains("Try:"),
            "no trailing 'Try:' with empty hint: {out:?}"
        );

        // Non-empty msg + whitespace-only hint — treat as empty.
        let out = format_with_hint("config error", "   ");
        assert_eq!(out, "config error.", "whitespace hint = empty hint");
        assert!(!out.contains("Try:"));

        // Both empty → empty string (no period-only artifact).
        let out = format_with_hint("", "");
        assert_eq!(out, "", "both empty must return empty string");

        // Both whitespace-only → empty string.
        let out = format_with_hint("   ", "\t\n");
        assert_eq!(out, "", "both whitespace must return empty string");

        // Line-break-only msg/hint reduces to whitespace, then empty.
        let out = format_with_hint("\r\n\n", "");
        assert_eq!(out, "");
    }

    /// Pin the broadened "single logical line" contract: CR (`\r`) and
    /// CRLF (`\r\n`) — common in Windows files and paste-from-Windows —
    /// must collapse the same way as plain `\n`. Stderr parsers and grep
    /// assertions on `^...$` would silently break otherwise. The CRLF
    /// case must produce ONE space, not two (a regression risk if a
    /// future refactor swaps the order of replace operations).
    #[test]
    fn format_with_hint_collapses_crlf_and_bare_cr_in_msg_and_hint() {
        // CRLF in msg
        let out = format_with_hint("line1\r\nline2", "step a");
        assert!(!out.contains('\r'), "no CR in output: {out:?}");
        assert!(!out.contains('\n'), "no LF in output: {out:?}");
        assert_eq!(out, "line1 line2. Try: step a", "CRLF → single space");

        // Bare CR in msg
        let out = format_with_hint("line1\rline2", "step a");
        assert!(!out.contains('\r'));
        assert_eq!(out, "line1 line2. Try: step a", "bare CR → space");

        // CRLF in hint
        let out = format_with_hint("config error", "step 1\r\nstep 2");
        assert!(!out.contains('\r'));
        assert!(!out.contains('\n'));
        assert_eq!(out, "config error. Try: step 1 step 2");

        // Bare CR in hint
        let out = format_with_hint("config error", "step 1\rstep 2");
        assert!(!out.contains('\r'));
        assert_eq!(out, "config error. Try: step 1 step 2");

        // Mixed line breaks in both — most realistic Windows-paste scenario.
        let out = format_with_hint("a\r\nb\nc\rd", "x\ry\nz\r\nw");
        assert!(!out.contains('\r'));
        assert!(!out.contains('\n'));
        assert_eq!(out, "a b c d. Try: x y z w");
    }

    #[test]
    fn format_dm_error_none_hint_still_collapses_line_breaks() {
        // Asymmetry guard: the `None`-hint branch took a different path
        // through the function (raw `format!("dm: {msg}")`) and used to
        // skip line-break normalisation. A multi-line `msg` with no hint
        // would emit literal `\n` to stderr, breaking the same single-line
        // parser contract that the `Some(hint)` branch already honors.
        let out = format_dm_error("line1\nline2", None);
        assert!(!out.contains('\n'), "no LF in output: {out:?}");
        assert_eq!(out, "dm: line1 line2");
        assert!(!out.contains("Try:"), "no Try: tail when None: {out}");

        // CRLF + bare CR also collapse, matching the broadened contract.
        let out = format_dm_error("a\r\nb\rc", None);
        assert!(!out.contains('\r'));
        assert!(!out.contains('\n'));
        assert_eq!(out, "dm: a b c");
    }

    #[test]
    fn format_with_hint_collapses_newlines_in_hint_too() {
        // Docstring claims "output stays on one logical line" — that has
        // to apply to both `msg` and `hint`, otherwise a caller passing a
        // multi-line hint silently breaks stderr parsers and test
        // assertions that grep for a single line.
        let out = format_with_hint("config error", "step 1\nstep 2");
        assert!(!out.contains('\n'), "no newlines in output: {out:?}");
        assert_eq!(out, "config error. Try: step 1 step 2");
    }
}
