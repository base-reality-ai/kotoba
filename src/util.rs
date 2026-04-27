//! Shared utility functions and heuristics.
//!
//! Includes safe string truncation, fallback environment helpers,
//! and common formatting routines.

/// Truncate a string slice to at most `max_bytes` bytes, respecting UTF-8 char boundaries.
pub(crate) fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Maximum characters of tool output to inject into conversation context.
/// ~12,500 tokens at the 4-chars/token heuristic.
pub(crate) const MAX_TOOL_OUTPUT_CHARS: usize = 50_000;

/// Truncate tool output for context injection, preserving UTF-8 boundaries.
/// Returns the original string if within limits.
pub(crate) fn truncate_tool_output(content: &str) -> String {
    if content.len() <= MAX_TOOL_OUTPUT_CHARS {
        return content.to_string();
    }
    let truncated = safe_truncate(content, MAX_TOOL_OUTPUT_CHARS);
    format!(
        "{}\n\n[output truncated: showing {}/{} chars — use offset/limit parameters for targeted reading]",
        truncated, MAX_TOOL_OUTPUT_CHARS, content.len()
    )
}

/// In-place truncation of a String to at most `max_bytes`, respecting UTF-8 char boundaries.
pub(crate) fn safe_string_truncate(s: &mut String, max_bytes: usize) {
    if s.len() > max_bytes {
        let mut end = max_bytes;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s.truncate(end);
    }
}

/// Classic Levenshtein edit distance between two strings, in byte-space.
/// Two-row rolling implementation, O(m*n) time, O(n) space.
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_truncate_ascii() {
        assert_eq!(safe_truncate("hello world", 5), "hello");
    }

    #[test]
    fn safe_truncate_short_passthrough() {
        assert_eq!(safe_truncate("hi", 100), "hi");
    }

    #[test]
    fn safe_truncate_exact_boundary() {
        assert_eq!(safe_truncate("abc", 3), "abc");
    }

    #[test]
    fn safe_truncate_utf8_walkback() {
        let s = "aé"; // 1 + 2 = 3 bytes
        assert_eq!(safe_truncate(s, 2), "a");
        assert_eq!(safe_truncate(s, 3), "aé");
    }

    #[test]
    fn safe_truncate_emoji() {
        let s = "a🦀b";
        assert_eq!(safe_truncate(s, 2), "a");
        assert_eq!(safe_truncate(s, 5), "a🦀");
    }

    #[test]
    fn safe_truncate_empty() {
        assert_eq!(safe_truncate("", 10), "");
    }

    #[test]
    fn safe_truncate_zero_limit() {
        assert_eq!(safe_truncate("hello", 0), "");
    }

    /// Pin multibyte boundary safety across all common UTF-8 widths
    /// (2-byte é, 3-byte €, 4-byte 🦀) for both the free-fn and the
    /// in-place mutator. Each truncation must:
    ///   1. Never panic, even when the limit lands mid-character.
    ///   2. Never return an invalid `&str` / `String` (Rust enforces this
    ///      via `is_char_boundary`, but a future refactor that switches to
    ///      raw byte-slicing would silently violate it — this test pins
    ///      against that direction).
    ///   3. Walk *back* to the previous boundary (not forward), so the
    ///      result never exceeds `max_bytes`.
    ///
    /// "café" at limit=4 is the tester-named regression seed: byte 4 is
    /// mid-é (0xC3 0xA9), so a naive byte-slice would panic.
    #[test]
    fn safe_truncate_never_splits_multibyte_chars_mid_byte() {
        // 2-byte char: é (0xC3 0xA9). "café" = c-a-f-0xC3-0xA9 = 5 bytes.
        // limit=4 lands MID-é → must walk back to byte 3, yielding "caf".
        let s = "café";
        assert_eq!(safe_truncate(s, 4), "caf", "mid-é must walk back");
        assert_eq!(safe_truncate(s, 3), "caf", "boundary at 3 stays");
        assert_eq!(safe_truncate(s, 5), "café", "full length unchanged");

        // 3-byte char: € (0xE2 0x82 0xAC). "a€b" = 5 bytes.
        // limit=2 mid-€ → walk back to byte 1 = "a".
        // limit=3 still mid-€ → walk back to byte 1 = "a".
        let s = "a€b";
        assert_eq!(safe_truncate(s, 2), "a", "mid-€ from byte 2 walks back");
        assert_eq!(safe_truncate(s, 3), "a", "mid-€ from byte 3 walks back");
        assert_eq!(safe_truncate(s, 4), "a€", "boundary after € stays");

        // 4-byte char: 🦀. "x🦀" = 5 bytes. limit=2,3,4 all mid-emoji.
        let s = "x🦀";
        for mid_limit in [2, 3, 4] {
            assert_eq!(
                safe_truncate(s, mid_limit),
                "x",
                "mid-🦀 limit={} must walk back",
                mid_limit
            );
        }
        assert_eq!(safe_truncate(s, 5), "x🦀");

        // In-place mutator parity — same boundary contract.
        let mut s = "café".to_string();
        safe_string_truncate(&mut s, 4);
        assert_eq!(s, "caf", "in-place mutator must also walk back");

        let mut s = "a€b".to_string();
        safe_string_truncate(&mut s, 3);
        assert_eq!(s, "a", "in-place mutator handles 3-byte char");

        let mut s = "x🦀".to_string();
        safe_string_truncate(&mut s, 3);
        assert_eq!(s, "x", "in-place mutator handles 4-byte char");
    }

    #[test]
    fn safe_truncate_multibyte_at_char_end() {
        let s = "aéb"; // 1 + 2 + 1 = 4 bytes
        assert_eq!(safe_truncate(s, 3), "aé");
    }

    #[test]
    fn safe_string_truncate_ascii() {
        let mut s = "hello".to_string();
        safe_string_truncate(&mut s, 5);
        assert_eq!(s, "hello");
    }

    #[test]
    fn safe_string_truncate_multibyte() {
        let mut s = "🦀🦀🦀🦀".to_string(); // 16 bytes
        safe_string_truncate(&mut s, 6);
        assert!(s.len() <= 6);
        assert_eq!(s, "🦀");
    }

    #[test]
    fn safe_string_truncate_short_unchanged() {
        let mut s = "short".to_string();
        safe_string_truncate(&mut s, 100);
        assert_eq!(s, "short");
    }

    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein("bash", "bash"), 0);
    }

    #[test]
    fn levenshtein_one_edit() {
        assert_eq!(levenshtein("bash", "bassh"), 1);
    }

    #[test]
    fn levenshtein_different() {
        assert_eq!(levenshtein("bash", "grep"), 4);
    }

    #[test]
    fn levenshtein_empty() {
        assert_eq!(levenshtein("", "abc"), 3);
    }

    #[test]
    fn levenshtein_is_byte_space_not_char_space() {
        // Function is documented as "byte-space" — it operates on
        // `as_bytes()`, so multi-byte UTF-8 chars contribute their byte
        // length, not 1. A caller treating the result as a char-space
        // edit distance would be surprised: e.g., the single-codepoint
        // diff between "é" and "e" reports as 2 (2 bytes deleted + 1
        // byte inserted, then minimised) rather than 1. Pin the
        // documented behavior so a future "upgrade to char-space"
        // refactor has to be intentional and update every caller's
        // assumption with it.
        // 'é' = [0xC3, 0xA9] (2 bytes), 'e' = [0x65] (1 byte).
        let d = levenshtein("é", "e");
        assert!(
            d >= 2,
            "byte-space distance for é→e must be ≥2 (not the char-space 1): got {}",
            d
        );

        // Two identical multi-byte strings still report 0 — byte-space
        // doesn't penalise multibyteness on equal inputs.
        assert_eq!(levenshtein("é", "é"), 0);
        assert_eq!(levenshtein("🦀", "🦀"), 0);

        // Mixed ASCII + multibyte: "café" vs "cafe" differs by é→e,
        // which is 2 bytes vs 1 byte. Edit distance ≥ 2.
        assert!(
            levenshtein("café", "cafe") >= 2,
            "café→cafe byte-space distance must be ≥2"
        );
    }

    #[test]
    fn safe_string_truncate_zero_limit_clears_string() {
        // Symmetric with `safe_truncate_zero_limit` for the free-fn
        // variant. The implementation has an `end > 0 &&` guard before
        // `is_char_boundary` (defensive against integer underflow if a
        // future refactor touches the loop) — this test confirms the
        // zero-limit path lands on `s.truncate(0)` and produces an
        // empty string rather than panicking.
        let mut s = "hello world".to_string();
        safe_string_truncate(&mut s, 0);
        assert_eq!(s, "");

        // Same on a multi-byte input — must not slice mid-char.
        let mut s = "🦀🦀".to_string();
        safe_string_truncate(&mut s, 0);
        assert_eq!(s, "");
    }
}
