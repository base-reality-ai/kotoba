//! Fuzzy matching of slash command names against the public registry.
//! The registry (`SLASH_COMMAND_NAMES`) remains in `commands.rs` — it
//! is consumed by `input.rs` for Tab autocomplete and by many tests,
//! so it stays at the module root.

use super::SLASH_COMMAND_NAMES;

/// Fuzzy-match a (possibly misspelled) slash command name against
/// `SLASH_COMMAND_NAMES`. Returns the best candidate or `None` when
/// nothing is close enough. The leading `/` must already be stripped.
pub fn suggest_slash_command(input: &str) -> Option<&'static str> {
    if input.is_empty() {
        return None;
    }

    // Substring match (skipped for 1-char inputs to avoid spurious matches
    // like "c" → "attach"; single chars fall through to Levenshtein which
    // picks the closest short command). The `registered.len() >= 3` guard
    // separately prevents short registered names like "cd"/"pr" from
    // eagerly swallowing arbitrary input.
    if input.len() >= 2 {
        for &registered in SLASH_COMMAND_NAMES {
            if registered.len() >= 3 && (registered.contains(input) || input.contains(registered)) {
                return Some(registered);
            }
        }
    }

    // Edit-distance match with a tight threshold (slash commands are short).
    let mut best: Option<(&'static str, usize)> = None;
    for &registered in SLASH_COMMAND_NAMES {
        let dist = crate::util::levenshtein(input, registered);
        if dist <= 2 && best.is_none_or(|(_, d)| dist < d) {
            best = Some((registered, dist));
        }
    }
    best.map(|(name, _)| name)
}
