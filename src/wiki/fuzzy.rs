//! Levenshtein-based fuzzy ranking over wiki index entries. Used by
//! `wiki_lookup` and `wiki_search` (and, in a future cycle, the
//! `/wiki search` slash command) to surface near-misses when an exact
//! or substring match returns nothing.

use super::IndexEntry;

/// Similarity threshold for "close enough": ~⅓ of `needle_lc.len()`,
/// clamped to `[2, 4]`. Both wiki tools share this contract, so the
/// formula lives here.
pub fn fuzzy_threshold(needle_lc: &str) -> usize {
    std::cmp::max(2, needle_lc.len().saturating_sub(needle_lc.len() * 2 / 3)).min(4)
}

/// Rank `entries` by min Levenshtein distance from `needle_lc` to
/// (lowercased title, lowercased path, basename of path sans `.md`).
/// Returns up to `top_n` entries sorted ascending by `(distance, path)`.
/// `category_filter`, when `Some`, restricts to entries whose
/// `IndexEntry::category.as_str()` matches.
pub fn rank_entries_by_levenshtein(
    entries: &[IndexEntry],
    needle_lc: &str,
    category_filter: Option<&str>,
    top_n: usize,
) -> Vec<(usize, String, String)> {
    let mut v: Vec<(usize, String, String)> = Vec::new();
    for entry in entries {
        if let Some(cat) = category_filter {
            if entry.category.as_str() != cat {
                continue;
            }
        }
        let title_l = entry.title.to_lowercase();
        let path_l = entry.path.to_lowercase();
        let base_full = path_l.rsplit('/').next().unwrap_or(path_l.as_str());
        let basename = base_full.strip_suffix(".md").unwrap_or(base_full);
        let d_title = crate::util::levenshtein(needle_lc, &title_l);
        let d_path = crate::util::levenshtein(needle_lc, &path_l);
        let d_base = crate::util::levenshtein(needle_lc, basename);
        let dist = d_title.min(d_path).min(d_base);
        v.push((dist, entry.path.clone(), entry.title.clone()));
    }
    v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    v.truncate(top_n);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wiki::PageType;

    fn entry(path: &str, title: &str, cat: PageType) -> IndexEntry {
        IndexEntry {
            path: path.to_string(),
            title: title.to_string(),
            category: cat,
            one_liner: String::new(),
            last_updated: None,
            outcome: None,
        }
    }

    #[test]
    fn threshold_floor_and_ceiling() {
        assert_eq!(fuzzy_threshold("a"), 2);
        assert_eq!(fuzzy_threshold("abcdef"), 2);
        assert_eq!(fuzzy_threshold("abcdefghi"), 3);
        assert_eq!(fuzzy_threshold("abcdefghijkl"), 4);
        assert_eq!(fuzzy_threshold("abcdefghijklmnopqrstuvwxyz"), 4);
    }

    #[test]
    fn rank_returns_empty_for_empty_entries() {
        let out = rank_entries_by_levenshtein(&[], "anything", None, 3);
        assert!(out.is_empty());
    }

    #[test]
    fn rank_sorts_by_distance_then_path() {
        // Both entries have basename distance 0 to needle "foo"; tie
        // breaks on path ascending.
        let entries = vec![
            entry("entities/foo_b.md", "Bee", PageType::Entity),
            entry("entities/foo_a.md", "Aye", PageType::Entity),
        ];
        let out = rank_entries_by_levenshtein(&entries, "foo_a", None, 3);
        // foo_a has dist 0 to basename, foo_b has dist 1 → foo_a first.
        assert_eq!(out[0].1, "entities/foo_a.md", "best match first: {:?}", out);
        assert_eq!(out[1].1, "entities/foo_b.md", "runner-up second: {:?}", out);
    }

    #[test]
    fn rank_respects_category_filter() {
        let entries = vec![
            entry("entities/auth.md", "Authish", PageType::Entity),
            entry("concepts/auth.md", "Authish", PageType::Concept),
        ];
        let out = rank_entries_by_levenshtein(&entries, "authish", Some("entity"), 3);
        assert_eq!(out.len(), 1, "filter should drop concept: {:?}", out);
        assert_eq!(out[0].1, "entities/auth.md");
    }

    #[test]
    fn rank_uses_basename_for_distance() {
        // Title is unrelated; full path has a long prefix; only basename
        // matches the needle. Basename-min must surface this entry.
        let entries = vec![entry("entities/sssssn.md", "x", PageType::Entity)];
        let out = rank_entries_by_levenshtein(&entries, "sssssn", None, 3);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].0, 0,
            "basename match should be distance 0: {:?}",
            out
        );
    }
}
