//! Wiki search — case-insensitive substring search across indexed pages.
//!
//! Split from `mod.rs` (Phase 1.1, Cycle 9). Uses split-impl: the `search`
//! method continues to extend the `Wiki` struct defined in the parent.
//! Helpers `count_occurrences` and `snippet_around` are search-only, so
//! they live here as module-local `fn` rather than `pub(super)` in the
//! parent — the directive's "search implementation" lives entirely here.

use std::io;

use crate::identity::Identity;

use super::{
    Layer, Wiki, WikiSearchHit, SEARCH_MAX_QUERY_LEN, SEARCH_MAX_RESULTS, SEARCH_SNIPPET_MAX,
    SEARCH_SNIPPET_SIDE,
};

impl Wiki {
    /// Case-insensitive substring search across all indexed wiki pages.
    ///
    /// Iterates `index.md` entries (the authoritative catalog) and reads
    /// each referenced page. Pages on disk but not in the index are
    /// ignored — surfacing drift is `/wiki lint`'s job.
    ///
    /// Empty or whitespace-only queries return `Ok(vec![])` so the caller
    /// can render a usage hint without branching on errors. Over-long
    /// queries are silently truncated to [`SEARCH_MAX_QUERY_LEN`] bytes
    /// (clamped to the nearest char boundary).
    ///
    /// Hits are ranked by `match_count` descending with a stable `path`
    /// ascending tiebreak, then capped at [`SEARCH_MAX_RESULTS`]. Per-page
    /// read errors are non-fatal: the offending page is skipped and a
    /// warning is pushed via `crate::warnings::push_warning`.
    pub fn search(&self, query: &str) -> io::Result<Vec<WikiSearchHit>> {
        self.search_inner(query, false)
    }

    /// Identity-aware search. Host-mode callers see host-layer hits before
    /// inherited kernel hits; kernel-mode callers keep the legacy flat rank.
    pub fn search_for_identity(
        &self,
        query: &str,
        identity: &Identity,
    ) -> io::Result<Vec<WikiSearchHit>> {
        self.search_inner(query, identity.is_host())
    }

    fn search_inner(&self, query: &str, stratify_layers: bool) -> io::Result<Vec<WikiSearchHit>> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        // Byte-cap then clamp to a char boundary so we don't slice through
        // a multi-byte codepoint (cycle-5 UTF-8 lesson).
        let capped = if trimmed.len() > SEARCH_MAX_QUERY_LEN {
            let mut end = SEARCH_MAX_QUERY_LEN;
            while end > 0 && !trimmed.is_char_boundary(end) {
                end -= 1;
            }
            &trimmed[..end]
        } else {
            trimmed
        };
        if capped.is_empty() {
            return Ok(Vec::new());
        }
        let needle = capped.to_lowercase();

        let idx = self.load_index()?;
        let mut hits: Vec<WikiSearchHit> = Vec::new();
        for entry in &idx.entries {
            let page = match self.read_page(&entry.path) {
                Ok(p) => p,
                Err(e) => {
                    crate::warnings::push_warning(format!(
                        "wiki: search skipped page {} — {}",
                        entry.path, e
                    ));
                    continue;
                }
            };

            let title_lc = page.title.to_lowercase();
            let body_lc = page.body.to_lowercase();
            let title_matches = count_occurrences(&title_lc, &needle);
            let body_matches = count_occurrences(&body_lc, &needle);
            let total = title_matches + body_matches;
            if total == 0 {
                continue;
            }

            // Snippet is extracted from `body_lc` (not `page.body`) so the match
            // offset is always valid — `to_lowercase()` can shift byte counts in
            // either direction (e.g. Turkish İ→i̇ grows 2→3, German ẞ→ß shrinks
            // 3→2). Trade: the snippet's surrounding context is lowercased too.
            // Acceptable because the snippet is informational, not a code-paste
            // target. (cycle-12 fix for BUG-11-1)
            let snippet = if body_matches > 0 {
                if let Some(idx) = body_lc.find(&needle) {
                    snippet_around(&body_lc, idx, needle.len())
                } else {
                    page.title.clone()
                }
            } else {
                page.title.clone()
            };

            hits.push(WikiSearchHit {
                path: entry.path.clone(),
                title: page.title,
                category: entry.category,
                layer: page.layer,
                match_count: total,
                snippet,
            });
        }

        hits.sort_by(|a, b| {
            let layer_order = if stratify_layers {
                layer_rank(a.layer).cmp(&layer_rank(b.layer))
            } else {
                std::cmp::Ordering::Equal
            };
            layer_order
                .then_with(|| b.match_count.cmp(&a.match_count))
                .then_with(|| a.path.cmp(&b.path))
        });
        hits.truncate(SEARCH_MAX_RESULTS);
        Ok(hits)
    }
}

fn layer_rank(layer: Layer) -> u8 {
    match layer {
        Layer::Host => 0,
        Layer::Kernel => 1,
    }
}

/// Count non-overlapping occurrences of `needle` in `haystack`. Both sides
/// are assumed already-lowercased by the caller (see [`Wiki::search`]). An
/// empty needle returns 0 — we never want to report infinite hits.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut pos = 0usize;
    let mut n = 0usize;
    while let Some(i) = haystack[pos..].find(needle) {
        n += 1;
        pos += i + needle.len();
        if pos >= haystack.len() {
            break;
        }
    }
    n
}

/// Extract a snippet of ≤[`SEARCH_SNIPPET_MAX`] bytes around a body match.
/// `match_byte_idx` is the byte offset of the match in `body`; `needle_len`
/// is the byte length of the (lowercased) needle — used to expand the
/// window past the match itself.
///
/// Always clamps window endpoints to char boundaries using `char_indices`
/// so multi-byte codepoints are never split (cycle-5 regression guard).
/// Embedded newlines collapse to spaces so the result is single-line
/// regardless of source formatting.
fn snippet_around(body: &str, match_byte_idx: usize, needle_len: usize) -> String {
    if body.is_empty() {
        return String::new();
    }
    let want_start = match_byte_idx.saturating_sub(SEARCH_SNIPPET_SIDE);
    let want_end = (match_byte_idx + needle_len + SEARCH_SNIPPET_SIDE).min(body.len());

    // Clamp endpoints outward to the nearest char boundaries so we never
    // slice through a multi-byte codepoint. `is_char_boundary(body.len())`
    // is always true, so the end loop terminates.
    let mut start = want_start;
    while start > 0 && !body.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = want_end;
    while end < body.len() && !body.is_char_boundary(end) {
        end += 1;
    }

    let slice = &body[start..end];
    let mut snippet = String::with_capacity(slice.len() + 6);
    if start > 0 {
        snippet.push('…');
    }
    for ch in slice.chars() {
        if ch == '\n' || ch == '\r' {
            snippet.push(' ');
        } else {
            snippet.push(ch);
        }
    }
    if end < body.len() {
        snippet.push('…');
    }

    // Final cap — if concatenation exceeded the max, truncate at a char
    // boundary and append an ellipsis.
    if snippet.len() > SEARCH_SNIPPET_MAX {
        let mut cut = SEARCH_SNIPPET_MAX;
        while cut > 0 && !snippet.is_char_boundary(cut) {
            cut -= 1;
        }
        snippet.truncate(cut);
        snippet.push('…');
    }
    snippet
}
