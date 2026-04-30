//! Wiki compaction bridge — synthesis page writers.
//!
//! Split from `mod.rs` (Phase 1.1, Cycle 10). Two public entry points
//! land on the split-impl extension of `Wiki`:
//!   * `write_compact_synthesis` — called by the compaction pipeline
//!     before messages are dropped.
//!   * `write_cycle_synthesis` — called by the chain runner after
//!     `ChainCycleComplete`.
//!
//! Helpers `chain_slug_kebab` and `render_cycle_node_output` are
//! single-caller, so they live here as module-local functions rather
//! than `pub(super)` in the parent.

use std::fmt::Write as _;
use std::fs;
use std::io;

use super::{
    auto_ingest_enabled, CycleNodeSnapshot, IndexEntry, Layer, PageType, Wiki, WikiIndex, WikiPage,
    CYCLE_SYNTHESIS_LOG_VERB, CYCLE_SYNTHESIS_OUTPUT_PER_NODE, INDEX_LOCK, SYNTHESIS_CYCLE_PREFIX,
};

/// Default cap on retained `synthesis/compact-*.md` pages. Long-running
/// chains otherwise accumulate one synthesis page per compaction event,
/// which has been observed to grow into the tens of thousands and
/// dominate `WikiStats` plus `index.md` size. Override at runtime via
/// `DM_WIKI_COMPACT_KEEP=<usize>`. Re-exported as `wiki::DEFAULT_COMPACT_KEEP`
/// for the `/wiki prune` slash-command default.
pub(crate) const DEFAULT_COMPACT_KEEP: usize = 200;

/// Compaction-page filename prefix. Both the writer
/// ([`Wiki::write_compact_synthesis`]) and the pruner
/// ([`Wiki::prune_compact_synthesis_to`]) reference this constant so a
/// rename can't silently desync the two paths.
pub(super) const COMPACT_SYNTH_PREFIX: &str = "synthesis/compact-";

impl Wiki {
    /// Persist a compaction summary as a `synthesis/` page. Invoked by the
    /// compaction pipeline before messages are dropped, so knowledge from
    /// the dying context survives.
    ///
    /// Returns the relative path of the page written, or `Ok(None)` if the
    /// `DM_WIKI_AUTO_INGEST` kill switch is off. Callers in compaction
    /// paths should swallow any `io::Error` — failing to write the wiki
    /// must not fail compaction itself.
    pub fn write_compact_synthesis(
        &self,
        summary: &str,
        messages_summarized: usize,
        sources: &[String],
    ) -> io::Result<Option<String>> {
        let layer = self.compact_synthesis_layer();
        self.write_compact_synthesis_with_layer(summary, messages_summarized, sources, layer)
    }

    /// Explicit-layer variant of [`Self::write_compact_synthesis`]. The
    /// default method derives this from `.dm/identity.toml`; this entry point
    /// keeps tests and future identity-aware callers from relying on ambient
    /// cwd or global state.
    pub fn write_compact_synthesis_with_layer(
        &self,
        summary: &str,
        messages_summarized: usize,
        sources: &[String],
        layer: Layer,
    ) -> io::Result<Option<String>> {
        if !auto_ingest_enabled() {
            return Ok(None);
        }

        let now = chrono::Local::now();
        let display_ts = now.format("%Y-%m-%d %H:%M:%S").to_string();
        // Nanosecond resolution so back-to-back compacts (within the same
        // millisecond) never collide on filename.
        let slug_ts = now.format("%Y%m%d-%H%M%S-%9f").to_string();
        let page_rel = format!("{}{}.md", COMPACT_SYNTH_PREFIX, slug_ts);

        let title = format!("Compaction summary — {}", display_ts);
        let msg_word = if messages_summarized == 1 {
            "message"
        } else {
            "messages"
        };
        let body = format!(
            "# {}\n\n\
             Captured from a full-summarization compact that condensed \
             {} {} into a single prefix.\n\n\
             ## Summary\n\n{}\n",
            title, messages_summarized, msg_word, summary,
        );

        let page = WikiPage {
            title: title.clone(),
            page_type: PageType::Synthesis,
            layer,
            sources: sources.to_vec(),
            last_updated: display_ts,
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body,
            extras: ::std::collections::BTreeMap::new(),
        };
        self.write_page(&page_rel, &page)?;

        {
            let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut idx = self.load_index().unwrap_or_default();
            idx.entries.push(IndexEntry {
                title,
                path: page_rel.clone(),
                one_liner: format!("Compact snapshot ({} {})", messages_summarized, msg_word),
                category: PageType::Synthesis,
                last_updated: Some(page.last_updated),
                outcome: None,
            });
            self.save_index(&idx)?;
        }

        let cap = std::env::var("DM_WIKI_COMPACT_KEEP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_COMPACT_KEEP);
        // Best-effort: a prune failure must never fail compaction.
        let _ = self.prune_compact_synthesis_to(cap);

        let _ = self.log().append("compact", &page_rel);

        Ok(Some(page_rel))
    }

    fn compact_synthesis_layer(&self) -> Layer {
        match crate::identity::load_at(&self.project_root()) {
            Ok(identity) if identity.is_host() => Layer::Host,
            Ok(_) => Layer::Kernel,
            Err(e) => {
                crate::warnings::push_warning(format!("identity: {}", e));
                Layer::Kernel
            }
        }
    }

    /// Cap retained `synthesis/compact-*.md` pages at `max_keep`, removing
    /// the oldest excess from both `index.md` and disk. Returns the number
    /// of pruned entries.
    ///
    /// Recency is read off the embedded `%Y%m%d-%H%M%S-%9f` slug in the
    /// page path — lexical sort matches chronological order at nanosecond
    /// resolution, so back-to-back compactions still rank deterministically.
    /// (`last_updated` only carries second resolution, which ties.)
    ///
    /// Other categories (Entity, Concept, Summary) and non-`compact-*`
    /// Synthesis pages (e.g. `cycle-*`, curated `run-*`) are left untouched
    /// and keep their relative order.
    ///
    /// Holds the crate-private `INDEX_LOCK` for the full read-mutate-write
    /// so concurrent writers can't reintroduce a pruned entry between this
    /// method's load and save.
    pub fn prune_compact_synthesis_to(&self, max_keep: usize) -> io::Result<usize> {
        let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let idx = self.load_index().unwrap_or_default();

        let (mut compact_synth, other): (Vec<IndexEntry>, Vec<IndexEntry>) =
            idx.entries.into_iter().partition(|e| {
                e.category == PageType::Synthesis && e.path.starts_with(COMPACT_SYNTH_PREFIX)
            });

        // Newest first — path embeds nanosecond slug, so lexical desc == newest-first.
        compact_synth.sort_by(|a, b| b.path.cmp(&a.path));

        let prune = if compact_synth.len() > max_keep {
            compact_synth.split_off(max_keep)
        } else {
            Vec::new()
        };
        let pruned_count = prune.len();

        for entry in &prune {
            let path = self.root.join(&entry.path);
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }

        if pruned_count == 0 {
            // No on-disk change — skip the index rewrite to avoid touching
            // mtimes and to keep the no-op cheap.
            return Ok(0);
        }

        let mut new_entries = compact_synth;
        new_entries.extend(other);
        self.save_index(&WikiIndex {
            entries: new_entries,
        })?;

        Ok(pruned_count)
    }

    /// Persist a chain-cycle snapshot as a `synthesis/` page so the wiki
    /// captures what each incubation cycle produced. Called by the chain
    /// runner after `ChainCycleComplete`; the next cycle's planner sees
    /// the accumulated trail via future `planner_brief` evolutions.
    ///
    /// Returns `Ok(None)` when the `DM_WIKI_AUTO_INGEST` kill switch is
    /// off. Callers in the chain pipeline should swallow any
    /// `io::Error` — a wiki write failure must never abort a cycle.
    ///
    /// The log verb is [`CYCLE_SYNTHESIS_LOG_VERB`] (not `"ingest"`), so
    /// [`Wiki::momentum`]'s ingest-only filter ignores these entries —
    /// a cycle synthesis is meta-history, not source churn.
    ///
    /// `outcome` is an opaque per-cycle signal (convention: `"green"` /
    /// `"red"` / `"mixed"`) written into the page frontmatter and surfaced
    /// by [`Wiki::planner_brief`] via `RecentCycle::outcome`. `None`
    /// omits the frontmatter line entirely (byte-identity with pre-C90
    /// synthesis pages); production callers currently always pass `None`
    /// because no outcome auto-detection is wired yet.
    pub fn write_cycle_synthesis(
        &self,
        cycle: usize,
        chain_name: &str,
        nodes: &[CycleNodeSnapshot],
        outcome: Option<&str>,
    ) -> io::Result<Option<String>> {
        if !auto_ingest_enabled() {
            return Ok(None);
        }

        let now = chrono::Local::now();
        let display_ts = now.format("%Y-%m-%d %H:%M:%S").to_string();
        let slug_ts = now.format("%Y%m%d-%H%M%S-%9f").to_string();
        let slug = chain_slug_kebab(chain_name);
        let page_rel = format!(
            "{}{:02}-{}-{}.md",
            SYNTHESIS_CYCLE_PREFIX, cycle, slug, slug_ts
        );

        let title = format!("Chain cycle {} — {}", cycle, chain_name);

        let mut body = format!(
            "# {}\n\n\
             Captured at end of cycle {} of `{}`.\n\n\
             ## Nodes\n\n",
            title, cycle, chain_name,
        );
        for snap in nodes {
            let rendered_output = render_cycle_node_output(&snap.output);
            writeln!(
                body,
                "- **{}** ({}):\n  {}",
                snap.name, snap.role, rendered_output,
            )
            .expect("write to String never fails");
        }

        let page = WikiPage {
            title: title.clone(),
            page_type: PageType::Synthesis,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: display_ts,
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: outcome.map(String::from),
            scope: vec![],
            body,
            extras: ::std::collections::BTreeMap::new(),
        };
        self.write_page(&page_rel, &page)?;

        {
            let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut idx = self.load_index().unwrap_or_default();
            idx.entries.push(IndexEntry {
                title,
                path: page_rel.clone(),
                one_liner: format!("Cycle {} of {}", cycle, chain_name),
                category: PageType::Synthesis,
                last_updated: Some(page.last_updated),
                outcome: page.outcome,
            });
            self.save_index(&idx)?;
        }

        let _ = self.log().append(CYCLE_SYNTHESIS_LOG_VERB, &page_rel);

        Ok(Some(page_rel))
    }
}

/// Render one cycle-node's `output` into the bullet body used by
/// [`Wiki::write_cycle_synthesis`]. Empty output renders as a
/// parenthesized placeholder; oversize output is truncated at the
/// nearest newline ≤ [`CYCLE_SYNTHESIS_OUTPUT_PER_NODE`] with a
/// `[...truncated]` marker so the bullet always ends cleanly.
///
/// Extracted as a free function so the truncation contract is tested
/// without constructing a whole Wiki+workspace fixture.
fn render_cycle_node_output(output: &str) -> String {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return "_(no output)_".to_string();
    }
    if trimmed.len() <= CYCLE_SYNTHESIS_OUTPUT_PER_NODE {
        return trimmed.to_string();
    }
    // Find the last newline at or before the cap so we never split a
    // line; fall back to the cap itself if the first line is longer
    // than the cap (still char-boundary safe).
    let mut cut = CYCLE_SYNTHESIS_OUTPUT_PER_NODE;
    while !trimmed.is_char_boundary(cut) && cut > 0 {
        cut -= 1;
    }
    let candidate = &trimmed[..cut];
    let end = candidate.rfind('\n').unwrap_or(cut);
    let kept = trimmed[..end].trim_end();
    format!("{}\n  [...truncated]", kept)
}

/// Kebab-case slug for chain filenames in [`Wiki::write_cycle_synthesis`].
/// `"Self-Improve / v2!"` → `"self-improve-v2"`. Every non-alphanumeric
/// run collapses to a single `-`; leading/trailing `-` trimmed;
/// ASCII-lowercased. Empty slug (e.g. all-punctuation input) falls back
/// to `"chain"` so a filename always has a readable segment.
///
/// Distinct from `entity_page_rel` (underscore, entity-scoped) and
/// `sanitize_dep_for_path` (underscore, concept-scoped, case-preserved)
/// because synthesis filenames embed multiple hyphenated segments
/// (`cycle-N-\<chain\>-\<ts\>`) and mixing separators would be ugly.
fn chain_slug_kebab(name: &str) -> String {
    let mut slug = String::with_capacity(name.len() + 4);
    let mut last_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            slug.push('-');
            last_dash = true;
        }
    }
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        "chain".to_string()
    } else {
        trimmed.to_string()
    }
}
