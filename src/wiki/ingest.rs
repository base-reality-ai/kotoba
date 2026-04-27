//! Wiki ingest pipeline — records a file read, writes/updates its entity
//! page, upserts the index, appends the log, and dedups on content hash.
//!
//! Split from `mod.rs` (Phase 1.1, Cycle 11). Uses split-impl: the
//! `ingest_file` / `ingest_file_internal` methods continue to extend the
//! `Wiki` struct defined in the parent. `ingest_file_internal` is
//! `pub(super)` because `Wiki::refresh` (in `mod.rs`) calls it directly to
//! force a rebuild on schema changes; the public `ingest_file` wrapper
//! pins `force=false` for normal ingests.
//!
//! Also hosts `detect_entity_kind`, `extract_dependencies`, and
//! `extract_purpose` — free-fn frontmatter extractors moved here in
//! Cycle 18 since `ingest_file_internal` is their sole production caller.

use std::fs;
use std::io;
use std::path::Path;

use super::{
    auto_ingest_enabled, collect_module_docs, content_hash, entity_kind_from_rust_keyword,
    entity_page_rel, extract_content_preview, extract_key_exports, ingest_cache_check_and_update,
    inject_wiki_links, rust_item_regex, rust_use_regex, EntityKind, IndexEntry, IngestOutcome,
    PageType, SkipReason, Wiki, WikiPage, INDEX_LOCK,
};

impl Wiki {
    /// Record that a file was read. Writes/updates a stub entity page, upserts
    /// the index entry, appends to the log, and dedups on content hash. All
    /// operations are idempotent.
    ///
    /// Synchronous I/O — callers in hot paths should wrap via
    /// `tokio::task::spawn_blocking`. `canonical_path` must already be
    /// canonicalized; `project_root` should also be canonical so the
    /// `strip_prefix` check succeeds for files inside the project.
    pub fn ingest_file(
        &self,
        project_root: &Path,
        canonical_path: &Path,
        content: &str,
    ) -> io::Result<IngestOutcome> {
        self.ingest_file_internal(project_root, canonical_path, content, false)
    }

    /// Internal variant with a `force` flag. `refresh()` sets `force=true`
    /// to bypass the content-hash short-circuit when the source bytes are
    /// unchanged but the wiki page's schema needs to be rebuilt (e.g.
    /// `entity_kind` was added in Cycle 38). The cache is *still* updated
    /// so subsequent forced ingests of the same bytes remain idempotent.
    pub(super) fn ingest_file_internal(
        &self,
        project_root: &Path,
        canonical_path: &Path,
        content: &str,
        force: bool,
    ) -> io::Result<IngestOutcome> {
        if !auto_ingest_enabled() {
            return Ok(IngestOutcome::Skipped(SkipReason::Disabled));
        }

        let Ok(rel) = canonical_path.strip_prefix(project_root) else {
            return Ok(IngestOutcome::Skipped(SkipReason::OutsideProject));
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.is_empty() || rel_str == "." {
            return Ok(IngestOutcome::Skipped(SkipReason::IneligiblePath));
        }
        if rel_str.starts_with(".dm/wiki/") {
            return Ok(IngestOutcome::Skipped(SkipReason::InsideWikiDir));
        }

        let key = canonical_path.to_string_lossy().into_owned();
        let hash = content_hash(content);
        let changed = ingest_cache_check_and_update(&key, hash);
        if !force && !changed {
            return Ok(IngestOutcome::Skipped(SkipReason::UnchangedSinceLast));
        }

        let page_rel = entity_page_rel(&rel_str);
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let preview = extract_content_preview(&rel_str, content);
        // Snapshot entity-source pairs from the current index (page-side
        // frontmatter is authoritative for sources; matches the
        // compute_inbound_links read pattern). Loaded *before* writing
        // this page so we know what other entities to cross-link to.
        // Race: a concurrent ingest landing between snapshot and write
        // will miss a link here; the next ingest of this same source
        // picks it up. Acceptable vs. widening INDEX_LOCK scope.
        let entity_srcs: Vec<(String, String)> = self
            .load_index()
            .unwrap_or_default()
            .entries
            .into_iter()
            .filter(|e| e.category == PageType::Entity)
            .filter_map(|e| {
                let page_abs = self.root.join(&e.path);
                let text = fs::read_to_string(&page_abs).ok()?;
                let page = WikiPage::parse(&text)?;
                let src = page.sources.into_iter().next()?;
                Some((e.path, src))
            })
            .collect();
        let linked_preview = inject_wiki_links(&preview, &entity_srcs, &rel_str);
        let body = if linked_preview.is_empty() {
            format!(
                "# {}\n\nSource file: `{}`\n\nLast observed: {}\n\n\
                 *(file is empty or whitespace-only — no preview extracted)*\n",
                rel_str, rel_str, now
            )
        } else {
            format!(
                "# {}\n\nSource file: `{}`\n\nLast observed: {}\n\n{}",
                rel_str, rel_str, now, linked_preview
            )
        };
        let page = WikiPage {
            title: rel_str.clone(),
            page_type: PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![rel_str.clone()],
            last_updated: now,
            entity_kind: detect_entity_kind(&rel_str, content),
            purpose: extract_purpose(&rel_str, content),
            key_exports: extract_key_exports(&rel_str, content),
            dependencies: extract_dependencies(&rel_str, content),
            outcome: None,
            scope: vec![],
            body,
        };
        self.write_page(&page_rel, &page)?;

        {
            let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut idx = self.load_index().unwrap_or_default();
            let entry = IndexEntry {
                title: rel_str.clone(),
                path: page_rel.clone(),
                one_liner: format!("File: {}", rel_str),
                category: PageType::Entity,
                last_updated: Some(page.last_updated),
                outcome: None,
            };
            if let Some(existing) = idx.entries.iter_mut().find(|e| e.path == page_rel) {
                *existing = entry;
            } else {
                idx.entries.push(entry);
            }
            self.save_index(&idx)?;
        }

        // Log errors are deliberately non-fatal — a dropped log line is
        // preferable to a failed ingest that already wrote the page.
        let _ = self.log().append("ingest", &rel_str);

        // Summary is now stale; mark but don't regenerate (lazy at consumer
        // boundary). Best-effort — ingest success is not blocked by a
        // marker-write failure.
        let _ = self.mark_summary_dirty();

        Ok(IngestOutcome::Ingested { page_rel })
    }
}

/// Infer the structural kind of a source file for `WikiPage.entity_kind`.
///
/// - Non-Rust paths → `None` (detection is Rust-only today).
/// - Zero represented kinds (empty file, or only `const`/`static`/`type`/
///   `mod`/`impl` — kinds the schema doesn't enumerate) → `None`.
/// - Exactly one represented kind → that kind.
/// - Multiple represented kinds → `Some(EntityKind::Unknown)` as the
///   "mixed" sentinel. Test coverage pins this so a future cycle
///   promoting "mixed" to a dedicated variant trips a canary.
///
/// Reuses `rust_item_regex`, so detection tracks the same item set that
/// `extract_rust_preview` already surfaces in the `## Items` section —
/// preview and kind can't disagree on what a file contains.
pub(super) fn detect_entity_kind(rel_path: &str, content: &str) -> Option<EntityKind> {
    if !rel_path.ends_with(".rs") {
        return None;
    }
    let re = rust_item_regex();
    let mut seen: std::collections::BTreeSet<EntityKind> = std::collections::BTreeSet::new();
    for cap in re.captures_iter(content) {
        let kind = cap.name("kind").map_or("", |m| m.as_str());
        if let Some(ek) = entity_kind_from_rust_keyword(kind) {
            seen.insert(ek);
        }
    }
    match seen.len() {
        0 => None,
        1 => seen.into_iter().next(),
        _ => Some(EntityKind::Unknown),
    }
}

/// `use foo::{\n    bar,\n    baz,\n};` groups are whitespace-collapsed
/// to `"foo::{bar, baz}"` so each dependency is a single string.
///
/// Known gaps match `extract_key_exports`: string-literal false
/// positives and missing nested-module `use` statements.
pub(super) fn extract_dependencies(rel_path: &str, content: &str) -> Vec<String> {
    if !rel_path.ends_with(".rs") {
        return Vec::new();
    }
    let re = rust_use_regex();
    let mut out: Vec<String> = Vec::new();
    for cap in re.captures_iter(content) {
        let raw = cap.get(1).map_or("", |m| m.as_str());
        // Strip top-level `as X` only when there's no group (`{`) in the
        // path — otherwise `foo::{bar as b, baz}` would be truncated at
        // the first ` as `, dropping `baz`. Group-internal renames are
        // preserved verbatim on purpose.
        let trimmed = if raw.contains('{') {
            raw
        } else {
            raw.split(" as ").next().unwrap_or(raw)
        };
        let collapsed: String = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
        if !collapsed.is_empty() {
            out.push(collapsed);
        }
    }
    out
}

/// Infer a 1-3 line purpose summary from a Rust source file's module
/// (`//!`) docs for `WikiPage.purpose`. Returns `None` for non-Rust
/// files, files with no module docs, or files whose docs are empty
/// after stripping.
///
/// The summary is the first paragraph of `//!` lines — consecutive
/// non-blank lines, capped at 3 — joined with single spaces. The
/// returned string contains no newlines, making it safe to emit as
/// a single-line YAML frontmatter value.
pub(super) fn extract_purpose(rel_path: &str, content: &str) -> Option<String> {
    if !rel_path.ends_with(".rs") {
        return None;
    }
    let docs = collect_module_docs(content);
    let mut para: Vec<&str> = Vec::new();
    for line in &docs {
        if line.is_empty() {
            break;
        }
        para.push(line);
        if para.len() >= 3 {
            break;
        }
    }
    let joined = para.join(" ").trim().to_string();
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}
