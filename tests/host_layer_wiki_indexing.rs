//! Tier-4 paradigm-gap regression test (kotoba v0.3).
//!
//! Demonstrates that host-layer wiki pages — the ones kotoba's host caps
//! produce under `.dm/wiki/entities/Vocabulary/`, `Kanji/`, `Persona/`, and
//! `synthesis/` — are invisible to `Wiki::search` / `Wiki::stats` because
//! they are never registered in `index.md`. The kernel's auto-ingest
//! pipeline (`Wiki::ingest_file`) skips any path under `.dm/wiki/` (see
//! `src/wiki/ingest.rs::ingest_file_internal`, the `InsideWikiDir` guard),
//! and host caps write pages with raw `std::fs::write` — they don't
//! invoke `Wiki::write_page` and they don't call `save_index`.
//!
//! This test pins the broken state. When the canonical-dm fix lands
//! (either kernel-side host-page ingest, or a `Wiki::register_host_page`
//! API kotoba can call), the asserts here flip from "should be empty" to
//! "should find the page" and the gap doc moves to RESOLVED.
//!
//! See `.dm/wiki/concepts/paradigm-gap-host-layer-wiki-ingest.md` for the
//! full diagnosis + recommended fix.

use dark_matter::wiki::{Layer, PageType, Wiki, WikiPage};

/// Mirrors what `host_caps::log_vocabulary_in` writes today: a
/// frontmatter-shaped markdown file under
/// `.dm/wiki/entities/Vocabulary/`, no index update.
fn write_host_vocab_like_kotoba(project_root: &std::path::Path, slug: &str, body: &str) {
    let dir = project_root
        .join(".dm")
        .join("wiki")
        .join("entities")
        .join("Vocabulary");
    std::fs::create_dir_all(&dir).expect("create vocab dir");
    let path = dir.join(format!("{}.md", slug));
    std::fs::write(&path, body).expect("write vocab page");
}

#[test]
fn host_layer_vocab_pages_are_invisible_to_wiki_search() {
    let tmp = tempfile::tempdir().expect("tempdir");
    // `Wiki::open` creates `.dm/wiki/` layout (entities/, concepts/, etc.)
    // and seeds an empty `index.md` if missing.
    let wiki = Wiki::open(tmp.path()).expect("open wiki");

    // Kotoba's host_log_vocabulary effectively does this — write a
    // well-formed entity page directly to the host-layer entities dir.
    write_host_vocab_like_kotoba(
        tmp.path(),
        "neko",
        "---\n\
title: 猫\n\
type: entity\n\
entity_kind: vocabulary\n\
layer: host\n\
last_updated: 2026-04-28 09:00:00\n\
---\n\n\
# 猫\n\n\
- **Kanji:** 猫\n\
- **Kana:** ねこ\n\
- **Meaning:** cat\n\
- **Part of speech:** noun\n\
- **Mastery:** Introduced\n",
    );

    // The page is on disk.
    let on_disk = tmp.path().join(".dm/wiki/entities/Vocabulary/neko.md");
    assert!(on_disk.exists(), "host cap should write to disk");

    // But Wiki::search is index-driven (`src/wiki/search.rs::search_inner`
    // iterates `idx.entries`), and the index is never updated by the host
    // cap — so the page is invisible to search.
    let hits = wiki.search("cat").expect("search");
    assert!(
        hits.is_empty(),
        "GAP: host-layer vocab page should be findable by search('cat'), \
         got {} hit(s). If this assert flips, update \
         paradigm-gap-host-layer-wiki-ingest.md to RESOLVED.",
        hits.len()
    );

    // Same gap surfaces in stats: index entry count is zero despite the
    // page existing on disk.
    let stats = wiki.stats().expect("stats");
    assert_eq!(
        stats.total_pages, 0,
        "GAP: index reports {} pages but disk has 1 host-layer entity",
        stats.total_pages
    );
}

#[test]
fn host_layer_pages_written_via_write_page_also_skip_index() {
    // Even when callers use the kernel API (`Wiki::write_page`) directly
    // — the path host caps *would* take if they were rewritten — the
    // index is still not updated. `write_page` is a pure file-write; only
    // `ingest_file_internal` and `save_index` mutate the catalog, and
    // `ingest_file_internal` rejects `.dm/wiki/` paths up front. This pins
    // the "we'd need a new API" half of the gap.
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");

    let page = WikiPage {
        title: "猫".to_string(),
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated: "2026-04-28 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "Kana: ねこ. Meaning: cat.".to_string(),
    };
    wiki.write_page("entities/Vocabulary/neko.md", &page)
        .expect("write_page");

    let hits = wiki.search("cat").expect("search");
    assert!(
        hits.is_empty(),
        "GAP: even Wiki::write_page does not update the index. \
         Host caps need an explicit `register_host_page` step or a kernel-side ingest pass."
    );
}
