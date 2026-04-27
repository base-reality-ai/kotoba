//! Wiki lint pipeline — integrity checks over index + page frontmatter.
//!
//! Split from `mod.rs` (Phase 1.1, Cycle 8). Uses split-impl: the `lint`
//! method continues to extend the `Wiki` struct defined in the parent,
//! but its body lives here so the 400-line rule pipeline doesn't bloat
//! `mod.rs`. Mod-private helpers (`entity_page_rel`,
//! `extract_key_exports`, `parse_page_item_names`, `parse_page_timestamp`,
//! `rust_item_regex`) are `pub(super)` re-imports from the parent — they
//! still serve `ingest`/`refresh`/`detect_entity_kind` callsites that
//! remain in `mod.rs` and will migrate as those callsites do.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

use super::{
    entity_page_rel, extract_key_exports, parse_page_item_names, parse_page_timestamp,
    rust_item_regex, PageType, Wiki, WikiLintFinding, WikiLintKind, WikiPage,
};

/// Scan `body` for path-shaped candidates anchored at `src/` and return
/// every `*.rs` candidate found.
///
/// Used by [`Wiki::lint`]'s `BodyPathMissing` rule and by the test-side
/// helper at `src/wiki/tests.rs::collect_phantom_paths`. Centralizing here
/// avoids the manual sync that previously caused a false-positive when
/// only the test helper was updated for nested paths.
///
/// The byte anchor is `src/`, but the scanner backward-extends through any
/// path-shaped prefix so nested layouts like `examples/host-skeleton/src/foo.rs`
/// register as one candidate rather than yielding a phantom `src/foo.rs`
/// substring.
pub(super) fn body_src_path_candidates(body: &str) -> std::collections::BTreeSet<String> {
    let mut paths: std::collections::BTreeSet<String> = Default::default();
    let bytes = body.as_bytes();
    let path_char = |c: u8| -> bool {
        c.is_ascii_alphanumeric() || c == b'/' || c == b'_' || c == b'-' || c == b'.'
    };
    let mut i = 0;
    while i + 4 <= bytes.len() {
        if &bytes[i..i + 4] == b"src/" {
            let mut start = i;
            while start > 0 && path_char(bytes[start - 1]) {
                start -= 1;
            }
            let mut j = i + 4;
            while j < bytes.len() {
                if !path_char(bytes[j]) {
                    break;
                }
                j += 1;
            }
            let candidate = &body[start..j];
            if candidate.ends_with(".rs") {
                paths.insert(candidate.to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    paths
}

impl Wiki {
    /// Static-consistency check over the wiki's current on-disk state. No
    /// LLM calls — just fast, deterministic rules:
    ///
    ///  1. `OrphanIndexEntry` — an index entry points to a page file that
    ///     does not exist on disk.
    ///  2. `UntrackedPage` — a `.md` file exists directly under one of the
    ///     four category subdirs but has no matching entry in `index.md`.
    ///  3. `CategoryMismatch` — an index entry's `category` field does not
    ///     match the category-subdir prefix of its `path`.
    ///  4. `IndexTimestampDrift` — an index entry's cached
    ///     `last_updated` marker disagrees with the linked page's
    ///     frontmatter timestamp. Legacy entries without a cache are
    ///     ignored.
    ///  5. `SourceMissing` / `SourceNewerThanPage` — a source path in a
    ///     page's `sources` frontmatter has been deleted or modified after
    ///     the page's `last_updated` timestamp. Synthesis pages are skipped
    ///     (their `sources` are compaction contributors, not tracked files).
    ///     Sibling rule `BodyPathMissing` covers the same drift class for
    ///     paths mentioned in body text rather than declared in
    ///     `sources:`; paths present in both surface only as
    ///     `SourceMissing` (dedup).
    ///  6. `MalformedPage` — page frontmatter could not be parsed, or
    ///     `last_updated` is not a valid `%Y-%m-%d %H:%M:%S` local-time
    ///     timestamp. Applies to every category. A `MalformedPage` page
    ///     is never checked for source-drift — the finding short-circuits
    ///     the rest of the per-page pass for that entry.
    ///  7. `ItemDrift` — an entity page's `## Items` section names a
    ///     top-level item no longer present in the current source file.
    ///  8. `EntityGap` — a `.rs` file under `<project>/src/` has no
    ///     documenting page (neither canonical nor source-listed).
    ///  9. `MissingEntityKind` — an entity page with `.rs` sources whose
    ///     `entity_kind` frontmatter is `None`. Flags pre-Cycle-38 pages
    ///     that survived the schema addition without a re-ingest.
    /// 10. `ConceptScopeUndocumented` — a Concept page declares `scope:`
    ///     prefixes but one or more `.rs` files under those prefixes is
    ///     not mentioned in the page body. Reverse-direction sibling of
    ///     `BodyPathMissing`; opt-in via the page's `scope:` field.
    ///     `mod.rs` files are skipped per C38 design.
    ///     Scope entries can be either a directory prefix (recursive
    ///     walk; `mod.rs` skipped) or a specific `.rs` file path (per-file
    ///     mention check, no walk).
    ///
    /// A missing/unreadable index yields zero findings (lint is best-effort
    /// and tolerant of partial scaffolds). A missing category directory is
    /// silently skipped — the read-only lint must not assume
    /// `ensure_layout` was called on the same handle.
    ///
    /// Findings are sorted by `(kind, path)` ascending so callers can rely
    /// on deterministic ordering.
    pub fn lint(&self) -> io::Result<Vec<WikiLintFinding>> {
        use std::collections::HashSet;

        let idx = self.load_index().unwrap_or_default();
        let index_paths: HashSet<&str> = idx.entries.iter().map(|e| e.path.as_str()).collect();

        let mut findings: Vec<WikiLintFinding> = Vec::new();

        // Rule 1: index entries pointing at missing files.
        for entry in &idx.entries {
            let abs = self.root.join(&entry.path);
            if !abs.is_file() {
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::OrphanIndexEntry,
                    path: entry.path.clone(),
                    detail: format!(
                        "index lists \"{}\" under {}; file not on disk",
                        entry.title,
                        entry.category.as_str(),
                    ),
                });
            }
        }

        // Rule 2: on-disk .md files under category dirs with no index entry.
        for cat in [
            PageType::Entity,
            PageType::Concept,
            PageType::Summary,
            PageType::Synthesis,
        ] {
            let dir = self.root.join(cat.category_dir());
            let rd = match fs::read_dir(&dir) {
                Ok(rd) => rd,
                Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            for ent in rd.flatten() {
                let Ok(ft) = ent.file_type() else {
                    continue;
                };
                if !ft.is_file() {
                    continue;
                }
                let name = ent.file_name();
                let name = name.to_string_lossy();
                if !name.ends_with(".md") {
                    continue;
                }
                let rel = format!("{}/{}", cat.category_dir(), name);
                if !index_paths.contains(rel.as_str()) {
                    findings.push(WikiLintFinding {
                        kind: WikiLintKind::UntrackedPage,
                        path: rel,
                        detail: format!(
                            "page exists on disk but is not listed in index ({})",
                            cat.as_str(),
                        ),
                    });
                }
            }
        }

        // Rule 3: category field disagrees with path subdir.
        for entry in &idx.entries {
            let expected = format!("{}/", entry.category.category_dir());
            if !entry.path.starts_with(&expected) {
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::CategoryMismatch,
                    path: entry.path.clone(),
                    detail: format!(
                        "category={} but path={} does not match {}",
                        entry.category.as_str(),
                        entry.path,
                        expected,
                    ),
                });
            }
        }

        // Rules 4, 5 & 8: per-page checks. Unified sweep — each index
        // entry whose file is on disk gets read and parsed exactly once,
        // then flows through:
        //   * Rule 5 `MalformedPage` — frontmatter parse fail, or
        //     `last_updated` not a valid timestamp. Universal (all
        //     categories, regardless of `sources`).
        //   * Rule 8 `MissingEntityKind` — entity page with `.rs`
        //     sources but `entity_kind: None`. Piggybacks on this sweep
        //     because the parsed `WikiPage` is already in hand.
        //   * Rule 4 `SourceMissing`/`SourceNewerThanPage` — only non-
        //     synthesis pages with non-empty `sources`. Synthesis is
        //     skipped because its `sources` list compaction contributors,
        //     not tracked source files.
        let proj = self.project_root();
        // Track every source path cited in any page's `sources` frontmatter.
        // A file listed as a source is considered wiki-covered even when the
        // hosting page sits at a non-canonical path (e.g. `entities/main.md`
        // covering `src/main.rs` instead of `entities/src_main_rs.md`). This
        // prevents EntityGap from firing false positives on files that are
        // already documented under a human-chosen page name.
        let mut documented_sources: HashSet<String> = HashSet::new();
        // first_source (.rs only) → list of Entity page paths claiming it.
        // Populated inline by the main entry-parse loop; drained into
        // DuplicateSource findings after ItemDrift, before the final sort.
        let mut by_first_source: HashMap<String, Vec<String>> = HashMap::new();
        for entry in &idx.entries {
            let abs = self.root.join(&entry.path);
            let Ok(text) = fs::read_to_string(&abs) else {
                continue; // Covered by OrphanIndexEntry.
            };
            let Some(page) = WikiPage::parse(&text) else {
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::MalformedPage,
                    path: entry.path.clone(),
                    detail: "page frontmatter could not be parsed".to_string(),
                });
                continue;
            };
            let Some(page_ts) = parse_page_timestamp(&page.last_updated) else {
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::MalformedPage,
                    path: entry.path.clone(),
                    detail: format!(
                        "unparseable last_updated={:?} (expected %Y-%m-%d %H:%M:%S local time)",
                        page.last_updated,
                    ),
                });
                continue;
            };
            if let Some(index_ts) = &entry.last_updated {
                if index_ts != &page.last_updated {
                    findings.push(WikiLintFinding {
                        kind: WikiLintKind::IndexTimestampDrift,
                        path: entry.path.clone(),
                        detail: format!(
                            "index cached last_updated={} but page frontmatter has {}. Try: update the index entry's `<!--updated:...-->` marker to match the page.",
                            index_ts, page.last_updated
                        ),
                    });
                }
            }
            for src in &page.sources {
                documented_sources.insert(src.clone());
            }
            if entry.category == PageType::Entity
                && page.entity_kind.is_none()
                && page.sources.iter().any(|s| s.ends_with(".rs"))
            {
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::MissingEntityKind,
                    path: entry.path.clone(),
                    detail: "entity_kind missing; re-ingest to populate".to_string(),
                });
            }
            if entry.category == PageType::Entity {
                if let Some(first) = page.sources.first() {
                    if first.ends_with(".rs") {
                        by_first_source
                            .entry(first.clone())
                            .or_default()
                            .push(entry.path.clone());
                    }
                }
            }
            // Rule 4c: ConceptScopeUndocumented — for Concept pages with a
            // non-empty `scope:` declaration, walk each scope prefix and emit a
            // finding for any `.rs` file (excluding `mod.rs`) that the body
            // doesn't mention. Reverse-direction sibling of `BodyPathMissing`.
            // Runs BEFORE the Synthesis/empty-sources early-continue below
            // because a Concept page with `scope:` set may legitimately have
            // empty `sources:` — the scope declaration is the documentary claim.
            if entry.category == PageType::Concept && !page.scope.is_empty() {
                for scope_prefix in &page.scope {
                    let scope_root = proj.join(scope_prefix);
                    // File-level scope: a scope entry ending in `.rs` and
                    // resolving to an existing file is a per-file claim.
                    // Assert the file is mentioned in the page body.
                    // Mirrors the directory-walk's body-mention check but
                    // doesn't recurse.
                    if scope_root.is_file() && scope_prefix.ends_with(".rs") {
                        let basename = scope_prefix.rsplit('/').next().unwrap_or(scope_prefix);
                        let stem = basename.strip_suffix(".rs").unwrap_or(basename);
                        let mentioned = page.body.contains(scope_prefix)
                            || page.body.contains(basename)
                            || page.body.contains(stem);
                        if !mentioned {
                            findings.push(WikiLintFinding {
                                kind: WikiLintKind::ConceptScopeUndocumented,
                                path: entry.path.clone(),
                                detail: format!(
                                    "scope '{}' is a specific file but the page does not mention it",
                                    scope_prefix,
                                ),
                            });
                        }
                        continue; // file-level branch handled; skip the directory walk
                    }
                    if !scope_root.is_dir() {
                        continue; // scope points at non-existent dir; skip silently
                    }
                    let mut stack: Vec<PathBuf> = vec![scope_root];
                    while let Some(dir) = stack.pop() {
                        let Ok(rd) = fs::read_dir(&dir) else {
                            continue;
                        };
                        for ent in rd.flatten() {
                            let path = ent.path();
                            let Ok(ft) = ent.file_type() else { continue };
                            if ft.is_dir() {
                                let skip = path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .is_some_and(|n| n.starts_with('.') || n == "target");
                                if !skip {
                                    stack.push(path);
                                }
                                continue;
                            }
                            if !ft.is_file() {
                                continue;
                            }
                            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                                continue;
                            }
                            if path.file_name().and_then(|n| n.to_str()) == Some("mod.rs") {
                                continue;
                            }
                            let Ok(rel) = path.strip_prefix(&proj) else {
                                continue;
                            };
                            let rel_str = rel.to_string_lossy().replace('\\', "/");
                            let basename = rel_str.rsplit('/').next().unwrap_or(&rel_str);
                            let stem = basename.strip_suffix(".rs").unwrap_or(basename);
                            let mentioned = page.body.contains(&rel_str)
                                || page.body.contains(basename)
                                || page.body.contains(stem);
                            if !mentioned {
                                findings.push(WikiLintFinding {
                                    kind: WikiLintKind::ConceptScopeUndocumented,
                                    path: entry.path.clone(),
                                    detail: format!(
                                        "scope '{}' includes {} which the page does not mention",
                                        scope_prefix, rel_str,
                                    ),
                                });
                            }
                        }
                    }
                }
            }

            if entry.category == PageType::Synthesis || page.sources.is_empty() {
                continue;
            }
            for src in &page.sources {
                let src_abs = proj.join(src);
                match fs::metadata(&src_abs) {
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {
                        findings.push(WikiLintFinding {
                            kind: WikiLintKind::SourceMissing,
                            path: entry.path.clone(),
                            detail: format!(
                                "source {} listed on page no longer exists under project root",
                                src,
                            ),
                        });
                    }
                    Err(_) => {}
                    Ok(md) => {
                        if let Ok(mtime) = md.modified() {
                            if mtime > page_ts {
                                findings.push(WikiLintFinding {
                                    kind: WikiLintKind::SourceNewerThanPage,
                                    path: entry.path.clone(),
                                    detail: format!(
                                        "source {} modified after page last_updated={}",
                                        src, page.last_updated,
                                    ),
                                });
                            }
                        }
                    }
                }
            }

            // Rule 4b: BodyPathMissing — paths mentioned in body text that
            // don't exist on disk. Skips paths already in `page.sources`
            // (those are SourceMissing's territory). Scan logic shared with
            // `src/wiki/tests.rs::collect_phantom_paths` via
            // [`body_src_path_candidates`].
            {
                let sources_set: HashSet<&str> = page.sources.iter().map(|s| s.as_str()).collect();
                for p in body_src_path_candidates(&page.body) {
                    if sources_set.contains(p.as_str()) {
                        continue;
                    }
                    if !proj.join(&p).is_file() {
                        findings.push(WikiLintFinding {
                            kind: WikiLintKind::BodyPathMissing,
                            path: entry.path.clone(),
                            detail: format!(
                                "body text mentions {} which does not exist under project root",
                                p,
                            ),
                        });
                    }
                }
            }
        }

        // Rule 6: EntityGap — `.rs` files under `<project_root>/src/` that
        // neither have a canonical entity page at `entity_page_rel(rel)` nor
        // are listed in any page's `sources` frontmatter. MVP scope: `src/`
        // with `.rs` only; `.toml`/`.md`/other languages and other roots
        // (`lib/`, `tests/`) are deferred until operators show they need it.
        let src_root = proj.join("src");
        if src_root.is_dir() {
            let mut stack: Vec<PathBuf> = vec![src_root];
            while let Some(dir) = stack.pop() {
                let Ok(rd) = fs::read_dir(&dir) else {
                    continue;
                };
                for entry in rd.flatten() {
                    let path = entry.path();
                    let Ok(ft) = entry.file_type() else {
                        continue;
                    };
                    if ft.is_dir() {
                        // Skip hidden directories (e.g. `.git`, `.cache`)
                        // and `target/` even though neither belongs under
                        // `src/` in well-formed projects — cheap defense
                        // against surprise repo layouts.
                        let skip = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .is_some_and(|n| n.starts_with('.') || n == "target");
                        if !skip {
                            stack.push(path);
                        }
                        continue;
                    }
                    if !ft.is_file() {
                        continue;
                    }
                    if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                        continue;
                    }
                    let Ok(rel) = path.strip_prefix(&proj) else {
                        continue;
                    };
                    let rel_str = rel.to_string_lossy().replace('\\', "/");
                    let canonical_page = entity_page_rel(&rel_str);
                    let canonical_abs = self.root.join(&canonical_page);
                    if canonical_abs.is_file() {
                        continue;
                    }
                    if documented_sources.contains(&rel_str) {
                        continue;
                    }
                    findings.push(WikiLintFinding {
                        kind: WikiLintKind::EntityGap,
                        path: rel_str,
                        detail: format!("no entity page at `{}`", canonical_page),
                    });
                }
            }
        }

        // Rule 7: ItemDrift — for each entity page whose source exists and
        // is Rust, diff the page's `## Items` names against a fresh
        // regex scan of the source. Names in the page but not the source
        // → symbolic drift. Sibling to Rule 5 (SourceNewerThanPage):
        // temporal drift says "mtime later," symbolic drift says "names
        // disagree." Both can fire together.
        for entry in &idx.entries {
            if entry.category != PageType::Entity {
                continue;
            }
            let abs = self.root.join(&entry.path);
            let Ok(text) = fs::read_to_string(&abs) else {
                continue; // OrphanIndexEntry covers this.
            };
            let Some(page) = WikiPage::parse(&text) else {
                continue; // MalformedPage already flagged in the Rule 4/5 loop.
            };
            let Some(first_src) = page.sources.first() else {
                continue;
            };
            if !first_src.ends_with(".rs") {
                continue;
            }
            let src_abs = proj.join(first_src);
            let Ok(src_content) = fs::read_to_string(&src_abs) else {
                continue; // SourceMissing already flagged — don't double-report.
            };

            // Rule 7b: ExportDrift — diff page.key_exports names against
            // the current source's extracted exports. Fires independently
            // of Rule 7 (ItemDrift); a page can drift in one or both
            // places. Name-only comparison for v1.
            if !page.key_exports.is_empty() {
                let current: HashSet<String> = extract_key_exports(first_src, &src_content)
                    .into_iter()
                    .map(|e| e.name)
                    .collect();
                let mut drifted: Vec<String> = page
                    .key_exports
                    .iter()
                    .map(|e| e.name.clone())
                    .filter(|n| !current.contains(n))
                    .collect();
                if !drifted.is_empty() {
                    drifted.sort();
                    drifted.dedup();
                    // 200-byte cap on the symbol list — mirrors ItemDrift's
                    // pattern below. Next-step suffix sits outside the cap
                    // so overflow cannot cost the actionable hint.
                    let prefix = "key_exports in frontmatter not in source: ";
                    let mut detail = String::from(prefix);
                    let mut first_sym = true;
                    let mut capped = false;
                    for name in &drifted {
                        let sep = if first_sym { "" } else { ", " };
                        let next_len = detail.len() + sep.len() + name.len();
                        if next_len > 200usize.saturating_sub(5) {
                            capped = true;
                            break;
                        }
                        detail.push_str(sep);
                        detail.push_str(name);
                        first_sym = false;
                    }
                    if capped {
                        detail.push_str(", …");
                    }
                    detail.push_str("; re-ingest to reconcile");
                    findings.push(WikiLintFinding {
                        kind: WikiLintKind::ExportDrift,
                        path: entry.path.clone(),
                        detail,
                    });
                }
            }

            let page_items = parse_page_item_names(&page.body);
            if page_items.is_empty() {
                continue; // Pre-Cycle-35 stub page; no drift to check.
            }
            let mut source_items: HashSet<String> = HashSet::new();
            for cap in rust_item_regex().captures_iter(&src_content) {
                if let Some(n) = cap.name("name") {
                    source_items.insert(n.as_str().to_string());
                }
            }
            let mut drifted: Vec<String> = page_items.difference(&source_items).cloned().collect();
            if drifted.is_empty() {
                continue;
            }
            drifted.sort();
            // Cap the detail at 200 bytes by trimming the joined list and
            // appending `, …` once we'd exceed. Never truncate mid-name.
            let prefix = "items in page not in source: ";
            let mut detail = String::from(prefix);
            let mut first = true;
            let mut capped = false;
            for name in &drifted {
                let sep = if first { "" } else { ", " };
                let next_len = detail.len() + sep.len() + name.len();
                // Leave 5 bytes of headroom for the `, …` marker.
                if next_len > 200usize.saturating_sub(5) {
                    capped = true;
                    break;
                }
                detail.push_str(sep);
                detail.push_str(name);
                first = false;
            }
            if capped {
                detail.push_str(", …");
            }
            findings.push(WikiLintFinding {
                kind: WikiLintKind::ItemDrift,
                path: entry.path.clone(),
                detail,
            });
        }

        // Rule 9: DuplicateSource — any first-source key with ≥ 2 Entity
        // pages claiming it is a page-to-page contradiction. Emits one
        // finding per claimant page; `others` lists the competing pages so
        // the operator can jump directly to the conflict. HashMap iteration
        // order is nondeterministic, but the final `findings.sort_by`
        // below orders by `(kind as u8, path)` so output is deterministic.
        for (source, mut pages) in by_first_source {
            if pages.len() < 2 {
                continue;
            }
            pages.sort();
            for (i, page_path) in pages.iter().enumerate() {
                let others: Vec<&str> = pages
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, p)| p.as_str())
                    .collect();
                let detail = format!(
                    "source `{}` also documented by: {}; delete the stale page \
                     and run `/wiki refresh` on the remaining one to reconcile",
                    source,
                    others.join(", "),
                );
                findings.push(WikiLintFinding {
                    kind: WikiLintKind::DuplicateSource,
                    path: page_path.clone(),
                    detail,
                });
            }
        }

        findings.sort_by(|a, b| {
            (a.kind as u8)
                .cmp(&(b.kind as u8))
                .then(a.path.cmp(&b.path))
        });
        Ok(findings)
    }
}
