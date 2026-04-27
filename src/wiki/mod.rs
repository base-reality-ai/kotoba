//! Wiki — persistent project knowledge layer.
//!
//! Implements the scaffold for Pillar 3 of the current directive (LLM Wiki).
//! The wiki lives at `<project>/.dm/wiki/` and consists of:
//!
//! - `index.md`            — catalog of every page, grouped by category.
//! - `schema.md`           — conventions for page format.
//! - `log.md`              — append-only activity log.
//! - `entities/`           — pages for concrete things (files, functions).
//! - `concepts/`           — pages for abstract things (patterns, decisions).
//! - `summaries/`          — directory and source roll-ups.
//! - `synthesis/`          — cross-cutting analysis.
//!
//! This increment delivers only the scaffold: types, layout creation, seeding,
//! and frontmatter round-tripping. Auto-ingest, compact-to-wiki, and slash
//! commands land in later cycles.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

mod types;
pub use types::{
    ConceptPagesReport, CycleNodeSnapshot, EntityKind, IndexEntry, IngestOutcome, KeyExport, Layer,
    MomentumReport, PageType, ProjectSummaryReport, SkipReason, WikiIndex, WikiLintFinding,
    WikiLintKind, WikiLog, WikiPage, WikiRefreshReport, WikiSearchHit, WikiSeedReport, WikiStats,
};

mod summary;
pub use summary::PlannerBrief;
mod compact;
pub(crate) use compact::DEFAULT_COMPACT_KEEP;
pub mod fuzzy;
mod index;
mod ingest;
mod lint;
mod search;

/// Byte cap for the model-facing context snippet. Keeps the system prompt
/// from ballooning on projects with large wikis.
pub(super) const CONTEXT_SNIPPET_MAX_BYTES: usize = 4096;

/// Minimum entity-page consumers a dependency must have before a
/// `concepts/dep-*.md` page is auto-generated. The ≥ 3 threshold keeps
/// one-offs (file-specific imports) from polluting the concept layer
/// while still surfacing genuinely shared idioms early.
pub(super) const CONCEPT_DEP_MIN_OCCURRENCES: usize = 3;

/// Default count-window for [`Wiki::momentum`] — how many of the most
/// recent `ingest` entries to aggregate. Count-based (not time-based) so
/// results are deterministic across wall-clock drift in tests.
pub const MOMENTUM_DEFAULT_WINDOW: usize = 100;

/// How many top paths / modules [`Wiki::momentum`] surfaces. Capped low
/// so the `## Momentum` section and `/wiki momentum` output stay glanceable.
pub const MOMENTUM_TOP_N: usize = 5;

/// Top-N hot paths surfaced by [`Wiki::planner_brief`]. Narrower than
/// [`MOMENTUM_TOP_N`] would suggest because the planner brief is appended
/// to the system prompt, where every character competes with the rest of
/// the context budget.
pub const PLANNER_BRIEF_HOT_PATHS: usize = 5;

/// Cap on drifting-page entries surfaced by [`Wiki::planner_brief`]. Same
/// budget rationale as [`PLANNER_BRIEF_HOT_PATHS`] — rank-truncated to
/// keep `sys_prompt` lean.
pub const PLANNER_BRIEF_MAX_DRIFT: usize = 5;

/// Upper cap on synthesis cycle-pages surfaced in
/// [`PlannerBrief::recent_cycles`]. Small by design — the planner only
/// needs the last handful for drift-awareness; deeper history lives on
/// disk for `file_read` lookup.
pub const PLANNER_BRIEF_RECENT_CYCLES: usize = 3;

/// Upper cap on synthesis cycle-pages surfaced in the `## Recent cycles`
/// section of [`Wiki::build_project_summary`]. Wider than
/// [`PLANNER_BRIEF_RECENT_CYCLES`] because the project summary is a
/// reference view humans read on `summaries/project.md`, not a budget-
/// constrained system-prompt injection.
pub const PROJECT_SUMMARY_RECENT_CYCLES: usize = 5;

/// Shared path prefix for chain-cycle synthesis pages. Both the
/// filename producer ([`Wiki::write_cycle_synthesis`]) and the parser
/// (`parse_cycle_number_from_synthesis_path`) reference this constant
/// so a future format change cannot silently break the round-trip.
pub const SYNTHESIS_CYCLE_PREFIX: &str = "synthesis/cycle-";

/// Character budget passed to [`PlannerBrief::render`] for the default
/// planner-role `sys_prompt` injection. Truncation is line-boundary aware
/// so the planner never receives half a bullet.
pub const PLANNER_BRIEF_BUDGET_CHARS: usize = 2048;

/// Hours after which the newest `RecentCycle.last_updated` renders with
/// a staleness warning. Tuned for typical incubation cadence — a chain
/// that goes >24h without a synthesis write is either paused or stuck,
/// and the planner should surface that to itself before planning.
pub const PLANNER_BRIEF_STALE_HOURS: i64 = 24;

/// Upper cap on entity/concept pages surfaced in
/// [`PlannerBrief::fresh_pages`]. Same lean-sys_prompt rationale as
/// [`PLANNER_BRIEF_HOT_PATHS`] — the planner gets the newest handful;
/// deeper history is a `file_read` away.
pub const PLANNER_BRIEF_FRESH_PAGES: usize = 5;

/// Age window for [`PlannerBrief::fresh_pages`] — a page counts as
/// "fresh" when its `last_updated` frontmatter is within this many hours
/// of composition time. Matches [`PLANNER_BRIEF_STALE_HOURS`] so a cycle
/// that's fresh-enough-to-not-warn also surfaces the pages it likely
/// touched.
pub const PLANNER_BRIEF_FRESH_HOURS: i64 = 24;

/// Byte cap for the `\<wiki_fresh\>` block emitted by
/// [`Wiki::fresh_pages_snippet`] into the session-start system prompt.
/// Sized to hold the top handful of entity/concept pages as
/// `- {title} — {one_liner}` lines without pushing the total prompt
/// past the 4096-char wiki-summary + 6144-char wiki-index budgets.
pub const FRESH_PAGES_BUDGET_CHARS: usize = 1024;

/// Per-node output cap in [`Wiki::write_cycle_synthesis`]. Bullets over
/// the cap are truncated at the nearest line boundary with a
/// `[...truncated]` marker appended. v1 keeps the body small so the
/// synthesis page stays glanceable and later cycles don't drown the
/// index under multi-MB outputs.
pub const CYCLE_SYNTHESIS_OUTPUT_PER_NODE: usize = 600;

/// Log verb for chain-cycle synthesis entries. Deliberately distinct
/// from `"ingest"` so [`Wiki::momentum`]'s ingest-only filter ignores
/// these — cycle-synthesis writes are meta-history, not source churn,
/// and mixing them into `hot_paths` would corrupt the momentum signal.
pub const CYCLE_SYNTHESIS_LOG_VERB: &str = "chain";

/// Backslash-escape characters that would otherwise terminate a link
/// destination in `[title](path)`. Only `\` and `)` are escaped — `(` is
/// legal inside a destination and parses fine.
///
/// Escape order is fixed (`\` first, then `)`) so the backslash introduced
/// by the second pass is not re-escaped by the first.
pub(super) fn escape_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str(r"\\"),
            ')' => out.push_str(r"\)"),
            c => out.push(c),
        }
    }
    out
}

/// Consume `s` up to the first unescaped `)`. Returns the decoded path and
/// the remainder after the `)`. Returns `None` if no terminator is found
/// (caller should skip the line, matching existing parse behaviour).
///
/// Recognizes `\\` → `\` and `\)` → `)`. Unknown escape sequences (e.g.
/// `\U`) are passed through verbatim so hand-edited paths like `C:\Users\…`
/// do not lose data.
pub(super) fn scan_path(s: &str) -> Option<(String, &str)> {
    let mut path = String::with_capacity(s.len());
    let mut it = s.char_indices().peekable();
    while let Some((i, ch)) = it.next() {
        if ch == '\\' {
            if let Some(&(_, next)) = it.peek() {
                match next {
                    '\\' => {
                        path.push('\\');
                        it.next();
                        continue;
                    }
                    ')' => {
                        path.push(')');
                        it.next();
                        continue;
                    }
                    _ => {
                        path.push('\\');
                        continue; // next iteration pushes `next` verbatim
                    }
                }
            }
        }
        if ch == ')' {
            return Some((path, &s[i + ch.len_utf8()..]));
        }
        path.push(ch);
    }
    None
}

/// In-process content-hash cache for ingest dedup. Keyed on the canonical
/// path string; value is the last ingested content's hash. Process-local —
/// persistent cache deferred.
static INGEST_CACHE: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Serializes all `load_index → mutate → save_index` sequences in this
/// process. The critical section is small (read + parse + mutate + write
/// of `index.md`), so contention is cheap; correctness dominates. Held
/// via `unwrap_or_else(|e| e.into_inner())` so a poisoned lock from a
/// prior panic doesn't wedge future callers.
pub(super) static INDEX_LOCK: Mutex<()> = Mutex::new(());

pub(super) fn ingest_cache_check_and_update(key: &str, hash: u64) -> bool {
    let mut guard = INGEST_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    match map.get(key) {
        Some(&prev) if prev == hash => false,
        _ => {
            map.insert(key.to_string(), hash);
            true
        }
    }
}

/// Test-only: reset the cache between tests so dedup state doesn't leak.
#[cfg(test)]
pub(crate) fn reset_ingest_cache_for_tests() {
    let mut guard = INGEST_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// Non-cryptographic hash of file content — used only for change detection
/// in the dedup cache.
pub(super) fn content_hash(content: &str) -> u64 {
    use std::hash::{DefaultHasher, Hasher};
    let mut h = DefaultHasher::new();
    h.write(content.as_bytes());
    h.finish()
}

/// `DM_WIKI_AUTO_INGEST=0|false|off` disables the auto-ingest hook without
/// a rebuild. Any other value (or unset) leaves ingest enabled.
pub(super) fn auto_ingest_enabled() -> bool {
    !matches!(
        std::env::var("DM_WIKI_AUTO_INGEST").as_deref(),
        Ok("0" | "false" | "off")
    )
}

/// Turn a project-relative path into a flat, safe entity-page filename under
/// `entities/`. `src/foo/bar.rs` → `entities/src_foo_bar_rs.md`. Every
/// non-alphanumeric character becomes `_`; runs collapse; leading/trailing
/// `_` trimmed; empty slug falls back to `unnamed`.
pub(super) fn entity_page_rel(project_rel: &str) -> String {
    let mut slug = String::with_capacity(project_rel.len() + 4);
    let mut prev_underscore = false;
    for ch in project_rel.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            slug.push('_');
            prev_underscore = true;
        }
    }
    let slug = slug.trim_matches('_');
    format!(
        "entities/{}.md",
        if slug.is_empty() { "unnamed" } else { slug }
    )
}

/// Scan `text` for substring matches of any known entity-page source path,
/// returning the matched entity-page paths (relative to `.dm/wiki/`).
///
/// `entity_srcs` is an iterator of `(page_path, source_path)` pairs drawn
/// from entity pages' frontmatter. `self_source` — if Some — is excluded
/// from matches so a source file's own text mentioning itself does not
/// inflate its own inbound count.
///
/// Matching is plain substring — Rust source rarely contains literal
/// file-path strings outside comments/tests, and false positives are
/// tolerable for a count-only metric. Precise AST-based resolution is an
/// LSP-scale follow-up (see Cycle 33 deferred list).
///
/// Result is sorted and deduplicated so callers get a stable iteration
/// order regardless of `entity_srcs` input order.
pub(crate) fn wiki_link_scan<'a, I>(
    text: &str,
    entity_srcs: I,
    self_source: Option<&str>,
) -> Vec<String>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut matched: Vec<String> = entity_srcs
        .into_iter()
        .filter(|(_, src)| Some(*src) != self_source && text.contains(src))
        .map(|(page, _)| page.to_string())
        .collect();
    matched.sort();
    matched.dedup();
    matched
}

/// Rewrite `body` by wrapping the first occurrence of each other-entity
/// source path with `[[…]]` wiki-link syntax. `entity_srcs` is the same
/// `(page_path, source_path)` shape as [`wiki_link_scan`]. `self_source`
/// is never wrapped — a page describing `src/x.rs` should not link to
/// itself via its own `Source file:` heading mention.
///
/// Already-wrapped occurrences (immediately preceded by `[[`) are left
/// alone so re-ingest is idempotent — running this twice never produces
/// `[[[[src/a.rs]]]]`.
///
/// Only the first occurrence per source path is wrapped; subsequent
/// mentions stay plain to avoid visual clutter.
pub(crate) fn inject_wiki_links(
    body: &str,
    entity_srcs: &[(String, String)],
    self_source: &str,
) -> String {
    let mut out = body.to_string();
    for (_page_path, src) in entity_srcs {
        if src == self_source || src.is_empty() {
            continue;
        }
        let Some(idx) = out.find(src.as_str()) else {
            continue;
        };
        // Idempotence: if the match is already inside `[[…]]`, leave it.
        // `ends_with` is UTF-8-safe when `idx` lands just after a multi-byte
        // char (slicing at `idx - 2` would panic mid-codepoint).
        if out[..idx].ends_with("[[") {
            continue;
        }
        let wrapped = format!("[[{}]]", src);
        out.replace_range(idx..idx + src.len(), &wrapped);
    }
    out
}

/// Max bytes the extracted preview section may add to an entity page body.
/// Keeps entity pages bounded so an ingest of a 100KB source file doesn't
/// inflate the wiki — and the context snippet, which lists page metadata
/// but not body, is unaffected either way.
pub(crate) const PREVIEW_MAX_BYTES: usize = 4000;

/// Max lines of generic (non-Rust) file preview. Rust preview uses its
/// own item cap (a regex pass over every line); the generic path just
/// shows the first N lines of the file verbatim.
const PREVIEW_GENERIC_MAX_LINES: usize = 30;

/// Lazily-compiled regex for top-level Rust items. Anchored at column 0
/// (no leading whitespace) so methods inside `impl { … }` blocks — which
/// are always indented — are skipped. Optional `pub` / `pub(crate)`
/// visibility prefix. Captures `vis`, `kind`, `name`.
///
/// Known gap: inline `mod foo { pub fn bar() {} }` loses `bar` because
/// the nested item is indented. dm itself splits into separate files,
/// so this costs little; a future `syn`-backed path would fix it.
static RUST_ITEM_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

pub(super) fn rust_item_regex() -> &'static regex::Regex {
    RUST_ITEM_RE.get_or_init(|| {
        regex::Regex::new(
            r"(?m)^(?P<vis>pub(?:\([^)]+\))?[[:space:]]+)?(?:async[[:space:]]+)?(?:unsafe[[:space:]]+)?(?P<kind>fn|struct|enum|trait|const|static|type|mod|impl)[[:space:]]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
        )
        .expect("rust item regex is a valid pattern")
    })
}

static RUST_USE_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();

/// Match a Rust `use` statement at the start of a line, optionally prefixed
/// by `pub` / `pub(crate)` / `pub(super)`. The captured group is the path
/// expression between `use ` and the terminating `;` — may contain newlines
/// when a `use foo::{ ... };` group spans multiple lines. The caller
/// collapses whitespace after capture.
///
/// Known gaps (same accepted trade-offs as `rust_item_regex`): matches
/// `use foo;` appearing as string-literal content starting at column 1;
/// misses `use` statements indented inside `mod foo { ... }` blocks.
/// A future `syn`-backed pass would fix both at once.
pub(super) fn rust_use_regex() -> &'static regex::Regex {
    RUST_USE_RE.get_or_init(|| {
        regex::Regex::new(r"(?m)^(?:pub(?:\([^)]+\))?[[:space:]]+)?use[[:space:]]+([^;]+);")
            .expect("rust use regex is a valid pattern")
    })
}

/// Extract top-level item names from a wiki entity page body by locating
/// the `## Items` section and parsing its bullet list. Returns an empty
/// set if the section is absent or unparseable.
///
/// Each recognized bullet looks like `` - `\<vis\>? \<kind\> \<name\>` `` (where
/// `\<kind\>` ∈ {`fn`, `struct`, `enum`, `trait`, `const`, `static`,
/// `type`, `mod`, `impl`} and `\<vis\>` is optional — matching the form
/// written by [`extract_rust_preview`]). Walking stops at the first
/// non-bullet line. The "(+N more; preview capped)" truncation marker
/// is recognized and skipped — the names behind the marker are unknown,
/// so the caller must treat the returned set as a lower bound.
pub(crate) fn parse_page_item_names(body: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let Some(items_at) = body.find("## Items\n") else {
        return out;
    };
    let after = &body[items_at + "## Items\n".len()..];
    for line in after.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !trimmed.starts_with("- ") {
            // Any non-bullet line (e.g. next `## Heading`, blank was
            // skipped above) ends the section.
            break;
        }
        if trimmed.starts_with("- *(") {
            // Truncation marker — skip, don't treat as an item.
            continue;
        }
        // Bullet body lives between the two backticks.
        let Some(open) = trimmed.find('`') else {
            continue;
        };
        let rest = &trimmed[open + 1..];
        let Some(close_rel) = rest.find('`') else {
            continue;
        };
        let inner = &rest[..close_rel];
        // Split on whitespace; the name is the last ident-shaped token.
        let mut name: Option<&str> = None;
        for tok in inner.split_whitespace() {
            let candidate = tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if candidate && !tok.is_empty() {
                let first = tok
                    .chars()
                    .next()
                    .expect("tok is non-empty (checked above)");
                if first.is_ascii_alphabetic() || first == '_' {
                    name = Some(tok);
                }
            }
        }
        if let Some(n) = name {
            out.insert(n.to_string());
        }
    }
    out
}

/// MVP content extraction for an ingested file. Regex-based (no `syn`
/// dep — Phase 3 MVP scope): Rust files get module `//!` docs + a
/// bulleted listing of top-level items; everything else gets a
/// first-N-lines code-fenced preview. Output is bounded by
/// [`PREVIEW_MAX_BYTES`] so entity pages stay small.
///
/// Empty/whitespace-only content returns an empty string — the caller
/// is expected to render a placeholder, not splice an empty Overview
/// heading with nothing under it.
pub(crate) fn extract_content_preview(rel_path: &str, content: &str) -> String {
    if content.trim().is_empty() {
        return String::new();
    }
    if rel_path.ends_with(".rs") {
        extract_rust_preview(content)
    } else {
        extract_generic_preview(content)
    }
}

/// Map a Rust item keyword (as captured by `rust_item_regex`'s `kind`
/// group) to the corresponding `EntityKind`. Returns `None` for kinds
/// not represented in the schema today (const/static/type/mod/impl).
///
/// Shared by `detect_entity_kind` (whole-file structural identity) and
/// `extract_key_exports` (per-item public-API list) so the two views of
/// the same file can't disagree on what counts as a known kind.
pub(super) fn entity_kind_from_rust_keyword(kw: &str) -> Option<EntityKind> {
    match kw {
        "fn" => Some(EntityKind::Function),
        "struct" => Some(EntityKind::Struct),
        "enum" => Some(EntityKind::Enum),
        "trait" => Some(EntityKind::Trait),
        _ => None,
    }
}

/// Extract the public-API exports of a Rust source file for
/// `WikiPage.key_exports`. Returns an empty `Vec` for non-Rust files,
/// files with no public items, or files whose only public items are
/// kinds not in `EntityKind` (const/static/type/mod/impl).
///
/// Order: as encountered in source (regex iteration order). Within a
/// single regex-engine version two ingests of the same content produce
/// the same order; callers should not otherwise rely on ordering.
///
/// Deliberate divergence from `detect_entity_kind`: this function skips
/// private items (no `vis` capture → not an export), whereas
/// `detect_entity_kind` counts all items because "what does this file
/// contain" is a visibility-independent question. A future `syn`-backed
/// pass will tighten both.
pub(super) fn extract_key_exports(rel_path: &str, content: &str) -> Vec<KeyExport> {
    if !rel_path.ends_with(".rs") {
        return Vec::new();
    }
    let re = rust_item_regex();
    let mut out = Vec::new();
    for cap in re.captures_iter(content) {
        if cap.name("vis").is_none() {
            continue;
        }
        let kind_str = cap.name("kind").map_or("", |m| m.as_str());
        let Some(kind) = entity_kind_from_rust_keyword(kind_str) else {
            continue;
        };
        let Some(name) = cap.name("name").map(|m| m.as_str()) else {
            continue;
        };
        out.push(KeyExport {
            kind,
            name: name.to_string(),
        });
    }
    out
}

/// Extract the `use` dependencies of a Rust source file for
/// `WikiPage.dependencies`. Returns an empty `Vec` for non-Rust files
/// or files with no `use` statements.
///
/// Preserves source order (first-seen). Includes `pub use` — a re-export
/// is still a dependency in the sense that this file pulls the target in.
/// Drops trailing `as X` renames on top-level paths (the wiki cares about
/// the import target, not the local alias). Multi-line
/// Map a dependency string to a filename-safe slug for
/// `concepts/dep-*.md` pages. Returns `None` for grouped imports
/// (containing `{`) — those are file-specific composition shapes, not
/// shared concepts — and for strings that reduce to empty after
/// non-alnum collapse.
///
/// `std::sync::Arc` → `Some("std_sync_Arc")`
/// `foo::{bar, baz}` → `None`
/// `::leading` → `Some("leading")` (leading/trailing `_` trimmed)
pub(super) fn sanitize_dep_for_path(dep: &str) -> Option<String> {
    if dep.contains('{') {
        return None;
    }
    let mut out = String::with_capacity(dep.len());
    let mut last_underscore = false;
    for ch in dep.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Strip timestamp lines from a concept-page body for write-avoidance
/// comparison. Matches the frontmatter field (`last_updated:`) and the
/// body heading line (`Last updated:`); leading whitespace is tolerated
/// so indented emissions still match.
pub(super) fn concept_body_sans_timestamp(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let t = line.trim_start();
            !(t.starts_with("last_updated:") || t.starts_with("Last updated:"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collect the run of `//!` module-doc lines at the top of a Rust source
/// file. Each returned string has the `//!` prefix stripped and leading
/// whitespace trimmed.
///
/// Semantics: leading blank lines and a shebang (`#!`) are skipped.
/// Once the first `//!` line is seen, the next blank or non-doc line
/// ends collection. A blank `//!` line (just `//!` with nothing after)
/// is kept as an empty `""` in the returned vec so callers can detect
/// paragraph boundaries inside the doc block.
pub(super) fn collect_module_docs(content: &str) -> Vec<&str> {
    let mut docs: Vec<&str> = Vec::new();
    for line in content.lines() {
        let t = line.trim_start();
        if t.is_empty() || t.starts_with("#!") {
            if docs.is_empty() {
                continue;
            }
            break;
        }
        if let Some(rest) = t.strip_prefix("//!") {
            docs.push(rest.trim_start());
        } else {
            break;
        }
    }
    docs
}

fn extract_rust_preview(content: &str) -> String {
    let mut out = String::new();

    // Module-level `//!` docs: consecutive run from top (skipping blank
    // lines and shebang). Stop at first non-doc, non-blank line.
    let docs = collect_module_docs(content);
    if !docs.is_empty() {
        out.push_str("## Module docs\n\n");
        for d in &docs {
            if out.len() + d.len() + 1 > PREVIEW_MAX_BYTES / 2 {
                out.push_str("…\n");
                break;
            }
            out.push_str(d);
            out.push('\n');
        }
        out.push('\n');
    }

    // Top-level items via regex. `impl Foo` has no `name` match for
    // items like `impl Foo for Bar` but we just list the first
    // identifier after `impl` — that's the concrete type, which is
    // what a reader wants to scan.
    let re = rust_item_regex();
    let mut items: Vec<String> = Vec::new();
    for cap in re.captures_iter(content) {
        let vis = cap.name("vis").map_or("", |m| m.as_str().trim());
        let kind = cap.name("kind").map_or("", |m| m.as_str());
        let name = cap.name("name").map_or("", |m| m.as_str());
        if kind.is_empty() || name.is_empty() {
            continue;
        }
        let line = if vis.is_empty() {
            format!("- `{} {}`", kind, name)
        } else {
            format!("- `{} {} {}`", vis, kind, name)
        };
        items.push(line);
    }
    if !items.is_empty() {
        out.push_str("## Items\n\n");
        let mut truncated = 0usize;
        for item in items {
            if out.len() + item.len() + 1 > PREVIEW_MAX_BYTES {
                truncated += 1;
                continue;
            }
            out.push_str(&item);
            out.push('\n');
        }
        if truncated > 0 {
            writeln!(out, "- *(+{} more; preview capped)*", truncated)
                .expect("write to String never fails");
        }
    }
    out
}

fn extract_generic_preview(content: &str) -> String {
    let mut out = String::from("## Preview\n\n```\n");
    let mut truncated = false;
    for (lines_shown, line) in content.lines().enumerate() {
        if lines_shown >= PREVIEW_GENERIC_MAX_LINES
            || out.len() + line.len() + 6 > PREVIEW_MAX_BYTES
        {
            truncated = true;
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("```\n");
    if truncated {
        out.push_str("\n*(preview truncated)*\n");
    }
    out
}

/// Collapse any whitespace run (including `\n`, `\r`, `\t`) into a single
/// space and trim. Used when serializing index entries so a `one_liner` or
/// `title` cannot break the `- [title](path) — one_liner` line format.
pub(super) fn sanitize_single_line(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

const INDEX_SEED: &str = "# Wiki Index\n\n\
This file is the catalog of all wiki pages. Each entry links to a page\n\
with a one-line summary. Updated by dm on every ingest.\n\n\
## Entities\n\n\
## Concepts\n\n\
## Summaries\n\n\
## Synthesis\n";

const SCHEMA_SEED: &str = "# Wiki Schema\n\n\
Conventions for pages under `.dm/wiki/`.\n\n\
## Page format\n\n\
Every page is markdown with a YAML frontmatter block:\n\n\
```\n\
---\n\
title: <page title>\n\
type: entity | concept | summary | synthesis\n\
layer: kernel | host  # optional; omitted means kernel\n\
sources:\n\
  - <relative path or identifier>\n\
last_updated: <YYYY-MM-DD HH:MM:SS>\n\
---\n\
<markdown body>\n\
```\n\n\
## Categories\n\n\
- **entity**: concrete artifacts — files, modules, functions, structs, config keys.\n\
- **concept**: abstract ideas — architecture decisions, patterns, invariants, trade-offs.\n\
- **summary**: roll-ups for a directory or source file.\n\
- **synthesis**: cross-cutting analysis — comparisons, contradictions, dependency maps.\n\n\
## Files\n\n\
- `index.md`  — content catalog (one line per page, grouped by category).\n\
- `log.md`    — append-only activity log.\n\
- `schema.md` — this file.\n";

/// Cap on hits returned by [`Wiki::search`]. Bounds result-list size so
/// callers don't have to paginate.
pub const SEARCH_MAX_RESULTS: usize = 20;

/// Cap on query bytes processed by [`Wiki::search`]. Longer queries are
/// silently truncated; never an error.
pub const SEARCH_MAX_QUERY_LEN: usize = 200;

/// Max bytes of snippet returned per hit, including any surrounding ellipsis.
pub(super) const SEARCH_SNIPPET_MAX: usize = 160;

/// Bytes of context on each side of the first body match before clamping.
pub(super) const SEARCH_SNIPPET_SIDE: usize = 60;

/// One row in [`PlannerBrief::recent_cycles`]. Surfaces the cycle number
/// and chain name so the planner can cite them directly, plus the wiki-
/// relative page path so the planner can `file_read` for the full body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecentCycle {
    pub cycle: usize,
    pub chain: String,
    pub page_path: String,
    /// Page `last_updated` frontmatter value, when the synthesis page is
    /// readable and well-formed. `None` on any I/O or parse error — the
    /// brief must never fail to render because a timestamp was missing.
    pub last_updated: Option<String>,
    /// Cycle outcome (`"green"` / `"red"` / `"mixed"` by convention, but
    /// opaque at this layer) sourced from the synthesis page's `outcome`
    /// frontmatter field. `None` for legacy synthesis pages that predate
    /// [`WikiPage::outcome`] and for cycles whose producer had no outcome
    /// signal. Rendered as a trailing `[{outcome}]` badge on the cycle
    /// line in [`PlannerBrief::render`]; `None` omits the badge entirely.
    pub outcome: Option<String>,
}

/// Extract the cycle number from a `synthesis/cycle-{NN}-{slug}-{ts}.md`
/// relative path. Returns `None` for `synthesis/compact-*` pages,
/// malformed paths, or non-numeric cycle segments.
///
/// Anchored on [`SYNTHESIS_CYCLE_PREFIX`] so the filename producer and
/// parser cannot drift apart — a future format change to
/// [`Wiki::write_cycle_synthesis`] must go through this const too.
fn parse_cycle_number_from_synthesis_path(path: &str) -> Option<usize> {
    let rest = path.strip_prefix(SYNTHESIS_CYCLE_PREFIX)?;
    let end = rest.find('-')?;
    rest[..end].parse::<usize>().ok()
}

/// Recover the chain name from a cycle-synthesis index one-liner.
/// [`Wiki::write_cycle_synthesis`] writes them as `"Cycle {N} of {chain}"`;
/// this helper reverses that. Returns `None` for one-liners that don't
/// match (e.g. compact synthesis's `"Compact snapshot (...)"`), so non-
/// cycle pages never leak into `PlannerBrief::recent_cycles`.
fn extract_chain_name_from_one_liner(one_liner: &str) -> Option<String> {
    let s = one_liner.strip_prefix("Cycle ")?;
    let of_idx = s.find(" of ")?;
    let chain = s[of_idx + " of ".len()..].trim();
    if chain.is_empty() {
        return None;
    }
    Some(chain.to_string())
}

/// Parse a wiki page's `last_updated` field into a `SystemTime`. Uses the
/// same `%Y-%m-%d %H:%M:%S` local-time format that `ingest_file` writes.
/// Returns `None` for malformed input or DST-ambiguous times (the latter
/// is a 1-hour window twice a year — silently skipping is strictly better
/// than falsely flagging or crashing).
pub(super) fn parse_page_timestamp(s: &str) -> Option<std::time::SystemTime> {
    use chrono::{Local, NaiveDateTime, TimeZone};
    let naive = NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S").ok()?;
    let local = Local.from_local_datetime(&naive).single()?;
    Some(local.into())
}

/// Handle to a wiki rooted at `\<project\>/.dm/wiki/`.
#[derive(Debug, Clone)]
pub struct Wiki {
    pub(super) root: PathBuf,
}

impl Wiki {
    /// Open (or create) the wiki for the given project root. Computes
    /// `\<project_root\>/.dm/wiki/` and ensures the layout exists.
    pub fn open(project_root: &Path) -> io::Result<Wiki> {
        let root = project_root.join(".dm").join("wiki");
        let wiki = Wiki { root };
        wiki.ensure_layout()?;
        Ok(wiki)
    }

    /// Create the directory layout and seed `index.md` + `schema.md` if
    /// either is missing. Existing files are left untouched.
    pub fn ensure_layout(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        for sub in ["entities", "concepts", "summaries", "synthesis"] {
            fs::create_dir_all(self.root.join(sub))?;
        }
        let index_path = self.root.join("index.md");
        if !index_path.exists() {
            fs::write(&index_path, INDEX_SEED)?;
        }
        let schema_path = self.root.join("schema.md");
        if !schema_path.exists() {
            fs::write(&schema_path, SCHEMA_SEED)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn log(&self) -> WikiLog {
        WikiLog::new(self.root.join("log.md"))
    }

    /// Compute inbound cross-reference counts per entity page.
    ///
    /// For every entity page in `idx`, reads its frontmatter to recover
    /// the canonical source path. Then, for every entity page whose
    /// source file exists under the project root, reads that source file
    /// and scans it (via [`wiki_link_scan`]) for occurrences of the
    /// OTHER entity pages' source paths; each hit bumps the matched
    /// page's inbound count.
    ///
    /// Pages whose frontmatter cannot be parsed, or whose source file is
    /// missing/unreadable, are silently skipped — a broken or deleted
    /// source degrades to zero inbound links, never an error (Prime
    /// Directive: no silent failures for writes, but read-side scans
    /// must tolerate drift).
    ///
    /// Complexity is O(N² · S) where N = entity-page count and S =
    /// average source-file size. At current wiki sizes (<100 pages,
    /// KB-scale files) this is milliseconds; a future cycle can add
    /// caching or incremental updates if `/wiki status` measurably slows.
    pub(crate) fn compute_inbound_links(
        &self,
        idx: &WikiIndex,
    ) -> io::Result<Vec<(String, usize)>> {
        let proj = self.project_root();
        let entity_srcs: Vec<(String, String)> = idx
            .entries
            .iter()
            .filter(|e| e.category == PageType::Entity)
            .filter_map(|e| {
                let page_abs = self.root.join(&e.path);
                let text = fs::read_to_string(&page_abs).ok()?;
                let page = WikiPage::parse(&text)?;
                let src = page.sources.into_iter().next()?;
                Some((e.path.clone(), src))
            })
            .collect();

        let mut counts: std::collections::HashMap<String, usize> = entity_srcs
            .iter()
            .map(|(page, _)| (page.clone(), 0usize))
            .collect();

        for (_page, self_src) in &entity_srcs {
            let abs = proj.join(self_src);
            let Ok(content) = fs::read_to_string(&abs) else {
                continue;
            };
            let iter = entity_srcs.iter().map(|(p, s)| (p.as_str(), s.as_str()));
            for target_page in wiki_link_scan(&content, iter, Some(self_src.as_str())) {
                if let Some(c) = counts.get_mut(&target_page) {
                    *c += 1;
                }
            }
        }

        Ok(counts.into_iter().collect())
    }

    /// Project root for this wiki handle. `self.root` is `\<proj\>/.dm/wiki/`,
    /// so the project root is two parents up. Falls back to the wiki root
    /// itself if somehow the handle was opened at a shallower path — callers
    /// use this only for resolving source-path lookups, which will simply
    /// report the sources as missing in that (never-in-practice) case.
    pub(super) fn project_root(&self) -> PathBuf {
        self.root
            .parent()
            .and_then(|p| p.parent())
            .map_or_else(|| self.root.clone(), Path::to_path_buf)
    }

    /// Scan `synthesis/cycle-*.md` index entries and return the newest
    /// `limit` cycles (by cycle number desc, `page_path` asc tiebreak) as
    /// [`RecentCycle`] rows. Shared by [`Wiki::planner_brief`] (tight
    /// context surface) and [`Wiki::build_project_summary`] (reference
    /// surface) so any cache-elision regression is caught by either
    /// consumer's tests.
    ///
    /// I/O cost (C92): both `last_updated` and `outcome` are normally
    /// read from the [`IndexEntry`] cache — zero disk reads on the hot
    /// path. The per-entry page read fires only when either field is
    /// missing from the index (legacy fallback), which self-heals on
    /// the next ingest that touches the entry.
    ///
    /// Best-effort: a missing / malformed index degrades to an empty
    /// vec via `load_index().unwrap_or_default()` so the helper never
    /// surfaces an error to callers whose next step is rendering.
    pub(crate) fn collect_recent_cycles(&self, limit: usize) -> Vec<RecentCycle> {
        let index = self.load_index().unwrap_or_default();
        let mut recents: Vec<RecentCycle> = index
            .entries
            .iter()
            .filter(|e| e.category == PageType::Synthesis)
            .filter_map(|e| {
                let cycle = parse_cycle_number_from_synthesis_path(&e.path)?;
                let chain = extract_chain_name_from_one_liner(&e.one_liner)?;
                let (last_updated, outcome) = if e.last_updated.is_some() && e.outcome.is_some() {
                    (e.last_updated.clone(), e.outcome.clone())
                } else {
                    let parsed_page = std::fs::read_to_string(self.root.join(&e.path))
                        .ok()
                        .and_then(|text| WikiPage::parse(&text));
                    let ts = match &e.last_updated {
                        Some(ts) => Some(ts.clone()),
                        None => parsed_page.as_ref().map(|p| p.last_updated.clone()),
                    };
                    let outc = match &e.outcome {
                        Some(v) => Some(v.clone()),
                        None => parsed_page.and_then(|p| p.outcome),
                    };
                    (ts, outc)
                };
                Some(RecentCycle {
                    cycle,
                    chain,
                    page_path: e.path.clone(),
                    last_updated,
                    outcome,
                })
            })
            .collect();
        recents.sort_by(|a, b| {
            b.cycle
                .cmp(&a.cycle)
                .then_with(|| a.page_path.cmp(&b.page_path))
        });
        recents.truncate(limit);
        recents
    }

    /// Check whether a source file is newer than its canonical entity page.
    ///
    /// Returns `Some(warning)` when:
    /// - the canonical entity page exists, AND
    /// - the source file's mtime is strictly later than the page's
    ///   `last_updated` timestamp.
    ///
    /// Returns `None` when the page is fresh, missing, or unreadable.
    /// This is the per-file counterpart to lint Rule 4
    /// (`SourceNewerThanPage`) — surfaced inline after edits so the model
    /// knows the wiki may be stale before the next `/wiki lint` cycle.
    pub fn check_source_drift(
        &self,
        project_root: &Path,
        canonical_path: &Path,
    ) -> io::Result<Option<String>> {
        let Ok(rel) = canonical_path.strip_prefix(project_root) else {
            return Ok(None);
        };
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.starts_with(".dm/wiki/") {
            return Ok(None);
        }

        let page_rel = entity_page_rel(&rel_str);
        let page = match self.read_page(&page_rel) {
            Ok(p) => p,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };

        let Some(page_ts) = parse_page_timestamp(&page.last_updated) else {
            return Ok(None);
        };

        let Ok(src_meta) = fs::metadata(canonical_path) else {
            return Ok(None);
        };
        let Ok(src_mtime) = src_meta.modified() else {
            return Ok(None);
        };

        if src_mtime > page_ts {
            Ok(Some(format!(
                "[wiki-drift] Wiki page '{}' may be stale (source file modified after last ingest at {}). \
                 Run /wiki refresh or re-read the file to update the wiki.",
                page_rel, page.last_updated,
            )))
        } else {
            Ok(None)
        }
    }
}

/// Test-only: poll `\<project_root\>/.dm/wiki/log.md` for `marker`, returning
/// `true` as soon as it appears and `false` after ~1 s of absence. Colocated
/// with `ingest_file` so the polling target stays in lockstep with the
/// hook's three-step write order (page → index → log): waiting on the log
/// line guarantees the page and index have already landed.
#[cfg(test)]
pub(crate) async fn wait_for_ingest_log_marker(project_root: &Path, marker: &str) -> bool {
    let log_path = project_root.join(".dm/wiki/log.md");
    for _ in 0..20 {
        if log_path.is_file() {
            if let Ok(s) = std::fs::read_to_string(&log_path) {
                if s.contains(marker) {
                    return true;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    false
}

/// Open or create the project wiki for the current working directory.
/// On I/O failure, pushes a warning through `crate::warnings` and returns
/// `None` rather than propagating — wiki init is best-effort.
// Note: reads `env::current_dir()` at session start only, before any tool
// can drift cwd, so the bug Finding 2 addresses (cwd drift during long
// sessions) is bounded to the compaction tee, not this entry point.
pub fn ensure_for_cwd() -> Option<Wiki> {
    let cwd = std::env::current_dir().ok()?;
    match Wiki::open(&cwd) {
        Ok(w) => Some(w),
        Err(e) => {
            crate::warnings::push_warning(format!(
                "wiki: failed to initialize at {}: {}",
                cwd.display(),
                e
            ));
            None
        }
    }
}

/// Reject relative wiki paths that could escape the wiki root.
pub(super) fn validate_rel(rel: &str) -> io::Result<()> {
    if rel.is_empty() || rel.starts_with('/') || rel.starts_with('\\') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsafe wiki page path: {}", rel),
        ));
    }
    for comp in rel.split(&['/', '\\'][..]) {
        if comp == ".." {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsafe wiki page path: {}", rel),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
