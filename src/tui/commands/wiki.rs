//! `/wiki` subcommand surface.
//!
//! Full command-family home: formatters (`format_wiki_*`), dispatch
//! handlers (`handle_wiki_*`), the subcommand-name table
//! (`WIKI_SUBCOMMAND_NAMES`), and the fuzzy suggester
//! (`suggest_wiki_subcommand`). Consumed by `commands::execute()`'s
//! `/wiki` match arms. Phase 1.2 (split-impl for `tui/commands.rs`).

use std::fmt::Write as _;

use super::SlashResult;

pub(super) const WIKI_SUBCOMMAND_NAMES: &[&str] = &[
    "status", "search", "lint", "refresh", "summary", "concepts", "momentum", "fresh", "planner",
    "stats", "prune", "seed",
];

/// Parse an optional `<category>:` prefix from a `/wiki search` query.
///
/// Recognizes the four `PageType` values as case-insensitive prefixes on
/// the first whitespace-delimited token: `entity:`, `concept:`,
/// `summary:`, `synthesis:`. Anything else (including unrecognized
/// prefixes like `foo:`) returns `(input.to_string(), None)` so the user
/// can still search for literal `foo:bar` strings.
///
/// Returns `(effective_query, category)`:
/// - `entity:auth tokens` → `("auth tokens", Some("entity"))`
/// - `entity: auth tokens` → `("auth tokens", Some("entity"))` (whitespace after `:` ok)
/// - `entity:` → `("", Some("entity"))` (caller treats empty query as usage hint)
/// - `auth tokens` → `("auth tokens", None)`
/// - `something:weird foo` → `("something:weird foo", None)`
pub(super) fn parse_category_prefix(input: &str) -> (String, Option<String>) {
    const KNOWN: &[&str] = &["entity", "concept", "summary", "synthesis"];
    let trimmed_lead = input.trim_start();
    if let Some(colon_idx) = trimmed_lead.find(':') {
        let prefix = &trimmed_lead[..colon_idx];
        let prefix_lc = prefix.to_lowercase();
        if KNOWN.contains(&prefix_lc.as_str()) && !prefix.contains(char::is_whitespace) {
            let rest = trimmed_lead[colon_idx + 1..].trim_start();
            return (rest.to_string(), Some(prefix_lc));
        }
    }
    (input.to_string(), None)
}

/// Fuzzy-match a (possibly misspelled) `/wiki` subcommand against
/// `WIKI_SUBCOMMAND_NAMES`. Mirrors `suggest_slash_command` but uses the
/// same tight Levenshtein ≤ 2 threshold — the subcommand namespace is
/// small (8 names, 4–8 chars each), and a looser threshold would flip
/// near-miss inputs to unrelated commands.
pub(super) fn suggest_wiki_subcommand(input: &str) -> Option<&'static str> {
    if input.is_empty() {
        return None;
    }

    // Substring match first — lets "stat" / "concept" map to "status" /
    // "concepts" before Levenshtein gets a chance to pick a shorter name.
    // All wiki subcommand names are ≥ 4 chars so no guard is needed.
    if input.len() >= 2 {
        for &registered in WIKI_SUBCOMMAND_NAMES {
            if registered.contains(input) || input.contains(registered) {
                return Some(registered);
            }
        }
    }

    let mut best: Option<(&'static str, usize)> = None;
    for &registered in WIKI_SUBCOMMAND_NAMES {
        let dist = crate::util::levenshtein(input, registered);
        if dist <= 2 && best.is_none_or(|(_, d)| dist < d) {
            best = Some((registered, dist));
        }
    }
    best.map(|(name, _)| name)
}

/// Render a `crate::wiki::WikiStats` snapshot (page counts + last-activity
/// timestamp) as a multi-line info block for the `/wiki status` slash.
/// Distinct from `format_wiki_stats` (the C7-added telemetry-counter
/// formatter for `/wiki stats`); the type-and-slash names sit one letter
/// apart, but the surfaces are unrelated. Categories are always listed in
/// canonical order (entity → concept → summary → synthesis) with
/// zero-count entries included so the output shape is stable.
pub(super) fn format_wiki_status(s: &crate::wiki::WikiStats) -> String {
    use crate::wiki::PageType;
    let get = |pt: PageType| s.by_category.get(&pt).copied().unwrap_or(0);
    let last = s.last_activity.as_deref().unwrap_or("(none)");
    let mut out = format!(
        "Wiki status:\n  \
         Root:        {}\n  \
         Pages:       {} total\n               \
         {} entities, {} concepts, {} summaries, {} synthesis\n  \
         Log:         {} entries\n  \
         Last update: {}",
        s.root.display(),
        s.total_pages,
        get(PageType::Entity),
        get(PageType::Concept),
        get(PageType::Summary),
        get(PageType::Synthesis),
        s.log_entries,
        last,
    );
    if !s.most_linked.is_empty() {
        out.push_str("\n  Most linked:");
        for (path, count) in &s.most_linked {
            write!(out, "\n    {}  ({})", path, count).expect("write to String never fails");
        }
    }
    if s.total_pages == 0 {
        out.push_str("\n\nWiki is empty — read files or let auto-ingest populate it.");
    }
    out
}

/// Render a `/wiki search` result set. Three shapes:
///  * empty query — show usage line
///  * non-empty query, zero hits — "No matches" line echoing the query
///  * non-empty query, N hits — header + one line per hit with path,
///    title, match count, and a snippet (title-only if no body match)
///
/// Back-compat wrapper: production callers should reach for
/// `format_wiki_search_with_fuzzy` so the slash command can surface
/// near-miss titles on a zero-hit query. The wrapper passes empty
/// fuzzy data, hitting the bare-404 branch — preserves byte-identical
/// output for the existing formatter tests.
pub(super) fn format_wiki_search(query: &str, hits: &[crate::wiki::WikiSearchHit]) -> String {
    format_wiki_search_with_fuzzy(query, hits, &[], 0)
}

/// Render a `/wiki search` result set with optional fuzzy near-miss
/// data for the zero-hit branch. When `hits.is_empty()` and
/// `fuzzy_ranked` is non-empty, branches on whether the closest entry
/// clears `max_dist` — accepted-fuzzy lists "Closest titles:", rejected
/// lists "above similarity threshold N", empty fuzzy keeps the bare 404.
pub(super) fn format_wiki_search_with_fuzzy(
    query: &str,
    hits: &[crate::wiki::WikiSearchHit],
    fuzzy_ranked: &[(usize, String, String)],
    max_dist: usize,
) -> String {
    let q = query.trim();
    if q.is_empty() {
        return "Usage: /wiki search [<cat>:]<query>\n\n\
                Case-insensitive substring + fuzzy match across wiki pages.\n\
                Optional category prefix: entity, concept, summary, synthesis.\n\
                Example: /wiki search concept:logging"
            .to_string();
    }
    if hits.is_empty() {
        let accepted = fuzzy_ranked.first().is_some_and(|(d, _, _)| *d <= max_dist);
        if accepted {
            let mut out = format!("No matches for '{}'. Closest titles:\n", q);
            for (d, p, t) in fuzzy_ranked.iter().take_while(|(d, _, _)| *d <= max_dist) {
                writeln!(out, "  - {} — {} (distance {})", p, t, d)
                    .expect("write to String never fails");
            }
            out.push_str("\nTip: try /wiki status to see all pages.");
            return out;
        }
        if !fuzzy_ranked.is_empty() {
            let mut out = format!(
                "No wiki matches for '{}'. Closest titles \
                 (above similarity threshold {}):\n",
                q, max_dist
            );
            for (d, p, t) in fuzzy_ranked {
                writeln!(out, "  - {} — {} (distance {})", p, t, d)
                    .expect("write to String never fails");
            }
            return out;
        }
        return format!("No wiki matches for '{}'.", q);
    }
    let mut out = format!(
        "Found {} match{} for '{}':",
        hits.len(),
        if hits.len() == 1 { "" } else { "es" },
        q,
    );
    for h in hits {
        write!(
            out,
            "\n  [{}x] {} — {}\n        {}",
            h.match_count, h.path, h.title, h.snippet,
        )
        .expect("write to String never fails");
    }
    out
}

/// Render a `/wiki lint` result set. Zero findings → single-line clean
/// message. Otherwise group by `WikiLintKind` with a per-group header and
/// indented bullets; empty groups are omitted.
pub(super) fn format_wiki_lint(findings: &[crate::wiki::WikiLintFinding]) -> String {
    use crate::wiki::WikiLintKind;
    if findings.is_empty() {
        return "Wiki lint: no issues found.".to_string();
    }
    let mut orphans: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut untracked: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut mismatches: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut index_timestamp_drift: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut source_missing: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut body_path_missing: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut concept_scope_undocumented: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut source_newer: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut item_drift: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut export_drift: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut entity_gap: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut missing_kind: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut duplicate_source: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    let mut malformed: Vec<&crate::wiki::WikiLintFinding> = Vec::new();
    for f in findings {
        match f.kind {
            WikiLintKind::OrphanIndexEntry => orphans.push(f),
            WikiLintKind::UntrackedPage => untracked.push(f),
            WikiLintKind::CategoryMismatch => mismatches.push(f),
            WikiLintKind::IndexTimestampDrift => index_timestamp_drift.push(f),
            WikiLintKind::SourceMissing => source_missing.push(f),
            WikiLintKind::BodyPathMissing => body_path_missing.push(f),
            WikiLintKind::ConceptScopeUndocumented => concept_scope_undocumented.push(f),
            WikiLintKind::SourceNewerThanPage => source_newer.push(f),
            WikiLintKind::ItemDrift => item_drift.push(f),
            WikiLintKind::ExportDrift => export_drift.push(f),
            WikiLintKind::EntityGap => entity_gap.push(f),
            WikiLintKind::MissingEntityKind => missing_kind.push(f),
            WikiLintKind::DuplicateSource => duplicate_source.push(f),
            WikiLintKind::MalformedPage => malformed.push(f),
        }
    }
    let mut out = format!("Wiki lint — {} finding(s):", findings.len());
    let push_group = |label: &str, items: &[&crate::wiki::WikiLintFinding], out: &mut String| {
        if items.is_empty() {
            return;
        }
        write!(out, "\n\n{} ({}):", label, items.len()).expect("write to String never fails");
        for f in items {
            write!(out, "\n  - {}: {}", f.path, f.detail).expect("write to String never fails");
        }
    };
    push_group("Orphan index entries", &orphans, &mut out);
    push_group("Untracked pages", &untracked, &mut out);
    push_group("Category mismatches", &mismatches, &mut out);
    push_group("Index timestamp drift", &index_timestamp_drift, &mut out);
    push_group("Missing source files", &source_missing, &mut out);
    push_group("Body-text phantom paths", &body_path_missing, &mut out);
    push_group(
        "Concept scope — undocumented files",
        &concept_scope_undocumented,
        &mut out,
    );
    push_group("Stale pages — source newer", &source_newer, &mut out);
    push_group(
        "Item drift — page items not in source",
        &item_drift,
        &mut out,
    );
    push_group(
        "Export drift — frontmatter exports not in source",
        &export_drift,
        &mut out,
    );
    push_group("Undocumented source files", &entity_gap, &mut out);
    push_group("Missing entity_kind", &missing_kind, &mut out);
    push_group(
        "Duplicate source contradictions",
        &duplicate_source,
        &mut out,
    );
    push_group("Malformed pages", &malformed, &mut out);
    out
}

/// Render a `/wiki concepts` report. Empty state → usage hint. Otherwise
/// a single-line summary plus up to 10 of the top detected dependencies
/// with their consumer counts and a trailing next-step line.
pub(super) fn format_wiki_concepts(report: &crate::wiki::ConceptPagesReport) -> String {
    if report.generated.is_empty() && report.refreshed.is_empty() && report.detected_deps.is_empty()
    {
        return "No shared dependencies meet the threshold (≥ 3 entity pages). \
                Next: run `/wiki refresh` after ingesting more Rust sources, \
                then retry `/wiki concepts`."
            .to_string();
    }
    let mut out = format!(
        "Concept detection: {} new, {} refreshed, {} total shared dep(s).\n",
        report.generated.len(),
        report.refreshed.len(),
        report.detected_deps.len(),
    );
    for (dep, n) in report.detected_deps.iter().take(10) {
        writeln!(out, "  • `{}` — {} consumer(s)", dep, n).expect("write to String never fails");
    }
    out.push_str(
        "\nNext: open `.dm/wiki/concepts/dep-*.md` to inspect, \
         or run `/wiki summary` to re-roll the project summary.",
    );
    out
}

/// Render a `/wiki momentum` report. Empty state → a single next-step line
/// nudging the operator toward ingest. Otherwise a one-line header,
/// top-5 hot paths, top-5 hot modules, and an actionable trailing hint
/// pointing at `/wiki refresh` (the verb that generates more ingest
/// entries).
pub(super) fn format_wiki_momentum(report: &crate::wiki::MomentumReport) -> String {
    if report.window_processed == 0 {
        return "No ingest activity recorded yet. \
                Next: run `/wiki refresh` after ingesting some Rust sources, \
                then retry `/wiki momentum`."
            .to_string();
    }
    let mut out = format!(
        "Momentum: {} ingest entr{} in window ({} total log line{}).\n",
        report.window_processed,
        if report.window_processed == 1 {
            "y"
        } else {
            "ies"
        },
        report.total_entries,
        if report.total_entries == 1 { "" } else { "s" },
    );
    out.push_str("\nHot paths:\n");
    for (p, n) in &report.hot_paths {
        writeln!(out, "  • `{}` — {} ingest(s)", p, n).expect("write to String never fails");
    }
    out.push_str("\nHot modules:\n");
    for (m, n) in &report.hot_modules {
        writeln!(out, "  • `{}` — {} ingest(s)", m, n).expect("write to String never fails");
    }
    out.push_str(
        "\nNext: focus review on the hottest module, \
         or run `/wiki refresh` to capture further churn.",
    );
    out
}

/// Render `/wiki summary` — one-line confirmation of the
/// `write_project_summary` result, plus a `Recent cycles:` trailer when
/// the report carries synthesis-page cycle outcomes. Mirrors the body-
/// side "Recent cycles" format (`cycle N (outcome)` when `Some`, bare
/// `cycle N` when `None`) so the TUI line stays visually consistent
/// with the page body. Chain name stays in the page body — the TUI
/// trailer is a glanceable index, not a full report.
pub(super) fn format_wiki_summary(report: &crate::wiki::ProjectSummaryReport) -> String {
    let kinds = if report.kind_counts.is_empty() {
        "no entity kinds".to_string()
    } else {
        report
            .kind_counts
            .iter()
            .map(|(k, n)| format!("{}: {}", k.as_str(), n))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = format!(
        "Project summary written to {} ({} entit{}, {})",
        report.path,
        report.entity_count,
        if report.entity_count == 1 { "y" } else { "ies" },
        kinds,
    );
    if !report.recent_cycles.is_empty() {
        let cycles = report
            .recent_cycles
            .iter()
            .map(|rc| match &rc.outcome {
                Some(o) => format!("cycle {} ({})", rc.cycle, o),
                None => format!("cycle {}", rc.cycle),
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(out, "\nRecent cycles: {}", cycles).expect("write to String never fails");
    }
    out
}

/// Render `/wiki fresh` — rank `Entity`/`Concept` pages by `last_updated`
/// descending and emit a ranked list with per-entry `(updated {ts})`
/// suffixes so operators can see the same top-K the model receives via
/// the session-start `<wiki_fresh>` block. Empty bucket → stub pointing
/// at `/wiki refresh`. Hard byte cap at `budget_chars`, truncated on line
/// boundaries so no partial entry lines reach the display.
pub(super) fn format_wiki_fresh(wiki: &crate::wiki::Wiki, budget_chars: usize) -> String {
    let stub = "No Entity or Concept pages yet. Try: `/wiki refresh` after touching source files.";
    let idx = match wiki.load_index() {
        Ok(idx) => idx,
        Err(e) => return format!("wiki fresh failed: {}. Try: /wiki status.", e),
    };
    let mut ranked: Vec<&crate::wiki::IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.category,
                crate::wiki::PageType::Entity | crate::wiki::PageType::Concept
            )
        })
        .collect();
    if ranked.is_empty() {
        return stub.to_string();
    }
    ranked.sort_by(|a, b| match (&a.last_updated, &b.last_updated) {
        (Some(x), Some(y)) => y.cmp(x).then_with(|| a.path.cmp(&b.path)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.path.cmp(&b.path),
    });

    let header = format!(
        "Fresh wiki pages ({} Entity/Concept total):\n",
        ranked.len()
    );
    let mut out = String::new();
    for e in ranked {
        let line = match &e.last_updated {
            Some(ts) => format!("- {} — {} (updated {})\n", e.title, e.one_liner, ts),
            None => format!("- {} — {}\n", e.title, e.one_liner),
        };
        let needed = if out.is_empty() {
            header.len() + line.len()
        } else {
            line.len()
        };
        if out.len() + needed > budget_chars {
            break;
        }
        if out.is_empty() {
            out.push_str(&header);
        }
        out.push_str(&line);
    }
    if out.is_empty() {
        return stub.to_string();
    }
    out
}

/// Render `/wiki planner` — the same brief the orchestration runner injects
/// into chain planner prompts. Empty brief → one-line placeholder pointing
/// at the next step; otherwise delegate to `PlannerBrief::render` so the
/// operator sees the exact section the planner will see.
/// Usage hint surfaced when the user types `/wiki <unrecognized>`. Listed
/// separately from the match arm so tests can drive it without needing to
/// build a full `App`. Prefixes a "Did you mean" line when the first
/// whitespace-delimited token of `sub` fuzzy-matches a known subcommand.
pub(super) fn format_wiki_unknown_usage(sub: &str) -> String {
    let needle = sub.split_whitespace().next().unwrap_or("");
    let suggestion = suggest_wiki_subcommand(needle);
    let prefix = match suggestion {
        Some(name) => format!("Did you mean /wiki {}?\n\n", name),
        None => String::new(),
    };
    // Group by operator intent so a typo'd subcommand surfaces a list
    // organized like a mental model: read first, then maintain, then act.
    // Each subcommand line uses 6-space indent under a 2-space group
    // header; the indent split is what
    // `format_wiki_unknown_usage_lists_all_twelve_subcommands` counts.
    format!(
        "{}Unknown /wiki subcommand: {}\n\n\
         Usage:\n  \
         Inspect:\n      \
         /wiki status           — show page counts and last activity\n      \
         /wiki search [cat:]<q> — substring + fuzzy match (cat: entity|concept|summary|synthesis)\n      \
         /wiki lint             — check wiki for orphans, untracked pages, category drift\n      \
         /wiki momentum         — show hot paths and modules from recent ingest activity\n      \
         /wiki fresh            — show top-K entity/concept pages ranked by last_updated\n      \
         /wiki planner          — show the orchestration planner brief (hot paths, drift, lint, recent cycles)\n      \
         /wiki stats            — show wiki tool-call + drift-warning counters\n  \
         Maintain:\n      \
         /wiki refresh          — re-ingest pages with stale or missing entity metadata\n      \
         /wiki summary          — regenerate summaries/project.md from entity pages\n      \
         /wiki concepts         — auto-detect shared dependencies and write concept pages\n  \
         Operator one-shot:\n      \
         /wiki prune [<N>]      — drop oldest synthesis/compact-*.md beyond cap (default {})\n      \
         /wiki seed [<dir>]     — recursively ingest .rs files under <dir> (default src/)",
        prefix,
        sub,
        crate::wiki::DEFAULT_COMPACT_KEEP,
    )
}

pub(super) fn format_wiki_planner_brief(brief: &crate::wiki::PlannerBrief) -> String {
    let placeholder =
        "No planner signals yet. Ingest Rust sources (edit/read files) or run `/wiki refresh`."
            .to_string();
    if brief.is_empty() {
        return placeholder;
    }
    brief
        .render(crate::wiki::PLANNER_BRIEF_BUDGET_CHARS)
        .unwrap_or(placeholder)
}

/// Render a `/wiki refresh` report. Clean state → single-line confirmation.
/// Otherwise lists refreshed sources, then any missing sources, then any
/// per-source ingest errors.
pub(super) fn format_wiki_refresh(report: &crate::wiki::WikiRefreshReport) -> String {
    if report.refreshed.is_empty() && report.missing_sources.is_empty() && report.errors.is_empty()
    {
        return format!(
            "Wiki is up to date ({} page-source pair{} checked).",
            report.up_to_date,
            if report.up_to_date == 1 { "" } else { "s" }
        );
    }
    let mut out = format!(
        "Refreshed {} page-source pair{} ({} already up to date):",
        report.refreshed.len(),
        if report.refreshed.len() == 1 { "" } else { "s" },
        report.up_to_date,
    );
    for src in &report.refreshed {
        write!(out, "\n  [ok] {}", src).expect("write to String never fails");
    }
    if !report.missing_sources.is_empty() {
        out.push_str("\nMissing sources (skipped):");
        for src in &report.missing_sources {
            write!(out, "\n  [missing] {}", src).expect("write to String never fails");
        }
    }
    if !report.errors.is_empty() {
        out.push_str("\nErrors:");
        for (src, err) in &report.errors {
            write!(out, "\n  [err] {}: {}", src, err).expect("write to String never fails");
        }
    }
    out
}

// ── /wiki dispatch handlers (C21) ────────────────────────────────────
// One sync handler per /wiki subcommand. Each opens the wiki at cwd,
// invokes the appropriate Wiki method, and renders via the matching
// formatter above. Extracted from `execute()` in C21 so the entire
// /wiki command surface (parse → handle → render) lives in this file.

/// Dispatcher for `/wiki status`.
pub(super) fn handle_wiki_status() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.stats() {
            Ok(s) => SlashResult::Info(format_wiki_status(&s)),
            Err(e) => SlashResult::Error(format!(
                "wiki status failed: {}. Try: /wiki refresh",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki search <query>`. Optional `<category>:` prefix
/// (`entity:`, `concept:`, `summary:`, `synthesis:`) restricts results.
/// Unknown prefixes pass through as literal query text — see
/// `parse_category_prefix`.
pub(super) fn handle_wiki_search(query: String) -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    let (effective_query, category) = parse_category_prefix(&query);
    // Specific guard: prefix parsed but rest is blank. Without this, the
    // user sees a generic "Usage: …" message that doesn't acknowledge
    // the category was accepted — making it look like the prefix was
    // wrong rather than that the query body was missing.
    if let Some(cat) = category.as_deref() {
        if effective_query.trim().is_empty() {
            return SlashResult::Info(format!(
                "/wiki search: category '{}' was parsed, but the query is empty. \
                 Try: /wiki search {}:<keyword>, e.g. /wiki search {}:auth",
                cat, cat, cat
            ));
        }
    }
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.search(&effective_query) {
            Ok(hits) => {
                // Filter substring hits by category (mirrors wiki_search tool).
                let hits: Vec<_> = if let Some(cat) = category.as_deref() {
                    hits.into_iter()
                        .filter(|h| h.category.as_str() == cat)
                        .collect()
                } else {
                    hits
                };

                let trimmed = effective_query.trim();
                if hits.is_empty() && !trimmed.is_empty() {
                    let needle = trimmed.to_lowercase();
                    let max_dist = crate::wiki::fuzzy::fuzzy_threshold(&needle);
                    let ranked = match wiki.load_index() {
                        Ok(idx) => crate::wiki::fuzzy::rank_entries_by_levenshtein(
                            &idx.entries,
                            &needle,
                            category.as_deref(),
                            3,
                        ),
                        Err(_) => Vec::new(),
                    };
                    SlashResult::Info(format_wiki_search_with_fuzzy(
                        &effective_query,
                        &hits,
                        &ranked,
                        max_dist,
                    ))
                } else {
                    SlashResult::Info(format_wiki_search(&effective_query, &hits))
                }
            }
            Err(e) => SlashResult::Error(format!(
                "wiki search failed: {}. Try: /wiki status to see if the wiki is reachable.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!(
            "wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.",
            e
        )),
    }
}

/// Dispatcher for `/wiki lint`.
pub(super) fn handle_wiki_lint() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.lint() {
            Ok(findings) => SlashResult::Info(format_wiki_lint(&findings)),
            Err(e) => SlashResult::Error(format!(
                "wiki lint failed: {}. Try: /wiki refresh first if pages are stale.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki refresh`.
pub(super) fn handle_wiki_refresh() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.refresh() {
            Ok(report) => SlashResult::Info(format_wiki_refresh(&report)),
            Err(e) => SlashResult::Error(format!(
                "wiki refresh failed: {}. Try: /wiki lint to see current state.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki summary`.
pub(super) fn handle_wiki_summary() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.write_project_summary() {
            Ok(report) => SlashResult::Info(format_wiki_summary(&report)),
            Err(e) => SlashResult::Error(format!(
                "wiki summary failed: {}. Try: /wiki status to check wiki health.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki concepts`.
pub(super) fn handle_wiki_concepts() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.write_concept_pages() {
            Ok(report) => SlashResult::Info(format_wiki_concepts(&report)),
            Err(e) => SlashResult::Error(format!(
                "wiki concepts failed: {}. Try: ensure `.dm/wiki/` exists and retry; \
                 `/wiki refresh` may surface the underlying cause.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki momentum`.
pub(super) fn handle_wiki_momentum() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.momentum(crate::wiki::MOMENTUM_DEFAULT_WINDOW) {
            Ok(report) => SlashResult::Info(format_wiki_momentum(&report)),
            Err(e) => SlashResult::Error(format!(
                "wiki momentum failed: {}. Try: /wiki status to check wiki health.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki fresh`.
pub(super) fn handle_wiki_fresh() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => SlashResult::Info(format_wiki_fresh(&wiki, 4096)),
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki planner`.
pub(super) fn handle_wiki_planner() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.planner_brief(
            crate::wiki::MOMENTUM_DEFAULT_WINDOW,
            crate::wiki::PLANNER_BRIEF_MAX_DRIFT,
        ) {
            Ok(brief) => SlashResult::Info(format_wiki_planner_brief(&brief)),
            Err(e) => SlashResult::Error(format!(
                "wiki planner brief failed: {}. Try: /wiki status to check wiki health.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!("wiki open failed: {}. Try: /wiki refresh to re-ingest, or check that .dm/wiki/ exists.", e)),
    }
}

/// Dispatcher for `/wiki <unrecognized>` — no wiki open needed.
pub(super) fn handle_wiki_unknown(sub: String) -> SlashResult {
    SlashResult::Info(format_wiki_unknown_usage(&sub))
}

/// Render the three wiki-telemetry metrics as an aligned status panel.
/// Tip line names each counter's source so operators can correlate the
/// numbers with the underlying signals.
pub(super) fn format_wiki_stats(
    tool_calls: usize,
    drift_warnings: usize,
    snippet_bytes: usize,
) -> String {
    format!(
        "Wiki telemetry (this session):\n  \
         tool_calls:       {}\n  \
         drift_warnings:   {}\n  \
         snippet_bytes:    {}\n\n\
         Tip: tool_calls counts wiki_lookup/wiki_search invocations; \
         drift_warnings counts [wiki-drift] markers in tool results; \
         snippet_bytes is the system-prompt wiki injection size.",
        tool_calls, drift_warnings, snippet_bytes
    )
}

/// Dispatcher for `/wiki stats` — snapshot telemetry globals and compute
/// snippet bytes on the fly so the number reflects whatever the next
/// session would inject. Wiki-less projects show 0 for `snippet_bytes`.
pub(super) fn handle_wiki_stats() -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    let (tool_calls, drift_warnings) = crate::telemetry::snapshot();
    let snippet_bytes = crate::api::load_wiki_snippet(&cwd).map_or(0, |s| s.len())
        + crate::api::load_wiki_fresh(&cwd).map_or(0, |s| s.len());
    SlashResult::Info(format_wiki_stats(tool_calls, drift_warnings, snippet_bytes))
}

/// Render the result of `/wiki prune`. Distinguishes the no-op case
/// (already within cap) from a positive prune count, and pluralizes the
/// noun. The cap is included in both messages so the operator can sanity-
/// check what they passed.
pub(super) fn format_wiki_prune(pruned: usize, max_keep: usize) -> String {
    if pruned == 0 {
        format!(
            "Compact synthesis already within cap (kept ≤ {}).",
            max_keep
        )
    } else {
        let noun = if pruned == 1 { "page" } else { "pages" };
        format!(
            "Pruned {} compact-* synthesis {} (cap: {}).",
            pruned, noun, max_keep
        )
    }
}

/// Render the result of `/wiki seed`. One header line summarising
/// totals, then up to 10 ingested rel paths and up to 5 errors so the
/// output stays under ~30 lines even for big trees. The clean-zero case
/// includes a `Try:` hint so the operator knows what to try next.
/// `symlinks_skipped` only appears in the header when non-zero — it's
/// a security/correctness signal worth surfacing, but most trees have
/// none and the noise would distract.
pub(super) fn format_wiki_seed(report: &crate::wiki::WikiSeedReport, dir: &str) -> String {
    let n_ing = report.ingested.len();
    let n_unc = report.skipped_unchanged.len();
    let n_err = report.errors.len();
    let n_oth = report.skipped_other;
    let n_sym = report.symlinks_skipped;

    if n_ing == 0 && n_unc == 0 && n_err == 0 && n_oth == 0 && n_sym == 0 {
        return format!(
            "Wiki seed: nothing to ingest under {}/. Try: /wiki seed src to widen scope.",
            dir
        );
    }

    // Special: only symlinks encountered. Distinguish from the generic
    // "nothing under dir" line so the operator sees the safety skip.
    if n_ing == 0 && n_unc == 0 && n_err == 0 && n_oth == 0 && n_sym > 0 {
        return format!(
            "Wiki seed: nothing ingested under {}/ ({} symlinks skipped). \
             Try: /wiki seed src to widen scope.",
            dir, n_sym
        );
    }

    let symlink_clause = if n_sym > 0 {
        format!(", {} symlinks skipped", n_sym)
    } else {
        String::new()
    };
    let mut out = format!(
        "Seeded {}/: {} ingested, {} unchanged, {} skipped{}, {} errors.",
        dir, n_ing, n_unc, n_oth, symlink_clause, n_err,
    );
    const SHOW_INGESTED: usize = 10;
    const SHOW_ERRORS: usize = 5;
    for rel in report.ingested.iter().take(SHOW_INGESTED) {
        write!(out, "\n  [ok] {}", rel).expect("write to String never fails");
    }
    if n_ing > SHOW_INGESTED {
        write!(out, "\n  … {} more ingested", n_ing - SHOW_INGESTED)
            .expect("write to String never fails");
    }
    for (rel, msg) in report.errors.iter().take(SHOW_ERRORS) {
        write!(out, "\n  [err] {}: {}", rel, msg).expect("write to String never fails");
    }
    if n_err > SHOW_ERRORS {
        write!(out, "\n  … {} more errors", n_err - SHOW_ERRORS)
            .expect("write to String never fails");
    }
    out
}

/// Dispatcher for `/wiki seed [<dir>]` — operator-deliberate bulk seed
/// of source files into entity pages. Mutates only `.dm/wiki/`; never
/// touches the source files themselves. Default dir is `src` so a
/// no-arg invocation matches the typical Rust project layout. For
/// re-ingest of *existing* entity pages whose sources drifted, use
/// `/wiki refresh` instead — it's index-walking, not filesystem-walking.
pub(super) fn handle_wiki_seed(arg: Option<String>) -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    let dir = arg.as_deref().unwrap_or("src");
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => {
            match wiki.seed_dir(std::path::Path::new(dir), &["rs"]) {
                Ok(report) => SlashResult::Info(format_wiki_seed(&report, dir)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => SlashResult::Error(format!(
                    "wiki seed failed: {}. Try: re-run from the project root, or pass an existing relative dir.",
                    e
                )),
                Err(e) => SlashResult::Error(format!(
                    "wiki seed failed: {}. Try: check filesystem permissions on {}/.",
                    e, dir
                )),
            }
        }
        Err(e) => SlashResult::Error(format!(
            "wiki open failed: {}. Try: cd into a project with .dm/wiki/, or run /wiki status.",
            e
        )),
    }
}

/// Dispatcher for `/wiki prune [<N>]` — operator-triggered cap on the
/// `synthesis/compact-*.md` backlog. Reuses the same method that
/// `write_compact_synthesis` calls automatically; surfaces it for one-shot
/// cleanup of historical accumulations that predate the cap.
pub(super) fn handle_wiki_prune(max_keep: usize) -> SlashResult {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(e) => {
            return SlashResult::Error(format!(
                "cannot read cwd: {}. Try: re-run from a directory you can access.",
                e
            ))
        }
    };
    match crate::wiki::Wiki::open(&cwd) {
        Ok(wiki) => match wiki.prune_compact_synthesis_to(max_keep) {
            Ok(pruned) => SlashResult::Info(format_wiki_prune(pruned, max_keep)),
            Err(e) => SlashResult::Error(format!(
                "wiki prune failed: {}. Try: check write permissions on .dm/wiki/synthesis/.",
                e
            )),
        },
        Err(e) => SlashResult::Error(format!(
            "wiki open failed: {}. Try: cd into a project with .dm/wiki/, or run /wiki refresh.",
            e
        )),
    }
}
