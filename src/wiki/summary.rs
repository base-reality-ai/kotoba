//! Summary + momentum + planner-brief rendering.
//!
//! Holds the `PlannerBrief` value type and its Markdown render path. The
//! filesystem-facing constructor (`Wiki::planner_brief`) stays on the
//! `Wiki` struct in `mod.rs`; this module holds only the pure rendering
//! logic so the behavior is unit-testable without a `Wiki` instance.
//! Holds the full project-comprehension surface via split-impl:
//! `build_project_summary` + `write_project_summary`, `momentum`,
//! `planner_brief`, the concept-pages producer/consumer pair, and the
//! three model-facing snippet emitters (`context_snippet`,
//! `project_summary_snippet`, `fresh_pages_snippet`).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::{fs, io};

use super::{
    concept_body_sans_timestamp, sanitize_dep_for_path, ConceptPagesReport, EntityKind, IndexEntry,
    MomentumReport, PageType, ProjectSummaryReport, RecentCycle, Wiki, WikiLintKind, WikiPage,
    CONCEPT_DEP_MIN_OCCURRENCES, CONTEXT_SNIPPET_MAX_BYTES, INDEX_LOCK, MOMENTUM_DEFAULT_WINDOW,
    MOMENTUM_TOP_N, PLANNER_BRIEF_FRESH_HOURS, PLANNER_BRIEF_FRESH_PAGES, PLANNER_BRIEF_HOT_PATHS,
    PLANNER_BRIEF_RECENT_CYCLES, PLANNER_BRIEF_STALE_HOURS, PROJECT_SUMMARY_RECENT_CYCLES,
};

/// Filename of the project-summary dirty marker, sibling to `index.md`
/// inside `.dm/wiki/`. Ingest touches it; session-start and `refresh()`
/// lazy-regenerate when set. Collapses bulk ingest to one summary write.
const SUMMARY_DIRTY_MARKER: &str = ".summary-dirty";

/// Structured briefing for chain `planner`-role nodes. Composes the three
/// wiki signals a planner cares about at cycle start: ingest momentum
/// (where activity is concentrated), entity pages whose body or
/// frontmatter has drifted from source, and a lint-kind tally so the
/// planner can see which contradiction classes are open.
///
/// `Default` is the empty-wiki shape. [`PlannerBrief::render`] returns
/// `None` for an empty brief so the runner can skip the section entirely
/// rather than emit a hollow header.
#[derive(Debug, Default, Clone)]
pub struct PlannerBrief {
    /// Top source paths by ingest frequency — piggybacks
    /// [`Wiki::momentum`]'s window + ordering, capped at
    /// [`PLANNER_BRIEF_HOT_PATHS`].
    pub hot_paths: Vec<(String, usize)>,
    /// Top source modules (path prefixes) by ingest frequency — forwards
    /// [`MomentumReport::hot_modules`] so the planner sees the coarse
    /// "which subsystem is hot" signal alongside the fine-grained
    /// `hot_paths`. Already ordered + capped by [`Wiki::momentum`].
    pub hot_modules: Vec<(String, usize)>,
    /// Entity page paths flagged by [`WikiLintKind::ItemDrift`] or
    /// [`WikiLintKind::ExportDrift`]. Deduped (one page can drift in
    /// both dimensions) and capped at `max_findings` (threaded through
    /// from [`Wiki::planner_brief`]'s parameter).
    pub drifting_pages: Vec<String>,
    /// All lint kinds observed this run with counts. Sorted by count
    /// desc, then kind ascending — gives the planner a stable,
    /// priority-ordered list without re-running `/wiki lint`.
    pub lint_counts: Vec<(WikiLintKind, usize)>,
    /// Recent chain-cycle synthesis pages surfaced so the planner can
    /// pick up where the last incubation cycle left off. Newest-first,
    /// capped at [`PLANNER_BRIEF_RECENT_CYCLES`]. Populated from
    /// `synthesis/cycle-*.md` index entries; compact synthesis pages
    /// are filtered out.
    pub recent_cycles: Vec<RecentCycle>,
    /// Entity + concept pages whose `last_updated` frontmatter is within
    /// [`PLANNER_BRIEF_FRESH_HOURS`] of composition time. Newest-first,
    /// capped at [`PLANNER_BRIEF_FRESH_PAGES`]. `(page_path, last_updated)`
    /// tuples; synthesis / summary / index entries are filtered out so
    /// the signal complements `recent_cycles` rather than duplicating it.
    pub fresh_pages: Vec<(String, String)>,
}

/// Parse `last_updated` (format `%Y-%m-%d %H:%M:%S` local time, per
/// `write_project_summary` / `write_cycle_synthesis`) and return the age
/// in hours vs `now`. `None` on parse failure or negative delta (clock
/// skew) — both degrade to "not stale" for the render check.
pub(super) fn cycle_age_hours(last_updated: &str, now: chrono::NaiveDateTime) -> Option<i64> {
    let ts = chrono::NaiveDateTime::parse_from_str(last_updated, "%Y-%m-%d %H:%M:%S").ok()?;
    let hours = (now - ts).num_hours();
    if hours < 0 {
        None
    } else {
        Some(hours)
    }
}

impl PlannerBrief {
    /// `true` when every field is empty — the runner uses this to skip
    /// injecting the section entirely on a fresh / silent wiki.
    pub fn is_empty(&self) -> bool {
        self.hot_paths.is_empty()
            && self.hot_modules.is_empty()
            && self.drifting_pages.is_empty()
            && self.lint_counts.is_empty()
            && self.recent_cycles.is_empty()
            && self.fresh_pages.is_empty()
    }

    /// Render as a `## Planner Brief` markdown section. Returns `None`
    /// when the brief is empty so callers can skip the append without an
    /// `is_empty()` precondition. `budget_chars` is a soft cap enforced
    /// at line boundaries — overflow appends `[...truncated]` on its own
    /// line so the planner never sees a half-cut bullet.
    pub fn render(&self, budget_chars: usize) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut out = String::from("## Planner Brief\n\n");
        // Staleness warning: if the newest recent-cycle synthesis page's
        // `last_updated` parses as >= PLANNER_BRIEF_STALE_HOURS old, nudge
        // the planner to refresh the wiki before planning. All failure
        // modes (missing cycle, missing timestamp, parse error, clock
        // skew) degrade silently to "not stale".
        if let Some(rc) = self.recent_cycles.first() {
            if let Some(ts) = &rc.last_updated {
                let now = chrono::Local::now().naive_local();
                if let Some(age) = cycle_age_hours(ts, now) {
                    if age >= PLANNER_BRIEF_STALE_HOURS {
                        // The "may be stale" phrase here is intentional UX
                        // prose — NOT the structured `[wiki-drift]` marker
                        // counted by `crate::telemetry`. See
                        // `src/tools/registry.rs::call` for the counter trigger.
                        write!(
                            out,
                            "⚠️  Latest cycle is {}h old — wiki context may be stale. \
                             Consider `/wiki refresh` before planning.\n\n",
                            age,
                        )
                        .expect("write to String never fails");
                    }
                }
            }
        }
        if !self.hot_paths.is_empty() {
            out.push_str("### Hot paths (per file)\n");
            for (path, n) in &self.hot_paths {
                writeln!(out, "- `{}` — {} ingest(s)", path, n)
                    .expect("write to String never fails");
            }
            out.push('\n');
        }
        if !self.hot_modules.is_empty() {
            out.push_str("### Hot modules (per directory)\n");
            for (module, n) in &self.hot_modules {
                writeln!(out, "- `{}` — {} ingest(s)", module, n)
                    .expect("write to String never fails");
            }
            out.push('\n');
        }
        if !self.drifting_pages.is_empty() {
            out.push_str("### Drifting pages\n");
            for page in &self.drifting_pages {
                writeln!(out, "- `{}`", page).expect("write to String never fails");
            }
            out.push('\n');
        }
        if !self.lint_counts.is_empty() {
            out.push_str("### Open lint findings\n");
            for (kind, n) in &self.lint_counts {
                writeln!(out, "- {:?}: {}", kind, n).expect("write to String never fails");
            }
            out.push('\n');
        }
        if !self.recent_cycles.is_empty() {
            out.push_str("### Recent cycles\n");
            for rc in &self.recent_cycles {
                write!(
                    out,
                    "- Cycle {} ({}): `.dm/wiki/{}`",
                    rc.cycle, rc.chain, rc.page_path
                )
                .expect("write to String never fails");
                if let Some(ts) = &rc.last_updated {
                    write!(out, " — {}", ts).expect("write to String never fails");
                }
                if let Some(outcome) = &rc.outcome {
                    write!(out, " [{}]", outcome).expect("write to String never fails");
                }
                out.push('\n');
            }
            out.push('\n');
        }
        if !self.fresh_pages.is_empty() {
            out.push_str("### Fresh entity pages (updated <24h)\n");
            for (path, ts) in &self.fresh_pages {
                writeln!(out, "- `.dm/wiki/{}` — {}", path, ts)
                    .expect("write to String never fails");
            }
            out.push('\n');
        }
        let trimmed = out.trim_end().to_string();
        if trimmed.len() <= budget_chars {
            return Some(trimmed);
        }
        // Line-boundary truncation: scan forward accumulating whole
        // lines until the next line would overflow, then append the
        // truncation marker. Mirrors `project_summary_snippet`'s approach.
        let marker = "\n[...truncated]";
        let room = budget_chars.saturating_sub(marker.len());
        let mut kept = String::with_capacity(room);
        for line in trimmed.lines() {
            if kept.len() + line.len() + 1 > room {
                break;
            }
            if !kept.is_empty() {
                kept.push('\n');
            }
            kept.push_str(line);
        }
        kept.push_str(marker);
        Some(kept)
    }
}

impl Wiki {
    /// Build an in-memory project summary page by aggregating `purpose`,
    /// `entity_kind`, and `dependencies` across every Entity page in the
    /// index. Pure: does not write to disk — see
    /// [`Wiki::write_project_summary`] for the persisting variant.
    ///
    /// Entity pages that fail to parse or read are silently skipped (the
    /// `/wiki lint` `MalformedPage` rule surfaces them separately). Empty
    /// wikis produce a well-formed page with placeholder sections rather
    /// than an error.
    pub fn build_project_summary(&self) -> io::Result<WikiPage> {
        use std::collections::BTreeMap;
        let idx = self.load_index()?;
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let entity_pages: Vec<(String, WikiPage)> = idx
            .entries
            .iter()
            .filter(|e| e.category == PageType::Entity)
            .filter_map(|e| {
                let abs = self.root.join(&e.path);
                let text = fs::read_to_string(&abs).ok()?;
                let page = WikiPage::parse(&text)?;
                Some((e.path.clone(), page))
            })
            .collect();

        let mut purposes: Vec<String> = Vec::new();
        let mut kind_counts: BTreeMap<EntityKind, usize> = BTreeMap::new();
        let mut dep_freq: HashMap<String, usize> = HashMap::new();
        let mut source_paths: Vec<String> = Vec::new();

        for (_path, page) in &entity_pages {
            if let Some(kind) = page.entity_kind {
                *kind_counts.entry(kind).or_insert(0) += 1;
            }
            if let Some(p) = &page.purpose {
                purposes.push(format!("- **{}** — {}", page.title, p));
            }
            for dep in &page.dependencies {
                *dep_freq.entry(dep.clone()).or_insert(0) += 1;
            }
            for source in &page.sources {
                if !source_paths.contains(source) {
                    source_paths.push(source.clone());
                }
            }
        }

        let purposes_section: String = if purposes.is_empty() {
            "*(no entity pages with `purpose` yet — run `/wiki refresh` after ingesting Rust sources)*".into()
        } else {
            purposes
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        };

        let arch_section: String = {
            let mut parts: Vec<String> = kind_counts
                .iter()
                .map(|(k, n)| format!("{}: {}", k.as_str(), n))
                .collect();
            if parts.is_empty() {
                parts.push("*(no entity kinds detected yet)*".into());
            }
            let mut top_deps: Vec<(String, usize)> =
                dep_freq.iter().map(|(k, v)| (k.clone(), *v)).collect();
            top_deps.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let deps_line = if top_deps.is_empty() {
                String::from("*(no tracked dependencies)*")
            } else {
                top_deps
                    .iter()
                    .take(10)
                    .map(|(d, n)| format!("`{}` ({})", d, n))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!(
                "{}\n\nMost-cited dependencies: {}",
                parts.join(", "),
                deps_line
            )
        };

        let momentum_section: String = {
            let report = self.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap_or_default();
            if report.window_processed == 0 {
                "*(no ingest activity yet — run `/wiki refresh` or ingest a source file)*".into()
            } else {
                let paths_line = report
                    .hot_paths
                    .iter()
                    .map(|(p, n)| format!("`{}` ({})", p, n))
                    .collect::<Vec<_>>()
                    .join(", ");
                let modules_line = report
                    .hot_modules
                    .iter()
                    .map(|(m, n)| format!("`{}` ({})", m, n))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Hot paths (last {} ingests): {}\n\nHot modules: {}",
                    report.window_processed, paths_line, modules_line,
                )
            }
        };

        let recent_cycles_section: String = {
            let cycles = self.collect_recent_cycles(PROJECT_SUMMARY_RECENT_CYCLES);
            if cycles.is_empty() {
                "*(no synthesis pages yet — chains haven't written cycle outcomes)*".into()
            } else {
                cycles
                    .iter()
                    .map(|rc| match &rc.outcome {
                        Some(o) => format!("- cycle {} ({}) — `{}`", rc.cycle, o, rc.chain),
                        None => format!("- cycle {} — `{}`", rc.cycle, rc.chain),
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        let body = format!(
            "# Project\n\n\
             Last updated: {}\n\n\
             ## Purpose\n\n{}\n\n\
             ## Recent cycles\n\n{}\n\n\
             ## Architecture\n\n{}\n\n\
             ## Momentum\n\n{}\n\n\
             ---\n\nGenerated from {} entity page(s).\n",
            now,
            purposes_section,
            recent_cycles_section,
            arch_section,
            momentum_section,
            entity_pages.len(),
        );

        Ok(WikiPage {
            title: "Project".to_string(),
            page_type: PageType::Summary,
            layer: crate::wiki::Layer::Kernel,
            sources: source_paths,
            last_updated: now,
            entity_kind: None,
            outcome: None,
            purpose: None,
            key_exports: Vec::new(),
            dependencies: Vec::new(),
            scope: vec![],
            body,
        })
    }

    /// Generate and persist the project summary page at
    /// `summaries/project.md`. Upserts the index entry, appends a
    /// `summary` verb to the log, and returns a [`ProjectSummaryReport`]
    /// with the derived aggregates for the caller to render.
    pub fn write_project_summary(&self) -> io::Result<ProjectSummaryReport> {
        use std::collections::BTreeMap;
        let page = self.build_project_summary()?;
        let rel = "summaries/project.md".to_string();
        self.write_page(&rel, &page)?;

        {
            let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut idx = self.load_index().unwrap_or_default();
            let entry = IndexEntry {
                title: page.title.clone(),
                path: rel.clone(),
                one_liner: format!("Project summary ({} entities)", page.sources.len()),
                category: PageType::Summary,
                last_updated: Some(page.last_updated),
                outcome: None,
            };
            if let Some(existing) = idx.entries.iter_mut().find(|e| e.path == rel) {
                *existing = entry;
            } else {
                idx.entries.push(entry);
            }
            self.save_index(&idx)?;
        }

        self.log().append("summary", &rel)?;

        // Re-derive stats for the report so the caller doesn't have to
        // re-parse the page. The index is read a second time to pick up
        // the summary entry we just wrote (not counted as an entity).
        let idx = self.load_index()?;
        let mut kind_counts: BTreeMap<EntityKind, usize> = BTreeMap::new();
        let mut dep_freq: HashMap<String, usize> = HashMap::new();
        let mut entity_count = 0usize;
        for entry in &idx.entries {
            if entry.category != PageType::Entity {
                continue;
            }
            let Ok(text) = fs::read_to_string(self.root.join(&entry.path)) else {
                continue;
            };
            let Some(p) = WikiPage::parse(&text) else {
                continue;
            };
            entity_count += 1;
            if let Some(k) = p.entity_kind {
                *kind_counts.entry(k).or_insert(0) += 1;
            }
            for d in &p.dependencies {
                *dep_freq.entry(d.clone()).or_insert(0) += 1;
            }
        }
        let mut top_dependencies: Vec<(String, usize)> = dep_freq.into_iter().collect();
        top_dependencies.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top_dependencies.truncate(10);

        let recent_cycles = self.collect_recent_cycles(PROJECT_SUMMARY_RECENT_CYCLES);

        // Explicit path (`/wiki summary`) resolves staleness even when the
        // caller bypassed `ensure_summary_current`. Pairs belt-and-suspenders
        // with the `ensure_summary_current` clear call.
        let _ = self.clear_summary_dirty();

        Ok(ProjectSummaryReport {
            path: rel,
            entity_count,
            kind_counts,
            top_dependencies,
            recent_cycles,
        })
    }

    /// Aggregate the last `window` `ingest`-verb entries from `log.md`
    /// into hot-path and hot-module frequency reports. Count-window (not
    /// time-window) because log timestamps are wall-clock and tests need
    /// deterministic behaviour. Non-ingest verbs (`summary`, `concept`,
    /// `compact`) are skipped — momentum is a source-churn signal, not
    /// an all-verb tail.
    ///
    /// A missing or unreadable `log.md` yields a [`MomentumReport::default`]
    /// (zero counts, empty vectors) rather than an error — the project
    /// summary's `## Momentum` section is rendered from this report and
    /// must degrade gracefully.
    ///
    /// Malformed lines (no `]` closer, no `|` between verb and subject)
    /// are silently skipped. `total_entries` counts every line read;
    /// `window_processed` counts only the ingest entries that passed
    /// parsing and contributed to the aggregates.
    pub fn momentum(&self, window: usize) -> io::Result<MomentumReport> {
        let log_path = self.log().path().to_path_buf();
        let text = match fs::read_to_string(&log_path) {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(MomentumReport::default()),
            Err(e) => return Err(e),
        };

        let all_lines: Vec<&str> = text.lines().collect();
        let total_entries = all_lines.len();

        let mut path_counts: HashMap<String, usize> = HashMap::new();
        let mut module_counts: HashMap<String, usize> = HashMap::new();
        let mut window_processed = 0usize;

        // Walk newest-first; stop once we've collected `window` ingest hits.
        for line in all_lines.iter().rev() {
            if window_processed >= window {
                break;
            }
            let Some(rest) = line.strip_prefix('[') else {
                continue;
            };
            let Some(close) = rest.find(']') else {
                continue;
            };
            let after = rest[close + 1..].trim_start();
            let Some(sep) = after.find('|') else {
                continue;
            };
            let verb = after[..sep].trim();
            if verb != "ingest" {
                continue;
            }
            let subject = after[sep + 1..].trim();
            if subject.is_empty() {
                continue;
            }
            *path_counts.entry(subject.to_string()).or_insert(0) += 1;
            *module_counts
                .entry(momentum_module_of(subject))
                .or_insert(0) += 1;
            window_processed += 1;
        }

        let mut hot_paths: Vec<(String, usize)> = path_counts.into_iter().collect();
        hot_paths.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hot_paths.truncate(MOMENTUM_TOP_N);

        let mut hot_modules: Vec<(String, usize)> = module_counts.into_iter().collect();
        hot_modules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        hot_modules.truncate(MOMENTUM_TOP_N);

        Ok(MomentumReport {
            total_entries,
            window_processed,
            hot_paths,
            hot_modules,
        })
    }

    /// Compose a [`PlannerBrief`] from momentum + lint so chain planner
    /// nodes start each cycle with a surfaced, structured picture of
    /// "where activity is" and "where comprehension has drifted".
    ///
    /// `window` is threaded through to [`Wiki::momentum`]; `max_findings`
    /// caps `drifting_pages`. Both momentum and lint errors degrade
    /// gracefully — a missing wiki / missing log / unparsable page yields
    /// an empty field, never an error, because the brief is best-effort
    /// context for a role whose next step is planning anyway.
    pub fn planner_brief(&self, window: usize, max_findings: usize) -> io::Result<PlannerBrief> {
        let mut brief = PlannerBrief::default();

        // Pull both hot_paths + hot_modules off the same momentum scan so
        // the planner sees fine-grained (paths) and coarse (modules)
        // views of the same activity window without a second log walk.
        let momentum = self.momentum(window).unwrap_or_default();
        brief.hot_modules = momentum
            .hot_modules
            .iter()
            .take(PLANNER_BRIEF_HOT_PATHS)
            .cloned()
            .collect();
        brief.hot_paths = momentum
            .hot_paths
            .into_iter()
            .take(PLANNER_BRIEF_HOT_PATHS)
            .collect();

        let findings = self.lint().unwrap_or_default();

        // `drifting_pages`: dedup while preserving iteration order — a
        // page drifting in both dimensions (ItemDrift + ExportDrift) is
        // one entry, not two. `findings` is already sorted by
        // `(kind as u8, path)` so the natural order here is ItemDrift
        // pages (asc) then ExportDrift pages (asc).
        let mut seen = std::collections::HashSet::new();
        for f in &findings {
            if matches!(f.kind, WikiLintKind::ItemDrift | WikiLintKind::ExportDrift)
                && seen.insert(f.path.clone())
            {
                brief.drifting_pages.push(f.path.clone());
                if brief.drifting_pages.len() >= max_findings {
                    break;
                }
            }
        }

        // `lint_counts`: tally every finding's kind, then sort count
        // desc / kind asc so the planner sees the loudest contradiction
        // class first. Uses HashMap + explicit sort rather than BTreeMap
        // because tie-break is count-first (BTreeMap would tie-break
        // kind-first).
        let mut counts: HashMap<WikiLintKind, usize> = HashMap::new();
        for f in &findings {
            *counts.entry(f.kind).or_insert(0) += 1;
        }
        let mut lint_counts: Vec<(WikiLintKind, usize)> = counts.into_iter().collect();
        lint_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| (a.0 as u8).cmp(&(b.0 as u8))));
        brief.lint_counts = lint_counts;

        // `recent_cycles`: delegated to [`Wiki::collect_recent_cycles`] so
        // the cache-elision logic stays in one place. `fresh_pages` below
        // still scans the index directly — it filters on a different
        // category set (Entity + Concept) and doesn't benefit from the
        // synthesis-specific helper.
        brief.recent_cycles = self.collect_recent_cycles(PLANNER_BRIEF_RECENT_CYCLES);

        // `fresh_pages` scans entity + concept entries; load the index
        // once (separate from the helper's internal load — bounded cost,
        // one extra parse of the already-warm file).
        let index = self.load_index().unwrap_or_default();

        // `fresh_pages`: entity + concept entries whose `last_updated` is
        // within PLANNER_BRIEF_FRESH_HOURS of now. Synthesis + Summary
        // are excluded by category (synthesis surfaces via recent_cycles;
        // summary is the global project-summary, not per-entity signal).
        // A missing file / malformed frontmatter / negative age (clock
        // skew) degrades via `filter_map` drop so the brief never errors.
        let now = chrono::Local::now().naive_local();
        let mut fresh: Vec<(String, String)> = index
            .entries
            .iter()
            .filter(|e| matches!(e.category, PageType::Entity | PageType::Concept))
            .filter_map(|e| {
                // Cache-hit path (C85): `last_updated` is already in the
                // index — no disk I/O. Legacy-index fallback reads the
                // page once and self-heals on next ingest.
                let ts = match &e.last_updated {
                    Some(ts) => ts.clone(),
                    None => std::fs::read_to_string(self.root.join(&e.path))
                        .ok()
                        .and_then(|text| WikiPage::parse(&text))
                        .map(|page| page.last_updated)?,
                };
                let age = cycle_age_hours(&ts, now)?;
                if age < PLANNER_BRIEF_FRESH_HOURS {
                    Some((e.path.clone(), ts))
                } else {
                    None
                }
            })
            .collect();
        // Sort newest-first by timestamp (lexical order == chronological
        // for `%Y-%m-%d %H:%M:%S`), tie-break on page path asc for
        // determinism.
        fresh.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        fresh.truncate(PLANNER_BRIEF_FRESH_PAGES);
        brief.fresh_pages = fresh;

        Ok(brief)
    }

    /// Build concept pages for dependencies shared by ≥
    /// `CONCEPT_DEP_MIN_OCCURRENCES` entity pages. Pure: does not write
    /// to disk — see [`Wiki::write_concept_pages`] for the persisting
    /// variant.
    ///
    /// Returned tuples are `(relative_path, page)` where `relative_path`
    /// is `concepts/dep-{slug}.md`. Grouped imports (see
    /// `sanitize_dep_for_path`) are excluded. Output order is
    /// freq-desc then path-asc so callers can snapshot-compare.
    pub fn build_concept_pages(&self) -> io::Result<Vec<(String, WikiPage)>> {
        let idx = self.load_index()?;
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        let mut dep_consumers: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for entry in idx
            .entries
            .iter()
            .filter(|e| e.category == PageType::Entity)
        {
            let abs = self.root.join(&entry.path);
            let Ok(text) = fs::read_to_string(&abs) else {
                continue;
            };
            let Some(page) = WikiPage::parse(&text) else {
                continue;
            };
            for dep in &page.dependencies {
                dep_consumers
                    .entry(dep.clone())
                    .or_default()
                    .push((page.title.clone(), entry.path.clone()));
            }
        }

        let mut shared: Vec<(String, Vec<(String, String)>)> = dep_consumers
            .into_iter()
            .filter(|(dep, consumers)| {
                consumers.len() >= CONCEPT_DEP_MIN_OCCURRENCES
                    && sanitize_dep_for_path(dep).is_some()
            })
            .collect();
        shared.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

        let mut out: Vec<(String, WikiPage)> = Vec::with_capacity(shared.len());
        for (dep, mut consumers) in shared {
            consumers.sort_by(|a, b| a.1.cmp(&b.1));
            let Some(slug) = sanitize_dep_for_path(&dep) else {
                continue;
            };
            let rel = format!("concepts/dep-{}.md", slug);
            let title = format!("Shared dependency: {}", dep);
            let bullets: String = consumers
                .iter()
                .map(|(t, p)| format!("- [{}]({}) — `{}`", t, p, p))
                .collect::<Vec<_>>()
                .join("\n");
            let body = format!(
                "# {}\n\n\
                 Last updated: {}\n\n\
                 ## Used by\n\n{}\n\n\
                 ---\n\n\
                 Auto-detected: {} entity page(s) depend on `{}`. \
                 Threshold is {} shared consumers.\n",
                title,
                now,
                bullets,
                consumers.len(),
                dep,
                CONCEPT_DEP_MIN_OCCURRENCES,
            );
            let sources: Vec<String> = consumers.iter().map(|(_, p)| p.clone()).collect();
            out.push((
                rel,
                WikiPage {
                    title,
                    page_type: PageType::Concept,
                    layer: crate::wiki::Layer::Kernel,
                    sources,
                    last_updated: now.clone(),
                    entity_kind: None,
                    purpose: None,
                    key_exports: Vec::new(),
                    dependencies: Vec::new(),
                    outcome: None,
                    scope: vec![],
                    body,
                },
            ));
        }
        Ok(out)
    }

    /// Generate and persist all detected concept pages. Upserts index
    /// entries under `INDEX_LOCK`, appends one `concept` log line per
    /// changed page, and marks the project summary dirty iff any page
    /// was newly created or had its body change. Returns a
    /// [`ConceptPagesReport`].
    ///
    /// Idempotent: re-running on an unchanged wiki produces an empty
    /// `generated`/`refreshed` set and does not touch the dirty marker.
    /// Write-avoidance strips timestamp lines before comparison — a run
    /// whose `now` differs from the previous run's timestamp but whose
    /// consumer set is unchanged is correctly treated as a noop.
    /// Hand-written concept pages not matching `concepts/dep-*.md` are
    /// never touched — this writer only manages auto-detected pages.
    pub fn write_concept_pages(&self) -> io::Result<ConceptPagesReport> {
        let pages = self.build_concept_pages()?;

        let mut generated: Vec<String> = Vec::new();
        let mut refreshed: Vec<String> = Vec::new();
        let mut detected_deps: Vec<(String, usize)> = pages
            .iter()
            .map(|(_, p)| {
                (
                    p.title
                        .trim_start_matches("Shared dependency: ")
                        .to_string(),
                    p.sources.len(),
                )
            })
            .collect();

        {
            let _idx_guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let mut idx = self.load_index().unwrap_or_default();

            for (rel, page) in &pages {
                let abs = self.root.join(rel);
                let existed = abs.is_file();
                let prev = if existed {
                    fs::read_to_string(&abs).ok()
                } else {
                    None
                };
                let new_text = page.to_markdown();
                let changed = match prev.as_deref() {
                    Some(existing) => {
                        concept_body_sans_timestamp(existing)
                            != concept_body_sans_timestamp(&new_text)
                    }
                    None => true,
                };
                if changed {
                    self.write_page(rel, page)?;
                    if existed {
                        refreshed.push(rel.clone());
                    } else {
                        generated.push(rel.clone());
                    }
                }

                let entry = IndexEntry {
                    title: page.title.clone(),
                    path: rel.clone(),
                    one_liner: format!(
                        "{} entity page(s) share this dependency",
                        page.sources.len()
                    ),
                    category: PageType::Concept,
                    last_updated: Some(page.last_updated.clone()),
                    outcome: None,
                };
                if let Some(existing) = idx.entries.iter_mut().find(|e| e.path == *rel) {
                    *existing = entry;
                } else {
                    idx.entries.push(entry);
                }
            }
            self.save_index(&idx)?;
        }

        for rel in generated.iter().chain(refreshed.iter()) {
            self.log().append("concept", rel)?;
        }

        if !generated.is_empty() || !refreshed.is_empty() {
            let _ = self.mark_summary_dirty();
        }

        detected_deps.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        Ok(ConceptPagesReport {
            generated,
            refreshed,
            detected_deps,
        })
    }

    /// Build a compact, model-facing summary of the wiki's current contents.
    /// Returns `None` when the wiki is empty or the index is unreadable, so
    /// callers can skip prompt injection without branching on errors.
    ///
    /// The snippet is bounded by `CONTEXT_SNIPPET_MAX_BYTES` on the body;
    /// a trailing "… N more page(s) not shown" line is appended when the
    /// body was truncated, so callers always see that the listing is partial.
    ///
    /// Identity is loaded from `.dm/identity.toml` under the project root,
    /// defaulting to kernel mode on any failure so the chain self-heals on a
    /// hand-edited file. Use [`Self::context_snippet_for`] when the caller
    /// already holds an `Identity`.
    pub fn context_snippet(&self) -> Option<String> {
        // `self.root()` is `<project>/.dm/wiki/`; `identity::load_at` wants the
        // *project* root (it appends `.dm/identity.toml`), so pass `project_root()`.
        // Earlier draft used `self.root()` and silently regressed to kernel
        // mode in every host project — the wiki snippet header would always
        // say "dark-matter Wiki" because the looked-up `.dm/wiki/.dm/identity.toml`
        // path doesn't exist and `load_at` returns the kernel default.
        let identity = crate::identity::load_at(&self.project_root()).unwrap_or_else(|e| {
            crate::warnings::push_warning(format!("identity: {}", e));
            crate::identity::Identity::default_kernel()
        });
        self.context_snippet_for(&identity)
    }

    /// Identity-explicit variant of [`Self::context_snippet`]. Host-mode
    /// callers see each category split into a host block and a kernel block
    /// separated by `---`; kernel-mode callers get the legacy flat list.
    /// Layer is read per-page from disk in host mode (kernel mode skips
    /// the per-page read so it pays no cost for the layer-aware path).
    pub fn context_snippet_for(&self, identity: &crate::identity::Identity) -> Option<String> {
        let idx = self.load_index().ok()?;
        if idx.entries.is_empty() {
            return None;
        }
        let display_name = identity.display_name();
        let stratify = identity.is_host();

        let mut out = format!(
            "## {} Wiki\n\n\
             This project has a persistent wiki at `.dm/wiki/`. Before \
             guessing about the codebase, query the wiki: use `wiki_search` \
             for keyword discovery and `wiki_lookup` for a specific page \
             (both tolerate typos via fuzzy fallback, and surface a \
             `[wiki-drift]` marker when a touched source file is newer \
             than its page). Pages listed below:\n",
            display_name
        );
        let mut truncated = 0usize;
        for cat in [
            PageType::Entity,
            PageType::Concept,
            PageType::Summary,
            PageType::Synthesis,
        ] {
            let cat_entries: Vec<&IndexEntry> =
                idx.entries.iter().filter(|e| e.category == cat).collect();
            if cat_entries.is_empty() {
                continue;
            }
            let header = format!("\n### {}\n", cat.as_str());
            if out.len() + header.len() > CONTEXT_SNIPPET_MAX_BYTES {
                truncated += cat_entries.len();
                continue;
            }
            out.push_str(&header);

            if stratify {
                // Read each page once to recover its layer; pages whose body
                // or frontmatter we cannot parse default to Kernel so the
                // entry still surfaces (degraded ordering is preferable to
                // silently dropping the page from the snippet).
                let (host_entries, kernel_entries): (Vec<&IndexEntry>, Vec<&IndexEntry>) =
                    cat_entries
                        .into_iter()
                        .partition(|e| self.entry_layer(e) == super::Layer::Host);

                let mut wrote_host = false;
                for e in &host_entries {
                    let line = format!("- {}: {}\n", e.title, e.one_liner);
                    if out.len() + line.len() > CONTEXT_SNIPPET_MAX_BYTES {
                        truncated += 1;
                        continue;
                    }
                    out.push_str(&line);
                    wrote_host = true;
                }
                if wrote_host && !kernel_entries.is_empty() {
                    let sep = "\n---\n";
                    if out.len() + sep.len() <= CONTEXT_SNIPPET_MAX_BYTES {
                        out.push_str(sep);
                    }
                }
                for e in &kernel_entries {
                    let line = format!("- {}: {}\n", e.title, e.one_liner);
                    if out.len() + line.len() > CONTEXT_SNIPPET_MAX_BYTES {
                        truncated += 1;
                        continue;
                    }
                    out.push_str(&line);
                }
            } else {
                for e in &cat_entries {
                    let line = format!("- {}: {}\n", e.title, e.one_liner);
                    if out.len() + line.len() > CONTEXT_SNIPPET_MAX_BYTES {
                        truncated += 1;
                        continue;
                    }
                    out.push_str(&line);
                }
            }
        }
        if truncated > 0 {
            // Tail is appended unconditionally — readers must know the
            // listing is partial even if that nudges us slightly past the
            // body cap.
            writeln!(
                out,
                "\n… {} more page(s) not shown; read `.dm/wiki/index.md` for the full catalog.",
                truncated
            )
            .expect("write to String never fails");
        }
        Some(out)
    }

    /// Read a page's `Layer` from disk. Falls back to `Kernel` on any
    /// read/parse failure so a malformed page still surfaces in the snippet
    /// (in its inherited-default position) rather than disappearing
    /// silently. Used by host-mode `context_snippet_for` only — in kernel
    /// mode, every entry is implicitly Kernel and the read is skipped.
    fn entry_layer(&self, entry: &IndexEntry) -> super::Layer {
        let path = self.root.join(&entry.path);
        match fs::read_to_string(&path) {
            Ok(text) => WikiPage::parse(&text)
                .map(|p| p.layer)
                .unwrap_or(super::Layer::Kernel),
            Err(_) => super::Layer::Kernel,
        }
    }

    /// Load the project summary page body for system-prompt injection.
    /// Returns `None` when `summaries/project.md` is absent, unreadable,
    /// or the page body is empty — callers skip injection in those cases.
    /// Bodies longer than `budget_chars` are truncated at a line boundary
    /// with a trailing `[...truncated]` marker so the model can tell the
    /// view is partial.
    pub fn project_summary_snippet(&self, budget_chars: usize) -> Option<String> {
        let path = self.root.join("summaries/project.md");
        let text = fs::read_to_string(&path).ok()?;
        let page = WikiPage::parse(&text)?;
        let body = page.body.trim_end();
        if body.is_empty() {
            return None;
        }
        if body.len() <= budget_chars {
            return Some(body.to_string());
        }
        let marker = "\n[...truncated]";
        let cutoff = budget_chars.saturating_sub(marker.len());
        let mut end = 0usize;
        for (idx, _) in body.char_indices() {
            if idx > cutoff {
                break;
            }
            end = idx;
        }
        let slice = &body[..end];
        let line_end = slice.rfind('\n').unwrap_or(end);
        let truncated = &body[..line_end];
        Some(format!("{}{}", truncated, marker))
    }

    /// Rank `IndexEntry` rows filtered to `Entity`/`Concept` categories by
    /// `last_updated` descending and emit a compact `- {title} — {one_liner}`
    /// listing for session-start system-prompt injection. Entries with no
    /// cached `last_updated` sort last (stable within the `None` bucket).
    ///
    /// Returns `None` when the index has no qualifying pages or is
    /// unreadable, so callers can omit the `\<wiki_fresh\>` block entirely
    /// without branching on errors. Fully in-memory on post-C85 indexes —
    /// every entry's `last_updated` is cached, so ranking is zero I/O.
    ///
    /// Output is truncated at a line boundary to fit within `budget_chars`
    /// bytes; no partial title lines ever reach the prompt.
    pub fn fresh_pages_snippet(&self, budget_chars: usize) -> Option<String> {
        let idx = self.load_index().ok()?;
        let mut ranked: Vec<&IndexEntry> = idx
            .entries
            .iter()
            .filter(|e| matches!(e.category, PageType::Entity | PageType::Concept))
            .collect();
        if ranked.is_empty() {
            return None;
        }
        // Primary: `last_updated` desc, with `None` sorting last. Secondary:
        // path asc for deterministic ordering across equal timestamps.
        ranked.sort_by(|a, b| match (&a.last_updated, &b.last_updated) {
            (Some(x), Some(y)) => y.cmp(x).then_with(|| a.path.cmp(&b.path)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.path.cmp(&b.path),
        });

        let header = "## Fresh wiki pages (recently updated)\n";
        let mut out = String::new();
        for e in ranked {
            let line = format!("- {} — {}\n", e.title, e.one_liner);
            // First emission reserves the header; subsequent lines only
            // need to fit themselves. Budget is a hard byte cap.
            let needed = if out.is_empty() {
                header.len() + line.len()
            } else {
                line.len()
            };
            if out.len() + needed > budget_chars {
                break;
            }
            if out.is_empty() {
                out.push_str(header);
            }
            out.push_str(&line);
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Touch `.dm/wiki/.summary-dirty` to signal that the project summary is
    /// stale. Called by `ingest_file_internal` after a successful page write.
    /// Cheap: a zero-byte file write. Bulk ingest collapses under the same
    /// marker — one consumer-boundary regeneration covers N ingests.
    pub fn mark_summary_dirty(&self) -> io::Result<()> {
        let path = self.root.join(SUMMARY_DIRTY_MARKER);
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&path)?;
        Ok(())
    }

    /// Whether `.dm/wiki/.summary-dirty` exists. Cheap stat check.
    pub fn is_summary_dirty(&self) -> bool {
        self.root.join(SUMMARY_DIRTY_MARKER).is_file()
    }

    /// Remove the dirty marker. `NotFound` is not an error — the caller
    /// already has what it wanted.
    pub fn clear_summary_dirty(&self) -> io::Result<()> {
        let path = self.root.join(SUMMARY_DIRTY_MARKER);
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Lazy regeneration at consumer boundaries. If the dirty marker is
    /// set, regenerate `summaries/project.md` via
    /// [`Wiki::write_project_summary`] (which clears the marker
    /// internally). If clean, a no-op returning `None`. Callers:
    ///   - end of [`Wiki::refresh`] (batch boundary)
    ///   - session-start disk-fallback in `build_system_prompt_inner`
    ///     (consumer boundary)
    ///   - [`crate::api::ApiState::current_wiki_summary`] (web hot path)
    ///
    /// Idempotent under concurrent callers: both regenerate with
    /// last-writer-wins on identical content. `INDEX_LOCK` inside
    /// `write_project_summary` serializes the index write. A second
    /// `mark_summary_dirty` that lands after the inner clear but before
    /// this function returns is preserved for the next cycle — exactly
    /// the correct semantics.
    pub fn ensure_summary_current(&self) -> io::Result<Option<ProjectSummaryReport>> {
        if !self.is_summary_dirty() {
            return Ok(None);
        }
        let report = self.write_project_summary()?;
        Ok(Some(report))
    }
}

/// Parent-module slug for [`Wiki::momentum`] aggregation. `src/foo/bar.rs`
/// → `"src/foo"`; `README.md` (no `/`) → `"."` (a sentinel for top-level
/// paths, chosen over `""` so it renders visibly in the momentum
/// summary). Trailing slashes are stripped; double slashes collapse under
/// `rfind('/')`'s behaviour.
pub fn momentum_module_of(path: &str) -> String {
    match path.rfind('/') {
        Some(0) => ".".to_string(),
        Some(i) => path[..i].to_string(),
        None => ".".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_age_hours_parses_recent_timestamp_returns_zero_ish() {
        let now = chrono::NaiveDate::from_ymd_opt(2026, 4, 18)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        // 30 minutes earlier → 0 full hours
        let ts = "2026-04-18 11:30:00";
        assert_eq!(cycle_age_hours(ts, now), Some(0));
    }

    #[test]
    fn cycle_age_hours_parses_old_timestamp_returns_large() {
        let now = chrono::NaiveDate::from_ymd_opt(2026, 4, 18)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        // 48h earlier
        let ts = "2026-04-16 12:00:00";
        assert_eq!(cycle_age_hours(ts, now), Some(48));
    }

    #[test]
    fn cycle_age_hours_returns_none_on_clock_skew() {
        let now = chrono::NaiveDate::from_ymd_opt(2026, 4, 18)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        // Future timestamp
        let ts = "2026-04-18 13:00:00";
        assert_eq!(cycle_age_hours(ts, now), None);
    }

    #[test]
    fn cycle_age_hours_returns_none_on_parse_failure() {
        let now = chrono::NaiveDate::from_ymd_opt(2026, 4, 18)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        assert_eq!(cycle_age_hours("not a timestamp", now), None);
    }

    #[test]
    fn momentum_module_of_strips_filename() {
        assert_eq!(momentum_module_of("src/foo/bar.rs"), "src/foo");
        assert_eq!(momentum_module_of("src/bar.rs"), "src");
    }

    #[test]
    fn momentum_module_of_top_level_uses_dot_sentinel() {
        assert_eq!(momentum_module_of("README.md"), ".");
        assert_eq!(momentum_module_of("Cargo.toml"), ".");
        assert_eq!(momentum_module_of(""), ".");
    }
}
