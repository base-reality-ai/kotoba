//! Wiki index + page CRUD + index-walking aggregates.
//!
//! On-disk read/write surface for `index.md` and individual wiki pages,
//! plus `Wiki::stats` (page counts, log tail, inbound-link tally) and
//! `Wiki::refresh` (re-ingest entity pages whose sources have drifted).
//! Split from `mod.rs` across Cycles 12 and 16 via split-impl.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::{
    parse_page_timestamp, validate_rel, IngestOutcome, PageType, SkipReason, Wiki, WikiIndex,
    WikiPage, WikiRefreshReport, WikiSeedReport, WikiStats,
};

impl Wiki {
    /// Read and parse `index.md`. A missing or unreadable index is treated
    /// as empty rather than an error, so callers can recover by re-indexing.
    pub fn load_index(&self) -> io::Result<WikiIndex> {
        let path = self.root.join("index.md");
        match fs::read_to_string(&path) {
            Ok(text) => Ok(WikiIndex::parse(&text)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(WikiIndex::default()),
            Err(e) => Err(e),
        }
    }

    /// Write `index.md` from the given index.
    pub fn save_index(&self, idx: &WikiIndex) -> io::Result<()> {
        let path = self.root.join("index.md");
        fs::write(path, idx.to_markdown())
    }

    /// Read and parse a page at `\<wiki_root\>/\<relative_path\>`.
    pub fn read_page(&self, relative_path: &str) -> io::Result<WikiPage> {
        validate_rel(relative_path)?;
        let path = self.root.join(relative_path);
        let text = fs::read_to_string(&path)?;
        WikiPage::parse(&text).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("malformed wiki page at {}", path.display()),
            )
        })
    }

    /// Serialize and write a page to `\<wiki_root\>/\<relative_path\>`. Creates
    /// parent directories if missing. Overwrites any existing file.
    pub fn write_page(&self, relative_path: &str, page: &WikiPage) -> io::Result<()> {
        validate_rel(relative_path)?;
        let path = self.root.join(relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, page.to_markdown())
    }

    /// Summary of wiki contents: page counts per category, log-file size,
    /// the most recent log entry, and the most-linked entity pages.
    /// Computed from `index.md`, `log.md`, and a one-shot scan over
    /// entity pages + their source files to tally inbound cross-references
    /// (see `Wiki::compute_inbound_links`). Intended for `/wiki status`.
    pub fn stats(&self) -> io::Result<WikiStats> {
        let idx = self.load_index()?;
        let mut by_category: std::collections::BTreeMap<PageType, usize> =
            std::collections::BTreeMap::new();
        for entry in &idx.entries {
            *by_category.entry(entry.category).or_insert(0) += 1;
        }

        let log_path = self.root.join("log.md");
        let (log_entries, last_activity) = match fs::read_to_string(&log_path) {
            Ok(text) => {
                let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
                let last = lines.last().map(|l| l.trim().to_string());
                (lines.len(), last)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => (0, None),
            Err(e) => return Err(e),
        };

        let counts = self.compute_inbound_links(&idx)?;
        let mut most_linked: Vec<(String, usize)> =
            counts.into_iter().filter(|(_, n)| *n > 0).collect();
        most_linked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        most_linked.truncate(5);

        Ok(WikiStats {
            root: self.root.clone(),
            total_pages: idx.entries.len(),
            by_category,
            log_entries,
            last_activity,
            most_linked,
        })
    }

    /// Walk the index and re-ingest entity pages whose `.rs` sources are
    /// newer than the page OR whose `entity_kind` frontmatter is `None`
    /// (a legacy-page signal post-Cycle-38). Re-ingest uses the forced
    /// path so content-hash dedup doesn't short-circuit schema backfills.
    ///
    /// Non-entity pages (Concept / Synthesis / Howto) are skipped. See
    /// [`WikiRefreshReport`] for output shape.
    pub fn refresh(&self) -> io::Result<WikiRefreshReport> {
        let mut report = WikiRefreshReport::default();
        let proj = self.project_root();
        let canonical_proj = proj.canonicalize().unwrap_or_else(|_| proj.clone());
        let idx = self.load_index()?;
        for entry in &idx.entries {
            if entry.category != PageType::Entity {
                continue;
            }
            let abs = self.root.join(&entry.path);
            let Ok(text) = fs::read_to_string(&abs) else {
                continue;
            };
            let Some(page) = WikiPage::parse(&text) else {
                continue;
            };
            let page_ts = parse_page_timestamp(&page.last_updated);
            for src in &page.sources {
                let src_abs = proj.join(src);
                let needs_kind_backfill = page.entity_kind.is_none() && src.ends_with(".rs");
                let needs_temporal_refresh = match (page_ts, fs::metadata(&src_abs)) {
                    (Some(ts), Ok(md)) => md.modified().map(|m| m > ts).unwrap_or(false),
                    _ => false,
                };
                if !needs_kind_backfill && !needs_temporal_refresh {
                    report.up_to_date += 1;
                    continue;
                }
                let Ok(content) = fs::read_to_string(&src_abs) else {
                    report.missing_sources.push(src.clone());
                    continue;
                };
                let Ok(canonical) = src_abs.canonicalize() else {
                    report.missing_sources.push(src.clone());
                    continue;
                };
                match self.ingest_file_internal(&canonical_proj, &canonical, &content, true) {
                    Ok(_) => report.refreshed.push(src.clone()),
                    Err(e) => report.errors.push((src.clone(), e.to_string())),
                }
            }
        }
        // Concept auto-detection runs as part of the batch boundary: new
        // or updated entity pages may have shifted dependency-frequency
        // buckets across the `CONCEPT_DEP_MIN_OCCURRENCES` threshold.
        // Must run *before* `ensure_summary_current` so the summary sees
        // fresh concept pages on first run. Non-fatal — refresh already
        // succeeded and the explicit `/wiki concepts` path remains.
        let _ = self.write_concept_pages();
        // Batch boundary: if anything was refreshed (or a prior ingest left
        // the marker set), regenerate now so session-start injection sees
        // fresh state. Non-fatal — refresh already succeeded.
        let _ = self.ensure_summary_current();
        Ok(report)
    }

    /// Operator-deliberate bulk seed of `.rs` (or other ext-filtered)
    /// source files into entity pages. Walks `rel_dir` recursively from
    /// the project root and feeds every matching file through the same
    /// `ingest_file` path that the tool pipeline uses, so dedup, drift
    /// markers, log entries, and entity-kind detection behave identically
    /// to single-file ingests.
    ///
    /// Skips dotfile dirs (`.git`, `.dm`, etc.), `target/`, and
    /// `node_modules/` — high-noise and rarely the operator's intent.
    /// Source files are not modified; only `.dm/wiki/` mutates.
    /// Distinct from [`Wiki::refresh`], which only re-ingests sources
    /// already referenced by an existing entity page.
    ///
    /// Returns [`io::Error`] only when `rel_dir` doesn't exist on disk.
    /// Per-file failures (read errors, ingest errors) accumulate in
    /// [`WikiSeedReport::errors`] and don't abort the walk.
    ///
    /// `ext_filter` is matched lower-case against the OS extension. An
    /// empty filter matches nothing — strict, predictable contract.
    pub fn seed_dir(&self, rel_dir: &Path, ext_filter: &[&str]) -> io::Result<WikiSeedReport> {
        let proj = self.project_root();
        let canonical_proj = proj.canonicalize().unwrap_or_else(|_| proj.clone());
        let abs = if rel_dir.is_absolute() {
            rel_dir.to_path_buf()
        } else {
            proj.join(rel_dir)
        };
        if !abs.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "seed dir does not exist: {}. Try: re-run from the project root or pass a relative path that exists.",
                    abs.display()
                ),
            ));
        }

        let mut collected: Vec<PathBuf> = Vec::new();
        let mut symlinks_skipped: usize = 0;
        walk_dir_collect(&abs, ext_filter, &mut collected, &mut symlinks_skipped)?;

        let mut report = WikiSeedReport {
            symlinks_skipped,
            ..WikiSeedReport::default()
        };
        for src_abs in &collected {
            let rel_display = src_abs
                .strip_prefix(&canonical_proj)
                .or_else(|_| src_abs.strip_prefix(&proj))
                .unwrap_or(src_abs)
                .display()
                .to_string();
            let content = match fs::read_to_string(src_abs) {
                Ok(c) => c,
                Err(e) => {
                    report.errors.push((rel_display, e.to_string()));
                    continue;
                }
            };
            let canonical = match src_abs.canonicalize() {
                Ok(p) => p,
                Err(e) => {
                    report.errors.push((rel_display, e.to_string()));
                    continue;
                }
            };
            match self.ingest_file(&canonical_proj, &canonical, &content) {
                Ok(IngestOutcome::Ingested { page_rel }) => report.ingested.push(page_rel),
                Ok(IngestOutcome::Skipped(SkipReason::UnchangedSinceLast)) => {
                    report.skipped_unchanged.push(rel_display)
                }
                Ok(IngestOutcome::Skipped(_)) => report.skipped_other += 1,
                Err(e) => report.errors.push((rel_display, e.to_string())),
            }
        }

        // Mirror `refresh()`'s batch-boundary regen so a fresh seed
        // immediately surfaces concept pages and a refreshed
        // `summaries/project.md`. Non-fatal — seed already succeeded.
        let _ = self.write_concept_pages();
        let _ = self.ensure_summary_current();

        Ok(report)
    }
}

/// Recursive directory walker for [`Wiki::seed_dir`]. Skips dirs whose
/// name starts with `.` (handles `.git`, `.dm`, `.cargo`), exact-name
/// matches `target` and `node_modules`, symlinks (counted via
/// `symlinks_skipped`), and any non-file entry. Files are kept only if
/// their lower-cased extension appears in `ext_filter`; an empty filter
/// keeps nothing.
///
/// The explicit `is_symlink()` branch is belt-and-suspenders: a symlink's
/// `FileType` returns false for both `is_dir()` and `is_file()`, so they'd
/// also fall through the existing arms — but the counter would never
/// increment without the dedicated check. Must come *before* the
/// dir/file arms; otherwise it would be unreachable.
fn walk_dir_collect(
    dir: &Path,
    ext_filter: &[&str],
    out: &mut Vec<PathBuf>,
    symlinks_skipped: &mut usize,
) -> io::Result<()> {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for ent in rd.flatten() {
        let Ok(ft) = ent.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            *symlinks_skipped += 1;
            continue;
        }
        let name_os = ent.file_name();
        let name = name_os.to_string_lossy();
        if ft.is_dir() {
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            walk_dir_collect(&ent.path(), ext_filter, out, symlinks_skipped)?;
        } else if ft.is_file() {
            let path = ent.path();
            let ext_match = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .map(|lc| ext_filter.iter().any(|f| f.eq_ignore_ascii_case(&lc)))
                .unwrap_or(false);
            if ext_match {
                out.push(path);
            }
        }
    }
    Ok(())
}
