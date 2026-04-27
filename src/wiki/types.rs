//! Wiki data types — pure value objects shared across wiki submodules.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::RecentCycle;
use super::{escape_path, sanitize_single_line, scan_path};

/// Result of an ingest attempt. Callers use the variant to surface
/// what happened (or why nothing did) in logs and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestOutcome {
    Ingested { page_rel: String },
    Skipped(SkipReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// `DM_WIKI_AUTO_INGEST=0|false|off` at the environment level.
    Disabled,
    /// File is under `.dm/wiki/` — ingest would recurse.
    InsideWikiDir,
    /// Canonical path escapes the project root.
    OutsideProject,
    /// Content hash matches the last ingest for this path.
    UnchangedSinceLast,
    /// Empty or otherwise non-ingestible path.
    IneligiblePath,
}

/// Category of a wiki page.
///
/// `Ord` follows declaration order (Entity < Concept < Summary < Synthesis),
/// matching the order used everywhere else in this module when iterating
/// categories. Used as a key in `WikiStats::by_category`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PageType {
    Entity,
    Concept,
    Summary,
    Synthesis,
}

impl PageType {
    pub fn as_str(self) -> &'static str {
        match self {
            PageType::Entity => "entity",
            PageType::Concept => "concept",
            PageType::Summary => "summary",
            PageType::Synthesis => "synthesis",
        }
    }

    /// Heading used in `index.md` for this category.
    pub fn index_heading(self) -> &'static str {
        match self {
            PageType::Entity => "## Entities",
            PageType::Concept => "## Concepts",
            PageType::Summary => "## Summaries",
            PageType::Synthesis => "## Synthesis",
        }
    }

    /// Subdirectory name where pages of this category live (plural form,
    /// matching `Wiki::ensure_layout`).
    pub fn category_dir(self) -> &'static str {
        match self {
            PageType::Entity => "entities",
            PageType::Concept => "concepts",
            PageType::Summary => "summaries",
            PageType::Synthesis => "synthesis",
        }
    }
}

impl std::str::FromStr for PageType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s.trim() {
            "entity" => Ok(PageType::Entity),
            "concept" => Ok(PageType::Concept),
            "summary" => Ok(PageType::Summary),
            "synthesis" => Ok(PageType::Synthesis),
            _ => Err(()),
        }
    }
}

/// Knowledge layer a wiki page belongs to.
///
/// Kernel pages document dm's inherited spine; host pages document the
/// spawned project's domain. Pages predating this field default to Kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    Kernel,
    Host,
}

impl Layer {
    pub fn as_str(self) -> &'static str {
        match self {
            Layer::Kernel => "kernel",
            Layer::Host => "host",
        }
    }
}

impl std::str::FromStr for Layer {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s.trim() {
            "kernel" => Ok(Layer::Kernel),
            "host" => Ok(Layer::Host),
            _ => Err(()),
        }
    }
}

/// Structural kind of an entity page's source symbol. Optional because pages
/// ingested before concept auto-detection shipped won't have one, and pages
/// that describe multiple symbols don't fit a single kind. Lives in the
/// frontmatter so it survives round-trips and is visible to wiki tooling
/// (e.g. `/wiki lint` can require that every Entity page has a concrete kind).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntityKind {
    Function,
    Struct,
    Enum,
    Trait,
    Unknown,
}

impl EntityKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EntityKind::Function => "function",
            EntityKind::Struct => "struct",
            EntityKind::Enum => "enum",
            EntityKind::Trait => "trait",
            EntityKind::Unknown => "unknown",
        }
    }
}

impl std::str::FromStr for EntityKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s.trim() {
            "function" => Ok(EntityKind::Function),
            "struct" => Ok(EntityKind::Struct),
            "enum" => Ok(EntityKind::Enum),
            "trait" => Ok(EntityKind::Trait),
            "unknown" => Ok(EntityKind::Unknown),
            _ => Err(()),
        }
    }
}

/// One public-API symbol exported by the source file a wiki page documents.
/// `kind` is restricted to the variants `EntityKind` enumerates today
/// (function/struct/enum/trait); const/static/type/mod/impl exports are
/// deliberately dropped until the schema grows to represent them.
/// `name` is the bare identifier as it appears in source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyExport {
    pub kind: EntityKind,
    pub name: String,
}

/// A single wiki page: frontmatter + markdown body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WikiPage {
    pub title: String,
    pub page_type: PageType,
    /// Knowledge layer this page belongs to. Omitted frontmatter defaults to
    /// Kernel so existing canonical pages remain byte-identical.
    pub layer: Layer,
    pub sources: Vec<String>,
    pub last_updated: String,
    /// Structural kind of the source symbol when `page_type == Entity` and
    /// the ingester detected one. `None` for Concept/Summary/Synthesis pages,
    /// for legacy pages predating this field, or when detection was ambiguous.
    pub entity_kind: Option<EntityKind>,
    /// 1-3 line summary drawn from source-file module docs (`//!`), or
    /// `None` for non-Rust files, doc-less sources, and legacy pages
    /// predating this field. Guaranteed to contain no newlines when
    /// produced by `extract_purpose` — safe as a single-line YAML value.
    /// Manual constructions inherit this invariant.
    pub purpose: Option<String>,
    /// Public-API symbols the source file exports. Empty `Vec` for non-Rust
    /// files, files with no public items, or legacy pages predating this
    /// field. An empty `Vec` is serialized as no frontmatter block (byte-
    /// identity with pages written before `key_exports` shipped).
    pub key_exports: Vec<KeyExport>,
    /// `use` imports the source file declares, as written (whitespace-
    /// collapsed for multi-line groups). Source order preserved. Empty `Vec`
    /// for non-Rust files, files with no `use` statements, or legacy pages
    /// predating this field. An empty `Vec` is serialized as no frontmatter
    /// block (byte-identity with pages written before `dependencies` shipped).
    pub dependencies: Vec<String>,
    /// Cycle outcome stamp for synthesis pages — opaque at the wiki layer.
    /// Convention (not enforced): `"green"` / `"red"` / `"mixed"`. `None` for
    /// non-synthesis pages, for legacy synthesis pages predating this field,
    /// and for synthesis writes where the caller had no outcome signal.
    /// Serialized as `outcome: \<value\>` in frontmatter when `Some`; the
    /// line is omitted entirely when `None` (byte-identity with pre-C90
    /// pages). Producer side: `Wiki::write_cycle_synthesis`. Consumer side:
    /// [`RecentCycle::outcome`] in the `PlannerBrief` render.
    pub outcome: Option<String>,
    /// Optional opt-in declaration: directories the concept page claims
    /// authoritative documentation over. When non-empty, the
    /// `ConceptScopeUndocumented` lint rule (step 2 of C38's reverse-drift
    /// arc) walks each prefix and reports `.rs` files not mentioned in
    /// the page body. Empty `Vec` means the page makes no scope claim
    /// and is unaffected by the rule. Pages predating this field
    /// round-trip with empty `scope`. Path-prefix match only (no globs)
    /// per C38 design.
    pub scope: Vec<String>,
    pub body: String,
}

impl WikiPage {
    /// Serialize to markdown with a YAML frontmatter block.
    ///
    /// The `entity_kind` line is only emitted when `Some` — legacy pages
    /// round-trip to their original on-disk form, and non-Entity page types
    /// stay noise-free.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("---\n");
        writeln!(out, "title: {}", self.title).expect("write to String never fails");
        writeln!(out, "type: {}", self.page_type.as_str()).expect("write to String never fails");
        if self.layer != Layer::Kernel {
            writeln!(out, "layer: {}", self.layer.as_str()).expect("write to String never fails");
        }
        if let Some(kind) = self.entity_kind {
            writeln!(out, "entity_kind: {}", kind.as_str()).expect("write to String never fails");
        }
        if let Some(purpose) = &self.purpose {
            writeln!(out, "purpose: {}", purpose).expect("write to String never fails");
        }
        if !self.key_exports.is_empty() {
            out.push_str("key_exports:\n");
            for export in &self.key_exports {
                writeln!(out, "  - {} {}", export.kind.as_str(), export.name)
                    .expect("write to String never fails");
            }
        }
        if !self.dependencies.is_empty() {
            out.push_str("dependencies:\n");
            for dep in &self.dependencies {
                writeln!(out, "  - {}", dep).expect("write to String never fails");
            }
        }
        if !self.scope.is_empty() {
            out.push_str("scope:\n");
            for s in &self.scope {
                writeln!(out, "  - {}", s).expect("write to String never fails");
            }
        }
        if let Some(outcome) = &self.outcome {
            writeln!(out, "outcome: {}", outcome).expect("write to String never fails");
        }
        out.push_str("sources:\n");
        for src in &self.sources {
            writeln!(out, "  - {}", src).expect("write to String never fails");
        }
        writeln!(out, "last_updated: {}", self.last_updated).expect("write to String never fails");
        out.push_str("---\n");
        out.push_str(&self.body);
        out
    }

    /// Parse a page from text. Returns `None` if the frontmatter is missing or malformed.
    ///
    /// Hand-rolled parser — the frontmatter fields are limited and known, so
    /// there's no need to pull in a full YAML dependency for the scaffold.
    pub fn parse(text: &str) -> Option<WikiPage> {
        let rest = text.strip_prefix("---\n")?;
        let end = rest.find("\n---\n")?;
        let header = &rest[..end];
        let body = &rest[end + "\n---\n".len()..];

        let mut title: Option<String> = None;
        let mut page_type: Option<PageType> = None;
        let mut layer = Layer::Kernel;
        let mut sources: Vec<String> = Vec::new();
        let mut last_updated: Option<String> = None;
        let mut entity_kind: Option<EntityKind> = None;
        let mut purpose: Option<String> = None;
        let mut key_exports: Vec<KeyExport> = Vec::new();
        let mut dependencies: Vec<String> = Vec::new();
        let mut outcome: Option<String> = None;
        let mut scope: Vec<String> = Vec::new();
        let mut in_sources = false;
        let mut in_key_exports = false;
        let mut in_dependencies = false;
        let mut in_scope = false;

        for line in header.lines() {
            if in_sources {
                if let Some(rest) = line.strip_prefix("  - ") {
                    sources.push(rest.to_string());
                    continue;
                }
                in_sources = false;
            }
            if in_key_exports {
                if let Some(rest) = line.strip_prefix("  - ") {
                    // Format: `  - <kind> <name>`. Malformed lines (missing
                    // space, unknown kind, empty name) are silently dropped
                    // — same forward-compat pattern as entity_kind parsing.
                    if let Some((kind_str, name)) = rest.split_once(' ') {
                        if let Ok(kind) = kind_str.parse::<EntityKind>() {
                            let name = name.trim();
                            if !name.is_empty() {
                                key_exports.push(KeyExport {
                                    kind,
                                    name: name.to_string(),
                                });
                            }
                        }
                    }
                    continue;
                }
                in_key_exports = false;
            }
            if in_dependencies {
                if let Some(rest) = line.strip_prefix("  - ") {
                    let trimmed = rest.trim();
                    if !trimmed.is_empty() {
                        dependencies.push(trimmed.to_string());
                    }
                    continue;
                }
                in_dependencies = false;
            }
            if in_scope {
                if let Some(rest) = line.strip_prefix("  - ") {
                    let trimmed = rest.trim();
                    if !trimmed.is_empty() {
                        scope.push(trimmed.to_string());
                    }
                    continue;
                }
                in_scope = false;
            }
            if let Some(v) = line.strip_prefix("title: ") {
                title = Some(v.to_string());
            } else if let Some(v) = line.strip_prefix("type: ") {
                page_type = v.parse::<PageType>().ok();
            } else if let Some(v) = line.strip_prefix("layer: ") {
                layer = v.parse::<Layer>().ok()?;
            } else if let Some(v) = line.strip_prefix("entity_kind: ") {
                // Malformed value silently drops to None rather than failing
                // the whole parse — preserves the "unknown field shouldn't
                // brick the wiki" property.
                entity_kind = v.parse::<EntityKind>().ok();
            } else if let Some(v) = line.strip_prefix("purpose: ") {
                purpose = Some(v.to_string());
            } else if let Some(v) = line.strip_prefix("outcome: ") {
                outcome = Some(v.to_string());
            } else if line == "sources:" {
                in_sources = true;
            } else if line == "key_exports:" {
                in_key_exports = true;
            } else if line == "dependencies:" {
                in_dependencies = true;
            } else if line == "scope:" {
                in_scope = true;
            } else if let Some(v) = line.strip_prefix("last_updated: ") {
                last_updated = Some(v.to_string());
            }
        }

        Some(WikiPage {
            title: title?,
            page_type: page_type?,
            layer,
            sources,
            last_updated: last_updated?,
            entity_kind,
            purpose,
            key_exports,
            dependencies,
            outcome,
            scope,
            body: body.to_string(),
        })
    }
}

/// One entry in `index.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    pub title: String,
    pub path: String,
    pub one_liner: String,
    pub category: PageType,
    /// Cached `last_updated` frontmatter of the linked page, emitted
    /// as a trailing `<!--updated:{ts}-->` HTML comment in `index.md`.
    /// `None` on legacy indexes (pre-C85) that lack the suffix;
    /// consumers (`recent_cycles`, `fresh_pages`) fall back to a
    /// per-entry file read in that case, which self-heals on the next
    /// ingest that touches the entry.
    pub last_updated: Option<String>,
    /// Cached `outcome` frontmatter of the linked page (C92), mirroring
    /// the `last_updated` cache pattern. Emitted as a trailing
    /// `<!--outcome:{v}-->` HTML comment after `<!--updated:...-->`
    /// when `Some`. `None` on legacy indexes (pre-C92) and on every
    /// non-Synthesis page (entity/concept/summary pages never carry
    /// an outcome). Lets the `planner_brief` composer skip per-entry
    /// page reads when both this and `last_updated` are cached.
    pub outcome: Option<String>,
}

/// Parsed view of `index.md`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WikiIndex {
    pub entries: Vec<IndexEntry>,
}

impl WikiIndex {
    /// Render the index back to markdown, grouped by category.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from(
            "# Wiki Index\n\n\
             This file is the catalog of all wiki pages. Each entry links to a page\n\
             with a one-line summary. Updated by dm on every ingest.\n\n",
        );
        for cat in [
            PageType::Entity,
            PageType::Concept,
            PageType::Summary,
            PageType::Synthesis,
        ] {
            out.push_str(cat.index_heading());
            out.push('\n');
            out.push('\n');
            for entry in self.entries.iter().filter(|e| e.category == cat) {
                write!(
                    out,
                    "- [{}]({}) — {}",
                    sanitize_single_line(&entry.title),
                    escape_path(&sanitize_single_line(&entry.path)),
                    sanitize_single_line(&entry.one_liner),
                )
                .expect("write to String never fails");
                if let Some(ts) = &entry.last_updated {
                    write!(out, " <!--updated:{}-->", sanitize_single_line(ts))
                        .expect("write to String never fails");
                }
                if let Some(v) = &entry.outcome {
                    write!(out, " <!--outcome:{}-->", sanitize_single_line(v))
                        .expect("write to String never fails");
                }
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }

    /// Parse `index.md`. Unknown or malformed lines are skipped — a broken
    /// index should not be a hard error, since the wiki can recover by
    /// re-indexing.
    pub fn parse(text: &str) -> WikiIndex {
        let mut entries = Vec::new();
        let mut current: Option<PageType> = None;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("## ") {
                current = match rest.trim() {
                    "Entities" => Some(PageType::Entity),
                    "Concepts" => Some(PageType::Concept),
                    "Summaries" => Some(PageType::Summary),
                    "Synthesis" => Some(PageType::Synthesis),
                    _ => None,
                };
                continue;
            }
            let Some(cat) = current else { continue };
            let Some(rest) = line.strip_prefix("- [") else {
                continue;
            };
            let Some(title_end) = rest.find("](") else {
                continue;
            };
            let title = &rest[..title_end];
            let after = &rest[title_end + 2..];
            let Some((path, tail)) = scan_path(after) else {
                continue;
            };
            let tail = tail.trim_start_matches(" — ");
            // Legacy entries have no suffix → everything is the one_liner,
            // last_updated + outcome stay None so consumers fall back to a
            // per-entry file read (self-heals on next ingest). Known
            // suffixes (`<!--updated:ts-->`, `<!--outcome:v-->`) are
            // peeled in either order; peel the RIGHTMOST comment each
            // pass so a `<!--updated:--> <!--outcome:-->` pair parses
            // regardless of emission order.
            let mut remaining = tail.to_string();
            let mut last_updated: Option<String> = None;
            let mut outcome: Option<String> = None;
            loop {
                let trimmed = remaining.trim_end();
                let Some(stripped) = trimmed.strip_suffix("-->") else {
                    break;
                };
                let updated_pos = stripped.rfind(" <!--updated:");
                let outcome_pos = stripped.rfind(" <!--outcome:");
                // Pick whichever comment is RIGHTMOST — that's the tail
                // suffix the current iteration should consume.
                match (updated_pos, outcome_pos) {
                    (Some(up), Some(op)) if op > up => {
                        let v = stripped[op + " <!--outcome:".len()..].trim().to_string();
                        if outcome.is_none() && !v.is_empty() {
                            outcome = Some(v);
                        }
                        remaining = stripped[..op].to_string();
                    }
                    (Some(up), _) => {
                        let v = stripped[up + " <!--updated:".len()..].trim().to_string();
                        if last_updated.is_none() && !v.is_empty() {
                            last_updated = Some(v);
                        }
                        remaining = stripped[..up].to_string();
                    }
                    (None, Some(op)) => {
                        let v = stripped[op + " <!--outcome:".len()..].trim().to_string();
                        if outcome.is_none() && !v.is_empty() {
                            outcome = Some(v);
                        }
                        remaining = stripped[..op].to_string();
                    }
                    (None, None) => break,
                }
            }
            let one_liner = remaining.trim().to_string();
            entries.push(IndexEntry {
                title: title.to_string(),
                path,
                one_liner,
                category: cat,
                last_updated,
                outcome,
            });
        }
        WikiIndex { entries }
    }
}

/// Append-only log (`log.md`). Every wiki operation records a line.
#[derive(Debug, Clone)]
pub struct WikiLog {
    path: PathBuf,
}

impl WikiLog {
    pub fn new(path: PathBuf) -> Self {
        WikiLog { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append `\[timestamp\] verb | subject`. Creates the file if missing.
    pub fn append(&self, verb: &str, subject: &str) -> io::Result<()> {
        use std::io::Write;
        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let line = format!("[{}] {} | {}\n", ts, verb, subject);
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }
}

/// Read-only snapshot of a wiki's contents for introspection (e.g. `/wiki status`).
///
/// Counts come from `index.md`; log metrics from `log.md`. Computed by
/// `Wiki::stats`.
#[derive(Debug, Clone)]
pub struct WikiStats {
    pub root: PathBuf,
    pub total_pages: usize,
    pub by_category: std::collections::BTreeMap<PageType, usize>,
    pub log_entries: usize,
    pub last_activity: Option<String>,
    /// Top-5 entity pages ranked by inbound cross-reference count, in
    /// descending order (tie-break: alphabetical page path). Entries with
    /// zero inbound links are excluded, so a wiki without cross-refs
    /// renders with an empty list (no "(none)" placeholder).
    pub most_linked: Vec<(String, usize)>,
}

/// One hit returned by `Wiki::search`. `snippet` is a short excerpt around
/// the first body match (or the title if only the title matched).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WikiSearchHit {
    pub path: String,
    pub title: String,
    pub category: PageType,
    pub layer: Layer,
    pub match_count: usize,
    pub snippet: String,
}

/// Kind of static-consistency issue found by `Wiki::lint`. Variant order
/// is load-bearing: `lint()` sorts findings by `(kind as u8, path)` so
/// tests can assert exact sequences and grouping. Declared order (and
/// therefore sort order) is: `OrphanIndexEntry` → `UntrackedPage` →
/// `CategoryMismatch` → `IndexTimestampDrift` → `SourceMissing` →
/// `BodyPathMissing` → `ConceptScopeUndocumented` →
/// `SourceNewerThanPage` → `ItemDrift` → `ExportDrift` → `EntityGap` →
/// `MissingEntityKind` →
/// `DuplicateSource` → `MalformedPage`.
/// `MalformedPage` must stay last so parse-failure findings never hide
/// behind content-level issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WikiLintKind {
    /// Index entry points to a page file that does not exist on disk.
    OrphanIndexEntry,
    /// Page file exists on disk but has no entry in the index.
    UntrackedPage,
    /// Index entry's `category` disagrees with its `path`'s subdir.
    CategoryMismatch,
    /// Index entry's cached `last_updated` marker disagrees with the
    /// linked page's frontmatter. This catches stale `<!--updated:...-->`
    /// comments in `index.md`; legacy entries without a cached timestamp
    /// are ignored.
    IndexTimestampDrift,
    /// A path listed in a page's `sources` frontmatter no longer exists
    /// on disk under the project root.
    SourceMissing,
    /// A `src/.../X.rs` path appears in a page's body text (markdown
    /// tables, prose, bullet lists) but does not exist on disk under
    /// the project root. Sibling of `SourceMissing` — that rule scans
    /// the structured `sources:` frontmatter; this rule scans body
    /// text, where historical phantoms (e.g. C23's `src/api/state.rs`
    /// in `concepts/module-structure.md`) lived in tables. The two
    /// rules are deduplicated: a path in both `sources:` and body
    /// surfaces only as `SourceMissing`.
    BodyPathMissing,
    /// Concept page declares `scope: <prefixes>` in its frontmatter, but
    /// one or more `.rs` files under that prefix is not mentioned in the
    /// page body. The reverse-direction sibling of `BodyPathMissing`:
    /// where `BodyPathMissing` catches body mentions of files that don't
    /// exist, this rule catches files that exist under a declared scope
    /// but are silently undocumented. Opt-in via the page's `scope:`
    /// field; pages without `scope:` are unaffected.
    ///
    /// MVP semantics (C38 design): path-prefix match (no globs);
    /// recursion into subdirectories; `mod.rs` files skipped; Concept
    /// pages only.
    ConceptScopeUndocumented,
    /// A source file listed in a page's `sources` has been modified after
    /// the page was last updated — consider re-ingesting.
    SourceNewerThanPage,
    /// Entity page's `## Items` section names an item that is no longer
    /// a top-level item in the current source file (renamed or removed).
    /// Content-level drift; fires only for `.rs` sources with a parseable
    /// `## Items` section in the page body. Pre-Cycle-35 stub pages
    /// (no Items section) are silently skipped.
    ItemDrift,
    /// Entity page's `key_exports` frontmatter names a symbol no longer
    /// present in the current source file's extracted exports. Sibling
    /// of [`WikiLintKind::ItemDrift`]: `ItemDrift` detects drift in the
    /// body's `## Items` list; `ExportDrift` detects drift in the
    /// structured frontmatter field. Both can fire together for the
    /// same page. v1 compares by symbol name only — kind-level drift
    /// (same name, different `fn`/`struct`/`enum`/`trait`) is out of
    /// scope. Pre-Cycle-42 pages with empty `key_exports` are silently
    /// skipped.
    ExportDrift,
    /// A source file exists under `project_root/src/` with extension
    /// `.rs`, but no wiki page documents it — neither a canonical entity
    /// page at `entity_page_rel(rel)` nor any other page listing the
    /// file in its `sources` frontmatter. Surfaces documentation coverage
    /// holes so operators can trigger `ingest_file` on gap paths.
    EntityGap,
    /// Entity page whose `sources` include a `.rs` file but whose
    /// `entity_kind` frontmatter is `None`. Signals a legacy page that
    /// survived the Cycle-38 schema addition without being re-ingested;
    /// re-running ingest will populate the field.
    MissingEntityKind,
    /// Two or more Entity pages list the same `.rs` file as their first
    /// source (`sources.first()`). Page-to-page contradiction — exactly
    /// one entity page should document any given source. Most common
    /// causes: a rename left behind the old page, an ingest under a new
    /// title didn't delete the old page, or a split that didn't clean
    /// up. The operator reconciles by deleting the stale page and
    /// running `/wiki refresh` on the remaining one.
    DuplicateSource,
    /// Page frontmatter could not be parsed, or `last_updated` is not a
    /// valid `%Y-%m-%d %H:%M:%S` timestamp. Detail describes the specific
    /// failure. Surfaced as a finding (rather than silently skipped) so
    /// the user can see — and repair — the blocker that would otherwise
    /// hide source-drift issues behind a pass-failure.
    MalformedPage,
}

/// One static-consistency finding from `Wiki::lint`. See [`WikiLintKind`]
/// for the rules checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WikiLintFinding {
    pub kind: WikiLintKind,
    /// Page path the finding refers to (e.g. "entities/foo.md"), or empty
    /// for findings not tied to a single page.
    pub path: String,
    /// Human-readable detail for the formatter to render per-line.
    pub detail: String,
}

/// Report from `Wiki::write_project_summary`. Exposes the derived
/// aggregates so callers can render a one-line confirmation without
/// re-parsing the generated page.
#[derive(Debug, Clone)]
pub struct ProjectSummaryReport {
    /// Relative path of the summary page — always `"summaries/project.md"`.
    pub path: String,
    /// Number of Entity pages that fed the aggregation.
    pub entity_count: usize,
    /// Count of entity pages per `EntityKind`. `BTreeMap` so iteration
    /// order is stable (`EntityKind` derives `Ord`).
    pub kind_counts: std::collections::BTreeMap<EntityKind, usize>,
    /// Top dependencies by citation frequency, descending. Capped at 10.
    #[allow(dead_code)]
    pub top_dependencies: Vec<(String, usize)>,
    /// Last-N cycle outcomes surfaced in the page body. Populated from
    /// `Wiki::collect_recent_cycles` with the
    /// `PROJECT_SUMMARY_RECENT_CYCLES` cap. Empty vec when no
    /// synthesis pages exist.
    pub recent_cycles: Vec<RecentCycle>,
}

/// Report from `Wiki::write_concept_pages`. `generated` holds paths of
/// concept pages newly created this run; `refreshed` holds paths of
/// existing concept pages whose body changed. `detected_deps` is the
/// full (dep, `occurrence_count`) set sorted by frequency desc, then
/// dep-string asc — a stable tiebreak for snapshot-testable output.
#[derive(Debug, Clone, Default)]
pub struct ConceptPagesReport {
    pub generated: Vec<String>,
    pub refreshed: Vec<String>,
    pub detected_deps: Vec<(String, usize)>,
}

/// Report from `Wiki::momentum`. Aggregates the most recent `ingest`
/// entries from `log.md` into hot-path and hot-module counts. `Default`
/// is load-bearing: `Wiki::build_project_summary` falls back to an
/// empty report if the log is missing or unreadable, and callers render
/// the empty shape as a "no activity yet" placeholder rather than an
/// error.
#[derive(Debug, Clone, Default)]
pub struct MomentumReport {
    /// Total lines seen in `log.md` (before any filtering). Helps the
    /// operator distinguish "no activity" from "plenty of activity but
    /// none in the window".
    pub total_entries: usize,
    /// Number of `ingest`-verb entries actually aggregated — at most
    /// `window`, at most `total_entries`.
    pub window_processed: usize,
    /// Top source paths by ingest frequency in the window, descending by
    /// count then ascending by path. Capped at `MOMENTUM_TOP_N`.
    pub hot_paths: Vec<(String, usize)>,
    /// Top parent modules (path → `momentum_module_of`) by ingest
    /// frequency in the window. Same ordering and cap as `hot_paths`.
    pub hot_modules: Vec<(String, usize)>,
}

/// One chain node's contribution to a cycle, fed to
/// `Wiki::write_cycle_synthesis`. The runner builds a `Vec` of these
/// in chain-DAG order at the end of each clean cycle so the wiki
/// captures what each role produced before the outputs roll out of
/// [`crate::orchestrate::types::ChainState::node_outputs`].
#[derive(Debug, Clone)]
pub struct CycleNodeSnapshot {
    pub name: String,
    pub role: String,
    /// Full output from the node; truncation to
    /// `CYCLE_SYNTHESIS_OUTPUT_PER_NODE` happens at render time so
    /// callers can hold the raw text without a pre-truncation round trip.
    pub output: String,
}

/// Report from `Wiki::refresh`. Each field accumulates across the walk
/// so callers can render a single summary.
#[derive(Debug, Default, Clone)]
pub struct WikiRefreshReport {
    /// Source rel-paths successfully re-ingested (one entry per source,
    /// multi-source pages push multiple entries).
    pub refreshed: Vec<String>,
    /// Count of (page, source) pairs that needed no refresh.
    pub up_to_date: usize,
    /// Sources referenced by a page but absent on disk.
    pub missing_sources: Vec<String>,
    /// `(source_rel, error_msg)` for ingest failures that aren't
    /// `SkipReason` outcomes.
    pub errors: Vec<(String, String)>,
}

/// Report from `Wiki::seed_dir`. Bulk operator-triggered ingest of
/// source files into entity pages. Distinguishes `Ingested` writes from
/// hash-dedup hits (so a re-run after editing one file shows a small
/// `ingested` list and a large `skipped_unchanged` tail) and lumps the
/// other `SkipReason` variants under a single counter — operators
/// generally only need the totals.
#[derive(Debug, Default, Clone)]
pub struct WikiSeedReport {
    /// Wiki rel paths newly written or rewritten (one per source file).
    pub ingested: Vec<String>,
    /// Source rel paths whose content hash matched the last ingest, so
    /// no page write happened.
    pub skipped_unchanged: Vec<String>,
    /// Count of `SkipReason::{Disabled, InsideWikiDir, OutsideProject,
    /// IneligiblePath}` outcomes — predictable skips that don't deserve
    /// per-path output.
    pub skipped_other: usize,
    /// Symbolic links encountered during the walk. Counted separately
    /// because they're a security/correctness signal — symlinked dirs
    /// could otherwise loop or escape the project root, and symlinked
    /// files would cause double-ingest. The walker never follows them.
    pub symlinks_skipped: usize,
    /// `(source_rel, error_msg)` pairs for I/O or ingest failures.
    pub errors: Vec<(String, String)>,
}
