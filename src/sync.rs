//! `dm sync` — explicit host-mode kernel update flow.
//!
//! Tier 2 starts with the command boundary and mode guard. The actual diff,
//! status, and apply machinery lands in later increments.

use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncArgs {
    pub dry_run: bool,
    pub abort: bool,
    pub status: bool,
}

/// Where a file sits relative to the kernel/host boundary that `dm sync`
/// enforces. Per the contract in `.dm/wiki/concepts/dm-sync.md`, sync
/// touches kernel-scope files and never touches host-owned files. Wiki
/// pages under `concepts/` and `entities/` are layer-dependent — the
/// classifier returns `WikiPage` and leaves the per-page `layer:` lookup
/// to the caller, who can read the frontmatter at sync time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncScope {
    /// Path is inside kernel scope: sync may write here on apply.
    Kernel,
    /// Path is host-owned: sync never touches it.
    Host,
    /// Path is a wiki page whose scope is determined by the page's
    /// `layer:` frontmatter. `Layer::Kernel` (or absent) → kernel scope;
    /// `Layer::Host` → host scope. The classifier defers this read.
    WikiPage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChangeKind {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl ChangeKind {
    fn note(self) -> Option<&'static str> {
        match self {
            ChangeKind::Added => None,
            ChangeKind::Modified => None,
            ChangeKind::Deleted => Some("deleted"),
            ChangeKind::Renamed => Some("renamed — review needed"),
        }
    }
}

/// Per-file decision sync makes about a path the canonical-side diff
/// surfaced. Captures both the resolved scope (after wiki-layer peek)
/// and, for kernel-scope paths, the host-side state relative to the
/// pinned canonical bytes.
///
/// `SyncPlanEntry` is the unit dry-run renders and apply (Tier 2 cycle 9)
/// will iterate to decide which kernel-side bytes are safe to stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SyncPlanEntry {
    /// Project-relative path as it appeared in the canonical diff.
    pub(crate) rel: String,
    pub(crate) change_kind: ChangeKind,
    /// Resolved scope after `classify_for_sync` and (for wiki pages)
    /// `resolve_wiki_layer`. Always `Kernel` or `Host` here — `WikiPage`
    /// never escapes the planner.
    pub(crate) scope: SyncScope,
    /// Short note describing how a wiki page's layer was decided. `None`
    /// for non-wiki paths.
    pub(crate) layer_note: Option<&'static str>,
    /// Host-side state for kernel-scope paths. `None` for host-scope
    /// paths since sync never plans to touch them.
    pub(crate) host_status: Option<HostDeltaStatus>,
}

/// How a kernel-scope file's host bytes relate to the canonical bytes
/// at the pinned revision. Drives the per-line hint in dry-run output
/// and the apply path's "safe to stage" decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostDeltaStatus {
    /// Host bytes match canonical@pinned exactly. Apply would update
    /// host bytes to canonical@HEAD with no operator review.
    CleanApply,
    /// Canonical adds a new file at HEAD that did not exist at pinned
    /// and does not exist in the host tree. Apply would create it.
    NewKernelFile,
    /// Host has modified the file relative to canonical@pinned.
    /// Apply must defer to operator review.
    HostModified,
    /// Host bytes could not be read (I/O error other than NotFound, or
    /// a binary read failure). Treat as "needs review" defensively.
    HostUnreadable,
}

impl HostDeltaStatus {
    fn display_label(self) -> &'static str {
        match self {
            HostDeltaStatus::CleanApply => "kernel-only change, would apply cleanly",
            HostDeltaStatus::NewKernelFile => "new kernel file, would apply cleanly",
            HostDeltaStatus::HostModified => "host-modified — review needed",
            HostDeltaStatus::HostUnreadable => "host state unreadable — review needed",
        }
    }
}

/// All per-file decisions for a single sync run. Built once by
/// `build_sync_plan`, rendered by dry-run and (Tier 2 cycle 9) consumed
/// by apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SyncPlan {
    pub(crate) pinned: String,
    pub(crate) entries: Vec<SyncPlanEntry>,
}

impl SyncPlan {
    pub(crate) fn kernel_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.scope == SyncScope::Kernel)
            .count()
    }

    pub(crate) fn host_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.scope == SyncScope::Host)
            .count()
    }
}

/// Classify a project-relative path against the sync contract.
///
/// `rel` MUST be relative to the host project root (e.g. `src/foo.rs`,
/// `.dm/identity.toml`, `Cargo.toml`). Absolute paths are treated as
/// `Host` defensively — sync never operates on paths outside the
/// project root.
///
/// The design-page contract in `.dm/wiki/concepts/dm-sync.md` is the
/// authoritative source for the cases this function covers; tests pin
/// every bullet from the "kernel scope" / "NOT touched" lists.
pub fn classify_for_sync(rel: &Path) -> SyncScope {
    if rel.is_absolute() {
        return SyncScope::Host;
    }
    let s = match rel.to_str() {
        Some(s) => s,
        // Non-UTF-8 paths default to host-owned: sync would not know how
        // to round-trip them through the canonical wiki anyway, and the
        // safe answer is "leave alone".
        None => return SyncScope::Host,
    };
    // Normalize separators for Windows-style paths so the prefix checks
    // work uniformly. dm itself is unix-only today, but the classifier
    // is platform-agnostic since it only inspects path strings.
    let s = s.replace('\\', "/");

    // Explicit host-owned files. These take precedence over any prefix
    // rule below: `.dm/identity.toml` lives under `.dm/` but is host-
    // owned, and `src/host_main.rs` lives under `src/` but is host-owned.
    if matches!(
        s.as_str(),
        ".dm/identity.toml"
            | ".dm/wiki/index.md"
            | ".dm/wiki/log.md"
            | ".dm/wiki/.summary-dirty"
            | "src/host_main.rs"
            | "src/domain.rs"
    ) {
        return SyncScope::Host;
    }

    // Host-shaped tests: the spawn convention reserves `tests/host_*.rs`
    // for host-added tests so they survive sync. (Path-level only — host
    // tests in other shapes need explicit operator opt-out for now.)
    if let Some(rest) = s.strip_prefix("tests/") {
        if rest.starts_with("host_") {
            return SyncScope::Host;
        }
        return SyncScope::Kernel;
    }

    // Wiki-page paths under concepts/ or entities/: layer-dependent.
    // summaries/ and synthesis/ are host-regenerated artifacts so they
    // are host-owned regardless of any frontmatter.
    if s.starts_with(".dm/wiki/concepts/") || s.starts_with(".dm/wiki/entities/") {
        return SyncScope::WikiPage;
    }
    if s.starts_with(".dm/wiki/summaries/") || s.starts_with(".dm/wiki/synthesis/") {
        return SyncScope::Host;
    }

    // Anything else under `.dm/` is host-owned — the spawn paradigm
    // keeps host runtime state inside `.dm/` (sessions, caches, etc.).
    if s.starts_with(".dm/") {
        return SyncScope::Host;
    }

    // Kernel-managed code, build, tests, and top-level manifests.
    if s.starts_with("src/") || s.starts_with("build/") || s == "Cargo.toml" || s == "Cargo.lock" {
        return SyncScope::Kernel;
    }

    // Anything outside the kernel scope is host-owned by default — the
    // host project's own files (e.g. host-added crates, host-only
    // directories) live above the kernel surface.
    SyncScope::Host
}

pub async fn run_sync(args: SyncArgs) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().map_err(|e| {
        anyhow::anyhow!(
            "Failed to read current directory: {}. Try: run `dm sync` from an accessible host project directory.",
            e
        )
    })?;
    let message = sync_message_for_project(&cwd, args)?;
    println!("{message}");
    Ok(())
}

pub fn sync_message_for_project(project_root: &Path, args: SyncArgs) -> anyhow::Result<String> {
    let identity = crate::identity::load_at(project_root)?;
    if identity.is_kernel() {
        anyhow::bail!(
            "`dm sync` is only meaningful in host mode. Try: run from a project spawned via `dm spawn`, or use git in canonical dm directly."
        );
    }

    if args.status {
        return sync_status_message(&identity);
    }
    if args.dry_run {
        return sync_dry_run_message(project_root, &identity);
    }
    if args.abort {
        return sync_abort_message(project_root, &identity);
    }
    sync_apply_stage_message(project_root, &identity)
}

/// Predictable persistent staging directory for an in-flight `dm sync`
/// apply. Lives under `.dm/` so it sits next to the host's identity and
/// wiki state and survives across `dm sync` invocations until either a
/// successful apply commit (Tier 2 cycle 11+) clears it or `dm sync
/// --abort` discards it. Per-host-project, never global.
fn sync_staging_dir(project_root: &Path) -> PathBuf {
    project_root.join(".dm").join("sync-staging")
}

fn sync_dry_run_message(
    project_root: &Path,
    identity: &crate::identity::Identity,
) -> anyhow::Result<String> {
    let repo = identity.canonical_dm_repo.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_repo not set; sync dry-run needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let pinned = identity.canonical_dm_revision.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_revision not set; sync dry-run needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let temp_root = make_sync_temp_dir()?;
    let clone_dir = temp_root.join("canonical");
    let result = (|| {
        run_git(
            None,
            &[
                "clone",
                "--quiet",
                repo,
                clone_dir.to_string_lossy().as_ref(),
            ],
            "clone canonical repository for dry-run",
        )?;
        ensure_revision_available(&clone_dir, repo, pinned)?;
        let diff = run_git(
            Some(&clone_dir),
            &["diff", "--name-status", &format!("{pinned}..HEAD")],
            "compute canonical dry-run diff",
        )?;
        let plan = build_sync_plan(project_root, &clone_dir, pinned, &diff);
        Ok(format_dry_run(identity, &plan))
    })();
    let _ = fs::remove_dir_all(&temp_root);
    result
}

fn sync_apply_stage_message(
    project_root: &Path,
    identity: &crate::identity::Identity,
) -> anyhow::Result<String> {
    let repo = identity.canonical_dm_repo.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_repo not set; sync apply needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let pinned = identity.canonical_dm_revision.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_revision not set; sync apply needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let staging_root = sync_staging_dir(project_root);
    if staging_root.exists() {
        anyhow::bail!(
            "sync staging already exists at {}. Try: run `dm sync --abort` to discard it before retrying, or inspect the staged tree manually.",
            staging_root.display(),
        );
    }
    fs::create_dir_all(&staging_root).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create sync staging directory {}: {}. Try: check write permissions on the host project's .dm directory.",
            staging_root.display(),
            e
        )
    })?;
    let clone_dir = staging_root.join("canonical");
    let staged_dir = staging_root.join("staged");
    let result = (|| {
        run_git(
            None,
            &[
                "clone",
                "--quiet",
                repo,
                clone_dir.to_string_lossy().as_ref(),
            ],
            "clone canonical repository for sync apply",
        )?;
        ensure_revision_available(&clone_dir, repo, pinned)?;
        let head = run_git(
            Some(&clone_dir),
            &["rev-parse", "HEAD"],
            "read canonical HEAD for sync apply",
        )?
        .trim()
        .to_string();
        let diff = run_git(
            Some(&clone_dir),
            &["diff", "--name-status", &format!("{pinned}..HEAD")],
            "compute canonical apply diff",
        )?;
        let plan = build_sync_plan(project_root, &clone_dir, pinned, &diff);
        mirror_host_tree(project_root, &staged_dir)?;
        let stage_message = stage_sync_plan(identity, &plan, &clone_dir, &head, &staged_dir)?;
        Ok((stage_message, plan, head))
    })();
    let (stage_message, plan, head) = match result {
        Err(e) => {
            let _ = fs::remove_dir_all(&staging_root);
            return Err(e);
        }
        Ok(parts) => parts,
    };
    // Cycle 12: cargo-check gate over the merged staged tree. On
    // failure, preserve the staging directory so the operator can
    // inspect the merged tree, fix host-modified files, and re-run.
    // The skip env var honors operators on slow machines and tests
    // that don't ship a compilable Rust crate in the staged mirror.
    match run_cargo_check_on_staged(&staged_dir) {
        Ok(note) => {
            let applied = apply_staged_kernel_entries(project_root, &staged_dir, &plan)?;
            write_updated_identity(project_root, identity, &head)?;
            append_sync_audit_log(project_root, &plan.pinned, &head, applied)?;
            fs::remove_dir_all(&staging_root).map_err(|e| {
                anyhow::anyhow!(
                    "Applied sync but failed to remove staging directory {}: {}. Try: remove it manually before running dm sync again.",
                    staging_root.display(),
                    e
                )
            })?;
            Ok(format!(
                "{stage_message}\n{note}\nApplied {applied} kernel file(s).\nUpdated canonical_dm_revision to {head}.\nRemoved sync staging."
            ))
        }
        Err(check_err) => Err(anyhow::anyhow!(
            "{stage_message}\ncargo check failed in staged tree:\n{check_err}\nTry: review the cargo errors above, fix host-modified files, then re-run `dm sync` — or run `dm sync --abort` to discard the staging tree."
        )),
    }
}

/// Run `cargo check` over the merged staged tree to gate the apply
/// path. Returns Ok with a status note ("passed" or "skipped") when the
/// gate is satisfied; Err with the raw cargo stderr when the merged
/// tree fails to compile.
///
/// `DM_SYNC_SKIP_CARGO_CHECK=1` (or any non-empty value) skips the
/// check. This is both a test seam — unit tests don't always set up a
/// compilable Rust crate in the staged mirror — and an operator escape
/// hatch for slow machines or air-gapped environments where running a
/// full check on the merged tree is impractical.
fn run_cargo_check_on_staged(staged_dir: &Path) -> anyhow::Result<String> {
    if std::env::var_os("DM_SYNC_SKIP_CARGO_CHECK")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return Ok("cargo check: skipped (DM_SYNC_SKIP_CARGO_CHECK set).".to_string());
    }
    let manifest = staged_dir.join("Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "check",
            "--manifest-path",
            manifest.to_string_lossy().as_ref(),
        ])
        .current_dir(staged_dir)
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to run `cargo check` on staged tree {}: {}. Try: install the Rust toolchain, or set DM_SYNC_SKIP_CARGO_CHECK=1 to skip the gate.",
                staged_dir.display(),
                e
            )
        })?;
    if output.status.success() {
        return Ok("cargo check: passed.".to_string());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    anyhow::bail!("{}", stderr.trim())
}

/// Discard a half-applied sync staging directory so a later `dm sync`
/// can re-stage from scratch. Idempotent — a missing staging dir is
/// reported informationally, not as an error, so operators can pipe
/// `dm sync --abort && dm sync` in scripts safely.
fn sync_abort_message(
    project_root: &Path,
    identity: &crate::identity::Identity,
) -> anyhow::Result<String> {
    let staging_root = sync_staging_dir(project_root);
    if !staging_root.exists() {
        return Ok(format!(
            "No pending sync staging at {}. Try: run `dm sync` to start one.",
            staging_root.display(),
        ));
    }
    fs::remove_dir_all(&staging_root).map_err(|e| {
        anyhow::anyhow!(
            "Failed to remove sync staging directory {}: {}. Try: check permissions or remove the directory manually.",
            staging_root.display(),
            e
        )
    })?;
    Ok(format!(
        "Discarded sync staging at {} for host project `{}`. Try: run `dm sync` to start a fresh apply.",
        staging_root.display(),
        identity.display_name(),
    ))
}

/// Read a wiki page from the canonical clone and resolve its sync scope
/// from the `layer:` frontmatter. Returns the resolved scope plus a short
/// note describing how the layer was determined — included in dry-run
/// output so an operator can see why a page landed in a given bucket.
///
/// Defensive fallback: when the file is unreadable or its frontmatter is
/// unparseable, the page is treated as host-scope so sync would refuse to
/// touch it. Sync would rather surface an unresolvable page than risk
/// clobbering one whose intent it cannot read.
fn resolve_wiki_layer(clone_dir: &Path, rel: &str) -> (SyncScope, &'static str) {
    let path = clone_dir.join(rel);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return (SyncScope::Host, "unreadable, defaulting to host-safe"),
    };
    match crate::wiki::WikiPage::parse(&text) {
        Some(page) => match page.layer {
            crate::wiki::Layer::Host => (SyncScope::Host, "layer: host"),
            crate::wiki::Layer::Kernel => (SyncScope::Kernel, "layer: kernel"),
        },
        None => (
            SyncScope::Host,
            "frontmatter unparseable, defaulting to host-safe",
        ),
    }
}

fn sync_status_message(identity: &crate::identity::Identity) -> anyhow::Result<String> {
    let repo = identity.canonical_dm_repo.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_repo not set; sync needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let pinned = identity.canonical_dm_revision.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "canonical_dm_revision not set; sync needs it. Try: re-spawn or set it manually in .dm/identity.toml."
        )
    })?;
    let head = canonical_head_for_repo(repo)?;
    let status = if pinned == head {
        "up to date"
    } else {
        "pinned and canonical HEAD differ"
    };

    Ok(format!(
        "Host project: {}\nPinned revision: {}\nCanonical repo: {}\nCanonical HEAD: {}\nStatus: {}",
        identity.display_name(),
        pinned,
        repo,
        head,
        status,
    ))
}

fn canonical_head_for_repo(repo: &str) -> anyhow::Result<String> {
    let output = Command::new("git")
        .args(["ls-remote", repo, "HEAD"])
        .current_dir("/")
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to run `git ls-remote {} HEAD`: {}. Try: install git and verify canonical_dm_repo is reachable.",
                repo,
                e
            )
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Failed to query canonical HEAD from {}: {}. Try: verify canonical_dm_repo is reachable.",
            repo,
            stderr.trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let sha = stdout
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().next())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not parse canonical HEAD from `git ls-remote {} HEAD`. Try: verify canonical_dm_repo points to a git repository with HEAD.",
                repo
            )
        })?;
    Ok(sha.to_string())
}

fn ensure_revision_available(clone_dir: &Path, repo: &str, rev: &str) -> anyhow::Result<()> {
    let exists = Command::new("git")
        .args(["cat-file", "-e", &format!("{rev}^{{commit}}")])
        .current_dir(clone_dir)
        .status()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to inspect pinned revision {}: {}. Try: verify git is installed.",
                rev,
                e
            )
        })?;
    if exists.success() {
        return Ok(());
    }
    run_git(
        Some(clone_dir),
        &["fetch", "--quiet", "origin", rev],
        &format!("fetch pinned revision {rev} from {repo}"),
    )?;
    Ok(())
}

fn host_delta_status(
    project_root: &Path,
    clone_dir: &Path,
    pinned: &str,
    rel: &str,
) -> HostDeltaStatus {
    let canonical_at_pinned = git_show_file_at_revision(clone_dir, pinned, rel);
    let host_path = project_root.join(rel);
    let host_current = fs::read(&host_path);
    match (canonical_at_pinned, host_current) {
        (Ok(Some(canonical)), Ok(host)) if canonical == host => HostDeltaStatus::CleanApply,
        (Ok(Some(_)), Ok(_)) => HostDeltaStatus::HostModified,
        (Ok(Some(_)), Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            HostDeltaStatus::HostModified
        }
        (Ok(None), Ok(_)) => HostDeltaStatus::HostModified,
        (Ok(None), Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            HostDeltaStatus::NewKernelFile
        }
        _ => HostDeltaStatus::HostUnreadable,
    }
}

/// Build the per-file plan for a single sync run.
///
/// Walks the canonical-side diff, classifies each path, resolves wiki
/// pages by reading their `layer:` frontmatter from the clone, and for
/// every kernel-scope path computes the host-vs-canonical@pinned delta.
/// Pure with respect to the working tree — no writes occur.
fn build_sync_plan(
    project_root: &Path,
    clone_dir: &Path,
    pinned: &str,
    diff_name_status: &str,
) -> SyncPlan {
    let entries: Vec<SyncPlanEntry> = diff_name_status
        .lines()
        .filter_map(parse_name_status_line)
        .map(|diff_entry| {
            let rel = diff_entry.rel;
            let initial = classify_for_sync(Path::new(&rel));
            let (scope, layer_note) = match initial {
                SyncScope::WikiPage => {
                    let (resolved, note) = resolve_wiki_layer(clone_dir, &rel);
                    (resolved, Some(note))
                }
                other => (other, None),
            };
            let host_status =
                if scope == SyncScope::Kernel && diff_entry.kind != ChangeKind::Renamed {
                    Some(host_delta_status(project_root, clone_dir, pinned, &rel))
                } else {
                    None
                };
            SyncPlanEntry {
                rel,
                change_kind: diff_entry.kind,
                scope,
                layer_note,
                host_status,
            }
        })
        .collect();
    SyncPlan {
        pinned: pinned.to_string(),
        entries,
    }
}

struct DiffEntry {
    kind: ChangeKind,
    rel: String,
}

fn parse_name_status_line(line: &str) -> Option<DiffEntry> {
    let mut parts = line.split('\t');
    let status = parts.next()?.trim();
    if status.is_empty() {
        return None;
    }
    let kind = match status.as_bytes()[0] as char {
        'A' => ChangeKind::Added,
        'M' => ChangeKind::Modified,
        'D' => ChangeKind::Deleted,
        'R' => ChangeKind::Renamed,
        _ => ChangeKind::Modified,
    };
    let rel = match kind {
        ChangeKind::Renamed => {
            let _old = parts.next()?;
            parts.next()?
        }
        _ => parts.next()?,
    };
    Some(DiffEntry {
        kind,
        rel: rel.trim().to_string(),
    })
}

fn git_show_file_at_revision(
    clone_dir: &Path,
    revision: &str,
    rel: &str,
) -> anyhow::Result<Option<Vec<u8>>> {
    let spec = format!("{revision}:{rel}");
    let output = Command::new("git")
        .args(["show", &spec])
        .current_dir(clone_dir)
        .output()
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to inspect {} at pinned revision: {}. Try: verify git is installed.",
                rel,
                e
            )
        })?;
    if output.status.success() {
        return Ok(Some(output.stdout));
    }
    Ok(None)
}

fn mirror_host_tree(project_root: &Path, staged_dir: &Path) -> anyhow::Result<()> {
    copy_mirror_entries(project_root, project_root, staged_dir)
}

fn copy_mirror_entries(
    project_root: &Path,
    current: &Path,
    staged_dir: &Path,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(current).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read host project directory {}: {}. Try: check permissions before running dm sync again.",
            current.display(),
            e
        )
    })? {
        let entry = entry.map_err(|e| {
            anyhow::anyhow!(
                "Failed to read host project directory entry in {}: {}. Try: check permissions before running dm sync again.",
                current.display(),
                e
            )
        })?;
        let path = entry.path();
        let rel = path.strip_prefix(project_root).map_err(|e| {
            anyhow::anyhow!(
                "Failed to mirror host path {}: {}. Try: run dm sync from the project root.",
                path.display(),
                e
            )
        })?;
        let file_type = entry.file_type().map_err(|e| {
            anyhow::anyhow!(
                "Failed to inspect host path {}: {}. Try: check permissions before running dm sync again.",
                path.display(),
                e
            )
        })?;
        if should_skip_mirror_path(rel) {
            continue;
        }
        let target = staged_dir.join(rel);
        if file_type.is_dir() {
            fs::create_dir_all(&target).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to mirror host directory {}: {}. Try: check permissions on .dm/sync-staging.",
                    target.display(),
                    e
                )
            })?;
            copy_mirror_entries(project_root, &path, staged_dir)?;
        } else if file_type.is_file() {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to create mirrored host directory {}: {}. Try: check permissions on .dm/sync-staging.",
                        parent.display(),
                        e
                    )
                })?;
            }
            fs::copy(&path, &target).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to mirror host file {} to {}: {}. Try: check permissions on .dm/sync-staging.",
                    path.display(),
                    target.display(),
                    e
                )
            })?;
        }
    }
    Ok(())
}

fn should_skip_mirror_path(rel: &Path) -> bool {
    let s = rel.to_string_lossy().replace('\\', "/");
    s == "target"
        || s.starts_with("target/")
        || s == ".git"
        || s.starts_with(".git/")
        || s == ".dm/sync-staging"
        || s.starts_with(".dm/sync-staging/")
        || s == ".dm/sessions"
        || s.starts_with(".dm/sessions/")
        || matches!(
            s.as_str(),
            ".dm/wiki/log.md" | ".dm/wiki/.summary-dirty" | ".dm/wiki/index.md"
        )
}

fn stage_sync_plan(
    identity: &crate::identity::Identity,
    plan: &SyncPlan,
    clone_dir: &Path,
    head: &str,
    staged_dir: &Path,
) -> anyhow::Result<String> {
    fs::create_dir_all(staged_dir).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create sync staging directory {}: {}. Try: check permissions on the system temp directory.",
            staged_dir.display(),
            e
        )
    })?;

    let mut staged = Vec::new();
    let mut deferred = Vec::new();
    for entry in &plan.entries {
        if entry.scope != SyncScope::Kernel {
            continue;
        }
        match entry.host_status {
            Some(HostDeltaStatus::CleanApply) if entry.change_kind == ChangeKind::Deleted => {
                staged.push(entry.rel.clone());
            }
            Some(HostDeltaStatus::CleanApply | HostDeltaStatus::NewKernelFile) => {
                match git_show_file_at_revision(clone_dir, head, &entry.rel)? {
                    Some(bytes) => {
                        let Some(target) = safe_join(staged_dir, &entry.rel) else {
                            deferred.push((
                                entry.rel.clone(),
                                "unsafe path — review needed".to_string(),
                            ));
                            continue;
                        };
                        if let Some(parent) = target.parent() {
                            fs::create_dir_all(parent).map_err(|e| {
                                anyhow::anyhow!(
                                    "Failed to create staging directory {}: {}. Try: check permissions on the system temp directory.",
                                    parent.display(),
                                    e
                                )
                            })?;
                        }
                        fs::write(&target, bytes).map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to stage {}: {}. Try: check permissions on the system temp directory.",
                                entry.rel,
                                e
                            )
                        })?;
                        staged.push(entry.rel.clone());
                    }
                    None => deferred.push((
                        entry.rel.clone(),
                        "canonical HEAD missing — review needed".to_string(),
                    )),
                }
            }
            Some(status) => {
                let reason = if entry.change_kind == ChangeKind::Deleted {
                    format!("deleted upstream; {}", status.display_label())
                } else {
                    status.display_label().to_string()
                };
                deferred.push((entry.rel.clone(), reason));
            }
            None => deferred.push((
                entry.rel.clone(),
                entry
                    .change_kind
                    .note()
                    .unwrap_or("missing host delta — review needed")
                    .to_string(),
            )),
        }
    }

    let mut out = format!(
        "Host project: {}\nPinned revision: {}\nCanonical HEAD: {}\nStaging directory: {}\nStaged {} kernel file(s).",
        identity.display_name(),
        plan.pinned,
        head,
        staged_dir.display(),
        staged.len(),
    );
    out.push_str(&format!(
        "\nDeferred for review: {} file(s)",
        deferred.len()
    ));
    for (rel, reason) in &deferred {
        out.push_str("\n  ");
        out.push_str(rel);
        out.push_str(" [");
        out.push_str(reason);
        out.push(']');
    }
    out.push_str("\nStaged tree ready; host tree unchanged until cargo check passes.");
    Ok(out)
}

fn apply_staged_kernel_entries(
    project_root: &Path,
    staged_dir: &Path,
    plan: &SyncPlan,
) -> anyhow::Result<usize> {
    let backup_dir = sync_staging_dir(project_root).join("backup");
    let mut applied = Vec::new();
    for entry in &plan.entries {
        if entry.scope != SyncScope::Kernel {
            continue;
        }
        if !matches!(
            entry.host_status,
            Some(HostDeltaStatus::CleanApply | HostDeltaStatus::NewKernelFile)
        ) {
            continue;
        }
        let result = if entry.change_kind == ChangeKind::Deleted {
            apply_one_deleted_entry(project_root, &backup_dir, entry, &mut applied)
        } else {
            apply_one_staged_entry(project_root, staged_dir, &backup_dir, entry, &mut applied)
        };
        if let Err(e) = result {
            rollback_applied_entries(applied);
            return Err(e);
        }
    }
    Ok(applied.len())
}

fn apply_one_deleted_entry(
    project_root: &Path,
    backup_dir: &Path,
    entry: &SyncPlanEntry,
    applied: &mut Vec<AppliedEntry>,
) -> anyhow::Result<()> {
    let target = safe_join(project_root, &entry.rel).ok_or_else(|| {
        anyhow::anyhow!(
            "Refusing to delete unsafe host path {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
            entry.rel
        )
    })?;
    if !target.exists() {
        return Ok(());
    }
    let backup = safe_join(backup_dir, &entry.rel).ok_or_else(|| {
        anyhow::anyhow!(
            "Refusing to back up unsafe deleted path {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
            entry.rel
        )
    })?;
    if let Some(parent) = backup.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create sync backup directory {}: {}. Try: check permissions before rerunning dm sync.",
                parent.display(),
                e
            )
        })?;
    }
    fs::rename(&target, &backup).map_err(|e| {
        anyhow::anyhow!(
            "Failed to remove deleted kernel file {}: {}. Try: check permissions before rerunning dm sync.",
            target.display(),
            e
        )
    })?;
    sync_parent(&target);
    applied.push(AppliedEntry {
        target,
        backup: Some(backup),
    });
    Ok(())
}

fn apply_one_staged_entry(
    project_root: &Path,
    staged_dir: &Path,
    backup_dir: &Path,
    entry: &SyncPlanEntry,
    applied: &mut Vec<AppliedEntry>,
) -> anyhow::Result<()> {
    let source = safe_join(staged_dir, &entry.rel).ok_or_else(|| {
        anyhow::anyhow!(
            "Refusing to apply unsafe staged path {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
            entry.rel
        )
    })?;
    let target = safe_join(project_root, &entry.rel).ok_or_else(|| {
        anyhow::anyhow!(
            "Refusing to apply unsafe host path {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
            entry.rel
        )
    })?;
    if !source.exists() {
        anyhow::bail!(
            "Staged file {} is missing. Try: run `dm sync --abort` and retry.",
            source.display()
        );
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!(
                "Failed to create host directory {}: {}. Try: check permissions before rerunning dm sync.",
                parent.display(),
                e
            )
        })?;
    }
    let backup = if target.exists() {
        let backup = safe_join(backup_dir, &entry.rel).ok_or_else(|| {
            anyhow::anyhow!(
                "Refusing to back up unsafe host path {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
                entry.rel
            )
        })?;
        if let Some(parent) = backup.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create sync backup directory {}: {}. Try: check permissions before rerunning dm sync.",
                    parent.display(),
                    e
                )
            })?;
        }
        fs::rename(&target, &backup).map_err(|e| {
            anyhow::anyhow!(
                "Failed to back up host file {}: {}. Try: check permissions before rerunning dm sync.",
                target.display(),
                e
            )
        })?;
        Some(backup)
    } else {
        None
    };
    if let Err(e) = fs::rename(&source, &target) {
        if let Some(backup) = &backup {
            let _ = fs::rename(backup, &target);
        }
        anyhow::bail!(
            "Failed to apply staged file {} to {}: {}. Try: inspect .dm/sync-staging and run `dm sync --abort`.",
            source.display(),
            target.display(),
            e
        );
    }
    sync_parent(&target);
    applied.push(AppliedEntry { target, backup });
    Ok(())
}

#[derive(Debug)]
struct AppliedEntry {
    target: PathBuf,
    backup: Option<PathBuf>,
}

fn rollback_applied_entries(mut applied: Vec<AppliedEntry>) {
    while let Some(entry) = applied.pop() {
        if let Some(backup) = entry.backup {
            let _ = fs::remove_file(&entry.target);
            let _ = fs::rename(backup, &entry.target);
            sync_parent(&entry.target);
        } else {
            let _ = fs::remove_file(&entry.target);
            sync_parent(&entry.target);
        }
    }
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }
}

fn write_updated_identity(
    project_root: &Path,
    identity: &crate::identity::Identity,
    head: &str,
) -> anyhow::Result<()> {
    let mut updated = identity.clone();
    updated.canonical_dm_revision = Some(head.to_string());
    let text = crate::identity::render_toml(&updated)?;
    let path = project_root
        .join(".dm")
        .join(crate::identity::IDENTITY_FILENAME);
    fs::write(&path, text).map_err(|e| {
        anyhow::anyhow!(
            "Failed to update {}: {}. Try: check permissions and inspect .dm/sync-staging before rerunning dm sync.",
            path.display(),
            e
        )
    })
}

fn append_sync_audit_log(
    project_root: &Path,
    pinned: &str,
    head: &str,
    applied: usize,
) -> anyhow::Result<()> {
    let wiki_dir = project_root.join(".dm").join("wiki");
    fs::create_dir_all(&wiki_dir).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create wiki directory {}: {}. Try: check permissions before rerunning dm sync.",
            wiki_dir.display(),
            e
        )
    })?;
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");
    let path = wiki_dir.join("log.md");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to append sync audit log {}: {}. Try: check permissions before rerunning dm sync.",
                path.display(),
                e
            )
        })?;
    writeln!(
        file,
        "[{timestamp}] dm sync: applied {applied} kernel file(s) ({pinned} -> {head})"
    )
    .map_err(|e| {
        anyhow::anyhow!(
            "Failed to write sync audit log {}: {}. Try: check permissions before rerunning dm sync.",
            path.display(),
            e
        )
    })
}

fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    let path = Path::new(rel);
    if path.is_absolute() {
        return None;
    }
    if path
        .components()
        .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        return None;
    }
    Some(root.join(path))
}

fn format_dry_run(identity: &crate::identity::Identity, plan: &SyncPlan) -> String {
    let mut out = format!(
        "Host project: {}\nPinned revision: {}\nFiles canonical changed since {}:",
        identity.display_name(),
        plan.pinned,
        plan.pinned,
    );

    if plan.entries.is_empty() {
        out.push_str("\n  (none)\nSummary: 0 changed file(s)");
        return out;
    }

    for entry in &plan.entries {
        out.push_str("\n  ");
        out.push_str(&entry.rel);
        out.push_str(" (");
        out.push_str(scope_label(entry.scope));
        if let Some(note) = entry.change_kind.note() {
            out.push_str(") [");
            out.push_str(note);
            out.push(']');
        } else {
            out.push(')');
        }
        if let Some(note) = entry.layer_note {
            out.push_str(" [");
            out.push_str(note);
            out.push(']');
        }
        if let Some(status) = entry.host_status {
            out.push_str(" [");
            out.push_str(status.display_label());
            out.push(']');
        }
    }
    out.push_str(&format!(
        "\nSummary: {} changed file(s): {} kernel, {} host",
        plan.entries.len(),
        plan.kernel_count(),
        plan.host_count(),
    ));
    out
}

fn scope_label(scope: SyncScope) -> &'static str {
    match scope {
        SyncScope::Kernel => "kernel",
        SyncScope::Host => "host — would not apply",
        SyncScope::WikiPage => "wiki — layer-dependent",
    }
}

fn run_git(cwd: Option<&Path>, args: &[&str], action: &str) -> anyhow::Result<String> {
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(cwd) = cwd {
        cmd.current_dir(cwd);
    } else {
        cmd.current_dir("/");
    }
    let output = cmd.output().map_err(|e| {
        anyhow::anyhow!(
            "Failed to {} with `git {}`: {}. Try: install git and verify canonical_dm_repo is reachable.",
            action,
            args.join(" "),
            e
        )
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Failed to {} with `git {}`: {}. Try: verify canonical_dm_repo and canonical_dm_revision are valid.",
            action,
            args.join(" "),
            stderr.trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn make_sync_temp_dir() -> anyhow::Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("dm-sync-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|e| {
        anyhow::anyhow!(
            "Failed to create sync staging directory {}: {}. Try: check permissions on the system temp directory.",
            dir.display(),
            e
        )
    })?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{render_toml, Identity, Mode, IDENTITY_FILENAME};

    fn write_identity(root: &Path, identity: &Identity) {
        let dm = root.join(".dm");
        std::fs::create_dir_all(&dm).unwrap();
        let toml = render_toml(identity).unwrap();
        std::fs::write(dm.join(IDENTITY_FILENAME), toml).unwrap();
    }

    fn write_minimal_crate(root: &Path) {
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"finance-app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .unwrap();
        let lib = root.join("src/lib.rs");
        std::fs::create_dir_all(lib.parent().unwrap()).unwrap();
        std::fs::write(lib, "pub fn host_lib() {}\n").unwrap();
    }

    #[test]
    fn sync_rejects_kernel_mode_with_next_step() {
        let tmp = tempfile::tempdir().unwrap();

        let err = sync_message_for_project(tmp.path(), SyncArgs::default()).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("only meaningful in host mode"), "{msg}");
        assert!(msg.contains("Try:"), "{msg}");
        assert!(msg.contains("dm spawn"), "{msg}");
    }

    #[test]
    fn sync_abort_with_no_pending_staging_is_informational() {
        // No staging dir exists yet — `--abort` is idempotent: prints an
        // informational note and returns Ok so operators can pipe
        // `dm sync --abort && dm sync` safely.
        let tmp = tempfile::tempdir().unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some("abc123".to_string()),
                canonical_dm_repo: None,
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                abort: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(msg.contains("No pending sync staging"), "{msg}");
        assert!(msg.contains(".dm/sync-staging"), "{msg}");
        assert!(msg.contains("Try:"), "{msg}");
    }

    fn run_git(dir: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap_or_else(|e| panic!("failed to run git {:?}: {}", args, e));
        assert!(
            output.status.success(),
            "git {:?} failed:\nstdout={}\nstderr={}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    fn current_head(repo: &Path) -> String {
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn make_canonical_repo() -> (tempfile::TempDir, String) {
        let repo = tempfile::tempdir().unwrap();
        run_git(repo.path(), &["init"]);
        std::fs::write(repo.path().join("README.md"), "one\n").unwrap();
        run_git(repo.path(), &["add", "README.md"]);
        run_git(
            repo.path(),
            &[
                "-c",
                "user.name=dm-test",
                "-c",
                "user.email=dm-test@example.invalid",
                "commit",
                "-m",
                "initial",
            ],
        );
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo.path())
            .output()
            .unwrap();
        assert!(output.status.success());
        let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (repo, head)
    }

    fn commit_file(repo: &Path, rel: &str, contents: &str, message: &str) -> String {
        let path = repo.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
        run_git(repo, &["add", rel]);
        run_git(
            repo,
            &[
                "-c",
                "user.name=dm-test",
                "-c",
                "user.email=dm-test@example.invalid",
                "commit",
                "-m",
                message,
            ],
        );
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(repo)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    #[test]
    fn sync_status_reports_matching_canonical_head() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, head) = make_canonical_repo();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(head.clone()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                status: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(msg.contains("Host project: finance-app"), "{msg}");
        assert!(msg.contains(&format!("Pinned revision: {head}")), "{msg}");
        assert!(msg.contains(&format!("Canonical HEAD: {head}")), "{msg}");
        assert!(msg.contains("Status: up to date"), "{msg}");
    }

    #[test]
    fn sync_status_reports_when_pinned_revision_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, head) = make_canonical_repo();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some("deadbeef".to_string()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                status: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(msg.contains("Pinned revision: deadbeef"), "{msg}");
        assert!(msg.contains(&format!("Canonical HEAD: {head}")), "{msg}");
        assert!(
            msg.contains("Status: pinned and canonical HEAD differ"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_reports_no_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, head) = make_canonical_repo();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(head),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(msg.contains("Files canonical changed since"), "{msg}");
        assert!(msg.contains("  (none)"), "{msg}");
        assert!(msg.contains("Summary: 0 changed file(s)"), "{msg}");
    }

    #[test]
    fn sync_dry_run_buckets_changed_paths_by_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, pinned) = make_canonical_repo();
        commit_file(
            repo.path(),
            "src/new_kernel.rs",
            "pub fn kernel() {}\n",
            "kernel",
        );
        commit_file(repo.path(), "src/host_main.rs", "fn main() {}\n", "host");
        commit_file(
            repo.path(),
            ".dm/wiki/concepts/kernel-page.md",
            "---\ntitle: Kernel page\ntype: concept\nlast_updated: 2026-04-26 00:00:00\n---\n# Kernel page\n",
            "wiki",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned.clone()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains(&format!("Files canonical changed since {pinned}:")),
            "{msg}"
        );
        assert!(msg.contains("src/new_kernel.rs (kernel)"), "{msg}");
        assert!(
            msg.contains("src/new_kernel.rs (kernel) [new kernel file, would apply cleanly]"),
            "{msg}"
        );
        assert!(
            msg.contains("src/host_main.rs (host — would not apply)"),
            "{msg}"
        );
        // Wiki page lacks an explicit `layer:` field → defaults to kernel
        // per the parser contract; dry-run resolves it accordingly and
        // tags the line so an operator sees how the layer was decided.
        assert!(
            msg.contains(".dm/wiki/concepts/kernel-page.md (kernel) [layer: kernel]"),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 3 changed file(s): 2 kernel, 1 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_flags_host_modified_kernel_path_for_review() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        let pinned = commit_file(
            repo.path(),
            "src/sync_target.rs",
            "pub fn value() -> u8 { 1 }\n",
            "add sync target",
        );
        commit_file(
            repo.path(),
            "src/sync_target.rs",
            "pub fn value() -> u8 { 2 }\n",
            "canonical update",
        );
        let host_path = tmp.path().join("src/sync_target.rs");
        std::fs::create_dir_all(host_path.parent().unwrap()).unwrap();
        std::fs::write(&host_path, "pub fn value() -> u8 { 99 }\n").unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains("src/sync_target.rs (kernel) [host-modified — review needed]"),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 1 changed file(s): 1 kernel, 0 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_reports_clean_canonical_deletion_precisely() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        let pinned = commit_file(
            repo.path(),
            "src/remove_me.rs",
            "pub fn remove_me() {}\n",
            "add removable file",
        );
        run_git(repo.path(), &["rm", "src/remove_me.rs"]);
        run_git(
            repo.path(),
            &[
                "-c",
                "user.name=dm-test",
                "-c",
                "user.email=dm-test@example.invalid",
                "commit",
                "-m",
                "delete removable file",
            ],
        );
        let host_file = tmp.path().join("src/remove_me.rs");
        std::fs::create_dir_all(host_file.parent().unwrap()).unwrap();
        std::fs::write(&host_file, "pub fn remove_me() {}\n").unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains(
                "src/remove_me.rs (kernel) [deleted] [kernel-only change, would apply cleanly]"
            ),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 1 changed file(s): 1 kernel, 0 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_apply_stages_clean_kernel_files_and_defers_host_modified() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        commit_file(
            repo.path(),
            "src/clean.rs",
            "pub fn clean() -> u8 { 1 }\n",
            "add clean file",
        );
        let pinned = commit_file(
            repo.path(),
            "src/conflict.rs",
            "pub fn conflict() -> u8 { 1 }\n",
            "add conflict file",
        );
        commit_file(
            repo.path(),
            "src/clean.rs",
            "pub fn clean() -> u8 { 2 }\n",
            "update clean file",
        );
        commit_file(
            repo.path(),
            "src/conflict.rs",
            "pub fn conflict() -> u8 { 2 }\n",
            "update conflict file",
        );
        let head = commit_file(
            repo.path(),
            "src/new_kernel.rs",
            "pub fn new_kernel() -> u8 { 3 }\n",
            "add new kernel file",
        );
        for (rel, contents) in [
            ("src/clean.rs", "pub fn clean() -> u8 { 1 }\n"),
            ("src/conflict.rs", "pub fn conflict() -> u8 { 99 }\n"),
        ] {
            let path = tmp.path().join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
        write_minimal_crate(tmp.path());
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(tmp.path(), SyncArgs::default()).unwrap();

        assert!(msg.contains("Staged 2 kernel file(s)."), "{msg}");
        assert!(msg.contains("Deferred for review: 1 file(s)"), "{msg}");
        assert!(
            msg.contains("src/conflict.rs [host-modified — review needed]"),
            "{msg}"
        );
        assert!(msg.contains("Applied 2 kernel file(s)."), "{msg}");
        assert!(
            msg.contains(&format!("Updated canonical_dm_revision to {head}")),
            "{msg}"
        );
        assert!(msg.contains("Removed sync staging."), "{msg}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("src/clean.rs")).unwrap(),
            "pub fn clean() -> u8 { 2 }\n"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("src/new_kernel.rs")).unwrap(),
            "pub fn new_kernel() -> u8 { 3 }\n"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("src/conflict.rs")).unwrap(),
            "pub fn conflict() -> u8 { 99 }\n"
        );
        let updated = crate::identity::load_at(tmp.path()).unwrap();
        assert_eq!(
            updated.canonical_dm_revision.as_deref(),
            Some(head.as_str())
        );
        let log = std::fs::read_to_string(tmp.path().join(".dm/wiki/log.md")).unwrap();
        assert!(log.contains("dm sync: applied 2 kernel file(s)"), "{log}");
        assert!(log.contains(&head), "{log}");
        assert!(!tmp.path().join(".dm/sync-staging").exists());
    }

    #[test]
    fn sync_apply_removes_clean_canonical_deletion() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        write_minimal_crate(tmp.path());
        commit_file(
            repo.path(),
            "Cargo.toml",
            "[package]\nname = \"finance-app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
            "add manifest",
        );
        commit_file(
            repo.path(),
            "src/lib.rs",
            "pub fn host_lib() {}\n",
            "add lib",
        );
        let pinned = commit_file(
            repo.path(),
            "src/remove_me.rs",
            "pub fn remove_me() {}\n",
            "add removable file",
        );
        run_git(repo.path(), &["rm", "src/remove_me.rs"]);
        run_git(
            repo.path(),
            &[
                "-c",
                "user.name=dm-test",
                "-c",
                "user.email=dm-test@example.invalid",
                "commit",
                "-m",
                "delete removable file",
            ],
        );
        let head = current_head(repo.path());
        std::fs::write(
            tmp.path().join("src/remove_me.rs"),
            "pub fn remove_me() {}\n",
        )
        .unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(tmp.path(), SyncArgs::default()).unwrap();

        assert!(msg.contains("Applied 1 kernel file(s)."), "{msg}");
        assert!(!tmp.path().join("src/remove_me.rs").exists());
        let updated = crate::identity::load_at(tmp.path()).unwrap();
        assert_eq!(
            updated.canonical_dm_revision.as_deref(),
            Some(head.as_str())
        );
        let log = std::fs::read_to_string(tmp.path().join(".dm/wiki/log.md")).unwrap();
        assert!(log.contains("dm sync: applied 1 kernel file(s)"), "{log}");
    }

    #[test]
    fn sync_apply_persists_staging_inside_dot_dm_sync_staging() {
        // Apply must put its staging tree at a predictable location
        // inside the host project's `.dm/` so a future `dm sync --abort`
        // (and the cycle-11 swap-in) can find it. /tmp/dm-sync-...
        // staging from cycle-9 is not enough — it's per-process and
        // cleans up on success path inconsistently.
        std::env::remove_var("DM_SYNC_SKIP_CARGO_CHECK");
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        let pinned = commit_file(
            repo.path(),
            "src/seed.rs",
            "pub fn seed() {}\n",
            "seed file",
        );
        commit_file(
            repo.path(),
            "src/added.rs",
            "pub fn added() {}\n",
            "add file",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );
        let host_main = tmp.path().join("src/host_main.rs");
        std::fs::create_dir_all(host_main.parent().unwrap()).unwrap();
        std::fs::write(&host_main, "fn main() { println!(\"host\"); }\n").unwrap();
        let target_file = tmp.path().join("target/debug/ignored");
        std::fs::create_dir_all(target_file.parent().unwrap()).unwrap();
        std::fs::write(&target_file, "build output").unwrap();
        let session_file = tmp.path().join(".dm/sessions/session.json");
        std::fs::create_dir_all(session_file.parent().unwrap()).unwrap();
        std::fs::write(&session_file, "{}").unwrap();
        let wiki_dir = tmp.path().join(".dm/wiki");
        std::fs::create_dir_all(&wiki_dir).unwrap();
        std::fs::write(wiki_dir.join("index.md"), "# runtime index\n").unwrap();
        std::fs::write(wiki_dir.join("log.md"), "runtime log\n").unwrap();

        let err = sync_message_for_project(tmp.path(), SyncArgs::default()).unwrap_err();
        let msg = err.to_string();

        let staging_root = tmp.path().join(".dm").join("sync-staging");
        assert!(
            staging_root.is_dir(),
            "staging root must exist at {}",
            staging_root.display()
        );
        assert!(
            staging_root.join("staged").is_dir(),
            "staged subdir must exist"
        );
        assert!(
            msg.contains(".dm/sync-staging/staged"),
            "message must surface persistent staging path: {msg}",
        );
        assert!(msg.contains("cargo check failed in staged tree"), "{msg}");
        assert_eq!(
            std::fs::read_to_string(staging_root.join("staged/src/host_main.rs")).unwrap(),
            "fn main() { println!(\"host\"); }\n"
        );
        assert!(
            !staging_root.join("staged/target/debug/ignored").exists(),
            "target/ must not be mirrored into staging"
        );
        assert!(
            !staging_root
                .join("staged/.dm/sessions/session.json")
                .exists(),
            ".dm/sessions/ must not be mirrored into staging"
        );
        assert!(
            !staging_root.join("staged/.dm/wiki/index.md").exists(),
            "runtime wiki index must not be mirrored into staging"
        );
        assert!(
            !staging_root.join("staged/.dm/wiki/log.md").exists(),
            "runtime wiki log must not be mirrored into staging"
        );
        let _ = std::fs::remove_dir_all(&staging_root);
    }

    #[test]
    fn sync_apply_rolls_back_previous_renames_on_later_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let staged = tmp.path().join(".dm/sync-staging/staged");
        std::fs::create_dir_all(staged.join("src/b")).unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/a.rs"), "old a\n").unwrap();
        std::fs::write(tmp.path().join("src/b"), "blocks directory\n").unwrap();
        std::fs::write(staged.join("src/a.rs"), "new a\n").unwrap();
        std::fs::write(staged.join("src/b/c.rs"), "new c\n").unwrap();
        let plan = SyncPlan {
            pinned: "old".to_string(),
            entries: vec![
                SyncPlanEntry {
                    rel: "src/a.rs".to_string(),
                    change_kind: ChangeKind::Modified,
                    scope: SyncScope::Kernel,
                    layer_note: None,
                    host_status: Some(HostDeltaStatus::CleanApply),
                },
                SyncPlanEntry {
                    rel: "src/b/c.rs".to_string(),
                    change_kind: ChangeKind::Added,
                    scope: SyncScope::Kernel,
                    layer_note: None,
                    host_status: Some(HostDeltaStatus::NewKernelFile),
                },
            ],
        };

        let err = apply_staged_kernel_entries(tmp.path(), &staged, &plan).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("Failed to create host directory"), "{msg}");
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("src/a.rs")).unwrap(),
            "old a\n"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("src/b")).unwrap(),
            "blocks directory\n"
        );
        assert!(!tmp.path().join("src/b/c.rs").exists());
    }

    #[test]
    fn sync_apply_errors_when_staging_already_exists() {
        // Re-running `dm sync` while a staging tree exists must refuse
        // and point at `--abort` so the operator never silently
        // overwrites an in-flight sync.
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _initial) = make_canonical_repo();
        let pinned = commit_file(
            repo.path(),
            "src/seed.rs",
            "pub fn seed() {}\n",
            "seed file",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );
        let staging_root = tmp.path().join(".dm").join("sync-staging");
        std::fs::create_dir_all(staging_root.join("staged")).unwrap();
        std::fs::write(staging_root.join("staged/leftover.txt"), "stale").unwrap();

        let err = sync_message_for_project(tmp.path(), SyncArgs::default()).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("sync staging already exists"), "{msg}",);
        assert!(msg.contains("dm sync --abort"), "{msg}");
        // Existing staging must NOT have been clobbered.
        assert_eq!(
            std::fs::read_to_string(staging_root.join("staged/leftover.txt")).unwrap(),
            "stale",
        );
    }

    #[test]
    fn sync_abort_removes_existing_staging_directory() {
        // `--abort` removes the persistent staging tree so a fresh
        // `dm sync` can start over. Idempotent on the next call.
        let tmp = tempfile::tempdir().unwrap();
        let staging_root = tmp.path().join(".dm").join("sync-staging");
        std::fs::create_dir_all(staging_root.join("staged/src")).unwrap();
        std::fs::write(staging_root.join("staged/src/a.rs"), "pub fn a() {}\n").unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some("abc123".to_string()),
                canonical_dm_repo: Some("/local/repo".to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                abort: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(msg.contains("Discarded sync staging"), "{msg}");
        assert!(msg.contains("finance-app"), "{msg}");
        assert!(!staging_root.exists(), "staging dir must be removed");
    }

    #[test]
    fn sync_dry_run_resolves_wiki_layer_host_as_out_of_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, pinned) = make_canonical_repo();
        commit_file(
            repo.path(),
            ".dm/wiki/concepts/host-page.md",
            "---\ntitle: Host page\ntype: concept\nlayer: host\nlast_updated: 2026-04-26 00:00:00\n---\n# Host body\n",
            "wiki host page",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned.clone()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains(".dm/wiki/concepts/host-page.md (host — would not apply) [layer: host]"),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 1 changed file(s): 0 kernel, 1 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_resolves_wiki_layer_kernel_as_in_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, pinned) = make_canonical_repo();
        commit_file(
            repo.path(),
            ".dm/wiki/concepts/kernel-explicit.md",
            "---\ntitle: Kernel explicit\ntype: concept\nlayer: kernel\nlast_updated: 2026-04-26 00:00:00\n---\n# Kernel body\n",
            "wiki kernel page",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned.clone()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains(".dm/wiki/concepts/kernel-explicit.md (kernel) [layer: kernel]"),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 1 changed file(s): 1 kernel, 0 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_treats_unparseable_wiki_page_as_host_safe() {
        // A wiki page whose frontmatter cannot be parsed (no `---`
        // delimiters, malformed YAML, etc.) must NOT be silently treated
        // as kernel-scope — sync would risk clobbering an
        // operator-edited page. Defensive default is host-scope.
        let tmp = tempfile::tempdir().unwrap();
        let (repo, pinned) = make_canonical_repo();
        commit_file(
            repo.path(),
            ".dm/wiki/concepts/garbled.md",
            "this is not valid frontmatter\nno YAML block at all\n",
            "garbled wiki page",
        );
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some(pinned.clone()),
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let msg = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap();

        assert!(
            msg.contains(".dm/wiki/concepts/garbled.md (host — would not apply) [frontmatter unparseable, defaulting to host-safe]"),
            "{msg}"
        );
        assert!(
            msg.contains("Summary: 1 changed file(s): 0 kernel, 1 host"),
            "{msg}"
        );
    }

    #[test]
    fn sync_dry_run_missing_canonical_revision_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        let (repo, _head) = make_canonical_repo();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: None,
                canonical_dm_repo: Some(repo.path().to_string_lossy().to_string()),
                source: None,
            },
        );

        let err = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                dry_run: true,
                ..SyncArgs::default()
            },
        )
        .unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("canonical_dm_revision not set"), "{msg}");
        assert!(msg.contains("Try:"), "{msg}");
    }

    #[test]
    fn classify_for_sync_treats_kernel_source_and_manifests_as_kernel() {
        for path in [
            "src/main.rs",
            "src/lib.rs",
            "src/wiki/mod.rs",
            "src/wiki/types.rs",
            "src/identity.rs",
            "src/sync.rs",
            "build/.glue/glue.yaml",
            "Cargo.toml",
            "Cargo.lock",
        ] {
            assert_eq!(
                classify_for_sync(Path::new(path)),
                SyncScope::Kernel,
                "expected Kernel for {path}",
            );
        }
    }

    #[test]
    fn classify_for_sync_treats_host_skeleton_files_as_host() {
        for path in [
            "src/host_main.rs",
            "src/domain.rs",
            ".dm/identity.toml",
            ".dm/wiki/index.md",
            ".dm/wiki/log.md",
            ".dm/wiki/.summary-dirty",
            ".dm/wiki/summaries/project.md",
            ".dm/wiki/synthesis/run-30-synthesis.md",
            ".dm/sessions/abc.json",
            "DM.md",
            "README.md",
        ] {
            assert_eq!(
                classify_for_sync(Path::new(path)),
                SyncScope::Host,
                "expected Host for {path}",
            );
        }
    }

    #[test]
    fn classify_for_sync_treats_wiki_concept_and_entity_pages_as_layer_dependent() {
        for path in [
            ".dm/wiki/concepts/spawn-identity.md",
            ".dm/wiki/concepts/dm-sync.md",
            ".dm/wiki/entities/src_identity_rs.md",
            ".dm/wiki/entities/src_lib_rs.md",
        ] {
            assert_eq!(
                classify_for_sync(Path::new(path)),
                SyncScope::WikiPage,
                "expected WikiPage for {path}",
            );
        }
    }

    #[test]
    fn classify_for_sync_kernel_tests_vs_host_tests() {
        assert_eq!(
            classify_for_sync(Path::new("tests/spawn_smoke.rs")),
            SyncScope::Kernel,
        );
        assert_eq!(
            classify_for_sync(Path::new("tests/wiki_layering.rs")),
            SyncScope::Kernel,
        );
        assert_eq!(
            classify_for_sync(Path::new("tests/host_finance.rs")),
            SyncScope::Host,
        );
        assert_eq!(
            classify_for_sync(Path::new("tests/host_smoke.rs")),
            SyncScope::Host,
        );
    }

    #[test]
    fn classify_for_sync_rejects_absolute_paths_as_host() {
        // Absolute paths must never escalate to Kernel scope — they're
        // outside the project root sync operates on, so refusing to
        // touch them is the safe default.
        assert_eq!(classify_for_sync(Path::new("/etc/passwd")), SyncScope::Host,);
        assert_eq!(
            classify_for_sync(Path::new("/home/user/dark-matter/src/main.rs")),
            SyncScope::Host,
        );
    }

    #[test]
    fn sync_status_missing_canonical_repo_is_actionable() {
        let tmp = tempfile::tempdir().unwrap();
        write_identity(
            tmp.path(),
            &Identity {
                mode: Mode::Host,
                host_project: Some("finance-app".to_string()),
                canonical_dm_revision: Some("abc123".to_string()),
                canonical_dm_repo: None,
                source: None,
            },
        );

        let err = sync_message_for_project(
            tmp.path(),
            SyncArgs {
                status: true,
                ..SyncArgs::default()
            },
        )
        .unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("canonical_dm_repo not set"), "{msg}");
        assert!(msg.contains("Try:"), "{msg}");
    }
}
