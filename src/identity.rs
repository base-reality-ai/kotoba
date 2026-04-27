//! Spawn-paradigm identity for a running dm.
//!
//! Every dm process is in exactly one mode:
//!
//! - `kernel` — canonical dm at `base-reality-ai/dark-matter`. Wiki tracks dm.
//! - `host`   — dm spawned into a project. Wiki tracks the host project.
//!
//! Identity is captured in `<project_root>/.dm/identity.toml`. When the file
//! is absent we default to `kernel` — this preserves canonical-dm behavior in
//! the source repo and in any pre-spawn-paradigm environment.
//!
//! See `VISION.md` (repo root) for the full conceptual frame.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Filename, relative to `<project_root>/.dm/`.
pub const IDENTITY_FILENAME: &str = "identity.toml";

/// Two modes a running dm can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Kernel,
    Host,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Kernel => "kernel",
            Mode::Host => "host",
        }
    }
}

/// Parsed identity for the current dm process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub mode: Mode,
    /// Required when `mode == Host`. None in kernel mode.
    pub host_project: Option<String>,
    /// Canonical dm git sha at spawn time. Written by `dm spawn`, omitted in
    /// kernel mode.
    pub canonical_dm_revision: Option<String>,
    /// Canonical dm repository URL (or local path) the host was spawned
    /// from. Written by `dm spawn` so `dm sync` can re-fetch from the same
    /// source instead of reverting to the hard-coded default. Omitted in
    /// kernel mode and in legacy host identity files predating Tier 2.
    pub canonical_dm_repo: Option<String>,
    /// Where this identity came from. `None` means defaulted (no file present).
    pub source: Option<PathBuf>,
}

impl Identity {
    /// Default identity used when no `.dm/identity.toml` exists.
    pub fn default_kernel() -> Self {
        Self {
            mode: Mode::Kernel,
            host_project: None,
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        }
    }

    /// Display name for UI surfaces (TUI title, session prompts, chain seeds).
    /// Kernel mode renders as `"dark-matter"`; host mode as the host project name.
    pub fn display_name(&self) -> &str {
        match (self.mode, self.host_project.as_deref()) {
            (Mode::Host, Some(name)) => name,
            _ => "dark-matter",
        }
    }

    pub fn is_host(&self) -> bool {
        self.mode == Mode::Host
    }

    #[allow(dead_code)]
    pub fn is_kernel(&self) -> bool {
        self.mode == Mode::Kernel
    }
}

/// On-disk schema. Kept separate from `Identity` so the public type can
/// carry derived fields (`source`) without leaking them through TOML.
#[derive(Debug, Deserialize, Serialize)]
struct IdentityFile {
    mode: Mode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host_project: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    canonical_dm_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    canonical_dm_repo: Option<String>,
}

/// Load identity for `project_root`. Looks for `<project_root>/.dm/identity.toml`.
/// Returns the kernel default when the file is absent.
///
/// Errors only on malformed TOML or a host-mode file missing `host_project`.
/// Read errors other than `NotFound` are propagated so callers can surface
/// disk problems instead of silently masking them as kernel mode.
pub fn load_at(project_root: &Path) -> anyhow::Result<Identity> {
    let path = project_root.join(".dm").join(IDENTITY_FILENAME);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Identity::default_kernel());
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "failed to read {}: {}. Try: ensure the file is readable, or delete it to fall back to kernel mode.",
                path.display(),
                e
            ));
        }
    };
    let parsed: IdentityFile = toml::from_str(&text).map_err(|e| {
        anyhow::anyhow!(
            "malformed {}: {}. Try: check the file matches `mode = \"kernel\" | \"host\"` with optional `host_project` and `canonical_dm_revision`.",
            path.display(),
            e
        )
    })?;
    if parsed.mode == Mode::Host
        && parsed
            .host_project
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
    {
        return Err(anyhow::anyhow!(
            "{} has mode = \"host\" but no `host_project` (or only whitespace). Try: set `host_project = \"<name>\"` or change `mode = \"kernel\"`.",
            path.display()
        ));
    }
    // Trim incidental whitespace from a hand-edited host_project so downstream
    // consumers (TUI title, daemon socket path, system_prompt framing) don't
    // render leading/trailing spaces. `dm spawn` already passes through
    // `validate_project_name` which rejects whitespace outright; this guards
    // the hand-edited / external-script path.
    let host_project = parsed.host_project.map(|s| s.trim().to_string());
    Ok(Identity {
        mode: parsed.mode,
        host_project,
        canonical_dm_revision: parsed.canonical_dm_revision,
        canonical_dm_repo: parsed.canonical_dm_repo,
        source: Some(path),
    })
}

/// Convenience: load identity rooted at the current working directory.
/// Returns the kernel default on any I/O failure resolving cwd, mirroring
/// the lenient behavior of `wiki::ensure_for_cwd`.
pub fn load_for_cwd() -> Identity {
    let cwd = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return Identity::default_kernel(),
    };
    match load_at(&cwd) {
        Ok(id) => id,
        Err(e) => {
            crate::warnings::push_warning(format!("identity: {}", e));
            Identity::default_kernel()
        }
    }
}

/// Serialize an identity for writing to `.dm/identity.toml`. Used by
/// `dm spawn` (via the library) to seal a fresh host project's identity.
/// Binary-side `mod identity;` doesn't reach this — `dm spawn` is
/// dispatched through `dark_matter::spawn::run_spawn`, so the binary's
/// identity module sees this fn as dead. The lib target's spawn.rs and
/// the in-module roundtrip tests cover its real usage.
#[allow(dead_code)]
pub fn render_toml(identity: &Identity) -> anyhow::Result<String> {
    let file = IdentityFile {
        mode: identity.mode,
        host_project: identity.host_project.clone(),
        canonical_dm_revision: identity.canonical_dm_revision.clone(),
        canonical_dm_repo: identity.canonical_dm_repo.clone(),
    };
    Ok(toml::to_string_pretty(&file)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_identity(root: &Path, contents: &str) {
        let dm = root.join(".dm");
        fs::create_dir_all(&dm).unwrap();
        fs::write(dm.join(IDENTITY_FILENAME), contents).unwrap();
    }

    #[test]
    fn missing_file_defaults_to_kernel() {
        let tmp = TempDir::new().unwrap();
        let id = load_at(tmp.path()).unwrap();
        assert_eq!(id, Identity::default_kernel());
        assert!(id.is_kernel());
        assert!(!id.is_host());
        assert_eq!(id.display_name(), "dark-matter");
        assert!(id.source.is_none());
    }

    #[test]
    fn parses_kernel_file() {
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "mode = \"kernel\"\n");
        let id = load_at(tmp.path()).unwrap();
        assert_eq!(id.mode, Mode::Kernel);
        assert!(id.host_project.is_none());
        assert!(id.canonical_dm_revision.is_none());
        assert_eq!(id.display_name(), "dark-matter");
        assert!(id.source.is_some());
    }

    #[test]
    fn parses_host_file() {
        let tmp = TempDir::new().unwrap();
        write_identity(
            tmp.path(),
            "mode = \"host\"\nhost_project = \"finance-app\"\ncanonical_dm_revision = \"abc123\"\n",
        );
        let id = load_at(tmp.path()).unwrap();
        assert_eq!(id.mode, Mode::Host);
        assert_eq!(id.host_project.as_deref(), Some("finance-app"));
        assert_eq!(id.canonical_dm_revision.as_deref(), Some("abc123"));
        assert!(id.is_host());
        assert!(!id.is_kernel());
        assert_eq!(id.display_name(), "finance-app");
    }

    #[test]
    fn host_without_project_is_rejected() {
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "mode = \"host\"\n");
        let err = load_at(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("host_project"), "err = {err}");
        assert!(err.contains("Try:"), "err missing next-step hint: {err}");
    }

    #[test]
    fn host_with_empty_project_is_rejected() {
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "mode = \"host\"\nhost_project = \"\"\n");
        let err = load_at(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("host_project"), "err = {err}");
    }

    #[test]
    fn host_with_whitespace_only_project_is_rejected() {
        // Hand-edited identity.toml could have `host_project = "   "` —
        // `dm spawn` rejects such names via `validate_project_name`, but
        // the loader is the last line of defense for externally-written
        // files. Trim+empty check beats raw is_empty.
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "mode = \"host\"\nhost_project = \"   \"\n");
        let err = load_at(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("host_project"), "err = {err}");
        assert!(
            err.contains("whitespace"),
            "err should mention whitespace: {err}"
        );
    }

    #[test]
    fn host_project_is_trimmed_on_load() {
        // Incidental leading/trailing whitespace gets stripped so downstream
        // consumers (TUI title, daemon socket path, system_prompt) don't
        // render visible spaces.
        let tmp = TempDir::new().unwrap();
        write_identity(
            tmp.path(),
            "mode = \"host\"\nhost_project = \"  finance-app  \"\n",
        );
        let id = load_at(tmp.path()).unwrap();
        assert_eq!(id.host_project.as_deref(), Some("finance-app"));
        assert_eq!(id.display_name(), "finance-app");
    }

    #[test]
    fn malformed_toml_is_rejected_with_hint() {
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "this is not valid = toml = at all\n");
        let err = load_at(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("malformed"), "err = {err}");
        assert!(err.contains("Try:"), "err missing next-step hint: {err}");
    }

    #[test]
    fn unknown_mode_is_rejected() {
        let tmp = TempDir::new().unwrap();
        write_identity(tmp.path(), "mode = \"satellite\"\n");
        let err = load_at(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("malformed"), "err = {err}");
    }

    #[test]
    fn render_toml_kernel_roundtrip() {
        let id = Identity::default_kernel();
        let text = render_toml(&id).unwrap();
        assert!(text.contains("mode = \"kernel\""));
        assert!(!text.contains("host_project"));
        assert!(!text.contains("canonical_dm_revision"));
    }

    #[test]
    fn render_toml_host_roundtrip() {
        let id = Identity {
            mode: Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: Some("deadbeef".into()),
            canonical_dm_repo: Some("https://github.com/base-reality-ai/dark-matter.git".into()),
            source: None,
        };
        let text = render_toml(&id).unwrap();
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join(".dm")).unwrap();
        fs::write(tmp.path().join(".dm").join(IDENTITY_FILENAME), &text).unwrap();
        let parsed = load_at(tmp.path()).unwrap();
        assert_eq!(parsed.mode, Mode::Host);
        assert_eq!(parsed.host_project.as_deref(), Some("finance-app"));
        assert_eq!(parsed.canonical_dm_revision.as_deref(), Some("deadbeef"));
        assert_eq!(
            parsed.canonical_dm_repo.as_deref(),
            Some("https://github.com/base-reality-ai/dark-matter.git"),
        );
    }

    #[test]
    fn legacy_host_file_without_canonical_dm_repo_loads_with_none() {
        // Pre-Tier-2 host identities omit `canonical_dm_repo`. They must
        // still load cleanly so existing spawned projects don't break on a
        // dm upgrade — the field defaults to None until the next
        // `dm spawn` (or future migration) populates it.
        let tmp = TempDir::new().unwrap();
        write_identity(
            tmp.path(),
            "mode = \"host\"\nhost_project = \"finance-app\"\ncanonical_dm_revision = \"deadbeef\"\n",
        );
        let id = load_at(tmp.path()).unwrap();
        assert_eq!(id.mode, Mode::Host);
        assert_eq!(id.host_project.as_deref(), Some("finance-app"));
        assert_eq!(id.canonical_dm_revision.as_deref(), Some("deadbeef"));
        assert!(id.canonical_dm_repo.is_none());
    }

    #[test]
    fn render_toml_kernel_omits_canonical_dm_repo() {
        // Kernel-mode identities never carry a repo URL; the serializer
        // skip_serializing_if guard must keep that line out so the
        // canonical wiki's identity stays minimal.
        let id = Identity::default_kernel();
        let text = render_toml(&id).unwrap();
        assert!(!text.contains("canonical_dm_repo"));
    }

    #[test]
    fn mode_as_str_matches_serde() {
        assert_eq!(Mode::Kernel.as_str(), "kernel");
        assert_eq!(Mode::Host.as_str(), "host");
    }
}
