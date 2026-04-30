//! Implements the `dm spawn` command for bootstrapping host projects.
//!
//! This module handles copying the canonical kernel, configuring the host
//! identity, and generating a starter domain project.

use crate::identity::{load_at, render_toml, Identity, Mode};
use anyhow::Context;
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub async fn run_spawn(project_name: &str, canonical_arg: Option<&str>) -> anyhow::Result<()> {
    validate_project_name(project_name)?;

    let cwd = std::env::current_dir()
        .context("Failed to read current directory. Try: run `dm spawn` from an accessible project parent directory.")?;
    let target_dir = cwd.join(project_name);

    if target_dir.exists() {
        anyhow::bail!(
            "Directory '{}' already exists. Cannot spawn dm into an existing directory.",
            project_name
        );
    }

    println!(
        "Spawning dark-matter kernel into {}...",
        target_dir.display()
    );

    // 1. Copy/clone canonical dm sources
    let canonical_repo = canonical_arg
        .map(|s| s.to_string())
        .or_else(|| std::env::var("DM_CANONICAL_REPO").ok())
        .unwrap_or_else(|| "https://github.com/base-reality-ai/dark-matter.git".to_string());
    refuse_host_mode_canonical_source(&canonical_repo, &cwd)?;

    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            &canonical_repo,
            target_dir
                .to_str()
                .context("Failed to convert spawn target path to UTF-8. Try: choose a parent directory with a UTF-8 path.")?,
        ])
        .status()
        .await
        .with_context(|| {
            format!(
                "Failed to clone canonical dm sources from {}. Try: install git and verify --canonical or DM_CANONICAL_REPO is reachable.",
                canonical_repo,
            )
        })?;

    if !status.success() {
        anyhow::bail!(
            "Failed to clone dark-matter into {}. Try: use --canonical <path-or-url> or set DM_CANONICAL_REPO to a reachable source.",
            target_dir.display()
        );
    }

    // Capture the cloned source's HEAD before we delete its .git. Running
    // rev-parse here (instead of in the caller's cwd) is what makes
    // `canonical_dm_revision` actually identify the snapshot we just
    // unpacked — the caller's cwd may be unrelated, ungit, or out of sync
    // with the canonical remote we cloned from.
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&target_dir)
        .output()
        .await
        .with_context(|| {
            format!(
                "Failed to execute 'git rev-parse' in {} to record the canonical dm revision. Try: ensure git is installed and the cloned source contains a valid .git directory before cleanup.",
                target_dir.display(),
            )
        })?;
    let canonical_dm_revision = if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    };

    // Clean up copied .git, target, .dm
    cleanup_spawned_kernel(&target_dir).await?;

    // 2. Write .dm/identity.toml
    let dm_dir = target_dir.join(".dm");
    tokio::fs::create_dir_all(&dm_dir).await.with_context(|| {
        format!(
            "Failed to create {}. Try: check write permissions on the spawn target directory.",
            dm_dir.display(),
        )
    })?;

    let identity = Identity {
        mode: Mode::Host,
        host_project: Some(project_name.to_string()),
        canonical_dm_revision,
        canonical_dm_repo: Some(canonical_repo.clone()),
        source: None,
    };
    let identity_toml = render_toml(&identity)
        .context("Failed to render host identity.toml. Try: report this dm bug; the generated host identity should always serialize.")?;
    let identity_path = dm_dir.join("identity.toml");
    tokio::fs::write(&identity_path, identity_toml)
        .await
        .with_context(|| {
            format!(
                "Failed to write {}. Try: check write permissions on the spawned project's .dm/ directory.",
                identity_path.display(),
            )
        })?;

    // 3. Create starter DM.md
    let dm_md_content = format!(
        "\
# {}

This is a host project built on the dark-matter kernel.
",
        project_name
    );
    let dm_md_path = target_dir.join("DM.md");
    tokio::fs::write(&dm_md_path, dm_md_content)
        .await
        .with_context(|| {
            format!(
                "Failed to write {}. Try: check write permissions on the spawned project directory.",
                dm_md_path.display(),
            )
        })?;

    // 4. Starter src/host_main.rs and Cargo.toml modifications
    let cargo_toml_path = target_dir.join("Cargo.toml");
    let cargo_toml = tokio::fs::read_to_string(&cargo_toml_path)
        .await
        .with_context(|| {
            format!(
                "Failed to read {}. Try: confirm the canonical dm source has a Cargo.toml.",
                cargo_toml_path.display(),
            )
        })?;
    let cargo_toml = rewrite_host_package_section(&cargo_toml, project_name);

    let bin_entry = format!(
        "\n[[bin]]\nname = \"{}\"\npath = \"src/host_main.rs\"\n",
        project_name
    );
    tokio::fs::write(&cargo_toml_path, format!("{}{}", cargo_toml, bin_entry))
        .await
        .with_context(|| {
            format!(
                "Failed to update {}. Try: check write permissions on the spawned project directory.",
                cargo_toml_path.display(),
            )
        })?;

    // Sweep `examples/*/Cargo.toml` so any example that depends on canonical dm
    // by package name (`dark-matter = { path = "../.." }`) instead points at
    // the spawned host's package name. The lib crate keeps `name = "dark_matter"`,
    // so example sources that `use dark_matter::*` still resolve. Without this,
    // every spawned project inherits a broken example until the operator hand-edits.
    rewrite_example_cargo_deps(&target_dir, project_name).await?;

    let host_main_content = format!(
        "\
use dark_matter::config::Config;
use dark_matter::wiki::{{IngestOutcome, Wiki}};
use std::env;
use std::path::Path;

mod domain;

#[tokio::main]
async fn main() -> anyhow::Result<()> {{
    // 1. Host Project domain logic.
    // Parse arguments and yield control to the dark-matter kernel if requested.
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && args[1] == \"dm\" {{
        println!(\"Delegating to dark-matter kernel...\");
        let config = Config::load()?;
        println!(\"Using Ollama model: {{}}\", config.model);
        
        // Example: To launch the TUI, you would uncomment the line below.
        // dark_matter::tui::run::run_tui(&config, None).await?;
        println!(\"TUI launch disabled in generated skeleton to prevent taking over terminal.\");
        return Ok(());
    }}

    let transactions = domain::sample_transactions();
    let balance = domain::projected_balance_cents(&transactions);
    let wiki_result = ingest_domain_module(Path::new(env!(\"CARGO_MANIFEST_DIR\")))?;

    // Host project's primary execution path
    println!(\"Hello from {project_name} host!\");
    println!(\"Projected balance: ${{:.2}}\", balance as f64 / 100.0);
    println!(\"Wiki tracked host file: {{}}\", wiki_result);
    println!(\"Run with `cargo run --bin {project_name} -- dm` to see the kernel delegation pattern.\");

    Ok(())
}}

fn ingest_domain_module(project_root: &Path) -> anyhow::Result<String> {{
    let project_root = project_root.canonicalize()?;
    let source_path = project_root.join(\"src/domain.rs\").canonicalize()?;
    let content = std::fs::read_to_string(&source_path)?;
    let wiki = Wiki::open(&project_root)?;

    let message = match wiki.ingest_file(&project_root, &source_path, &content)? {{
        IngestOutcome::Ingested {{ page_rel }} => page_rel,
        IngestOutcome::Skipped(reason) => format!(\"skipped: {{:?}}\", reason),
    }};

    Ok(message)
}}
"
    );
    let host_main_path = target_dir.join("src").join("host_main.rs");
    tokio::fs::write(&host_main_path, host_main_content)
        .await
        .with_context(|| {
            format!(
                "Failed to write {}. Try: check write permissions on the spawned project's src directory.",
                host_main_path.display(),
            )
        })?;

    let domain_content = "\
//! Tiny host-domain module for the embedding example.
//!
//! In a real spawned project this file would be the app's own domain logic.
//! The skeleton ingests this file into `.dm/wiki/` to show dm tracking host
//! code rather than only its kernel internals.

#[derive(Debug, Clone, Copy)]
pub struct Transaction {
    pub amount_cents: i64,
}

pub fn sample_transactions() -> Vec<Transaction> {
    vec![
        Transaction {
            amount_cents: 12_500,
        },
        Transaction {
            amount_cents: -4_000,
        },
        Transaction {
            amount_cents: 1_550,
        },
    ]
}

pub fn projected_balance_cents(transactions: &[Transaction]) -> i64 {
    transactions.iter().map(|tx| tx.amount_cents).sum()
}
";
    let domain_path = target_dir.join("src").join("domain.rs");
    tokio::fs::write(&domain_path, domain_content)
        .await
        .with_context(|| {
            format!(
                "Failed to write {}. Try: check write permissions on the spawned project's src directory.",
                domain_path.display(),
            )
        })?;

    // 5. Initialize .dm/wiki/ via the canonical layout: four category subdirs
    // (`entities/`, `concepts/`, `summaries/`, `synthesis/`) plus `index.md`
    // and `schema.md`. Going through `Wiki::open` keeps spawn's seed in
    // lockstep with what kernel-mode `Wiki::open` writes elsewhere — the
    // host project's `.dm/wiki/` is layer-ready from day zero, and the
    // schema doc that lands references the optional `layer:` frontmatter
    // field.
    crate::wiki::Wiki::open(&target_dir).with_context(|| {
        format!(
            "Failed to initialize wiki layout under {}. Try: check write permissions on the spawned project's .dm directory.",
            target_dir.display(),
        )
    })?;

    // 6. Next-step instructions
    print!("{}", spawn_success_message(project_name));

    Ok(())
}

fn spawn_success_message(project_name: &str) -> String {
    format!(
        "\nSuccessfully spawned {project_name}!\n\
Next steps:\n  cd {project_name}\n  cargo run --bin {project_name}   # Run your host application\n  cargo run --bin dm     # Run the dark-matter TUI/agent\n\
\n\
Host state:\n  .dm/identity.toml      # mode = \"host\", host_project = \"{project_name}\"\n  .dm/wiki/              # host project wiki\n"
    )
}

/// Rewrite the `[package]` table of canonical dm's `Cargo.toml` for a host
/// project. Strips fields that describe canonical dm (`description`,
/// `repository`, `homepage`, `documentation`, `keywords`, `categories`,
/// `license`, `license-file`, `readme`, `authors`) and resets `name`/`version`
/// to host defaults. Lines outside the `[package]` table — `[lib]`, `[[bin]]`,
/// `[dependencies]`, profile sections — are preserved verbatim, so the lib
/// crate keeps its `dark_matter` identity for `use dark_matter::…` imports.
fn rewrite_host_package_section(cargo_toml: &str, project_name: &str) -> String {
    const STRIP_KEYS: &[&str] = &[
        "description",
        "repository",
        "homepage",
        "documentation",
        "keywords",
        "categories",
        "license",
        "license-file",
        "readme",
        "authors",
    ];
    let mut out = String::with_capacity(cargo_toml.len());
    let mut in_package = false;
    let mut name_emitted = false;
    let mut version_emitted = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim_start();
        // Section header: a single-bracket [name]. Double-bracket [[name]]
        // (e.g. `[[bin]]`) are array-of-tables and never our `[package]` table.
        if trimmed.starts_with('[') && !trimmed.starts_with("[[") {
            in_package = trimmed.starts_with("[package]");
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if in_package {
            let key = trimmed.split('=').next().unwrap_or("").trim();
            if STRIP_KEYS.contains(&key) {
                continue;
            }
            if key == "name" {
                if !name_emitted {
                    out.push_str(&format!("name = \"{}\"\n", project_name));
                    name_emitted = true;
                }
                continue;
            }
            if key == "version" {
                if !version_emitted {
                    out.push_str("version = \"0.1.0\"\n");
                    version_emitted = true;
                }
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Rewrite path-style `dark-matter = { path = ... }` dependency lines in any
/// `examples/*/Cargo.toml` to the host project's package name. Idempotent:
/// once rewritten, the substring no longer matches. Missing `examples/` dir
/// is a no-op (canonical dm without examples still spawns cleanly).
async fn rewrite_example_cargo_deps(spawn_root: &Path, host_project: &str) -> anyhow::Result<()> {
    let examples_dir = spawn_root.join("examples");
    if !examples_dir.exists() {
        return Ok(());
    }
    let mut entries = tokio::fs::read_dir(&examples_dir).await.with_context(|| {
        format!(
            "Failed to read {}. Try: check read permissions on the spawned project's examples directory.",
            examples_dir.display(),
        )
    })?;
    while let Some(entry) = entries.next_entry().await.with_context(|| {
        format!(
            "Failed to enumerate entries under {}. Try: re-run after resolving filesystem errors on the spawned project's examples directory.",
            examples_dir.display(),
        )
    })? {
        let cargo_path = entry.path().join("Cargo.toml");
        if !cargo_path.is_file() {
            continue;
        }
        let body = tokio::fs::read_to_string(&cargo_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to read {}. Try: check permissions on the example's Cargo.toml.",
                    cargo_path.display(),
                )
            })?;
        let rewritten = body.replace(
            "dark-matter = { path",
            &format!("{} = {{ path", host_project),
        );
        if rewritten != body {
            tokio::fs::write(&cargo_path, rewritten)
                .await
                .with_context(|| {
                    format!(
                        "Failed to write {}. Try: check write permissions on the example's Cargo.toml.",
                        cargo_path.display(),
                    )
                })?;
        }
    }
    Ok(())
}

fn refuse_host_mode_canonical_source(canonical_repo: &str, cwd: &Path) -> anyhow::Result<()> {
    let Some(source_root) = canonical_repo_local_path(canonical_repo, cwd) else {
        return Ok(());
    };
    let identity = load_at(&source_root).with_context(|| {
        format!(
            "Failed to read canonical source identity at {}. Try: fix or remove its .dm/identity.toml before using it as --canonical.",
            source_root.display(),
        )
    })?;
    if identity.mode == Mode::Host {
        anyhow::bail!(
            "refusing to spawn from a host project's source tree (mode = host). Try: --canonical https://github.com/base-reality-ai/Dark-Matter"
        );
    }
    Ok(())
}

fn canonical_repo_local_path(canonical_repo: &str, cwd: &Path) -> Option<PathBuf> {
    if let Some(rest) = canonical_repo.strip_prefix("file://") {
        if rest.is_empty() {
            return None;
        }
        return Some(PathBuf::from(rest));
    }
    if canonical_repo.contains("://") || canonical_repo.contains('@') {
        return None;
    }
    let path = PathBuf::from(canonical_repo);
    let local_path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    if local_path.exists() {
        Some(local_path)
    } else {
        None
    }
}

pub(crate) async fn cleanup_spawned_kernel(target_dir: &Path) -> anyhow::Result<()> {
    // These removals are strictly required: failing to remove .dm would cause the spawned project
    // to inherit the kernel's identity and wiki; failing to remove .git or target would bloat the
    // new workspace or confuse version control. Therefore, non-NotFound failures are hard errors.
    for dir in &[".git", "target", ".dm", ".dm-workspace"] {
        let path = target_dir.join(dir);
        if let Err(e) = tokio::fs::remove_dir_all(&path).await {
            if e.kind() != std::io::ErrorKind::NotFound {
                anyhow::bail!(
                    "Failed to clean up {}; the spawned project may be corrupted. Try: check permissions or remove it manually. Error: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    Ok(())
}

pub fn validate_project_name(project_name: &str) -> anyhow::Result<()> {
    if project_name.is_empty() {
        anyhow::bail!(
            "Invalid project name: name must not be empty. Try: use 1-64 ASCII letters, numbers, '-' or '_'."
        );
    }
    if project_name.len() > 64 {
        anyhow::bail!(
            "Invalid project name '{}': name is longer than 64 characters. Try: use a shorter name with ASCII letters, numbers, '-' or '_'.",
            project_name
        );
    }
    if !project_name
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        anyhow::bail!(
            "Invalid project name '{}': only ASCII letters, numbers, '-' and '_' are allowed. Try: use a simple name like finance-app.",
            project_name
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_repo_local_path, refuse_host_mode_canonical_source, rewrite_host_package_section,
        spawn_success_message, validate_project_name,
    };

    #[test]
    fn validate_project_name_accepts_safe_names() {
        for name in ["finance-app", "finance_app", "FinanceApp42"] {
            validate_project_name(name).expect(name);
        }
    }

    #[test]
    fn validate_project_name_rejects_path_or_empty_names_with_hint() {
        for name in ["", "../finance", "finance/app", "finance app", ".hidden"] {
            let err = validate_project_name(name).unwrap_err().to_string();
            assert!(err.contains("Invalid project name"), "err = {err}");
            assert!(err.contains("Try:"), "err missing next-step hint: {err}");
        }
    }

    #[test]
    fn validate_project_name_rejects_overlong_name() {
        let name = "a".repeat(65);
        let err = validate_project_name(&name).unwrap_err().to_string();
        assert!(err.contains("longer than 64"), "err = {err}");
        assert!(err.contains("Try:"), "err missing next-step hint: {err}");
    }

    // Mirrors the canonical dark-matter `[package]` block at the top of
    // `Cargo.toml` so the rewriter's behavior is locked against the real
    // shape we're rewriting in `dm spawn`.
    const CANONICAL_PACKAGE: &str = "\
[package]
name = \"dark-matter\"
version = \"4.0.0\"
edition = \"2021\"
description = \"Dark Matter — local AI coding agent powered by Ollama\"
license = \"MIT\"
repository = \"https://github.com/base-reality-ai/dark-matter\"
homepage = \"https://github.com/base-reality-ai/dark-matter\"
readme = \"README.md\"
keywords = [\"llm\", \"ollama\", \"cli\", \"agent\", \"code\"]
categories = [\"command-line-utilities\", \"development-tools\"]
publish = false

[lib]
name = \"dark_matter\"
path = \"src/lib.rs\"

[[bin]]
name = \"dm\"
path = \"src/main.rs\"

[dependencies]
tokio = { version = \"1\" }
";

    #[test]
    fn rewrite_strips_canonical_metadata_and_resets_name_version() {
        let out = rewrite_host_package_section(CANONICAL_PACKAGE, "finance-app");
        assert!(out.contains("name = \"finance-app\""));
        assert!(out.contains("version = \"0.1.0\""));
        // Canonical metadata fields must be stripped.
        for stripped in [
            "Dark Matter — local AI coding",
            "base-reality-ai",
            "license = \"MIT\"",
            "keywords = [",
            "categories = [",
            "readme = \"README.md\"",
        ] {
            assert!(
                !out.contains(stripped),
                "host Cargo.toml still leaks {stripped:?} from canonical dm:\n{out}",
            );
        }
        // Original kernel name must not survive.
        assert!(!out.contains("name = \"dark-matter\""));
    }

    #[test]
    fn rewrite_preserves_lib_bin_and_dependencies_verbatim() {
        let out = rewrite_host_package_section(CANONICAL_PACKAGE, "finance-app");
        // Lib block: name must stay `dark_matter` so host code can `use dark_matter::…`.
        assert!(out.contains("[lib]\nname = \"dark_matter\""));
        // Existing kernel binary must remain so `cargo run --bin dm` keeps working.
        assert!(out.contains("[[bin]]\nname = \"dm\""));
        // Dependencies block must pass through untouched.
        assert!(out.contains("[dependencies]"));
        assert!(out.contains("tokio = { version = \"1\" }"));
        // publish = false belongs to the package table; should pass through.
        assert!(out.contains("publish = false"));
    }

    #[test]
    fn rewrite_idempotent_on_already_host_package() {
        let host_first = rewrite_host_package_section(CANONICAL_PACKAGE, "finance-app");
        let host_second = rewrite_host_package_section(&host_first, "finance-app");
        assert_eq!(host_first, host_second);
    }

    #[test]
    fn rewrite_only_touches_package_table() {
        // A `repository = ` line inside `[dependencies.foo]` must not be
        // stripped — only the `[package]` table's repository field should go.
        let input = "\
[package]
name = \"dark-matter\"
version = \"4.0.0\"
repository = \"https://github.com/base-reality-ai/dark-matter\"

[dependencies.foo]
git = \"https://example.com/foo.git\"
";
        let out = rewrite_host_package_section(input, "host");
        assert!(!out.contains("repository = \"https://github.com/base-reality-ai/dark-matter\""));
        assert!(out.contains("git = \"https://example.com/foo.git\""));
    }

    #[test]
    fn spawn_success_message_points_at_host_identity_and_wiki() {
        let out = spawn_success_message("finance-app");
        assert!(out.contains("\n  cd finance-app"), "missing cd hint: {out}");
        assert!(
            out.contains("\n  cargo run --bin finance-app"),
            "missing host binary hint: {out}"
        );
        assert!(
            out.contains("\n  .dm/identity.toml") && out.contains("host_project = \"finance-app\""),
            "missing host identity hint: {out}"
        );
        assert!(
            out.contains("\n  .dm/wiki/"),
            "missing host wiki hint: {out}"
        );
    }

    #[test]
    fn canonical_repo_local_path_accepts_file_url_absolute_and_relative_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();
        let local = cwd.join("canonical");
        std::fs::create_dir_all(&local).unwrap();

        assert_eq!(
            canonical_repo_local_path(&format!("file://{}", local.display()), cwd).as_deref(),
            Some(local.as_path()),
        );
        assert_eq!(
            canonical_repo_local_path(local.to_str().unwrap(), cwd).as_deref(),
            Some(local.as_path()),
        );
        assert_eq!(
            canonical_repo_local_path("canonical", cwd).as_deref(),
            Some(local.as_path()),
        );
    }

    #[test]
    fn canonical_repo_local_path_skips_remote_urls_and_scp_syntax() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd = tmp.path();
        assert!(canonical_repo_local_path(
            "https://github.com/base-reality-ai/dark-matter.git",
            cwd,
        )
        .is_none());
        assert!(canonical_repo_local_path("git@github.com:org/repo.git", cwd).is_none());
        assert!(canonical_repo_local_path("missing-local-path", cwd).is_none());
    }

    #[test]
    fn refuse_host_mode_canonical_source_rejects_local_host_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().join("host-source");
        let dm_dir = canonical.join(".dm");
        std::fs::create_dir_all(&dm_dir).unwrap();
        std::fs::write(
            dm_dir.join(crate::identity::IDENTITY_FILENAME),
            "mode = \"host\"\nhost_project = \"kotoba\"\n",
        )
        .unwrap();

        let err = refuse_host_mode_canonical_source(
            &format!("file://{}", canonical.display()),
            tmp.path(),
        )
        .unwrap_err()
        .to_string();

        assert!(
            err.contains("refusing to spawn from a host project's source tree"),
            "wrong refusal: {err}",
        );
        assert!(
            err.contains("--canonical https://github.com/base-reality-ai/Dark-Matter"),
            "missing canonical retry hint: {err}",
        );
    }

    #[test]
    fn refuse_host_mode_canonical_source_allows_kernel_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let canonical = tmp.path().join("kernel-source");
        let dm_dir = canonical.join(".dm");
        std::fs::create_dir_all(&dm_dir).unwrap();
        std::fs::write(
            dm_dir.join(crate::identity::IDENTITY_FILENAME),
            "mode = \"kernel\"\n",
        )
        .unwrap();

        refuse_host_mode_canonical_source(canonical.to_str().unwrap(), tmp.path())
            .expect("kernel-mode canonical source should be allowed");
    }

    #[tokio::test]
    async fn rewrite_example_cargo_deps_rewrites_canonical_path_dep() {
        let tmp = tempfile::tempdir().unwrap();
        let example_dir = tmp.path().join("examples").join("host-skeleton");
        tokio::fs::create_dir_all(&example_dir).await.unwrap();
        let cargo = example_dir.join("Cargo.toml");
        let body = "\
[package]
name = \"host-skeleton\"
version = \"0.1.0\"
edition = \"2021\"

[dependencies]
tokio = { version = \"1\", features = [\"full\"] }
dark-matter = { path = \"../..\" }
serde_json = \"1\"
";
        tokio::fs::write(&cargo, body).await.unwrap();

        super::rewrite_example_cargo_deps(tmp.path(), "finance-app")
            .await
            .unwrap();

        let after = tokio::fs::read_to_string(&cargo).await.unwrap();
        assert!(
            after.contains("finance-app = { path = \"../..\" }"),
            "rewrite missed the dark-matter path dep:\n{after}",
        );
        assert!(
            !after.contains("dark-matter = { path"),
            "stale dark-matter path dep survived rewrite:\n{after}",
        );
        // Unrelated lines must pass through verbatim.
        assert!(after.contains("tokio = { version = \"1\""));
        assert!(after.contains("serde_json = \"1\""));
        assert!(after.contains("name = \"host-skeleton\""));
    }

    #[tokio::test]
    async fn rewrite_example_cargo_deps_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let example_dir = tmp.path().join("examples").join("host-skeleton");
        tokio::fs::create_dir_all(&example_dir).await.unwrap();
        let cargo = example_dir.join("Cargo.toml");
        let body = "\
[dependencies]
dark-matter = { path = \"../..\" }
";
        tokio::fs::write(&cargo, body).await.unwrap();

        super::rewrite_example_cargo_deps(tmp.path(), "finance-app")
            .await
            .unwrap();
        let after_first = tokio::fs::read_to_string(&cargo).await.unwrap();
        super::rewrite_example_cargo_deps(tmp.path(), "finance-app")
            .await
            .unwrap();
        let after_second = tokio::fs::read_to_string(&cargo).await.unwrap();

        assert_eq!(after_first, after_second);
    }

    #[tokio::test]
    async fn rewrite_example_cargo_deps_missing_examples_dir_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        // No examples/ dir under tmp.
        super::rewrite_example_cargo_deps(tmp.path(), "finance-app")
            .await
            .expect("missing examples/ should be a clean no-op");
    }

    #[tokio::test]
    async fn rewrite_example_cargo_deps_leaves_unrelated_cargo_untouched() {
        let tmp = tempfile::tempdir().unwrap();
        let example_dir = tmp.path().join("examples").join("plain");
        tokio::fs::create_dir_all(&example_dir).await.unwrap();
        let cargo = example_dir.join("Cargo.toml");
        // No `dark-matter = { path ...` line — must round-trip byte-identically.
        let body = "\
[package]
name = \"plain\"

[dependencies]
serde = \"1\"
";
        tokio::fs::write(&cargo, body).await.unwrap();
        super::rewrite_example_cargo_deps(tmp.path(), "finance-app")
            .await
            .unwrap();
        let after = tokio::fs::read_to_string(&cargo).await.unwrap();
        assert_eq!(after, body, "unrelated example Cargo.toml must not change");
    }

    #[tokio::test]
    async fn cleanup_spawned_kernel_silently_ignores_missing_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let result = super::cleanup_spawned_kernel(tmp.path()).await;
        assert!(result.is_ok(), "cleanup should ignore NotFound errors");
    }

    #[tokio::test]
    async fn cleanup_spawned_kernel_removes_existing_dirs() {
        let tmp = tempfile::tempdir().unwrap();

        let dm_dir = tmp.path().join(".dm");
        tokio::fs::create_dir_all(&dm_dir).await.unwrap();

        let result = super::cleanup_spawned_kernel(tmp.path()).await;
        assert!(result.is_ok(), "cleanup should succeed");
        assert!(!dm_dir.exists(), ".dm dir should be removed");
    }
}
