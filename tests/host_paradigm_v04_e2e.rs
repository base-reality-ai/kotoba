//! Tier-4 end-to-end acceptance test for the kotoba v0.3 paradigm-gap
//! triple-fix (canonical-dm v0.4 acceptance test).
//!
//! Mirrors `tests/host_mode_e2e.rs` (Run-31 acceptance) but exercises:
//!
//! - Tier 1 — `Wiki::register_host_page` makes a host-layer page
//!   visible to `Wiki::search` and `idx.entries`.
//! - Tier 2 — `orchestrate::save_last_chain_pointer` lands the pointer
//!   under `<project>/.dm/last_chain.json` in host mode (not `~/.dm/`).
//! - Tier 3 — `host::invoke_tool` dispatches an installed host
//!   capability deterministically (the daemon `host.invoke` RPC arm
//!   is a thin wrapper around this primitive).
//!
//! Final assert: the isolated `~/.dm/` is byte-identical to its
//! pre-test snapshot — Run-31's no-leakage invariant must hold.
//!
//! Single-binary because `install_host_capabilities` writes to a
//! process-global `OnceLock` and `current_dir` / `HOME` are
//! process-global state.

use async_trait::async_trait;
use dark_matter::host::{install_host_capabilities, invoke_tool, HostCapabilities};
use dark_matter::tools::{Tool, ToolResult};
use dark_matter::wiki::{Layer, PageType, Wiki, WikiPage};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use tempfile::TempDir;

struct E2eHost;

impl HostCapabilities for E2eHost {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostE2eTool)]
    }
}

struct HostE2eTool;

#[async_trait]
impl Tool for HostE2eTool {
    fn name(&self) -> &'static str {
        "host_paradigm_e2e"
    }
    fn description(&self) -> &'static str {
        "End-to-end paradigm-gap probe."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "msg": { "type": "string" } }
        })
    }
    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let msg = args.get("msg").and_then(Value::as_str).unwrap_or("");
        Ok(ToolResult {
            content: format!("e2e:{msg}"),
            is_error: false,
        })
    }
}

fn write_host_identity(root: &Path) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"v04-e2e\"\n",
    )
    .expect("write identity");
}

fn snapshot_tree(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let rel = p
                .strip_prefix(root)
                .expect("strip prefix")
                .to_string_lossy()
                .to_string();
            if p.is_dir() {
                out.insert(format!("{rel}/"), Vec::new());
                walk(root, &p, out);
            } else {
                out.insert(rel, std::fs::read(&p).expect("read snapshot file"));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn host_paradigm_v04_three_tier_fix_lands_no_leakage() {
    let home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path());

    // Pre-populate ~/.dm with a sentinel so we can byte-compare the
    // snapshot after the test exercises every routing path.
    let global_dm = home.path().join(".dm");
    std::fs::create_dir_all(&global_dm).expect("create global .dm");
    std::fs::write(global_dm.join("sentinel.txt"), "untouched").expect("write sentinel");
    let global_before = snapshot_tree(&global_dm);

    let prior_home = std::env::var_os("HOME");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_var("HOME", home.path());
    std::env::set_current_dir(project.path()).expect("chdir project");

    // ── Tier 1 — register_host_page ──────────────────────────────────
    let wiki = Wiki::open(project.path()).expect("open wiki");
    let host_page = WikiPage {
        title: "猫".to_string(),
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated: "2026-04-28 12:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "Kana: ねこ\nMeaning: cat\n".to_string(),
    };
    wiki.register_host_page(
        "entities/Vocabulary/neko.md",
        &host_page,
        "vocab: neko (cat)",
    )
    .expect("register_host_page");
    let hits = wiki.search("cat").expect("search");
    assert_eq!(hits.len(), 1, "host page must be findable: {hits:?}");
    assert_eq!(hits[0].path, "entities/Vocabulary/neko.md");

    // ── Tier 2 — chain pointer routes project-local ─────────────────
    let workspace = project.path().join("chain-workspace");
    std::fs::create_dir_all(&workspace).expect("create workspace");
    dark_matter::orchestrate::save_last_chain_pointer(&workspace);
    let project_pointer = project.path().join(".dm").join("last_chain.json");
    let leaked_pointer = home.path().join(".dm").join("last_chain.json");
    assert!(
        project_pointer.is_file(),
        "Tier 2: chain pointer must land at {}",
        project_pointer.display()
    );
    assert!(
        !leaked_pointer.exists(),
        "Tier 2: chain pointer must NOT leak to {}",
        leaked_pointer.display()
    );

    // ── Tier 3 — host.invoke dispatch primitive ─────────────────────
    install_host_capabilities(Box::new(E2eHost)).expect("install host caps");
    let result = invoke_tool("host_paradigm_e2e", json!({ "msg": "wired" }))
        .await
        .expect("invoke_tool");
    assert!(!result.is_error, "tool must succeed: {}", result.content);
    assert_eq!(result.content, "e2e:wired");

    // Restore process state before snapshot comparison so a panic in the
    // tail of the test still leaves env clean.
    if let Some(prev) = prior_home {
        std::env::set_var("HOME", prev);
    } else {
        std::env::remove_var("HOME");
    }
    if let Some(prev) = prior_cwd {
        let _ = std::env::set_current_dir(prev);
    }

    // ── Acceptance — `~/.dm/` is byte-identical to pre-test state ──
    let global_after = snapshot_tree(&global_dm);
    assert_eq!(
        global_after, global_before,
        "host-mode artifacts must NOT mutate the operator-global ~/.dm/"
    );

    // ── Belt-and-suspenders — every project-local artifact is present
    let project_dm = project.path().join(".dm");
    assert!(project_dm.join("wiki/index.md").is_file());
    assert!(
        project_dm
            .join("wiki/entities/Vocabulary/neko.md")
            .is_file(),
        "host-layer wiki page must be on disk"
    );
    assert!(project_dm.join("last_chain.json").is_file());
}
