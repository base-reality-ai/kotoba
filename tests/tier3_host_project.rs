//! Integration test: real host-project validation (Tier 3 cycle 1).
//!
//! Exercises the spawn paradigm against itself with a coherent
//! "task-tracker" host project. Tier 1 (wiki layering), Run-26
//! (identity), and Run-27 (capability extension) must work together
//! when a single host project carries all three. The synthetic
//! coverage in `tests/host_capabilities.rs` (capability install) and
//! `tests/wiki_layering.rs` (search stratification) each verifies one
//! axis in isolation; this test verifies the integration:
//!
//! - A host project's `.dm/identity.toml` declares `mode = "host"`.
//! - That project installs `HostCapabilities` exposing `host_add_task`
//!   and `host_list_tasks` — concrete domain tools, not probes.
//! - That project's `.dm/wiki/concepts/task-data-model.md` carries
//!   `layer: host` so it lives alongside (and is filtered against)
//!   inherited kernel-layer pages.
//! - `wiki_search` in host mode surfaces the host concept with the
//!   `[host]` badge ahead of any kernel-layer match for the same
//!   keyword.
//! - The default tool registry merges the host tools so the host
//!   project's TUI / chain / sub-agents all see them.
//!
//! Lives in its own integration-test binary for the same reason
//! `tests/host_capabilities.rs` does: `install_host_capabilities`
//! writes to a process-global `OnceLock` and must not collide with
//! other test binaries' installations.

use async_trait::async_trait;
use dark_matter::host::{install_host_capabilities, HostCapabilities};
use dark_matter::identity::{Identity, Mode};
use dark_matter::tools::registry::default_registry;
use dark_matter::tools::{Tool, ToolResult};
use dark_matter::wiki::{IndexEntry, Layer, PageType, Wiki, WikiPage};
use serde_json::{json, Value};
use std::sync::Mutex;

/// Process-shared store the tracker tools mutate. Real host projects
/// would back this with a SQLite handle, a JSON file, etc.; for the
/// test, an in-memory Mutex keeps the harness self-contained.
static TASKS: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct TaskTracker;

impl HostCapabilities for TaskTracker {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostAddTaskTool), Box::new(HostListTasksTool)]
    }
}

struct HostAddTaskTool;

#[async_trait]
impl Tool for HostAddTaskTool {
    fn name(&self) -> &'static str {
        "host_add_task"
    }
    fn description(&self) -> &'static str {
        "Append a task to the tracker. The host_ prefix marks it as host-domain."
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "title": { "type": "string" } },
            "required": ["title"]
        })
    }
    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let title = args
            .get("title")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("host_add_task: missing `title`"))?;
        TASKS.lock().unwrap().push(title.to_string());
        Ok(ToolResult {
            content: format!("added: {title}"),
            is_error: false,
        })
    }
}

struct HostListTasksTool;

#[async_trait]
impl Tool for HostListTasksTool {
    fn name(&self) -> &'static str {
        "host_list_tasks"
    }
    fn description(&self) -> &'static str {
        "List every task tracked by the host project."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    async fn call(&self, _args: Value) -> anyhow::Result<ToolResult> {
        let tasks = TASKS.lock().unwrap();
        Ok(ToolResult {
            content: tasks.join("\n"),
            is_error: false,
        })
    }
}

fn write_host_identity(root: &std::path::Path) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).unwrap();
    std::fs::write(
        dm.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"task-tracker\"\n",
    )
    .unwrap();
}

fn add_layered_concept(wiki: &Wiki, rel: &str, title: &str, layer: Layer, body: &str) {
    let page = WikiPage {
        title: title.to_string(),
        page_type: PageType::Concept,
        layer,
        sources: vec![],
        last_updated: "2026-04-26 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: body.to_string(),
        extras: ::std::collections::BTreeMap::new(),
    };
    wiki.write_page(rel, &page).expect("write page");
    let mut idx = wiki.load_index().expect("load index");
    idx.entries.push(IndexEntry {
        title: title.to_string(),
        path: rel.to_string(),
        one_liner: format!("{title} one-liner"),
        category: PageType::Concept,
        last_updated: Some("2026-04-26 00:00:00".to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).expect("save index");
}

#[tokio::test]
async fn host_project_combines_identity_capability_and_layered_wiki() {
    let tmp = tempfile::tempdir().expect("tempdir");

    // 1. Identity: declare this project a host. The same protocol
    //    `dm spawn` writes — manually constructed here so the test
    //    isn't gated on the spawn binary's runtime.
    write_host_identity(tmp.path());
    let host_identity = Identity {
        mode: Mode::Host,
        host_project: Some("task-tracker".to_string()),
        canonical_dm_revision: None,
        canonical_dm_repo: None,
        source: None,
    };

    // 2. Capability extension: install the tracker as the process's
    //    host capabilities. Mirrors what a real host crate does at
    //    startup. (Process-global OnceLock — the file-per-binary
    //    convention keeps this safe across test binaries.)
    install_host_capabilities(Box::new(TaskTracker)).expect("install tracker");

    // 3. Layered wiki: the host project's wiki carries kernel-layer
    //    pages it inherited (or could inherit) from canonical AND a
    //    host-layer page describing its own data model. Both must
    //    coexist without blurring.
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
    add_layered_concept(
        &wiki,
        "concepts/task-data-model.md",
        "Task data model",
        Layer::Host,
        "task tracker host-domain explanation",
    );
    add_layered_concept(
        &wiki,
        "concepts/wiki-layering.md",
        "Wiki layering",
        Layer::Kernel,
        "task layer kernel concept",
    );

    // 4. Tool registry: the default constructor every dm runtime path
    //    uses (TUI, chain, sub-agent, web) must surface the tracker
    //    tools alongside kernel tools.
    let registry = default_registry(
        "tier3-test",
        tmp.path(),
        "http://localhost:11434",
        "gemma:1b",
        "mxbai-embed-large",
    );
    let tool_names = registry.tool_names();
    assert!(
        tool_names.contains(&"host_add_task"),
        "host_add_task must be registered. tools: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"host_list_tasks"),
        "host_list_tasks must be registered. tools: {tool_names:?}"
    );
    assert!(
        tool_names.contains(&"bash"),
        "kernel tools must coexist with host tools. tools: {tool_names:?}"
    );

    // 5. Host capability dispatch: tools must actually round-trip
    //    through the registry's `call(...)`, not just appear in the
    //    name list.
    let added = registry
        .call("host_add_task", json!({"title": "ship Tier 3 cycle 1"}))
        .await
        .expect("host_add_task dispatch");
    assert!(!added.is_error, "host_add_task call failed: {added:?}");
    assert_eq!(added.content, "added: ship Tier 3 cycle 1");
    let listed = registry
        .call("host_list_tasks", json!({}))
        .await
        .expect("host_list_tasks dispatch");
    assert!(!listed.is_error);
    assert!(
        listed.content.contains("ship Tier 3 cycle 1"),
        "host_list_tasks must surface added task. content: {}",
        listed.content
    );

    // 6. Layered wiki search: in host mode, the host-layer concept
    //    surfaces ahead of the kernel-layer match for the same
    //    keyword. This is the Tier 1 stratification working in the
    //    same project that has the host capability installed.
    let hits = wiki
        .search_for_identity("task", &host_identity)
        .expect("identity-aware search");
    assert!(
        hits.len() >= 2,
        "search must find both layered concepts. hits: {hits:?}"
    );
    assert_eq!(
        hits[0].path, "concepts/task-data-model.md",
        "host-layer concept must rank first in host mode. hits: {hits:?}"
    );
    assert_eq!(hits[0].layer, Layer::Host);
    assert!(
        hits.iter().any(|h| h.layer == Layer::Kernel),
        "kernel-layer concept must still surface (after the host hit). hits: {hits:?}"
    );

    // 7. Snippet stratification: the session-start wiki snippet for
    //    this host project lists the host concept above the
    //    `\n---\n` separator and the kernel concept below.
    let snippet = wiki
        .context_snippet_for(&host_identity)
        .expect("host snippet");
    let host_pos = snippet
        .find("Task data model")
        .expect("host concept in snippet");
    let sep_pos = snippet.find("\n---\n").expect("layer separator");
    let kernel_pos = snippet
        .find("Wiki layering")
        .expect("kernel concept in snippet");
    assert!(
        host_pos < sep_pos && sep_pos < kernel_pos,
        "snippet must place host above separator and kernel below:\n{snippet}"
    );
}
