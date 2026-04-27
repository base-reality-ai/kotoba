use dark_matter::identity::{self, Identity, Mode};
use dark_matter::tools::wiki_search::WikiSearchTool;
use dark_matter::tools::Tool;
use dark_matter::wiki::{IndexEntry, Layer, PageType, Wiki, WikiPage};
use serde_json::json;

fn host_identity() -> Identity {
    Identity {
        mode: Mode::Host,
        host_project: Some("finance-app".to_string()),
        canonical_dm_revision: Some("abc123".to_string()),
        canonical_dm_repo: None,
        source: None,
    }
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

fn write_identity(root: &std::path::Path, identity: &Identity) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    let toml = identity::render_toml(identity).expect("render identity");
    std::fs::write(dm.join(identity::IDENTITY_FILENAME), toml).expect("write identity");
}

struct CwdRestore {
    original: std::path::PathBuf,
}

impl CwdRestore {
    fn enter(path: &std::path::Path) -> Self {
        let original = std::env::current_dir().expect("current dir");
        std::env::set_current_dir(path).expect("set current dir");
        Self { original }
    }
}

impl Drop for CwdRestore {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.original);
    }
}

#[tokio::test]
async fn host_mode_wiki_layering_is_end_to_end() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
    let host = host_identity();
    write_identity(tmp.path(), &host);

    add_layered_concept(
        &wiki,
        "concepts/kernel-ledger.md",
        "Kernel ledger",
        Layer::Kernel,
        "ledger ledger ledger inherited kernel explanation",
    );
    add_layered_concept(
        &wiki,
        "concepts/host-ledger.md",
        "Host ledger",
        Layer::Host,
        "ledger host-domain explanation",
    );

    let hits = wiki
        .search_for_identity("ledger", &host)
        .expect("identity-aware search");
    assert_eq!(hits[0].path, "concepts/host-ledger.md");
    assert_eq!(hits[0].layer, Layer::Host);
    assert_eq!(hits[1].path, "concepts/kernel-ledger.md");
    assert_eq!(hits[1].layer, Layer::Kernel);

    let snippet = wiki
        .context_snippet_for(&host)
        .expect("host context snippet");
    let host_pos = snippet.find("Host ledger").expect("host snippet entry");
    let sep_pos = snippet.find("\n---\n").expect("layer separator");
    let kernel_pos = snippet.find("Kernel ledger").expect("kernel snippet entry");
    assert!(
        host_pos < sep_pos && sep_pos < kernel_pos,
        "host snippet must place host entry above separator and kernel below:\n{snippet}",
    );

    let _cwd = CwdRestore::enter(tmp.path());
    let result = WikiSearchTool
        .call(json!({"query": "ledger"}))
        .await
        .expect("wiki_search call");
    assert!(!result.is_error, "{}", result.content);
    let host_badge = result.content.find("[host]").expect("host badge");
    let kernel_badge = result.content.find("[kernel]").expect("kernel badge");
    assert!(
        host_badge < kernel_badge,
        "wiki_search must surface host-layer result before kernel-layer result:\n{}",
        result.content,
    );
}

#[test]
fn kernel_mode_wiki_layering_stays_flat() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
    let kernel = Identity::default_kernel();

    add_layered_concept(
        &wiki,
        "concepts/host-ledger.md",
        "Host ledger",
        Layer::Host,
        "ledger host-domain explanation",
    );
    add_layered_concept(
        &wiki,
        "concepts/kernel-ledger.md",
        "Kernel ledger",
        Layer::Kernel,
        "ledger ledger ledger inherited kernel explanation",
    );

    let hits = wiki
        .search_for_identity("ledger", &kernel)
        .expect("kernel search");
    assert_eq!(hits[0].path, "concepts/kernel-ledger.md");
    assert_eq!(hits[1].path, "concepts/host-ledger.md");

    let snippet = wiki
        .context_snippet_for(&kernel)
        .expect("kernel context snippet");
    assert!(
        !snippet.contains("\n---\n"),
        "kernel snippet must remain flat:\n{snippet}",
    );
}
