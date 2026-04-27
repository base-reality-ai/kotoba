use std::fmt::Write as _;

use super::ingest::{detect_entity_kind, extract_dependencies, extract_purpose};
use super::*;
use tempfile::TempDir;

#[test]
fn open_creates_layout() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let root = wiki.root();
    assert!(root.join("entities").is_dir());
    assert!(root.join("concepts").is_dir());
    assert!(root.join("summaries").is_dir());
    assert!(root.join("synthesis").is_dir());
    assert!(root.join("index.md").is_file());
    assert!(root.join("schema.md").is_file());
}

#[test]
fn open_is_idempotent() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let index_path = wiki.root().join("index.md");
    fs::write(&index_path, "# Custom Index\n\nkeep me\n").unwrap();
    let _ = Wiki::open(tmp.path()).unwrap();
    let contents = fs::read_to_string(&index_path).unwrap();
    assert!(contents.contains("keep me"));
}

#[test]
fn log_append_format() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    log.append("ingest", "src/foo.rs").unwrap();
    let text = fs::read_to_string(log.path()).unwrap();
    let line = text.lines().next().unwrap();
    let re = regex::Regex::new(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ingest \| src/foo\.rs$")
        .unwrap();
    assert!(re.is_match(line), "line did not match: {}", line);
}

#[test]
fn page_roundtrip() {
    let page = WikiPage {
        title: "compaction.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![
            "src/compaction.rs".to_string(),
            "src/session/mod.rs".to_string(),
        ],
        last_updated: "2026-04-17 12:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# compaction.rs\n\nOrchestrates the three-stage compactor.\n".to_string(),
    };
    let md = page.to_markdown();
    let parsed = WikiPage::parse(&md).expect("parse");
    assert_eq!(parsed, page);
}

#[test]
fn layer_as_str_covers_all_variants() {
    assert_eq!(Layer::Kernel.as_str(), "kernel");
    assert_eq!(Layer::Host.as_str(), "host");
}

#[test]
fn layer_from_str_covers_all_variants() {
    assert_eq!("kernel".parse::<Layer>(), Ok(Layer::Kernel));
    assert_eq!("host".parse::<Layer>(), Ok(Layer::Host));
}

#[test]
fn layer_from_str_rejects_unknown_input() {
    assert!("garbage".parse::<Layer>().is_err());
    assert!("".parse::<Layer>().is_err());
    assert!("Kernel".parse::<Layer>().is_err(), "case-sensitive");
}

#[test]
fn page_without_layer_defaults_to_kernel_and_omits_line_in_markdown() {
    let legacy = "---\n\
                  title: legacy\n\
                  type: concept\n\
                  sources:\n  - src/foo.rs\n\
                  last_updated: 2026-04-17 00:00:00\n\
                  ---\n\
                  # legacy body\n";
    let parsed = WikiPage::parse(legacy).expect("legacy page must parse");
    assert_eq!(parsed.layer, Layer::Kernel);

    let md = parsed.to_markdown();
    assert!(
        !md.contains("\nlayer: "),
        "kernel default must not emit a layer line: {}",
        md
    );
}

#[test]
fn page_with_explicit_kernel_layer_parses_as_kernel() {
    let md = "---\n\
              title: explicit kernel\n\
              type: concept\n\
              layer: kernel\n\
              sources:\n  - src/foo.rs\n\
              last_updated: 2026-04-17 00:00:00\n\
              ---\n\
              # body\n";
    let parsed = WikiPage::parse(md).expect("explicit kernel layer must parse");
    assert_eq!(parsed.layer, Layer::Kernel);
}

#[test]
fn page_with_host_layer_round_trips_through_markdown() {
    let page = WikiPage {
        title: "host concept".to_string(),
        page_type: PageType::Concept,
        layer: Layer::Host,
        sources: vec!["src/domain.rs".to_string()],
        last_updated: "2026-04-26 20:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# host concept\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        md.contains("layer: host\n"),
        "host layer must be explicit in frontmatter: {}",
        md
    );
    let parsed = WikiPage::parse(&md).expect("host page must parse");
    assert_eq!(parsed, page);
}

#[test]
fn page_with_unknown_layer_value_is_malformed() {
    let md = "---\n\
              title: bad layer\n\
              type: concept\n\
              layer: substrate\n\
              sources:\n  - src/foo.rs\n\
              last_updated: 2026-04-17 00:00:00\n\
              ---\n\
              # body\n";
    assert!(WikiPage::parse(md).is_none());
}

#[test]
fn entity_kind_as_str_covers_all_variants() {
    assert_eq!(EntityKind::Function.as_str(), "function");
    assert_eq!(EntityKind::Struct.as_str(), "struct");
    assert_eq!(EntityKind::Enum.as_str(), "enum");
    assert_eq!(EntityKind::Trait.as_str(), "trait");
    assert_eq!(EntityKind::Unknown.as_str(), "unknown");
}

#[test]
fn entity_kind_from_str_covers_all_variants() {
    assert_eq!("function".parse::<EntityKind>(), Ok(EntityKind::Function));
    assert_eq!("struct".parse::<EntityKind>(), Ok(EntityKind::Struct));
    assert_eq!("enum".parse::<EntityKind>(), Ok(EntityKind::Enum));
    assert_eq!("trait".parse::<EntityKind>(), Ok(EntityKind::Trait));
    assert_eq!("unknown".parse::<EntityKind>(), Ok(EntityKind::Unknown));
}

#[test]
fn entity_kind_from_str_rejects_unknown_input() {
    // Malformed tokens must Err — the parser relies on .ok() to coerce to
    // None without failing the whole page parse.
    assert!("garbage".parse::<EntityKind>().is_err());
    assert!("".parse::<EntityKind>().is_err());
    assert!("Function".parse::<EntityKind>().is_err(), "case-sensitive");
}

#[test]
fn page_with_entity_kind_round_trips_through_markdown() {
    let page = WikiPage {
        title: "short_id".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/session/mod.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: Some(EntityKind::Function),
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# short_id\n\nBorrowed 8-char prefix helper.\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        md.contains("entity_kind: function\n"),
        "markdown should carry the kind line: {}",
        md
    );
    let parsed = WikiPage::parse(&md).expect("parse");
    assert_eq!(parsed, page);
}

#[test]
fn page_without_entity_kind_omits_line_in_markdown() {
    let page = WikiPage {
        title: "concept".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# concept\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        !md.contains("entity_kind"),
        "None must not emit an entity_kind line: {}",
        md
    );
    let parsed = WikiPage::parse(&md).expect("parse");
    assert_eq!(parsed.entity_kind, None);
}

#[test]
fn page_parses_legacy_markdown_without_entity_kind_line() {
    // Pre-Cycle 38 on-disk format had no entity_kind line. Parsing must
    // yield None without error so existing wiki pages continue to load.
    let legacy = "---\n\
                      title: legacy\n\
                      type: entity\n\
                      sources:\n  - src/foo.rs\n\
                      last_updated: 2026-04-17 00:00:00\n\
                      ---\n\
                      # legacy body\n";
    let parsed = WikiPage::parse(legacy).expect("legacy page must parse");
    assert_eq!(parsed.entity_kind, None);
    assert_eq!(parsed.page_type, PageType::Entity);
    assert_eq!(parsed.sources, vec!["src/foo.rs".to_string()]);
}

#[test]
fn page_round_trips_each_entity_kind_variant() {
    for kind in [
        EntityKind::Function,
        EntityKind::Struct,
        EntityKind::Enum,
        EntityKind::Trait,
        EntityKind::Unknown,
    ] {
        let page = WikiPage {
            title: "v".to_string(),
            page_type: PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/v.rs".to_string()],
            last_updated: "2026-04-18 09:00:00".to_string(),
            entity_kind: Some(kind),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "# v\n".to_string(),
        };
        let parsed = WikiPage::parse(&page.to_markdown()).expect("parse");
        assert_eq!(
            parsed.entity_kind,
            Some(kind),
            "variant {:?} must survive round-trip",
            kind
        );
    }
}

#[test]
fn page_parse_ignores_malformed_entity_kind_value() {
    // A garbage value in the frontmatter must not fail the whole parse —
    // we want forward-compat (future variants) and defensive degradation
    // against hand-edited pages. Downstream treats it as None.
    let md = "---\n\
                  title: t\n\
                  type: entity\n\
                  entity_kind: banana\n\
                  sources:\n  - src/t.rs\n\
                  last_updated: 2026-04-18 09:00:00\n\
                  ---\n\
                  # t\n";
    let parsed = WikiPage::parse(md).expect("must still parse despite bad kind");
    assert_eq!(parsed.entity_kind, None);
    assert_eq!(parsed.title, "t");
}

#[test]
fn detect_entity_kind_single_function_file_returns_function() {
    assert_eq!(
        detect_entity_kind("src/x.rs", "pub fn foo() {}\n"),
        Some(EntityKind::Function)
    );
}

#[test]
fn detect_entity_kind_single_struct_file_returns_struct() {
    assert_eq!(
        detect_entity_kind("src/x.rs", "pub struct Foo;\n"),
        Some(EntityKind::Struct)
    );
}

#[test]
fn detect_entity_kind_single_enum_file_returns_enum() {
    assert_eq!(
        detect_entity_kind("src/x.rs", "pub enum E { A, B }\n"),
        Some(EntityKind::Enum)
    );
}

#[test]
fn detect_entity_kind_single_trait_file_returns_trait() {
    assert_eq!(
        detect_entity_kind("src/x.rs", "pub trait T {}\n"),
        Some(EntityKind::Trait)
    );
}

#[test]
fn detect_entity_kind_mixed_kinds_returns_unknown() {
    // Multiple represented kinds collapse to the Unknown sentinel. Pins the
    // behavior so a future cycle promoting "mixed" to a dedicated variant
    // (or giving Unknown a payload) has a canary here.
    let src = "pub fn foo() {}\npub struct Bar;\n";
    assert_eq!(
        detect_entity_kind("src/x.rs", src),
        Some(EntityKind::Unknown)
    );
}

#[test]
fn detect_entity_kind_non_rust_path_returns_none() {
    // Content contains Rust-like tokens but the path extension gate wins.
    assert_eq!(
        detect_entity_kind("docs/readme.md", "pub fn foo() {}\npub struct Bar;\n"),
        None
    );
}

#[test]
fn detect_entity_kind_const_only_file_returns_none() {
    // const/static/type/mod/impl are not in EntityKind's schema; a file
    // made only of them must resolve to None, not accidentally map to
    // one of the represented kinds.
    assert_eq!(
        detect_entity_kind("src/x.rs", "pub const X: u32 = 1;\n"),
        None
    );
}

#[test]
fn ingest_file_populates_entity_kind_for_single_kind_rust_file() {
    // End-to-end: regex parsing can be correct while wiring is broken.
    // Write a single-function source, ingest, re-read the page, and
    // confirm the frontmatter carries entity_kind=function.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();
    let src_abs = project_root.join("foo.rs");
    fs::write(&src_abs, "pub fn main() {}\n").unwrap();
    let wiki_root = project_root.join(".dm").join("wiki");
    let wiki = Wiki::open(&wiki_root).unwrap();

    let outcome = wiki.ingest_file(project_root, &src_abs, "pub fn main() {}\n");
    assert!(outcome.is_ok(), "ingest must succeed: {:?}", outcome);

    let page_rel = entity_page_rel("foo.rs");
    let page_text = fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let parsed = WikiPage::parse(&page_text).expect("ingested page must parse");
    assert_eq!(parsed.entity_kind, Some(EntityKind::Function));
    assert!(
        page_text.contains("entity_kind: function\n"),
        "frontmatter must carry the kind line: {}",
        page_text
    );
}

#[test]
fn extract_key_exports_captures_single_pub_function() {
    let exports = extract_key_exports("src/x.rs", "pub fn foo() {}\n");
    assert_eq!(
        exports,
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "foo".to_string(),
        }]
    );
}

#[test]
fn extract_key_exports_captures_mixed_kinds_and_skips_unrepresented() {
    // Includes a `pub const` — which is not in EntityKind's schema today
    // and must be dropped. Order is source-order (regex iteration order).
    let src = "pub fn foo() {}\npub struct Bar;\npub const X: u32 = 1;\n";
    let exports = extract_key_exports("src/x.rs", src);
    assert_eq!(
        exports,
        vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "foo".to_string(),
            },
            KeyExport {
                kind: EntityKind::Struct,
                name: "Bar".to_string(),
            },
        ]
    );
}

#[test]
fn extract_key_exports_skips_private_items() {
    // Private items have no `vis` capture. "Key exports" is a public-API
    // concept; this is the deliberate divergence from detect_entity_kind
    // (which counts all items, including private, because structural
    // identity is visibility-independent).
    let src = "fn private() {}\npub fn public() {}\n";
    let exports = extract_key_exports("src/x.rs", src);
    assert_eq!(
        exports,
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "public".to_string(),
        }]
    );
}

#[test]
fn extract_key_exports_returns_empty_for_non_rust_file() {
    // Non-Rust path gate mirrors detect_entity_kind. Even if the content
    // looks Rust-like, extraction is Rust-only today.
    let exports = extract_key_exports("docs/notes.md", "pub fn foo() {}\n");
    assert!(exports.is_empty());
}

#[test]
fn key_exports_round_trip_through_markdown() {
    // Serialize a page with exports, parse the output, and confirm the
    // exports survive verbatim. Also asserts the frontmatter line ordering:
    // entity_kind -> key_exports -> sources (sibling ingest-derived blocks
    // live together, before sources).
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: Some(EntityKind::Unknown),
        purpose: None,
        key_exports: vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "foo".to_string(),
            },
            KeyExport {
                kind: EntityKind::Struct,
                name: "Bar".to_string(),
            },
        ],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    let ek_at = md.find("entity_kind:").expect("entity_kind line");
    let ke_at = md.find("key_exports:").expect("key_exports line");
    let sr_at = md.find("sources:").expect("sources line");
    assert!(
        ek_at < ke_at && ke_at < sr_at,
        "ordering must be entity_kind -> key_exports -> sources, got: {}",
        md
    );
    let reparsed = WikiPage::parse(&md).expect("round-trip must parse");
    assert_eq!(reparsed.key_exports, page.key_exports);
}

#[test]
fn wiki_page_round_trips_scope_field() {
    // C40 schema addition: scope: Vec<String> survives to_markdown
    // ↔ parse, including empty (back-compat) and non-empty cases.
    let with_scope = WikiPage {
        title: "Probe".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/foo.rs".to_string()],
        last_updated: "2026-04-26 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec!["src/wiki/".to_string(), "src/tui/commands/".to_string()],
        body: "test body".to_string(),
    };
    let serialized = with_scope.to_markdown();
    assert!(
        serialized.contains("scope:\n  - src/wiki/\n  - src/tui/commands/"),
        "scope block must serialize: {}",
        serialized
    );
    let parsed = WikiPage::parse(&serialized).expect("parse");
    assert_eq!(parsed.scope, with_scope.scope, "scope round-trip");

    // Back-compat: page with empty scope serializes WITHOUT a scope block,
    // and re-parses with empty scope.
    let no_scope = WikiPage {
        scope: vec![],
        ..with_scope
    };
    let serialized = no_scope.to_markdown();
    assert!(
        !serialized.contains("scope:"),
        "empty scope must not emit a block: {}",
        serialized
    );
    let parsed = WikiPage::parse(&serialized).expect("parse");
    assert!(parsed.scope.is_empty(), "empty scope round-trips empty");
}

#[test]
fn key_exports_omitted_from_markdown_when_empty() {
    // Legacy byte-identity: a page with no exports emits no block at all.
    // Pages written before key_exports shipped continue to produce the
    // same bytes after round-tripping.
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        !md.contains("key_exports"),
        "empty key_exports must not emit a block: {}",
        md
    );
}

#[test]
fn legacy_page_without_key_exports_line_parses_with_empty_vec() {
    // Backward-compat load: a hand-crafted page from before key_exports
    // shipped must parse cleanly and expose an empty Vec.
    let md = "---\n\
                  title: src/x.rs\n\
                  type: entity\n\
                  sources:\n  - src/x.rs\n\
                  last_updated: 2026-04-18 09:00:00\n\
                  ---\n\
                  # x\n";
    let parsed = WikiPage::parse(md).expect("legacy page must parse");
    assert!(parsed.key_exports.is_empty());
    assert_eq!(parsed.sources, vec!["src/x.rs".to_string()]);
}

#[test]
fn ingest_file_populates_key_exports_end_to_end() {
    // Full-stack: wiring can break between the pure extractor and the
    // page on disk. Write a source with two public exports, ingest,
    // re-read, and confirm both the parsed Vec and the raw frontmatter
    // block survive.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();
    let src_abs = project_root.join("foo.rs");
    let content = "pub fn a() {}\npub struct B;\n";
    fs::write(&src_abs, content).unwrap();
    let wiki_root = project_root.join(".dm").join("wiki");
    let wiki = Wiki::open(&wiki_root).unwrap();

    let outcome = wiki.ingest_file(project_root, &src_abs, content);
    assert!(outcome.is_ok(), "ingest must succeed: {:?}", outcome);

    let page_rel = entity_page_rel("foo.rs");
    let page_text = fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let parsed = WikiPage::parse(&page_text).expect("ingested page must parse");
    assert_eq!(
        parsed.key_exports,
        vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "a".to_string(),
            },
            KeyExport {
                kind: EntityKind::Struct,
                name: "B".to_string(),
            },
        ]
    );
    assert!(
        page_text.contains("key_exports:\n  - function a\n  - struct B\n"),
        "frontmatter must carry the exports block: {}",
        page_text
    );
}

#[test]
fn extract_dependencies_captures_uses_in_source_order() {
    let src = "use std::path::PathBuf;\n\
                   use crate::config::Config;\n\
                   use tokio::sync::Mutex;\n\
                   pub fn main() {}\n";
    let deps = extract_dependencies("src/x.rs", src);
    assert_eq!(
        deps,
        vec![
            "std::path::PathBuf".to_string(),
            "crate::config::Config".to_string(),
            "tokio::sync::Mutex".to_string(),
        ]
    );
}

#[test]
fn extract_dependencies_strips_as_rename_and_accepts_pub_use() {
    // Top-level `as X` rename drops to the import target; `pub use`
    // re-exports are still dependencies (the file pulls the target in).
    let src = "use foo::bar as baz;\n\
                   pub use qux::quux;\n";
    let deps = extract_dependencies("src/x.rs", src);
    assert_eq!(deps, vec!["foo::bar".to_string(), "qux::quux".to_string()]);
}

#[test]
fn extract_dependencies_collapses_multiline_brace_group() {
    // Multi-line `use foo::{ bar, baz, };` collapses internal whitespace
    // (including newlines and trailing comma whitespace) to a single
    // string. Returns one entry, not three.
    let src = "use foo::{\n    bar,\n    baz,\n};\n";
    let deps = extract_dependencies("src/x.rs", src);
    assert_eq!(deps.len(), 1, "multi-line group is one dep: {:?}", deps);
    assert_eq!(deps[0], "foo::{ bar, baz, }");
}

#[test]
fn extract_dependencies_returns_empty_for_non_rust_file() {
    // Path gate mirrors extract_key_exports / detect_entity_kind.
    let deps = extract_dependencies("docs/notes.md", "use foo::bar;\n");
    assert!(deps.is_empty());
}

#[test]
fn extract_dependencies_preserves_group_internal_as_rename() {
    // Code comment in src/wiki/ingest.rs spells out the contract:
    // "Group-internal renames are preserved verbatim on purpose."
    // The ` as ` strip must only fire when no `{` is present —
    // otherwise `foo::{bar as b, baz}` would be truncated at the
    // first ` as `, dropping `baz`. Pin this so a future refactor
    // can't silently regress to the lossy split.
    let src = "use foo::{bar as b, baz};\n";
    let deps = extract_dependencies("src/x.rs", src);
    assert_eq!(deps.len(), 1, "single use is one dep: {:?}", deps);
    let dep = &deps[0];
    assert!(
        dep.contains("baz"),
        "group-internal sibling must survive the as-rename strip: {}",
        dep
    );
    assert!(
        dep.contains("bar as b"),
        "group-internal rename must be preserved verbatim: {}",
        dep
    );
}

/// Pin nested-brace handling in `extract_dependencies`. The regex
/// captures everything between `use` and the first `;`, so
/// `use foo::{bar::{baz, qux}, quux};` yields a single dep string
/// containing all three inner names. A future refactor that switches
/// to a brace-counting parser must still surface every leaf name —
/// this test passes whether the function emits one dep or several,
/// as long as `baz`, `qux`, and `quux` all appear *somewhere* in the
/// dep list.
#[test]
fn extract_dependencies_preserves_nested_brace_groups() {
    let src = "use foo::{bar::{baz, qux}, quux};\n";
    let deps = extract_dependencies("src/x.rs", src);
    assert!(
        !deps.is_empty(),
        "nested group must produce at least one dep"
    );
    let joined = deps.join(" | ");
    for needle in ["baz", "qux", "quux"] {
        assert!(
            joined.contains(needle),
            "leaf name `{}` must appear in extracted deps; got: {}",
            needle,
            joined
        );
    }
    // Outer brace must not have been silently truncated either —
    // `foo` (the head segment) should be visible somewhere.
    assert!(
        joined.contains("foo"),
        "head segment `foo` lost: {}",
        joined
    );
}

#[test]
fn dependencies_round_trip_through_markdown() {
    // Extends the Cycle 40 canary: asserts entity_kind < key_exports <
    // dependencies < sources ordering in the serialized frontmatter, and
    // that dependencies survive serialize -> parse unchanged.
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: Some(EntityKind::Function),
        purpose: None,
        key_exports: vec![KeyExport {
            kind: EntityKind::Function,
            name: "main".to_string(),
        }],
        dependencies: vec![
            "std::path::PathBuf".to_string(),
            "tokio::sync::Mutex".to_string(),
        ],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    let ek_at = md.find("entity_kind:").expect("entity_kind line");
    let ke_at = md.find("key_exports:").expect("key_exports line");
    let dp_at = md.find("dependencies:").expect("dependencies line");
    let sr_at = md.find("sources:").expect("sources line");
    assert!(
        ek_at < ke_at && ke_at < dp_at && dp_at < sr_at,
        "ordering must be entity_kind -> key_exports -> dependencies -> sources, got: {}",
        md
    );
    let reparsed = WikiPage::parse(&md).expect("round-trip must parse");
    assert_eq!(reparsed.dependencies, page.dependencies);
}

#[test]
fn dependencies_omitted_from_markdown_when_empty() {
    // Legacy byte-identity: empty dependencies emits no block at all.
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        !md.contains("dependencies"),
        "empty dependencies must not emit a block: {}",
        md
    );
}

#[test]
fn legacy_page_without_dependencies_line_parses_with_empty_vec() {
    // Backward-compat: a page from before this field shipped must parse
    // cleanly and expose an empty Vec.
    let md = "---\n\
                  title: src/x.rs\n\
                  type: entity\n\
                  sources:\n  - src/x.rs\n\
                  last_updated: 2026-04-18 09:00:00\n\
                  ---\n\
                  # x\n";
    let parsed = WikiPage::parse(md).expect("legacy page must parse");
    assert!(parsed.dependencies.is_empty());
}

#[test]
fn ingest_file_entity_extraction_respects_env_disable() {
    // Pins the invariant that makes ENV_LOCK guards load-bearing for
    // the sibling `ingest_file_populates_*_end_to_end` tests: when
    // `DM_WIKI_AUTO_INGEST=0` is set, `ingest_file` silently writes
    // no page, and the siblings' "ingested page must parse" assertions
    // fail. If a future cycle rewires `ingest_file` to bypass the env
    // gate, this canary catches the regression before the flake
    // returns under parallel test execution.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", "0");
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();
    let src_abs = project_root.join("foo.rs");
    let content = "pub fn a() {}\npub struct B;\n";
    fs::write(&src_abs, content).unwrap();
    let wiki_root = project_root.join(".dm").join("wiki");
    let wiki = Wiki::open(&wiki_root).unwrap();

    let outcome = wiki.ingest_file(project_root, &src_abs, content);
    assert!(
        outcome.is_ok(),
        "ingest_file should not error when env-disabled, just no-op: {:?}",
        outcome,
    );

    let page_rel = entity_page_rel("foo.rs");
    let page_path = wiki.root().join(&page_rel);
    assert!(
        !page_path.exists(),
        "ingest_file must no-op (not write page) when \
             DM_WIKI_AUTO_INGEST=0; page unexpectedly at {:?}",
        page_path,
    );
}

#[test]
fn ingest_file_populates_dependencies_end_to_end() {
    // Full-stack: write a source with two single-line `use` statements,
    // ingest, re-read the page, and confirm both the parsed Vec and the
    // raw frontmatter block round-trip.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();
    let src_abs = project_root.join("foo.rs");
    let content = "use std::path::PathBuf;\n\
                       use tokio::sync::Mutex;\n\
                       pub fn a() {}\n";
    fs::write(&src_abs, content).unwrap();
    let wiki_root = project_root.join(".dm").join("wiki");
    let wiki = Wiki::open(&wiki_root).unwrap();

    let outcome = wiki.ingest_file(project_root, &src_abs, content);
    assert!(outcome.is_ok(), "ingest must succeed: {:?}", outcome);

    let page_rel = entity_page_rel("foo.rs");
    let page_text = fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let parsed = WikiPage::parse(&page_text).expect("ingested page must parse");
    assert_eq!(
        parsed.dependencies,
        vec![
            "std::path::PathBuf".to_string(),
            "tokio::sync::Mutex".to_string(),
        ]
    );
    assert!(
        page_text.contains("dependencies:\n  - std::path::PathBuf\n  - tokio::sync::Mutex\n"),
        "frontmatter must carry the dependencies block: {}",
        page_text
    );
}

#[test]
fn extract_purpose_returns_first_paragraph_joined_with_spaces() {
    let src = "//! First line of summary.\n\
                   //! Second line continuation.\n\
                   //!\n\
                   //! Deep detail that should NOT be in purpose.\n\
                   pub fn foo() {}\n";
    let purpose = extract_purpose("src/x.rs", src);
    assert_eq!(
        purpose,
        Some("First line of summary. Second line continuation.".to_string())
    );
}

#[test]
fn extract_purpose_caps_at_three_lines() {
    // Five consecutive non-blank `//!` lines → only the first 3 are
    // joined. The 4th and 5th must not leak into the summary.
    let src = "//! Line 1.\n\
                   //! Line 2.\n\
                   //! Line 3.\n\
                   //! Line 4 should be dropped.\n\
                   //! Line 5 should be dropped.\n\
                   pub fn foo() {}\n";
    let purpose = extract_purpose("src/x.rs", src).expect("some purpose");
    assert!(purpose.contains("Line 1."), "missing L1: {}", purpose);
    assert!(purpose.contains("Line 2."), "missing L2: {}", purpose);
    assert!(purpose.contains("Line 3."), "missing L3: {}", purpose);
    assert!(!purpose.contains("Line 4"), "leaked L4: {}", purpose);
    assert!(!purpose.contains("Line 5"), "leaked L5: {}", purpose);
}

#[test]
fn extract_purpose_stops_at_first_blank_slash_bang_line() {
    // A blank `//!` line terminates the first paragraph even before
    // the 3-line cap fires. Subsequent content is deep-detail.
    let src = "//! Line 1.\n//!\n//! Line after blank.\n";
    let purpose = extract_purpose("src/x.rs", src);
    assert_eq!(purpose, Some("Line 1.".to_string()));
}

#[test]
fn extract_purpose_returns_none_for_non_rust_file() {
    let purpose = extract_purpose("docs/notes.md", "//! valid rust doc\n");
    assert!(purpose.is_none());
}

#[test]
fn extract_purpose_returns_none_when_no_doc_comments() {
    let purpose = extract_purpose("src/x.rs", "pub fn foo() {}\n");
    assert!(purpose.is_none());
}

#[test]
fn purpose_round_trip_through_markdown() {
    // Extends the Cycle 40/41 canary: entity_kind -> purpose ->
    // key_exports -> dependencies -> sources must appear in that order.
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: Some(EntityKind::Function),
        purpose: Some("A test module.".to_string()),
        key_exports: vec![KeyExport {
            kind: EntityKind::Function,
            name: "main".to_string(),
        }],
        dependencies: vec!["std::path::PathBuf".to_string()],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    let ek_at = md.find("entity_kind:").expect("entity_kind line");
    let pu_at = md.find("purpose:").expect("purpose line");
    let ke_at = md.find("key_exports:").expect("key_exports line");
    let dp_at = md.find("dependencies:").expect("dependencies line");
    let sr_at = md.find("sources:").expect("sources line");
    assert!(
            ek_at < pu_at && pu_at < ke_at && ke_at < dp_at && dp_at < sr_at,
            "ordering must be entity_kind -> purpose -> key_exports -> dependencies -> sources, got: {}",
            md
        );
    let reparsed = WikiPage::parse(&md).expect("round-trip must parse");
    assert_eq!(reparsed.purpose, page.purpose);
}

#[test]
fn purpose_omitted_from_markdown_when_none_and_legacy_pages_parse_with_none() {
    // (a) None purpose must not emit a "purpose:" line — byte-identity
    // with pages written before this field shipped.
    let page = WikiPage {
        title: "src/x.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/x.rs".to_string()],
        last_updated: "2026-04-18 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# x\n".to_string(),
    };
    let md = page.to_markdown();
    assert!(
        !md.contains("purpose:"),
        "None purpose must not emit a line: {}",
        md
    );

    // (b) A hand-crafted frontmatter without a purpose line parses
    // cleanly with None.
    let legacy = "---\n\
                      title: src/x.rs\n\
                      type: entity\n\
                      sources:\n  - src/x.rs\n\
                      last_updated: 2026-04-18 09:00:00\n\
                      ---\n\
                      # x\n";
    let parsed = WikiPage::parse(legacy).expect("legacy page must parse");
    assert!(parsed.purpose.is_none());
}

#[test]
fn ingest_file_populates_purpose_end_to_end() {
    // Full-stack: module docs drive the purpose field end-to-end.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();
    let src_abs = project_root.join("foo.rs");
    let content = "//! Utility helpers for Foo.\n\
                       //! Exposes the build() function.\n\
                       \n\
                       pub fn build() -> u32 { 42 }\n";
    fs::write(&src_abs, content).unwrap();
    let wiki_root = project_root.join(".dm").join("wiki");
    let wiki = Wiki::open(&wiki_root).unwrap();

    let outcome = wiki.ingest_file(project_root, &src_abs, content);
    assert!(outcome.is_ok(), "ingest must succeed: {:?}", outcome);

    let page_rel = entity_page_rel("foo.rs");
    let page_text = fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let parsed = WikiPage::parse(&page_text).expect("ingested page must parse");
    assert_eq!(
        parsed.purpose,
        Some("Utility helpers for Foo. Exposes the build() function.".to_string())
    );
    assert!(
        page_text.contains("purpose: Utility helpers for Foo. Exposes the build() function.\n"),
        "frontmatter must carry the purpose line: {}",
        page_text
    );
}

#[test]
fn load_empty_index() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = wiki.load_index().unwrap();
    assert!(idx.entries.is_empty());
}

#[test]
fn index_roundtrip_with_entries() {
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "Wiki".to_string(),
                path: "entities/wiki.md".to_string(),
                one_liner: "Persistent project knowledge layer.".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "Compaction pipeline".to_string(),
                path: "concepts/compaction.md".to_string(),
                one_liner: "Three-stage context trimming.".to_string(),
                category: PageType::Concept,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    let md = idx.to_markdown();
    let parsed = WikiIndex::parse(&md);
    assert_eq!(parsed, idx);
}

#[test]
fn page_parse_no_leading_fence_returns_none() {
    let text = "title: x\ntype: entity\nsources:\nlast_updated: 2026-04-17\n\nbody";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_missing_closing_fence_returns_none() {
    let text = "---\ntitle: x\ntype: entity\nsources:\nlast_updated: 2026-04-17\nbody\n";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_missing_title_returns_none() {
    let text = "---\ntype: entity\nsources:\nlast_updated: 2026-04-17 00:00:00\n---\nbody\n";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_missing_type_returns_none() {
    let text = "---\ntitle: x\nsources:\nlast_updated: 2026-04-17 00:00:00\n---\nbody\n";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_missing_last_updated_returns_none() {
    let text = "---\ntitle: x\ntype: entity\nsources:\n---\nbody\n";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_unknown_type_returns_none() {
    let text =
        "---\ntitle: x\ntype: gibberish\nsources:\nlast_updated: 2026-04-17 00:00:00\n---\nbody\n";
    assert!(WikiPage::parse(text).is_none());
}

#[test]
fn page_parse_empty_string_returns_none() {
    assert!(WikiPage::parse("").is_none());
}

#[test]
fn page_parse_preserves_multi_paragraph_body() {
    let page = WikiPage {
        title: "x".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# Heading\n\nPara 1.\n\nPara 2 with `---` inside.\n".to_string(),
    };
    let md = page.to_markdown();
    let parsed = WikiPage::parse(&md).expect("parse");
    assert_eq!(parsed, page);
}

#[test]
fn index_parse_ignores_unknown_headings_and_noise() {
    let text = "\
# Wiki Index\n\n\
Random preamble.\n\n\
## Unknown\n\n\
- [Ignored](path.md) — should not appear\n\n\
## Entities\n\n\
not a list line\n\
- [Foo](entities/foo.md) — foo summary\n\
- broken line without link format\n\
- [Bar](entities/bar.md) — bar summary\n\n\
## Concepts\n\n\
- [Baz](concepts/baz.md) — baz summary\n";
    let idx = WikiIndex::parse(text);
    let titles: Vec<&str> = idx.entries.iter().map(|e| e.title.as_str()).collect();
    assert_eq!(titles, vec!["Foo", "Bar", "Baz"]);
    assert_eq!(idx.entries[0].category, PageType::Entity);
    assert_eq!(idx.entries[2].category, PageType::Concept);
    // Entries under an unknown heading must be discarded.
    assert!(idx.entries.iter().all(|e| e.title != "Ignored"));
}

#[test]
fn index_parse_tolerates_missing_em_dash() {
    let text = "## Entities\n\n- [Foo](entities/foo.md) no em-dash here\n";
    let idx = WikiIndex::parse(text);
    assert_eq!(idx.entries.len(), 1);
    assert_eq!(idx.entries[0].one_liner, "no em-dash here");
}

#[test]
fn load_index_returns_empty_on_missing_file() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    fs::remove_file(wiki.root().join("index.md")).unwrap();
    let idx = wiki.load_index().unwrap();
    assert!(idx.entries.is_empty());
}

#[test]
fn save_and_load_index_roundtrips() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Wiki".to_string(),
            path: "entities/wiki.md".to_string(),
            one_liner: "test".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let loaded = wiki.load_index().unwrap();
    assert_eq!(loaded, idx);
}

#[test]
fn ensure_layout_concurrent_calls_are_safe() {
    use std::sync::Arc;
    use std::thread;
    let tmp = Arc::new(TempDir::new().unwrap());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let tmp = Arc::clone(&tmp);
        handles.push(thread::spawn(move || Wiki::open(tmp.path()).expect("open")));
    }
    for h in handles {
        let w = h.join().unwrap();
        assert!(w.root().join("index.md").is_file());
        assert!(w.root().join("schema.md").is_file());
        assert!(w.root().join("entities").is_dir());
    }
    // Every thread wrote the same seed, so the final file must still be a
    // valid parse of the seeded index (empty entries).
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = wiki.load_index().unwrap();
    assert!(idx.entries.is_empty());
}

#[test]
fn log_append_multiple_entries_are_ordered() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    log.append("ingest", "a").unwrap();
    log.append("ingest", "b").unwrap();
    log.append("lint", "c").unwrap();
    let text = fs::read_to_string(log.path()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].ends_with("ingest | a"));
    assert!(lines[1].ends_with("ingest | b"));
    assert!(lines[2].ends_with("lint | c"));
}

#[test]
fn page_type_from_str_valid_and_invalid() {
    assert_eq!("entity".parse::<PageType>(), Ok(PageType::Entity));
    assert_eq!("concept".parse::<PageType>(), Ok(PageType::Concept));
    assert_eq!("summary".parse::<PageType>(), Ok(PageType::Summary));
    assert_eq!("synthesis".parse::<PageType>(), Ok(PageType::Synthesis));
    // Whitespace tolerance.
    assert_eq!("  entity  ".parse::<PageType>(), Ok(PageType::Entity));
    assert_eq!("\tconcept\n".parse::<PageType>(), Ok(PageType::Concept));
    // Errors.
    assert!("bogus".parse::<PageType>().is_err());
    assert!("".parse::<PageType>().is_err());
    assert!("Entity".parse::<PageType>().is_err()); // case-sensitive
}

#[test]
fn write_page_and_read_page_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = WikiPage {
        title: "foo".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/foo.rs".to_string()],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# foo\n\nNotes.\n".to_string(),
    };
    wiki.write_page("entities/foo.md", &page).unwrap();
    let read = wiki.read_page("entities/foo.md").unwrap();
    assert_eq!(read, page);
}

#[test]
fn write_page_creates_subdir() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = WikiPage {
        title: "deep".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("entities/new/deep.md", &page).unwrap();
    assert!(wiki.root().join("entities/new").is_dir());
    assert!(wiki.root().join("entities/new/deep.md").is_file());
}

#[test]
fn read_page_on_missing_file_returns_error() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let err = wiki.read_page("entities/nope.md").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::NotFound);
}

#[test]
fn read_page_on_malformed_content_returns_invalid_data() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let path = wiki.root().join("entities/broken.md");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, "not a page").unwrap();
    let err = wiki.read_page("entities/broken.md").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn write_page_rejects_parent_traversal() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = WikiPage {
        title: "escape".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    let err = wiki.write_page("../escape.md", &page).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    let err = wiki.write_page("/etc/passwd", &page).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    let err = wiki
        .write_page("entities/../../../boom.md", &page)
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    let err = wiki.write_page("", &page).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn read_page_rejects_parent_traversal() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let err = wiki.read_page("../escape.md").unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn validate_rel_rejects_windows_style_traversal() {
    // Backslash leading segment.
    assert_eq!(
        validate_rel("\\server\\share").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    // Pure backslash parent traversal.
    assert_eq!(
        validate_rel("..\\escape.md").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    // Mixed separators with parent traversal.
    assert_eq!(
        validate_rel("entities\\..\\..\\boom.md")
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        validate_rel("entities/..\\boom.md").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

#[test]
fn validate_rel_accepts_reasonable_paths() {
    // Sanity: normal relative paths must pass.
    assert!(validate_rel("entities/foo.md").is_ok());
    assert!(validate_rel("entities/sub/bar.md").is_ok());
    assert!(validate_rel("concepts/architecture.md").is_ok());
    // A single "." component is not a parent traversal.
    assert!(validate_rel("entities/./foo.md").is_ok());
}

#[test]
fn write_page_backslash_traversal_stays_in_root() {
    // Even with a traversal-like backslash literal on Unix (where `\` is
    // not a path separator), validate_rel must still reject it.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = WikiPage {
        title: "x".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    let err = wiki.write_page("..\\escape.md", &page).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
}

#[test]
fn ensure_for_cwd_creates_layout_in_cwd() {
    let _guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let tmp = TempDir::new().unwrap();
    let orig = std::env::current_dir().unwrap();
    std::env::set_current_dir(tmp.path()).unwrap();
    let result = ensure_for_cwd();
    // Always restore before assertions so a panic doesn't leave the test
    // runner stranded in the tempdir.
    std::env::set_current_dir(&orig).unwrap();
    let wiki = result.expect("wiki");
    assert!(wiki.root().join("index.md").is_file());
    assert!(wiki.root().join("schema.md").is_file());
    assert!(wiki.root().join("entities").is_dir());
}

#[test]
fn context_snippet_empty_returns_none() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    assert!(wiki.context_snippet().is_none());
}

#[test]
fn context_snippet_with_entries_formats_by_category() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "Compaction".to_string(),
                path: "entities/compaction.md".to_string(),
                one_liner: "Three-stage context trimming.".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "Session".to_string(),
                path: "entities/session.md".to_string(),
                one_liner: "Persistent conversation state.".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "Wiki pattern".to_string(),
                path: "concepts/wiki.md".to_string(),
                one_liner: "Karpathy LLM wiki.".to_string(),
                category: PageType::Concept,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    wiki.save_index(&idx).unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(snippet.contains("## dark-matter Wiki"));
    assert!(snippet.contains("### entity\n"));
    assert!(snippet.contains("### concept\n"));
    assert!(!snippet.contains("### summary"));
    assert!(!snippet.contains("### synthesis"));
    assert!(snippet.contains("- Compaction: Three-stage context trimming."));
    assert!(snippet.contains("- Session: Persistent conversation state."));
    assert!(snippet.contains("- Wiki pattern: Karpathy LLM wiki."));
    // No truncation tail on a small index.
    assert!(!snippet.contains("more page(s) not shown"));
}

#[test]
fn context_snippet_uses_title_only_format_no_paths() {
    // Directive target #2 (C19): the snippet is a TOC of entity names
    // + one-liners, NOT a path dump. Regression guard: a refactor
    // that reintroduces `.dm/wiki/...` per-line paths trips this test.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.save_index(&WikiIndex {
        entries: vec![IndexEntry {
            title: "Compaction".to_string(),
            path: "entities/compaction.md".to_string(),
            one_liner: "Three-stage context trimming.".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    })
    .unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(
        snippet.contains("- Compaction: Three-stage context trimming."),
        "title-based per-line format: {}",
        snippet
    );
    assert!(
        !snippet.contains(".dm/wiki/entities/compaction.md"),
        "no per-line path dump: {}",
        snippet
    );
    // The truncation tail at the bottom of the snippet still names the
    // catalog file by path — that's where the model reads the full
    // index when truncated, and stays unchanged.
}

#[test]
fn context_snippet_preamble_points_at_wiki_tools() {
    // The session-start snippet must direct the model to the
    // wiki_lookup / wiki_search tools introduced in C1, not the
    // pre-C1 `file_read` advice. Regression guard: a refactor that
    // reintroduces "file_read tool" in the preamble (or drops the
    // wiki_search reference) would silently suppress the wiki
    // tooling we built across C1-C11.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Probe".to_string(),
            path: "entities/probe.md".to_string(),
            one_liner: "test".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(
        snippet.contains("wiki_search"),
        "must mention wiki_search: {}",
        snippet
    );
    assert!(
        snippet.contains("wiki_lookup"),
        "must mention wiki_lookup: {}",
        snippet
    );
    assert!(
        snippet.contains("[wiki-drift]"),
        "must mention drift marker: {}",
        snippet
    );
    assert!(
        !snippet.contains("file_read tool"),
        "must NOT advise file_read for wiki pages: {}",
        snippet
    );
}

// Identity threading regression — the snippet's `## <name> Wiki`
// header must reflect the host project's `.dm/identity.toml` (when
// present), not the kernel default. Locks down the cycle-3 wiring
// and catches the cycle-13 regression where `load_at(self.root())`
// looked up the wrong path and silently fell back to kernel mode.
#[test]
fn context_snippet_header_uses_host_identity_when_present() {
    let tmp = TempDir::new().unwrap();
    // Write `.dm/identity.toml` BEFORE opening the wiki — `load_at`
    // reads from project_root which is `tmp.path()`.
    let dm = tmp.path().join(".dm");
    std::fs::create_dir_all(&dm).unwrap();
    std::fs::write(
        dm.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"finance-app\"\n",
    )
    .unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.save_index(&WikiIndex {
        entries: vec![IndexEntry {
            title: "Marker".into(),
            path: "entities/marker.md".into(),
            one_liner: "marker".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    })
    .unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(
        snippet.starts_with("## finance-app Wiki"),
        "header must use host project name, got: {}",
        &snippet[..snippet.len().min(80)],
    );
    assert!(
        !snippet.starts_with("## dark-matter Wiki"),
        "host mode must not render canonical name",
    );
}

#[test]
fn context_snippet_header_defaults_to_kernel_when_no_identity_file() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.save_index(&WikiIndex {
        entries: vec![IndexEntry {
            title: "Marker".into(),
            path: "entities/marker.md".into(),
            one_liner: "marker".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    })
    .unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(
        snippet.starts_with("## dark-matter Wiki"),
        "kernel default expected, got: {}",
        &snippet[..snippet.len().min(80)],
    );
}

#[test]
fn context_snippet_for_kernel_identity_renders_flat_per_category() {
    // Kernel-mode identity must keep the legacy flat list — host pages
    // (if any are mistakenly present in a kernel-mode project) appear
    // inline with kernel pages, no `---` separator inserted, no per-page
    // disk reads to recover layer.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page_with_layer(
        &wiki,
        "concepts/host-a.md",
        "Host A",
        PageType::Concept,
        crate::wiki::Layer::Host,
        "host body",
    );
    add_page_with_layer(
        &wiki,
        "concepts/kernel-b.md",
        "Kernel B",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "kernel body",
    );
    let snippet = wiki
        .context_snippet_for(&crate::identity::Identity::default_kernel())
        .expect("snippet");
    assert!(
        !snippet.contains("\n---\n"),
        "kernel mode must not insert layer separator. snippet:\n{snippet}"
    );
    assert!(snippet.contains("- Host A: Host A\n"));
    assert!(snippet.contains("- Kernel B: Kernel B\n"));
}

#[test]
fn context_snippet_for_host_identity_stratifies_host_then_kernel_with_separator() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Add in interleaved order so we know stratification, not insertion
    // order, drives the result.
    add_page_with_layer(
        &wiki,
        "concepts/kernel-a.md",
        "Kernel A",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "kernel body",
    );
    add_page_with_layer(
        &wiki,
        "concepts/host-b.md",
        "Host B",
        PageType::Concept,
        crate::wiki::Layer::Host,
        "host body",
    );
    add_page_with_layer(
        &wiki,
        "concepts/kernel-c.md",
        "Kernel C",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "kernel body",
    );
    add_page_with_layer(
        &wiki,
        "concepts/host-d.md",
        "Host D",
        PageType::Concept,
        crate::wiki::Layer::Host,
        "host body",
    );
    let identity = crate::identity::Identity {
        mode: crate::identity::Mode::Host,
        host_project: Some("finance-app".to_string()),
        canonical_dm_revision: None,
        canonical_dm_repo: None,
        source: None,
    };
    let snippet = wiki.context_snippet_for(&identity).expect("snippet");

    let host_b = snippet.find("- Host B: Host B").expect("host B in snippet");
    let host_d = snippet.find("- Host D: Host D").expect("host D in snippet");
    let sep = snippet.find("\n---\n").expect("layer separator");
    let kernel_a = snippet.find("- Kernel A: Kernel A").expect("kernel A");
    let kernel_c = snippet.find("- Kernel C: Kernel C").expect("kernel C");
    assert!(
        host_b < sep && host_d < sep,
        "host pages must precede separator. snippet:\n{snippet}"
    );
    assert!(
        sep < kernel_a && sep < kernel_c,
        "kernel pages must follow separator. snippet:\n{snippet}"
    );
}

#[test]
fn context_snippet_for_host_identity_omits_separator_when_one_layer_empty() {
    // Host mode but every page is kernel-layer (matches a freshly spawned
    // project before any host content is added). The snippet must list
    // the kernel entries flat with no dangling `---`.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page_with_layer(
        &wiki,
        "concepts/kernel-only.md",
        "Kernel Only",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "kernel body",
    );
    let identity = crate::identity::Identity {
        mode: crate::identity::Mode::Host,
        host_project: Some("finance-app".to_string()),
        canonical_dm_revision: None,
        canonical_dm_repo: None,
        source: None,
    };
    let snippet = wiki.context_snippet_for(&identity).expect("snippet");
    assert!(
        !snippet.contains("\n---\n"),
        "single-layer category must not emit separator. snippet:\n{snippet}"
    );
    assert!(snippet.contains("- Kernel Only: Kernel Only\n"));
}

#[test]
fn context_snippet_size_bounded() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let entries = (0..500)
        .map(|i| IndexEntry {
            title: format!("Page {}", i),
            path: format!("entities/page-{:04}.md", i),
            one_liner: "x".repeat(40),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        })
        .collect();
    wiki.save_index(&WikiIndex { entries }).unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    // Body is capped by CONTEXT_SNIPPET_MAX_BYTES; the truncation tail is
    // appended unconditionally, so allow ~200 bytes of slack for it.
    assert!(
        snippet.len() <= CONTEXT_SNIPPET_MAX_BYTES + 200,
        "snippet unexpectedly large: {} bytes",
        snippet.len()
    );
    assert!(snippet.contains("more page(s) not shown"));
    assert!(snippet.contains(".dm/wiki/index.md"));
}

#[test]
fn index_roundtrip_sanitizes_newlines_in_one_liner() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Compaction".into(),
            path: "concepts/compaction.md".into(),
            one_liner: "line one\nline two\nline three".into(),
            category: PageType::Concept,
            last_updated: None,
            outcome: None,
        }],
    };
    let md = idx.to_markdown();
    let parsed = WikiIndex::parse(&md);
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].one_liner, "line one line two line three");
}

#[test]
fn index_roundtrip_sanitizes_cr_crlf_and_tab_in_one_liner() {
    for raw in ["a\rb", "a\r\nb", "a\tb"] {
        let idx = WikiIndex {
            entries: vec![IndexEntry {
                title: "t".into(),
                path: "entities/t.md".into(),
                one_liner: raw.into(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            }],
        };
        let parsed = WikiIndex::parse(&idx.to_markdown());
        assert_eq!(parsed.entries.len(), 1, "dropped entry for input {:?}", raw);
        assert_eq!(
            parsed.entries[0].one_liner, "a b",
            "bad round-trip for {:?}",
            raw
        );
    }
}

#[test]
fn index_roundtrip_sanitizes_newline_in_title() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "split\ntitle".into(),
            path: "entities/foo.md".into(),
            one_liner: "notes".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].title, "split title");
}

#[test]
fn index_roundtrip_lossless_for_single_line_chars() {
    let busy = "()[]*_~`!@#$%^&*{}|;:,./<>?+= —";
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "title".into(),
            path: "entities/foo.md".into(),
            one_liner: busy.into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].one_liner, busy);
}

#[test]
fn index_roundtrip_collapses_multiple_spaces() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "t".into(),
            path: "entities/t.md".into(),
            one_liner: "a    b".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].one_liner, "a b");
}

#[test]
fn sanitize_single_line_empty_stays_empty() {
    assert_eq!(sanitize_single_line(""), "");
}

#[test]
fn sanitize_single_line_only_whitespace_returns_empty() {
    assert_eq!(sanitize_single_line("   \n\t  "), "");
}

#[test]
fn sanitize_single_line_collapses_unicode_whitespace() {
    // NBSP (U+00A0), line separator (U+2028), paragraph separator
    // (U+2029), narrow NBSP (U+202F), ideographic space (U+3000) are all
    // Unicode whitespace per char::is_whitespace and must collapse.
    assert_eq!(sanitize_single_line("a\u{00A0}b"), "a b");
    assert_eq!(sanitize_single_line("a\u{2028}b"), "a b");
    assert_eq!(sanitize_single_line("a\u{2029}b"), "a b");
    assert_eq!(sanitize_single_line("a\u{202F}b"), "a b");
    assert_eq!(sanitize_single_line("a\u{3000}b"), "a b");
    // Mix of ASCII and Unicode whitespace collapses to one space.
    assert_eq!(sanitize_single_line("a \u{00A0}\t\u{2028}b"), "a b");
    // Leading and trailing Unicode whitespace is stripped.
    assert_eq!(sanitize_single_line("\u{00A0}foo\u{2029}"), "foo");
}

#[test]
fn index_roundtrip_sanitizes_unicode_whitespace_in_one_liner() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "NBSP".to_string(),
            path: "entities/nbsp.md".to_string(),
            // Paragraph separator, NBSP, line separator all embedded.
            one_liner: "foo\u{2029}bar\u{00A0}baz\u{2028}qux".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].one_liner, "foo bar baz qux");
}

// Happy path: a `]` inside a title (without immediately following `(`)
// roundtrips losslessly — the parser keys on the literal `](` bigram.
// This is a positive result worth guarding against a regression.
#[test]
fn index_roundtrip_title_with_bare_closing_bracket_is_lossless() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "has ] bracket".to_string(),
            path: "entities/b.md".to_string(),
            one_liner: "summary".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].title, "has ] bracket");
    assert_eq!(parsed.entries[0].path, "entities/b.md");
    assert_eq!(parsed.entries[0].one_liner, "summary");
}

// Known gap (not a regression; documented for a future builder): titles
// literally containing `](` and paths literally containing `)` break the
// parser because `WikiIndex::to_markdown` only sanitizes whitespace, not
// markdown-link-breaking characters. These tests record current behavior
// so a future escaping fix will know which assertions to flip.
#[test]
fn index_roundtrip_known_gap_title_with_literal_link_close() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "foo](bar".to_string(),
            path: "entities/b.md".to_string(),
            one_liner: "summary".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    // Parser cuts title at the first `](`, so title ends up as "foo".
    // Assert this directly so the test fails loudly if a future fix
    // starts preserving the original title.
    assert_eq!(parsed.entries.len(), 1);
    assert_ne!(
            parsed.entries[0].title, "foo](bar",
            "title with literal '](' now roundtrips losslessly — escaping has landed, update this assertion"
        );
}

#[test]
fn index_roundtrip_path_with_close_paren() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "p".to_string(),
            path: "entities/weird) name.md".to_string(),
            one_liner: "summary".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, "entities/weird) name.md");
    assert_eq!(parsed.entries[0].one_liner, "summary");
}

#[test]
fn index_roundtrip_path_with_backslash() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Raw string".into(),
            path: r"entities\backslash.md".into(),
            one_liner: "Windows-style path".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, r"entities\backslash.md");
}

#[test]
fn index_roundtrip_path_with_multiple_parens() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "t".into(),
            path: "entities/foo)bar)baz.md".into(),
            one_liner: "three closers".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, "entities/foo)bar)baz.md");
}

#[test]
fn index_roundtrip_path_with_backslash_close_paren_mix() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "mix".into(),
            path: r"entities/a\b)c.md".into(),
            one_liner: "mix".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, r"entities/a\b)c.md");
}

#[test]
fn index_roundtrip_path_with_multibyte_char() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "cafe".into(),
            path: "entities/café)bar.md".into(),
            one_liner: "multi-byte".into(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    let parsed = WikiIndex::parse(&idx.to_markdown());
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, "entities/café)bar.md");
}

#[test]
fn escape_path_order_invariance() {
    assert_eq!(escape_path(r"a\b"), r"a\\b");
    assert_eq!(escape_path("a)b"), r"a\)b");
    assert_eq!(escape_path(r"a\)b"), r"a\\\)b");
}

#[test]
fn scan_path_unknown_escape_passes_through() {
    let (path, tail) = scan_path(r"a\xb)tail").expect("terminator");
    assert_eq!(path, r"a\xb");
    assert_eq!(tail, "tail");
}

#[test]
fn scan_path_no_terminator_returns_none() {
    assert!(scan_path("nothing-here").is_none());
}

#[test]
fn escape_path_stress_all_parens() {
    // Path is only `)` characters — every one must be escaped and
    // decoded.
    let escaped = escape_path(")))");
    assert_eq!(escaped, r"\)\)\)");
    let serialized = format!("{})tail", escaped);
    let (decoded, tail) = scan_path(&serialized).unwrap();
    assert_eq!(decoded, ")))");
    assert_eq!(tail, "tail");
}

#[test]
fn escape_path_stress_all_backslashes() {
    // Path is only backslashes — escape doubles each, decoder halves.
    let input = r"\\\\"; // 4 backslashes
    let escaped = escape_path(input);
    assert_eq!(escaped, r"\\\\\\\\"); // 8 backslashes
    let serialized = format!("{})tail", escaped);
    let (decoded, tail) = scan_path(&serialized).unwrap();
    assert_eq!(decoded, input);
    assert_eq!(tail, "tail");
}

#[test]
fn escape_path_empty_input() {
    assert_eq!(escape_path(""), "");
    // `)` alone is a valid terminator: empty path, empty tail.
    let (decoded, tail) = scan_path(")").unwrap();
    assert_eq!(decoded, "");
    assert_eq!(tail, "");
}

#[test]
fn index_roundtrip_path_empty_is_allowed() {
    // An empty `path` is a degenerate but non-crashing case. Confirms
    // save_index + load_index don't drop the entry and don't panic.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "empty-path".to_string(),
            path: String::new(),
            one_liner: "tail after empty path survives".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, "");
    assert_eq!(
        parsed.entries[0].one_liner,
        "tail after empty path survives"
    );
}

#[test]
fn index_roundtrip_one_liner_with_newline_then_paren_survives() {
    // Exactly the builder's flagged pipeline:
    // raw  = "ab\n)cd"
    // sanitized (via to_markdown) = "ab )cd"     (newline → space)
    // escaped in path? No — this is the one_liner, only collapsed.
    // For a path, same input:
    // path sanitize = "ab )cd" → escape → r"ab \)cd" → parse → "ab )cd"
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "x".to_string(),
            path: "entities/ab\n)cd.md".to_string(),
            one_liner: "ab\n)cd".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    // Path: sanitize collapses `\n` to space, then escape + parse is
    // lossless w.r.t. the collapsed form.
    assert_eq!(parsed.entries[0].path, "entities/ab )cd.md");
    // One-liner: same sanitize, no escape needed for `)` (not a
    // structural character outside the path slot).
    assert_eq!(parsed.entries[0].one_liner, "ab )cd");
}

#[test]
fn index_roundtrip_path_trailing_backslash_survives() {
    // Odd-backslash-count edge case: a path that ends in a single `\`
    // must still roundtrip because escape doubles it first.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "t".to_string(),
            path: r"entities\trailing\".to_string(),
            one_liner: "summary".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let parsed = wiki.load_index().unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.entries[0].path, r"entities\trailing\");
    assert_eq!(parsed.entries[0].one_liner, "summary");
}

#[test]
fn context_snippet_unicode_whitespace_lands_clean() {
    // End-to-end: a one-liner with NBSP + line sep survives save → load
    // → snippet as a single-line entry with normal spaces, so the
    // model-facing prompt stays well-formed markdown.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.save_index(&WikiIndex {
        entries: vec![IndexEntry {
            title: "Unicode".to_string(),
            path: "entities/u.md".to_string(),
            one_liner: "alpha\u{00A0}beta\u{2028}gamma".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    })
    .unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    assert!(snippet.contains("- Unicode: alpha beta gamma"));
    // No raw Unicode whitespace should leak into the snippet.
    assert!(!snippet.contains('\u{00A0}'));
    assert!(!snippet.contains('\u{2028}'));
    assert!(!snippet.contains('\u{2029}'));
}

#[test]
fn context_snippet_skips_unreadable_index() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Start from a valid (empty) index, then remove it so load_index
    // returns an empty WikiIndex via its NotFound fallback.
    wiki.save_index(&WikiIndex::default()).unwrap();
    fs::remove_file(wiki.root().join("index.md")).unwrap();
    assert!(wiki.context_snippet().is_none());
}

#[test]
fn context_snippet_survives_pathological_entries() {
    // The goal: the snippet should never panic, must remain well-formed
    // markdown (every list line starts with `- ` on its own line), and
    // must carry every one_liner through — even weird ones. If a future
    // auto-ingest writes entries like these, the prompt still parses.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let long_title = "T".repeat(400);
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "path with spaces".to_string(),
                path: "entities/spaces in path.md".to_string(),
                one_liner: "has spaces".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "newline in one-liner".to_string(),
                path: "entities/nl.md".to_string(),
                one_liner: "line one\nline two".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: long_title.clone(),
                path: "entities/huge-title.md".to_string(),
                one_liner: "short".to_string(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "backticks".to_string(),
                path: "entities/tick`s.md".to_string(),
                one_liner: "one_liner with a ` inside".to_string(),
                category: PageType::Concept,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    wiki.save_index(&idx).unwrap();
    let snippet = wiki.context_snippet().expect("snippet");
    // Sanity: the header is present and the snippet did not panic.
    assert!(snippet.starts_with("## dark-matter Wiki\n"));
    // Titles without newlines survive the save → load → snippet trip.
    assert!(snippet.contains("- path with spaces: has spaces"));
    assert!(snippet.contains("- newline in one-liner:"));
    assert!(snippet.contains("- backticks: one_liner with a ` inside"));
    // `WikiIndex::to_markdown` now sanitizes `one_liner` whitespace to a
    // single space so newlines cannot split the record. Both halves of a
    // multiline one-liner survive the save → load → snippet trip, joined
    // by a space.
    assert!(snippet.contains("line one line two"));
    // The long title IS emitted now that the helper renders titles in
    // the per-line format (400 chars fits well under the 4 KB cap).
    assert!(snippet.contains(&long_title));
    // The tail line for truncation should NOT be present — total size
    // is well under the cap.
    assert!(!snippet.contains("more page(s) not shown"));
}

#[test]
fn context_snippet_matches_save_index_roundtrip() {
    // Regression guard: an index that roundtrips through save_index /
    // load_index must still produce a snippet containing its entries.
    // This catches subtle formatting drift between `to_markdown` and
    // `parse` that would silently break the splice.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "roundtrip".to_string(),
            path: "entities/roundtrip.md".to_string(),
            one_liner: "ensure format stability.".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        }],
    };
    wiki.save_index(&idx).unwrap();
    let snippet = wiki.context_snippet().expect("snippet after roundtrip");
    assert!(snippet.contains("- roundtrip: ensure format stability."));
}

#[test]
fn project_summary_snippet_absent_returns_none() {
    // No summaries/project.md written → loader is a no-op. Callers rely
    // on this to skip the `<wiki_summary>` block without branching.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    assert!(wiki.project_summary_snippet(4096).is_none());
}

#[test]
fn project_summary_snippet_returns_body_without_frontmatter() {
    // After write_project_summary runs, the loader returns the body
    // only — frontmatter must never reach the system prompt (it's
    // noise for the model).
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/session.md",
        "session",
        "src/session.rs",
        Some(EntityKind::Struct),
        Some("persistent conversation state"),
        vec![],
    );
    wiki.write_project_summary().unwrap();
    let snippet = wiki.project_summary_snippet(8192).expect("snippet");
    assert!(snippet.contains("# Project"));
    assert!(
        !snippet.starts_with("---\n"),
        "frontmatter must be stripped"
    );
    assert!(
        !snippet.contains("last_updated:"),
        "frontmatter fields must be stripped"
    );
}

#[test]
fn project_summary_snippet_truncates_at_line_boundary() {
    // Over-budget bodies are truncated at a `\n` boundary with a
    // trailing marker so the model can tell the view is partial.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let body = (0..200)
        .map(|i| format!("- line {:03} with some filler text\n", i))
        .collect::<String>();
    let page = WikiPage {
        title: "Project".to_string(),
        page_type: PageType::Summary,
        layer: crate::wiki::Layer::Kernel,
        entity_kind: None,
        sources: Vec::new(),
        last_updated: "2026-04-18T00:00:00Z".to_string(),
        body,
        purpose: None,
        key_exports: Vec::new(),
        dependencies: Vec::new(),
        outcome: None,
        scope: vec![],
    };
    fs::create_dir_all(wiki.root().join("summaries")).unwrap();
    fs::write(wiki.root().join("summaries/project.md"), page.to_markdown()).unwrap();

    let budget = 512;
    let snippet = wiki.project_summary_snippet(budget).expect("snippet");
    assert!(snippet.ends_with("[...truncated]"));
    // Body before the marker ends at a newline — no mid-line cut.
    let before = snippet
        .trim_end_matches("[...truncated]")
        .trim_end_matches('\n');
    assert!(
        before.is_empty() || !before.ends_with(' '),
        "truncation should land at a line boundary: {:?}",
        before
    );
    // Snippet stays near the budget (marker is appended after the cut).
    assert!(
        snippet.len() <= budget + "\n[...truncated]".len(),
        "snippet unexpectedly large: {} bytes (budget {})",
        snippet.len(),
        budget
    );
}

#[test]
fn project_summary_snippet_malformed_page_returns_none() {
    // A file at summaries/project.md that doesn't parse as a WikiPage
    // (e.g., missing frontmatter) must not crash — return None so the
    // builder silently omits the block.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    fs::create_dir_all(wiki.root().join("summaries")).unwrap();
    fs::write(
        wiki.root().join("summaries/project.md"),
        "no frontmatter here, just raw text",
    )
    .unwrap();
    assert!(wiki.project_summary_snippet(4096).is_none());
}

// ── auto-ingest ───────────────────────────────────────────────────────

/// Serializes tests that touch `DM_WIKI_AUTO_INGEST` so the process-wide
/// env var doesn't leak between threads.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}
impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prev }
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var(self.key, v),
            None => std::env::remove_var(self.key),
        }
    }
}

#[tokio::test]
async fn wait_for_ingest_log_marker_returns_true_when_marker_present() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    wiki.log().append("ingest", "probe.txt").unwrap();

    let hit = wait_for_ingest_log_marker(&proj, "ingest | probe.txt").await;
    assert!(
        hit,
        "helper should observe the marker that was written before the poll"
    );
}

#[tokio::test]
async fn wait_for_ingest_log_marker_times_out_when_marker_absent() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let _ = Wiki::open(&proj).unwrap();

    let start = std::time::Instant::now();
    let hit = wait_for_ingest_log_marker(&proj, "ingest | missing.txt").await;
    let elapsed = start.elapsed();

    assert!(!hit, "helper must return false when marker never appears");
    assert!(
        elapsed >= std::time::Duration::from_millis(900),
        "timeout should poll ~20×50ms before giving up, got {:?}",
        elapsed
    );
}

#[test]
fn ingest_file_writes_page_updates_index_appends_log() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("foo.rs");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "hello world").unwrap();

    let outcome = wiki.ingest_file(&proj, &file, "hello world").unwrap();
    assert_eq!(
        outcome,
        IngestOutcome::Ingested {
            page_rel: "entities/src_foo_rs.md".into()
        }
    );

    let page_path = wiki.root().join("entities/src_foo_rs.md");
    assert!(page_path.is_file(), "page not written");
    let text = fs::read_to_string(&page_path).unwrap();
    let page = WikiPage::parse(&text).expect("page parses");
    assert_eq!(page.page_type, PageType::Entity);
    assert_eq!(page.title, "src/foo.rs");

    let idx = wiki.load_index().unwrap();
    assert_eq!(idx.entries.len(), 1);
    assert_eq!(idx.entries[0].path, "entities/src_foo_rs.md");

    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert!(log.contains("ingest | src/foo.rs"), "log: {:?}", log);
}

#[test]
fn ingest_file_is_idempotent_on_same_content() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("a.txt");
    fs::write(&file, "x").unwrap();

    let first = wiki.ingest_file(&proj, &file, "x").unwrap();
    let second = wiki.ingest_file(&proj, &file, "x").unwrap();
    assert!(matches!(first, IngestOutcome::Ingested { .. }));
    assert_eq!(
        second,
        IngestOutcome::Skipped(SkipReason::UnchangedSinceLast)
    );

    let idx = wiki.load_index().unwrap();
    assert_eq!(idx.entries.len(), 1);
    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert_eq!(log.matches("ingest | a.txt").count(), 1);
}

#[test]
fn ingest_file_updates_existing_entry_on_content_change() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("b.txt");
    fs::write(&file, "a").unwrap();

    let first = wiki.ingest_file(&proj, &file, "a").unwrap();
    let second = wiki.ingest_file(&proj, &file, "b").unwrap();
    assert!(matches!(first, IngestOutcome::Ingested { .. }));
    assert!(matches!(second, IngestOutcome::Ingested { .. }));

    let idx = wiki.load_index().unwrap();
    assert_eq!(idx.entries.len(), 1, "upsert, not append");
    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert_eq!(log.matches("ingest | b.txt").count(), 2);
}

#[test]
fn ingest_file_skips_paths_inside_wiki() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let inner = wiki.root().join("entities/foo.md");
    fs::create_dir_all(inner.parent().unwrap()).unwrap();
    fs::write(&inner, "ignored").unwrap();

    let outcome = wiki.ingest_file(&proj, &inner, "ignored").unwrap();
    assert_eq!(outcome, IngestOutcome::Skipped(SkipReason::InsideWikiDir));
    let idx = wiki.load_index().unwrap();
    assert!(idx.entries.is_empty());
}

#[test]
fn ingest_file_skips_paths_outside_project() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let proj_tmp = TempDir::new().unwrap();
    let outside_tmp = TempDir::new().unwrap();
    let proj = proj_tmp.path().canonicalize().unwrap();
    let outside = outside_tmp.path().canonicalize().unwrap().join("x.txt");
    fs::write(&outside, "hi").unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let outcome = wiki.ingest_file(&proj, &outside, "hi").unwrap();
    assert_eq!(outcome, IngestOutcome::Skipped(SkipReason::OutsideProject));
    let idx = wiki.load_index().unwrap();
    assert!(idx.entries.is_empty());
}

#[test]
fn ingest_file_respects_env_disable() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", "0");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("c.txt");
    fs::write(&file, "nope").unwrap();

    let outcome = wiki.ingest_file(&proj, &file, "nope").unwrap();
    assert_eq!(outcome, IngestOutcome::Skipped(SkipReason::Disabled));
}

#[test]
fn entity_page_rel_slug_rules() {
    assert_eq!(
        entity_page_rel("src/foo/bar.rs"),
        "entities/src_foo_bar_rs.md"
    );
    assert_eq!(entity_page_rel("README"), "entities/readme.md");
    assert_eq!(entity_page_rel("a//b//c"), "entities/a_b_c.md");
    assert_eq!(entity_page_rel(""), "entities/unnamed.md");
    assert_eq!(entity_page_rel("---"), "entities/unnamed.md");
    let unicode = entity_page_rel("Üñîçødé.txt");
    assert!(unicode.starts_with("entities/"));
    assert!(unicode.ends_with(".md"));
}

#[test]
fn content_hash_changes_with_content() {
    assert_ne!(content_hash("a"), content_hash("b"));
    assert_eq!(content_hash("a"), content_hash("a"));
}

#[test]
fn extract_rust_preview_lists_top_level_functions() {
    let src = "\
pub fn foo(x: u32) -> u32 { x + 1 }
fn helper() {}
pub async fn fetch_it() {}
pub(crate) unsafe fn peek() {}
";
    let out = extract_content_preview("src/lib.rs", src);
    assert!(out.contains("## Items"), "items heading expected: {}", out);
    assert!(out.contains("`pub fn foo`"), "missing pub fn foo: {}", out);
    assert!(out.contains("`fn helper`"), "missing fn helper: {}", out);
    assert!(
        out.contains("`pub fn fetch_it`"),
        "async-before-fn should still emit signature: {}",
        out
    );
    assert!(
        out.contains("`pub(crate) fn peek`"),
        "pub(crate)+unsafe preserved: {}",
        out
    );
}

#[test]
fn extract_rust_preview_lists_structs_and_enums() {
    let src = "\
pub struct Foo { x: u32 }
enum Color { Red, Green, Blue }
pub(super) struct Bar;
";
    let out = extract_content_preview("src/types.rs", src);
    assert!(out.contains("`pub struct Foo`"), "pub struct Foo: {}", out);
    assert!(out.contains("`enum Color`"), "enum Color: {}", out);
    assert!(
        out.contains("`pub(super) struct Bar`"),
        "visibility kept: {}",
        out
    );
}

#[test]
fn extract_rust_preview_lists_traits_and_impls() {
    let src = "\
pub trait Widget { fn render(&self); }
impl Widget for Foo { fn render(&self) {} }
pub type Result<T> = std::result::Result<T, anyhow::Error>;
const TIMEOUT_MS: u64 = 5000;
pub static GLOBAL_COUNTER: u32 = 0;
mod inner { pub fn x() {} }
";
    let out = extract_content_preview("src/traits.rs", src);
    assert!(out.contains("`pub trait Widget`"), "trait: {}", out);
    assert!(out.contains("`impl Widget`"), "impl: {}", out);
    assert!(out.contains("`pub type Result`"), "type alias: {}", out);
    assert!(out.contains("`const TIMEOUT_MS`"), "const: {}", out);
    assert!(
        out.contains("`pub static GLOBAL_COUNTER`"),
        "static: {}",
        out
    );
    assert!(out.contains("`mod inner`"), "mod: {}", out);
}

#[test]
fn extract_rust_preview_captures_module_docs() {
    let src = "\
//! # Widget library
//!
//! This module exposes the core Widget trait and a handful of
//! concrete implementations.

pub trait Widget {}
";
    let out = extract_content_preview("src/lib.rs", src);
    assert!(
        out.contains("## Module docs"),
        "module docs heading: {}",
        out
    );
    assert!(out.contains("Widget library"), "doc body captured: {}", out);
    assert!(
        out.contains("concrete implementations."),
        "multi-line doc captured: {}",
        out
    );
    // Items section still appears.
    assert!(
        out.contains("`pub trait Widget`"),
        "items after docs: {}",
        out
    );
}

#[test]
fn extract_preview_non_rust_file_uses_generic_fenced_preview() {
    let src = "# My Project\n\nHello world.\n\n## Usage\n\nRun `foo`.\n";
    let out = extract_content_preview("README.md", src);
    assert!(
        out.contains("## Preview"),
        "generic preview heading: {}",
        out
    );
    assert!(out.contains("```"), "fenced block present: {}", out);
    assert!(out.contains("My Project"), "content preserved: {}", out);
    assert!(
        !out.contains("## Items"),
        "non-rust should not emit Items heading: {}",
        out
    );
}

#[test]
fn extract_preview_empty_content_returns_empty_string() {
    assert_eq!(extract_content_preview("src/lib.rs", ""), "");
    assert_eq!(extract_content_preview("anything.txt", "   \n\n\t"), "");
}

#[test]
fn extract_preview_bounded_by_max_bytes() {
    // Build a source with many pub fn declarations so the Items
    // section is forced past the cap, and verify the truncation
    // marker appears + size stays within a small margin of the cap.
    let mut src = String::new();
    for i in 0..600 {
        writeln!(
            src,
            "pub fn function_with_a_sufficiently_long_name_{:04}() {{}}",
            i
        )
        .expect("write to String never fails");
    }
    let out = extract_content_preview("src/big.rs", &src);
    assert!(
        out.len() <= PREVIEW_MAX_BYTES + 200,
        "preview must stay near cap ({}): got {} bytes",
        PREVIEW_MAX_BYTES,
        out.len()
    );
    assert!(
        out.contains("preview capped"),
        "truncation marker expected: len={}",
        out.len()
    );
}

#[test]
fn ingest_file_writes_real_preview_not_stub() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_dir = proj.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let file = src_dir.join("widget.rs");
    let content = "//! Widget module docs.\n\npub fn build() -> u32 { 42 }\n";
    std::fs::write(&file, content).unwrap();

    let outcome = wiki.ingest_file(&proj, &file, content).unwrap();
    let page_rel = match outcome {
        IngestOutcome::Ingested { page_rel } => page_rel,
        other => panic!("expected Ingested, got {:?}", other),
    };
    let page = wiki.read_page(&page_rel).unwrap();
    assert!(
        !page.body.contains("Stub page"),
        "body still has stub marker: {}",
        page.body
    );
    assert!(
        page.body.contains("Widget module docs."),
        "module docs must be spliced into body: {}",
        page.body
    );
    assert!(
        page.body.contains("`pub fn build`"),
        "top-level pub fn must be listed: {}",
        page.body
    );
}

// ─── Cycle 36: inject_wiki_links ──────────────────────────────────────

#[test]
fn inject_wiki_links_wraps_first_occurrence_only() {
    let body = "src/a.rs and later src/a.rs again";
    let srcs = vec![("entity/a.md".to_string(), "src/a.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, "[[src/a.rs]] and later src/a.rs again");
}

#[test]
fn inject_wiki_links_skips_self_source() {
    let body = "this mentions src/x.rs in its preview";
    let srcs = vec![("entity/x.md".to_string(), "src/x.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, body, "self-source must never be wrapped");
}

#[test]
fn inject_wiki_links_idempotent_on_already_wrapped() {
    let body = "see [[src/a.rs]] for details";
    let srcs = vec![("entity/a.md".to_string(), "src/a.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, body, "already-wrapped occurrence must stay untouched");
}

#[test]
fn inject_wiki_links_handles_empty_entity_srcs() {
    let body = "nothing to link here";
    let out = inject_wiki_links(body, &[], "src/x.rs");
    assert_eq!(out, body);
}

#[test]
fn inject_wiki_links_multiple_different_sources() {
    let body = "first src/a.rs then src/b.rs closing";
    let srcs = vec![
        ("entity/a.md".to_string(), "src/a.rs".to_string()),
        ("entity/b.md".to_string(), "src/b.rs".to_string()),
    ];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, "first [[src/a.rs]] then [[src/b.rs]] closing");
}

// ─── Cycle 37: inject_wiki_links UTF-8 safety ─────────────────────────

#[test]
fn inject_wiki_links_handles_multibyte_char_before_src() {
    // `€` is 3 bytes; source path starts at byte offset 3. The old
    // `&out[idx-2..idx]` implementation sliced mid-codepoint and
    // panicked. `ends_with` is boundary-safe.
    let body = "€src/a.rs";
    let srcs = vec![("entity/a.md".to_string(), "src/a.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, "€[[src/a.rs]]");
}

#[test]
fn inject_wiki_links_handles_cjk_char_before_src() {
    let body = "日src/a.rs";
    let srcs = vec![("entity/a.md".to_string(), "src/a.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, "日[[src/a.rs]]");
}

#[test]
fn inject_wiki_links_handles_emoji_before_src() {
    let body = "🦀src/a.rs";
    let srcs = vec![("entity/a.md".to_string(), "src/a.rs".to_string())];
    let out = inject_wiki_links(body, &srcs, "src/x.rs");
    assert_eq!(out, "🦀[[src/a.rs]]");
}

// ─── Cycle 37: parse_page_item_names ──────────────────────────────────

#[test]
fn parse_page_item_names_returns_set_from_items_section() {
    let body = "# title\n\n## Items\n\n- `pub fn foo`\n- `struct Bar`\n";
    let set = parse_page_item_names(body);
    assert!(set.contains("foo"), "expected foo: {:?}", set);
    assert!(set.contains("Bar"), "expected Bar: {:?}", set);
    assert_eq!(set.len(), 2);
}

#[test]
fn parse_page_item_names_ignores_truncation_marker() {
    let body = "## Items\n\n- `pub fn foo`\n- *(+3 more; preview capped)*\n";
    let set = parse_page_item_names(body);
    assert!(set.contains("foo"));
    assert!(
        !set.iter()
            .any(|n| n.contains("more") || n.contains("capped")),
        "truncation marker leaked in: {:?}",
        set,
    );
    assert_eq!(set.len(), 1);
}

#[test]
fn parse_page_item_names_empty_when_no_items_section() {
    let body = "# title\n\n## Module docs\n\nHello world.\n";
    let set = parse_page_item_names(body);
    assert!(set.is_empty(), "unexpected items: {:?}", set);
}

#[test]
fn parse_page_item_names_stops_at_next_heading() {
    let body = "## Items\n\n- `fn a`\n\n## Other\n\n- `fn b`\n";
    let set = parse_page_item_names(body);
    assert!(set.contains("a"));
    assert!(!set.contains("b"), "crossed heading: {:?}", set);
    assert_eq!(set.len(), 1);
}

#[test]
fn parse_page_item_names_handles_pub_crate_vis() {
    let body = "## Items\n\n- `pub(crate) fn helper`\n- `pub(super) struct Wrap`\n";
    let set = parse_page_item_names(body);
    assert!(set.contains("helper"), "helper: {:?}", set);
    assert!(set.contains("Wrap"), "Wrap: {:?}", set);
}

// ─── Cycle 36: ingest_file wiki-link injection ────────────────────────

#[test]
fn ingest_file_injects_wiki_links_for_known_sources() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_dir = proj.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // First ingest: the referenced helper file. Establishes the entity
    // page + index entry that a later ingest can cross-link to.
    let helper = src_dir.join("helper.rs");
    let helper_content = "//! Helper utilities.\npub fn help() {}\n";
    std::fs::write(&helper, helper_content).unwrap();
    wiki.ingest_file(&proj, &helper, helper_content).unwrap();

    // Second ingest: a file whose module docs mention src/helper.rs.
    let caller = src_dir.join("caller.rs");
    let caller_content = "//! Caller that delegates to src/helper.rs for work.\npub fn run() {}\n";
    std::fs::write(&caller, caller_content).unwrap();
    let outcome = wiki.ingest_file(&proj, &caller, caller_content).unwrap();

    let page_rel = match outcome {
        IngestOutcome::Ingested { page_rel } => page_rel,
        other => panic!("expected Ingested, got {:?}", other),
    };
    let page = wiki.read_page(&page_rel).unwrap();
    assert!(
        page.body.contains("[[src/helper.rs]]"),
        "expected [[src/helper.rs]] wiki-link in caller page body: {}",
        page.body
    );
}

#[test]
fn ingest_file_omits_wiki_links_when_index_empty() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_dir = proj.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let file = src_dir.join("lonely.rs");
    let content = "//! Lonely module.\npub fn solo() {}\n";
    std::fs::write(&file, content).unwrap();

    let outcome = wiki.ingest_file(&proj, &file, content).unwrap();
    let page_rel = match outcome {
        IngestOutcome::Ingested { page_rel } => page_rel,
        other => panic!("expected Ingested, got {:?}", other),
    };
    let page = wiki.read_page(&page_rel).unwrap();
    assert!(
        !page.body.contains("[["),
        "empty index ⇒ no wiki-links should appear: {}",
        page.body
    );
}

#[test]
fn ingest_file_does_not_self_link() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_dir = proj.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // Seed index with an entity page for src/self.rs so the snapshot
    // load sees itself as a candidate. Use a second ingest, not direct
    // index manipulation, so the frontmatter path is exercised.
    let file = src_dir.join("self.rs");
    let content =
        "//! Self-referential module mentioning src/self.rs explicitly.\npub fn me() {}\n";
    std::fs::write(&file, content).unwrap();
    wiki.ingest_file(&proj, &file, content).unwrap();

    // Re-ingest with changed content to force a second write. The
    // second pass has the first page already in the index, so if
    // self-linking weren't suppressed the module-doc mention would
    // get wrapped.
    let content2 =
        "//! Self-referential module mentioning src/self.rs explicitly.\npub fn me_v2() {}\n";
    std::fs::write(&file, content2).unwrap();
    let outcome = wiki.ingest_file(&proj, &file, content2).unwrap();
    let page_rel = match outcome {
        IngestOutcome::Ingested { page_rel } => page_rel,
        other => panic!("expected Ingested, got {:?}", other),
    };
    let page = wiki.read_page(&page_rel).unwrap();
    assert!(
        !page.body.contains("[[src/self.rs]]"),
        "a page must not wiki-link to its own source file: {}",
        page.body
    );
}

// ─── Cycle 36: impl-block filter (regex anchor at column 0) ───────────

#[test]
fn extract_rust_preview_skips_impl_block_methods() {
    let src = "\
pub fn top() {}

impl Foo {
    pub fn method_a(&self) {}
    fn method_b(&self) {}
}
";
    let out = extract_content_preview("src/thing.rs", src);
    assert!(
        out.contains("`pub fn top`"),
        "top-level fn present: {}",
        out
    );
    assert!(
        !out.contains("`pub fn method_a`") && !out.contains("`fn method_b`"),
        "impl-block methods must not appear in items: {}",
        out
    );
}

#[test]
fn extract_rust_preview_still_captures_column_zero_items() {
    let src = "\
pub fn alpha() {}
pub struct Beta;
fn gamma() {}
pub(crate) enum Delta { X }
";
    let out = extract_content_preview("src/col0.rs", src);
    assert!(out.contains("`pub fn alpha`"), "alpha: {}", out);
    assert!(out.contains("`pub struct Beta`"), "Beta: {}", out);
    assert!(out.contains("`fn gamma`"), "gamma: {}", out);
    assert!(out.contains("`pub(crate) enum Delta`"), "Delta: {}", out);
}

#[test]
fn ingest_appears_in_context_snippet_after_ingest() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("lib.rs");
    fs::write(&file, "pub fn x(){}").unwrap();

    wiki.ingest_file(&proj, &file, "pub fn x(){}").unwrap();

    let snippet = wiki.context_snippet().expect("snippet");
    // C19: snippet renders title + one-liner, not path. The page's
    // title is "File: lib.rs"; presence proves the ingested entry
    // surfaces in the next session's context.
    assert!(snippet.contains("File: lib.rs"));
}

/// Hammer the same (path, content) from many threads. The cache mutex
/// should let exactly one caller through; the rest must see
/// `UnchangedSinceLast`. End state: 1 index entry, 1 log line.
#[test]
fn ingest_file_concurrent_same_content_dedups_to_one() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("hot.txt");
    fs::write(&file, "x").unwrap();

    let mut handles = Vec::new();
    for _ in 0..20 {
        let wiki = wiki.clone();
        let proj = proj.clone();
        let file = file.clone();
        handles.push(std::thread::spawn(move || {
            wiki.ingest_file(&proj, &file, "x").unwrap()
        }));
    }
    let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let ingested = outcomes
        .iter()
        .filter(|o| matches!(o, IngestOutcome::Ingested { .. }))
        .count();
    let skipped = outcomes
        .iter()
        .filter(|o| matches!(o, IngestOutcome::Skipped(SkipReason::UnchangedSinceLast)))
        .count();
    assert_eq!(ingested, 1, "exactly one winner: {:?}", outcomes);
    assert_eq!(skipped, 19, "all others dedup: {:?}", outcomes);

    let idx = wiki.load_index().unwrap();
    assert_eq!(idx.entries.len(), 1);
    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert_eq!(log.matches("ingest | hot.txt").count(), 1);
}

/// Concurrent ingests with distinct content per thread admit every caller
/// past the dedup cache (each has a unique hash). The index load/save
/// sequence is not locked — characterize the observable end state so any
/// regression in convergence or duplication is caught.
///
/// Current behavior: last-writer wins on the page; the index may end up
/// with 1-to-N entries depending on interleave. This test accepts any
/// count ≥ 1 and only fails if the page file is missing.
#[test]
fn ingest_file_concurrent_distinct_content_is_non_crashing() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("race.txt");
    fs::write(&file, "x").unwrap();

    let mut handles = Vec::new();
    for i in 0..16 {
        let wiki = wiki.clone();
        let proj = proj.clone();
        let file = file.clone();
        handles.push(std::thread::spawn(move || {
            wiki.ingest_file(&proj, &file, &format!("content-{}", i))
                .unwrap()
        }));
    }
    for h in handles {
        let _ = h.join().unwrap();
    }

    let page = wiki.root().join("entities/race_txt.md");
    assert!(page.is_file(), "page must exist after race");

    let idx = wiki.load_index().unwrap();
    let race_entries = idx
        .entries
        .iter()
        .filter(|e| e.path == "entities/race_txt.md")
        .count();
    assert_eq!(
        race_entries, 1,
        "upsert under INDEX_LOCK must collapse 16 concurrent writers on the \
             same page_rel to exactly one index entry; got {}",
        race_entries
    );
}

/// Two distinct project paths can slug to the same `entities/*.md`
/// filename (`a/b.rs` and `a_b.rs` both → `entities/a_b_rs.md`).
/// Characterize the current behavior: last write wins, 1 page, 1 entry.
/// Flip to multi-entry assertions when hash-suffixing is added.
#[test]
fn ingest_file_slug_collision_last_writer_wins() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let file_slash = proj.join("a").join("b.rs");
    fs::create_dir_all(file_slash.parent().unwrap()).unwrap();
    fs::write(&file_slash, "first").unwrap();
    let file_flat = proj.join("a_b.rs");
    fs::write(&file_flat, "second").unwrap();

    let first = wiki.ingest_file(&proj, &file_slash, "first").unwrap();
    let second = wiki.ingest_file(&proj, &file_flat, "second").unwrap();

    match (&first, &second) {
        (IngestOutcome::Ingested { page_rel: p1 }, IngestOutcome::Ingested { page_rel: p2 }) => {
            assert_eq!(p1, p2, "both must slug to the same page");
            assert_eq!(p1, "entities/a_b_rs.md");
        }
        other => panic!("expected two Ingested outcomes, got {:?}", other),
    }

    // Page body reflects the last writer.
    let page = wiki.read_page("entities/a_b_rs.md").expect("page readable");
    assert_eq!(page.title, "a_b.rs", "last writer's title wins");

    // Index collapses to one entry via upsert-on-path.
    let idx = wiki.load_index().unwrap();
    let colliding: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.path == "entities/a_b_rs.md")
        .collect();
    assert_eq!(colliding.len(), 1, "upsert, not duplicate");
    assert_eq!(colliding[0].title, "a_b.rs");
}

/// A symlink inside the project whose target is outside must be rejected
/// as `OutsideProject` — the `file_read` hook canonicalizes before calling
/// ingest, so the symlink itself never reaches this code; only the
/// resolved target does.
#[cfg(unix)]
#[test]
fn ingest_file_symlink_target_outside_is_outside_project() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();

    let proj_tmp = TempDir::new().unwrap();
    let outside_tmp = TempDir::new().unwrap();
    let proj = proj_tmp.path().canonicalize().unwrap();
    let outside = outside_tmp.path().canonicalize().unwrap().join("real.txt");
    fs::write(&outside, "leaked").unwrap();

    let link = proj.join("link.txt");
    std::os::unix::fs::symlink(&outside, &link).unwrap();

    let resolved = link.canonicalize().unwrap();
    assert_eq!(
        resolved, outside,
        "canonicalize should resolve the symlink to the outside target"
    );

    let wiki = Wiki::open(&proj).unwrap();
    let outcome = wiki.ingest_file(&proj, &resolved, "leaked").unwrap();
    assert_eq!(outcome, IngestOutcome::Skipped(SkipReason::OutsideProject));

    let page = proj.join(".dm/wiki/entities/link_txt.md");
    assert!(!page.exists(), "no entity page for escaped symlink");
}

/// Calling `ingest_file` with `canonical_path == project_root` yields
/// a relative path of `""` or `"."`, which must be rejected as
/// `IneligiblePath` rather than producing an `entities/unnamed.md` page.
#[test]
fn ingest_file_project_root_itself_is_ineligible() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let outcome = wiki.ingest_file(&proj, &proj, "whatever").unwrap();
    assert_eq!(outcome, IngestOutcome::Skipped(SkipReason::IneligiblePath));
    let unnamed = wiki.root().join("entities/unnamed.md");
    assert!(!unnamed.exists(), "must not create unnamed.md");
}

/// Empty content has a stable, distinct hash — ensures the dedup cache
/// doesn't conflate empty files with some default sentinel.
#[test]
fn content_hash_empty_is_stable_and_distinct() {
    assert_eq!(content_hash(""), content_hash(""));
    assert_ne!(content_hash(""), content_hash(" "));
    assert_ne!(content_hash(""), content_hash("\0"));
}

/// `DM_WIKI_AUTO_INGEST=false` and `=off` must both disable (not just
/// `0`). Matches the documented contract in `auto_ingest_enabled`.
#[test]
fn ingest_file_respects_env_disable_all_variants() {
    for variant in ["0", "false", "off"] {
        let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", variant);
        reset_ingest_cache_for_tests();
        let tmp = TempDir::new().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = Wiki::open(&proj).unwrap();
        let file = proj.join("v.txt");
        fs::write(&file, "c").unwrap();

        let outcome = wiki.ingest_file(&proj, &file, "c").unwrap();
        assert_eq!(
            outcome,
            IngestOutcome::Skipped(SkipReason::Disabled),
            "variant {:?} should disable",
            variant
        );
    }
}

// ── compact-to-wiki bridge ────────────────────────────────────────────

#[test]
fn write_compact_synthesis_writes_page_index_and_log() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_compact_synthesis("Fixed bug in main.rs", 12, &["src/main.rs".into()])
        .unwrap()
        .expect("should return page path when enabled");
    assert!(out.starts_with("synthesis/compact-"), "got: {}", out);
    assert!(out.ends_with(".md"), "got: {}", out);

    let page_path = wiki.root().join(&out);
    assert!(page_path.is_file(), "page not on disk: {:?}", page_path);
    let text = fs::read_to_string(&page_path).unwrap();
    let page = WikiPage::parse(&text).expect("page parses");
    assert_eq!(page.page_type, PageType::Synthesis);
    assert_eq!(page.sources, vec!["src/main.rs".to_string()]);
    assert!(
        page.body.contains("Fixed bug in main.rs"),
        "body: {}",
        page.body
    );
    assert!(page.body.contains("12 messages"), "body: {}", page.body);

    let idx = wiki.load_index().unwrap();
    let synthesis_entries: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .collect();
    assert_eq!(synthesis_entries.len(), 1);
    assert_eq!(synthesis_entries[0].path, out);

    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert!(
        log.contains(&format!("compact | {}", out)),
        "log: {:?}",
        log
    );
}

#[test]
fn write_compact_synthesis_pluralizes_message_count() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let one = wiki.write_compact_synthesis("s", 1, &[]).unwrap().unwrap();
    let two = wiki.write_compact_synthesis("s", 2, &[]).unwrap().unwrap();

    let body_one = fs::read_to_string(wiki.root().join(&one)).unwrap();
    assert!(body_one.contains("1 message"), "one: {}", body_one);
    assert!(
        !body_one.contains("1 messages"),
        "one singular: {}",
        body_one
    );

    let body_two = fs::read_to_string(wiki.root().join(&two)).unwrap();
    assert!(body_two.contains("2 messages"), "two: {}", body_two);
}

#[test]
fn write_compact_synthesis_back_to_back_no_overwrite() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let a = wiki
        .write_compact_synthesis("first", 3, &[])
        .unwrap()
        .unwrap();
    let b = wiki
        .write_compact_synthesis("second", 4, &[])
        .unwrap()
        .unwrap();

    assert_ne!(a, b, "back-to-back calls must produce distinct paths");
    assert!(wiki.root().join(&a).is_file());
    assert!(wiki.root().join(&b).is_file());

    let idx = wiki.load_index().unwrap();
    let count = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .count();
    assert_eq!(count, 2);
}

#[test]
fn write_compact_synthesis_empty_sources_works() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_compact_synthesis("no sources", 5, &[])
        .unwrap()
        .unwrap();
    let text = fs::read_to_string(wiki.root().join(&out)).unwrap();
    let page = WikiPage::parse(&text).expect("parses");
    assert!(page.sources.is_empty(), "sources: {:?}", page.sources);
}

#[test]
fn write_compact_synthesis_respects_env_disable() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", "0");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_compact_synthesis("nope", 1, &["src/x.rs".into()])
        .unwrap();
    assert!(out.is_none(), "disabled should return None");

    let synth_dir = wiki.root().join("synthesis");
    let has_compact = fs::read_dir(&synth_dir).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("compact-")
    });
    assert!(!has_compact, "disabled must not write any compact-*.md");

    let idx = wiki.load_index().unwrap();
    let synth_count = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .count();
    assert_eq!(synth_count, 0, "index must be untouched");
}

#[test]
fn write_compact_synthesis_preserves_summary_text_verbatim() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let summary = "### Key decisions\n\
                       - Added `foo` — see `src/foo.rs`\n\
                       ```rs\n\
                       pub fn foo() {}\n\
                       ```\n\
                       Unicode café ✓";
    let out = wiki
        .write_compact_synthesis(summary, 7, &[])
        .unwrap()
        .unwrap();

    let text = fs::read_to_string(wiki.root().join(&out)).unwrap();
    let page = WikiPage::parse(&text).expect("parses");
    assert!(
        page.body.contains(summary),
        "body missing verbatim summary:\n{}",
        page.body
    );
}

#[test]
fn write_compact_synthesis_respects_env_ingest_guard_for_synthesis_too() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", "off");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki.write_compact_synthesis("off-case", 9, &[]).unwrap();
    assert!(out.is_none(), "'off' must also disable synthesis writes");
}

/// After a compact synthesis lands, the next session's `context_snippet`
/// must surface the new page under the Synthesis category. Proves the
/// "compacted knowledge carries into the next session" invariant end-to-end.
#[test]
fn compact_synthesis_surfaces_in_next_session_context_snippet() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_compact_synthesis(
            "discovered race in run_conversation",
            37,
            &["src/conversation.rs".into()],
        )
        .unwrap()
        .expect("page written");

    let snippet = wiki.context_snippet().expect("snippet should exist");
    assert!(
        snippet.contains("### synthesis"),
        "snippet missing synthesis category: {}",
        snippet
    );
    // C19: snippet renders titles, not paths. The synthesis page's
    // title starts with "Compaction summary" — that's enough to pin
    // the cross-session knowledge-carry invariant.
    let _ = page_rel;
    assert!(
        snippet.contains("Compaction summary"),
        "snippet missing synthesis page title: {}",
        snippet
    );
    assert!(
        snippet.contains("Compact snapshot (37 messages)"),
        "snippet missing one-liner: {}",
        snippet
    );
}

/// Two threads each call `Wiki::open(&proj)` *independently* and ingest
/// the same project path with distinct content. `INGEST_CACHE` admits
/// both writers (distinct hashes), so both hit the index critical
/// section. `INDEX_LOCK` is a process-global static — independent `Wiki`
/// handles share it — so the two upserts must collapse to exactly one
/// entry for that `page_rel`, with no duplication.
#[test]
fn two_wiki_handles_concurrent_ingest_upserts_to_one() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let _ = Wiki::open(&proj).unwrap();
    let file = proj.join("shared.txt");
    fs::write(&file, "seed").unwrap();

    let proj_a = proj.clone();
    let file_a = file.clone();
    let h_a = std::thread::spawn(move || {
        let wiki = Wiki::open(&proj_a).unwrap();
        wiki.ingest_file(&proj_a, &file_a, "content-a").unwrap()
    });
    let proj_b = proj.clone();
    let file_b = file.clone();
    let h_b = std::thread::spawn(move || {
        let wiki = Wiki::open(&proj_b).unwrap();
        wiki.ingest_file(&proj_b, &file_b, "content-b").unwrap()
    });
    let _ = h_a.join().unwrap();
    let _ = h_b.join().unwrap();

    let wiki = Wiki::open(&proj).unwrap();
    let idx = wiki.load_index().unwrap();
    let entries: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.path == "entities/shared_txt.md")
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "two independent Wiki handles must still collapse to one entry via INDEX_LOCK"
    );
    let page = wiki.root().join("entities/shared_txt.md");
    assert!(page.is_file());
}

/// Two independent `Wiki` handles each call `write_compact_synthesis`
/// concurrently. Nanosecond-stamped filenames are distinct, both pages
/// land on disk, both index entries must survive.
#[test]
fn two_wiki_handles_concurrent_compact_both_entries_survive() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let _ = Wiki::open(&proj).unwrap();

    let proj_a = proj.clone();
    let h_a = std::thread::spawn(move || {
        let wiki = Wiki::open(&proj_a).unwrap();
        wiki.write_compact_synthesis("first compact", 5, &[])
            .unwrap()
    });
    let proj_b = proj.clone();
    let h_b = std::thread::spawn(move || {
        let wiki = Wiki::open(&proj_b).unwrap();
        wiki.write_compact_synthesis("second compact", 7, &[])
            .unwrap()
    });
    let page_a = h_a.join().unwrap().expect("a enabled");
    let page_b = h_b.join().unwrap().expect("b enabled");
    assert_ne!(page_a, page_b, "distinct nanosecond filenames");

    let wiki = Wiki::open(&proj).unwrap();
    let idx = wiki.load_index().unwrap();
    let synth_paths: std::collections::HashSet<String> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .map(|e| e.path.clone())
        .collect();
    assert!(synth_paths.contains(&page_a), "page_a lost from index");
    assert!(synth_paths.contains(&page_b), "page_b lost from index");
    assert_eq!(synth_paths.len(), 2, "exactly two synthesis entries");

    for page_rel in [&page_a, &page_b] {
        assert!(
            wiki.root().join(page_rel).is_file(),
            "page {} missing on disk",
            page_rel
        );
    }
}

/// Mixed concurrent workload: half the threads ingest, half compact.
/// Index updates from both call sites share `INDEX_LOCK`, so no entries
/// may be dropped regardless of which path wrote them.
#[test]
fn mixed_concurrent_ingest_and_compact_no_entries_lost() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let mut handles = Vec::new();
    for i in 0..8 {
        let wiki = wiki.clone();
        let proj = proj.clone();
        handles.push(std::thread::spawn(move || {
            if i % 2 == 0 {
                let file = proj.join(format!("f{}.rs", i));
                std::fs::write(&file, format!("c{}", i)).unwrap();
                wiki.ingest_file(&proj, &file, &format!("c{}", i)).unwrap();
                None
            } else {
                Some(
                    wiki.write_compact_synthesis(&format!("s{}", i), i + 1, &[])
                        .unwrap()
                        .expect("enabled"),
                )
            }
        }));
    }
    let compact_rels: Vec<String> = handles
        .into_iter()
        .filter_map(|h| h.join().unwrap())
        .collect();

    let idx = wiki.load_index().unwrap();
    let entity_entries = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Entity)
        .count();
    let synth_paths: std::collections::HashSet<String> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .map(|e| e.path.clone())
        .collect();
    assert_eq!(entity_entries, 4, "four distinct ingested files");
    assert_eq!(synth_paths.len(), 4, "four synthesis entries");
    for rel in &compact_rels {
        assert!(synth_paths.contains(rel), "compact {} dropped", rel);
    }
}

/// STRICT: pages on disk and index synthesis entries must match set-wise
/// under 8 concurrent `write_compact_synthesis` calls. `INDEX_LOCK`
/// serializes the `load_index → push → save_index` critical section, so
/// no index entry may be lost.
#[test]
fn write_compact_synthesis_concurrent_index_strict() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let mut handles = Vec::new();
    for i in 0..8 {
        let wiki = wiki.clone();
        handles.push(std::thread::spawn(move || {
            wiki.write_compact_synthesis(&format!("payload {}", i), i + 1, &[])
                .unwrap()
                .expect("enabled")
        }));
    }
    for h in handles {
        let _ = h.join().unwrap();
    }

    let pages_on_disk: std::collections::HashSet<String> =
        std::fs::read_dir(wiki.root().join("synthesis"))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| format!("synthesis/{}", e.file_name().to_string_lossy()))
            .collect();

    let idx = wiki.load_index().unwrap();
    let index_synth_paths: std::collections::HashSet<String> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .map(|e| e.path.clone())
        .collect();

    assert_eq!(
        pages_on_disk,
        index_synth_paths,
        "index dropped synthesis entries under concurrent compact — \
             pages_on_disk - index = {:?}",
        pages_on_disk
            .difference(&index_synth_paths)
            .collect::<Vec<_>>()
    );
}

/// Three back-to-back compacts (single-threaded) each land a distinct
/// page, entry, and log line. Nanosecond filename guard exercised.
#[test]
fn write_compact_synthesis_three_back_to_back_all_land() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let a = wiki.write_compact_synthesis("A", 1, &[]).unwrap().unwrap();
    let b = wiki.write_compact_synthesis("B", 2, &[]).unwrap().unwrap();
    let c = wiki.write_compact_synthesis("C", 3, &[]).unwrap().unwrap();
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);

    let synthesis_dir = wiki.root().join("synthesis");
    let on_disk = fs::read_dir(&synthesis_dir).unwrap().count();
    assert_eq!(on_disk, 3, "three pages must exist");

    let idx = wiki.load_index().unwrap();
    let synth: Vec<_> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .collect();
    assert_eq!(synth.len(), 3, "three index entries, serially");

    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert_eq!(log.matches("compact | synthesis/compact-").count(), 3);
}

/// Summary text containing YAML-frontmatter-lookalike lines must not
/// corrupt the page's own frontmatter. The body is injected under a
/// `## Summary` heading, so it parses as body (not header) regardless
/// of what's inside.
#[test]
fn write_compact_synthesis_summary_with_frontmatter_lookalike_parses_safely() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let evil = "---\ntitle: not-the-title\ntype: entity\n---\nbody";
    let page_rel = wiki.write_compact_synthesis(evil, 1, &[]).unwrap().unwrap();

    let text = fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let page = WikiPage::parse(&text).expect("still parses");
    assert_eq!(page.page_type, PageType::Synthesis);
    assert!(
        page.title.starts_with("Compaction summary"),
        "title hijacked: {}",
        page.title
    );
    assert!(page.body.contains("not-the-title"), "summary preserved");
}

// ── compact-synthesis pruning ────────────────────────────────────────

/// `prune_compact_synthesis_to(N)` keeps the N most recent compact
/// pages and drops both index entries and on-disk files for the rest.
#[test]
fn prune_compact_synthesis_caps_to_max_keep() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    std::env::remove_var("DM_WIKI_COMPACT_KEEP");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let mut written: Vec<String> = Vec::new();
    for i in 0..5 {
        let rel = wiki
            .write_compact_synthesis(&format!("payload {}", i), i + 1, &[])
            .unwrap()
            .unwrap();
        written.push(rel);
    }
    // Slug timestamps embed nanoseconds — sort gives chronological order.
    let mut sorted = written.clone();
    sorted.sort();
    let oldest_two = &sorted[..2];
    let kept_three: std::collections::HashSet<&String> = sorted[2..].iter().collect();

    let pruned = wiki.prune_compact_synthesis_to(3).unwrap();
    assert_eq!(pruned, 2, "should report 2 pruned");

    let idx = wiki.load_index().unwrap();
    let compact_entries: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis && e.path.starts_with("synthesis/compact-"))
        .collect();
    assert_eq!(
        compact_entries.len(),
        3,
        "index should retain 3 compact entries"
    );
    for entry in &compact_entries {
        assert!(
            kept_three.contains(&entry.path),
            "kept entry {} not in expected newest-3",
            entry.path
        );
    }

    for old in oldest_two {
        assert!(
            !wiki.root().join(old).exists(),
            "oldest file {} should be removed",
            old
        );
    }
    for kept in &sorted[2..] {
        assert!(
            wiki.root().join(kept).exists(),
            "kept file {} should remain",
            kept
        );
    }
}

/// `prune_compact_synthesis_to(0)` removes every compact-synthesis
/// page and reports the full count.
#[test]
fn prune_compact_synthesis_zero_clears_all() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    std::env::remove_var("DM_WIKI_COMPACT_KEEP");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let mut written: Vec<String> = Vec::new();
    for i in 0..3 {
        written.push(
            wiki.write_compact_synthesis(&format!("p{}", i), i + 1, &[])
                .unwrap()
                .unwrap(),
        );
    }

    let pruned = wiki.prune_compact_synthesis_to(0).unwrap();
    assert_eq!(pruned, 3);

    let idx = wiki.load_index().unwrap();
    let compact_count = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis && e.path.starts_with("synthesis/compact-"))
        .count();
    assert_eq!(compact_count, 0, "no compact entries should remain");

    for rel in &written {
        assert!(
            !wiki.root().join(rel).exists(),
            "file {} should be removed",
            rel
        );
    }
}

/// Pruning compact-synthesis pages must leave entity entries and
/// non-`compact-*` synthesis entries (e.g. cycle/run pages) untouched
/// and in their original relative order.
#[test]
fn prune_compact_synthesis_preserves_other_categories() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    std::env::remove_var("DM_WIKI_COMPACT_KEEP");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Pre-seed an entity entry and a curated `run-*` synthesis entry.
    let entity_entry = IndexEntry {
        title: "src/foo.rs".into(),
        path: "entities/src_foo_rs.md".into(),
        one_liner: "foo entity".into(),
        category: PageType::Entity,
        last_updated: Some("2026-01-01 00:00:00".into()),
        outcome: None,
    };
    let run_entry = IndexEntry {
        title: "Curated run note".into(),
        path: "synthesis/run-curated.md".into(),
        one_liner: "operator-curated note".into(),
        category: PageType::Synthesis,
        last_updated: Some("2026-01-02 00:00:00".into()),
        outcome: None,
    };
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(entity_entry.clone());
    idx.entries.push(run_entry.clone());
    wiki.save_index(&idx).unwrap();

    for i in 0..4 {
        wiki.write_compact_synthesis(&format!("c{}", i), i + 1, &[])
            .unwrap()
            .unwrap();
    }

    let pruned = wiki.prune_compact_synthesis_to(1).unwrap();
    assert_eq!(pruned, 3);

    let idx = wiki.load_index().unwrap();
    let other: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| {
            !(e.category == PageType::Synthesis && e.path.starts_with("synthesis/compact-"))
        })
        .collect();
    let entity_pos = other
        .iter()
        .position(|e| **e == entity_entry)
        .expect("entity entry must survive prune untouched");
    let run_pos = other
        .iter()
        .position(|e| **e == run_entry)
        .expect("run-* entry must survive prune untouched");
    assert!(
        entity_pos < run_pos,
        "original relative order (entity before run-*) must be preserved",
    );

    let compact_count = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis && e.path.starts_with("synthesis/compact-"))
        .count();
    assert_eq!(compact_count, 1, "exactly 1 compact entry should remain");
}

/// `write_compact_synthesis` reads `DM_WIKI_COMPACT_KEEP` after every
/// write and self-prunes, so long-running chains can't accumulate
/// thousands of compact pages.
#[test]
fn write_compact_synthesis_self_bounds_via_env() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let _cap_guard = EnvGuard::set("DM_WIKI_COMPACT_KEEP", "2");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    for i in 0..4 {
        wiki.write_compact_synthesis(&format!("payload {}", i), i + 1, &[])
            .unwrap()
            .unwrap();
    }

    let idx = wiki.load_index().unwrap();
    let compact_entries: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis && e.path.starts_with("synthesis/compact-"))
        .collect();
    assert_eq!(
        compact_entries.len(),
        2,
        "index should be bounded at DM_WIKI_COMPACT_KEEP=2"
    );

    let synth_dir = wiki.root().join("synthesis");
    let on_disk: Vec<_> = fs::read_dir(&synth_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("compact-"))
        .collect();
    assert_eq!(
        on_disk.len(),
        2,
        "disk should be bounded at DM_WIKI_COMPACT_KEEP=2"
    );
}

// ── Wiki::seed_dir tests ─────────────────────────────────────────────

#[test]
fn seed_dir_ingests_rust_files_recursive() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src/sub")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();
    fs::write(proj.join("src/sub/b.rs"), "//! b").unwrap();
    fs::write(proj.join("src/c.txt"), "not rust").unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert_eq!(
        report.ingested.len(),
        2,
        "should ingest exactly 2 .rs files, got {:?}",
        report.ingested
    );
    assert!(wiki.root().join("entities/src_a_rs.md").is_file());
    assert!(wiki.root().join("entities/src_sub_b_rs.md").is_file());
    assert!(
        !wiki.root().join("entities/src_c_txt.md").exists(),
        ".txt file must not be ingested"
    );
}

#[test]
fn seed_dir_skips_dot_target_node_modules() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::create_dir_all(proj.join(".git")).unwrap();
    fs::create_dir_all(proj.join("target/debug")).unwrap();
    fs::create_dir_all(proj.join("node_modules/some-pkg")).unwrap();
    fs::write(proj.join("src/keep.rs"), "//! keep").unwrap();
    fs::write(proj.join(".git/x.rs"), "//! x").unwrap();
    fs::write(proj.join("target/debug/y.rs"), "//! y").unwrap();
    fs::write(proj.join("node_modules/some-pkg/z.rs"), "//! z").unwrap();

    let report = wiki.seed_dir(std::path::Path::new("."), &["rs"]).unwrap();
    assert_eq!(
        report.ingested.len(),
        1,
        "only src/keep.rs should be ingested, got {:?}",
        report.ingested
    );
    assert_eq!(report.ingested[0], "entities/src_keep_rs.md");
}

#[test]
fn seed_dir_idempotent_second_call_unchanged() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();
    fs::write(proj.join("src/b.rs"), "//! b").unwrap();

    let first = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert_eq!(first.ingested.len(), 2);

    let second = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert!(
        second.ingested.is_empty(),
        "second seed should ingest nothing, got {:?}",
        second.ingested
    );
    assert_eq!(
        second.skipped_unchanged.len(),
        first.ingested.len(),
        "unchanged count should match the first run's ingested count",
    );
}

#[test]
fn seed_dir_errors_on_missing_dir() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let err = wiki
        .seed_dir(std::path::Path::new("nope"), &["rs"])
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn seed_dir_respects_ext_filter() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();

    // Empty ext_filter = strict mismatch contract: nothing matches.
    let report = wiki.seed_dir(std::path::Path::new("src"), &[]).unwrap();
    assert!(report.ingested.is_empty());
    assert_eq!(report.skipped_unchanged.len(), 0);
    assert_eq!(report.errors.len(), 0);
}

#[test]
fn seed_dir_no_symlinks_keeps_counter_zero() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src/sub")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();
    fs::write(proj.join("src/sub/b.rs"), "//! b").unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert_eq!(
        report.symlinks_skipped, 0,
        "no-symlink tree should keep counter at zero"
    );
    assert_eq!(report.ingested.len(), 2);
}

#[cfg(unix)]
#[test]
fn seed_dir_skips_symlinked_dir_no_recursion() {
    use std::os::unix::fs::symlink;

    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();
    // Cyclic symlink: src/loop -> src (parent). If the walker followed
    // it, the next test run would never terminate.
    symlink(proj.join("src"), proj.join("src/loop")).unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert_eq!(
        report.ingested.len(),
        1,
        "cyclic symlink must not cause repeat ingest, got {:?}",
        report.ingested
    );
    assert!(
        report.symlinks_skipped >= 1,
        "expected at least one symlink skip, got {}",
        report.symlinks_skipped,
    );
}

#[cfg(unix)]
#[test]
fn seed_dir_does_not_escape_project_via_symlinked_dir() {
    use std::os::unix::fs::symlink;

    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();

    let outside = TempDir::new().unwrap();
    let outside_root = outside.path().canonicalize().unwrap();
    fs::write(outside_root.join("evil.rs"), "//! evil").unwrap();

    let proj_tmp = TempDir::new().unwrap();
    let proj = proj_tmp.path().canonicalize().unwrap();
    assert_ne!(
        outside_root, proj,
        "test fixture broken: outside dir collided with project root",
    );
    let wiki = Wiki::open(&proj).unwrap();
    fs::create_dir_all(proj.join("src")).unwrap();
    symlink(outside_root.clone(), proj.join("src/escape")).unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert!(
        report.ingested.is_empty(),
        "symlink to out-of-tree dir must not produce ingest, got {:?}",
        report.ingested,
    );
    assert!(
        !wiki.root().join("entities/src_outside_evil_rs.md").exists(),
        "out-of-tree evil.rs page must not exist on disk",
    );
    assert!(
        report.symlinks_skipped >= 1,
        "expected at least one symlink skip, got {}",
        report.symlinks_skipped,
    );
}

#[cfg(unix)]
#[test]
fn seed_dir_skips_symlinked_file() {
    use std::os::unix::fs::symlink;

    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::write(proj.join("src/real.rs"), "//! real").unwrap();
    symlink("real.rs", proj.join("src/link.rs")).unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert_eq!(
        report.ingested.len(),
        1,
        "only the real file's page should be written, got {:?}",
        report.ingested,
    );
    assert!(
        report.symlinks_skipped >= 1,
        "expected at least one symlink skip, got {}",
        report.symlinks_skipped,
    );
}

#[test]
fn seed_dir_honors_disable_env() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _disable = EnvGuard::set("DM_WIKI_AUTO_INGEST", "0");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    fs::create_dir_all(proj.join("src")).unwrap();
    fs::write(proj.join("src/a.rs"), "//! a").unwrap();
    fs::write(proj.join("src/b.rs"), "//! b").unwrap();

    let report = wiki.seed_dir(std::path::Path::new("src"), &["rs"]).unwrap();
    assert!(
        report.ingested.is_empty(),
        "disabled env must yield zero ingests, got {:?}",
        report.ingested
    );
    assert_eq!(
        report.skipped_other, 2,
        "both files should land in skipped_other when ingest is disabled"
    );
}

/// `prune_compact_synthesis_to` must be reachable through the public
/// `Wiki` surface (no `pub(super)` workaround) so `/wiki prune` and any
/// future external caller can invoke it without crate-internal access.
/// The 0-cap clears all and returns the count.
#[test]
fn prune_compact_synthesis_to_is_publicly_reachable() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    std::env::remove_var("DM_WIKI_COMPACT_KEEP");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki: crate::wiki::Wiki = crate::wiki::Wiki::open(&proj).unwrap();

    for i in 0..3 {
        wiki.write_compact_synthesis(&format!("p{}", i), i + 1, &[])
            .unwrap()
            .unwrap();
    }
    let pruned: usize = wiki.prune_compact_synthesis_to(0).unwrap();
    assert_eq!(pruned, 3);
}

// ── Wiki::write_cycle_synthesis tests (Cycle 55) ──────────────────────

#[test]
fn write_cycle_synthesis_writes_page_index_and_log() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let nodes = vec![
        CycleNodeSnapshot {
            name: "planner".into(),
            role: "planner".into(),
            output: "plan text".into(),
        },
        CycleNodeSnapshot {
            name: "builder".into(),
            role: "builder".into(),
            output: "built it".into(),
        },
    ];
    let out = wiki
        .write_cycle_synthesis(3, "incubation", &nodes, None)
        .unwrap()
        .expect("should return page path when enabled");
    assert!(out.starts_with("synthesis/cycle-03-"), "got: {}", out);
    assert!(out.ends_with(".md"), "got: {}", out);

    let page_path = wiki.root().join(&out);
    assert!(page_path.is_file(), "page not on disk: {:?}", page_path);
    let text = fs::read_to_string(&page_path).unwrap();
    let page = WikiPage::parse(&text).expect("page parses");
    assert_eq!(page.page_type, PageType::Synthesis);
    assert!(page.sources.is_empty());
    assert!(page.body.contains("Chain cycle 3 — incubation"));
    assert!(page.body.contains("**planner** (planner)"));
    assert!(page.body.contains("**builder** (builder)"));
    assert!(page.body.contains("plan text"));
    assert!(page.body.contains("built it"));

    let idx = wiki.load_index().unwrap();
    let entry = idx
        .entries
        .iter()
        .find(|e| e.path == out)
        .expect("index entry present");
    assert_eq!(entry.category, PageType::Synthesis);
    assert_eq!(entry.one_liner, "Cycle 3 of incubation");

    let log = fs::read_to_string(wiki.root().join("log.md")).unwrap();
    assert!(
        log.contains(&format!("{} | {}", CYCLE_SYNTHESIS_LOG_VERB, out)),
        "log: {:?}",
        log
    );
}

#[test]
fn write_cycle_synthesis_respects_env_disable() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _guard = EnvGuard::set("DM_WIKI_AUTO_INGEST", "0");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_cycle_synthesis(1, "incubation", &[], None)
        .unwrap();
    assert!(out.is_none(), "disabled should return None");

    let synth_dir = wiki.root().join("synthesis");
    if synth_dir.is_dir() {
        let has_cycle = fs::read_dir(&synth_dir).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("cycle-")
        });
        assert!(!has_cycle, "disabled must not write any cycle-*.md");
    }
}

#[test]
fn write_cycle_synthesis_kebab_slugs_exotic_chain_name() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_cycle_synthesis(7, "Self-Improve / v2!", &[], None)
        .unwrap()
        .unwrap();
    assert!(
        out.contains("self-improve-v2"),
        "slug missing in path: {}",
        out
    );
    assert!(
        !out.contains("Self-Improve"),
        "case/punct leaked into filename: {}",
        out
    );
}

#[test]
fn write_cycle_synthesis_back_to_back_no_overwrite() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let a = wiki
        .write_cycle_synthesis(1, "chain", &[], None)
        .unwrap()
        .unwrap();
    let b = wiki
        .write_cycle_synthesis(1, "chain", &[], None)
        .unwrap()
        .unwrap();
    assert_ne!(a, b, "back-to-back calls must produce distinct paths");
    assert!(wiki.root().join(&a).is_file());
    assert!(wiki.root().join(&b).is_file());
}

#[test]
fn write_cycle_synthesis_empty_nodes_produces_valid_page() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let out = wiki
        .write_cycle_synthesis(4, "empty-chain", &[], None)
        .unwrap()
        .unwrap();
    let text = fs::read_to_string(wiki.root().join(&out)).unwrap();
    let page = WikiPage::parse(&text).expect("parses");
    assert_eq!(page.page_type, PageType::Synthesis);
    assert!(page.body.contains("## Nodes"));
}

#[test]
fn write_cycle_synthesis_empty_output_renders_placeholder() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let nodes = vec![CycleNodeSnapshot {
        name: "builder".into(),
        role: "builder".into(),
        output: "".into(),
    }];
    let out = wiki
        .write_cycle_synthesis(1, "chain", &nodes, None)
        .unwrap()
        .unwrap();
    let text = fs::read_to_string(wiki.root().join(&out)).unwrap();
    assert!(
        text.contains("_(no output)_"),
        "missing placeholder: {}",
        text
    );
}

#[test]
fn write_cycle_synthesis_truncates_oversize_output() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Many newline-separated lines, well above the cap.
    let big: String = (0..200)
        .map(|i| format!("line {:03}", i))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(big.len() > CYCLE_SYNTHESIS_OUTPUT_PER_NODE);

    let nodes = vec![CycleNodeSnapshot {
        name: "builder".into(),
        role: "builder".into(),
        output: big,
    }];
    let out = wiki
        .write_cycle_synthesis(1, "chain", &nodes, None)
        .unwrap()
        .unwrap();
    let text = fs::read_to_string(wiki.root().join(&out)).unwrap();
    assert!(
        text.contains("[...truncated]"),
        "missing truncation marker: {}",
        text
    );
    assert!(
        !text.contains("line 199"),
        "tail should be dropped: {}",
        text
    );
}

#[test]
fn write_cycle_synthesis_page_passes_lint_clean() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let nodes = vec![CycleNodeSnapshot {
        name: "planner".into(),
        role: "planner".into(),
        output: "done".into(),
    }];
    let out = wiki
        .write_cycle_synthesis(2, "chain", &nodes, None)
        .unwrap()
        .unwrap();

    let findings = wiki.lint().unwrap();
    let our_findings: Vec<_> = findings.iter().filter(|f| f.path == out).collect();
    assert!(
        our_findings.is_empty(),
        "cycle synthesis page should not trip lint: {:?}",
        our_findings
    );
}

// ── Wiki::stats() tests (cycle 10 — `/wiki status` data layer) ─────────

#[test]
fn stats_on_empty_wiki_returns_zeros() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let s = wiki.stats().unwrap();
    assert_eq!(s.total_pages, 0);
    assert!(s.by_category.is_empty());
    assert_eq!(s.log_entries, 0);
    assert!(s.last_activity.is_none());
    assert_eq!(s.root, wiki.root());
}

#[test]
fn stats_counts_by_category() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();

    // Manually construct an index covering three categories, two entities.
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "a".into(),
                path: "entities/a.md".into(),
                one_liner: "first".into(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "b".into(),
                path: "entities/b.md".into(),
                one_liner: "second".into(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "cx".into(),
                path: "concepts/cx.md".into(),
                one_liner: "concept".into(),
                category: PageType::Concept,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "syn".into(),
                path: "synthesis/syn.md".into(),
                one_liner: "synth".into(),
                category: PageType::Synthesis,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    wiki.save_index(&idx).unwrap();

    let s = wiki.stats().unwrap();
    assert_eq!(s.total_pages, 4);
    assert_eq!(s.by_category.get(&PageType::Entity).copied(), Some(2));
    assert_eq!(s.by_category.get(&PageType::Concept).copied(), Some(1));
    assert_eq!(s.by_category.get(&PageType::Synthesis).copied(), Some(1));
    assert_eq!(s.by_category.get(&PageType::Summary).copied(), None);
}

#[test]
fn stats_last_activity_is_last_log_line() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.log().append("ingest", "a").unwrap();
    wiki.log().append("compact", "b").unwrap();
    let s = wiki.stats().unwrap();
    assert_eq!(s.log_entries, 2);
    let last = s.last_activity.expect("log should have last entry");
    assert!(
        last.contains("compact"),
        "last line should be compact: {}",
        last
    );
    assert!(last.contains('b'), "subject 'b' should appear: {}", last);
}

// ── Wiki::search tests (cycle 11) ────────────────────────────────────
//
// Helper: write a page at the given rel path and upsert a matching index
// entry so search can discover it via `load_index().entries`.
fn add_page(wiki: &Wiki, rel: &str, title: &str, category: PageType, body: &str) {
    add_page_with_layer(wiki, rel, title, category, crate::wiki::Layer::Kernel, body);
}

fn add_page_with_layer(
    wiki: &Wiki,
    rel: &str,
    title: &str,
    category: PageType,
    layer: crate::wiki::Layer,
    body: &str,
) {
    let page = WikiPage {
        title: title.to_string(),
        page_type: category,
        layer,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: body.to_string(),
    };
    wiki.write_page(rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: title.to_string(),
        path: rel.to_string(),
        one_liner: title.to_string(),
        category,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

#[test]
fn search_empty_query_returns_empty() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(
        &wiki,
        "entities/a.md",
        "alpha",
        PageType::Entity,
        "something",
    );
    assert!(wiki.search("").unwrap().is_empty());
    assert!(wiki.search("   ").unwrap().is_empty());
    assert!(wiki.search("\t\n").unwrap().is_empty());
}

#[test]
fn search_on_empty_wiki_returns_empty() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let hits = wiki.search("anything").unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_matches_body_case_insensitive() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(
        &wiki,
        "entities/a.md",
        "alpha",
        PageType::Entity,
        "Compaction pipeline trims context.",
    );
    let hits_lower = wiki.search("compaction").unwrap();
    assert_eq!(hits_lower.len(), 1);
    assert_eq!(hits_lower[0].path, "entities/a.md");
    assert_eq!(hits_lower[0].match_count, 1);

    let hits_upper = wiki.search("COMPACTION").unwrap();
    assert_eq!(hits_upper.len(), 1);
    assert_eq!(hits_upper[0].match_count, 1);

    let hits_mixed = wiki.search("CoMpAcTiOn").unwrap();
    assert_eq!(hits_mixed.len(), 1);
}

#[test]
fn search_matches_title() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(
        &wiki,
        "entities/a.md",
        "Compactor",
        PageType::Entity,
        "Body without the keyword.",
    );
    let hits = wiki.search("compactor").unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].title, "Compactor");
    // Snippet falls back to title when only the title matched.
    assert_eq!(hits[0].snippet, "Compactor");
}

#[test]
fn search_ranks_by_match_count_desc() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/a.md", "A", PageType::Entity, "foo");
    add_page(&wiki, "entities/b.md", "B", PageType::Entity, "foo foo foo");
    add_page(&wiki, "entities/c.md", "C", PageType::Entity, "foo foo");
    let hits = wiki.search("foo").unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].path, "entities/b.md");
    assert_eq!(hits[0].match_count, 3);
    assert_eq!(hits[1].path, "entities/c.md");
    assert_eq!(hits[1].match_count, 2);
    assert_eq!(hits[2].path, "entities/a.md");
    assert_eq!(hits[2].match_count, 1);
}

#[test]
fn search_for_kernel_identity_keeps_flat_match_count_ranking() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page_with_layer(
        &wiki,
        "concepts/host.md",
        "Host",
        PageType::Concept,
        crate::wiki::Layer::Host,
        "foo",
    );
    add_page_with_layer(
        &wiki,
        "concepts/kernel.md",
        "Kernel",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "foo foo foo",
    );

    let hits = wiki
        .search_for_identity("foo", &crate::identity::Identity::default_kernel())
        .unwrap();

    assert_eq!(hits[0].path, "concepts/kernel.md");
    assert_eq!(hits[0].layer, crate::wiki::Layer::Kernel);
    assert_eq!(hits[1].path, "concepts/host.md");
    assert_eq!(hits[1].layer, crate::wiki::Layer::Host);
}

#[test]
fn search_for_host_identity_stratifies_host_before_kernel() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page_with_layer(
        &wiki,
        "concepts/kernel.md",
        "Kernel",
        PageType::Concept,
        crate::wiki::Layer::Kernel,
        "foo foo foo",
    );
    add_page_with_layer(
        &wiki,
        "concepts/host.md",
        "Host",
        PageType::Concept,
        crate::wiki::Layer::Host,
        "foo",
    );
    let identity = crate::identity::Identity {
        mode: crate::identity::Mode::Host,
        host_project: Some("finance-app".to_string()),
        canonical_dm_revision: Some("abc123".to_string()),
        canonical_dm_repo: None,
        source: None,
    };

    let hits = wiki.search_for_identity("foo", &identity).unwrap();

    assert_eq!(hits[0].path, "concepts/host.md");
    assert_eq!(hits[0].layer, crate::wiki::Layer::Host);
    assert_eq!(hits[1].path, "concepts/kernel.md");
    assert_eq!(hits[1].layer, crate::wiki::Layer::Kernel);
}

#[test]
fn search_truncates_to_cap() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for i in 0..(SEARCH_MAX_RESULTS + 5) {
        add_page(
            &wiki,
            &format!("entities/p{:02}.md", i),
            &format!("page-{}", i),
            PageType::Entity,
            "needle body",
        );
    }
    let hits = wiki.search("needle").unwrap();
    assert_eq!(hits.len(), SEARCH_MAX_RESULTS);
}

#[test]
fn search_snippet_is_utf8_safe_around_multibyte_match() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Multibyte characters around an ASCII keyword — snippet_around
    // must clamp start/end to char boundaries.
    let body = "日本語ではこれがキーワードpipelineです。非常に長い説明文が続きます日本語テキスト。";
    add_page(&wiki, "entities/m.md", "multibyte", PageType::Entity, body);
    let hits = wiki.search("pipeline").unwrap();
    assert_eq!(hits.len(), 1);
    // Snippet is a valid UTF-8 string (implicit: Rust String invariant)
    // and contains the matched keyword.
    assert!(
        hits[0].snippet.contains("pipeline"),
        "snippet should contain keyword: {:?}",
        hits[0].snippet
    );
    // And it round-trips through char iteration without panicking on
    // incomplete codepoints.
    let _: String = hits[0].snippet.chars().collect();
}

// ── cycle-11 tester: additional search coverage ──────────────────────

// Pinned regression guard for per-page read failures. Build an index
// entry for a page that doesn't exist on disk, confirm the search
// doesn't fail as a whole — the missing page is skipped, a warning is
// pushed, and the good page still matches.
#[test]
fn search_skips_missing_page_with_warning_and_continues() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Good page — actually on disk.
    add_page(
        &wiki,
        "entities/real.md",
        "real",
        PageType::Entity,
        "findme in body",
    );

    // Index-only ghost entry — no file will be written for this path.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghost".into(),
        path: "entities/ghost.md".into(),
        one_liner: "ghost".into(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let _warnings_guard = crate::warnings::WARNINGS_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = crate::warnings::drain_warnings();
    let hits = wiki.search("findme").unwrap();
    let warnings = crate::warnings::drain_warnings();

    assert_eq!(hits.len(), 1, "real page should still match: {:?}", hits);
    assert_eq!(hits[0].path, "entities/real.md");
    assert!(
        warnings.iter().any(|w| w.contains("ghost.md")),
        "missing page should surface a warning: {:?}",
        warnings
    );
}

// Query longer than SEARCH_MAX_QUERY_LEN (200 bytes) must not panic and
// must not overflow internal buffers. Silently truncating to a char
// boundary is the documented contract.
#[test]
fn search_truncates_overlong_query_without_panic() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/p.md", "p", PageType::Entity, "hello world");
    // 500-byte ASCII query — well over the 200-byte cap.
    let long = "z".repeat(500);
    let hits = wiki.search(&long).unwrap();
    assert!(
        hits.is_empty(),
        "no real content contains 500 z's: {:?}",
        hits
    );
}

// Multi-byte query at the exact cap boundary: must clamp inward to a
// char boundary (never mid-codepoint). Uses 3-byte characters so a naive
// byte-cap at 200 would slice a codepoint; the `is_char_boundary` loop
// must back off cleanly.
#[test]
fn search_truncates_overlong_multibyte_query_to_char_boundary() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/j.md", "j", PageType::Entity, "日本語");
    // 300 copies of 3-byte char = 900 bytes; clamping at 200 bytes lands
    // between codepoints (200/3 = 66r2) — the loop must back off.
    let long: String = "日".repeat(300);
    let hits = wiki.search(&long).unwrap();
    // May or may not match depending on clamp, but must not panic.
    let _ = hits;
}

// Tiebreak contract: when match_count is equal, hits sort by path
// ascending. Locks the deterministic ordering so users see a stable
// result list across runs.
#[test]
fn search_tiebreak_path_ascending_when_counts_equal() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/z.md", "z", PageType::Entity, "foo");
    add_page(&wiki, "entities/a.md", "a", PageType::Entity, "foo");
    add_page(&wiki, "entities/m.md", "m", PageType::Entity, "foo");
    let hits = wiki.search("foo").unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].path, "entities/a.md", "order: {:?}", hits);
    assert_eq!(hits[1].path, "entities/m.md");
    assert_eq!(hits[2].path, "entities/z.md");
}

/// End-to-end validation that the C9/C10 frontmatter backfill on
/// `concepts/error-handling.md` and `concepts/module-structure.md`
/// makes those pages discoverable via `wiki.search()`. Without proper
/// `--- ... ---` frontmatter, `WikiPage::parse` returns `None` and
/// search silently skips the page with a warning. C10 verified the
/// frontmatter parses; this test verifies it survives the search path.
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips gracefully
/// when the test runs outside the package root.
#[test]
fn c9_c10_backfilled_concept_pages_are_search_findable() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(project_root) = std::env::current_dir() else {
        return; // unable to read cwd — skip
    };
    if !project_root
        .join(".dm/wiki/concepts/error-handling.md")
        .is_file()
    {
        return;
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");

    let hits = wiki.search("logging system").expect("search ok");
    assert!(
        hits.iter().any(|h| h.path == "concepts/error-handling.md"),
        "concepts/error-handling.md must surface for 'logging system' \
             after C9/C10 frontmatter backfill — got paths: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    let hits = wiki.search("Module Structure").expect("search ok");
    assert!(
        hits.iter()
            .any(|h| h.path == "concepts/module-structure.md"),
        "concepts/module-structure.md must surface for 'Module Structure' \
             after C9/C10 frontmatter backfill — got paths: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );
}

/// Regression guard: `module-structure.md` declared `scope:` in C42
/// for `src/wiki/` and `src/tui/commands/`. The new
/// `ConceptScopeUndocumented` rule (C41) should emit zero findings
/// against the live page — every `.rs` file (excluding `mod.rs`) under
/// those prefixes must remain mentioned in the page body. A future
/// cycle that adds a new submodule without updating the table trips
/// this test, surfacing the documentation gap before merge.
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips when run
/// outside the package root.
#[test]
fn module_structure_scope_undocumented_is_zero() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let project_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return,
    };
    if !project_root
        .join(".dm/wiki/concepts/module-structure.md")
        .is_file()
    {
        return;
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");
    let findings = wiki.lint().expect("lint");
    let scope_findings: Vec<_> = findings
        .iter()
        .filter(|f| {
            f.kind == WikiLintKind::ConceptScopeUndocumented
                && f.path == "concepts/module-structure.md"
        })
        .collect();
    assert!(
        scope_findings.is_empty(),
        "module-structure.md must mention every .rs file (sans mod.rs) \
             under its declared scope. Missing files: {:?}",
        scope_findings.iter().map(|f| &f.detail).collect::<Vec<_>>()
    );
}

/// C45 dogfood: error-handling.md declared file-level scope on
/// `src/logging.rs` and `src/warnings.rs`. The page already
/// mentions both extensively, so the rule should emit zero
/// findings. A future cycle that drops one of the mentions trips
/// this test by design.
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips when
/// run outside the package root.
#[test]
fn error_handling_scope_undocumented_is_zero() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let project_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return,
    };
    if !project_root
        .join(".dm/wiki/concepts/error-handling.md")
        .is_file()
    {
        return;
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");
    let findings = wiki.lint().expect("lint");
    let scope_findings: Vec<_> = findings
        .iter()
        .filter(|f| {
            f.kind == WikiLintKind::ConceptScopeUndocumented
                && f.path == "concepts/error-handling.md"
        })
        .collect();
    assert!(
        scope_findings.is_empty(),
        "error-handling.md must mention every file in its scope. \
             Missing: {:?}",
        scope_findings.iter().map(|f| &f.detail).collect::<Vec<_>>()
    );
}

/// Live regression guard: `wiki-tooling.md` declared file-level scope
/// at C46 on the four central wiki-tooling source files
/// (`wiki_lookup.rs`, `wiki_search.rs`, `fuzzy.rs`, `telemetry.rs`).
/// The page already mentions all four extensively, so the C41 rule
/// should emit zero `ConceptScopeUndocumented` findings. A future
/// cycle that drops a mention (e.g., removes the "Shared fuzzy
/// helper" section) trips this test by design.
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips when run
/// outside the package root.
#[test]
fn wiki_tooling_scope_undocumented_is_zero() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let project_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return,
    };
    if !project_root
        .join(".dm/wiki/concepts/wiki-tooling.md")
        .is_file()
    {
        return;
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");
    let findings = wiki.lint().expect("lint");
    let scope_findings: Vec<_> = findings
        .iter()
        .filter(|f| {
            f.kind == WikiLintKind::ConceptScopeUndocumented && f.path == "concepts/wiki-tooling.md"
        })
        .collect();
    assert!(
        scope_findings.is_empty(),
        "wiki-tooling.md must mention every file in its declared \
             scope. Missing: {:?}",
        scope_findings.iter().map(|f| &f.detail).collect::<Vec<_>>()
    );
}

/// Run-state self-check: `Wiki::lint()` against the live `.dm/wiki/`
/// should emit zero findings for the strict-zero rule subset.
///
/// "Strict-zero" rules are those where any finding is real drift,
/// not natural code-vs-doc divergence:
/// - `BodyPathMissing`: page mentions a `src/...rs` path that doesn't
///   exist on disk
/// - `ConceptScopeUndocumented`: page declared scope but missed a file
/// - `SourceMissing`: page's `sources:` cites a deleted file
/// - `MalformedPage`: page frontmatter fails to parse
/// - `OrphanIndexEntry`: index entry points at a missing page file
/// - `CategoryMismatch`: index entry's category disagrees with subdir
///
/// Tolerable rules (NOT in the strict subset): `EntityGap` (most
/// `.rs` files don't need entity pages), `SourceNewerThanPage`
/// (natural after edits), `ItemDrift` / `ExportDrift` (entity-page
/// edge cases), `MissingEntityKind` (legacy pages),
/// `DuplicateSource` (rare), `UntrackedPage` (synthesis pages may
/// exist before indexing).
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips when
/// run outside the package root.
#[test]
fn live_wiki_lint_has_zero_strict_findings() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let project_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return,
    };
    if !project_root.join(".dm/wiki/index.md").is_file() {
        return;
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");
    let findings = wiki.lint().expect("lint");

    let strict_kinds = [
        WikiLintKind::BodyPathMissing,
        WikiLintKind::ConceptScopeUndocumented,
        WikiLintKind::SourceMissing,
        WikiLintKind::MalformedPage,
        WikiLintKind::OrphanIndexEntry,
        WikiLintKind::CategoryMismatch,
    ];

    let strict_findings: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| strict_kinds.contains(&f.kind))
        .collect();

    assert!(
        strict_findings.is_empty(),
        "live `.dm/wiki/` has {} strict-zero finding(s):\n  {}",
        strict_findings.len(),
        strict_findings
            .iter()
            .map(|f| format!("{:?} | {} | {}", f.kind, f.path, f.detail))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// E2E validation that the snippet injected into every system prompt
/// for this project stays within the directive's 8KB ceiling AND
/// holds the C12 (preamble) + C19 (title-only) format contracts.
///
/// Reads this repo's actual `.dm/wiki/`. Skips gracefully when the
/// test runs outside the package root.
///
/// Why this test exists: the directive tracks
/// `wiki_snippet_bytes — target: <8KB` against the *real* index.
/// A unit test against a synthetic 1-entry wiki would not trip if
/// `CONTEXT_SNIPPET_MAX_BYTES` were widened or the preamble grew
/// unboundedly. This test pins the contract end-to-end against the
/// repo's actual ~3-page index.
#[test]
fn snippet_against_repo_wiki_fits_directive_budget_and_format() {
    const DIRECTIVE_CEILING: usize = 8 * 1024;

    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(project_root) = std::env::current_dir() else {
        return;
    };
    if !project_root.join(".dm/wiki/index.md").is_file() {
        return; // running outside the package root — skip
    }
    let wiki = Wiki::open(&project_root).expect("open repo wiki");
    let Some(snippet) = wiki.context_snippet() else {
        return; // wiki has no entries — skip
    };

    // (a) Directive byte budget.
    assert!(
        snippet.len() < DIRECTIVE_CEILING,
        "snippet must fit the directive's <8KB ceiling — got {} bytes",
        snippet.len()
    );

    // (b) C12 preamble survived intact.
    assert!(
        snippet.starts_with("## dark-matter Wiki\n"),
        "preamble header missing — snippet preview: {}",
        &snippet[..snippet.len().min(120)]
    );
    assert!(
        snippet.contains("wiki_search") && snippet.contains("wiki_lookup"),
        "C12 tool-recommendation preamble missing"
    );

    // (c) C19 title-only format — no per-line `.dm/wiki/...` paths
    // appear in the rendered TOC. The truncation tail (which names
    // `.dm/wiki/index.md` once at the bottom) is allowed; the per-
    // entry lines are NOT.
    //
    // Strategy: split off the truncation tail before scanning for
    // `.dm/wiki/`. The tail begins with "\n… " (the ellipsis line).
    let body = match snippet.find("\n… ") {
        Some(idx) => &snippet[..idx],
        None => snippet.as_str(),
    };
    assert!(
        !body.contains(".dm/wiki/entities/"),
        "C19 format violation — per-line entity path leaked: \n{}",
        body
    );
    assert!(
        !body.contains(".dm/wiki/concepts/"),
        "C19 format violation — per-line concept path leaked: \n{}",
        body
    );
}

/// Scan a `.dm/wiki/concepts/*.md` page for `src/.../X.rs` paths
/// and return any that don't exist on disk under `project_root`.
/// Empty Vec means clean. Used by the per-page phantom-file
/// regression tests below.
///
/// Hand-rolled byte scan — leading `src/` and trailing `.rs` are
/// anchors; the inter-anchor character set covers every legitimate
/// Rust path the project uses. Avoids dragging in `regex` for a
/// one-off check. Deduplicates via `BTreeSet`.
fn collect_phantom_paths(
    project_root: &std::path::Path,
    page_path: &std::path::Path,
) -> Vec<String> {
    let Ok(body) = std::fs::read_to_string(page_path) else {
        return Vec::new();
    };
    super::lint::body_src_path_candidates(&body)
        .into_iter()
        .filter(|p| !project_root.join(p).is_file())
        .collect()
}

/// Phantom-file drift guard for `concepts/module-structure.md`.
///
/// Every `src/.../X.rs` path the page mentions must exist on disk.
/// Catches the C23 failure mode (page claimed `src/api/handlers.rs`
/// and `src/api/state.rs` for an over-eager split that never
/// happened — silently broken until manually audited).
///
/// Does NOT enforce the inverse (every `src/**.rs` must be
/// documented) — the page intentionally uses "and ~15 others"
/// for `src/tools/`. That's a future-cycle problem.
///
/// Read-only. Skips gracefully when run outside the package root.
#[test]
fn module_structure_page_names_no_phantom_files() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(project_root) = std::env::current_dir() else {
        return;
    };
    let page = project_root.join(".dm/wiki/concepts/module-structure.md");
    if !page.is_file() {
        return;
    }
    let phantoms = collect_phantom_paths(&project_root, &page);
    assert!(
        phantoms.is_empty(),
        "concepts/module-structure.md mentions {} phantom file(s):\n  {}",
        phantoms.len(),
        phantoms.join("\n  ")
    );
}

/// Phantom-file drift guard for `concepts/error-handling.md`. Same
/// contract as the module-structure guard above; mechanically
/// extended via the shared `collect_phantom_paths` helper so all
/// three audited concept pages stay aligned with disk reality.
#[test]
fn error_handling_page_names_no_phantom_files() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(project_root) = std::env::current_dir() else {
        return;
    };
    let page = project_root.join(".dm/wiki/concepts/error-handling.md");
    if !page.is_file() {
        return;
    }
    let phantoms = collect_phantom_paths(&project_root, &page);
    assert!(
        phantoms.is_empty(),
        "concepts/error-handling.md mentions {} phantom file(s):\n  {}",
        phantoms.len(),
        phantoms.join("\n  ")
    );
}

/// Phantom-file drift guard for `concepts/wiki-tooling.md`. Same
/// contract as the module-structure guard above.
#[test]
fn wiki_tooling_page_names_no_phantom_files() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(project_root) = std::env::current_dir() else {
        return;
    };
    let page = project_root.join(".dm/wiki/concepts/wiki-tooling.md");
    if !page.is_file() {
        return;
    }
    let phantoms = collect_phantom_paths(&project_root, &page);
    assert!(
        phantoms.is_empty(),
        "concepts/wiki-tooling.md mentions {} phantom file(s):\n  {}",
        phantoms.len(),
        phantoms.join("\n  ")
    );
}

/// Negative-path validation for `collect_phantom_paths`. The C25/C26
/// positive tests prove "current concept pages have no phantoms" but
/// would also pass if the scan logic were broken (always returned
/// empty). This test fixes that tautology by feeding a known-phantom
/// fixture through the same helper and asserting the phantoms are
/// detected and the real path is not.
#[test]
fn collect_phantom_paths_detects_known_phantoms_in_fixture() {
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();

    // Create exactly one real source file. Anything else mentioned
    // in the fixture page is a phantom by construction.
    std::fs::create_dir_all(project_root.join("src/wiki")).unwrap();
    std::fs::write(project_root.join("src/wiki/mod.rs"), "// real source file").unwrap();

    // Three phantoms — including the historical C23 case
    // (`src/api/state.rs`) so the test names what bug it guards.
    let page_path = project_root.join("fixture.md");
    let fixture = "---\n\
                       title: Fixture\n\
                       type: concept\n\
                       sources:\n\
                         - src/wiki/mod.rs\n\
                       last_updated: 2026-04-25 00:00:00\n\
                       ---\n\
                       # Fixture\n\
                       \n\
                       Real path: `src/wiki/mod.rs` exists on disk.\n\
                       \n\
                       Phantom paths (should all be flagged):\n\
                       - `src/wiki/nonexistent.rs`\n\
                       - `src/api/state.rs`\n\
                       - `src/imaginary/path.rs`\n";
    std::fs::write(&page_path, fixture).unwrap();

    let phantoms = collect_phantom_paths(project_root, &page_path);

    assert!(
        !phantoms.iter().any(|p| p == "src/wiki/mod.rs"),
        "real path must NOT be flagged as phantom: {:?}",
        phantoms
    );
    assert!(
        phantoms.iter().any(|p| p == "src/wiki/nonexistent.rs"),
        "must detect src/wiki/nonexistent.rs: {:?}",
        phantoms
    );
    assert!(
        phantoms.iter().any(|p| p == "src/api/state.rs"),
        "must detect src/api/state.rs (the C23 historical phantom): {:?}",
        phantoms
    );
    assert!(
        phantoms.iter().any(|p| p == "src/imaginary/path.rs"),
        "must detect src/imaginary/path.rs: {:?}",
        phantoms
    );
    assert_eq!(
        phantoms.len(),
        3,
        "expected exactly 3 phantoms (1 real + 3 phantom paths in fixture): {:?}",
        phantoms
    );
}

#[test]
fn collect_phantom_paths_respects_nested_src_prefixes() {
    let tmp = TempDir::new().unwrap();
    let project_root = tmp.path();

    std::fs::create_dir_all(project_root.join("examples/host-skeleton/src")).unwrap();
    std::fs::write(
        project_root.join("examples/host-skeleton/src/host_caps.rs"),
        "// real nested source file",
    )
    .unwrap();

    let page_path = project_root.join("fixture.md");
    let fixture = "# Fixture\n\n\
        Nested real path: `examples/host-skeleton/src/host_caps.rs`.\n";
    std::fs::write(&page_path, fixture).unwrap();

    let phantoms = collect_phantom_paths(project_root, &page_path);
    assert!(
        phantoms.is_empty(),
        "nested src/ path should resolve from its full prefix, not as src/host_caps.rs: {:?}",
        phantoms
    );
}

#[test]
fn body_src_path_candidates_extracts_root_and_nested_paths() {
    let body = "\
        Root path: `src/wiki/lint.rs`.\n\
        Nested path: `examples/host-skeleton/src/host_caps.rs`.\n\
        Duplicate root path: `src/wiki/lint.rs`.\n";

    let candidates = super::lint::body_src_path_candidates(body);
    let expected: std::collections::BTreeSet<String> = [
        "examples/host-skeleton/src/host_caps.rs".to_string(),
        "src/wiki/lint.rs".to_string(),
    ]
    .into_iter()
    .collect();

    assert_eq!(candidates, expected);
}

// Whitespace-only query is not the same as empty string — must still
// return empty without erroring. Builder's `search_empty_query_returns_empty`
// tests `""` explicitly; this locks the same behavior for `"   \t\n"`.
#[test]
fn search_whitespace_only_query_returns_empty() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/p.md", "p", PageType::Entity, "hello");
    let hits = wiki.search("   \t\n  ").unwrap();
    assert!(hits.is_empty());
}

// UTF-8 case-folding byte-length drift. Turkish `İ` (2 bytes) lowercases
// to `i` + combining dot (3 bytes), so `body_lc.find(needle)` can return
// an offset that doesn't correspond to the same position in `page.body`.
// The current implementation feeds that offset straight into
// `snippet_around(&page.body, ...)`. For short bodies the 60-byte window
// typically still covers the true match position, so the keyword lands
// in the snippet and this test passes; it guards against regressions
// where the drift isn't absorbed by the context window. A matching
// `#[ignore]`d pathological-case test below documents the failure mode
// for large bodies with many length-changing characters.
#[test]
fn search_snippet_contains_keyword_with_turkish_prefix() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let body = format!("{} hello world", "İ".repeat(3));
    add_page(&wiki, "entities/tr.md", "tr", PageType::Entity, &body);
    let hits = wiki.search("hello").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet.contains("hello"),
        "snippet should contain keyword with small Turkish prefix: {:?}",
        hits[0].snippet
    );
}

// Pathological byte-length drift: many Turkish `İ`s BEFORE the match.
// Before the cycle-12 fix, `body_lc.find(needle)` returned an offset
// past `page.body.len()` (positive drift +1 byte per İ), so snippet
// extraction off `page.body` yielded an empty/"…"-only window that
// didn't contain the keyword. Since cycle-12, the snippet is sourced
// from `body_lc` — same string the offset points into — so the window
// is always valid. Trade: the snippet text is lowercased.
#[test]
fn search_snippet_handles_case_expanding_prefix() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // 200 × 1-byte drift = 200 bytes between body and body_lc. That's
    // way past the 60-byte snippet side, so the pre-fix window fell
    // off the end of page.body.
    let body = format!("{} hello world", "İ".repeat(200));
    add_page(&wiki, "entities/tr.md", "tr", PageType::Entity, &body);
    let hits = wiki.search("hello").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet.contains("hello"),
        "snippet should contain matched keyword even with heavy byte-length \
             drift from to_lowercase(); got: {:?}",
        hits[0].snippet
    );
    assert!(
        hits[0].snippet.contains("hello world"),
        "snippet should contain the full matched phrase with ~60 chars of \
             surrounding context; got: {:?}",
        hits[0].snippet
    );
    // Snippet is sourced from body_lc, so the raw uppercase İ must
    // never appear — only its lowercase form (i with combining dot).
    assert!(
        !hits[0].snippet.contains('İ'),
        "snippet is sourced from body_lc; raw İ should never appear; got: {:?}",
        hits[0].snippet
    );
}

// Negative-drift direction of BUG-11-1: German `ẞ` (3 bytes) lowercases
// to `ß` (2 bytes). 200 × ẞ means body_lc is 200 bytes SHORTER than
// page.body. Before cycle-12, body_lc.find(keyword) would land at an
// offset that, applied to page.body, centers the snippet ~200 bytes
// earlier in body — missing the keyword entirely. This is the
// symmetric case of `search_snippet_handles_case_expanding_prefix`:
// guards against a future refactor "optimizing" the fix back to
// &page.body without realizing both drift directions break.
#[test]
fn search_snippet_handles_case_shrinking_prefix() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let body = format!("{} foundit", "ẞ".repeat(200));
    add_page(&wiki, "entities/de.md", "de", PageType::Entity, &body);
    let hits = wiki.search("foundit").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet.contains("foundit"),
        "snippet must contain keyword even with shrinking-case drift; got: {:?}",
        hits[0].snippet
    );
}

// Documents the intentional cycle-12 trade-off: the snippet is
// sourced from the lowercased body, so ASCII mixed-case in the body
// appears lowercased in the snippet. Locking this so a future
// "preserve original casing" change is a deliberate decision with
// explicit test churn, not a silent regression of the fix.
#[test]
fn search_snippet_uses_lowercased_context_intentional_trade() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(
        &wiki,
        "entities/c.md",
        "c",
        PageType::Entity,
        "AAAA BIGWORD BBBB",
    );
    let hits = wiki.search("BIGWORD").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet.contains("bigword"),
        "snippet should show the matched keyword in lowercase (body_lc sourced); got: {:?}",
        hits[0].snippet
    );
    assert!(
        !hits[0].snippet.contains("BIGWORD"),
        "original uppercase should NOT appear in snippet (body_lc sourced); got: {:?}",
        hits[0].snippet
    );
}

// ── cycle-12 tester: post-fix coverage ──────────────────────────────

// After the BUG-11-1 fix, the 160-byte snippet cap clamps against
// `body_lc`, not `page.body`. When the body also has case-expanding
// prefix drift, both clamps need to compose: the `body_lc` slice fits
// under SEARCH_SNIPPET_MAX and the keyword still appears. Protects
// against a future refactor that tries to re-apply the cap in the
// wrong coordinate system.
#[test]
fn search_snippet_cap_composes_with_case_expanding_prefix() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Long case-expanding prefix + lots of ASCII context + keyword +
    // more ASCII context. Ensures the snippet is cap-bounded (not
    // slice-bounded by body_lc edges) while match_byte_idx is well
    // past page.body.len() due to drift.
    let mut body = "İ".repeat(300);
    body.push_str(&"z".repeat(500));
    body.push_str(" hello ");
    body.push_str(&"z".repeat(500));
    add_page(&wiki, "entities/cap.md", "cap", PageType::Entity, &body);
    let hits = wiki.search("hello").unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet.contains("hello"),
        "snippet must contain keyword with composed cap + drift; got len={} snip={:?}",
        hits[0].snippet.len(),
        hits[0].snippet
    );
    // The snippet is capped at SEARCH_SNIPPET_MAX (160) plus the
    // trailing ellipsis byte width. UTF-8 '…' is 3 bytes; the cap loop
    // truncates BEFORE appending '…', so final byte length is
    // ≤ SEARCH_SNIPPET_MAX + '…'.len_utf8() = 163. Lock that bound.
    assert!(
        hits[0].snippet.len() <= SEARCH_SNIPPET_MAX + '…'.len_utf8(),
        "snippet must honor SEARCH_SNIPPET_MAX cap even with drift; got {} bytes",
        hits[0].snippet.len()
    );
    // Validity — chars() round-trips without panicking.
    let _: String = hits[0].snippet.chars().collect();
}

// Case-expanding NEEDLE (as opposed to body): the query itself contains
// a character whose `to_lowercase()` byte length differs from its
// original. Both sides of the comparison are lowercased, so this must
// still match. Previously unexercised — ensures the normalization is
// symmetric in query direction too.
#[test]
fn search_matches_body_when_needle_is_case_expanding() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Body contains the *uppercase* form İ; query is the uppercase
    // form as well. After `to_lowercase()`, both sides become "i̇"
    // (i + combining dot above, 3 bytes). Match count should be 1.
    let body = "header text İmeta tag here.";
    add_page(&wiki, "entities/n.md", "n", PageType::Entity, body);
    let hits = wiki.search("İ").unwrap();
    assert_eq!(hits.len(), 1, "case-expanding needle must still match");
    assert!(
        hits[0].match_count >= 1,
        "match_count should be ≥1: {:?}",
        hits[0]
    );
    // Snippet is body_lc-sourced, so raw uppercase İ never appears.
    assert!(
        !hits[0].snippet.contains('İ'),
        "snippet is lowercased; raw İ forbidden: {:?}",
        hits[0].snippet
    );
}

// ── Wiki::lint tests (cycle 13) ──────────────────────────────────────
//
// `add_page` (defined above) writes both the page file and upserts a
// matching index entry — i.e., it produces a consistent state. For
// lint tests that need divergence between disk and index, either
// (a) call `write_page` without updating the index, or (b) mutate
// the index directly via `save_index` to produce a phantom entry.

#[test]
fn lint_clean_wiki_returns_no_findings() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page(&wiki, "entities/a.md", "a", PageType::Entity, "body");
    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "expected no findings, got {:?}",
        findings
    );
}

#[test]
fn lint_detects_orphan_index_entry() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghost".to_string(),
        path: "entities/ghost.md".to_string(),
        one_liner: "vanished".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "entities/ghost.md");
    assert!(findings[0].detail.contains("ghost"));
}

#[test]
fn lint_detects_untracked_page() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = WikiPage {
        title: "loose".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("concepts/loose.md", &page).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::UntrackedPage);
    assert_eq!(findings[0].path, "concepts/loose.md");
}

#[test]
fn lint_detects_category_mismatch() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Write the file so the finding is NOT also an orphan.
    let page = WikiPage {
        title: "miscat".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("concepts/foo.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "miscat".to_string(),
        path: "concepts/foo.md".to_string(),
        one_liner: "category wrong".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::CategoryMismatch);
    assert_eq!(findings[0].path, "concepts/foo.md");
}

#[test]
fn lint_multiple_findings_sorted_deterministically() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let empty_page = |cat: PageType| WikiPage {
        title: "x".to_string(),
        page_type: cat,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    // One untracked page.
    wiki.write_page("summaries/loose.md", &empty_page(PageType::Summary))
        .unwrap();
    // Two category-mismatch pages (must exist on disk, or they'd
    // also count as orphans and skew the finding count).
    wiki.write_page("entities/m1.md", &empty_page(PageType::Entity))
        .unwrap();
    wiki.write_page("synthesis/m2.md", &empty_page(PageType::Synthesis))
        .unwrap();

    let mut idx = wiki.load_index().unwrap();
    // Two orphan index entries (no file on disk).
    idx.entries.push(IndexEntry {
        title: "g2".to_string(),
        path: "concepts/g2.md".to_string(),
        one_liner: String::new(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "g1".to_string(),
        path: "entities/g1.md".to_string(),
        one_liner: String::new(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    // Two mismatches — category field disagrees with path prefix.
    idx.entries.push(IndexEntry {
        title: "m1".to_string(),
        path: "entities/m1.md".to_string(),
        one_liner: String::new(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "m2".to_string(),
        path: "synthesis/m2.md".to_string(),
        one_liner: String::new(),
        category: PageType::Summary,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 5, "findings = {:?}", findings);

    // Sort: OrphanIndexEntry < UntrackedPage < CategoryMismatch,
    // each group sorted by path ascending.
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "concepts/g2.md");
    assert_eq!(findings[1].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[1].path, "entities/g1.md");
    assert_eq!(findings[2].kind, WikiLintKind::UntrackedPage);
    assert_eq!(findings[2].path, "summaries/loose.md");
    assert_eq!(findings[3].kind, WikiLintKind::CategoryMismatch);
    assert_eq!(findings[3].path, "entities/m1.md");
    assert_eq!(findings[4].kind, WikiLintKind::CategoryMismatch);
    assert_eq!(findings[4].path, "synthesis/m2.md");
}

#[test]
fn lint_ignores_non_md_files_and_subdirs_under_categories() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Non-.md: manual notes must not trigger UntrackedPage.
    fs::write(wiki.root().join("entities/README.txt"), "hello").unwrap();
    // Nested .md: one-level-deep convention; nested pages are out
    // of scope for cycle-13 lint.
    fs::create_dir_all(wiki.root().join("entities/subdir")).unwrap();
    fs::write(wiki.root().join("entities/subdir/nested.md"), "# nested\n").unwrap();

    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "expected no findings, got {:?}",
        findings
    );
}

#[test]
fn lint_tolerates_missing_category_dir() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Simulate a partial scaffold — synthesis dir removed between
    // `Wiki::open` and `lint()`. The read-only lint must not panic
    // or error; must report no findings caused by the absence.
    fs::remove_dir_all(wiki.root().join("synthesis")).unwrap();
    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "expected no findings, got {:?}",
        findings
    );
}

// Rules 1 and 3 are independent: an entry can be simultaneously
// orphan (file missing) AND category-mismatched (path doesn't match
// category). Lint must emit BOTH findings for the same entry.
#[test]
fn lint_orphan_and_category_mismatch_fire_independently_for_same_entry() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghostmiscat".to_string(),
        // path is under entities/, but category says Concept, AND the
        // file is not on disk — both rules 1 and 3 should trigger.
        path: "entities/gm.md".to_string(),
        one_liner: "vanished and miscategorized".to_string(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 2, "findings = {:?}", findings);
    // Sort order: Orphan (0) < CategoryMismatch (2), with UntrackedPage
    // (1) absent — so indices are [0]=Orphan, [1]=CategoryMismatch.
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "entities/gm.md");
    assert_eq!(findings[1].kind, WikiLintKind::CategoryMismatch);
    assert_eq!(findings[1].path, "entities/gm.md");
}

// `Wiki::lint()` uses `load_index().unwrap_or_default()`, and
// `WikiIndex::parse` is tolerant — malformed lines are skipped. A
// garbage `index.md` must still let lint run: zero index entries →
// all on-disk pages surface as UntrackedPage, none as orphan or
// mismatch. Locks the recovery story: a corrupt index is never a
// hard failure for /wiki lint.
#[test]
fn lint_with_unparseable_index_surfaces_disk_pages_as_untracked() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Put two real pages on disk (consistent with index if we hadn't
    // stomped it — but we do, below).
    let page = |cat: PageType| WikiPage {
        title: "t".to_string(),
        page_type: cat,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: "2026-04-17 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("entities/a.md", &page(PageType::Entity))
        .unwrap();
    wiki.write_page("concepts/b.md", &page(PageType::Concept))
        .unwrap();
    // Stomp index.md with garbage. WikiIndex::parse skips every
    // unrecognized line, yielding an empty WikiIndex.
    fs::write(
        wiki.root().join("index.md"),
        "not a wiki index\n<<< corrupted >>>\n- [bogus] no link\nrandom garbage\n",
    )
    .unwrap();

    let findings = wiki.lint().unwrap();
    // Both disk pages have no matching index entry → UntrackedPage.
    assert_eq!(findings.len(), 2, "findings = {:?}", findings);
    for f in &findings {
        assert_eq!(
            f.kind,
            WikiLintKind::UntrackedPage,
            "corrupt index must not yield orphan/mismatch findings: {:?}",
            f
        );
    }
    // Sort by path within the kind group: concepts/b.md < entities/a.md.
    assert_eq!(findings[0].path, "concepts/b.md");
    assert_eq!(findings[1].path, "entities/a.md");
}

// ── Wiki::lint source-drift tests (cycle 14) ─────────────────────────
//
// These exercise rules 4a `SourceMissing` and 4b `SourceNewerThanPage`
// added in cycle 14. The project root (for resolving source paths) is
// `wiki.root().parent().parent()`, which in tests is `tmp.path()`.

/// Fixed `last_updated` baseline the helpers below line up against.
/// Picked far enough in the past that `set_mtime` can place source
/// mtime on either side without bumping against "now".
const TEST_PAGE_TS_STR: &str = "2026-04-17 00:00:00";

/// Write a page at `rel` with the given sources + upsert a matching
/// index entry. Mirrors `add_page` above but exposes `sources` and
/// uses the cycle-14 baseline timestamp.
fn add_page_with_sources(
    wiki: &Wiki,
    rel: &str,
    title: &str,
    category: PageType,
    sources: Vec<String>,
    last_updated: &str,
) {
    // Mirror `ingest_file`'s schema: an entity page with `.rs` sources
    // populates `entity_kind`. Without this, legacy fixtures would trip
    // the Rule 8 `MissingEntityKind` lint that isn't the focus of the
    // test using the helper.
    let entity_kind = if category == PageType::Entity && sources.iter().any(|s| s.ends_with(".rs"))
    {
        Some(EntityKind::Unknown)
    } else {
        None
    };
    let page = WikiPage {
        title: title.to_string(),
        page_type: category,
        layer: crate::wiki::Layer::Kernel,
        sources,
        last_updated: last_updated.to_string(),
        entity_kind,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page(rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: title.to_string(),
        path: rel.to_string(),
        one_liner: title.to_string(),
        category,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

/// Set the mtime of an on-disk file explicitly so source-drift tests
/// don't race against wall-clock. Uses the stable-since-1.75 stdlib
/// `FileTimes::set_modified`, avoiding a dev-dep on `filetime`.
fn set_mtime(path: &Path, t: std::time::SystemTime) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open source for set_times");
    let times = std::fs::FileTimes::new().set_modified(t);
    file.set_times(times).expect("set_times");
}

#[test]
fn lint_index_timestamp_drift_flags_stale_index_cache() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    add_page_with_sources(
        &wiki,
        "concepts/p.md",
        "p",
        PageType::Concept,
        vec![],
        "2026-04-26 19:45:00",
    );
    let mut idx = wiki.load_index().unwrap();
    idx.entries[0].last_updated = Some("2026-04-26 17:30:00".to_string());
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::IndexTimestampDrift);
    assert_eq!(findings[0].path, "concepts/p.md");
    assert!(
        findings[0].detail.contains("Try:"),
        "detail must include next-step hint: {:?}",
        findings[0].detail
    );
}

#[test]
fn lint_source_missing_flagged() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Source listed on the page never existed — SourceMissing fires.
    add_page_with_sources(
        &wiki,
        "entities/p.md",
        "p",
        PageType::Entity,
        vec!["src/ghost.rs".to_string()],
        TEST_PAGE_TS_STR,
    );
    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::SourceMissing);
    assert_eq!(findings[0].path, "entities/p.md");
    assert!(
        findings[0].detail.contains("src/ghost.rs"),
        "detail must name the missing source: {:?}",
        findings[0].detail,
    );
}

#[test]
fn lint_body_path_missing_fires_for_body_text_phantom() {
    // C29: BodyPathMissing rule. A page that mentions a
    // `src/.../X.rs` path in body text (not in `sources:`) but the
    // file doesn't exist must surface a BodyPathMissing finding.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();

    // Real source file: gets listed in `sources:`.
    std::fs::create_dir_all(proj.join("src/wiki")).unwrap();
    std::fs::write(proj.join("src/wiki/mod.rs"), "// real").unwrap();

    // Page with a body-only phantom mention.
    let page = WikiPage {
        title: "Drift Probe".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/wiki/mod.rs".to_string()],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# Drift\n\nReal: `src/wiki/mod.rs`. \
                   Phantom: `src/imaginary/path.rs` (not on disk).\n"
            .to_string(),
    };
    wiki.write_page("concepts/drift-probe.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/drift-probe.md".to_string(),
        title: "Drift Probe".to_string(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    let body_phantoms: Vec<_> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::BodyPathMissing)
        .collect();
    assert_eq!(
        body_phantoms.len(),
        1,
        "expected exactly one BodyPathMissing finding: {:?}",
        findings
    );
    assert!(
        body_phantoms[0].detail.contains("src/imaginary/path.rs"),
        "finding must name the phantom path: {}",
        body_phantoms[0].detail
    );
    assert_eq!(body_phantoms[0].path, "concepts/drift-probe.md");
}

#[test]
fn lint_body_path_missing_skips_paths_in_sources_block() {
    // Dedup: if a phantom is in BOTH `sources:` AND body, only
    // SourceMissing fires. BodyPathMissing's job is body-only.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    let page = WikiPage {
        title: "Dup".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/imaginary/path.rs".to_string()], // phantom in sources
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "Body also names `src/imaginary/path.rs` here.".to_string(),
    };
    wiki.write_page("concepts/dup.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/dup.md".to_string(),
        title: "Dup".to_string(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let source_missing_count = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::SourceMissing)
        .count();
    let body_missing_count = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::BodyPathMissing)
        .count();
    assert_eq!(source_missing_count, 1, "should fire as SourceMissing");
    assert_eq!(
        body_missing_count, 0,
        "should NOT also fire BodyPathMissing"
    );
}

#[test]
fn lint_concept_scope_undocumented_fires_when_file_not_mentioned() {
    // C41 positive: page declares scope but body mentions only one
    // of two .rs files. The unmentioned file fires the rule. Also
    // verifies mod.rs skip — a mod.rs file is not mentioned but must
    // not appear in any finding.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src/probe")).unwrap();
    std::fs::write(proj.join("src/probe/mod.rs"), "// mod").unwrap();
    std::fs::write(proj.join("src/probe/foo.rs"), "// foo").unwrap();
    std::fs::write(proj.join("src/probe/bar.rs"), "// bar").unwrap();
    let page = WikiPage {
        title: "Probe Module".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec!["src/probe/".to_string()],
        body: "Mentions only foo (not the other one).".to_string(),
    };
    wiki.write_page("concepts/probe-module.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/probe-module.md".to_string(),
        title: page.title.clone(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let scope_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ConceptScopeUndocumented)
        .collect();
    assert_eq!(
        scope_findings.len(),
        1,
        "exactly one undoc'd file expected: {:?}",
        findings
    );
    assert!(
        scope_findings[0].detail.contains("src/probe/bar.rs"),
        "must name bar.rs: {}",
        scope_findings[0].detail
    );
    // mod.rs skip verification — no finding mentions mod.rs.
    for f in &findings {
        assert!(
            !f.detail.contains("src/probe/mod.rs"),
            "mod.rs must be skipped: {:?}",
            f
        );
    }
}

#[test]
fn lint_concept_scope_undocumented_silent_when_all_mentioned() {
    // C41 negative: page declares scope and body mentions every
    // .rs file in the scope. No ConceptScopeUndocumented finding.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src/cover")).unwrap();
    std::fs::write(proj.join("src/cover/alpha.rs"), "// a").unwrap();
    std::fs::write(proj.join("src/cover/beta.rs"), "// b").unwrap();
    let page = WikiPage {
        title: "Cover Module".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec!["src/cover/".to_string()],
        body: "Lists alpha and beta both, fully documented.".to_string(),
    };
    wiki.write_page("concepts/cover-module.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/cover-module.md".to_string(),
        title: page.title.clone(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let scope_count = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ConceptScopeUndocumented)
        .count();
    assert_eq!(
        scope_count, 0,
        "all files mentioned — no ConceptScopeUndocumented: {:?}",
        findings
    );
}

#[test]
fn lint_concept_scope_undocumented_skips_pages_without_scope() {
    // C41 opt-in: page has empty scope. Even with .rs files present
    // in the project, the rule must not fire.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src/orphan")).unwrap();
    std::fs::write(proj.join("src/orphan/lonely.rs"), "// lonely").unwrap();
    let page = WikiPage {
        title: "Unscoped Concept".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "Has nothing to say about lonely.".to_string(),
    };
    wiki.write_page("concepts/unscoped.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/unscoped.md".to_string(),
        title: page.title.clone(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let scope_count = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ConceptScopeUndocumented)
        .count();
    assert_eq!(scope_count, 0, "empty scope must not fire: {:?}", findings);
}

#[test]
fn lint_concept_scope_undocumented_fires_for_file_level_scope() {
    // C45: scope entries that are .rs files (not directories) get
    // a per-file mention check, not a walk. Page declares two real
    // files in scope; body mentions only one. The unmentioned file
    // fires the rule with the "specific file" branch detail.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/logging.rs"), "// real").unwrap();
    std::fs::write(proj.join("src/warnings.rs"), "// real").unwrap();
    let page = WikiPage {
        title: "File-Scoped Probe".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec!["src/logging.rs".to_string(), "src/warnings.rs".to_string()],
        body: "Mentions logging only.".to_string(),
    };
    wiki.write_page("concepts/probe.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/probe.md".to_string(),
        title: page.title.clone(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let scope_findings: Vec<_> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ConceptScopeUndocumented)
        .collect();
    assert_eq!(
        scope_findings.len(),
        1,
        "exactly one missing: {:?}",
        findings
    );
    assert!(
        scope_findings[0].detail.contains("src/warnings.rs"),
        "must name warnings.rs: {}",
        scope_findings[0].detail
    );
    assert!(
        scope_findings[0].detail.contains("specific file"),
        "must name file-level branch: {}",
        scope_findings[0].detail
    );
}

#[test]
fn lint_concept_scope_undocumented_silent_for_file_level_scope_when_mentioned() {
    // C45: same fixture as the positive file-level test, but the
    // body mentions both files. Zero findings.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/logging.rs"), "// real").unwrap();
    std::fs::write(proj.join("src/warnings.rs"), "// real").unwrap();
    let page = WikiPage {
        title: "Both Mentioned".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec!["src/logging.rs".to_string(), "src/warnings.rs".to_string()],
        body: "Mentions logging and warnings together.".to_string(),
    };
    wiki.write_page("concepts/both.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        path: "concepts/both.md".to_string(),
        title: page.title.clone(),
        category: PageType::Concept,
        one_liner: "test".to_string(),
        last_updated: Some(TEST_PAGE_TS_STR.to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
    let findings = wiki.lint().unwrap();
    let scope_count = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ConceptScopeUndocumented)
        .count();
    assert_eq!(scope_count, 0, "no findings expected: {:?}", findings);
}

#[test]
fn lint_source_newer_than_page_flagged() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Create a real source file under the project root.
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let src_rel = "src/main.rs";
    let src_abs = tmp.path().join(src_rel);
    std::fs::write(&src_abs, "fn main() {}").unwrap();
    // Bump mtime to one day AFTER the page's last_updated.
    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();
    let newer = page_ts + std::time::Duration::from_secs(86_400);
    set_mtime(&src_abs, newer);

    add_page_with_sources(
        &wiki,
        "entities/main.md",
        "main",
        PageType::Entity,
        vec![src_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::SourceNewerThanPage);
    assert_eq!(findings[0].path, "entities/main.md");
    assert!(
        findings[0].detail.contains(src_rel),
        "detail must name the stale source: {:?}",
        findings[0].detail,
    );
}

#[test]
fn lint_source_same_age_or_older_not_flagged() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();

    // (a) older: mtime = page_ts - 1 day.
    let older_rel = "src/older.rs";
    let older_abs = tmp.path().join(older_rel);
    std::fs::write(&older_abs, "// older").unwrap();
    set_mtime(&older_abs, page_ts - std::time::Duration::from_secs(86_400));

    // (b) same-instant: mtime = page_ts exactly. `mtime > page_ts` is
    // strict, so equal must NOT fire.
    let same_rel = "src/same.rs";
    let same_abs = tmp.path().join(same_rel);
    std::fs::write(&same_abs, "// same").unwrap();
    set_mtime(&same_abs, page_ts);

    add_page_with_sources(
        &wiki,
        "entities/fresh.md",
        "fresh",
        PageType::Entity,
        vec![older_rel.to_string(), same_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "expected no findings, got {:?}",
        findings
    );
}

#[test]
fn lint_synthesis_pages_skipped_for_source_drift() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Even with an obviously missing source, synthesis pages must not
    // produce a SourceMissing finding — their `sources` are compaction
    // contributors, not tracked files.
    add_page_with_sources(
        &wiki,
        "synthesis/overview.md",
        "overview",
        PageType::Synthesis,
        vec!["contributor/does/not/exist.md".to_string()],
        TEST_PAGE_TS_STR,
    );
    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "synthesis pages must be skipped for source-drift, got {:?}",
        findings,
    );
}

#[test]
fn lint_page_with_empty_sources_not_drift_checked() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // No sources at all — nothing to check. Must produce zero findings
    // (not even iterate into fs::metadata).
    add_page_with_sources(
        &wiki,
        "entities/empty.md",
        "empty",
        PageType::Entity,
        vec![],
        TEST_PAGE_TS_STR,
    );
    let findings = wiki.lint().unwrap();
    assert!(
        findings.is_empty(),
        "expected no findings, got {:?}",
        findings
    );
}

#[test]
fn lint_malformed_last_updated_yields_malformed_page_finding() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Garbage timestamp → parse_page_timestamp returns None. Cycle 15
    // surfaces this as a `MalformedPage` finding instead of a silent
    // skip. The source-drift check must still short-circuit (zero
    // SourceMissing/SourceNewerThanPage findings) — the malformed
    // page is a blocker, not a source-drift candidate.
    add_page_with_sources(
        &wiki,
        "entities/bad.md",
        "bad",
        PageType::Entity,
        vec!["src/anything.rs".to_string()],
        "not a real timestamp",
    );
    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[0].path, "entities/bad.md");
    assert!(
        findings[0].detail.contains("not a real timestamp"),
        "detail must echo the bad value for debuggability: {:?}",
        findings[0].detail,
    );
}

// One page with four sources: two missing, one stale, one fresh.
// Rule 4 iterates the whole `sources` list; each problematic source
// becomes its OWN finding, all attributed to the same page path.
// Locks per-source independence and that a single page can appear
// in both SourceMissing and SourceNewerThanPage groups.
#[test]
fn lint_multiple_sources_on_one_page_yield_per_source_findings() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();

    // Stale source: exists, mtime > page_ts.
    let stale_rel = "src/stale.rs";
    let stale_abs = tmp.path().join(stale_rel);
    std::fs::write(&stale_abs, "// stale").unwrap();
    set_mtime(&stale_abs, page_ts + std::time::Duration::from_secs(3_600));
    // Fresh source: exists, mtime < page_ts — must NOT fire.
    let fresh_rel = "src/fresh.rs";
    let fresh_abs = tmp.path().join(fresh_rel);
    std::fs::write(&fresh_abs, "// fresh").unwrap();
    set_mtime(&fresh_abs, page_ts - std::time::Duration::from_secs(3_600));

    add_page_with_sources(
        &wiki,
        "entities/p.md",
        "p",
        PageType::Entity,
        vec![
            "src/gone1.rs".to_string(),
            "src/gone2.rs".to_string(),
            stale_rel.to_string(),
            fresh_rel.to_string(),
        ],
        TEST_PAGE_TS_STR,
    );
    let findings = wiki.lint().unwrap();
    // 2 missing + 1 newer + 0 fresh = 3 findings, all on entities/p.md.
    assert_eq!(findings.len(), 3, "findings = {:?}", findings);
    for f in &findings {
        assert_eq!(
            f.path, "entities/p.md",
            "all findings must be attributed to the owning page",
        );
    }
    // Sort: SourceMissing (3) < SourceNewerThanPage (4). Among
    // same-kind same-path entries, insertion order is preserved by
    // Vec::sort_by (stable) — so gone1 precedes gone2.
    assert_eq!(findings[0].kind, WikiLintKind::SourceMissing);
    assert!(
        findings[0].detail.contains("src/gone1.rs"),
        "detail[0] must name gone1: {:?}",
        findings[0].detail,
    );
    assert_eq!(findings[1].kind, WikiLintKind::SourceMissing);
    assert!(
        findings[1].detail.contains("src/gone2.rs"),
        "detail[1] must name gone2: {:?}",
        findings[1].detail,
    );
    assert_eq!(findings[2].kind, WikiLintKind::SourceNewerThanPage);
    assert!(
        findings[2].detail.contains(stale_rel),
        "detail[2] must name the stale source: {:?}",
        findings[2].detail,
    );
}

// If the page file itself is missing (OrphanIndexEntry fires), rule 4
// must NOT also emit source-drift findings from the page's index
// entry. The implementation achieves this by bailing on
// `fs::read_to_string(&abs)` errors — locking this behavior prevents a
// refactor from double-counting orphan pages as missing sources too.
#[test]
fn lint_orphan_page_does_not_also_produce_source_drift_findings() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Index entry points at a page that doesn't exist, but claims
    // sources — which is nonsensical because we can't read the page
    // to know its sources. Rule 1 catches the orphan; rule 4 must
    // short-circuit silently.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghost".to_string(),
        path: "entities/ghost.md".to_string(),
        one_liner: "never written".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "entities/ghost.md");
}

// `WikiPage::parse` returning None (unrecognized body, no frontmatter,
// corrupt markdown) must short-circuit rule 4 (no false source-drift
// findings) AND produce exactly one `MalformedPage` finding from
// rule 5. Cycle 15 promoted the silent-skip to a visible finding.
#[test]
fn lint_unparseable_page_body_yields_malformed_page_finding() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Write raw garbage directly to disk (not via write_page, which
    // would produce a parseable file).
    let abs = wiki.root().join("entities/garbage.md");
    std::fs::write(&abs, "this is not a wiki page\njust some text\n").unwrap();
    // Upsert an index entry so rule 1 doesn't fire.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "garbage".to_string(),
        path: "entities/garbage.md".to_string(),
        one_liner: "unparseable".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[0].path, "entities/garbage.md");
    // Rule 4 must have been skipped — no SourceMissing/SourceNewer
    // on a page whose frontmatter can't even be parsed.
    for f in &findings {
        assert_ne!(f.kind, WikiLintKind::SourceMissing);
        assert_ne!(f.kind, WikiLintKind::SourceNewerThanPage);
    }
}

#[test]
fn lint_source_drift_findings_sort_after_existing_kinds() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();

    // One OrphanIndexEntry: index points at a file that doesn't exist.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghost".to_string(),
        path: "entities/ghost.md".to_string(),
        one_liner: "vanished".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    // One UntrackedPage: on-disk page with no index entry.
    let untracked = WikiPage {
        title: "loose".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("concepts/loose.md", &untracked).unwrap();

    // One SourceMissing + one SourceNewerThanPage, on two separate
    // pages so each has a distinct `path` sort key.
    add_page_with_sources(
        &wiki,
        "entities/miss.md",
        "miss",
        PageType::Entity,
        vec!["src/vanished.rs".to_string()],
        TEST_PAGE_TS_STR,
    );
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let newer_rel = "src/newer.rs";
    let newer_abs = tmp.path().join(newer_rel);
    std::fs::write(&newer_abs, "// newer").unwrap();
    set_mtime(&newer_abs, page_ts + std::time::Duration::from_secs(86_400));
    add_page_with_sources(
        &wiki,
        "entities/stale.md",
        "stale",
        PageType::Entity,
        vec![newer_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    let findings = wiki.lint().unwrap();
    // Expected sort: Orphan (0) < Untracked (1) < CategoryMismatch (2)
    //              < SourceMissing (3) < SourceNewerThanPage (4).
    assert_eq!(findings.len(), 4, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "entities/ghost.md");
    assert_eq!(findings[1].kind, WikiLintKind::UntrackedPage);
    assert_eq!(findings[1].path, "concepts/loose.md");
    assert_eq!(findings[2].kind, WikiLintKind::SourceMissing);
    assert_eq!(findings[2].path, "entities/miss.md");
    assert_eq!(findings[3].kind, WikiLintKind::SourceNewerThanPage);
    assert_eq!(findings[3].path, "entities/stale.md");
}

// ── Wiki::lint MalformedPage tests (cycle 15) ────────────────────────
//
// Cycle 15 promotes two parse-failure cases (frontmatter parse fail,
// timestamp parse fail) from silent skips to visible `MalformedPage`
// findings. `lint_malformed_last_updated_yields_malformed_page_finding`
// and `lint_unparseable_page_body_yields_malformed_page_finding` cover
// the primary branches; the tests below lock the universality
// (applies to synthesis too) and the sort-ordering tail.

/// Synthesis pages are skipped for source-drift but NOT for malformed-
/// frontmatter. A synthesis page with unparseable timestamp must still
/// surface as a `MalformedPage` finding — otherwise a broken synthesis
/// page could silently shadow real work from the tester's view.
#[test]
fn lint_malformed_synthesis_page_still_flagged() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Synthesis page, valid frontmatter shape, but bogus timestamp.
    // Synthesis pages are skipped for rule 4 (source-drift), so prior
    // to cycle 15 this would have produced zero findings.
    add_page_with_sources(
        &wiki,
        "synthesis/overview.md",
        "overview",
        PageType::Synthesis,
        vec![],
        "not a timestamp",
    );
    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 1, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[0].path, "synthesis/overview.md");
}

/// Full sort-order lock covering all five `WikiLintKind` variants.
/// Each of Orphan, Untracked, `CategoryMismatch`, `SourceMissing`,
/// `SourceNewerThanPage`, `MalformedPage` contributes exactly one finding
/// on a distinct path. Cycle 15 appends `MalformedPage` at the end of
/// the enum, so it must sort last.
#[test]
fn lint_malformed_page_sorts_after_all_other_kinds() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();

    // Orphan: index points at a missing file.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "ghost".to_string(),
        path: "entities/ghost.md".to_string(),
        one_liner: "vanished".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    // Untracked: on-disk page with no index entry.
    let untracked = WikiPage {
        title: "loose".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("concepts/loose.md", &untracked).unwrap();

    // CategoryMismatch: path under entities/ but index category says
    // Concept. Write the page so it's not also an orphan.
    let miscat = WikiPage {
        title: "miscat".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: TEST_PAGE_TS_STR.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: String::new(),
    };
    wiki.write_page("entities/mc.md", &miscat).unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "mc".to_string(),
        path: "entities/mc.md".to_string(),
        one_liner: "mc".to_string(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    // SourceMissing.
    add_page_with_sources(
        &wiki,
        "entities/miss.md",
        "miss",
        PageType::Entity,
        vec!["src/vanished.rs".to_string()],
        TEST_PAGE_TS_STR,
    );

    // SourceNewerThanPage.
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    let newer_rel = "src/newer.rs";
    let newer_abs = tmp.path().join(newer_rel);
    std::fs::write(&newer_abs, "// newer").unwrap();
    set_mtime(&newer_abs, page_ts + std::time::Duration::from_secs(86_400));
    add_page_with_sources(
        &wiki,
        "entities/stale.md",
        "stale",
        PageType::Entity,
        vec![newer_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    // MalformedPage: write raw garbage where a page should be, then
    // register it in the index so rule 5 iterates into it.
    let garbage_abs = wiki.root().join("entities/broken.md");
    std::fs::write(&garbage_abs, "no frontmatter here\n").unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "broken".to_string(),
        path: "entities/broken.md".to_string(),
        one_liner: "broken".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    // Expected sort: Orphan (0) < Untracked (1) < CategoryMismatch (2)
    //  < SourceMissing (3) < SourceNewerThanPage (4) < MalformedPage (5).
    assert_eq!(findings.len(), 6, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::OrphanIndexEntry);
    assert_eq!(findings[0].path, "entities/ghost.md");
    assert_eq!(findings[1].kind, WikiLintKind::UntrackedPage);
    assert_eq!(findings[1].path, "concepts/loose.md");
    assert_eq!(findings[2].kind, WikiLintKind::CategoryMismatch);
    assert_eq!(findings[2].path, "entities/mc.md");
    assert_eq!(findings[3].kind, WikiLintKind::SourceMissing);
    assert_eq!(findings[3].path, "entities/miss.md");
    assert_eq!(findings[4].kind, WikiLintKind::SourceNewerThanPage);
    assert_eq!(findings[4].path, "entities/stale.md");
    assert_eq!(findings[5].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[5].path, "entities/broken.md");
}

// Multiple malformed pages must each surface as a distinct finding,
// sorted by path ascending within the MalformedPage group. Builder's
// `lint_malformed_page_sorts_after_all_other_kinds` exercises a single
// MalformedPage — this locks the intra-group ordering.
#[test]
fn lint_multiple_malformed_pages_sort_by_path_within_group() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Two malformed pages written directly as garbage, registered in
    // the index. Deliberately register z.md first so the stable sort
    // has to actually re-order.
    let z_abs = wiki.root().join("entities/z.md");
    std::fs::write(&z_abs, "no frontmatter z\n").unwrap();
    let a_abs = wiki.root().join("entities/a.md");
    std::fs::write(&a_abs, "no frontmatter a\n").unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "z".to_string(),
        path: "entities/z.md".to_string(),
        one_liner: "z".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "a".to_string(),
        path: "entities/a.md".to_string(),
        one_liner: "a".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 2, "findings = {:?}", findings);
    assert_eq!(findings[0].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[0].path, "entities/a.md");
    assert_eq!(findings[1].kind, WikiLintKind::MalformedPage);
    assert_eq!(findings[1].path, "entities/z.md");
}

// Both MalformedPage branches (unparseable frontmatter vs unparseable
// last_updated) must produce distinguishable detail strings — the
// operator needs to know WHICH parse failed to pick a fix. The
// frontmatter branch has a fixed detail; the timestamp branch echoes
// the bad value. Locks that the two branches don't collapse to the
// same message under future refactors.
#[test]
fn lint_distinguishes_frontmatter_vs_timestamp_malformed_details() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();

    // (a) Page with bad frontmatter (raw garbage).
    let fm_abs = wiki.root().join("entities/fm_bad.md");
    std::fs::write(&fm_abs, "no frontmatter at all\n").unwrap();
    let mut idx = wiki.load_index().unwrap();
    idx.entries.push(IndexEntry {
        title: "fm_bad".to_string(),
        path: "entities/fm_bad.md".to_string(),
        one_liner: "fm_bad".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    // (b) Page with valid frontmatter but bogus timestamp.
    add_page_with_sources(
        &wiki,
        "entities/ts_bad.md",
        "ts_bad",
        PageType::Entity,
        vec![],
        "xyzzy-not-a-timestamp",
    );

    let findings = wiki.lint().unwrap();
    assert_eq!(findings.len(), 2, "findings = {:?}", findings);
    // Sort by path ascending: fm_bad < ts_bad.
    assert_eq!(findings[0].path, "entities/fm_bad.md");
    assert_eq!(findings[0].kind, WikiLintKind::MalformedPage);
    assert!(
        findings[0].detail.contains("frontmatter"),
        "frontmatter-parse detail must name frontmatter: {:?}",
        findings[0].detail,
    );
    assert!(
        !findings[0].detail.contains("xyzzy"),
        "frontmatter-parse detail must not echo timestamp: {:?}",
        findings[0].detail,
    );

    assert_eq!(findings[1].path, "entities/ts_bad.md");
    assert_eq!(findings[1].kind, WikiLintKind::MalformedPage);
    assert!(
        findings[1].detail.contains("last_updated"),
        "timestamp-parse detail must name last_updated: {:?}",
        findings[1].detail,
    );
    assert!(
        findings[1].detail.contains("xyzzy-not-a-timestamp"),
        "timestamp-parse detail must echo the bad value: {:?}",
        findings[1].detail,
    );
}

// ── Wiki::check_source_drift tests (wiki comprehension dogfooding) ───
//
// Per-file inline drift check surfaced after tool edits. Mirrors lint
// Rule 4b but returns a warning string instead of a lint finding.

#[test]
fn check_source_drift_detects_stale_page() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();

    let src_rel = "src/main.rs";
    let src_abs = tmp.path().join(src_rel);
    std::fs::write(&src_abs, "fn main() {}").unwrap();

    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();
    let newer = page_ts + std::time::Duration::from_secs(86_400);
    set_mtime(&src_abs, newer);

    // Write at the canonical entity-page path so check_source_drift
    // can find it by source-file path alone.
    add_page_with_sources(
        &wiki,
        "entities/src_main_rs.md",
        "main",
        PageType::Entity,
        vec![src_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    let warning = wiki
        .check_source_drift(tmp.path(), &src_abs)
        .unwrap()
        .expect("should detect drift");
    assert!(
        warning.contains("[wiki-drift]"),
        "warning should carry [wiki-drift] marker: {}",
        warning
    );
    assert!(
        warning.contains("may be stale"),
        "warning should say stale: {}",
        warning
    );
    assert!(
        warning.contains("entities/src_main_rs.md"),
        "warning should name page: {}",
        warning
    );
}

#[test]
fn check_source_drift_none_when_page_is_fresh() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();

    let src_rel = "src/fresh.rs";
    let src_abs = tmp.path().join(src_rel);
    std::fs::write(&src_abs, "// fresh").unwrap();

    let page_ts = parse_page_timestamp(TEST_PAGE_TS_STR).unwrap();
    // Source is OLDER than the page — no drift.
    set_mtime(&src_abs, page_ts - std::time::Duration::from_secs(3_600));

    add_page_with_sources(
        &wiki,
        "entities/src_fresh_rs.md",
        "fresh",
        PageType::Entity,
        vec![src_rel.to_string()],
        TEST_PAGE_TS_STR,
    );

    let result = wiki.check_source_drift(tmp.path(), &src_abs).unwrap();
    assert!(result.is_none(), "fresh page should produce no warning");
}

#[test]
fn check_source_drift_none_when_no_entity_page() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();

    let src_abs = tmp.path().join("src/orphan.rs");
    std::fs::write(&src_abs, "// orphan").unwrap();

    // No entity page exists — no drift to report.
    let result = wiki.check_source_drift(tmp.path(), &src_abs).unwrap();
    assert!(result.is_none(), "missing page should produce no warning");
}

#[test]
fn check_source_drift_none_for_wiki_internal_path() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();

    let wiki_file = wiki.root().join("index.md");
    // Wiki-internal paths are silently skipped.
    let result = wiki.check_source_drift(tmp.path(), &wiki_file).unwrap();
    assert!(
        result.is_none(),
        "wiki-internal path should produce no warning"
    );
}

// ── Wiki::lint EntityGap tests (cycle 30 — Phase 3.2 closure) ────────
//
// EntityGap surfaces coverage holes: .rs files under `<project>/src/`
// that no page documents, either canonically (at `entity_page_rel(rel)`)
// or by citation in any page's `sources` frontmatter.

#[test]
fn lint_entity_gap_flags_rs_file_without_page() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/foo.rs"), "fn main() {}").unwrap();

    let findings = wiki.lint().unwrap();
    let gaps: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::EntityGap)
        .collect();
    assert_eq!(
        gaps.len(),
        1,
        "expected one EntityGap; findings = {:?}",
        findings
    );
    assert_eq!(gaps[0].path, "src/foo.rs");
    assert!(
        gaps[0].detail.contains("entities/src_foo_rs.md"),
        "detail must name the expected canonical page; got {:?}",
        gaps[0].detail,
    );
}

#[test]
fn lint_entity_gap_silent_when_entity_page_exists() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("foo.rs");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "fn main() {}").unwrap();
    wiki.ingest_file(&proj, &file, "fn main() {}").unwrap();

    let findings = wiki.lint().unwrap();
    let gaps: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::EntityGap)
        .collect();
    assert!(
        gaps.is_empty(),
        "ingested file must not produce EntityGap; got {:?}",
        gaps
    );
}

#[test]
fn lint_entity_gap_recursively_walks_src_subdirs() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src/tools")).unwrap();
    std::fs::create_dir_all(tmp.path().join("src/wiki")).unwrap();
    std::fs::write(tmp.path().join("src/foo.rs"), "// foo").unwrap();
    std::fs::write(tmp.path().join("src/tools/bar.rs"), "// bar").unwrap();
    std::fs::write(tmp.path().join("src/wiki/baz.rs"), "// baz").unwrap();

    let findings = wiki.lint().unwrap();
    let gaps: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::EntityGap)
        .collect();
    let paths: Vec<&str> = gaps.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        vec!["src/foo.rs", "src/tools/bar.rs", "src/wiki/baz.rs"],
        "EntityGap findings must cover each rs file in sort order; got {:?}",
        paths
    );
}

#[test]
fn lint_entity_gap_ignores_non_rs_files() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    // Non-.rs extensions → ignored by extension filter.
    std::fs::write(tmp.path().join("src/notes.md"), "# readme").unwrap();
    std::fs::write(tmp.path().join("src/Cargo.toml.bak"), "").unwrap();
    // Hidden subdirectory → skipped by name-starts-with('.') guard so
    // it does not recurse into it.
    std::fs::create_dir_all(tmp.path().join("src/.hidden")).unwrap();
    std::fs::write(tmp.path().join("src/.hidden/inner.rs"), "// inner").unwrap();

    let findings = wiki.lint().unwrap();
    let gaps: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::EntityGap)
        .collect();
    assert!(
        gaps.is_empty(),
        "non-rs and hidden-dir entries must not produce EntityGap; got {:?}",
        gaps
    );
}

// ─── Cycle 37: ItemDrift (Rule 7) ──────────────────────────────────────

/// Helper: install a wiki entity page with the given body + an index
/// entry pointing to it. Keeps test fixtures compact.
fn install_entity_page(
    wiki: &Wiki,
    page_rel: &str,
    source_rel: &str,
    body: &str,
    last_updated: &str,
) {
    let page = WikiPage {
        title: source_rel.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![source_rel.to_string()],
        last_updated: last_updated.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: body.to_string(),
    };
    wiki.write_page(page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: source_rel.to_string(),
        path: page_rel.to_string(),
        one_liner: format!("File: {}", source_rel),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

#[test]
fn lint_flags_item_drift_when_page_has_item_not_in_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn alive() {}\n").unwrap();

    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        "# src/x.rs\n\n## Items\n\n- `pub fn ghost`\n- `pub fn alive`\n",
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    assert_eq!(drift.len(), 1, "exactly one ItemDrift: {:?}", findings);
    assert_eq!(drift[0].path, "entities/src_x_rs.md");
    assert!(
        drift[0].detail.contains("ghost"),
        "detail must name the drifted item: {:?}",
        drift[0].detail
    );
    assert!(
        !drift[0].detail.contains("alive"),
        "live items must not appear in drift list: {:?}",
        drift[0].detail
    );
}

#[test]
fn lint_does_not_flag_item_drift_when_items_match() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn foo() {}\npub struct Bar;\n").unwrap();

    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        "# src/x.rs\n\n## Items\n\n- `pub fn foo`\n- `pub struct Bar`\n",
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    assert!(drift.is_empty(), "no drift expected: {:?}", drift);
}

#[test]
fn lint_item_drift_skips_missing_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    // No src/x.rs on disk.

    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        "# src/x.rs\n\n## Items\n\n- `pub fn foo`\n",
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    let missing: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::SourceMissing)
        .collect();
    assert!(
        drift.is_empty(),
        "ItemDrift must defer to SourceMissing: {:?}",
        drift
    );
    assert_eq!(
        missing.len(),
        1,
        "SourceMissing fires instead: {:?}",
        missing
    );
}

#[test]
fn lint_item_drift_skips_non_rs_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::write(proj.join("foo.toml"), "[package]\nname=\"x\"\n").unwrap();

    install_entity_page(
        &wiki,
        "entities/foo_toml.md",
        "foo.toml",
        "# foo.toml\n\n## Items\n\n- `pub fn fake`\n",
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    assert!(drift.is_empty(), "non-Rust sources exempt: {:?}", drift);
}

#[test]
fn lint_item_drift_skips_page_without_items_section() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn alive() {}\n").unwrap();

    // Stub body without `## Items` (pre-Cycle-35 shape).
    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        "# src/x.rs\n\nSource file: `src/x.rs`\n",
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    assert!(drift.is_empty(), "stub pages don't drift: {:?}", drift);
}

#[test]
fn lint_item_drift_fires_alongside_source_newer_than_page() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Source has `alive` only; mtime is `now`.
    std::fs::write(proj.join("src/x.rs"), "pub fn alive() {}\n").unwrap();

    // Page last_updated deep in the past so the source is newer.
    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        "# src/x.rs\n\n## Items\n\n- `pub fn ghost`\n",
        "1970-01-02 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .count();
    let newer = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::SourceNewerThanPage)
        .count();
    assert_eq!(drift, 1, "ItemDrift must fire: {:?}", findings);
    assert_eq!(
        newer, 1,
        "SourceNewerThanPage must fire too: {:?}",
        findings
    );
}

#[test]
fn lint_item_drift_detail_caps_long_lists() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn alive() {}\n").unwrap();

    // Build a page body naming 50 drifted items.
    let mut body = String::from("# src/x.rs\n\n## Items\n\n");
    for i in 0..50 {
        writeln!(body, "- `pub fn ghost_{:02}`", i).expect("write to String never fails");
    }
    install_entity_page(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        &body,
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ItemDrift)
        .collect();
    assert_eq!(drift.len(), 1);
    assert!(
        drift[0].detail.len() <= 200,
        "detail must be ≤200 bytes; got len={}",
        drift[0].detail.len()
    );
    assert!(
        drift[0].detail.ends_with(", …"),
        "capped detail must end with truncation marker: {:?}",
        drift[0].detail
    );
}

#[test]
fn lint_item_drift_sorted_between_source_newer_and_entity_gap() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();

    // Source A — triggers SourceNewerThanPage + ItemDrift (page has `old`,
    // source has `cur`; page timestamp in 1970).
    std::fs::write(proj.join("src/a.rs"), "pub fn cur() {}\n").unwrap();
    install_entity_page(
        &wiki,
        "entities/src_a_rs.md",
        "src/a.rs",
        "# src/a.rs\n\n## Items\n\n- `pub fn old`\n",
        "1970-01-02 00:00:00",
    );
    // Source B — uncovered .rs → triggers EntityGap.
    std::fs::write(proj.join("src/b.rs"), "pub fn only_b() {}\n").unwrap();

    let findings = wiki.lint().unwrap();
    // Kind discriminants must appear in the sequence: SourceNewerThanPage
    // (4), ItemDrift (5), EntityGap (6). Ignore unrelated findings.
    let kinds: Vec<WikiLintKind> = findings
        .iter()
        .map(|f| f.kind)
        .filter(|k| {
            matches!(
                k,
                WikiLintKind::SourceNewerThanPage
                    | WikiLintKind::ItemDrift
                    | WikiLintKind::EntityGap
            )
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            WikiLintKind::SourceNewerThanPage,
            WikiLintKind::ItemDrift,
            WikiLintKind::EntityGap,
        ],
        "enum-ordered output: {:?}",
        findings
    );
}

// ─── Cycle 52: ExportDrift (Rule 7b) ───────────────────────────────────
//
// Frontmatter-layer parallel to ItemDrift: flags entity pages whose
// `key_exports` names a symbol no longer present in the current source.
// Piggybacks on the ItemDrift loop — both need the same .rs read.

/// Install an entity page whose `key_exports` are explicitly set. Used
/// exclusively by the `ExportDrift` cluster — the generic
/// `install_entity_page` helper hard-codes `key_exports: vec![]` for
/// pre-Cycle-42 legacy fixtures.
fn install_entity_page_with_exports(
    wiki: &Wiki,
    page_rel: &str,
    source_rel: &str,
    exports: Vec<KeyExport>,
    last_updated: &str,
) {
    let page = WikiPage {
        title: source_rel.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![source_rel.to_string()],
        last_updated: last_updated.to_string(),
        entity_kind: Some(EntityKind::Unknown),
        purpose: None,
        key_exports: exports,
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", source_rel),
    };
    wiki.write_page(page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: source_rel.to_string(),
        path: page_rel.to_string(),
        one_liner: format!("File: {}", source_rel),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

#[test]
fn lint_detects_export_drift_when_frontmatter_names_removed_symbol() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Source now has only `foo`; page still claims both `foo` and `bar`.
    std::fs::write(proj.join("src/x.rs"), "pub fn foo() {}\n").unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "foo".to_string(),
            },
            KeyExport {
                kind: EntityKind::Function,
                name: "bar".to_string(),
            },
        ],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .collect();
    assert_eq!(drift.len(), 1, "findings = {:?}", findings);
    assert_eq!(drift[0].path, "entities/src_x_rs.md");
    assert!(
        drift[0].detail.contains("bar"),
        "detail must name the drifted symbol; got: {}",
        drift[0].detail,
    );
    assert!(
        !drift[0].detail.contains("foo"),
        "detail must not name the still-present symbol; got: {}",
        drift[0].detail,
    );
}

#[test]
fn lint_no_export_drift_when_all_frontmatter_symbols_exist_in_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn foo() {}\npub struct Bar;\n").unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "foo".to_string(),
            },
            KeyExport {
                kind: EntityKind::Struct,
                name: "Bar".to_string(),
            },
        ],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .count();
    assert_eq!(
        drift, 0,
        "perfectly aligned exports → no drift; got: {:?}",
        findings
    );
}

#[test]
fn lint_export_drift_skips_page_with_empty_key_exports() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Source has several public items the (empty) page doesn't claim.
    std::fs::write(
        proj.join("src/x.rs"),
        "pub fn foo() {}\npub fn bar() {}\npub struct Baz;\n",
    )
    .unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        vec![], // Pre-Cycle-42 legacy shape.
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .count();
    assert_eq!(
        drift, 0,
        "empty key_exports must skip (pre-Cycle-42 parallel); got: {:?}",
        findings,
    );
}

#[test]
fn lint_export_drift_skips_non_rs_first_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("docs")).unwrap();
    std::fs::write(proj.join("docs/x.md"), "doc content").unwrap();
    // Page cites a .md source but has non-empty key_exports.
    install_entity_page_with_exports(
        &wiki,
        "entities/docs_x.md",
        "docs/x.md",
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "ghost".to_string(),
        }],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .count();
    assert_eq!(
        drift, 0,
        "non-.rs first-source must not trigger; got: {:?}",
        findings,
    );
}

#[test]
fn lint_export_drift_skips_when_source_missing() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    // No src/ tree — source file is absent from disk.
    install_entity_page_with_exports(
        &wiki,
        "entities/src_gone_rs.md",
        "src/gone.rs",
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "vanished".to_string(),
        }],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let source_missing = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::SourceMissing)
        .count();
    let export_drift = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .count();
    assert_eq!(
        source_missing, 1,
        "absent source must fire SourceMissing; got: {:?}",
        findings,
    );
    assert_eq!(
        export_drift, 0,
        "missing source must NOT also fire ExportDrift (avoid double-report); got: {:?}",
        findings,
    );
}

#[test]
fn lint_export_drift_detail_includes_drifted_names_and_next_step() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/x.rs"), "pub fn foo() {}\n").unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        vec![
            KeyExport {
                kind: EntityKind::Function,
                name: "foo".to_string(),
            },
            KeyExport {
                kind: EntityKind::Struct,
                name: "Bar".to_string(),
            },
            KeyExport {
                kind: EntityKind::Enum,
                name: "Baz".to_string(),
            },
        ],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .collect();
    assert_eq!(drift.len(), 1, "findings = {:?}", findings);
    let detail = &drift[0].detail;
    assert!(
        detail.contains("Bar"),
        "detail must list drifted Bar; got: {}",
        detail,
    );
    assert!(
        detail.contains("Baz"),
        "detail must list drifted Baz; got: {}",
        detail,
    );
    // ASCII-sorted: `Bar` precedes `Baz`.
    let bar_pos = detail.find("Bar").unwrap();
    let baz_pos = detail.find("Baz").unwrap();
    assert!(
        bar_pos < baz_pos,
        "detail must list names ASCII-sorted; got: {}",
        detail,
    );
    assert!(
        detail.ends_with("; re-ingest to reconcile"),
        "detail must carry actionable next-step suffix; got: {}",
        detail,
    );
}

#[test]
fn lint_export_drift_caps_detail_at_200_bytes_with_ellipsis() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Source defines nothing — every page-claimed export drifts.
    std::fs::write(proj.join("src/x.rs"), "// empty\n").unwrap();
    let exports: Vec<KeyExport> = (0..50)
        .map(|i| KeyExport {
            kind: EntityKind::Function,
            name: format!("ghost_symbol_{:02}", i),
        })
        .collect();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_x_rs.md",
        "src/x.rs",
        exports,
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let drift: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::ExportDrift)
        .collect();
    assert_eq!(drift.len(), 1, "findings = {:?}", findings);
    let detail = &drift[0].detail;
    // The `; re-ingest to reconcile` suffix sits outside the cap.
    let suffix = "; re-ingest to reconcile";
    assert!(
        detail.ends_with(suffix),
        "capped detail must still carry the next-step suffix; got: {}",
        detail,
    );
    let symbol_list_bytes = detail.len() - suffix.len();
    assert!(
        symbol_list_bytes <= 200,
        "symbol-list portion must be ≤ 200 bytes; got len={} detail={:?}",
        symbol_list_bytes,
        detail,
    );
    assert!(
        detail.contains(", …;"),
        "capped detail must include the truncation marker before the next-step suffix; got: {}",
        detail,
    );
}

#[test]
fn lint_export_drift_ordered_between_item_drift_and_entity_gap() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();

    // ItemDrift source: body `## Items` names `old`, source has `cur`.
    std::fs::write(proj.join("src/item.rs"), "pub fn cur() {}\n").unwrap();
    install_entity_page(
        &wiki,
        "entities/src_item_rs.md",
        "src/item.rs",
        "# src/item.rs\n\n## Items\n\n- `pub fn old`\n",
        "2099-01-01 00:00:00",
    );
    // ExportDrift source: frontmatter claims `ghost`, source defines nothing.
    std::fs::write(proj.join("src/exp.rs"), "// empty\n").unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/src_exp_rs.md",
        "src/exp.rs",
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "ghost".to_string(),
        }],
        "2099-01-01 00:00:00",
    );
    // EntityGap source: uncovered .rs under src/.
    std::fs::write(proj.join("src/gap.rs"), "pub fn only_gap() {}\n").unwrap();

    let findings = wiki.lint().unwrap();
    let kinds: Vec<WikiLintKind> = findings
        .iter()
        .map(|f| f.kind)
        .filter(|k| {
            matches!(
                k,
                WikiLintKind::ItemDrift | WikiLintKind::ExportDrift | WikiLintKind::EntityGap
            )
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            WikiLintKind::ItemDrift,
            WikiLintKind::ExportDrift,
            WikiLintKind::EntityGap,
        ],
        "enum-ordered output: {:?}",
        findings,
    );
}

// ─── Cycle 43: MissingEntityKind (Rule 8) ──────────────────────────────
//
// Flags entity pages whose `sources` include a `.rs` file but whose
// `entity_kind` frontmatter is `None` — a legacy-page signal that
// re-ingesting will populate. Piggybacks on the Rules 4/5 unified
// page-read sweep.

/// Install an entity page with `entity_kind` left unset — mirrors
/// pre-Cycle-38 fixture shape that Rule 8 targets.
fn install_entity_page_missing_kind(
    wiki: &Wiki,
    page_rel: &str,
    sources: Vec<String>,
    last_updated: &str,
) {
    let title = sources.first().cloned().unwrap_or_default();
    let page = WikiPage {
        title: title.clone(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources,
        last_updated: last_updated.to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", title),
    };
    wiki.write_page(page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: title.clone(),
        path: page_rel.to_string(),
        one_liner: format!("File: {}", title),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

#[test]
fn lint_missing_entity_kind_flags_legacy_rs_entity_page() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/legacy.rs"), "pub fn f() {}\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/src_legacy_rs.md",
        vec!["src/legacy.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert_eq!(hits.len(), 1, "expected one finding; got {:?}", findings);
    assert_eq!(hits[0].path, "entities/src_legacy_rs.md");
    assert!(
        hits[0].detail.contains("entity_kind"),
        "detail must name entity_kind: {:?}",
        hits[0].detail,
    );
    assert!(
        hits[0].detail.contains("re-ingest"),
        "detail must hint at the fix: {:?}",
        hits[0].detail,
    );
}

#[test]
fn lint_missing_entity_kind_silent_when_entity_kind_set() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/ok.rs"), "pub fn f() {}\n").unwrap();
    add_page_with_sources(
        &wiki,
        "entities/src_ok_rs.md",
        "src/ok.rs",
        PageType::Entity,
        vec!["src/ok.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert!(
        hits.is_empty(),
        "helper sets entity_kind for .rs entity pages; got {:?}",
        hits,
    );
}

#[test]
fn lint_missing_entity_kind_silent_for_non_rs_sources() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("docs")).unwrap();
    std::fs::write(proj.join("docs/note.md"), "# note\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/docs_note_md.md",
        vec!["docs/note.md".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert!(
        hits.is_empty(),
        "non-rs sources must not trip Rule 8; got {:?}",
        hits,
    );
}

#[test]
fn lint_missing_entity_kind_silent_for_concept_page() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/c.rs"), "pub fn f() {}\n").unwrap();
    // Concept page citing a .rs source — Rule 8 targets Entity only.
    let page = WikiPage {
        title: "concept".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/c.rs".to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# concept\n".to_string(),
    };
    wiki.write_page("concepts/c.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "concept".to_string(),
        path: "concepts/c.md".to_string(),
        one_liner: "concept".to_string(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert!(
        hits.is_empty(),
        "non-entity pages must not trip Rule 8; got {:?}",
        hits,
    );
}

#[test]
fn lint_missing_entity_kind_silent_for_empty_sources_entity_page() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/no_sources.md",
        vec![],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert!(
        hits.is_empty(),
        "entity pages with no sources cannot signal schema drift; got {:?}",
        hits,
    );
}

#[test]
fn lint_missing_entity_kind_fires_per_page_not_per_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/a.rs"), "pub fn a() {}\n").unwrap();
    std::fs::write(proj.join("src/b.rs"), "pub fn b() {}\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/multi.md",
        vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let hits: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::MissingEntityKind)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "Rule 8 emits once per page regardless of source count; got {:?}",
        hits,
    );
    assert_eq!(hits[0].path, "entities/multi.md");
}

#[test]
fn lint_missing_entity_kind_sorts_between_entity_gap_and_malformed() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();

    // Rule 7 EntityGap: uncovered .rs under src/.
    std::fs::write(proj.join("src/gap.rs"), "pub fn g() {}\n").unwrap();
    // Rule 8 MissingEntityKind: legacy entity page with .rs source.
    std::fs::write(proj.join("src/legacy.rs"), "pub fn l() {}\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/src_legacy_rs.md",
        vec!["src/legacy.rs".to_string()],
        "2099-01-01 00:00:00",
    );
    // Rule 5 MalformedPage: raw garbage.
    std::fs::create_dir_all(wiki.root().join("entities")).unwrap();
    std::fs::write(wiki.root().join("entities/broken.md"), "this is not yaml").unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "broken".to_string(),
        path: "entities/broken.md".to_string(),
        one_liner: "broken".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    let kinds: Vec<WikiLintKind> = findings
        .iter()
        .map(|f| f.kind)
        .filter(|k| {
            matches!(
                k,
                WikiLintKind::EntityGap
                    | WikiLintKind::MissingEntityKind
                    | WikiLintKind::MalformedPage
            )
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            WikiLintKind::EntityGap,
            WikiLintKind::MissingEntityKind,
            WikiLintKind::MalformedPage,
        ],
        "enum-ordered output: {:?}",
        findings,
    );
}

// Cycle 51: DuplicateSource — two or more Entity pages listing the
// same .rs file as their first source. Real-world trigger: a rename
// left behind a stale page, a split produced two pages for the same
// file, or an ingest under a new title didn't delete the old page.

#[test]
fn lint_detects_duplicate_source_across_two_entity_pages() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Stale-free source so SourceNewerThanPage doesn't fire (future
    // timestamp on pages).
    std::fs::write(proj.join("src/foo.rs"), "pub fn foo() {}\n").unwrap();
    add_page_with_sources(
        &wiki,
        "entities/a.md",
        "a",
        PageType::Entity,
        vec!["src/foo.rs".to_string()],
        "2099-01-01 00:00:00",
    );
    add_page_with_sources(
        &wiki,
        "entities/b.md",
        "b",
        PageType::Entity,
        vec!["src/foo.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let dups: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .collect();
    assert_eq!(
        dups.len(),
        2,
        "two pages sharing a source must produce 2 findings; got: {:?}",
        findings,
    );
    // Sort by path puts a.md before b.md.
    assert_eq!(dups[0].path, "entities/a.md");
    assert_eq!(dups[1].path, "entities/b.md");
}

#[test]
fn lint_duplicate_source_three_pages_produces_three_findings() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/foo.rs"), "pub fn foo() {}\n").unwrap();
    for rel in ["entities/a.md", "entities/b.md", "entities/c.md"] {
        add_page_with_sources(
            &wiki,
            rel,
            rel,
            PageType::Entity,
            vec!["src/foo.rs".to_string()],
            "2099-01-01 00:00:00",
        );
    }

    let findings = wiki.lint().unwrap();
    let dups: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .collect();
    assert_eq!(
        dups.len(),
        3,
        "N pages sharing a source → N findings; got: {:?}",
        findings,
    );
    // Each claimant carries a finding so jumps-to-conflict are direct.
    let paths: Vec<&str> = dups.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        vec!["entities/a.md", "entities/b.md", "entities/c.md"]
    );
}

#[test]
fn lint_duplicate_source_detail_lists_other_pages_and_next_step() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/foo.rs"), "pub fn foo() {}\n").unwrap();
    for rel in ["entities/a.md", "entities/b.md", "entities/c.md"] {
        add_page_with_sources(
            &wiki,
            rel,
            rel,
            PageType::Entity,
            vec!["src/foo.rs".to_string()],
            "2099-01-01 00:00:00",
        );
    }

    let findings = wiki.lint().unwrap();
    let a_finding = findings
        .iter()
        .find(|f| f.kind == WikiLintKind::DuplicateSource && f.path == "entities/a.md")
        .expect("a.md finding");
    assert!(
        a_finding.detail.contains("src/foo.rs"),
        "detail must cite the source filename; got: {}",
        a_finding.detail,
    );
    assert!(
        a_finding.detail.contains("entities/b.md"),
        "detail must list other page b.md; got: {}",
        a_finding.detail,
    );
    assert!(
        a_finding.detail.contains("entities/c.md"),
        "detail must list other page c.md; got: {}",
        a_finding.detail,
    );
    assert!(
        a_finding.detail.contains("delete the stale page"),
        "detail must include actionable next-step; got: {}",
        a_finding.detail,
    );
    assert!(
        a_finding.detail.contains("/wiki refresh"),
        "detail must reference /wiki refresh next-step; got: {}",
        a_finding.detail,
    );
}

#[test]
fn lint_no_duplicate_source_when_sources_unique() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    for (rel, src) in [
        ("entities/a.md", "src/a.rs"),
        ("entities/b.md", "src/b.rs"),
        ("entities/c.md", "src/c.rs"),
    ] {
        std::fs::write(proj.join(src), "pub fn x() {}\n").unwrap();
        add_page_with_sources(
            &wiki,
            rel,
            rel,
            PageType::Entity,
            vec![src.to_string()],
            "2099-01-01 00:00:00",
        );
    }

    let findings = wiki.lint().unwrap();
    let dups = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .count();
    assert_eq!(
        dups, 0,
        "unique first-sources must not trigger; got: {:?}",
        findings
    );
}

#[test]
fn lint_duplicate_source_ignores_non_rs_first_source() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    // Two entity pages sharing a .md first-source. Out of scope for
    // this rule — only .rs qualifies (same policy as MissingEntityKind).
    std::fs::create_dir_all(proj.join("docs")).unwrap();
    std::fs::write(proj.join("docs/x.md"), "doc").unwrap();
    add_page_with_sources(
        &wiki,
        "entities/a.md",
        "a",
        PageType::Entity,
        vec!["docs/x.md".to_string()],
        "2099-01-01 00:00:00",
    );
    add_page_with_sources(
        &wiki,
        "entities/b.md",
        "b",
        PageType::Entity,
        vec!["docs/x.md".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let dups = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .count();
    assert_eq!(
        dups, 0,
        "non-.rs first-source must not trigger; got: {:?}",
        findings,
    );
}

#[test]
fn lint_duplicate_source_ignores_non_entity_category() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/foo.rs"), "pub fn foo() {}\n").unwrap();
    // Concept page citing a .rs source alongside an Entity page —
    // concept/entity cohabitation is legitimate (the concept page is
    // about a pattern that appears in that file), not a contradiction.
    add_page_with_sources(
        &wiki,
        "concepts/c.md",
        "c",
        PageType::Concept,
        vec!["src/foo.rs".to_string()],
        "2099-01-01 00:00:00",
    );
    add_page_with_sources(
        &wiki,
        "entities/e.md",
        "e",
        PageType::Entity,
        vec!["src/foo.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let dups = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .count();
    assert_eq!(
        dups, 0,
        "concept/entity same-source cohabitation must not trigger; got: {:?}",
        findings,
    );
}

#[test]
fn lint_duplicate_source_sort_position_between_missing_entity_kind_and_malformed_page() {
    // Variant-position canary. Construct a wiki that triggers
    // MissingEntityKind, DuplicateSource, and MalformedPage; assert
    // sorted ordering places them in that exact sequence. Guards
    // against accidental enum reordering in future cycles.
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();

    // MissingEntityKind: a legacy page lacking entity_kind.
    std::fs::write(proj.join("src/legacy.rs"), "pub fn l() {}\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/z_legacy.md",
        vec!["src/legacy.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    // DuplicateSource: two Entity pages sharing src/dup.rs.
    std::fs::write(proj.join("src/dup.rs"), "pub fn d() {}\n").unwrap();
    add_page_with_sources(
        &wiki,
        "entities/dup_a.md",
        "dup_a",
        PageType::Entity,
        vec!["src/dup.rs".to_string()],
        "2099-01-01 00:00:00",
    );
    add_page_with_sources(
        &wiki,
        "entities/dup_b.md",
        "dup_b",
        PageType::Entity,
        vec!["src/dup.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    // MalformedPage: raw garbage registered in the index.
    std::fs::create_dir_all(wiki.root().join("entities")).unwrap();
    std::fs::write(wiki.root().join("entities/broken.md"), "no frontmatter").unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "broken".to_string(),
        path: "entities/broken.md".to_string(),
        one_liner: "broken".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let findings = wiki.lint().unwrap();
    let kinds: Vec<WikiLintKind> = findings
        .iter()
        .map(|f| f.kind)
        .filter(|k| {
            matches!(
                k,
                WikiLintKind::MissingEntityKind
                    | WikiLintKind::DuplicateSource
                    | WikiLintKind::MalformedPage
            )
        })
        .collect();
    // One MissingEntityKind + two DuplicateSource (N=2 → N findings)
    // + one MalformedPage = 4, in declared-enum order.
    assert_eq!(
        kinds,
        vec![
            WikiLintKind::MissingEntityKind,
            WikiLintKind::DuplicateSource,
            WikiLintKind::DuplicateSource,
            WikiLintKind::MalformedPage,
        ],
        "enum-ordered output: {:?}",
        findings,
    );
}

#[test]
fn lint_duplicate_source_skips_entity_with_empty_sources() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(proj.join("src/foo.rs"), "pub fn foo() {}\n").unwrap();
    // One entity page with empty sources (edge case from a partial
    // manual edit), paired with one entity page covering src/foo.rs.
    // Neither should trigger — there's no shared first-source.
    add_page_with_sources(
        &wiki,
        "entities/empty.md",
        "empty",
        PageType::Entity,
        vec![],
        "2099-01-01 00:00:00",
    );
    add_page_with_sources(
        &wiki,
        "entities/full.md",
        "full",
        PageType::Entity,
        vec!["src/foo.rs".to_string()],
        "2099-01-01 00:00:00",
    );

    let findings = wiki.lint().unwrap();
    let dups: Vec<&WikiLintFinding> = findings
        .iter()
        .filter(|f| f.kind == WikiLintKind::DuplicateSource)
        .collect();
    assert!(
        dups.is_empty(),
        "empty-sources page must not participate in DuplicateSource; got: {:?}",
        findings,
    );
}

#[test]
fn lint_missing_entity_kind_cleared_by_reingest() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/foo.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn foo() {}\n").unwrap();

    let page_rel = entity_page_rel(src_rel);
    install_entity_page_missing_kind(
        &wiki,
        &page_rel,
        vec![src_rel.to_string()],
        "2099-01-01 00:00:00",
    );
    let before = wiki.lint().unwrap();
    assert!(
        before
            .iter()
            .any(|f| f.kind == WikiLintKind::MissingEntityKind),
        "pre-reingest lint must flag: {:?}",
        before,
    );

    // Re-ingest: replaces the page with a fully-populated one.
    wiki.ingest_file(&proj, &src_abs, "pub fn foo() {}\n")
        .unwrap();

    let after = wiki.lint().unwrap();
    assert!(
        after
            .iter()
            .all(|f| f.kind != WikiLintKind::MissingEntityKind),
        "re-ingest must clear Rule 8: {:?}",
        after,
    );
}

// ─── Cycle 44: /wiki refresh ───────────────────────────────────────────
//
// `Wiki::refresh()` walks the index and re-ingests Entity pages whose
// `.rs` sources are newer than the page OR whose `entity_kind` is
// `None`. Uses `ingest_file_internal(..., force=true)` so the content-
// hash cache doesn't short-circuit schema backfills.

#[test]
fn refresh_reingests_page_when_source_newer_than_page_ts() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/stale.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn cur() {}\n").unwrap();
    // Stage entity page with an ancient timestamp so the source mtime
    // is newer. entity_kind is set so only the temporal rule fires.
    let page = WikiPage {
        title: src_rel.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![src_rel.to_string()],
        last_updated: "2020-01-01 00:00:00".to_string(),
        entity_kind: Some(EntityKind::Function),
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", src_rel),
    };
    let page_rel = entity_page_rel(src_rel);
    wiki.write_page(&page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: src_rel.to_string(),
        path: page_rel.clone(),
        one_liner: src_rel.to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let report = wiki.refresh().unwrap();
    assert_eq!(
        report.refreshed,
        vec![src_rel.to_string()],
        "expected stale source to be refreshed; report = {:?}",
        report,
    );
}

#[test]
fn refresh_reingests_legacy_entity_page_missing_kind() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/legacy.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn foo() {}\n").unwrap();
    // Stage legacy page: entity_kind None, .rs source, timestamp from
    // the far future so the temporal rule *cannot* fire. This isolates
    // the force-flag validation.
    let page = WikiPage {
        title: src_rel.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![src_rel.to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", src_rel),
    };
    let page_rel = entity_page_rel(src_rel);
    wiki.write_page(&page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: src_rel.to_string(),
        path: page_rel.clone(),
        one_liner: src_rel.to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let report = wiki.refresh().unwrap();
    assert_eq!(
        report.refreshed,
        vec![src_rel.to_string()],
        "force-flag must re-ingest even when source bytes unchanged; report = {:?}",
        report,
    );
    // On-disk page should now carry a populated entity_kind.
    let text = std::fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    let reparsed = WikiPage::parse(&text).expect("re-parseable after refresh");
    assert_eq!(
        reparsed.entity_kind,
        Some(EntityKind::Function),
        "refresh must populate entity_kind; page = {:?}",
        reparsed,
    );
}

#[test]
fn refresh_skips_up_to_date_pages() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/ok.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn ok() {}\n").unwrap();
    // Page timestamp in the far future so mtime < page_ts.
    let page = WikiPage {
        title: src_rel.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![src_rel.to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: Some(EntityKind::Function),
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", src_rel),
    };
    let page_rel = entity_page_rel(src_rel);
    wiki.write_page(&page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: src_rel.to_string(),
        path: page_rel.clone(),
        one_liner: src_rel.to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let report = wiki.refresh().unwrap();
    assert!(
        report.refreshed.is_empty(),
        "up-to-date page must not refresh; report = {:?}",
        report,
    );
    assert!(
        report.up_to_date >= 1,
        "up_to_date counter must increment; report = {:?}",
        report,
    );
}

#[test]
fn refresh_skips_non_entity_pages() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/c.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn c() {}\n").unwrap();
    // Concept page with a .rs source + missing entity_kind. Rule 8
    // wouldn't fire on it either; refresh must likewise ignore.
    let page = WikiPage {
        title: "concept".to_string(),
        page_type: PageType::Concept,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![src_rel.to_string()],
        last_updated: "2020-01-01 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# concept\n".to_string(),
    };
    wiki.write_page("concepts/c.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "concept".to_string(),
        path: "concepts/c.md".to_string(),
        one_liner: "concept".to_string(),
        category: PageType::Concept,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let report = wiki.refresh().unwrap();
    assert!(
        report.refreshed.is_empty(),
        "non-Entity page must not refresh; report = {:?}",
        report,
    );
    assert!(
        report.missing_sources.is_empty(),
        "non-Entity page should not be scanned for sources; report = {:?}",
        report,
    );
    assert!(
        report.errors.is_empty(),
        "non-Entity page should produce no errors; report = {:?}",
        report,
    );
}

#[test]
fn refresh_handles_missing_source_file_gracefully() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    // Page references a .rs source that doesn't exist on disk. The
    // `needs_kind_backfill` branch triggers (entity_kind None + .rs)
    // so refresh tries to read the file and gets NotFound.
    let page = WikiPage {
        title: "gone".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/gone.rs".to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# gone\n".to_string(),
    };
    wiki.write_page("entities/gone.md", &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "gone".to_string(),
        path: "entities/gone.md".to_string(),
        one_liner: "gone".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let report = wiki.refresh().unwrap();
    assert_eq!(
        report.missing_sources,
        vec!["src/gone.rs".to_string()],
        "absent source must be reported as missing; report = {:?}",
        report,
    );
    assert!(
        report.errors.is_empty(),
        "missing source is not an error; report = {:?}",
        report,
    );
    assert!(
        report.refreshed.is_empty(),
        "missing source must not count as refreshed; report = {:?}",
        report,
    );
}

#[test]
fn refresh_clears_missing_entity_kind_lint_finding() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src_rel = "src/legacy2.rs";
    let src_abs = proj.join(src_rel);
    std::fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    std::fs::write(&src_abs, "pub fn legacy() {}\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        &entity_page_rel(src_rel),
        vec![src_rel.to_string()],
        "2099-01-01 00:00:00",
    );

    let before = wiki.lint().unwrap();
    assert!(
        before
            .iter()
            .any(|f| f.kind == WikiLintKind::MissingEntityKind),
        "pre-refresh lint must flag MissingEntityKind; findings = {:?}",
        before,
    );

    let report = wiki.refresh().unwrap();
    assert_eq!(
        report.refreshed,
        vec![src_rel.to_string()],
        "refresh must re-ingest the legacy page; report = {:?}",
        report,
    );

    let after = wiki.lint().unwrap();
    assert!(
        after
            .iter()
            .all(|f| f.kind != WikiLintKind::MissingEntityKind),
        "post-refresh lint must not flag MissingEntityKind; findings = {:?}",
        after,
    );
}

// ─── Cycle 45: project summary page ────────────────────────────────────
//
// `Wiki::build_project_summary()` aggregates `purpose`, `entity_kind`,
// and `dependencies` from Entity pages into a Summary page.
// `Wiki::write_project_summary()` persists it at `summaries/project.md`,
// upserts the index entry, and appends a `summary` verb to the log.

/// Test helper: install an Entity page with specific `purpose` +
/// `entity_kind` + `dependencies` populated, matching post-Cycle-42
/// schema shape.
fn install_rich_entity_page(
    wiki: &Wiki,
    page_rel: &str,
    title: &str,
    source_rel: &str,
    kind: Option<EntityKind>,
    purpose: Option<&str>,
    dependencies: Vec<String>,
) {
    let page = WikiPage {
        title: title.to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![source_rel.to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: kind,
        purpose: purpose.map(|p| p.to_string()),
        key_exports: vec![],
        dependencies,
        outcome: None,
        scope: vec![],
        body: format!("# {}\n", title),
    };
    wiki.write_page(page_rel, &page).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: title.to_string(),
        path: page_rel.to_string(),
        one_liner: format!("File: {}", source_rel),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();
}

#[test]
fn build_project_summary_aggregates_entity_purposes() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "auth",
        "src/auth.rs",
        Some(EntityKind::Function),
        Some("handles authentication"),
        vec![],
    );
    install_rich_entity_page(
        &wiki,
        "entities/b.md",
        "routing",
        "src/router.rs",
        Some(EntityKind::Struct),
        Some("routes requests"),
        vec![],
    );
    install_rich_entity_page(
        &wiki,
        "entities/c.md",
        "storage",
        "src/store.rs",
        Some(EntityKind::Trait),
        Some("persists data"),
        vec![],
    );

    let page = wiki.build_project_summary().unwrap();
    assert!(page.body.contains("**auth** — handles authentication"));
    assert!(page.body.contains("**routing** — routes requests"));
    assert!(page.body.contains("**storage** — persists data"));
}

#[test]
fn build_project_summary_sources_are_source_files_not_entity_pages() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "auth",
        "src/auth.rs",
        Some(EntityKind::Function),
        Some("handles authentication"),
        vec![],
    );
    install_rich_entity_page(
        &wiki,
        "entities/b.md",
        "routing",
        "src/router.rs",
        Some(EntityKind::Struct),
        Some("routes requests"),
        vec![],
    );

    let page = wiki.build_project_summary().unwrap();
    assert_eq!(
        page.sources,
        vec!["src/auth.rs".to_string(), "src/router.rs".to_string()],
        "summary sources must cite tracked source files, not wiki entity pages",
    );
}

#[test]
fn build_project_summary_counts_entity_kinds() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/f1.md",
        "f1",
        "src/f1.rs",
        Some(EntityKind::Function),
        None,
        vec![],
    );
    install_rich_entity_page(
        &wiki,
        "entities/f2.md",
        "f2",
        "src/f2.rs",
        Some(EntityKind::Function),
        None,
        vec![],
    );
    install_rich_entity_page(
        &wiki,
        "entities/s1.md",
        "s1",
        "src/s1.rs",
        Some(EntityKind::Struct),
        None,
        vec![],
    );

    let page = wiki.build_project_summary().unwrap();
    // BTreeMap iteration order follows EntityKind's derived Ord
    // (declaration order: Function < Struct). The body must show
    // Functions first.
    assert!(
        page.body.contains("function: 2, struct: 1"),
        "body missing expected counts: {}",
        page.body,
    );
}

#[test]
fn build_project_summary_lists_top_dependencies_by_frequency() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "a",
        "src/a.rs",
        Some(EntityKind::Function),
        None,
        vec!["std::io".to_string()],
    );
    install_rich_entity_page(
        &wiki,
        "entities/b.md",
        "b",
        "src/b.rs",
        Some(EntityKind::Function),
        None,
        vec!["std::io".to_string(), "serde".to_string()],
    );
    install_rich_entity_page(
        &wiki,
        "entities/c.md",
        "c",
        "src/c.rs",
        Some(EntityKind::Function),
        None,
        vec!["std::io".to_string(), "tokio".to_string()],
    );

    let page = wiki.build_project_summary().unwrap();
    // `std::io` appears 3x, serde and tokio appear 1x each.
    // Frequency sort puts `std::io` first with count 3.
    let deps_idx = page
        .body
        .find("Most-cited dependencies:")
        .expect("summary body must have dependencies line");
    let deps_line = &page.body[deps_idx..];
    assert!(
        deps_line.contains("`std::io` (3)"),
        "top dep must be std::io (3): {}",
        deps_line,
    );
    let io_pos = deps_line.find("std::io").unwrap();
    let serde_pos = deps_line.find("serde").unwrap_or(usize::MAX);
    let tokio_pos = deps_line.find("tokio").unwrap_or(usize::MAX);
    assert!(
        io_pos < serde_pos && io_pos < tokio_pos,
        "std::io must sort before less-frequent deps: {}",
        deps_line,
    );
}

#[test]
fn build_project_summary_momentum_section_renames_and_surfaces_hot_paths() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Three ingests of `src/hot.rs`, one of `src/cold.rs` — `hot.rs`
    // wins both paths and modules (both live under `src`).
    for _ in 0..3 {
        log.append("ingest", "src/hot.rs").unwrap();
    }
    log.append("ingest", "src/cold.rs").unwrap();
    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("## Momentum"),
        "summary body must use the ## Momentum heading: {}",
        page.body,
    );
    assert!(
        !page.body.contains("Recent activity"),
        "summary body must not retain the old Recent activity heading: {}",
        page.body,
    );
    assert!(
        page.body.contains("`src/hot.rs` (3)"),
        "hot path with count must appear: {}",
        page.body,
    );
    assert!(
        page.body.contains("`src/cold.rs` (1)"),
        "less-hot path must also appear: {}",
        page.body,
    );
    // Code fences removed — section is prose, not a pasted log tail.
    assert!(
        !page.body.contains("```\n["),
        "Momentum section should not render as a code-fenced log dump: {}",
        page.body,
    );
}

#[test]
fn build_project_summary_momentum_section_empty_wiki_uses_placeholder() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("## Momentum"),
        "empty-wiki body must still render the Momentum heading: {}",
        page.body,
    );
    assert!(
        page.body.contains("no ingest activity yet"),
        "empty-wiki body must render the no-activity placeholder: {}",
        page.body,
    );
}

#[test]
fn momentum_report_default_on_missing_log() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // log.md does not exist yet — must not error out.
    let report = wiki.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap();
    assert_eq!(report.total_entries, 0);
    assert_eq!(report.window_processed, 0);
    assert!(report.hot_paths.is_empty());
    assert!(report.hot_modules.is_empty());
}

#[test]
fn momentum_ignores_non_ingest_verbs() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Non-ingest verbs must not inflate the counts.
    log.append("summary", "summaries/project.md").unwrap();
    log.append("concept", "concepts/dep-serde.md").unwrap();
    log.append("compact", "entities/foo.md").unwrap();
    log.append("ingest", "src/real.rs").unwrap();
    let report = wiki.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap();
    assert_eq!(report.total_entries, 4);
    assert_eq!(report.window_processed, 1);
    assert_eq!(
        report.hot_paths,
        vec![("src/real.rs".to_string(), 1)],
        "only the ingest entry should appear in hot_paths: {:?}",
        report,
    );
}

#[test]
fn momentum_ranks_by_frequency_then_path_asc() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Two ingests each for `src/a.rs` and `src/b.rs` — tied on count,
    // so `src/a.rs` must come first alphabetically.
    for _ in 0..2 {
        log.append("ingest", "src/a.rs").unwrap();
        log.append("ingest", "src/b.rs").unwrap();
    }
    log.append("ingest", "src/c.rs").unwrap();
    let report = wiki.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap();
    assert_eq!(
        report.hot_paths,
        vec![
            ("src/a.rs".to_string(), 2),
            ("src/b.rs".to_string(), 2),
            ("src/c.rs".to_string(), 1),
        ],
        "tie-break on path ascending: {:?}",
        report,
    );
}

#[test]
fn momentum_truncates_to_top_n() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Ten distinct paths, each ingested once — cap at MOMENTUM_TOP_N.
    for i in 0..10 {
        log.append("ingest", &format!("src/m{:02}.rs", i)).unwrap();
    }
    let report = wiki.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap();
    assert_eq!(report.hot_paths.len(), MOMENTUM_TOP_N);
    // Alphabetical tie-break: src/m00..src/m04 should win.
    let names: Vec<String> = report.hot_paths.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(
        names,
        vec![
            "src/m00.rs".to_string(),
            "src/m01.rs".to_string(),
            "src/m02.rs".to_string(),
            "src/m03.rs".to_string(),
            "src/m04.rs".to_string(),
        ],
        "MOMENTUM_TOP_N truncation order: {:?}",
        names,
    );
}

#[test]
fn momentum_count_window_walks_newest_first() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Four older ingests of `src/old.rs`, then three newer ingests of
    // `src/new.rs`. With window = 3 only the newest three hit.
    for _ in 0..4 {
        log.append("ingest", "src/old.rs").unwrap();
    }
    for _ in 0..3 {
        log.append("ingest", "src/new.rs").unwrap();
    }
    let report = wiki.momentum(3).unwrap();
    assert_eq!(report.total_entries, 7);
    assert_eq!(report.window_processed, 3);
    assert_eq!(
        report.hot_paths,
        vec![("src/new.rs".to_string(), 3)],
        "window must walk newest-first: {:?}",
        report,
    );
}

#[test]
fn momentum_aggregates_hot_modules() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Three ingests under `src/wiki/`, two under `src/tools/` — the
    // modules report must reflect the bucket, not the raw paths.
    log.append("ingest", "src/wiki/mod.rs").unwrap();
    log.append("ingest", "src/wiki/lint.rs").unwrap();
    log.append("ingest", "src/wiki/momentum.rs").unwrap();
    log.append("ingest", "src/tools/bash.rs").unwrap();
    log.append("ingest", "src/tools/grep.rs").unwrap();
    let report = wiki.momentum(MOMENTUM_DEFAULT_WINDOW).unwrap();
    assert_eq!(
        report.hot_modules,
        vec![("src/wiki".to_string(), 3), ("src/tools".to_string(), 2),],
        "hot_modules must bucket by parent dir desc-count: {:?}",
        report,
    );
}

// ── Wiki::planner_brief (cycle 54) ────────────────────────────────────

#[test]
fn planner_brief_empty_wiki_is_empty() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert!(
        brief.is_empty(),
        "fresh wiki yields empty brief: {:?}",
        brief
    );
    assert!(
        brief.render(PLANNER_BRIEF_BUDGET_CHARS).is_none(),
        "empty brief renders to None",
    );
}

#[test]
fn planner_brief_surfaces_top_hot_paths_from_momentum() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Three paths with 3/2/1 ingests respectively — hot_paths must
    // preserve that order and cap at PLANNER_BRIEF_HOT_PATHS.
    for _ in 0..3 {
        log.append("ingest", "src/a.rs").unwrap();
    }
    for _ in 0..2 {
        log.append("ingest", "src/b.rs").unwrap();
    }
    log.append("ingest", "src/c.rs").unwrap();
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.hot_paths,
        vec![
            ("src/a.rs".to_string(), 3),
            ("src/b.rs".to_string(), 2),
            ("src/c.rs".to_string(), 1),
        ],
    );
}

#[test]
fn planner_brief_includes_item_and_export_drift_pages_deduped() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();

    // Page `dual.md` drifts in BOTH dimensions — must appear once.
    // Source has only `keep_me`; page claims `drifted` in body AND
    // `also_drifted` in key_exports.
    std::fs::write(proj.join("src/dual.rs"), "pub fn keep_me() {}\n").unwrap();
    install_entity_page_with_exports(
        &wiki,
        "entities/dual.md",
        "src/dual.rs",
        vec![KeyExport {
            kind: EntityKind::Function,
            name: "also_drifted".to_string(),
        }],
        "2099-01-01 00:00:00",
    );
    // Overwrite body so ItemDrift also fires (Items section).
    let page = WikiPage {
        title: "src/dual.rs".to_string(),
        page_type: PageType::Entity,
        layer: crate::wiki::Layer::Kernel,
        sources: vec!["src/dual.rs".to_string()],
        last_updated: "2099-01-01 00:00:00".to_string(),
        entity_kind: Some(EntityKind::Unknown),
        purpose: None,
        key_exports: vec![KeyExport {
            kind: EntityKind::Function,
            name: "also_drifted".to_string(),
        }],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# src/dual.rs\n\n## Items\n- `drifted`\n".to_string(),
    };
    wiki.write_page("entities/dual.md", &page).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief
            .drifting_pages
            .iter()
            .filter(|p| p.as_str() == "entities/dual.md")
            .count(),
        1,
        "dual-drift page must be deduped: {:?}",
        brief.drifting_pages,
    );
}

#[test]
fn planner_brief_tallies_lint_kinds_with_count_desc_kind_asc_ordering() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Craft 3 OrphanIndexEntry and 1 CategoryMismatch via a
    // hand-written index. Category mismatch means path under
    // `entities/` but category Synthesis (or similar) — simplest
    // is to list nonexistent entity paths (orphans) plus one
    // category-mismatched entry that also doesn't exist on disk.
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "a".to_string(),
                path: "entities/a.md".to_string(),
                one_liner: String::new(),
                category: PageType::Synthesis,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "b".to_string(),
                path: "entities/b.md".to_string(),
                one_liner: String::new(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "c".to_string(),
                path: "entities/c.md".to_string(),
                one_liner: String::new(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
            IndexEntry {
                title: "d".to_string(),
                path: "entities/d.md".to_string(),
                one_liner: String::new(),
                category: PageType::Entity,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    wiki.save_index(&idx).unwrap();
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    // OrphanIndexEntry has enum-variant index 0, CategoryMismatch 2.
    // Three OrphanIndexEntry (b/c/d — all missing on disk) vs one
    // CategoryMismatch (a — category Synthesis but path entities/)
    // plus a under entities/ is also missing → orphan too, so
    // OrphanIndexEntry count is 4, CategoryMismatch is 1.
    let orphan_count = brief
        .lint_counts
        .iter()
        .find(|(k, _)| *k == WikiLintKind::OrphanIndexEntry)
        .map_or(0, |(_, n)| *n);
    let mismatch_count = brief
        .lint_counts
        .iter()
        .find(|(k, _)| *k == WikiLintKind::CategoryMismatch)
        .map_or(0, |(_, n)| *n);
    assert!(
        orphan_count >= 3,
        "expected ≥3 orphans: {:?}",
        brief.lint_counts,
    );
    assert_eq!(
        mismatch_count, 1,
        "one category mismatch: {:?}",
        brief.lint_counts
    );
    // First entry must be the larger count, and for ties, the
    // lower-enum-index kind. Orphan wins on both axes here.
    assert_eq!(
        brief.lint_counts.first().map(|(k, _)| *k),
        Some(WikiLintKind::OrphanIndexEntry),
        "count desc ordering: {:?}",
        brief.lint_counts,
    );
}

#[test]
fn planner_brief_caps_drifting_pages_at_max_findings() {
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path();
    let wiki = Wiki::open(proj).unwrap();
    std::fs::create_dir_all(proj.join("src")).unwrap();
    // Create 8 item-drift pages; cap is PLANNER_BRIEF_MAX_DRIFT (5).
    for i in 0..8 {
        let src_rel = format!("src/d{}.rs", i);
        let page_rel = format!("entities/d{}.md", i);
        std::fs::write(proj.join(&src_rel), "pub fn kept() {}\n").unwrap();
        install_entity_page(
            &wiki,
            &page_rel,
            &src_rel,
            &format!("# {}\n\n## Items\n- `removed`\n", src_rel),
            "2099-01-01 00:00:00",
        );
    }
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.drifting_pages.len(),
        PLANNER_BRIEF_MAX_DRIFT,
        "drifting_pages must be capped at max_findings: {:?}",
        brief.drifting_pages,
    );
}

#[test]
fn planner_brief_render_respects_budget_and_line_boundary() {
    // Build a brief with enough content to exceed a tight budget.
    let brief = PlannerBrief {
        hot_paths: (0..30)
            .map(|i| (format!("src/mod_with_longer_name_{:02}.rs", i), 1))
            .collect(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(200).expect("non-empty brief renders");
    assert!(
        rendered.len() <= 200,
        "render must honour budget: len={}, body={:?}",
        rendered.len(),
        rendered,
    );
    assert!(
        rendered.ends_with("[...truncated]"),
        "oversized render must end with truncation marker: {:?}",
        rendered,
    );
    // No line in the output ends mid-path (each non-empty content
    // line must be valid markdown — either header, bullet, or blank).
    for line in rendered.lines() {
        if line == "[...truncated]" || line.is_empty() {
            continue;
        }
        assert!(
            line.starts_with('#') || line.starts_with("- "),
            "truncation must respect line boundaries; got: {:?}",
            line,
        );
    }
}

#[test]
fn planner_brief_render_empty_returns_none() {
    let brief = PlannerBrief::default();
    assert!(
        brief.render(PLANNER_BRIEF_BUDGET_CHARS).is_none(),
        "empty brief renders to None",
    );
}

#[test]
fn planner_brief_excludes_non_ingest_momentum() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // `concept`/`summary` verbs must not inflate hot_paths.
    log.append("concept", "concepts/dep-x.md").unwrap();
    log.append("summary", "summaries/project.md").unwrap();
    log.append("ingest", "src/real.rs").unwrap();
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.hot_paths,
        vec![("src/real.rs".to_string(), 1)],
        "only ingest verbs feed hot_paths: {:?}",
        brief.hot_paths,
    );
}

// ── PlannerBrief.recent_cycles tests (Cycle 57) ──────────────────────

#[test]
fn parse_cycle_number_from_synthesis_cycle_path() {
    assert_eq!(
        parse_cycle_number_from_synthesis_path(
            "synthesis/cycle-05-my-chain-20260418-143000-123456789.md"
        ),
        Some(5)
    );
}

#[test]
fn parse_cycle_number_rejects_compact_synthesis_path() {
    // Compact synthesis pages must not surface as recent cycles.
    assert_eq!(
        parse_cycle_number_from_synthesis_path("synthesis/compact-20260418-143000-123456789.md"),
        None
    );
}

#[test]
fn parse_cycle_number_handles_unpadded_and_large_cycle_numbers() {
    assert_eq!(
        parse_cycle_number_from_synthesis_path("synthesis/cycle-1-x-.md"),
        Some(1)
    );
    assert_eq!(
        parse_cycle_number_from_synthesis_path("synthesis/cycle-9999-x-.md"),
        Some(9999)
    );
}

#[test]
fn extract_chain_name_from_one_liner_strips_prefix() {
    assert_eq!(
        extract_chain_name_from_one_liner("Cycle 5 of self-improve"),
        Some("self-improve".to_string())
    );
}

#[test]
fn extract_chain_name_from_one_liner_rejects_non_matching() {
    assert_eq!(
        extract_chain_name_from_one_liner("Compact snapshot (12 messages)"),
        None
    );
}

#[test]
fn planner_brief_surfaces_recent_cycles_newest_first() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Write five cycle synthesis pages via the runner-facing API so
    // filename format stays in lockstep with production.
    for cycle in 1..=5 {
        wiki.write_cycle_synthesis(cycle, "demo", &[], None)
            .unwrap();
    }

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let cycles: Vec<usize> = brief.recent_cycles.iter().map(|r| r.cycle).collect();
    assert_eq!(
        cycles,
        vec![5, 4, 3],
        "recent_cycles must be newest-first and capped at 3: {:?}",
        brief.recent_cycles
    );
    for rc in &brief.recent_cycles {
        assert_eq!(rc.chain, "demo");
        assert!(
            rc.page_path.starts_with(SYNTHESIS_CYCLE_PREFIX),
            "page_path must reference the cycle prefix: {}",
            rc.page_path
        );
    }
}

#[test]
fn planner_brief_recent_cycles_capped_at_three() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    for cycle in 1..=10 {
        wiki.write_cycle_synthesis(cycle, "demo", &[], None)
            .unwrap();
    }

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.recent_cycles.len(),
        PLANNER_BRIEF_RECENT_CYCLES,
        "recent_cycles must honour PLANNER_BRIEF_RECENT_CYCLES cap"
    );
}

#[test]
fn planner_brief_render_includes_recent_cycles_section() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![
            RecentCycle {
                cycle: 9,
                chain: "self-improve".to_string(),
                page_path: "synthesis/cycle-09-self-improve-ts.md".to_string(),
                last_updated: None,
                outcome: None,
            },
            RecentCycle {
                cycle: 8,
                chain: "self-improve".to_string(),
                page_path: "synthesis/cycle-08-self-improve-ts.md".to_string(),
                last_updated: None,
                outcome: None,
            },
        ],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("### Recent cycles"),
        "missing Recent cycles header: {}",
        rendered
    );
    assert!(
        rendered
            .contains("- Cycle 9 (self-improve): `.dm/wiki/synthesis/cycle-09-self-improve-ts.md`"),
        "missing first bullet: {}",
        rendered
    );
    assert!(
        rendered
            .contains("- Cycle 8 (self-improve): `.dm/wiki/synthesis/cycle-08-self-improve-ts.md`"),
        "missing second bullet: {}",
        rendered
    );
}

#[test]
fn planner_brief_is_empty_false_when_only_recent_cycles_set() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "x".to_string(),
            page_path: "synthesis/cycle-01-x-ts.md".to_string(),
            last_updated: None,
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    assert!(!brief.is_empty());
    assert!(
        brief.render(1024).is_some(),
        "brief with only recent_cycles must render"
    );
}

// ── RecentCycle.last_updated tests (Cycle 79) ────────────────────────

#[test]
fn recent_cycle_default_last_updated_is_none() {
    // Explicit construction — no `Default` impl, so verify that the
    // `last_updated: None` default stays an `Option` the enrichment
    // layer can opt into rather than a required field.
    let rc = RecentCycle {
        cycle: 1,
        chain: "demo".to_string(),
        page_path: "synthesis/cycle-01-demo-ts.md".to_string(),
        last_updated: None,
        outcome: None,
    };
    assert!(rc.last_updated.is_none());
}

#[test]
fn planner_brief_recent_cycles_last_updated_populated_when_page_readable() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    wiki.write_cycle_synthesis(7, "demo", &[], None).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(brief.recent_cycles.len(), 1);
    assert!(
        brief.recent_cycles[0].last_updated.is_some(),
        "last_updated must be populated from synthesis page frontmatter: {:?}",
        brief.recent_cycles[0],
    );
}

#[test]
fn planner_brief_recent_cycles_last_updated_matches_page_frontmatter() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(4, "demo", &[], None)
        .unwrap()
        .expect("auto-ingest on in tests");

    let raw = std::fs::read_to_string(proj.join(".dm/wiki").join(&page_rel)).unwrap();
    let parsed = WikiPage::parse(&raw).expect("synthesis page frontmatter valid");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.recent_cycles[0].last_updated,
        Some(parsed.last_updated),
        "brief timestamp must round-trip from page frontmatter",
    );
}

#[test]
fn planner_brief_recent_cycles_last_updated_none_when_file_missing() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(2, "demo", &[], None)
        .unwrap()
        .expect("auto-ingest on in tests");

    // Simulate a legacy pre-C85 index (no cached `last_updated`) so the
    // file-read fallback path is the one under test.
    let mut idx = wiki.load_index().unwrap();
    for e in &mut idx.entries {
        e.last_updated = None;
    }
    wiki.save_index(&idx).unwrap();

    // Delete the page body but leave the index entry in place — the
    // enrichment must degrade to `None` without dropping the row.
    std::fs::remove_file(proj.join(".dm/wiki").join(&page_rel)).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.recent_cycles.len(),
        1,
        "row must survive missing file"
    );
    assert_eq!(
        brief.recent_cycles[0].last_updated, None,
        "missing file must degrade to None",
    );
}

#[test]
fn planner_brief_recent_cycles_last_updated_none_when_frontmatter_malformed() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(3, "demo", &[], None)
        .unwrap()
        .expect("auto-ingest on in tests");

    // Simulate a legacy pre-C85 index (no cached `last_updated`) so the
    // file-read fallback path is the one under test.
    let mut idx = wiki.load_index().unwrap();
    for e in &mut idx.entries {
        e.last_updated = None;
    }
    wiki.save_index(&idx).unwrap();

    // Overwrite with raw markdown lacking the `---\n` frontmatter
    // fences — `WikiPage::parse` returns `None` and enrichment degrades.
    std::fs::write(
        proj.join(".dm/wiki").join(&page_rel),
        "no frontmatter here, just body\n",
    )
    .unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(brief.recent_cycles.len(), 1);
    assert_eq!(
        brief.recent_cycles[0].last_updated, None,
        "malformed frontmatter must degrade to None",
    );
}

#[test]
fn planner_brief_render_includes_timestamp_when_present() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 9,
            chain: "self-improve".to_string(),
            page_path: "synthesis/cycle-09-self-improve-ts.md".to_string(),
            last_updated: Some("2026-04-18 14:00:00".to_string()),
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("2026-04-18 14:00:00"),
        "timestamp must be surfaced on the cycle line: {}",
        rendered,
    );
    assert!(
        rendered.contains(" — 2026-04-18 14:00:00"),
        "timestamp must be separated with ' — ' delimiter: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_omits_timestamp_dash_when_absent() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 9,
            chain: "self-improve".to_string(),
            page_path: "synthesis/cycle-09-self-improve-ts.md".to_string(),
            last_updated: None,
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    // Guard: no timestamp suffix attached when `last_updated` is None.
    // Narrow check uses the `" — 20"` prefix that only timestamps produce.
    assert!(
        !rendered.contains(" — 20"),
        "no timestamp delimiter when last_updated is None: {}",
        rendered,
    );
}

#[test]
fn recent_cycle_last_updated_preserves_newest_first_ordering() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "alpha", &[], None).unwrap();
    wiki.write_cycle_synthesis(2, "beta", &[], None).unwrap();
    wiki.write_cycle_synthesis(3, "gamma", &[], None).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let cycles: Vec<usize> = brief.recent_cycles.iter().map(|r| r.cycle).collect();
    assert_eq!(
        cycles,
        vec![3, 2, 1],
        "newest-first ordering must survive last_updated enrichment",
    );
    for rc in &brief.recent_cycles {
        assert!(
            rc.last_updated.is_some(),
            "every row must carry a timestamp when files are readable: {:?}",
            rc,
        );
    }
}

#[test]
fn planner_brief_render_no_staleness_warning_when_recent() {
    let now_str = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "demo".to_string(),
            page_path: "synthesis/cycle-01-demo-ts.md".to_string(),
            last_updated: Some(now_str),
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        !rendered.contains("stale"),
        "fresh cycle must not trigger staleness warning: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_no_staleness_warning_when_no_recent_cycles() {
    let brief = PlannerBrief {
        hot_paths: vec![("src/demo.rs".to_string(), 3)],
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        !rendered.contains("stale"),
        "no recent_cycles means no staleness warning: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_no_staleness_warning_when_timestamp_absent() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "demo".to_string(),
            page_path: "synthesis/cycle-01-demo-ts.md".to_string(),
            last_updated: None,
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        !rendered.contains("stale"),
        "missing last_updated must not trigger staleness warning: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_contains_staleness_warning_when_cycle_is_old() {
    // Timestamp anchored in 2020 so this test can't go green by
    // accident on any plausible future system clock.
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "demo".to_string(),
            page_path: "synthesis/cycle-01-demo-ts.md".to_string(),
            last_updated: Some("2020-01-01 00:00:00".to_string()),
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("stale"),
        "old cycle must surface staleness warning: {}",
        rendered,
    );
    assert!(
        rendered.contains("/wiki refresh"),
        "staleness warning must include next-steps hint: {}",
        rendered,
    );
}

// ── C84 hot_modules + fresh_pages tests ──────────────────────────────

#[test]
fn planner_brief_exposes_hot_modules_from_momentum() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let log = wiki.log();
    // Two modules (`src/foo` via foo/*.rs, `src/bar` via bar/*.rs)
    // with 3 vs 2 ingests. Module counts must preserve that ordering
    // and forward through `planner_brief.hot_modules`.
    for _ in 0..3 {
        log.append("ingest", "src/foo/a.rs").unwrap();
    }
    for _ in 0..2 {
        log.append("ingest", "src/bar/b.rs").unwrap();
    }
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.hot_modules,
        vec![("src/foo".to_string(), 3), ("src/bar".to_string(), 2),],
        "hot_modules must forward momentum ordering",
    );
}

#[test]
fn planner_brief_render_includes_hot_modules_section() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: vec![("src/foo".to_string(), 7)],
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("### Hot modules (per directory)"),
        "hot_modules section header missing: {}",
        rendered,
    );
    assert!(
        rendered.contains("`src/foo`"),
        "hot_modules module name missing: {}",
        rendered,
    );
    assert!(
        rendered.contains("7 ingest(s)"),
        "hot_modules count missing: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_omits_hot_modules_when_empty() {
    let brief = PlannerBrief {
        hot_paths: vec![("src/only.rs".to_string(), 1)],
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        !rendered.contains("### Hot modules"),
        "empty hot_modules must not emit section header: {}",
        rendered,
    );
}

#[test]
fn planner_brief_exposes_fresh_entity_pages() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let now_str = chrono::Local::now()
        .naive_local()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    install_entity_page_with_exports(&wiki, "entities/fresh.md", "src/fresh.rs", vec![], &now_str);
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.fresh_pages,
        vec![("entities/fresh.md".to_string(), now_str)],
        "entity page with `now` last_updated must surface in fresh_pages",
    );
}

#[test]
fn planner_brief_filters_synthesis_and_summary_from_fresh_pages() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let now_str = chrono::Local::now()
        .naive_local()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();

    // Entity page — should surface.
    install_entity_page_with_exports(&wiki, "entities/keep.md", "src/keep.rs", vec![], &now_str);

    // Synthesis page — category filter must drop it. Write the page
    // file so frontmatter reads would succeed, then register the
    // index entry under PageType::Synthesis.
    let synth_page = WikiPage {
        title: "cycle-99".to_string(),
        page_type: PageType::Synthesis,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: now_str.clone(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# cycle-99\n".to_string(),
    };
    wiki.write_page("synthesis/cycle-99-demo-ts.md", &synth_page)
        .unwrap();

    // Summary page — also filtered.
    let summary_page = WikiPage {
        title: "Project".to_string(),
        page_type: PageType::Summary,
        layer: crate::wiki::Layer::Kernel,
        sources: vec![],
        last_updated: now_str.clone(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# Project\n".to_string(),
    };
    wiki.write_page("summaries/project.md", &summary_page)
        .unwrap();

    // Manually register both non-entity pages in the index under
    // their respective categories.
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "cycle-99".to_string(),
        path: "synthesis/cycle-99-demo-ts.md".to_string(),
        one_liner: "Chain demo cycle 99".to_string(),
        category: PageType::Synthesis,
        last_updated: None,
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "Project".to_string(),
        path: "summaries/project.md".to_string(),
        one_liner: "Project summary".to_string(),
        category: PageType::Summary,
        last_updated: None,
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let paths: Vec<&str> = brief.fresh_pages.iter().map(|(p, _)| p.as_str()).collect();
    assert!(
        paths.contains(&"entities/keep.md"),
        "entity page must surface: paths={:?}",
        paths,
    );
    assert!(
        !paths.contains(&"synthesis/cycle-99-demo-ts.md"),
        "synthesis page must be filtered: paths={:?}",
        paths,
    );
    assert!(
        !paths.contains(&"summaries/project.md"),
        "summary page must be filtered: paths={:?}",
        paths,
    );
}

#[test]
fn planner_brief_render_includes_fresh_pages_section() {
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: vec![(
            "entities/fresh.md".to_string(),
            "2026-04-19 12:00:00".to_string(),
        )],
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("### Fresh entity pages (updated <24h)"),
        "fresh_pages section header missing: {}",
        rendered,
    );
    assert!(
        rendered.contains(".dm/wiki/entities/fresh.md"),
        "fresh_pages path missing: {}",
        rendered,
    );
    assert!(
        rendered.contains("2026-04-19 12:00:00"),
        "fresh_pages timestamp missing: {}",
        rendered,
    );
}

#[test]
fn planner_brief_caps_fresh_pages() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let now_str = chrono::Local::now()
        .naive_local()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    // Install 2× the cap so truncation is unambiguously exercised.
    let install_count = PLANNER_BRIEF_FRESH_PAGES * 2;
    for i in 0..install_count {
        install_entity_page_with_exports(
            &wiki,
            &format!("entities/e_{:02}.md", i),
            &format!("src/e_{:02}.rs", i),
            vec![],
            &now_str,
        );
    }
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(
        brief.fresh_pages.len(),
        PLANNER_BRIEF_FRESH_PAGES,
        "fresh_pages must cap at PLANNER_BRIEF_FRESH_PAGES; got {}",
        brief.fresh_pages.len(),
    );
}

#[test]
fn planner_brief_is_empty_respects_hot_modules_and_fresh_pages() {
    // hot_modules alone must flip is_empty to false.
    let with_modules = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: vec![("src/foo".to_string(), 1)],
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: Vec::new(),
    };
    assert!(
        !with_modules.is_empty(),
        "hot_modules-only brief must not be empty",
    );

    // fresh_pages alone must flip is_empty to false.
    let with_fresh = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: Vec::new(),
        fresh_pages: vec![(
            "entities/fresh.md".to_string(),
            "2026-04-19 12:00:00".to_string(),
        )],
    };
    assert!(
        !with_fresh.is_empty(),
        "fresh_pages-only brief must not be empty",
    );

    // All-empty must still be empty (regression guard).
    let empty = PlannerBrief::default();
    assert!(empty.is_empty(), "default brief must be empty");
}

#[test]
fn build_project_summary_handles_empty_wiki() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = wiki.build_project_summary().unwrap();
    assert_eq!(page.page_type, PageType::Summary);
    assert_eq!(page.title, "Project");
    assert!(
        page.body.contains("Generated from 0 entity page(s)."),
        "empty-wiki body must advertise zero entities: {}",
        page.body,
    );
    assert!(
        page.body.contains("*(no entity pages with `purpose` yet"),
        "empty-wiki body must render the empty-purpose placeholder: {}",
        page.body,
    );
}

#[test]
fn write_project_summary_upserts_index_entry() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "a",
        "src/a.rs",
        Some(EntityKind::Function),
        Some("alpha"),
        vec![],
    );

    let first = wiki.write_project_summary().unwrap();
    assert_eq!(first.path, "summaries/project.md");
    assert_eq!(first.entity_count, 1);

    // Add another entity then regenerate — exactly one summary entry
    // should exist; the one_liner should reflect the new entity count.
    install_rich_entity_page(
        &wiki,
        "entities/b.md",
        "b",
        "src/b.rs",
        Some(EntityKind::Struct),
        Some("beta"),
        vec![],
    );
    let second = wiki.write_project_summary().unwrap();
    assert_eq!(second.entity_count, 2);

    let idx = wiki.load_index().unwrap();
    let summary_entries: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.path == "summaries/project.md")
        .collect();
    assert_eq!(
        summary_entries.len(),
        1,
        "index must carry exactly one summary entry after upsert: {:?}",
        idx.entries,
    );
    assert!(
        summary_entries[0].one_liner.contains("2 entities"),
        "one_liner must reflect current entity count: {}",
        summary_entries[0].one_liner,
    );
}

#[test]
fn write_project_summary_appends_to_log() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.write_project_summary().unwrap();

    let log_text = fs::read_to_string(wiki.log().path()).unwrap();
    let has_entry = log_text
        .lines()
        .any(|l| l.contains("summary |") && l.ends_with("summaries/project.md"));
    assert!(
        has_entry,
        "log must gain a `summary | summaries/project.md` line: {}",
        log_text,
    );
}

// ── Cycle 47: summary-dirty marker ──────────────────────────────────────

#[test]
fn mark_and_clear_summary_dirty_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    assert!(!wiki.is_summary_dirty(), "fresh wiki must not be dirty");
    wiki.mark_summary_dirty().unwrap();
    assert!(wiki.is_summary_dirty(), "mark must flip the flag");
    wiki.clear_summary_dirty().unwrap();
    assert!(!wiki.is_summary_dirty(), "clear must unflip the flag");
}

#[test]
fn clear_summary_dirty_is_idempotent_on_missing_marker() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // No mark set — clear must not error.
    wiki.clear_summary_dirty().unwrap();
    // Second clear still fine.
    wiki.clear_summary_dirty().unwrap();
}

#[test]
fn ensure_summary_current_regenerates_when_dirty_with_entities() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/cfg.md",
        "cfg",
        "src/cfg.rs",
        Some(EntityKind::Struct),
        Some("reads configuration from disk"),
        vec![],
    );
    wiki.mark_summary_dirty().unwrap();
    let report = wiki
        .ensure_summary_current()
        .unwrap()
        .expect("dirty marker must trigger regeneration");
    assert_eq!(report.entity_count, 1);
    assert!(
        wiki.root().join("summaries/project.md").is_file(),
        "summary file must exist after regeneration"
    );
    assert!(
        !wiki.is_summary_dirty(),
        "marker must be cleared after regeneration"
    );
}

#[test]
fn ensure_summary_current_noop_on_clean_wiki() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let report = wiki.ensure_summary_current().unwrap();
    assert!(report.is_none(), "clean wiki must produce Ok(None)");
    assert!(
        !wiki.root().join("summaries/project.md").is_file(),
        "no regeneration must occur when clean"
    );
}

#[test]
fn ensure_summary_current_writes_empty_summary_when_dirty_no_entities() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.mark_summary_dirty().unwrap();
    let report = wiki
        .ensure_summary_current()
        .unwrap()
        .expect("dirty marker must trigger regeneration");
    assert_eq!(report.entity_count, 0);
    let body = fs::read_to_string(wiki.root().join("summaries/project.md")).unwrap();
    assert!(
        body.contains("Generated from 0 entity page(s)."),
        "zero-entity summary must reflect empty state: {}",
        body
    );
    assert!(!wiki.is_summary_dirty());
}

#[test]
fn ingest_file_marks_summary_dirty_on_successful_write() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let src = proj.join("src/app.rs");
    fs::create_dir_all(src.parent().unwrap()).unwrap();
    let content = "pub fn launch() {}\n";
    fs::write(&src, content).unwrap();
    assert!(!wiki.is_summary_dirty(), "precondition: clean");
    wiki.ingest_file(&proj, &src, content).unwrap();
    assert!(
        wiki.is_summary_dirty(),
        "successful ingest must mark the summary dirty"
    );
}

#[test]
fn refresh_regenerates_summary_and_clears_marker() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Stage: a .rs source with an entity page whose entity_kind is None
    // (legacy shape) and a stale last_updated. refresh() must re-ingest,
    // populate entity_kind, then regenerate the summary at batch end.
    let src_rel = "src/legacy.rs";
    let src_abs = proj.join(src_rel);
    fs::create_dir_all(src_abs.parent().unwrap()).unwrap();
    fs::write(&src_abs, "pub struct Legacy;\n").unwrap();
    install_entity_page_missing_kind(
        &wiki,
        "entities/src_legacy_rs.md",
        vec![src_rel.to_string()],
        "1970-01-01 00:00:00",
    );
    wiki.clear_summary_dirty().unwrap();

    let _ = wiki.refresh().unwrap();
    assert!(
        wiki.root().join("summaries/project.md").is_file(),
        "refresh must leave a summary file on disk"
    );
    assert!(
        !wiki.is_summary_dirty(),
        "refresh must clear the dirty marker at the batch boundary"
    );
}

#[test]
fn write_project_summary_clears_dirty_marker() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    wiki.mark_summary_dirty().unwrap();
    wiki.write_project_summary().unwrap();
    assert!(
        !wiki.is_summary_dirty(),
        "explicit /wiki summary must clear the marker itself"
    );
}

// ── Phase 3.6 cross-link foundation (Cycle 33) ──────────────────────────

#[test]
fn inbound_links_zero_for_lone_entity() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let foo = proj.join("src/foo.rs");
    fs::create_dir_all(foo.parent().unwrap()).unwrap();
    fs::write(&foo, "fn main() {}").unwrap();
    wiki.ingest_file(&proj, &foo, "fn main() {}").unwrap();

    let stats = wiki.stats().unwrap();
    assert!(
        stats.most_linked.is_empty(),
        "lone entity should have no inbound links, got {:?}",
        stats.most_linked
    );
}

#[test]
fn inbound_links_bumps_when_other_file_mentions_source() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let foo = proj.join("src/foo.rs");
    let bar = proj.join("src/bar.rs");
    fs::create_dir_all(foo.parent().unwrap()).unwrap();
    fs::write(&foo, "").unwrap();
    let bar_content = "uses src/foo.rs";
    fs::write(&bar, bar_content).unwrap();
    wiki.ingest_file(&proj, &foo, "").unwrap();
    wiki.ingest_file(&proj, &bar, bar_content).unwrap();

    let stats = wiki.stats().unwrap();
    assert_eq!(
        stats.most_linked,
        vec![("entities/src_foo_rs.md".to_string(), 1)],
        "foo should have 1 inbound link from bar, bar zero (hidden from list)"
    );
}

#[test]
fn inbound_links_does_not_self_link() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let foo = proj.join("src/foo.rs");
    fs::create_dir_all(foo.parent().unwrap()).unwrap();
    let content = "// self-reference: src/foo.rs";
    fs::write(&foo, content).unwrap();
    wiki.ingest_file(&proj, &foo, content).unwrap();

    let stats = wiki.stats().unwrap();
    assert!(
        stats.most_linked.is_empty(),
        "self-mention must not inflate own inbound count, got {:?}",
        stats.most_linked
    );
}

#[test]
fn inbound_links_idempotent_on_reingest() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let foo = proj.join("src/foo.rs");
    let bar = proj.join("src/bar.rs");
    fs::create_dir_all(foo.parent().unwrap()).unwrap();
    fs::write(&foo, "").unwrap();
    let bar_content = "refers to src/foo.rs";
    fs::write(&bar, bar_content).unwrap();
    wiki.ingest_file(&proj, &foo, "").unwrap();
    wiki.ingest_file(&proj, &bar, bar_content).unwrap();
    // Bust the dedup cache so re-ingest actually runs.
    reset_ingest_cache_for_tests();
    wiki.ingest_file(&proj, &bar, bar_content).unwrap();
    reset_ingest_cache_for_tests();
    wiki.ingest_file(&proj, &bar, bar_content).unwrap();

    let stats = wiki.stats().unwrap();
    assert_eq!(
        stats.most_linked,
        vec![("entities/src_foo_rs.md".to_string(), 1)],
        "full recompute must not accumulate on re-ingest; got {:?}",
        stats.most_linked
    );
}

#[test]
fn inbound_links_drop_when_source_file_deleted() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let foo = proj.join("src/foo.rs");
    let bar = proj.join("src/bar.rs");
    fs::create_dir_all(foo.parent().unwrap()).unwrap();
    fs::write(&foo, "").unwrap();
    let bar_content = "references src/foo.rs";
    fs::write(&bar, bar_content).unwrap();
    wiki.ingest_file(&proj, &foo, "").unwrap();
    wiki.ingest_file(&proj, &bar, bar_content).unwrap();

    let stats_before = wiki.stats().unwrap();
    assert_eq!(stats_before.most_linked.len(), 1, "precondition: 1 inbound");

    fs::remove_file(&bar).unwrap();

    let stats_after = wiki.stats().unwrap();
    assert!(
        stats_after.most_linked.is_empty(),
        "foo's inbound must drop to 0 once bar's source is gone, got {:?}",
        stats_after.most_linked
    );
}

#[test]
fn wiki_link_scan_filters_to_entity_pages_via_caller() {
    // The filter is enforced by the caller (compute_inbound_links)
    // which only feeds entity-page (page,src) pairs into the helper.
    // This test exercises the helper's core contract: a mix of inputs
    // is scanned faithfully, and self_source excludes only the matching
    // source.
    let pairs = [
        ("entities/a.md", "src/a.rs"),
        ("entities/b.md", "src/b.rs"),
        ("entities/c.md", "src/c.rs"),
    ];
    let text = "uses src/a.rs and src/b.rs but not c";
    let matched = wiki_link_scan(text, pairs.iter().map(|(p, s)| (*p, *s)), Some("src/b.rs"));
    assert_eq!(
        matched,
        vec!["entities/a.md".to_string()],
        "b is self-excluded, c is not mentioned"
    );
}

#[test]
fn wiki_stats_most_linked_returns_top_five_by_count() {
    // Build 7 entity pages whose inbound counts will be [5,4,3,2,1,1,0]
    // by having them reference each other's source paths.
    //
    // We bake this by writing pages + sources directly and calling
    // stats(), bypassing ingest. Names are single-letter-prefixed so
    // tie-break (alphabetical path) is deterministic.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Seven pages a..g. Sources live under src/.
    let names = ["a", "b", "c", "d", "e", "f", "g"];
    let src_dir = proj.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    for n in &names {
        fs::write(src_dir.join(format!("{}.rs", n)), "").unwrap();
    }

    // Desired inbound counts:
    //   a:5 (mentioned by b,c,d,e,f)
    //   b:4 (mentioned by c,d,e,f)
    //   c:3 (d,e,f)
    //   d:2 (e,f)
    //   e:1 (f)
    //   f:1 (g)
    //   g:0
    let mentions: std::collections::HashMap<&str, Vec<&str>> = [
        ("a", vec![]),
        ("b", vec!["src/a.rs"]),
        ("c", vec!["src/a.rs", "src/b.rs"]),
        ("d", vec!["src/a.rs", "src/b.rs", "src/c.rs"]),
        ("e", vec!["src/a.rs", "src/b.rs", "src/c.rs", "src/d.rs"]),
        (
            "f",
            vec!["src/a.rs", "src/b.rs", "src/c.rs", "src/d.rs", "src/e.rs"],
        ),
        ("g", vec!["src/f.rs"]),
    ]
    .iter()
    .cloned()
    .collect();
    for n in &names {
        let content = mentions[n].join("\n");
        fs::write(src_dir.join(format!("{}.rs", n)), &content).unwrap();
        wiki.ingest_file(&proj, &src_dir.join(format!("{}.rs", n)), &content)
            .unwrap();
    }

    let stats = wiki.stats().unwrap();
    assert_eq!(
        stats.most_linked.len(),
        5,
        "top-5 cap; got {:?}",
        stats.most_linked
    );
    assert_eq!(
        stats.most_linked,
        vec![
            ("entities/src_a_rs.md".to_string(), 5),
            ("entities/src_b_rs.md".to_string(), 4),
            ("entities/src_c_rs.md".to_string(), 3),
            ("entities/src_d_rs.md".to_string(), 2),
            // tie between e(1) and f(1) breaks alphabetically
            ("entities/src_e_rs.md".to_string(), 1),
        ],
    );
}

#[test]
fn wiki_stats_most_linked_hides_zero_count() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let a = proj.join("src/a.rs");
    let b = proj.join("src/b.rs");
    fs::create_dir_all(a.parent().unwrap()).unwrap();
    fs::write(&a, "no mentions").unwrap();
    fs::write(&b, "no mentions either").unwrap();
    wiki.ingest_file(&proj, &a, "no mentions").unwrap();
    wiki.ingest_file(&proj, &b, "no mentions either").unwrap();

    let stats = wiki.stats().unwrap();
    assert!(
        stats.most_linked.is_empty(),
        "zero-count entries must be filtered out, got {:?}",
        stats.most_linked
    );
}

#[test]
fn wiki_stats_back_compat_with_pre_cycle_33_index() {
    // Older wikis' index.md files are literally the same format —
    // this cycle persists no new fields in markdown. Write a fully
    // pre-Cycle-33-shaped index.md and load it.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let legacy = "\
# Wiki Index

## Entities

- [src/foo.rs](entities/src_foo_rs.md) — File: src/foo.rs

## Concepts

## Summaries

## Synthesis
";
    fs::write(wiki.root().join("index.md"), legacy).unwrap();

    let idx = wiki.load_index().unwrap();
    assert_eq!(idx.entries.len(), 1);
    assert_eq!(idx.entries[0].path, "entities/src_foo_rs.md");

    // Stats must still work even though the page file and source
    // file don't exist — compute_inbound_links tolerates missing
    // reads (skip, not error).
    let stats = wiki.stats().unwrap();
    assert_eq!(stats.total_pages, 1);
    assert!(stats.most_linked.is_empty());
}

// ─── Cycle 49: concept auto-detection ──────────────────────────────────
//
// `sanitize_dep_for_path` maps a dep string to a filename-safe slug.
// `build_concept_pages` / `write_concept_pages` emit
// `concepts/dep-*.md` pages for dependencies shared by ≥
// `CONCEPT_DEP_MIN_OCCURRENCES` entity pages. Grouped imports are
// filtered out (file-specific, not a shared concept).

#[test]
fn sanitize_dep_for_path_collapses_non_alnum_and_rejects_groups() {
    assert_eq!(
        sanitize_dep_for_path("std::sync::Arc"),
        Some("std_sync_Arc".to_string()),
    );
    assert_eq!(sanitize_dep_for_path("foo::{bar, baz}"), None);
    assert_eq!(
        sanitize_dep_for_path("::leading"),
        Some("leading".to_string()),
        "leading underscores must be trimmed",
    );
    assert_eq!(sanitize_dep_for_path(""), None);
    assert_eq!(sanitize_dep_for_path("::::"), None);
    assert_eq!(
        sanitize_dep_for_path("tokio::sync::Mutex"),
        Some("tokio_sync_Mutex".to_string()),
    );
}

#[test]
fn build_concept_pages_returns_empty_below_threshold() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "a",
        "src/a.rs",
        Some(EntityKind::Function),
        None,
        vec!["tokio::sync::Mutex".to_string()],
    );
    install_rich_entity_page(
        &wiki,
        "entities/b.md",
        "b",
        "src/b.rs",
        Some(EntityKind::Function),
        None,
        vec!["tokio::sync::Mutex".to_string()],
    );
    let pages = wiki.build_concept_pages().unwrap();
    assert!(
        pages.is_empty(),
        "< {} consumers must not emit a page; got {:?}",
        CONCEPT_DEP_MIN_OCCURRENCES,
        pages,
    );
}

#[test]
fn build_concept_pages_detects_at_threshold() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["tokio::sync::Mutex".to_string()],
        );
    }
    let pages = wiki.build_concept_pages().unwrap();
    assert_eq!(pages.len(), 1, "exactly one concept page at threshold");
    let (rel, page) = &pages[0];
    assert_eq!(rel, "concepts/dep-tokio_sync_Mutex.md");
    assert_eq!(page.title, "Shared dependency: tokio::sync::Mutex");
    assert_eq!(page.page_type, PageType::Concept);
    assert_eq!(
        page.sources,
        vec![
            "entities/a.md".to_string(),
            "entities/b.md".to_string(),
            "entities/c.md".to_string(),
        ],
        "sources must be consumer page-paths sorted asc",
    );
}

#[test]
fn build_concept_pages_skips_grouped_imports() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["foo::{a, b}".to_string()],
        );
    }
    let pages = wiki.build_concept_pages().unwrap();
    assert!(
        pages.is_empty(),
        "grouped imports must never produce concept pages; got {:?}",
        pages,
    );
}

#[test]
fn build_concept_pages_sorts_freq_desc_then_path_asc() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // `std::sync::Arc` shared by 4 entities, `tokio::sync::Mutex` by 3.
    // Plus a second dep at the same count-3 tier to test alpha tiebreak.
    for (rel, title, src, deps) in [
        (
            "entities/a.md",
            "a",
            "src/a.rs",
            vec![
                "std::sync::Arc".to_string(),
                "tokio::sync::Mutex".to_string(),
                "anyhow::Result".to_string(),
            ],
        ),
        (
            "entities/b.md",
            "b",
            "src/b.rs",
            vec![
                "std::sync::Arc".to_string(),
                "tokio::sync::Mutex".to_string(),
                "anyhow::Result".to_string(),
            ],
        ),
        (
            "entities/c.md",
            "c",
            "src/c.rs",
            vec![
                "std::sync::Arc".to_string(),
                "tokio::sync::Mutex".to_string(),
                "anyhow::Result".to_string(),
            ],
        ),
        (
            "entities/d.md",
            "d",
            "src/d.rs",
            vec!["std::sync::Arc".to_string()],
        ),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            deps,
        );
    }
    let pages = wiki.build_concept_pages().unwrap();
    assert_eq!(pages.len(), 3, "three deps meet threshold");
    // First: highest count (4 — std::sync::Arc).
    assert_eq!(pages[0].0, "concepts/dep-std_sync_Arc.md");
    // Next two at count 3 — alphabetical by DEP string asc:
    //   "anyhow::Result" < "tokio::sync::Mutex".
    assert_eq!(pages[1].0, "concepts/dep-anyhow_Result.md");
    assert_eq!(pages[2].0, "concepts/dep-tokio_sync_Mutex.md");
}

#[test]
fn write_concept_pages_creates_new_pages_and_logs() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    let report = wiki.write_concept_pages().unwrap();
    assert_eq!(report.generated.len(), 1, "one page newly created");
    assert_eq!(report.generated[0], "concepts/dep-std_sync_Arc.md");
    assert!(report.refreshed.is_empty());
    assert!(
        wiki.root().join("concepts/dep-std_sync_Arc.md").is_file(),
        "concept page must be written to disk",
    );
    // Log must have one `concept` entry for the generated page.
    let log_text = fs::read_to_string(wiki.log().path()).unwrap();
    assert!(
        log_text
            .lines()
            .any(|l| l.contains("concept") && l.contains("concepts/dep-std_sync_Arc.md")),
        "log must carry a concept entry; got:\n{}",
        log_text,
    );
    // Index must carry a Concept entry for the page.
    let idx = wiki.load_index().unwrap();
    let concept_entries: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Concept)
        .collect();
    assert_eq!(concept_entries.len(), 1);
    assert_eq!(concept_entries[0].path, "concepts/dep-std_sync_Arc.md");
}

#[test]
fn write_concept_pages_is_idempotent_when_unchanged() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    let first = wiki.write_concept_pages().unwrap();
    assert_eq!(first.generated.len(), 1);

    // Clear the marker so we can prove the second run doesn't touch it.
    wiki.clear_summary_dirty().unwrap();
    assert!(!wiki.is_summary_dirty());

    let second = wiki.write_concept_pages().unwrap();
    assert!(
        second.generated.is_empty(),
        "second run must not regenerate unchanged pages; got {:?}",
        second.generated,
    );
    assert!(
        second.refreshed.is_empty(),
        "second run must not mark unchanged pages as refreshed; got {:?}",
        second.refreshed,
    );
    assert_eq!(
        second.detected_deps.len(),
        1,
        "detected_deps remains stable across idempotent runs",
    );
    assert!(
        !wiki.is_summary_dirty(),
        "idempotent run must not touch the summary-dirty marker",
    );
}

#[test]
fn write_concept_pages_marks_summary_dirty_on_change() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    // Clean slate — ensure no residual marker.
    wiki.clear_summary_dirty().unwrap();
    assert!(!wiki.is_summary_dirty());
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    wiki.write_concept_pages().unwrap();
    assert!(
        wiki.is_summary_dirty(),
        "creating a new concept page must mark the summary dirty",
    );
}

// Cycle 50: write-avoidance must strip timestamp lines before
// comparing — otherwise the fresh `chrono::Local::now()` stamped
// into both frontmatter and body on every build_concept_pages()
// call guarantees the equality check fires whenever two runs
// straddle a second boundary.

#[test]
fn concept_body_sans_timestamp_strips_both_timestamp_forms() {
    let input = "---\n\
                     title: Shared dependency: std::sync::Arc\n\
                     type: concept\n\
                     sources:\n  - entities/a.md\n\
                     last_updated: 2026-04-18 09:00:00\n\
                     ---\n\n\
                     # Shared dependency: std::sync::Arc\n\n\
                     Last updated: 2026-04-18 09:00:00\n\n\
                     body text\n";
    let stripped = concept_body_sans_timestamp(input);
    assert!(
        !stripped.contains("last_updated:"),
        "frontmatter last_updated must be stripped; got:\n{}",
        stripped,
    );
    assert!(
        !stripped.contains("Last updated:"),
        "body Last updated line must be stripped; got:\n{}",
        stripped,
    );
    assert!(
        stripped.contains("# Shared dependency: std::sync::Arc"),
        "heading must be preserved; got:\n{}",
        stripped,
    );
    assert!(
        stripped.contains("body text"),
        "body text must be preserved; got:\n{}",
        stripped,
    );
}

#[test]
fn concept_body_sans_timestamp_tolerates_indented_key() {
    let input = "    last_updated: 2026-04-18 09:00:00\n\
                     \t\tLast updated: 2026-04-18 09:00:00\n\
                     keep me\n";
    let stripped = concept_body_sans_timestamp(input);
    assert!(
        !stripped.contains("last_updated:"),
        "indented last_updated must be stripped; got:\n{}",
        stripped,
    );
    assert!(
        !stripped.contains("Last updated:"),
        "tab-indented Last updated must be stripped; got:\n{}",
        stripped,
    );
    assert!(
        stripped.contains("keep me"),
        "non-matching line must survive; got:\n{}",
        stripped,
    );
}

#[test]
fn concept_body_sans_timestamp_preserves_non_matching_lines_verbatim() {
    let input = "alpha\nbeta\ngamma\n";
    let stripped = concept_body_sans_timestamp(input);
    // lines().collect().join("\n") drops the trailing newline —
    // that's fine for comparison purposes since both sides get
    // the same treatment.
    assert_eq!(stripped, "alpha\nbeta\ngamma");
}

#[test]
fn write_concept_pages_skips_write_when_only_timestamp_differs() {
    // Flake-repro: manually install a concept page whose body matches
    // what build_concept_pages() would emit, but with a stale
    // timestamp. The second write_concept_pages() call would otherwise
    // see different bytes and spuriously refresh.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    // First run creates the concept page.
    let first = wiki.write_concept_pages().unwrap();
    assert_eq!(first.generated.len(), 1);
    let concept_rel = &first.generated[0];

    // Overwrite the on-disk page with a stale timestamp (same body,
    // different second). This simulates a second-boundary straddle
    // without waiting.
    let abs = wiki.root().join(concept_rel);
    let current = fs::read_to_string(&abs).unwrap();
    let stale = current
        .replace("last_updated: ", "last_updated: STALE_")
        .replace("Last updated: ", "Last updated: STALE_");
    fs::write(&abs, &stale).unwrap();

    wiki.clear_summary_dirty().unwrap();
    assert!(!wiki.is_summary_dirty());

    let second = wiki.write_concept_pages().unwrap();
    assert!(
        second.refreshed.is_empty(),
        "timestamp-only difference must not count as refreshed; got {:?}",
        second.refreshed,
    );
    assert!(
        second.generated.is_empty(),
        "existing page must not be reported as generated; got {:?}",
        second.generated,
    );
    assert!(
        !wiki.is_summary_dirty(),
        "timestamp-only rerun must not mark summary dirty",
    );

    // And the on-disk bytes must still carry the stale marker —
    // proves write_page was NOT called.
    let after = fs::read_to_string(&abs).unwrap();
    assert!(
        after.contains("STALE_"),
        "on-disk page must be untouched; got:\n{}",
        after,
    );
}

#[test]
fn write_concept_pages_still_writes_when_consumer_added() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    let first = wiki.write_concept_pages().unwrap();
    assert_eq!(first.generated.len(), 1);
    wiki.clear_summary_dirty().unwrap();

    // Add a fourth consumer — concept page's `sources` must grow.
    install_rich_entity_page(
        &wiki,
        "entities/d.md",
        "d",
        "src/d.rs",
        Some(EntityKind::Function),
        None,
        vec!["std::sync::Arc".to_string()],
    );

    let second = wiki.write_concept_pages().unwrap();
    assert_eq!(
        second.refreshed.len(),
        1,
        "adding a consumer must refresh the concept page; got {:?}",
        second.refreshed,
    );
    assert!(
        second.generated.is_empty(),
        "existing page must not be re-generated; got {:?}",
        second.generated,
    );
    assert!(
        wiki.is_summary_dirty(),
        "consumer-set change must mark summary dirty",
    );
}

#[test]
fn write_concept_pages_still_writes_when_consumer_removed() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
        ("entities/d.md", "d", "src/d.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    let first = wiki.write_concept_pages().unwrap();
    assert_eq!(first.generated.len(), 1);
    wiki.clear_summary_dirty().unwrap();

    // Remove one consumer — still ≥ CONCEPT_DEP_MIN_OCCURRENCES,
    // so the page stays but with a smaller `sources` set.
    let mut idx = wiki.load_index().unwrap();
    idx.entries.retain(|e| e.path != "entities/d.md");
    wiki.save_index(&idx).unwrap();
    fs::remove_file(wiki.root().join("entities/d.md")).unwrap();

    let second = wiki.write_concept_pages().unwrap();
    assert_eq!(
        second.refreshed.len(),
        1,
        "removing a consumer must refresh the concept page; got {:?}",
        second.refreshed,
    );
    assert!(
        wiki.is_summary_dirty(),
        "consumer-set change must mark summary dirty",
    );
}

#[test]
fn write_concept_pages_on_noop_rerun_preserves_on_disk_timestamp() {
    // The first run stamps some `now_1` into the page. A second
    // run with an identical consumer set must NOT overwrite that
    // timestamp with `now_2`, even if they differ — because the
    // normalized comparison treats them as equal.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    for (rel, title, src) in [
        ("entities/a.md", "a", "src/a.rs"),
        ("entities/b.md", "b", "src/b.rs"),
        ("entities/c.md", "c", "src/c.rs"),
    ] {
        install_rich_entity_page(
            &wiki,
            rel,
            title,
            src,
            Some(EntityKind::Function),
            None,
            vec!["std::sync::Arc".to_string()],
        );
    }
    wiki.write_concept_pages().unwrap();
    let concept_path = wiki.root().join("concepts/dep-std_sync_Arc.md");
    let bytes_before = fs::read(&concept_path).unwrap();

    // Force a distinguishable on-disk state: overwrite the page with
    // a sentinel timestamp. If the second run spuriously rewrites,
    // the sentinel will be replaced with a fresh chrono::Local::now().
    let current = String::from_utf8(bytes_before.clone()).unwrap();
    let sentinel = current
        .replace(
            "last_updated: ",
            "last_updated: 2000-01-01 00:00:00_SENTINEL_",
        )
        .replace(
            "Last updated: ",
            "Last updated: 2000-01-01 00:00:00_SENTINEL_",
        );
    fs::write(&concept_path, &sentinel).unwrap();

    let _ = wiki.write_concept_pages().unwrap();

    let after = fs::read_to_string(&concept_path).unwrap();
    assert!(
        after.contains("SENTINEL_"),
        "noop rerun must preserve the on-disk timestamp bytes; got:\n{}",
        after,
    );
}

// ── C85 IndexEntry.last_updated cache tests ─────────────────────────

#[test]
fn index_entry_struct_has_last_updated_field() {
    // Shape sanity — the struct must expose an `Option<String>` field
    // named `last_updated`. A default-value construction proves both
    // that the field exists and that `None` is a legal state.
    let entry = IndexEntry {
        title: "t".to_string(),
        path: "entities/t.md".to_string(),
        one_liner: "x".to_string(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    };
    assert_eq!(entry.last_updated, None);
    let with_ts = IndexEntry {
        last_updated: Some("2026-04-19 00:00:00".to_string()),
        ..entry.clone()
    };
    assert_eq!(with_ts.last_updated.as_deref(), Some("2026-04-19 00:00:00"),);
}

#[test]
fn index_roundtrip_with_last_updated_suffix() {
    // `to_markdown` + `parse` must preserve `last_updated: Some(ts)`
    // across the HTML-comment suffix encoding. Mix one timestamped
    // row with one legacy row to exercise both branches in one pass.
    let idx = WikiIndex {
        entries: vec![
            IndexEntry {
                title: "Wiki".to_string(),
                path: "entities/wiki.md".to_string(),
                one_liner: "Persistent knowledge.".to_string(),
                category: PageType::Entity,
                last_updated: Some("2026-04-19 12:00:00".to_string()),
                outcome: None,
            },
            IndexEntry {
                title: "Compaction".to_string(),
                path: "concepts/compaction.md".to_string(),
                one_liner: "Three-stage context trimming.".to_string(),
                category: PageType::Concept,
                last_updated: None,
                outcome: None,
            },
        ],
    };
    let md = idx.to_markdown();
    assert!(
        md.contains("<!--updated:2026-04-19 12:00:00-->"),
        "HTML-comment suffix must be emitted: {}",
        md,
    );
    let parsed = WikiIndex::parse(&md);
    assert_eq!(parsed, idx, "roundtrip must preserve last_updated");
}

#[test]
fn index_parses_legacy_entries_as_last_updated_none() {
    // Pre-C85 `index.md` files have no `<!--updated:…-->` suffix;
    // they must parse cleanly with `last_updated = None` so upgrades
    // degrade gracefully to the file-read fallback path.
    let legacy = "## Entities\n\n- [Foo](entities/foo.md) — foo summary\n\
            \n## Concepts\n\n- [Bar](concepts/bar.md) — bar summary\n";
    let idx = WikiIndex::parse(legacy);
    assert_eq!(idx.entries.len(), 2);
    assert!(
        idx.entries.iter().all(|e| e.last_updated.is_none()),
        "legacy entries must parse as last_updated=None; got {:?}",
        idx.entries,
    );
    assert_eq!(idx.entries[0].one_liner, "foo summary");
    assert_eq!(idx.entries[1].one_liner, "bar summary");
}

#[test]
fn ingest_file_populates_last_updated_in_index_entry() {
    // `ingest_file` writes an entity page AND updates the index;
    // the new index entry must carry `last_updated = Some(ts)`
    // matching the page frontmatter so future planner_brief calls
    // skip the file read.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("cached.rs");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "pub fn cached() {}").unwrap();

    wiki.ingest_file(&proj, &file, "pub fn cached() {}")
        .unwrap();

    let idx = wiki.load_index().unwrap();
    let entry = idx
        .entries
        .iter()
        .find(|e| e.path == "entities/src_cached_rs.md")
        .expect("entry exists");
    let ts = entry
        .last_updated
        .as_ref()
        .expect("C85: ingest must cache last_updated");
    let page_raw = fs::read_to_string(wiki.root().join(&entry.path)).unwrap();
    let page = WikiPage::parse(&page_raw).expect("page parses");
    assert_eq!(
        ts, &page.last_updated,
        "index cache must match page frontmatter",
    );
}

#[test]
fn cycle_synthesis_populates_last_updated_in_index_entry() {
    // Symmetric check for the synthesis write path. `recent_cycles`
    // reads from the cache; if the write path fails to populate it,
    // every planner cycle would pay an extra file read per row.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(7, "demo-chain", &[], None)
        .unwrap()
        .expect("auto-ingest on in tests");

    let idx = wiki.load_index().unwrap();
    let entry = idx
        .entries
        .iter()
        .find(|e| e.path == page_rel)
        .expect("synthesis entry registered");
    let ts = entry
        .last_updated
        .as_ref()
        .expect("C85: cycle_synthesis must cache last_updated");
    let page_raw = fs::read_to_string(wiki.root().join(&entry.path)).unwrap();
    let page = WikiPage::parse(&page_raw).expect("page parses");
    assert_eq!(
        ts, &page.last_updated,
        "synthesis index cache must match page frontmatter",
    );
}

#[test]
fn recent_cycles_reads_last_updated_from_index_without_file_io() {
    // Write the synthesis page so the index gets a cached timestamp,
    // then corrupt the page file. The cache-hit path must surface
    // the timestamp WITHOUT touching the broken file.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(11, "demo", &[], None)
        .unwrap()
        .expect("auto-ingest on in tests");

    let cached_ts = wiki
        .load_index()
        .unwrap()
        .entries
        .iter()
        .find(|e| e.path == page_rel)
        .and_then(|e| e.last_updated.clone())
        .expect("cache populated");

    // Corrupt the page so any file-read fallback would yield None.
    fs::write(
        proj.join(".dm/wiki").join(&page_rel),
        "no frontmatter here\n",
    )
    .unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(brief.recent_cycles.len(), 1);
    assert_eq!(
        brief.recent_cycles[0].last_updated,
        Some(cached_ts),
        "cache-hit path must bypass file read for recent_cycles",
    );
}

#[test]
fn fresh_pages_reads_last_updated_from_index_without_file_io() {
    // Symmetric cache-hit check for `fresh_pages`. Ingest populates
    // the index cache; corrupting the page file must not drop the
    // row from fresh_pages.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("fresh_cache.rs");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "pub fn f() {}").unwrap();
    wiki.ingest_file(&proj, &file, "pub fn f() {}").unwrap();

    let page_rel = "entities/src_fresh_cache_rs.md".to_string();
    let cached_ts = wiki
        .load_index()
        .unwrap()
        .entries
        .iter()
        .find(|e| e.path == page_rel)
        .and_then(|e| e.last_updated.clone())
        .expect("cache populated");

    // Corrupt the page body — cache must still surface the entry.
    fs::write(
        proj.join(".dm/wiki").join(&page_rel),
        "no frontmatter here\n",
    )
    .unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let hit = brief
        .fresh_pages
        .iter()
        .find(|(p, _)| p == &page_rel)
        .expect("row survives corruption via cache");
    assert_eq!(
        hit.1, cached_ts,
        "fresh_pages must use cached last_updated when present",
    );
}

#[test]
fn fresh_pages_falls_back_to_file_for_legacy_index_entries() {
    // Migration safety net: a legacy index (no cache) must still
    // produce a valid fresh_pages row by reading the page file.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("legacy.rs");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "pub fn l() {}").unwrap();
    wiki.ingest_file(&proj, &file, "pub fn l() {}").unwrap();

    let page_rel = "entities/src_legacy_rs.md".to_string();

    // Simulate a legacy pre-C85 index — clear the cache so the
    // file-read fallback is the one under test.
    let mut idx = wiki.load_index().unwrap();
    for e in &mut idx.entries {
        e.last_updated = None;
    }
    wiki.save_index(&idx).unwrap();

    // Read the page timestamp that the fallback path should recover.
    let page_raw = fs::read_to_string(proj.join(".dm/wiki").join(&page_rel)).unwrap();
    let expected_ts = WikiPage::parse(&page_raw)
        .expect("page parses")
        .last_updated;

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let hit = brief
        .fresh_pages
        .iter()
        .find(|(p, _)| p == &page_rel)
        .expect("legacy entry must still surface via file fallback");
    assert_eq!(
        hit.1, expected_ts,
        "fallback must recover timestamp from page frontmatter",
    );
}

// ── C86 fresh_pages_snippet tests ────────────────────────────────────

#[test]
fn fresh_pages_snippet_ranks_entity_concept_by_last_updated_desc() {
    // Seed a mix of categories + timestamps. Only Entity + Concept
    // rows should land in the snippet, and only in newest-first
    // order — the session-start prompt wants "what's live right now"
    // first, not a random sample.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.extend(vec![
        IndexEntry {
            title: "older entity".to_string(),
            path: "entities/old.md".to_string(),
            one_liner: "old ent".to_string(),
            category: PageType::Entity,
            last_updated: Some("2026-04-18 10:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "newest entity".to_string(),
            path: "entities/new.md".to_string(),
            one_liner: "new ent".to_string(),
            category: PageType::Entity,
            last_updated: Some("2026-04-19 14:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "mid concept".to_string(),
            path: "concepts/mid.md".to_string(),
            one_liner: "mid con".to_string(),
            category: PageType::Concept,
            last_updated: Some("2026-04-19 08:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "filtered synthesis".to_string(),
            path: "synthesis/cycle-01.md".to_string(),
            one_liner: "filtered".to_string(),
            category: PageType::Synthesis,
            last_updated: Some("2026-04-19 23:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "filtered summary".to_string(),
            path: "summaries/project.md".to_string(),
            one_liner: "filtered".to_string(),
            category: PageType::Summary,
            last_updated: Some("2026-04-19 22:00:00".to_string()),
            outcome: None,
        },
    ]);
    wiki.save_index(&idx).unwrap();

    let snip = wiki.fresh_pages_snippet(4096).expect("snippet must render");
    let new_pos = snip.find("newest entity").expect("newest present");
    let mid_pos = snip.find("mid concept").expect("mid present");
    let old_pos = snip.find("older entity").expect("old present");
    assert!(
        new_pos < mid_pos && mid_pos < old_pos,
        "order must be newest → oldest: snippet=\n{}",
        snip,
    );
    assert!(
        !snip.contains("filtered synthesis"),
        "synthesis must be filtered: {}",
        snip,
    );
    assert!(
        !snip.contains("filtered summary"),
        "summary must be filtered: {}",
        snip,
    );
}

#[test]
fn fresh_pages_snippet_stable_on_missing_last_updated() {
    // `None` timestamps sort last; ties within the None bucket break
    // on path asc for deterministic output across runs.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.extend(vec![
        IndexEntry {
            title: "has-ts".to_string(),
            path: "entities/a.md".to_string(),
            one_liner: "timestamped".to_string(),
            category: PageType::Entity,
            last_updated: Some("2026-04-19 08:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "z-legacy".to_string(),
            path: "entities/z.md".to_string(),
            one_liner: "legacy z".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        },
        IndexEntry {
            title: "b-legacy".to_string(),
            path: "entities/b.md".to_string(),
            one_liner: "legacy b".to_string(),
            category: PageType::Entity,
            last_updated: None,
            outcome: None,
        },
    ]);
    wiki.save_index(&idx).unwrap();

    let snip = wiki.fresh_pages_snippet(4096).expect("renders");
    let ts_pos = snip.find("has-ts").unwrap();
    let b_pos = snip.find("b-legacy").unwrap();
    let z_pos = snip.find("z-legacy").unwrap();
    assert!(ts_pos < b_pos, "timestamped row must rank ahead of None");
    assert!(b_pos < z_pos, "None bucket must tie-break on path asc");
}

#[test]
fn fresh_pages_snippet_respects_byte_budget() {
    // 50 synthetic rows with wide `one_liner` text. Budget 256 bytes
    // must force truncation at a line boundary — partial title lines
    // would corrupt the prompt block.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    for i in 0..50 {
        idx.entries.push(IndexEntry {
            title: format!("title-{:02}", i),
            path: format!("entities/e{:02}.md", i),
            one_liner: "a".repeat(40),
            category: PageType::Entity,
            last_updated: Some(format!("2026-04-19 {:02}:00:00", i % 24)),
            outcome: None,
        });
    }
    wiki.save_index(&idx).unwrap();

    let snip = wiki.fresh_pages_snippet(256).expect("renders");
    assert!(snip.len() <= 256, "budget violated: {} > 256", snip.len(),);
    // Must end on `\n` — each bullet we commit is terminated, so a
    // budget-aware break lands cleanly at a line boundary.
    assert!(
        snip.ends_with('\n'),
        "truncation must land at a line boundary: tail=\n{}",
        snip.lines().last().unwrap_or(""),
    );
}

#[test]
fn fresh_pages_snippet_returns_none_when_no_entity_or_concept() {
    // A wiki with only Synthesis + Summary entries has nothing to
    // rank — the snippet must be suppressed entirely rather than
    // emit an empty header.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.extend(vec![
        IndexEntry {
            title: "cycle-01".to_string(),
            path: "synthesis/cycle-01.md".to_string(),
            one_liner: "s".to_string(),
            category: PageType::Synthesis,
            last_updated: Some("2026-04-19 12:00:00".to_string()),
            outcome: None,
        },
        IndexEntry {
            title: "Project".to_string(),
            path: "summaries/project.md".to_string(),
            one_liner: "p".to_string(),
            category: PageType::Summary,
            last_updated: Some("2026-04-19 11:00:00".to_string()),
            outcome: None,
        },
    ]);
    wiki.save_index(&idx).unwrap();

    assert_eq!(
        wiki.fresh_pages_snippet(4096),
        None,
        "synthesis/summary-only wiki must suppress the block",
    );
}

#[test]
fn fresh_pages_snippet_returns_none_when_empty_wiki() {
    // A freshly-initialized wiki with no entries must suppress the
    // block — no `<wiki_fresh>` header leaks into the prompt for
    // projects that haven't started incubating yet.
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    assert_eq!(wiki.fresh_pages_snippet(4096), None);
}

#[test]
fn fresh_pages_snippet_zero_io_on_cache_hit() {
    // Post-C85 indexes cache `last_updated` per-entry. The snippet
    // path must NOT re-read page files — we prove this by corrupting
    // the underlying page bytes after the index is saved and
    // asserting the snippet still renders.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    reset_ingest_cache_for_tests();
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let file = proj.join("src").join("cached_snip.rs");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "pub fn c() {}").unwrap();
    wiki.ingest_file(&proj, &file, "pub fn c() {}").unwrap();

    // Capture the cached-rich snippet rendering.
    let before = wiki
        .fresh_pages_snippet(4096)
        .expect("snippet renders from cache");

    // Corrupt every entity page so any file fallback would fail.
    let page_rel = "entities/src_cached_snip_rs.md";
    fs::write(
        proj.join(".dm/wiki").join(page_rel),
        "no frontmatter here\n",
    )
    .unwrap();

    let after = wiki
        .fresh_pages_snippet(4096)
        .expect("snippet still renders — no file I/O on cache hit");
    assert_eq!(
        before, after,
        "cache-hit path must not read the corrupted page file",
    );
    assert!(
        after.contains("src/cached_snip.rs"),
        "entity title must appear: {}",
        after,
    );
}

// ── C88 chain incubation-loop observability tests ───────────────────
//
// These exercise the public-API surface the integration test file
// `tests/chain_incubation_loop.rs` depends on. Each test asserts a
// narrow invariant of the observability chain that runs
//   write_cycle_synthesis → planner_brief → PlannerBrief::render
// and the co-surface `fresh_pages_snippet` that <wiki_fresh> uses.

#[test]
fn planner_brief_render_preserves_synthesis_page_path_verbatim() {
    // Page paths are embedded in the rendered brief prefixed with
    // `.dm/wiki/`. Anything else (HTML-escape, slash-normalization,
    // quote wrapping) would break file-open hints the planner gives
    // to human operators. Pin the byte-for-byte shape.
    let page_path = "synthesis/cycle-42-continuous-dev-20260419-120000-000000000.md";
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 42,
            chain: "continuous-dev".to_string(),
            page_path: page_path.to_string(),
            last_updated: None,
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(4096).expect("non-empty brief renders");
    let expected = format!(".dm/wiki/{}", page_path);
    assert!(
        rendered.contains(&expected),
        "path must appear verbatim under .dm/wiki/: expected substring {:?}; got: {}",
        expected,
        rendered,
    );
}

#[test]
fn planner_brief_render_orders_recent_cycles_newest_first() {
    // `planner_brief` sorts by cycle desc, then page_path asc — so
    // writing cycles 1, 2, 3 in that order must surface cycle 3
    // before cycle 2 before cycle 1 in the render. Guards against
    // the "incubation timeline goes backwards" regression.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    for cycle in 1..=3 {
        wiki.write_cycle_synthesis(cycle, "loop", &[], None)
            .unwrap();
    }

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let rendered = brief.render(8192).expect("non-empty brief renders");
    let p3 = rendered.find("Cycle 3 (loop)").expect("cycle 3 present");
    let p2 = rendered.find("Cycle 2 (loop)").expect("cycle 2 present");
    let p1 = rendered.find("Cycle 1 (loop)").expect("cycle 1 present");
    assert!(
        p3 < p2 && p2 < p1,
        "order must be 3 → 2 → 1: got positions {}/{}/{}; render: {}",
        p3,
        p2,
        p1,
        rendered,
    );
}

#[test]
fn planner_brief_render_respects_budget_truncation_on_cycles_block() {
    // Render-time budget is enforced on the cycles block too, not
    // just hot_paths. Construct 20 RecentCycle entries directly
    // (bypassing `planner_brief`'s built-in cap) + tight budget →
    // truncation marker lands on a line boundary, no partial bullet.
    let recents: Vec<RecentCycle> = (0..20)
        .map(|i| RecentCycle {
            cycle: i,
            chain: "loop".to_string(),
            page_path: format!("synthesis/cycle-{:02}-loop-20260419-120000.md", i),
            last_updated: None,
            outcome: None,
        })
        .collect();
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: recents,
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(400).expect("non-empty brief renders");
    assert!(
        rendered.len() <= 400,
        "budget violated: {} > 400; body: {:?}",
        rendered.len(),
        rendered,
    );
    assert!(
        rendered.ends_with("[...truncated]"),
        "oversized render must end with truncation marker: {:?}",
        rendered,
    );
    for line in rendered.lines() {
        if line == "[...truncated]" || line.is_empty() {
            continue;
        }
        assert!(
            line.starts_with('#') || line.starts_with("- "),
            "truncation must respect line boundaries; got: {:?}",
            line,
        );
    }
}

#[test]
fn write_cycle_synthesis_twice_for_same_cycle_produces_two_index_entries() {
    // Non-idempotent contract: re-running cycle N writes a second
    // snapshot with a distinct slug_ts. Both must land in the index
    // so both are discoverable by `planner_brief` and `/wiki search`.
    // Complements `write_cycle_synthesis_back_to_back_no_overwrite`,
    // which only asserts distinct page paths on disk.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let a = wiki
        .write_cycle_synthesis(1, "chain", &[], None)
        .unwrap()
        .unwrap();
    let b = wiki
        .write_cycle_synthesis(1, "chain", &[], None)
        .unwrap()
        .unwrap();
    assert_ne!(a, b, "back-to-back must produce distinct paths");

    let idx = wiki.load_index().unwrap();
    let synth: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .collect();
    assert_eq!(
        synth.len(),
        2,
        "both cycle-1 snapshots must appear in index; got: {:?}",
        synth,
    );
    let paths: std::collections::HashSet<&str> = synth.iter().map(|e| e.path.as_str()).collect();
    assert!(paths.contains(a.as_str()), "first path missing from index");
    assert!(paths.contains(b.as_str()), "second path missing from index");
}

#[test]
fn fresh_pages_snippet_unaffected_by_synthesis_page_count() {
    // `<wiki_fresh>` filters to Entity|Concept — synthesis page churn
    // from incubation cycles must not perturb the block the model
    // sees at session start. Seed identical Entity entries, then
    // write N synthesis pages and assert the fresh snippet is
    // byte-identical before/after.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "entity alpha".to_string(),
        path: "entities/alpha.md".to_string(),
        one_liner: "alpha ent".to_string(),
        category: PageType::Entity,
        last_updated: Some("2026-04-19 10:00:00".to_string()),
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "concept beta".to_string(),
        path: "concepts/beta.md".to_string(),
        one_liner: "beta con".to_string(),
        category: PageType::Concept,
        last_updated: Some("2026-04-19 09:00:00".to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).unwrap();

    let before = wiki.fresh_pages_snippet(4096).expect("renders before");
    for cycle in 1..=5 {
        wiki.write_cycle_synthesis(cycle, "loop", &[], None)
            .unwrap();
    }
    let after = wiki.fresh_pages_snippet(4096).expect("renders after");
    assert_eq!(
        before, after,
        "synthesis-page churn must not perturb the fresh snippet",
    );
}

#[test]
fn planner_brief_render_survives_corrupt_synthesis_page_bodies() {
    // The brief pipeline must never panic on a malformed synthesis
    // page. The index carries `last_updated` (post-C85 cache), so
    // a scrambled page body is observed only by consumers that
    // re-read the file — `render()` must degrade to "page omitted
    // or unchanged" without blowing up.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    let page_rel = wiki
        .write_cycle_synthesis(1, "loop", &[], None)
        .unwrap()
        .unwrap();
    let full = wiki.root().join(&page_rel);
    fs::write(&full, "not a wiki page at all\n\x00\x01binary junk").unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let rendered = brief.render(8192);
    assert!(
        rendered.is_some(),
        "brief with at least one recent cycle must still render",
    );
}

#[test]
fn wiki_index_remains_parseable_after_three_cycle_synthesis_writes() {
    // Three writes — each appends one Synthesis entry to index.md.
    // The index must still round-trip through parse afterwards and
    // every synthesis-categorized entry must have non-empty path +
    // title (guards against empty-slug filename generation and
    // index-serialization drift).
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    for cycle in 1..=3 {
        wiki.write_cycle_synthesis(cycle, "loop", &[], None)
            .unwrap();
    }

    let idx = wiki.load_index().expect("index must parse");
    let synth: Vec<&IndexEntry> = idx
        .entries
        .iter()
        .filter(|e| e.category == PageType::Synthesis)
        .collect();
    assert!(
        synth.len() >= 3,
        "expected ≥ 3 synthesis entries; got {}",
        synth.len(),
    );
    for e in &synth {
        assert!(!e.path.is_empty(), "synthesis entry path must be non-empty");
        assert!(
            !e.title.is_empty(),
            "synthesis entry title must be non-empty",
        );
    }
}

#[test]
fn planner_brief_recent_cycles_capped_at_planner_brief_recent_cycles_constant() {
    // `planner_brief` truncates `recent_cycles` at
    // `PLANNER_BRIEF_RECENT_CYCLES` regardless of how many synthesis
    // pages exist — the cap is a wiki-level constant, not the
    // `max_findings` argument (which bounds drifting_pages). Writing
    // 10 cycles must still produce at most the constant's worth of
    // RecentCycle entries.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();
    for cycle in 1..=10 {
        wiki.write_cycle_synthesis(cycle, "loop", &[], None)
            .unwrap();
    }

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert!(
        brief.recent_cycles.len() <= PLANNER_BRIEF_RECENT_CYCLES,
        "recent_cycles must be capped at PLANNER_BRIEF_RECENT_CYCLES ({}); got {}",
        PLANNER_BRIEF_RECENT_CYCLES,
        brief.recent_cycles.len(),
    );
}

// ── C90: outcome end-to-end round trip ──────────────────────────────

#[test]
fn write_cycle_synthesis_with_none_outcome_emits_no_frontmatter_line() {
    // Byte-identity with pre-C90 synthesis pages: `None` omits the
    // `outcome:` frontmatter line entirely. Guards against the
    // serializer emitting `outcome: \n` or `outcome: null` for the
    // default case, which would invalidate every pre-C90 parse path.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(1, "loop", &[], None)
        .unwrap()
        .unwrap();
    let text = std::fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    assert!(
        !text.contains("outcome:"),
        "None outcome must not emit any frontmatter line; got: {}",
        text,
    );
}

#[test]
fn write_cycle_synthesis_with_some_outcome_round_trips_through_parse() {
    // `Some("green")` → frontmatter `outcome: green` → `WikiPage::parse`
    // → `WikiPage.outcome == Some("green")`. End-to-end disk round trip
    // through the public API; no direct frontmatter string matching.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(1, "loop", &[], Some("green"))
        .unwrap()
        .unwrap();
    let text = std::fs::read_to_string(wiki.root().join(&page_rel)).unwrap();
    assert!(
        text.contains("outcome: green\n"),
        "Some(\"green\") must emit `outcome: green` in frontmatter; got: {}",
        text,
    );
    let parsed = WikiPage::parse(&text).expect("parse synthesis page");
    assert_eq!(parsed.outcome.as_deref(), Some("green"));
}

#[test]
fn planner_brief_recent_cycle_outcome_none_when_synthesis_has_no_outcome() {
    // Composer must leave `RecentCycle.outcome` as `None` when the
    // underlying synthesis page has no `outcome:` frontmatter line.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "loop", &[], None)
        .unwrap()
        .unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(brief.recent_cycles.len(), 1);
    assert_eq!(brief.recent_cycles[0].outcome, None);
}

#[test]
fn planner_brief_recent_cycle_outcome_populated_when_synthesis_has_outcome() {
    // Composer must surface `Some("yellow")` on the RecentCycle when
    // the synthesis page's frontmatter carries `outcome: yellow`.
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "loop", &[], Some("yellow"))
        .unwrap()
        .unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    assert_eq!(brief.recent_cycles.len(), 1);
    assert_eq!(brief.recent_cycles[0].outcome.as_deref(), Some("yellow"));
}

#[test]
fn planner_brief_render_omits_outcome_badge_when_none() {
    // Render must not emit the `[...]` suffix when outcome is None.
    // Negative check guards against `format!(" [{}]", opt)` regressions
    // that would render `[None]` or a stray pair of brackets.
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "loop".to_string(),
            page_path: "synthesis/cycle-01-loop-ts.md".to_string(),
            last_updated: None,
            outcome: None,
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        !rendered.contains("[green]")
            && !rendered.contains("[yellow]")
            && !rendered.contains("[red]")
            && !rendered.contains("[None]"),
        "None outcome must not emit any bracketed badge; got: {}",
        rendered,
    );
}

#[test]
fn planner_brief_render_includes_outcome_badge_when_some() {
    // Positive check: the `[{outcome}]` suffix lands on the cycle line.
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: vec![RecentCycle {
            cycle: 1,
            chain: "loop".to_string(),
            page_path: "synthesis/cycle-01-loop-ts.md".to_string(),
            last_updated: None,
            outcome: Some("green".to_string()),
        }],
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("[green]"),
        "Some(\"green\") outcome must surface as `[green]` badge; got: {}",
        rendered,
    );
    // And the badge must sit on the same line as the cycle bullet.
    let cycle_line = rendered
        .lines()
        .find(|l| l.contains("Cycle 1 (loop)"))
        .expect("cycle line present");
    assert!(
        cycle_line.contains("[green]"),
        "badge must appear on the cycle bullet line: {}",
        cycle_line,
    );
}

#[test]
fn planner_brief_render_outcome_badge_respects_budget_truncation() {
    // Budget enforcement must include the outcome badge — the render
    // truncates at line boundaries, so a tight budget must still
    // produce `<= budget` output and end with the truncation marker.
    let recents: Vec<RecentCycle> = (0..20)
        .map(|i| RecentCycle {
            cycle: i,
            chain: "loop".to_string(),
            page_path: format!("synthesis/cycle-{:02}-loop-20260419-120000.md", i),
            last_updated: None,
            outcome: Some("green".to_string()),
        })
        .collect();
    let brief = PlannerBrief {
        hot_paths: Vec::new(),
        hot_modules: Vec::new(),
        drifting_pages: Vec::new(),
        lint_counts: Vec::new(),
        recent_cycles: recents,
        fresh_pages: Vec::new(),
    };
    let rendered = brief.render(400).expect("non-empty brief renders");
    assert!(
        rendered.len() <= 400,
        "budget violated even with outcome badges: {} > 400; body: {:?}",
        rendered.len(),
        rendered,
    );
    assert!(
        rendered.ends_with("[...truncated]"),
        "oversized render must end with truncation marker: {:?}",
        rendered,
    );
}

#[test]
fn wiki_page_outcome_field_serialize_parse_roundtrip_preserves_arbitrary_strings() {
    // The `outcome` field is an opaque string to the wiki layer — no
    // enum validation, no auto-detection. Arbitrary values must round
    // trip through `to_markdown` → `WikiPage::parse` without mutation.
    for v in ["green", "yellow", "red", "flaky-tests", "n/a"] {
        let page = WikiPage {
            title: "synthesis-candidate".to_string(),
            page_type: PageType::Synthesis,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: "2026-04-19 12:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: Some(v.to_string()),
            scope: vec![],
            body: "# candidate\n".to_string(),
        };
        let md = page.to_markdown();
        let parsed = WikiPage::parse(&md).expect("round-trip parse");
        assert_eq!(
            parsed.outcome.as_deref(),
            Some(v),
            "outcome value {:?} must round trip; got: {:?}",
            v,
            parsed.outcome,
        );
    }
}

// ── C92: IndexEntry.outcome cache (mirrors C85 last_updated cache) ──

/// `IndexEntry`'s outcome cache serializes as a trailing
/// `<!--outcome:{v}-->` HTML comment after the `<!--updated:...-->`
/// suffix — separate comment (not extending the `updated:` grammar)
/// so legacy parsers and future fields don't collide.
#[test]
fn index_entry_serializes_outcome_as_html_comment_suffix() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Cycle 1".to_string(),
            path: "synthesis/cycle-01.md".to_string(),
            one_liner: "Cycle 1 of loop".to_string(),
            category: PageType::Synthesis,
            last_updated: Some("2026-04-19 12:00:00".to_string()),
            outcome: Some("green".to_string()),
        }],
    };
    let md = idx.to_markdown();
    assert!(
        md.contains("<!--updated:2026-04-19 12:00:00--> <!--outcome:green-->"),
        "outcome must follow updated as a separate HTML comment: {}",
        md,
    );
}

/// Parser accepts the outcome suffix when it sits after an updated
/// suffix on the same line and round-trips the value into
/// `IndexEntry.outcome`.
#[test]
fn index_entry_parses_outcome_suffix_alongside_updated_suffix() {
    let text = "\
# Wiki Index

## Synthesis

- [Cycle 1](synthesis/cycle-01.md) — Cycle 1 of loop <!--updated:2026-04-19 12:00:00--> <!--outcome:yellow-->
";
    let idx = WikiIndex::parse(text);
    assert_eq!(idx.entries.len(), 1);
    let e = &idx.entries[0];
    assert_eq!(e.last_updated.as_deref(), Some("2026-04-19 12:00:00"));
    assert_eq!(e.outcome.as_deref(), Some("yellow"));
    assert_eq!(e.one_liner, "Cycle 1 of loop");
}

/// Parser must handle outcome before updated as well — the peeling
/// loop strips trailing known suffixes in either order, so an
/// index file written by a future tool (or hand-edited) with the
/// comments reversed still yields the same `IndexEntry`.
#[test]
fn index_entry_parses_outcome_suffix_in_either_order() {
    let forward = "\
# Wiki Index

## Synthesis

- [C](synthesis/c.md) — C of loop <!--updated:2026-04-19 12:00:00--> <!--outcome:red-->
";
    let reversed = "\
# Wiki Index

## Synthesis

- [C](synthesis/c.md) — C of loop <!--outcome:red--> <!--updated:2026-04-19 12:00:00-->
";
    let a = WikiIndex::parse(forward).entries.pop().unwrap();
    let b = WikiIndex::parse(reversed).entries.pop().unwrap();
    assert_eq!(a.last_updated, b.last_updated);
    assert_eq!(a.outcome, b.outcome);
    assert_eq!(a.one_liner, b.one_liner);
}

/// Legacy indexes (pre-C92, with or without the C85 updated suffix)
/// parse cleanly — outcome stays `None`, consumers fall back to a
/// page read, self-healing on next ingest.
#[test]
fn index_entry_parses_legacy_line_without_outcome_suffix() {
    let text = "\
# Wiki Index

## Entities

- [Module](entities/module.md) — A module <!--updated:2026-04-19 09:00:00-->
- [Older](entities/older.md) — An older entry
";
    let idx = WikiIndex::parse(text);
    assert_eq!(idx.entries.len(), 2);
    let with_ts = idx
        .entries
        .iter()
        .find(|e| e.path == "entities/module.md")
        .unwrap();
    assert_eq!(with_ts.last_updated.as_deref(), Some("2026-04-19 09:00:00"));
    assert_eq!(with_ts.outcome, None);
    let legacy = idx
        .entries
        .iter()
        .find(|e| e.path == "entities/older.md")
        .unwrap();
    assert_eq!(legacy.last_updated, None);
    assert_eq!(legacy.outcome, None);
}

/// Round trip through `to_markdown` → `parse` preserves
/// `outcome: None` byte-identically — no spurious `<!--outcome:-->`
/// suffix when the field is None.
#[test]
fn index_entry_round_trip_preserves_outcome_none() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Entry".to_string(),
            path: "entities/entry.md".to_string(),
            one_liner: "A test entry".to_string(),
            category: PageType::Entity,
            last_updated: Some("2026-04-19 10:00:00".to_string()),
            outcome: None,
        }],
    };
    let md = idx.to_markdown();
    assert!(
        !md.contains("<!--outcome:"),
        "None outcome must not emit any outcome comment: {}",
        md,
    );
    let parsed = WikiIndex::parse(&md);
    assert_eq!(parsed, idx);
}

/// Round trip through `to_markdown` → `parse` preserves
/// `outcome: Some(...)` exactly.
#[test]
fn index_entry_round_trip_preserves_outcome_some() {
    let idx = WikiIndex {
        entries: vec![IndexEntry {
            title: "Cycle 42".to_string(),
            path: "synthesis/cycle-42.md".to_string(),
            one_liner: "Cycle 42 of loop".to_string(),
            category: PageType::Synthesis,
            last_updated: Some("2026-04-19 12:00:00".to_string()),
            outcome: Some("green".to_string()),
        }],
    };
    let md = idx.to_markdown();
    let parsed = WikiIndex::parse(&md);
    assert_eq!(parsed, idx);
}

/// `write_cycle_synthesis(..., Some("green"))` must land the outcome
/// in the `IndexEntry`'s cached field — so future `planner_brief`
/// invocations can skip the page read.
#[test]
fn wiki_ingest_populates_outcome_in_index_entry_from_cycle_synthesis() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "loop", &[], Some("green"))
        .unwrap()
        .unwrap();

    let idx = wiki.load_index().unwrap();
    let synth = idx
        .entries
        .iter()
        .find(|e| e.category == PageType::Synthesis)
        .expect("synthesis entry");
    assert_eq!(synth.outcome.as_deref(), Some("green"));
}

/// Elision proof: after an ingest that caches both `last_updated`
/// and `outcome`, deleting the synthesis page from disk must NOT
/// affect the rendered brief's outcome badge — the composer read
/// only the cached index fields, never the page file.
#[test]
fn planner_brief_skips_page_read_when_both_last_updated_and_outcome_cached() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(7, "loop", &[], Some("green"))
        .unwrap()
        .unwrap();

    // Delete the page file while leaving the cached index entry intact.
    std::fs::remove_file(wiki.root().join(&page_rel)).expect("remove page");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let rc = brief
        .recent_cycles
        .iter()
        .find(|rc| rc.cycle == 7)
        .expect("cycle 7 present");
    assert_eq!(
        rc.outcome.as_deref(),
        Some("green"),
        "cache hit must surface outcome without reading deleted page",
    );
    assert!(
        rc.last_updated.is_some(),
        "cached last_updated must still surface",
    );
}

/// Fallback proof: when the index entry's outcome is None (legacy
/// pre-C92 index), the composer re-reads the page and recovers the
/// outcome from the page's frontmatter. Self-heals on next ingest.
#[test]
fn planner_brief_falls_back_to_page_read_when_outcome_missing_from_cache() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(8, "loop", &[], Some("yellow"))
        .unwrap()
        .unwrap();

    // Simulate a legacy index by rewriting it with outcome=None on
    // every entry. The page file on disk still carries the outcome
    // in its frontmatter, so the composer must read it.
    let mut idx = wiki.load_index().unwrap();
    for e in &mut idx.entries {
        e.outcome = None;
    }
    wiki.save_index(&idx).unwrap();

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let rc = brief
        .recent_cycles
        .iter()
        .find(|rc| rc.cycle == 8)
        .expect("cycle 8 present");
    assert_eq!(
        rc.outcome.as_deref(),
        Some("yellow"),
        "legacy cache miss must fall back to page read; page_rel={}",
        page_rel,
    );
}

/// End-to-end from cached index to rendered brief: when outcome is
/// cached in the `IndexEntry`, the rendered brief includes the badge
/// without reading the page file.
#[test]
fn planner_brief_renders_outcome_badge_from_index_cache_without_reading_page() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(9, "loop", &[], Some("green"))
        .unwrap()
        .unwrap();
    std::fs::remove_file(wiki.root().join(&page_rel)).expect("remove page");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .unwrap();
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("[green]"),
        "cached outcome badge must appear in the rendered brief: {}",
        rendered,
    );
}

// ── C93: project summary surfaces recent cycle outcomes ──────────────

/// The `## Recent cycles` section in `build_project_summary` surfaces
/// cycle number, outcome, and chain name for each recent cycle so a
/// human reading `summaries/project.md` sees the incubation trail
/// without opening individual synthesis pages.
#[test]
fn build_project_summary_surfaces_recent_cycle_outcomes() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_cycle_synthesis(2, "demo", &[], Some("yellow"))
        .unwrap()
        .unwrap();
    wiki.write_cycle_synthesis(3, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();

    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("## Recent cycles"),
        "summary body must include Recent cycles section: {}",
        page.body,
    );
    for cycle in ["cycle 1", "cycle 2", "cycle 3"] {
        assert!(
            page.body.contains(cycle),
            "summary must surface {}: {}",
            cycle,
            page.body,
        );
    }
    assert!(
        page.body.matches("green").count() >= 2,
        "two green cycles must both render: {}",
        page.body,
    );
    assert!(
        page.body.contains("yellow"),
        "yellow cycle must render: {}",
        page.body,
    );
    assert!(
        page.body.contains("`demo`"),
        "chain name must appear backticked: {}",
        page.body,
    );
}

/// Recent cycles render newest-first in the summary body — the
/// helper's `b.cycle.cmp(&a.cycle)` ordering must be preserved by
/// the rendering loop.
#[test]
fn build_project_summary_recent_cycles_newest_first() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Write cycles out of order so cycle number — not insertion
    // order — is what drives the rendered sort.
    wiki.write_cycle_synthesis(3, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_cycle_synthesis(7, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_cycle_synthesis(5, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();

    let page = wiki.build_project_summary().unwrap();
    let pos7 = page.body.find("cycle 7").expect("cycle 7 present");
    let pos5 = page.body.find("cycle 5").expect("cycle 5 present");
    let pos3 = page.body.find("cycle 3").expect("cycle 3 present");
    assert!(
        pos7 < pos5 && pos5 < pos3,
        "cycles must render newest-first (7, 5, 3); got positions 7={} 5={} 3={}\n{}",
        pos7,
        pos5,
        pos3,
        page.body,
    );
}

/// The helper's `limit` argument is threaded from
/// `PROJECT_SUMMARY_RECENT_CYCLES` — writing more synthesis entries
/// than the cap must still surface exactly `cap` rows.
#[test]
fn build_project_summary_recent_cycles_capped_at_constant() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let over = PROJECT_SUMMARY_RECENT_CYCLES + 2;
    for i in 1..=over {
        wiki.write_cycle_synthesis(i, "demo", &[], Some("green"))
            .unwrap()
            .unwrap();
    }

    let page = wiki.build_project_summary().unwrap();
    // Rows are prefixed with `- cycle ` in the renderer; count them.
    let row_count = page.body.matches("- cycle ").count();
    assert_eq!(
        row_count, PROJECT_SUMMARY_RECENT_CYCLES,
        "recent cycles must cap at {}; got {} rows\nbody: {}",
        PROJECT_SUMMARY_RECENT_CYCLES, row_count, page.body,
    );
}

/// Empty wiki — no synthesis pages — renders the Recent cycles
/// header with a placeholder. Pairs with the empty-wiki regression
/// test at :12345; the section header must always appear.
#[test]
fn build_project_summary_recent_cycles_empty_wiki_uses_placeholder() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("## Recent cycles"),
        "Recent cycles header must appear even when empty: {}",
        page.body,
    );
    assert!(
        page.body
            .contains("*(no synthesis pages yet — chains haven't written cycle outcomes)*"),
        "empty-wiki body must render the Recent cycles placeholder: {}",
        page.body,
    );
}

/// Synthesis page with `outcome: None` renders as the literal
/// `- cycle N — chain` (no parens) — the match arm for `None` must
/// not inject an empty `()` segment.
#[test]
fn build_project_summary_recent_cycles_none_outcome_renders_without_parens() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(11, "demo", &[], None)
        .unwrap()
        .unwrap();

    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("- cycle 11 — `demo`"),
        "None outcome must render without parens: {}",
        page.body,
    );
    assert!(
        !page.body.contains("cycle 11 ()"),
        "must not render empty parens for None outcome: {}",
        page.body,
    );
    assert!(
        !page
            .body
            .contains("*(no synthesis pages yet — chains haven't written cycle outcomes)*",),
        "placeholder must not appear when a synthesis page exists: {}",
        page.body,
    );
}

/// Zero-I/O cache proof: after `write_cycle_synthesis` populates the
/// `IndexEntry`'s `outcome` cache (C92), deleting the page file from
/// disk must still surface the outcome in the summary body — the
/// `collect_recent_cycles` helper reads from the cache.
#[test]
fn build_project_summary_recent_cycles_uses_index_cache_when_available() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    let page_rel = wiki
        .write_cycle_synthesis(21, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    // Delete the page — cache must carry the outcome forward.
    std::fs::remove_file(wiki.root().join(&page_rel)).expect("remove page");

    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("cycle 21 (green)"),
        "cached outcome must render even after page deletion: {}",
        page.body,
    );
}

/// Legacy fallback: when the `IndexEntry` outcome cache is missing
/// (pre-C92 indexes), the helper reads the page and recovers the
/// outcome. Simulated by zeroing `outcome` on the index entry.
#[test]
fn build_project_summary_recent_cycles_fallback_reads_page_when_outcome_cache_missing() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(31, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();

    // Simulate a legacy pre-C92 index: rewrite the index with the
    // outcome field zeroed so only a page read can recover it.
    let mut idx = wiki.load_index().unwrap();
    for e in &mut idx.entries {
        if e.category == PageType::Synthesis {
            e.outcome = None;
        }
    }
    wiki.save_index(&idx).unwrap();

    let page = wiki.build_project_summary().unwrap();
    assert!(
        page.body.contains("cycle 31 (green)"),
        "legacy-cache-miss fallback must read the page: {}",
        page.body,
    );
}

/// `ProjectSummaryReport.recent_cycles` is populated by
/// `write_project_summary` — callers (TUI `/wiki build`, REST API)
/// can surface the same signal without re-parsing the page body.
#[test]
fn project_summary_report_includes_recent_cycles_vec() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(41, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_cycle_synthesis(42, "demo", &[], Some("yellow"))
        .unwrap()
        .unwrap();

    let report = wiki.write_project_summary().unwrap();
    assert_eq!(
        report.recent_cycles.len(),
        2,
        "report.recent_cycles must match synthesis entry count: {:?}",
        report.recent_cycles,
    );
    let outcomes: Vec<Option<String>> = report
        .recent_cycles
        .iter()
        .map(|rc| rc.outcome.clone())
        .collect();
    // Newest-first: cycle 42 (yellow) then cycle 41 (green).
    assert_eq!(
        outcomes,
        vec![Some("yellow".to_string()), Some("green".to_string())],
    );
    assert_eq!(report.recent_cycles[0].cycle, 42);
    assert_eq!(report.recent_cycles[1].cycle, 41);
}

// ── C95: section order so Recent cycles survives snippet truncation ──

/// Locks the new section order: Purpose → Recent cycles →
/// Architecture → Momentum. Recent cycles moved from body-bottom
/// (pre-C95) to position 2 so it survives `project_summary_snippet`
/// truncation under any budget that keeps Purpose.
#[test]
fn build_project_summary_body_orders_purpose_recent_architecture_momentum() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    wiki.write_cycle_synthesis(1, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    let page = wiki.build_project_summary().unwrap();

    let p_purpose = page.body.find("## Purpose").expect("Purpose header");
    let p_recent = page
        .body
        .find("## Recent cycles")
        .expect("Recent cycles header");
    let p_arch = page.body.find("## Architecture").expect("Architecture");
    let p_momentum = page.body.find("## Momentum").expect("Momentum");

    assert!(
        p_purpose < p_recent,
        "Recent cycles must follow Purpose: purpose={} recent={}\n{}",
        p_purpose,
        p_recent,
        page.body,
    );
    assert!(
        p_recent < p_arch,
        "Architecture must follow Recent cycles: recent={} arch={}\n{}",
        p_recent,
        p_arch,
        page.body,
    );
    assert!(
        p_arch < p_momentum,
        "Momentum must follow Architecture: arch={} momentum={}\n{}",
        p_arch,
        p_momentum,
        page.body,
    );
}

/// Budget sized to cut inside Momentum — Recent cycles (now at
/// position 2) must survive. Proves the invariant: reordering is
/// sufficient; no snippet-side code change is required.
///
/// Inflates sections that sit AFTER Recent cycles (Architecture via
/// long dep names, Momentum via log entries) while keeping Purpose
/// compact — Purpose bloat would precede Recent cycles and push it
/// out of the snippet, which is the opposite of what we're proving.
#[test]
fn project_summary_snippet_preserves_recent_cycles_when_momentum_truncated() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Minimal Purpose (1 short entity), heavy Architecture (long dep
    // names, 10 distinct deps to fill the cap), heavy Momentum (50
    // ingest log entries across 6 hot paths).
    for i in 0..6 {
        install_rich_entity_page(
            &wiki,
            &format!("entities/e{}.md", i),
            &format!("e{}", i),
            &format!("src/some_longer_path/module_{}.rs", i),
            Some(EntityKind::Function),
            Some("p"),
            (0..6)
                .map(|j| {
                    format!(
                        "really_long_dependency_crate_name_for_inflation_{}_{}",
                        i, j
                    )
                })
                .collect(),
        );
    }
    let log = wiki.log();
    for i in 0..50 {
        log.append(
            "ingest",
            &format!("src/some_longer_path/module_{}.rs", i % 6),
        )
        .unwrap();
    }
    wiki.write_cycle_synthesis(77, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_project_summary().unwrap();

    // Budget comfortably larger than Purpose+Recent prefix, smaller
    // than full body. Recent cycles must survive.
    let snippet = wiki
        .project_summary_snippet(500)
        .expect("snippet under budget");
    assert!(
        snippet.contains("## Recent cycles"),
        "Recent cycles header must survive truncation: {}",
        snippet,
    );
    assert!(
        snippet.contains("cycle 77"),
        "cycle 77 row must survive truncation: {}",
        snippet,
    );
    assert!(
        snippet.contains("[...truncated]"),
        "truncation marker must be present: {}",
        snippet,
    );
}

/// Even tighter budget cutting mid-Architecture — Recent cycles
/// still survives because it precedes Architecture.
#[test]
fn project_summary_snippet_preserves_recent_cycles_when_architecture_truncated() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    // Many entities with long deps lists to bloat Architecture.
    for i in 0..10 {
        install_rich_entity_page(
            &wiki,
            &format!("entities/e{}.md", i),
            &format!("entity {}", i),
            &format!("src/e{}.rs", i),
            Some(EntityKind::Function),
            Some("purpose line"),
            (0..8)
                .map(|j| format!("long_dep_name_{}_{}", i, j))
                .collect(),
        );
    }
    wiki.write_cycle_synthesis(88, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_project_summary().unwrap();

    let snippet = wiki
        .project_summary_snippet(600)
        .expect("snippet under budget");
    assert!(
        snippet.contains("## Recent cycles"),
        "Recent cycles must survive mid-Architecture truncation: {}",
        snippet,
    );
    assert!(
        snippet.contains("cycle 88"),
        "cycle 88 must appear in pre-Architecture slice: {}",
        snippet,
    );
}

/// Very tight budget (Purpose + Recent only fit) — snippet still
/// carries both. Architecture/Momentum land past the cut.
///
/// Same fixture shape as the "momentum truncated" test: compact
/// Purpose (3 entities, 1-char purposes) keeps Purpose+Recent under
/// the 400-char cut, while heavy Architecture (long dep names) and
/// Momentum (many log entries) bloat the body past the budget.
#[test]
fn project_summary_snippet_under_tight_budget_still_emits_purpose_and_recent_cycles() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    for i in 0..3 {
        install_rich_entity_page(
            &wiki,
            &format!("entities/e{}.md", i),
            &format!("e{}", i),
            &format!("src/some_longer_path/module_{}.rs", i),
            Some(EntityKind::Function),
            Some("p"),
            (0..6)
                .map(|j| {
                    format!(
                        "really_long_dependency_crate_name_for_inflation_{}_{}",
                        i, j
                    )
                })
                .collect(),
        );
    }
    let log = wiki.log();
    for i in 0..20 {
        log.append(
            "ingest",
            &format!("src/some_longer_path/module_{}.rs", i % 3),
        )
        .unwrap();
    }
    wiki.write_cycle_synthesis(99, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_project_summary().unwrap();

    let snippet = wiki
        .project_summary_snippet(400)
        .expect("snippet under budget");
    assert!(
        snippet.contains("## Purpose"),
        "Purpose header must appear: {}",
        snippet,
    );
    assert!(
        snippet.contains("## Recent cycles"),
        "Recent cycles header must appear: {}",
        snippet,
    );
    assert!(
        snippet.contains("cycle 99 (green)"),
        "cycle 99 row must appear under tight budget: {}",
        snippet,
    );
    assert!(
        snippet.contains("[...truncated]"),
        "tight budget must leave truncation marker: {}",
        snippet,
    );
}

/// `[...truncated]` marker lands AFTER Recent cycles — locks that
/// Recent cycles is NOT the section that gets sliced by truncation.
///
/// Purpose kept compact (4 entities, 1-char purposes) so Recent
/// cycles lands before the 500-char cut; Architecture + Momentum
/// carry the bulk of the body so truncation fires past Recent cycles.
#[test]
fn project_summary_snippet_emits_truncation_marker_after_recent_cycles_block() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    for i in 0..4 {
        install_rich_entity_page(
            &wiki,
            &format!("entities/e{}.md", i),
            &format!("e{}", i),
            &format!("src/some_longer_path/module_{}.rs", i),
            Some(EntityKind::Function),
            Some("p"),
            (0..6)
                .map(|j| {
                    format!(
                        "really_long_dependency_crate_name_for_inflation_{}_{}",
                        i, j
                    )
                })
                .collect(),
        );
    }
    let log = wiki.log();
    for i in 0..30 {
        log.append(
            "ingest",
            &format!("src/some_longer_path/module_{}.rs", i % 4),
        )
        .unwrap();
    }
    wiki.write_cycle_synthesis(101, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_project_summary().unwrap();

    let snippet = wiki
        .project_summary_snippet(500)
        .expect("snippet under budget");
    let p_recent = snippet
        .find("## Recent cycles")
        .expect("Recent cycles present");
    let p_marker = snippet.find("[...truncated]").expect("marker present");
    assert!(
        p_recent < p_marker,
        "truncation marker must follow Recent cycles: recent={} marker={}\n{}",
        p_recent,
        p_marker,
        snippet,
    );
}

/// Empty-wiki placeholders must respect the same section order as
/// populated bodies — reordering is structural, not content-gated.
#[test]
fn build_project_summary_empty_wiki_preserves_new_section_order() {
    let tmp = TempDir::new().unwrap();
    let wiki = Wiki::open(tmp.path()).unwrap();
    let page = wiki.build_project_summary().unwrap();

    let p_purpose = page.body.find("## Purpose").expect("Purpose");
    let p_recent = page.body.find("## Recent cycles").expect("Recent cycles");
    let p_arch = page.body.find("## Architecture").expect("Architecture");
    let p_momentum = page.body.find("## Momentum").expect("Momentum");
    assert!(
        p_purpose < p_recent && p_recent < p_arch && p_arch < p_momentum,
        "empty-wiki placeholders must honor new order: {}",
        page.body,
    );
}

/// Section order is a structural invariant: two runs with
/// different synthesis counts must place Recent cycles at the same
/// relative position (after Purpose, before Architecture).
#[test]
fn project_summary_snippet_section_order_independent_of_content_size() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let run = |synthesis_count: usize| -> String {
        std::env::remove_var("DM_WIKI_AUTO_INGEST");
        let tmp = TempDir::new().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = Wiki::open(&proj).unwrap();
        install_rich_entity_page(
            &wiki,
            "entities/a.md",
            "auth",
            "src/auth.rs",
            Some(EntityKind::Function),
            Some("purpose"),
            vec![],
        );
        for i in 1..=synthesis_count {
            wiki.write_cycle_synthesis(i, "demo", &[], Some("green"))
                .unwrap()
                .unwrap();
        }
        wiki.build_project_summary().unwrap().body
    };

    for count in [3usize, 5usize] {
        let body = run(count);
        let p_purpose = body.find("## Purpose").expect("Purpose");
        let p_recent = body.find("## Recent cycles").expect("Recent cycles");
        let p_arch = body.find("## Architecture").expect("Architecture");
        assert!(
            p_purpose < p_recent && p_recent < p_arch,
            "synthesis_count={} must not perturb section order; body:\n{}",
            count,
            body,
        );
    }
}

/// Generous budget — full body fits, no truncation marker. Regression
/// guard that the reorder didn't accidentally break the happy path
/// where the snippet returns the whole body verbatim.
#[test]
fn project_summary_snippet_full_body_fits_under_generous_budget() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("DM_WIKI_AUTO_INGEST");
    let tmp = TempDir::new().unwrap();
    let proj = tmp.path().canonicalize().unwrap();
    let wiki = Wiki::open(&proj).unwrap();

    install_rich_entity_page(
        &wiki,
        "entities/a.md",
        "auth",
        "src/auth.rs",
        Some(EntityKind::Function),
        Some("purpose"),
        vec!["serde".to_string()],
    );
    wiki.write_cycle_synthesis(42, "demo", &[], Some("green"))
        .unwrap()
        .unwrap();
    wiki.write_project_summary().unwrap();

    let snippet = wiki
        .project_summary_snippet(16384)
        .expect("snippet under budget");
    for section in [
        "## Purpose",
        "## Recent cycles",
        "## Architecture",
        "## Momentum",
    ] {
        assert!(
            snippet.contains(section),
            "{} must appear in generous-budget snippet: {}",
            section,
            snippet,
        );
    }
    assert!(
        !snippet.contains("[...truncated]"),
        "generous budget must not truncate: {}",
        snippet,
    );
    let p_purpose = snippet.find("## Purpose").unwrap();
    let p_recent = snippet.find("## Recent cycles").unwrap();
    let p_arch = snippet.find("## Architecture").unwrap();
    let p_momentum = snippet.find("## Momentum").unwrap();
    assert!(
        p_purpose < p_recent && p_recent < p_arch && p_arch < p_momentum,
        "full-body snippet must carry new section order: {}",
        snippet,
    );
}
