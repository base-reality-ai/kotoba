//! v0.5 paradigm-gap acceptance sweep — exercises Tiers 1-6 of the
//! kotoba-backlog closure run end-to-end against a single host fixture.
//!
//! Mirrors `tests/host_paradigm_v04_e2e.rs` (Run-32 acceptance) and pins the
//! six gap closures shipped in this initiative:
//!
//! - **Tier 1** — `dm spawn` rewrites `examples/*/Cargo.toml` to point at the
//!   new host package name.
//! - **Tier 2** — `dm spawn --canonical <host-source>` refuses to clone from
//!   a host project's tree (mode = host) before the clone runs.
//! - **Tier 3** — `Wiki::register_host_page` round-trips `WikiPage.extras`
//!   custom frontmatter both in-memory and on disk.
//! - **Tier 4** — `Wiki::write_compact_synthesis` lands `layer: host` when
//!   the project's `.dm/identity.toml` says `mode = "host"`.
//! - **Tier 5** — `run_conversation_capture_with_client_in_config_dir`
//!   accepts a non-Ollama `LlmClient` impl and drives the loop to completion.
//! - **Tier 6** — `apply_session_instruction_update` swaps the active
//!   persona's system instruction mid-session, with the next sync rewriting
//!   the first `role=system` message and preserving prior transcript context.
//!
//! Final assert: the operator-global `~/.dm/` is byte-identical to its
//! pre-test snapshot — the no-leakage invariant Run-31 introduced and every
//! subsequent paradigm sweep has had to honor.
//!
//! Single-binary multi-thread runner because the test sets `HOME` and `cwd`
//! (process-global state), shells out to `dm spawn`, and then calls async
//! capture functions that depend on the cwd-based identity routing.

use async_trait::async_trait;
use dark_matter::conversation::{
    apply_session_instruction_update, instruction_update_from_tool_result, system_msg, user_msg,
    SessionInstructionUpdate,
};
use dark_matter::llm::LlmClient;
use dark_matter::ollama::types::{ChatResponse, ChunkMessage, ToolDefinition};
use dark_matter::session::Session;
use dark_matter::tools::registry::ToolRegistry;
use dark_matter::wiki::{Layer, PageType, Wiki, WikiPage};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Mock `LlmClient` that returns a scripted response without touching any
/// HTTP backend. Used by Tier 5 to prove the trait is genuinely the seam.
struct MockClient {
    response_text: String,
    chat_calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl LlmClient for MockClient {
    fn model(&self) -> &str {
        "mock/sweep-model"
    }

    async fn model_context_limit(&self, _model: &str) -> usize {
        16384
    }

    async fn chat(
        &self,
        _messages: &[Value],
        _tools: &[ToolDefinition],
    ) -> anyhow::Result<ChatResponse> {
        self.chat_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(ChatResponse {
            message: ChunkMessage {
                role: Some("assistant".to_string()),
                content: self.response_text.clone(),
                thinking: None,
                tool_calls: vec![],
            },
            prompt_tokens: 3,
            completion_tokens: 5,
            duration_ms: 0,
        })
    }
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

/// One consolidated test. Marked `multi_thread` because `dm spawn` shells out
/// to `git` and `cargo`; a single-thread runtime can't supervise the child.
///
/// Ignored in kotoba: this is canonical's v0.5 acceptance test, written
/// under the assumption that the workspace root is kernel-mode. Kotoba is
/// host-mode, so the Tier 2 spawn step ("kernel canonical spawn must
/// succeed") correctly refuses with Gap #6's new error. The test is fully
/// valid in canonical context — kept here so future canonical syncs land
/// without churn, and so kotoba remains aware of the canonical invariants.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "canonical-side acceptance test; refused by Gap #6 from kotoba's host-mode root"]
async fn v0_5_paradigm_sweep_all_tiers_no_leakage() {
    let home = TempDir::new().expect("home tempdir");
    let parent = TempDir::new().expect("parent tempdir for spawn target");
    let project_name = "v05-sweep";

    // Pre-populate ~/.dm with a sentinel so we can byte-compare the
    // snapshot after the sweep runs every routing path.
    let global_dm = home.path().join(".dm");
    std::fs::create_dir_all(&global_dm).expect("create global .dm");
    std::fs::write(global_dm.join("sentinel.txt"), "untouched").expect("write sentinel");
    let global_before = snapshot_tree(&global_dm);

    let prior_home = std::env::var_os("HOME");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_var("HOME", home.path());

    // ── Tier 2 (success path) — spawn from kernel canonical succeeds ──
    let workspace_root = std::env::current_dir().expect("workspace cwd");
    let canonical = format!("file://{}", workspace_root.display());
    let dm_bin = env!("CARGO_BIN_EXE_dm");

    let status = Command::new(dm_bin)
        .arg("spawn")
        .arg(project_name)
        .arg("--canonical")
        .arg(&canonical)
        .current_dir(parent.path())
        .status()
        .expect("dm spawn from kernel canonical should run");
    assert!(status.success(), "Tier 2: kernel canonical spawn must succeed");
    let project = parent.path().join(project_name);
    assert!(
        project.join(".dm/identity.toml").is_file(),
        "Tier 2: spawned project must have identity.toml"
    );

    // ── Tier 1 — examples/*/Cargo.toml rewrite landed ──
    let example_cargo = project.join("examples/host-skeleton/Cargo.toml");
    if example_cargo.exists() {
        let body = std::fs::read_to_string(&example_cargo).expect("read example cargo");
        assert!(
            body.contains(&format!("{} = {{ path", project_name)),
            "Tier 1: example Cargo.toml must reference host package, got:\n{body}",
        );
        assert!(
            !body.contains("dark-matter = { path"),
            "Tier 1: stale dark-matter path dep must not survive spawn rewrite"
        );
    }

    // ── Tier 2 (refusal path) — spawn from a host source tree fails ──
    let host_src = parent.path().join("host-fixture");
    let dm_dir = host_src.join(".dm");
    std::fs::create_dir_all(&dm_dir).expect("create host fixture .dm");
    std::fs::write(
        dm_dir.join("identity.toml"),
        "mode = \"host\"\nhost_project = \"some-host\"\n",
    )
    .expect("write host fixture identity");

    let refused = Command::new(dm_bin)
        .arg("spawn")
        .arg("attempted-clone")
        .arg("--canonical")
        .arg(format!("file://{}", host_src.display()))
        .current_dir(parent.path())
        .output()
        .expect("dm spawn refusal-path must run");
    assert!(
        !refused.status.success(),
        "Tier 2: spawn from host-mode source must fail"
    );
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(
        stderr.contains("refusing to spawn from a host project's source tree"),
        "Tier 2: stderr must carry refusal hint, got:\n{stderr}",
    );
    assert!(
        !parent.path().join("attempted-clone").exists(),
        "Tier 2: refusal must occur before any target directory is created"
    );

    // From here on, work inside the spawned host project so Wiki/identity-
    // aware behaviors see `mode = host`.
    std::env::set_current_dir(&project).expect("chdir to spawned project");

    // ── Tier 3 — register_host_page round-trips extras ──
    let wiki = Wiki::open(&project).expect("open spawned wiki");
    let mut extras = BTreeMap::new();
    extras.insert("sessions_count".to_string(), "5".to_string());
    extras.insert("persona_id".to_string(), "Hiro".to_string());
    let host_page = WikiPage {
        title: "Hiro".to_string(),
        page_type: PageType::Entity,
        layer: Layer::Host,
        sources: vec![],
        last_updated: "2026-04-29 09:00:00".to_string(),
        entity_kind: None,
        purpose: None,
        key_exports: vec![],
        dependencies: vec![],
        outcome: None,
        scope: vec![],
        body: "# Hiro\n\nHost-defined persona page for the v0.5 sweep.\n".to_string(),
        extras: extras.clone(),
    };
    wiki.register_host_page("entities/Personas/Hiro.md", &host_page, "persona: Hiro")
        .expect("register_host_page");
    let read_back = wiki
        .read_page("entities/Personas/Hiro.md")
        .expect("read_page");
    assert_eq!(
        read_back.extras, extras,
        "Tier 3: extras must round-trip through register_host_page → read_page"
    );
    let on_disk = project.join(".dm/wiki/entities/Personas/Hiro.md");
    let raw = std::fs::read_to_string(&on_disk).expect("read raw page");
    assert!(
        raw.contains("\nsessions_count: 5\n") && raw.contains("\npersona_id: Hiro\n"),
        "Tier 3: raw page must contain extras lines, got:\n{raw}",
    );

    // ── Tier 4 — compact synthesis carries layer: host in host-mode ──
    let compact_rel = wiki
        .write_compact_synthesis(
            "v0.5 sweep compaction marker",
            6,
            &["src/domain.rs".to_string()],
        )
        .expect("write_compact_synthesis")
        .expect("compact synthesis page should be written when ingest is enabled");
    let compact_text = std::fs::read_to_string(wiki.root().join(&compact_rel))
        .expect("read compact synthesis");
    assert!(
        compact_text.contains("\nlayer: host\n"),
        "Tier 4: host-mode compact synthesis must carry layer: host, got:\n{compact_text}",
    );
    let compact_page = WikiPage::parse(&compact_text).expect("parse compact synthesis");
    assert_eq!(compact_page.layer, Layer::Host);

    // ── Tier 5 — run capture against a non-Ollama LlmClient ──
    let registry = ToolRegistry::new();
    let mock = MockClient {
        response_text: "v0.5 sweep mock reply".to_string(),
        chat_calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let capture = dark_matter::conversation::run_conversation_capture_with_client_in_config_dir(
        "ping",
        "v05-sweep",
        &mock,
        &registry,
        project.join(".dm").as_path(),
    )
    .await
    .expect("Tier 5 capture against mock client must succeed");
    assert_eq!(
        capture.text, "v0.5 sweep mock reply",
        "Tier 5: mock LlmClient response must surface through capture"
    );
    assert_eq!(
        mock.chat_calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Tier 5: a no-tool-calls response should consume exactly one chat round"
    );

    // ── Tier 6 — mid-session persona swap rewrites the system message ──
    let mut session = Session::new(
        project.to_string_lossy().to_string(),
        "mock/sweep-model".to_string(),
    );
    let mut messages = vec![
        system_msg("Yuki original system prompt"),
        user_msg("turn 1 user prompt"),
    ];
    assert_eq!(
        messages[0]["content"], "Yuki original system prompt",
        "Tier 6 baseline: turn-1 system prompt is the original"
    );

    let persona_payload = "Persona 'Hiro' loaded\n\nYou are Hiro. Teach gently.";
    let update = instruction_update_from_tool_result("host_invoke_persona", persona_payload, false)
        .expect("Tier 6: persona invocation must yield an instruction update");
    assert_eq!(
        update,
        SessionInstructionUpdate::ActivatePersona {
            persona: "Hiro".to_string(),
            instruction: persona_payload.to_string(),
        }
    );
    apply_session_instruction_update(&mut session, &mut messages, update);

    assert_eq!(
        session.active_persona.as_deref(),
        Some("Hiro"),
        "Tier 6: session active_persona must reflect the swap"
    );
    assert_eq!(
        messages[0]["content"], persona_payload,
        "Tier 6: turn-2 system prompt must reflect the new persona"
    );
    assert_eq!(
        messages[1]["content"], "turn 1 user prompt",
        "Tier 6: prior transcript context must survive the swap"
    );

    // Restore process state before the final acceptance assertion so a panic
    // in the snapshot diff still leaves env clean for sibling tests.
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
        "v0.5 sweep must NOT mutate the operator-global ~/.dm/"
    );
}
