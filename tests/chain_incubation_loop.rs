//! C88 — chain incubation-loop observability via public API.
//!
//! These integration tests exercise only the `pub` surface exposed by
//! `dark_matter::wiki` — `Wiki`, `PlannerBrief`, `RecentCycle`,
//! `IndexEntry`, `PageType`, `WikiIndex`, and the module-level constants
//! (`MOMENTUM_DEFAULT_WINDOW`, `PLANNER_BRIEF_MAX_DRIFT`,
//! `PLANNER_BRIEF_BUDGET_CHARS`). No `pub(crate)` escapes, no visibility
//! promotions.
//!
//! They demonstrate end-to-end observability of the chain-incubation
//! feedback loop from a consumer's perspective:
//!
//!   `write_cycle_synthesis(N)` → `planner_brief` → `PlannerBrief::render`
//!
//! The LLM-in-the-loop variant (running `run_conversation_capture_with_turns`
//! through a mock model provider) is gated on a `MockModelProvider` trait
//! scaffold that does not yet exist — tracked as a follow-up cycle.
//!
//! Integration tests compile as a separate binary, so nothing here
//! contributes to the `--lib` or `--bin dm` ratchet. They live adjacent
//! to `tests/compaction_stress.rs` and follow the same `TempDir` +
//! canonicalize pattern.

use dark_matter::wiki::{
    IndexEntry, PageType, PlannerBrief, RecentCycle, Wiki, MOMENTUM_DEFAULT_WINDOW,
    PLANNER_BRIEF_MAX_DRIFT,
};

use tempfile::TempDir;

/// Cycle N's synthesis page must surface in cycle N+1's planner brief
/// — the minimal observability contract that unblocks every higher-
/// level incubation feature. Proves the round trip works through
/// `Wiki::write_cycle_synthesis` → `Wiki::planner_brief` →
/// `PlannerBrief::render` without touching any internal helper.
#[test]
fn single_cycle_synthesis_appears_in_next_planner_brief() {
    let tmp = TempDir::new().expect("tempdir");
    let proj = tmp.path().canonicalize().expect("canonicalize");
    let wiki = Wiki::open(&proj).expect("open wiki");

    let page_path = wiki
        .write_cycle_synthesis(1, "continuous-dev", &[], None)
        .expect("write synthesis")
        .expect("page path returned (auto-ingest enabled by default)");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .expect("planner_brief");
    assert!(
        !brief.is_empty(),
        "brief must be non-empty after a synthesis write",
    );
    let rendered = brief.render(8192).expect("non-empty brief renders");
    let expected = format!(".dm/wiki/{}", page_path);
    assert!(
        rendered.contains(&expected),
        "rendered brief must reference cycle-1 synthesis path {:?}; got: {}",
        expected,
        rendered,
    );
}

/// Multi-cycle accretion: two synthesis writes with distinct cycle
/// numbers produce a newest-first ordering in the render. Guards
/// against regression where the planner would see old cycles ahead of
/// new ones.
#[test]
fn two_cycle_accretion_brief_lists_both_pages_newer_first() {
    let tmp = TempDir::new().expect("tempdir");
    let proj = tmp.path().canonicalize().expect("canonicalize");
    let wiki = Wiki::open(&proj).expect("open wiki");

    wiki.write_cycle_synthesis(1, "loop", &[], None)
        .expect("cycle 1")
        .expect("enabled");
    wiki.write_cycle_synthesis(2, "loop", &[], None)
        .expect("cycle 2")
        .expect("enabled");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .expect("planner_brief");
    let rendered = brief.render(8192).expect("non-empty brief renders");
    let p2 = rendered.find("Cycle 2 (loop)").expect("cycle 2 present");
    let p1 = rendered.find("Cycle 1 (loop)").expect("cycle 1 present");
    assert!(
        p2 < p1,
        "cycle 2 must precede cycle 1: p2={}, p1={}; render: {}",
        p2,
        p1,
        rendered,
    );
}

/// `<wiki_fresh>` filters to Entity|Concept — synthesis pages MUST NOT
/// leak into the session-start prompt block. A wiki whose index holds
/// only Synthesis entries must suppress the snippet entirely (returns
/// `None`) rather than emit an empty header.
#[test]
fn synthesis_pages_excluded_from_wiki_fresh_snippet() {
    let tmp = TempDir::new().expect("tempdir");
    let proj = tmp.path().canonicalize().expect("canonicalize");
    let wiki = Wiki::open(&proj).expect("open wiki");

    for cycle in 1..=3 {
        wiki.write_cycle_synthesis(cycle, "loop", &[], None)
            .expect("write synthesis")
            .expect("enabled");
    }

    assert_eq!(
        wiki.fresh_pages_snippet(1024),
        None,
        "synthesis-only wiki must suppress the <wiki_fresh> block",
    );
}

/// The two co-composing signals — `PlannerBrief` (synthesis-aware) and
/// `fresh_pages_snippet` (entity/concept-only) — must not cross-
/// contaminate on a shared workspace. The brief surfaces the synthesis
/// page; the snippet surfaces the entity/concept pages; neither leaks
/// the other's content.
#[test]
fn brief_and_fresh_compose_on_shared_workspace() {
    let tmp = TempDir::new().expect("tempdir");
    let proj = tmp.path().canonicalize().expect("canonicalize");
    let wiki = Wiki::open(&proj).expect("open wiki");

    // Seed Entity + Concept entries via the public index API so we
    // don't depend on the ingest pipeline's dedup cache (which is
    // lib-private to reset). `save_index` is pub, `IndexEntry` is pub.
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "module alpha".to_string(),
        path: "entities/alpha.md".to_string(),
        one_liner: "ent alpha".to_string(),
        category: PageType::Entity,
        last_updated: Some("2026-04-19 10:00:00".to_string()),
        outcome: None,
    });
    idx.entries.push(IndexEntry {
        title: "pattern beta".to_string(),
        path: "concepts/beta.md".to_string(),
        one_liner: "con beta".to_string(),
        category: PageType::Concept,
        last_updated: Some("2026-04-19 09:00:00".to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).expect("save_index");

    // Now write a synthesis page on top of that workspace.
    let synth_path = wiki
        .write_cycle_synthesis(1, "loop", &[], None)
        .expect("write synthesis")
        .expect("enabled");

    // Brief surfaces synthesis (via recent_cycles), but its render
    // must NOT contain the raw entity/concept titles — those live in
    // a different signal (`fresh_pages` shows paths, not one_liners).
    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .expect("planner_brief");
    let rendered_brief = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered_brief.contains(&synth_path),
        "brief must surface synthesis path {}: got: {}",
        synth_path,
        rendered_brief,
    );
    assert!(
        !rendered_brief.contains("ent alpha"),
        "brief must not leak entity one_liner content: {}",
        rendered_brief,
    );

    // Snippet surfaces entity/concept; must NOT contain the synthesis
    // path at all — fresh filters it out by category.
    let snippet = wiki
        .fresh_pages_snippet(4096)
        .expect("snippet renders from entity+concept entries");
    assert!(
        snippet.contains("module alpha"),
        "snippet must surface entity title: {}",
        snippet,
    );
    assert!(
        snippet.contains("pattern beta"),
        "snippet must surface concept title: {}",
        snippet,
    );
    assert!(
        !snippet.contains(&synth_path),
        "snippet must not leak synthesis path: {}",
        snippet,
    );
}

/// Compile-time canary. The fact that this file compiles at all proves
/// every `Wiki`/`PlannerBrief` method + every type this integration
/// binary constructs is `pub` in `dark_matter::wiki` — no
/// `pub(crate)` escapes, no visibility promotions needed. Keeping this
/// as a test (not just a `use` statement) makes the contract legible
/// in test output.
#[test]
fn incubation_observability_uses_only_public_api() {
    // The construction below mirrors the types used by the other four
    // tests. If any of these were `pub(crate)`, this file would fail
    // to compile and `cargo test --tests` would surface the breakage.
    let _: PlannerBrief = PlannerBrief::default();
    let _: RecentCycle = RecentCycle {
        cycle: 0,
        chain: String::new(),
        page_path: String::new(),
        last_updated: None,
        outcome: None,
    };
    let _: IndexEntry = IndexEntry {
        title: String::new(),
        path: String::new(),
        one_liner: String::new(),
        category: PageType::Entity,
        last_updated: None,
        outcome: None,
    };
}

/// C90 — end-to-end outcome round trip through the public API.
/// `write_cycle_synthesis(..., Some("green"))` → the `[green]` badge
/// must reach the rendered `PlannerBrief` without any internal
/// plumbing escaping into callers.
#[test]
fn cycle_outcome_round_trips_through_planner_brief_render() {
    let tmp = TempDir::new().expect("tempdir");
    let proj = tmp.path().canonicalize().expect("canonicalize");
    let wiki = Wiki::open(&proj).expect("open wiki");

    wiki.write_cycle_synthesis(1, "continuous-dev", &[], Some("green"))
        .expect("write synthesis")
        .expect("enabled");

    let brief = wiki
        .planner_brief(MOMENTUM_DEFAULT_WINDOW, PLANNER_BRIEF_MAX_DRIFT)
        .expect("planner_brief");
    assert_eq!(
        brief
            .recent_cycles
            .first()
            .and_then(|rc| rc.outcome.as_deref()),
        Some("green"),
        "outcome must round-trip into the brief's recent_cycles",
    );
    let rendered = brief.render(8192).expect("non-empty brief renders");
    assert!(
        rendered.contains("[green]"),
        "rendered brief must contain outcome badge [green]; got: {}",
        rendered,
    );
}
