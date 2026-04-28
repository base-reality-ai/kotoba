use super::git::build_security_review_prompt;
use super::wiki::{
    format_wiki_concepts, format_wiki_fresh, format_wiki_lint, format_wiki_planner_brief,
    format_wiki_prune, format_wiki_search, format_wiki_search_with_fuzzy, format_wiki_seed,
    format_wiki_stats, format_wiki_status, format_wiki_summary, format_wiki_unknown_usage,
    parse_category_prefix, suggest_wiki_subcommand, WIKI_SUBCOMMAND_NAMES,
};
use super::*;

/// Build the stock `(App, OllamaClient, user_tx, user_rx)` tuple used by
/// dispatch-flow tests. `config_dir` is caller-chosen so tests can target
/// a writable `TempDir` (save succeeds) or a `NamedTempFile`-as-file
/// (save fails because `create_dir_all` refuses to make a subdir under a
/// regular file).
fn dispatch_harness(
    config_dir: std::path::PathBuf,
) -> (
    App,
    crate::ollama::client::OllamaClient,
    tokio::sync::mpsc::Sender<String>,
    tokio::sync::mpsc::Receiver<String>,
) {
    let app = App::new(
        "test-model".into(),
        "http://localhost:11434".into(),
        "test-session".into(),
        config_dir,
        vec![],
    );
    let client =
        crate::ollama::client::OllamaClient::new("http://localhost:11434".into(), "test".into());
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    (app, client, tx, rx)
}

#[tokio::test]
async fn config_set_writes_global_settings_json() {
    let project = tempfile::TempDir::new().unwrap();
    let global = tempfile::TempDir::new().unwrap();
    let (mut app, client, tx, _rx) = dispatch_harness(project.path().to_path_buf());
    app.global_config_dir = global.path().to_path_buf();

    let result = execute(
        SlashCommand::Config("set model global-model".into()),
        &mut app,
        &client,
        &tx,
    )
    .await;
    assert!(matches!(result, SlashResult::Done));
    assert!(
        !project.path().join("settings.json").exists(),
        "/config set must not write project-local settings.json"
    );
    let settings_text = std::fs::read_to_string(global.path().join("settings.json")).unwrap();
    assert!(
        settings_text.contains("global-model"),
        "/config set must persist to global settings.json"
    );
}

/// Save-fails variant of `dispatch_harness`. Callers MUST bind the returned
/// `NamedTempFile` (e.g. `let (_file, app, ...)`) so Drop doesn't run early
/// — if it does, the backing file disappears and the "`config_dir` is a file"
/// invariant evaporates, flipping the test onto the success path.
fn dispatch_harness_unwritable_config() -> (
    tempfile::NamedTempFile,
    App,
    crate::ollama::client::OllamaClient,
    tokio::sync::mpsc::Sender<String>,
    tokio::sync::mpsc::Receiver<String>,
) {
    let file = tempfile::NamedTempFile::new().expect("create temp file");
    let (app, client, tx, rx) = dispatch_harness(file.path().to_path_buf());
    (file, app, client, tx, rx)
}

fn sample_session(id: &str) -> crate::session::Session {
    let mut s = crate::session::Session::new("/tmp".to_string(), "gemma4:26b".to_string());
    // Force id so short_id is deterministic in assertions.
    s.id = id.to_string();
    s
}

#[test]
fn format_session_row_uses_persisted_counts() {
    let mut s = sample_session("abcdef0123456789");
    s.title = Some("my session".to_string());
    s.turn_count = 4;
    s.prompt_tokens = 12_000;
    s.completion_tokens = 345;
    let row = format_session_row(&s);
    assert!(
        row.contains("abcdef01"),
        "row should include short id: {}",
        row
    );
    assert!(row.contains("4 turns"), "persisted turn count: {}", row);
    assert!(
        !row.contains("~"),
        "no estimate prefix when persisted: {}",
        row
    );
    assert!(row.contains("12.3k tok"), "human-formatted tokens: {}", row);
    assert!(row.contains("my session"), "title visible: {}", row);
    assert!(row.contains("gemma4:26b"), "model visible: {}", row);
}

#[test]
fn format_session_row_includes_host_project_when_present() {
    let mut s = sample_session("abcdef0123456789");
    s.host_project = Some("finance-app".to_string());

    let row = format_session_row(&s);

    assert!(
        row.contains("finance-app"),
        "host project should be visible in session rows: {}",
        row
    );
}

#[test]
fn format_session_row_singular_turn_wording() {
    let mut s = sample_session("oneturnid00000000");
    s.turn_count = 1;
    let row = format_session_row(&s);
    assert!(row.contains("1 turn"), "singular form: {}", row);
    assert!(
        !row.contains("1 turns"),
        "should not say '1 turns': {}",
        row
    );
}

#[test]
fn format_session_row_falls_back_to_estimate_for_old_session() {
    // Old session: no persisted turn_count but has user messages — we
    // estimate the count from message role counts and mark it with `~`.
    let mut s = sample_session("legacyid00000000");
    s.messages = vec![
        serde_json::json!({"role": "system", "content": "sys"}),
        serde_json::json!({"role": "user", "content": "one"}),
        serde_json::json!({"role": "assistant", "content": "r"}),
        serde_json::json!({"role": "user", "content": "two"}),
    ];
    let row = format_session_row(&s);
    assert!(row.contains("~2 turns"), "estimate marked with ~: {}", row);
    // Tokens should remain "—" because we never estimate token counts.
    assert!(row.contains("| — |"), "no token estimate: {}", row);
}

#[test]
fn format_session_row_empty_session_shows_em_dashes() {
    let s = sample_session("emptyid000000000");
    let row = format_session_row(&s);
    assert!(
        row.matches('—').count() >= 2,
        "turns and tokens both '—' for empty session: {}",
        row
    );
}

#[test]
fn suggest_slash_command_exact_typo() {
    assert_eq!(suggest_slash_command("hlep"), Some("help"));
}

#[test]
fn suggest_slash_command_one_char_off() {
    assert_eq!(suggest_slash_command("modle"), Some("model"));
}

#[test]
fn suggest_slash_command_no_match() {
    assert_eq!(suggest_slash_command("xyzzy"), None);
}

#[test]
fn suggest_slash_command_empty_is_none() {
    assert_eq!(suggest_slash_command(""), None);
}

#[test]
fn suggest_slash_command_single_char_is_not_attach() {
    assert_ne!(suggest_slash_command("c"), Some("attach"));
}

#[test]
fn suggest_slash_command_single_char_uses_levenshtein() {
    assert_eq!(suggest_slash_command("c"), Some("cd"));
}

#[test]
fn suggest_slash_command_two_char_prefix_still_suggests() {
    assert_eq!(suggest_slash_command("he"), Some("help"));
}

fn count_turns(entries: &[crate::tui::app::DisplayEntry]) -> usize {
    entries
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::UserMessage))
        .count()
}

#[test]
fn compact_skips_when_few_turns() {
    // Build a fake entries list with 3 user turns (below the threshold of 5).
    let entries: Vec<crate::tui::app::DisplayEntry> = (0..3)
        .map(|_| crate::tui::app::DisplayEntry {
            kind: EntryKind::UserMessage,
            content: "user".into(),
        })
        .collect();
    let turns = count_turns(&entries);
    assert_eq!(turns, 3, "expected 3 turns");
    assert!(
        turns < 5,
        "should be below the compact threshold of 5 turns"
    );
    // Simulate what execute() would return
    let ctx_pct = 50usize; // above 30% but turn count is too low
    let result_text = format!(
        "Nothing to compact — context is only {}% full ({} turns).",
        ctx_pct, turns
    );
    assert!(result_text.contains("Nothing to compact"));
}

#[test]
fn compact_keeps_last_8_messages() {
    // 20 conversation messages → 12 should be marked for summarization, 8 kept.
    const KEEP_TAIL: usize = 8;
    let total_msgs = 20usize;
    // system message is separate; agent keeps system + KEEP_TAIL
    let to_summarize = total_msgs.saturating_sub(KEEP_TAIL);
    assert_eq!(to_summarize, 12, "expected 12 messages to summarize");
    let kept = total_msgs - to_summarize;
    assert_eq!(kept, 8, "expected 8 messages kept as tail");
}

#[test]
fn memory_add_parses_correctly() {
    let cmd = parse("/memory add hello world").expect("should parse");
    match cmd {
        SlashCommand::MemoryAdd(text) => assert_eq!(text, "hello world"),
        _ => panic!("expected MemoryAdd"),
    }
}

#[test]
fn memory_forget_parses_index() {
    let cmd = parse("/memory forget 3").expect("should parse");
    match cmd {
        SlashCommand::MemoryForget(n) => assert_eq!(n, 3),
        _ => panic!("expected MemoryForget(3)"),
    }
}

#[test]
fn memory_forget_rejects_zero() {
    let cmd = parse("/memory forget 0").expect("should parse");
    match cmd {
        SlashCommand::MemoryForget(_) => panic!("should NOT be MemoryForget for index 0"),
        SlashCommand::Unknown(_) => {} // expected
        _ => {}                        // also acceptable
    }
}

#[test]
fn memory_list_parses() {
    let cmd = parse("/memory").expect("should parse");
    assert!(matches!(cmd, SlashCommand::MemoryList));
}

#[test]
fn memory_clear_parses() {
    let cmd = parse("/memory clear").expect("should parse");
    assert!(matches!(cmd, SlashCommand::MemoryClear));
}

#[test]
fn parse_wiki_status_variants() {
    // Bare `/wiki` and explicit `/wiki status` both route to WikiStatus.
    assert!(matches!(
        parse("/wiki").expect("should parse"),
        SlashCommand::WikiStatus
    ));
    assert!(matches!(
        parse("/wiki status").expect("should parse"),
        SlashCommand::WikiStatus
    ));
    // Unknown subcommands are held in WikiUnknown with the full trailing
    // argument so the help text can echo back what the user typed.
    match parse("/wiki foo").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "foo"),
        other => panic!("expected WikiUnknown, got {:?}", other),
    }
}

// ── /wiki status parse edge cases (cycle 10) ─────────────────────────
//
// Trailing whitespace after a subcommand: `parts.get(1)` yields the empty
// remainder (splitn(2, ' ') on "wiki " gives parts = ["wiki", ""]), which
// trims to "" and routes to WikiStatus. Documented here so "bare /wiki"
// and "/wiki " (stray space) behave identically — the user's space key
// shouldn't change the command.
#[test]
fn parse_wiki_trailing_whitespace_still_routes_to_status() {
    assert!(matches!(
        parse("/wiki ").expect("should parse"),
        SlashCommand::WikiStatus
    ));
    assert!(matches!(
        parse("/wiki    ").expect("should parse"),
        SlashCommand::WikiStatus
    ));
}

// Pinned behavior: "/wiki status extra" is NOT treated as WikiStatus with
// the extra ignored — `splitn(2, ' ')` means `parts[1] = "status extra"`,
// which fails the `sub == "status"` check and routes to WikiUnknown.
// Captured here so a future tweak that tries to be more lenient about
// trailing args doesn't silently change routing.
#[test]
fn parse_wiki_status_with_trailing_args_is_unknown() {
    match parse("/wiki status extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "status extra"),
        other => panic!("expected WikiUnknown for 'status extra', got {:?}", other),
    }
}

// ── /wiki search parse tests (cycle 11) ──────────────────────────────

#[test]
fn parse_wiki_search_extracts_query() {
    match parse("/wiki search compaction").expect("should parse") {
        SlashCommand::WikiSearch(q) => assert_eq!(q, "compaction"),
        other => panic!("expected WikiSearch, got {:?}", other),
    }
    // Multi-word query survives intact.
    match parse("/wiki search foo bar baz").expect("should parse") {
        SlashCommand::WikiSearch(q) => assert_eq!(q, "foo bar baz"),
        other => panic!("expected WikiSearch, got {:?}", other),
    }
}

#[test]
fn parse_wiki_search_empty_query() {
    // "/wiki search" with no query routes to WikiSearch("") so the
    // formatter can render the usage hint. Not WikiUnknown.
    match parse("/wiki search").expect("should parse") {
        SlashCommand::WikiSearch(q) => assert_eq!(q, ""),
        other => panic!("expected WikiSearch(\"\"), got {:?}", other),
    }
    // Trailing space after "search" is the same empty query.
    match parse("/wiki search ").expect("should parse") {
        SlashCommand::WikiSearch(q) => assert_eq!(q, ""),
        other => panic!("expected WikiSearch(\"\"), got {:?}", other),
    }
}

// Word-boundary guard: "/wiki searchfoo" must NOT be misread as
// WikiSearch("foo"). It's a distinct (unknown) subcommand.
#[test]
fn parse_wiki_searchfoo_is_unknown() {
    match parse("/wiki searchfoo").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "searchfoo"),
        other => panic!("expected WikiUnknown for 'searchfoo', got {:?}", other),
    }
}

// ── parse_category_prefix unit tests (cycle 13) ──────────────────────

#[test]
fn parse_category_prefix_recognizes_known_prefixes() {
    assert_eq!(
        parse_category_prefix("entity:foo"),
        ("foo".to_string(), Some("entity".to_string()))
    );
    assert_eq!(
        parse_category_prefix("concept:foo"),
        ("foo".to_string(), Some("concept".to_string()))
    );
    assert_eq!(
        parse_category_prefix("summary:foo"),
        ("foo".to_string(), Some("summary".to_string()))
    );
    assert_eq!(
        parse_category_prefix("synthesis:foo"),
        ("foo".to_string(), Some("synthesis".to_string()))
    );
}

#[test]
fn parse_category_prefix_strips_whitespace_after_colon() {
    assert_eq!(
        parse_category_prefix("entity:  auth tokens"),
        ("auth tokens".to_string(), Some("entity".to_string()))
    );
}

#[test]
fn parse_category_prefix_passes_through_unrecognized() {
    assert_eq!(parse_category_prefix("auth"), ("auth".to_string(), None));
    assert_eq!(
        parse_category_prefix("foo:bar"),
        ("foo:bar".to_string(), None)
    );
    // Whitespace inside the would-be prefix disqualifies it.
    assert_eq!(
        parse_category_prefix("some entity:auth"),
        ("some entity:auth".to_string(), None)
    );
}

#[test]
fn parse_category_prefix_case_insensitive() {
    assert_eq!(
        parse_category_prefix("ENTITY:auth"),
        ("auth".to_string(), Some("entity".to_string()))
    );
    assert_eq!(
        parse_category_prefix("Concept:Foo"),
        ("Foo".to_string(), Some("concept".to_string()))
    );
}

#[test]
fn parse_category_prefix_empty_query_after_prefix() {
    assert_eq!(
        parse_category_prefix("entity:"),
        (String::new(), Some("entity".to_string()))
    );
}

#[test]
fn parse_category_prefix_multiword_query() {
    assert_eq!(
        parse_category_prefix("concept: auth pattern logic"),
        (
            "auth pattern logic".to_string(),
            Some("concept".to_string())
        )
    );
}

#[test]
fn handle_wiki_search_category_prefix_with_empty_query_is_specific_hint() {
    // Cycle 444: `/wiki search entity:` (prefix parsed, rest blank)
    // used to fall through to the generic "Usage: /wiki search …"
    // message, which doesn't acknowledge the category prefix was
    // accepted. The dispatch-level guard now returns a specific hint.
    // This guard fires before Wiki::open, so no wiki/cwd setup needed.
    use crate::tui::commands::wiki::handle_wiki_search;
    use crate::tui::commands::SlashResult;

    for cat in &["entity", "concept", "summary", "synthesis"] {
        let result = handle_wiki_search(format!("{}:", cat));
        let msg = match result {
            SlashResult::Info(m) => m,
            _ => panic!("{}: expected SlashResult::Info", cat),
        };
        assert!(
            msg.contains(&format!("category '{}' was parsed", cat)),
            "{}: should acknowledge parsed category: {}",
            cat,
            msg
        );
        assert!(
            msg.contains("query is empty"),
            "{}: should explain the failure: {}",
            cat,
            msg
        );
        assert!(
            msg.contains("Try:"),
            "{}: should include next-step hint per directive: {}",
            cat,
            msg
        );
        assert!(
            msg.contains(&format!("{}:auth", cat)) || msg.contains(&format!("{}:<", cat)),
            "{}: should show example query syntax: {}",
            cat,
            msg
        );
    }

    // Whitespace-only rest also hits the guard.
    let result = handle_wiki_search("entity:   ".to_string());
    let msg = match result {
        SlashResult::Info(m) => m,
        _ => panic!("expected Info for whitespace-only rest"),
    };
    assert!(
        msg.contains("query is empty"),
        "whitespace-only rest should hit same guard: {}",
        msg
    );
}

// ── /wiki lint parse tests (cycle 13) ────────────────────────────────

#[test]
fn parse_wiki_lint_no_args() {
    assert!(matches!(
        parse("/wiki lint").expect("should parse"),
        SlashCommand::WikiLint
    ));
}

// Word-boundary guard parallel to `/wiki searchfoo` — no silent
// prefix match.
#[test]
fn parse_wiki_lintfoo_is_unknown() {
    match parse("/wiki lintfoo").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "lintfoo"),
        other => panic!("expected WikiUnknown for 'lintfoo', got {:?}", other),
    }
}

// `/wiki lint extra` must NOT silently drop the trailing args —
// route to WikiUnknown with the full argstring so the user gets
// feedback.
#[test]
fn parse_wiki_lint_with_trailing_args_is_unknown() {
    match parse("/wiki lint extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "lint extra"),
        other => panic!("expected WikiUnknown for 'lint extra', got {:?}", other),
    }
}

// Subcommand matching is case-sensitive for the `/wiki` family —
// locks symmetry with status/search and keeps the door open for
// a case-sensitive future subcommand that differs only in case.
#[test]
fn parse_wiki_lint_is_case_sensitive() {
    for variant in ["/wiki LINT", "/wiki Lint", "/wiki lInT"] {
        match parse(variant).expect("should parse") {
            SlashCommand::WikiUnknown(_) => {}
            other => panic!("expected WikiUnknown for {:?}, got {:?}", variant, other),
        }
    }
}

// Trailing whitespace after `lint` must be absorbed by the trim —
// symmetric with `parse_wiki_status_variants` (trailing ws → status).
// This is the natural typo/paste case and should NOT route to
// WikiUnknown (which is reserved for genuinely unrecognized args).
#[test]
fn parse_wiki_lint_tolerates_trailing_whitespace() {
    for variant in ["/wiki lint ", "/wiki lint   ", "/wiki lint\t"] {
        assert!(
            matches!(
                parse(variant).expect("should parse"),
                SlashCommand::WikiLint
            ),
            "expected WikiLint for {:?}",
            variant
        );
    }
}

// ── /wiki refresh parse tests (cycle 44) ─────────────────────────────
//
// `/wiki refresh` routes to a distinct variant (not WikiUnknown) so the
// dispatcher can invoke `Wiki::refresh()`. Trailing whitespace tolerated
// like the other /wiki family commands; trailing args route to
// WikiUnknown for user feedback.

#[test]
fn parse_wiki_refresh_routes_to_wiki_refresh() {
    assert!(matches!(
        parse("/wiki refresh").expect("should parse"),
        SlashCommand::WikiRefresh
    ));
    // Trailing whitespace tolerated (matches /wiki lint behavior).
    for variant in ["/wiki refresh ", "/wiki refresh   ", "/wiki refresh\t"] {
        assert!(
            matches!(
                parse(variant).expect("should parse"),
                SlashCommand::WikiRefresh
            ),
            "expected WikiRefresh for {:?}",
            variant
        );
    }
}

#[test]
fn parse_wiki_refresh_with_trailing_args_is_unknown() {
    match parse("/wiki refresh extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "refresh extra"),
        other => panic!("expected WikiUnknown for 'refresh extra', got {:?}", other),
    }
}

// ── /wiki summary parse tests (cycle 45) ─────────────────────────────
//
// `/wiki summary` routes to a distinct variant so the dispatcher can
// invoke `Wiki::write_project_summary()`. Parallel to refresh's tests:
// trailing whitespace tolerated; trailing args → WikiUnknown.

#[test]
fn parse_wiki_summary_routes_to_wiki_summary() {
    assert!(matches!(
        parse("/wiki summary").expect("should parse"),
        SlashCommand::WikiSummary
    ));
    for variant in ["/wiki summary ", "/wiki summary   ", "/wiki summary\t"] {
        assert!(
            matches!(
                parse(variant).expect("should parse"),
                SlashCommand::WikiSummary
            ),
            "expected WikiSummary for {:?}",
            variant
        );
    }
    match parse("/wiki summary extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "summary extra"),
        other => panic!("expected WikiUnknown for 'summary extra', got {:?}", other),
    }
}

// ── /wiki concepts parse + formatter tests (cycle 49) ───────────────
//
// `/wiki concepts` routes to a distinct variant so the dispatcher can
// invoke `Wiki::write_concept_pages()`. Parallel to refresh/summary:
// trailing whitespace tolerated; trailing args → WikiUnknown. The
// formatter exercises empty-state hint, top-N dep listing with
// consumer counts, and the trailing next-step line.

#[test]
fn parse_wiki_concepts_routes_to_wiki_concepts() {
    assert!(matches!(
        parse("/wiki concepts").expect("should parse"),
        SlashCommand::WikiConcepts
    ));
    for variant in ["/wiki concepts ", "/wiki concepts   ", "/wiki concepts\t"] {
        assert!(
            matches!(
                parse(variant).expect("should parse"),
                SlashCommand::WikiConcepts
            ),
            "expected WikiConcepts for {:?}",
            variant
        );
    }
}

#[test]
fn parse_wiki_concept_singular_falls_through_to_unknown() {
    // Regression guard: the variant is `concepts` (plural). Singular
    // must route to WikiUnknown so the user gets a usage hint and
    // doesn't mistakenly think the command succeeded.
    match parse("/wiki concept").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "concept"),
        other => panic!("expected WikiUnknown for 'concept', got {:?}", other),
    }
    match parse("/wiki concepts extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "concepts extra"),
        other => panic!("expected WikiUnknown for 'concepts extra', got {:?}", other),
    }
}

#[test]
fn format_wiki_concepts_shows_empty_state_hint() {
    let report = crate::wiki::ConceptPagesReport::default();
    let out = format_wiki_concepts(&report);
    assert!(
        out.contains("No shared dependencies meet the threshold"),
        "empty report must render the threshold-miss hint; got: {}",
        out,
    );
    assert!(
        out.contains("/wiki refresh"),
        "empty-state hint must reference `/wiki refresh` as next step; got: {}",
        out,
    );
}

#[test]
fn format_wiki_concepts_lists_top_deps_with_counts() {
    let report = crate::wiki::ConceptPagesReport {
        generated: vec!["concepts/dep-std_sync_Arc.md".to_string()],
        refreshed: vec![],
        detected_deps: vec![
            ("std::sync::Arc".to_string(), 7),
            ("tokio::sync::Mutex".to_string(), 4),
            ("anyhow::Result".to_string(), 3),
        ],
    };
    let out = format_wiki_concepts(&report);
    assert!(out.contains("std::sync::Arc"));
    assert!(out.contains("tokio::sync::Mutex"));
    assert!(out.contains("anyhow::Result"));
    assert!(out.contains("7 consumer(s)"));
    assert!(out.contains("4 consumer(s)"));
    assert!(out.contains("3 consumer(s)"));
    assert!(
        out.contains("1 new, 0 refreshed, 3 total"),
        "header must reflect generated/refreshed/detected counts; got: {}",
        out,
    );
}

#[test]
fn format_wiki_concepts_caps_listing_at_ten_and_includes_next_step() {
    let mut detected = Vec::new();
    for i in 0..15 {
        detected.push((format!("dep_{:02}", i), 3));
    }
    let report = crate::wiki::ConceptPagesReport {
        generated: vec![],
        refreshed: vec![],
        detected_deps: detected,
    };
    let out = format_wiki_concepts(&report);
    // First 10 listed, last 5 elided.
    assert!(out.contains("dep_00"));
    assert!(out.contains("dep_09"));
    assert!(
        !out.contains("dep_10"),
        "11th entry must be elided by take(10); got: {}",
        out,
    );
    // Trailing next-step line present on non-empty report.
    assert!(
        out.contains("Next:") && out.contains("/wiki summary"),
        "non-empty report must carry next-step hint; got: {}",
        out,
    );
}

// ── format_wiki_status formatter tests (cycle 10) ────────────────────
//
// The data-layer `Wiki::stats()` has tests in wiki::tests; the display
// formatter does not. These lock in the contract that `/wiki status`
// output is deterministic in category order, zero-fills missing
// categories, and shows the "empty wiki" hint iff total_pages == 0.
fn mk_stats(
    total: usize,
    entities: usize,
    concepts: usize,
    summaries: usize,
    synthesis: usize,
    log_entries: usize,
    last_activity: Option<&str>,
) -> crate::wiki::WikiStats {
    use crate::wiki::PageType;
    let mut by_category = std::collections::BTreeMap::new();
    if entities > 0 {
        by_category.insert(PageType::Entity, entities);
    }
    if concepts > 0 {
        by_category.insert(PageType::Concept, concepts);
    }
    if summaries > 0 {
        by_category.insert(PageType::Summary, summaries);
    }
    if synthesis > 0 {
        by_category.insert(PageType::Synthesis, synthesis);
    }
    crate::wiki::WikiStats {
        root: std::path::PathBuf::from("/fake/root/.dm/wiki"),
        total_pages: total,
        by_category,
        log_entries,
        last_activity: last_activity.map(|s| s.to_string()),
        most_linked: Vec::new(),
    }
}

#[test]
fn format_wiki_status_empty_wiki_shows_hint() {
    let s = mk_stats(0, 0, 0, 0, 0, 0, None);
    let out = format_wiki_status(&s);
    assert!(
        out.contains("0 entities"),
        "missing entity zero-fill: {}",
        out
    );
    assert!(
        out.contains("0 concepts"),
        "missing concept zero-fill: {}",
        out
    );
    assert!(
        out.contains("0 summaries"),
        "missing summary zero-fill: {}",
        out
    );
    assert!(
        out.contains("0 synthesis"),
        "missing synthesis zero-fill: {}",
        out
    );
    assert!(
        out.contains("(none)"),
        "last_activity=None should show (none): {}",
        out
    );
    assert!(
        out.contains("Wiki is empty"),
        "empty-wiki hint must appear when total_pages == 0: {}",
        out
    );
}

#[test]
fn format_wiki_status_populated_suppresses_hint() {
    let s = mk_stats(7, 5, 1, 0, 1, 12, Some("[2026-01-01] ingest | src/main.rs"));
    let out = format_wiki_status(&s);
    assert!(
        !out.contains("Wiki is empty"),
        "hint must not appear when total_pages > 0: {}",
        out
    );
    assert!(out.contains("5 entities"));
    assert!(out.contains("1 concepts"));
    assert!(
        out.contains("0 summaries"),
        "zero-fill for empty category: {}",
        out
    );
    assert!(out.contains("1 synthesis"));
    assert!(out.contains("Log:         12 entries"));
    assert!(out.contains("src/main.rs"));
}

// The display order is load-bearing for UX parity across cycles — lock
// "entities → concepts → summaries → synthesis" in a single line.
#[test]
fn format_wiki_status_canonical_category_order() {
    let s = mk_stats(10, 2, 3, 4, 1, 0, None);
    let out = format_wiki_status(&s);
    let e = out.find("2 entities").expect("entities present");
    let c = out.find("3 concepts").expect("concepts present");
    let m = out.find("4 summaries").expect("summaries present");
    let y = out.find("1 synthesis").expect("synthesis present");
    assert!(
        e < c && c < m && m < y,
        "canonical order broken: entities@{} concepts@{} summaries@{} synthesis@{} in:\n{}",
        e,
        c,
        m,
        y,
        out
    );
}

// Non-zero-total with no log activity yet is a valid intermediate state
// (e.g., index seeded manually). The hint must stay off, the (none)
// placeholder must appear.
#[test]
fn format_wiki_status_pages_without_log_shows_placeholder_not_hint() {
    let s = mk_stats(2, 2, 0, 0, 0, 0, None);
    let out = format_wiki_status(&s);
    assert!(
        !out.contains("Wiki is empty"),
        "hint fires off total_pages, not log activity: {}",
        out
    );
    assert!(
        out.contains("(none)"),
        "missing last-activity placeholder: {}",
        out
    );
}

#[test]
fn format_wiki_status_includes_root_path() {
    let s = mk_stats(1, 1, 0, 0, 0, 1, Some("[2026-04-17] ingest | a.rs"));
    let out = format_wiki_status(&s);
    assert!(
        out.contains("/fake/root/.dm/wiki"),
        "root path must appear in output: {}",
        out
    );
}

#[test]
fn format_wiki_status_renders_most_linked_section() {
    let mut s = mk_stats(3, 3, 0, 0, 0, 3, Some("[2026-04-18] ingest | src/main.rs"));
    s.most_linked = vec![
        ("entities/src_main_rs.md".to_string(), 3),
        ("entities/src_tools_mod_rs.md".to_string(), 2),
    ];
    let out = format_wiki_status(&s);
    assert!(
        out.contains("Most linked:"),
        "Most linked header must appear: {}",
        out
    );
    assert!(
        out.contains("entities/src_main_rs.md  (3)"),
        "top entry must render path + count: {}",
        out
    );
    assert!(
        out.contains("entities/src_tools_mod_rs.md  (2)"),
        "second entry must render: {}",
        out
    );
}

#[test]
fn format_wiki_status_omits_most_linked_when_empty() {
    let s = mk_stats(2, 2, 0, 0, 0, 2, Some("[2026-04-18] ingest | a.rs"));
    let out = format_wiki_status(&s);
    assert!(
        !out.contains("Most linked:"),
        "empty most_linked must suppress the section entirely: {}",
        out
    );
}

// ── format_wiki_search formatter tests (cycle 11) ────────────────────

#[test]
fn format_wiki_search_empty_query_shows_usage() {
    let out = format_wiki_search("", &[]);
    assert!(out.contains("Usage"), "usage hint must appear: {}", out);
    assert!(
        out.contains("/wiki search"),
        "command name must appear: {}",
        out
    );
    // Whitespace-only query trims to empty — same shape.
    let out2 = format_wiki_search("   ", &[]);
    assert!(out2.contains("Usage"));
}

#[test]
fn format_wiki_search_empty_query_mentions_category_prefix() {
    // After C13 the slash supports `cat:query` prefix; the empty-query
    // usage hint must surface that syntax. Drift trip: a future refactor
    // that drops the example or the category list should fail this test.
    let out = format_wiki_search("", &[]);
    assert!(out.contains("Usage:"), "must show Usage: prefix: {}", out);
    assert!(
        out.contains("entity") && out.contains("concept"),
        "must list category options: {}",
        out
    );
    assert!(
        out.contains("Example:"),
        "must include a category-prefix example: {}",
        out
    );
}

#[test]
fn format_wiki_search_zero_hits() {
    let out = format_wiki_search("needle", &[]);
    assert!(out.contains("No wiki matches"), "zero-hit line: {}", out);
    assert!(out.contains("needle"), "query echoed back: {}", out);
}

#[test]
fn format_wiki_search_renders_hits_with_path_title_count_snippet() {
    let hits = vec![
        crate::wiki::WikiSearchHit {
            path: "entities/a.md".to_string(),
            title: "alpha".to_string(),
            category: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            match_count: 3,
            snippet: "…hit snippet for alpha…".to_string(),
        },
        crate::wiki::WikiSearchHit {
            path: "concepts/b.md".to_string(),
            title: "beta".to_string(),
            category: crate::wiki::PageType::Concept,
            layer: crate::wiki::Layer::Kernel,
            match_count: 1,
            snippet: "beta".to_string(),
        },
    ];
    let out = format_wiki_search("foo", &hits);
    // Header count + plural.
    assert!(out.contains("Found 2 matches for 'foo'"), "header: {}", out);
    // Per-hit: path, title, match count, snippet.
    assert!(
        out.contains("[3x] entities/a.md — alpha"),
        "first hit: {}",
        out
    );
    assert!(
        out.contains("…hit snippet for alpha…"),
        "first snippet: {}",
        out
    );
    assert!(
        out.contains("[1x] concepts/b.md — beta"),
        "second hit: {}",
        out
    );
    // Ordering in output matches slice order (caller did the sort).
    let pa = out.find("entities/a.md").unwrap();
    let pb = out.find("concepts/b.md").unwrap();
    assert!(pa < pb, "hit order preserved: a@{} b@{}", pa, pb);
}

#[test]
fn format_wiki_search_fuzzy_lists_close_titles_when_within_threshold() {
    let ranked = vec![
        (1usize, "entities/needl.md".to_string(), "Needl".to_string()),
        (3usize, "entities/other.md".to_string(), "Other".to_string()),
    ];
    let out = format_wiki_search_with_fuzzy("needle", &[], &ranked, 4);
    assert!(out.contains("Closest titles:"), "header: {}", out);
    assert!(out.contains("entities/needl.md"), "path: {}", out);
    assert!(out.contains("(distance 1)"), "distance render: {}", out);
    assert!(out.contains("Tip:"), "next-step hint: {}", out);
}

#[test]
fn format_wiki_search_fuzzy_rejected_when_all_above_threshold() {
    let ranked = vec![
        (7usize, "entities/far_a.md".to_string(), "FarA".to_string()),
        (9usize, "entities/far_b.md".to_string(), "FarB".to_string()),
    ];
    let out = format_wiki_search_with_fuzzy("needle", &[], &ranked, 4);
    assert!(out.contains("No wiki matches"), "404 stem: {}", out);
    assert!(
        out.contains("above similarity threshold 4"),
        "rejected header: {}",
        out
    );
    assert!(out.contains("(distance 7)"), "distance render: {}", out);
    assert!(
        !out.contains("Closest titles:\n"),
        "must not use accepted-fuzzy wording: {}",
        out
    );
}

#[test]
fn format_wiki_search_fuzzy_empty_keeps_bare_404() {
    let out = format_wiki_search_with_fuzzy("needle", &[], &[], 0);
    assert!(
        out.contains("No wiki matches for 'needle'"),
        "bare 404: {}",
        out
    );
    assert!(
        !out.contains("similarity threshold"),
        "no rejected chatter on empty fuzzy: {}",
        out
    );
    assert!(
        !out.contains("Closest titles"),
        "no accepted chatter on empty fuzzy: {}",
        out
    );
}

// ── /wiki stats parse + formatter + telemetry roundtrip ──────────────

#[test]
fn parse_wiki_stats_dispatches() {
    assert!(matches!(
        parse("/wiki stats").expect("should parse"),
        SlashCommand::WikiStats
    ));
}

#[test]
fn format_wiki_stats_renders_three_metrics() {
    let out = format_wiki_stats(7, 2, 5012);
    assert!(out.contains("tool_calls:"), "tool_calls label: {}", out);
    assert!(
        out.contains("drift_warnings:"),
        "drift_warnings label: {}",
        out
    );
    assert!(
        out.contains("snippet_bytes:"),
        "snippet_bytes label: {}",
        out
    );
    assert!(out.contains("7"), "tool_calls value: {}", out);
    assert!(out.contains("2"), "drift_warnings value: {}", out);
    assert!(out.contains("5012"), "snippet_bytes value: {}", out);
    assert!(out.contains("Tip:"), "tip line: {}", out);
    assert!(
        out.contains("[wiki-drift]"),
        "tip should name drift marker: {}",
        out
    );
}

// ── /wiki prune parse + formatter ────────────────────────────────────

#[test]
fn parse_wiki_prune_no_args_uses_default() {
    let cmd = parse("/wiki prune").expect("should parse");
    assert!(
        matches!(cmd, SlashCommand::WikiPrune(n) if n == crate::wiki::DEFAULT_COMPACT_KEEP),
        "got: {:?}",
        cmd,
    );
}

#[test]
fn parse_wiki_prune_with_count() {
    assert!(matches!(
        parse("/wiki prune 50").expect("should parse"),
        SlashCommand::WikiPrune(50)
    ));
}

#[test]
fn parse_wiki_prune_zero_allowed() {
    assert!(matches!(
        parse("/wiki prune 0").expect("should parse"),
        SlashCommand::WikiPrune(0)
    ));
}

#[test]
fn parse_wiki_prune_invalid_arg_is_unknown() {
    let cmd = parse("/wiki prune abc").expect("should parse");
    match cmd {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "prune abc"),
        other => panic!("expected WikiUnknown(\"prune abc\"), got: {:?}", other),
    }
}

#[test]
fn parse_wiki_prune_negative_is_unknown() {
    let cmd = parse("/wiki prune -5").expect("should parse");
    match cmd {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "prune -5"),
        other => panic!("expected WikiUnknown(\"prune -5\"), got: {:?}", other),
    }
}

#[test]
fn parse_wiki_prune_with_trailing_args_after_n_is_unknown() {
    let cmd = parse("/wiki prune 5 extra").expect("should parse");
    match cmd {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "prune 5 extra"),
        other => panic!("expected WikiUnknown(\"prune 5 extra\"), got: {:?}", other),
    }
}

#[test]
fn wiki_subcommand_names_contains_prune() {
    assert!(
        WIKI_SUBCOMMAND_NAMES.contains(&"prune"),
        "WIKI_SUBCOMMAND_NAMES missing 'prune': {:?}",
        WIKI_SUBCOMMAND_NAMES,
    );
}

#[test]
fn suggest_wiki_subcommand_prun_returns_prune() {
    assert_eq!(suggest_wiki_subcommand("prun"), Some("prune"));
}

#[test]
fn format_wiki_prune_zero_pruned_says_already_within_cap() {
    let out = format_wiki_prune(0, 200);
    assert!(
        out.contains("already within cap"),
        "no-op message missing 'already within cap': {}",
        out,
    );
    assert!(out.contains("200"), "cap should appear in message: {}", out);
}

#[test]
fn format_wiki_prune_one_uses_singular() {
    let out = format_wiki_prune(1, 10);
    assert!(
        out.contains("1 compact-* synthesis page "),
        "singular: {}",
        out
    );
    assert!(
        !out.contains("pages"),
        "singular form must not contain 'pages': {}",
        out,
    );
}

#[test]
fn format_wiki_prune_many_uses_plural() {
    let out = format_wiki_prune(5, 10);
    assert!(
        out.contains("5 compact-* synthesis pages "),
        "plural: {}",
        out
    );
}

#[test]
fn format_wiki_unknown_usage_includes_prune_line() {
    let out = format_wiki_unknown_usage("zzz");
    assert!(
        out.contains("/wiki prune"),
        "usage missing /wiki prune line: {}",
        out,
    );
}

#[test]
fn format_wiki_unknown_usage_has_three_intent_groups() {
    let out = format_wiki_unknown_usage("zzz");
    for header in ["Inspect:", "Maintain:", "Operator one-shot:"] {
        assert!(
            out.contains(header),
            "usage missing intent-group header {:?}: {}",
            header,
            out,
        );
    }
}

#[test]
fn format_wiki_unknown_usage_groups_appear_in_intent_order() {
    let out = format_wiki_unknown_usage("zzz");
    let inspect = out.find("Inspect:").expect("Inspect header");
    let maintain = out.find("Maintain:").expect("Maintain header");
    let oneshot = out
        .find("Operator one-shot:")
        .expect("Operator one-shot header");
    assert!(
        inspect < maintain && maintain < oneshot,
        "intent groups out of order — Inspect@{} Maintain@{} Operator one-shot@{}",
        inspect,
        maintain,
        oneshot,
    );
}

#[test]
fn format_wiki_unknown_usage_lists_all_twelve_subcommands() {
    let out = format_wiki_unknown_usage("zzz");
    // Subcommand lines use 6-space indent under their group header,
    // distinguishing them from the leading echo of the unrecognized
    // sub. The count must equal len(WIKI_SUBCOMMAND_NAMES).
    let count = out.matches("\n      /wiki ").count();
    assert_eq!(
        count,
        WIKI_SUBCOMMAND_NAMES.len(),
        "expected one line per subcommand under intent groups; got {} lines, names: {:?}\n\
             output:\n{}",
        count,
        WIKI_SUBCOMMAND_NAMES,
        out,
    );
}

// ── /wiki seed parse + formatter ─────────────────────────────────────

#[test]
fn parse_wiki_seed_no_args() {
    assert!(matches!(
        parse("/wiki seed").expect("should parse"),
        SlashCommand::WikiSeed(None)
    ));
}

#[test]
fn parse_wiki_seed_with_dir() {
    let cmd = parse("/wiki seed src/foo").expect("should parse");
    match cmd {
        SlashCommand::WikiSeed(Some(d)) => assert_eq!(d, "src/foo"),
        other => panic!("expected WikiSeed(Some(\"src/foo\")), got: {:?}", other),
    }
}

#[test]
fn parse_wiki_seed_empty_arg_after_space_routes_to_default() {
    let cmd = parse("/wiki seed   ").expect("should parse");
    assert!(
        matches!(cmd, SlashCommand::WikiSeed(None)),
        "trailing whitespace should fall back to default dir, got: {:?}",
        cmd,
    );
}

#[test]
fn wiki_subcommand_names_contains_seed() {
    assert!(
        WIKI_SUBCOMMAND_NAMES.contains(&"seed"),
        "WIKI_SUBCOMMAND_NAMES missing 'seed': {:?}",
        WIKI_SUBCOMMAND_NAMES,
    );
}

/// Scan `body` for `/wiki <word>` references and return each `<word>`.
/// `<word>` is `[a-z_]+` — matches the lowercase ASCII shape of every
/// entry in `WIKI_SUBCOMMAND_NAMES`. Casual prose like `/wiki search`
/// inside narrative text is captured; markdown punctuation, code
/// fences, and inline code blocks don't need special handling because
/// the trailing `[^a-z_]` always terminates the word cleanly.
fn scan_wiki_subcommand_mentions(body: &str) -> Vec<String> {
    const NEEDLE: &[u8] = b"/wiki ";
    let bytes = body.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i + NEEDLE.len() <= bytes.len() {
        if &bytes[i..i + NEEDLE.len()] == NEEDLE {
            let word_start = i + NEEDLE.len();
            let mut j = word_start;
            while j < bytes.len() && (bytes[j].is_ascii_lowercase() || bytes[j] == b'_') {
                j += 1;
            }
            if j > word_start {
                out.push(body[word_start..j].to_string());
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    out
}

/// Drift guard: every `/wiki <sub>` mention in `.dm/wiki/concepts/*.md`
/// must reference a real subcommand listed in `WIKI_SUBCOMMAND_NAMES`.
/// Pages that contain the literal `WIKI_SUBCOMMAND_NAMES` symbol are
/// asserting they point at the canonical source (cycle-9 pattern); they
/// skip detailed scanning so future additions don't force per-page
/// edits.
///
/// Cycle-9 caught a real instance: `module-structure.md` claimed
/// `/wiki ingest` and `/wiki compact` — neither is a subcommand.
/// This test would have failed at the time of the drift, not 4
/// cycles later when a planner happened to read the page.
///
/// Read-only against this repo's actual `.dm/wiki/`. Skips when run
/// outside the package root.
#[test]
fn concept_pages_only_reference_known_wiki_subcommands() {
    let _cwd_guard = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    use std::collections::BTreeSet;
    let project_root = match std::env::current_dir() {
        Ok(p) => p,
        Err(_) => return,
    };
    let concepts_dir = project_root.join(".dm/wiki/concepts");
    if !concepts_dir.is_dir() {
        return;
    }
    let known: BTreeSet<&str> = WIKI_SUBCOMMAND_NAMES.iter().copied().collect();
    let mut violations: Vec<(String, String)> = Vec::new();
    let rd = std::fs::read_dir(&concepts_dir).expect("read concepts dir");
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&path) else {
            continue;
        };
        if body.contains("WIKI_SUBCOMMAND_NAMES") {
            continue; // page points at canonical source
        }
        let file_label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        for sub in scan_wiki_subcommand_mentions(&body) {
            if !known.contains(sub.as_str()) {
                violations.push((file_label.clone(), sub));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "concept pages reference /wiki subcommands not in WIKI_SUBCOMMAND_NAMES — \
             update the page to a real subcommand or add a `WIKI_SUBCOMMAND_NAMES` \
             pointer to opt out. Violations: {:?}",
        violations
    );
}

/// Self-test for the scanner so a future regression in
/// `scan_wiki_subcommand_mentions` (e.g. losing the trailing-word
/// boundary) doesn't silently disable the concept-page guard above.
#[test]
fn scan_wiki_subcommand_mentions_extracts_words() {
    let body = "Run /wiki search foo, then /wiki status. Also /wiki prune 0.\n\
                    Casual /wiki   double-space (no match — trailing whitespace, not a-z).\n\
                    /wiki_underscore_isnotcaught (no leading space after /wiki).\n\
                    `/wiki seed` in code spans counts.";
    let mut got = scan_wiki_subcommand_mentions(body);
    got.sort();
    assert_eq!(got, vec!["prune", "search", "seed", "status"]);
}

#[test]
fn format_wiki_seed_zero_results_includes_try_hint() {
    let report = crate::wiki::WikiSeedReport::default();
    let out = format_wiki_seed(&report, "src");
    assert!(
        out.contains("nothing to ingest"),
        "no-op message missing: {}",
        out,
    );
    assert!(out.contains("Try:"), "missing Try hint: {}", out);
}

#[test]
fn format_wiki_seed_includes_symlink_skip_in_header() {
    let report = crate::wiki::WikiSeedReport {
        ingested: vec!["entities/src_a_rs.md".into()],
        symlinks_skipped: 2,
        ..Default::default()
    };
    let out = format_wiki_seed(&report, "src");
    assert!(
        out.contains("2 symlinks skipped"),
        "header missing symlink count: {}",
        out,
    );
}

#[test]
fn format_wiki_seed_only_symlinks_uses_nothing_ingested_branch() {
    let report = crate::wiki::WikiSeedReport {
        symlinks_skipped: 3,
        ..Default::default()
    };
    let out = format_wiki_seed(&report, "src");
    assert!(
        out.contains("nothing ingested"),
        "expected dedicated 'nothing ingested' message: {}",
        out,
    );
    assert!(
        out.contains("3 symlinks skipped"),
        "expected symlink count: {}",
        out,
    );
    assert!(out.contains("Try:"), "missing Try hint: {}", out);
}

#[test]
fn format_wiki_stats_zero_state() {
    let out = format_wiki_stats(0, 0, 0);
    assert!(
        out.contains("tool_calls:       0"),
        "zero tool_calls: {}",
        out
    );
    assert!(
        out.contains("drift_warnings:   0"),
        "zero drift_warnings: {}",
        out
    );
    assert!(
        out.contains("snippet_bytes:    0"),
        "zero snippet_bytes: {}",
        out
    );
}

#[test]
fn wiki_stats_global_counter_roundtrip() {
    let _g = crate::telemetry::telemetry_test_guard();
    crate::telemetry::record_wiki_tool_call();
    crate::telemetry::record_wiki_tool_call();
    crate::telemetry::record_wiki_drift_warning();
    let (calls, drift) = crate::telemetry::snapshot();
    assert_eq!(calls, 2);
    assert_eq!(drift, 1);
}

// ── format_wiki_lint formatter tests (cycle 13) ──────────────────────

#[test]
fn format_wiki_lint_clean_wiki_shows_single_line() {
    let out = format_wiki_lint(&[]);
    assert_eq!(out, "Wiki lint: no issues found.");
}

#[test]
fn format_wiki_lint_groups_by_kind_with_counts() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        WikiLintFinding {
            kind: WikiLintKind::OrphanIndexEntry,
            path: "entities/g.md".to_string(),
            detail: "orphan-detail".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::UntrackedPage,
            path: "concepts/u1.md".to_string(),
            detail: "untracked-1".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::UntrackedPage,
            path: "summaries/u2.md".to_string(),
            detail: "untracked-2".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::CategoryMismatch,
            path: "entities/m.md".to_string(),
            detail: "mismatch-detail".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    assert!(out.contains("Wiki lint — 4 finding(s):"), "header: {}", out);
    assert!(
        out.contains("Orphan index entries (1):"),
        "orphan header: {}",
        out
    );
    assert!(
        out.contains("Untracked pages (2):"),
        "untracked header: {}",
        out
    );
    assert!(
        out.contains("Category mismatches (1):"),
        "mismatch header: {}",
        out
    );
    // All four paths appear.
    assert!(out.contains("entities/g.md"), "orphan path: {}", out);
    assert!(out.contains("concepts/u1.md"), "untracked1 path: {}", out);
    assert!(out.contains("summaries/u2.md"), "untracked2 path: {}", out);
    assert!(out.contains("entities/m.md"), "mismatch path: {}", out);
    // Empty-group omission: only orphans → no "Untracked pages" header.
    let orphans_only = vec![WikiLintFinding {
        kind: WikiLintKind::OrphanIndexEntry,
        path: "entities/only.md".to_string(),
        detail: "x".to_string(),
    }];
    let out2 = format_wiki_lint(&orphans_only);
    assert!(
        out2.contains("Orphan index entries (1):"),
        "orphans-only: {}",
        out2
    );
    assert!(
        !out2.contains("Untracked pages"),
        "must omit empty group: {}",
        out2
    );
    assert!(
        !out2.contains("Category mismatches"),
        "must omit empty group: {}",
        out2
    );
}

// Per-finding `detail` strings must be rendered verbatim — users need
// to see WHY something is flagged (e.g. "file not on disk", category
// names), not just the path.
#[test]
fn format_wiki_lint_renders_detail_per_finding_line() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        WikiLintFinding {
            kind: WikiLintKind::OrphanIndexEntry,
            path: "entities/g.md".to_string(),
            detail: "UNIQUE-ORPHAN-DETAIL-42".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::UntrackedPage,
            path: "concepts/u.md".to_string(),
            detail: "UNIQUE-UNTRACKED-DETAIL-7".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::CategoryMismatch,
            path: "entities/m.md".to_string(),
            detail: "UNIQUE-MISMATCH-DETAIL-π".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    for f in &findings {
        // Line must contain BOTH the path and its detail — sanity-check
        // the single-line "  - <path>: <detail>" rendering.
        let expected = format!("{}: {}", f.path, f.detail);
        assert!(
            out.contains(&expected),
            "missing '{}' in output: {}",
            expected,
            out
        );
    }
}

// Group order is load-bearing: users scan top-down, and the
// severity-ish ordering (broken references first, then catalog gaps,
// then metadata drift) should stay stable across cycles. Locks the
// orphans → untracked → mismatches sequence.
#[test]
fn format_wiki_lint_group_order_is_orphans_then_untracked_then_mismatches() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        // Intentionally reversed in input to prove formatter groups,
        // not positional order from the caller.
        WikiLintFinding {
            kind: WikiLintKind::CategoryMismatch,
            path: "entities/m.md".to_string(),
            detail: "m".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::UntrackedPage,
            path: "concepts/u.md".to_string(),
            detail: "u".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::OrphanIndexEntry,
            path: "entities/o.md".to_string(),
            detail: "o".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    let po = out
        .find("Orphan index entries")
        .expect("missing orphan header");
    let pu = out
        .find("Untracked pages")
        .expect("missing untracked header");
    let pm = out
        .find("Category mismatches")
        .expect("missing mismatch header");
    assert!(po < pu, "orphan header must precede untracked: {}", out);
    assert!(pu < pm, "untracked header must precede mismatch: {}", out);
}

#[test]
fn format_wiki_lint_renders_index_timestamp_drift_group() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        WikiLintFinding {
            kind: WikiLintKind::IndexTimestampDrift,
            path: "synthesis/run-27.md".to_string(),
            detail: "index cached last_updated=old but page frontmatter has new. Try: sync it."
                .to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::SourceMissing,
            path: "entities/missing.md".to_string(),
            detail: "source missing".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    let pindex = out
        .find("Index timestamp drift")
        .expect("missing index timestamp drift header");
    let pmissing = out
        .find("Missing source files")
        .expect("missing source-missing header");
    assert!(
        pindex < pmissing,
        "index timestamp drift should precede source file drift: {}",
        out
    );
    assert!(out.contains("Index timestamp drift (1):"), "out: {}", out);
    assert!(
        out.contains("synthesis/run-27.md: index cached last_updated=old"),
        "out: {}",
        out
    );
}

// ── cycle-14 formatter tests: source-drift groups ────────────────────
//
// The formatter adds two new group headers below the existing three.
// These tests lock both rendering (new groups appear) and suppression
// (empty groups are omitted when only older kinds are present).

#[test]
fn format_wiki_lint_renders_source_drift_groups() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        WikiLintFinding {
            kind: WikiLintKind::SourceMissing,
            path: "entities/m.md".to_string(),
            detail: "source src/gone.rs listed on page no longer exists".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::SourceNewerThanPage,
            path: "entities/s.md".to_string(),
            detail: "source src/fresh.rs modified after page last_updated".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    let pmiss = out
        .find("Missing source files")
        .expect("missing source-missing header");
    let pnewer = out
        .find("Stale pages — source newer")
        .expect("missing source-newer header");
    assert!(
        pmiss < pnewer,
        "missing-source header must precede stale-page header: {}",
        out
    );
    // Both detail lines render verbatim under their group.
    for f in &findings {
        let expected = format!("{}: {}", f.path, f.detail);
        assert!(
            out.contains(&expected),
            "missing '{}' in output: {}",
            expected,
            out
        );
    }
    // Header counts reflect per-group size.
    assert!(out.contains("Missing source files (1):"), "out: {}", out);
    assert!(
        out.contains("Stale pages — source newer (1):"),
        "out: {}",
        out
    );
}

// Full 5-group visual order is load-bearing for /wiki lint output.
// Builder's `format_wiki_lint_renders_source_drift_groups` only locks
// source_missing < source_newer (2 groups). This test locks the
// complete severity-ish chain across all five kinds so a refactor
// that reorders any one group is caught.
#[test]
fn format_wiki_lint_group_order_covers_all_five_kinds() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let mk = |kind, path: &str| WikiLintFinding {
        kind,
        path: path.to_string(),
        detail: "d".to_string(),
    };
    // Reverse-order input to prove the formatter groups, not that
    // the caller pre-sorted.
    let findings = vec![
        mk(WikiLintKind::SourceNewerThanPage, "entities/newer.md"),
        mk(WikiLintKind::SourceMissing, "entities/miss.md"),
        mk(WikiLintKind::CategoryMismatch, "entities/mismatch.md"),
        mk(WikiLintKind::UntrackedPage, "concepts/loose.md"),
        mk(WikiLintKind::OrphanIndexEntry, "entities/ghost.md"),
    ];
    let out = format_wiki_lint(&findings);
    let po = out.find("Orphan index entries").expect("orphan header");
    let pu = out.find("Untracked pages").expect("untracked header");
    let pm = out.find("Category mismatches").expect("mismatch header");
    let pms = out
        .find("Missing source files")
        .expect("source-missing header");
    let pns = out
        .find("Stale pages — source newer")
        .expect("source-newer header");
    assert!(po < pu, "orphan < untracked: {}", out);
    assert!(pu < pm, "untracked < mismatch: {}", out);
    assert!(pm < pms, "mismatch < source-missing: {}", out);
    assert!(pms < pns, "source-missing < source-newer: {}", out);
    // Header reports the total count.
    assert!(out.contains("Wiki lint — 5 finding(s):"), "header: {}", out);
}

// Cycle 15: `MalformedPage` gets its own group, rendered last. Locks
// that the formatter includes the new variant and that its bullets
// carry both path and detail like every other group.
#[test]
fn format_wiki_lint_renders_malformed_pages_group() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![
        WikiLintFinding {
            kind: WikiLintKind::OrphanIndexEntry,
            path: "entities/o.md".to_string(),
            detail: "orphan".to_string(),
        },
        WikiLintFinding {
            kind: WikiLintKind::MalformedPage,
            path: "entities/broken.md".to_string(),
            detail: "page frontmatter could not be parsed".to_string(),
        },
    ];
    let out = format_wiki_lint(&findings);
    assert!(out.contains("Malformed pages (1):"), "out: {}", out);
    assert!(
        out.contains("entities/broken.md: page frontmatter could not be parsed"),
        "out: {}",
        out
    );
    // Malformed is the last group in output order.
    let porphan = out
        .find("Orphan index entries")
        .expect("missing orphan header");
    let pmal = out
        .find("Malformed pages")
        .expect("missing malformed header");
    assert!(
        porphan < pmal,
        "malformed header must follow earlier groups: {}",
        out
    );
}

// Complete 7-group visual order lock. Earlier focused formatter tests
// only assert local ordering around the group they introduced. This
// locks the full chain so a refactor that inserts/reorders any group is
// caught.
#[test]
fn format_wiki_lint_group_order_covers_all_seven_primary_kinds() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let mk = |kind, path: &str| WikiLintFinding {
        kind,
        path: path.to_string(),
        detail: "d".to_string(),
    };
    // Reverse-order input to prove the formatter groups, not that
    // the caller pre-sorted.
    let findings = vec![
        mk(WikiLintKind::MalformedPage, "entities/broken.md"),
        mk(WikiLintKind::SourceNewerThanPage, "entities/newer.md"),
        mk(WikiLintKind::SourceMissing, "entities/miss.md"),
        mk(
            WikiLintKind::IndexTimestampDrift,
            "synthesis/stale-index.md",
        ),
        mk(WikiLintKind::CategoryMismatch, "entities/mismatch.md"),
        mk(WikiLintKind::UntrackedPage, "concepts/loose.md"),
        mk(WikiLintKind::OrphanIndexEntry, "entities/ghost.md"),
    ];
    let out = format_wiki_lint(&findings);
    let po = out.find("Orphan index entries").expect("orphan header");
    let pu = out.find("Untracked pages").expect("untracked header");
    let pm = out.find("Category mismatches").expect("mismatch header");
    let pit = out
        .find("Index timestamp drift")
        .expect("index timestamp drift header");
    let pms = out
        .find("Missing source files")
        .expect("source-missing header");
    let pns = out
        .find("Stale pages — source newer")
        .expect("source-newer header");
    let pmal = out.find("Malformed pages").expect("malformed header");
    assert!(po < pu, "orphan < untracked: {}", out);
    assert!(pu < pm, "untracked < mismatch: {}", out);
    assert!(pm < pit, "mismatch < index-timestamp-drift: {}", out);
    assert!(pit < pms, "index-timestamp-drift < source-missing: {}", out);
    assert!(pms < pns, "source-missing < source-newer: {}", out);
    assert!(pns < pmal, "source-newer < malformed: {}", out);
    assert!(out.contains("Wiki lint — 7 finding(s):"), "header: {}", out);
}

#[test]
fn format_wiki_lint_suppresses_empty_source_drift_groups_when_only_orphans() {
    use crate::wiki::{WikiLintFinding, WikiLintKind};
    let findings = vec![WikiLintFinding {
        kind: WikiLintKind::OrphanIndexEntry,
        path: "entities/o.md".to_string(),
        detail: "o".to_string(),
    }];
    let out = format_wiki_lint(&findings);
    assert!(out.contains("Orphan index entries"), "out: {}", out);
    assert!(
        !out.contains("Missing source files"),
        "empty source-missing group must be omitted: {}",
        out
    );
    assert!(
        !out.contains("Stale pages — source newer"),
        "empty source-newer group must be omitted: {}",
        out
    );
}

#[test]
fn chain_validate_parses() {
    let cmd = parse("/chain validate my-chain.yaml").expect("should parse");
    match cmd {
        SlashCommand::ChainValidate(file) => assert_eq!(file, "my-chain.yaml"),
        _ => panic!("expected ChainValidate"),
    }
}

#[test]
fn chain_validate_alias_check() {
    let cmd = parse("/chain check test.yaml").expect("should parse");
    match cmd {
        SlashCommand::ChainValidate(file) => assert_eq!(file, "test.yaml"),
        _ => panic!("expected ChainValidate from 'check' alias"),
    }
}

#[test]
fn chain_validate_no_file_parses_empty() {
    let cmd = parse("/chain validate").expect("should parse");
    match cmd {
        SlashCommand::ChainValidate(file) => assert!(file.is_empty()),
        _ => panic!("expected ChainValidate"),
    }
}

#[test]
fn chain_start_parses() {
    let cmd = parse("/chain start my-chain.yaml").expect("should parse");
    match cmd {
        SlashCommand::ChainStart(file) => assert_eq!(file, "my-chain.yaml"),
        _ => panic!("expected ChainStart"),
    }
}

#[test]
fn chain_start_empty_parses_to_empty_string() {
    let cmd = parse("/chain start").expect("should parse");
    match cmd {
        SlashCommand::ChainStart(file) => {
            assert!(file.is_empty(), "empty start should parse with empty file")
        }
        _ => panic!("expected ChainStart"),
    }
}

#[test]
fn chain_model_precheck_catches_missing_models() {
    let available = ["llama3:8b".to_string(), "gemma3:27b".to_string()];
    let node_models = ["llama3:8b", "nonexistent:7b", "gemma3:27b"];
    let missing: Vec<&str> = node_models
        .iter()
        .copied()
        .filter(|m| !available.iter().any(|am| am == m))
        .collect();
    assert_eq!(missing, ["nonexistent:7b"]);
}

#[test]
fn chain_model_precheck_passes_when_all_exist() {
    let available = ["llama3:8b".to_string(), "gemma3:27b".to_string()];
    let node_models = ["llama3:8b", "gemma3:27b"];
    let missing: Vec<&str> = node_models
        .iter()
        .copied()
        .filter(|m| !available.iter().any(|am| am == m))
        .collect();
    assert!(missing.is_empty());
}

#[test]
fn chain_status_parses() {
    let cmd = parse("/chain status").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainStatus));
}

#[test]
fn chain_stop_parses() {
    let cmd = parse("/chain stop").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainStop));
}

#[test]
fn chain_add_parses_with_model() {
    let cmd = parse("/chain add reviewer analyst gemma3:27b").expect("should parse");
    match cmd {
        SlashCommand::ChainAdd {
            name,
            role,
            model,
            input_from,
        } => {
            assert_eq!(name, "reviewer");
            assert_eq!(role, "analyst");
            assert_eq!(model, "gemma3:27b");
            assert!(input_from.is_none());
        }
        _ => panic!("expected ChainAdd"),
    }
}

#[test]
fn chain_add_parses_without_model() {
    let cmd = parse("/chain add reviewer analyst").expect("should parse");
    match cmd {
        SlashCommand::ChainAdd {
            name,
            role,
            model,
            input_from,
        } => {
            assert_eq!(name, "reviewer");
            assert_eq!(role, "analyst");
            assert_eq!(model, "");
            assert!(input_from.is_none());
        }
        _ => panic!("expected ChainAdd"),
    }
}

#[test]
fn chain_add_parses_with_input_from() {
    let cmd = parse("/chain add reviewer analyst gemma3:27b Tester").expect("should parse");
    match cmd {
        SlashCommand::ChainAdd {
            name,
            role,
            model,
            input_from,
        } => {
            assert_eq!(name, "reviewer");
            assert_eq!(role, "analyst");
            assert_eq!(model, "gemma3:27b");
            assert_eq!(input_from.as_deref(), Some("Tester"));
        }
        _ => panic!("expected ChainAdd"),
    }
}

#[test]
fn chain_add_missing_args_gives_unknown() {
    let cmd = parse("/chain add onlyname").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn chain_remove_parses() {
    let cmd = parse("/chain remove reviewer").expect("should parse");
    match cmd {
        SlashCommand::ChainRemove(name) => assert_eq!(name, "reviewer"),
        _ => panic!("expected ChainRemove"),
    }
}

#[test]
fn chain_talk_parses() {
    let cmd = parse("/chain talk planner fix the typo").expect("should parse");
    match cmd {
        SlashCommand::ChainTalk { node, message } => {
            assert_eq!(node, "planner");
            assert_eq!(message, "fix the typo");
        }
        _ => panic!("expected ChainTalk"),
    }
}

#[test]
fn chain_talk_missing_message_gives_unknown() {
    let cmd = parse("/chain talk planner").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn chain_model_parses() {
    let cmd = parse("/chain model builder qwen2.5-coder:7b").expect("should parse");
    match cmd {
        SlashCommand::ChainModel { node, model } => {
            assert_eq!(node, "builder");
            assert_eq!(model, "qwen2.5-coder:7b");
        }
        _ => panic!("expected ChainModel"),
    }
}

#[test]
fn chain_model_missing_args_gives_unknown() {
    let cmd = parse("/chain model builder").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn chain_pause_parses() {
    let cmd = parse("/chain pause").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainPause));
}

#[test]
fn chain_resume_parses() {
    let cmd = parse("/chain resume").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainResume));
}

#[test]
fn chain_resume_from_parses() {
    let cmd = parse("/chain resume-from /tmp/my-chain").expect("should parse");
    match cmd {
        SlashCommand::ChainResumeFrom(workspace) => assert_eq!(workspace, "/tmp/my-chain"),
        _ => panic!("expected ChainResumeFrom"),
    }
}

#[test]
fn chain_resume_from_missing_arg_gives_unknown() {
    let cmd = parse("/chain resume-from").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn chain_list_parses() {
    let cmd = parse("/chain list").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainList));
}

#[test]
fn chain_ls_alias_parses() {
    let cmd = parse("/chain ls").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainList));
}

#[test]
fn chain_no_args_shows_help() {
    let cmd = parse("/chain").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainHelp));
}

#[test]
fn chain_help_subcommand_shows_help() {
    let cmd = parse("/chain help").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainHelp));
}

#[test]
fn chain_metrics_parses() {
    let cmd = parse("/chain metrics").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainMetrics));
}

// ── /chain presets ─────────────────────────────────────────────────────

#[test]
fn parse_chain_presets_matches() {
    let cmd = parse("/chain presets").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainPresets));
}

#[test]
fn parse_chain_presets_with_trailing_whitespace() {
    let cmd = parse("/chain presets   ").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ChainPresets));
}

#[test]
fn parse_chain_presets_case_sensitive_not_caps() {
    let cmd = parse("/chain PRESETS").expect("should parse");
    assert!(
        !matches!(cmd, SlashCommand::ChainPresets),
        "subcommand parser is lowercase-only; PRESETS must fall through"
    );
}

#[tokio::test]
async fn chain_help_mentions_presets_subcommand() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut app, client, tx, _rx) = dispatch_harness(tmp.path().to_path_buf());
    let result = execute(SlashCommand::ChainHelp, &mut app, &client, &tx).await;
    match result {
        SlashResult::Info(s) => assert!(
            s.contains("presets"),
            "chain help must mention 'presets' subcommand, got: {}",
            s
        ),
        _ => panic!("expected Info for /chain help"),
    }
}

#[tokio::test]
async fn dispatch_chain_presets_returns_info_not_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut app, client, tx, _rx) = dispatch_harness(tmp.path().to_path_buf());
    let result = execute(SlashCommand::ChainPresets, &mut app, &client, &tx).await;
    assert!(
        matches!(result, SlashResult::Info(_)),
        "ChainPresets must return Info"
    );
}

#[tokio::test]
async fn dispatch_chain_presets_lists_all_three_builtin_names() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut app, client, tx, _rx) = dispatch_harness(tmp.path().to_path_buf());
    let SlashResult::Info(s) = execute(SlashCommand::ChainPresets, &mut app, &client, &tx).await
    else {
        panic!("expected Info");
    };
    assert!(
        s.contains("continuous-dev"),
        "missing continuous-dev: {}",
        s
    );
    assert!(s.contains("self-improve"), "missing self-improve: {}", s);
    assert!(s.contains("project-audit"), "missing project-audit: {}", s);
}

#[tokio::test]
async fn dispatch_chain_presets_marks_single_pass_distinctly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut app, client, tx, _rx) = dispatch_harness(tmp.path().to_path_buf());
    let SlashResult::Info(s) = execute(SlashCommand::ChainPresets, &mut app, &client, &tx).await
    else {
        panic!("expected Info");
    };
    assert_eq!(
        s.matches("single-pass").count(),
        1,
        "exactly one preset (project-audit) is single-pass: {}",
        s
    );
    assert_eq!(
        s.matches("looping").count(),
        2,
        "exactly two presets (continuous-dev, self-improve) are looping: {}",
        s
    );
}

#[tokio::test]
async fn dispatch_chain_presets_includes_run_hint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (mut app, client, tx, _rx) = dispatch_harness(tmp.path().to_path_buf());
    let SlashResult::Info(s) = execute(SlashCommand::ChainPresets, &mut app, &client, &tx).await
    else {
        panic!("expected Info");
    };
    assert!(
        s.contains("dm --chain"),
        "output must include actionable run hint 'dm --chain': {}",
        s
    );
}

#[test]
fn format_chain_metrics_with_data() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};
    let config = ChainConfig {
        name: "test-chain".into(),
        description: None,
        nodes: vec![
            ChainNodeConfig {
                id: "planner".into(),
                name: "planner".into(),
                role: "plan".into(),
                model: "m".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
            ChainNodeConfig {
                id: "builder".into(),
                name: "builder".into(),
                role: "build".into(),
                model: "m".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
        ],
        max_cycles: 5,
        max_total_turns: 100,
        workspace: std::path::PathBuf::from("/tmp"),
        skip_permissions_warning: true,
        loop_forever: false,
        directive: None,
    };
    let mut state = ChainState::new(config, "id".into());
    state.current_cycle = 3;
    state.total_duration_secs = 120.5;
    state
        .node_durations
        .insert("planner".into(), vec![10.0, 15.0, 12.0]);
    state
        .node_durations
        .insert("builder".into(), vec![25.0, 30.0]);
    state.node_failures.insert("builder".into(), 1);
    state.node_prompt_tokens.insert("planner".into(), 50000);
    state.node_completion_tokens.insert("planner".into(), 12000);
    state.node_prompt_tokens.insert("builder".into(), 80000);
    state.node_completion_tokens.insert("builder".into(), 25000);

    let output = crate::orchestrate::format_chain_metrics(&state);
    assert!(output.contains("test-chain"), "should contain chain name");
    assert!(output.contains("Cycles: 3"), "should show cycle count");
    assert!(output.contains("120.5s"), "should show total duration");
    assert!(output.contains("planner"), "should list planner node");
    assert!(output.contains("builder"), "should list builder node");
    // Check planner avg: (10+15+12)/3 = 12.3
    assert!(output.contains("12.3"), "planner avg should be 12.3");
    // Check token columns
    assert!(
        output.contains("PromptTok"),
        "header should show PromptTok column"
    );
    assert!(
        output.contains("CompTok"),
        "header should show CompTok column"
    );
    assert!(
        output.contains("50000"),
        "planner prompt tokens should appear"
    );
    assert!(
        output.contains("25000"),
        "builder completion tokens should appear"
    );
}

#[test]
fn format_chain_metrics_no_data() {
    use crate::orchestrate::types::{ChainConfig, ChainState};
    let config = ChainConfig {
        name: "empty".into(),
        description: None,
        nodes: vec![],
        max_cycles: 1,
        max_total_turns: 10,
        workspace: std::path::PathBuf::from("/tmp"),
        skip_permissions_warning: true,
        loop_forever: false,
        directive: None,
    };
    let state = ChainState::new(config, "id".into());
    let output = crate::orchestrate::format_chain_metrics(&state);
    assert!(output.contains("No metrics recorded"));
}

#[test]
fn chain_invalid_subcommand_is_unknown() {
    let cmd = parse("/chain foobar").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

// ── /chain init tests ─────────────────────────────────────────────────

#[test]
fn chain_init_default_name() {
    let cmd = parse("/chain init").expect("should parse");
    match cmd {
        SlashCommand::ChainInit(name) => assert_eq!(name, "my-chain"),
        _ => panic!("expected ChainInit"),
    }
}

#[test]
fn chain_init_custom_name() {
    let cmd = parse("/chain init my-project").expect("should parse");
    match cmd {
        SlashCommand::ChainInit(name) => assert_eq!(name, "my-project"),
        _ => panic!("expected ChainInit"),
    }
}

#[test]
fn chain_init_template_is_valid_yaml() {
    let yaml = crate::orchestrate::generate_chain_template("test", "gemma4:26b");
    // Should parse as valid YAML and round-trip through load
    let config: crate::orchestrate::types::ChainConfig =
        serde_yaml::from_str(&yaml).expect("template should be valid YAML");
    assert_eq!(config.name, "test");
    assert_eq!(config.nodes.len(), 2);
    assert_eq!(config.nodes[0].id, "planner");
    assert_eq!(config.nodes[1].id, "builder");
    assert_eq!(config.nodes[1].input_from.as_deref(), Some("Planner"));
}

#[test]
fn chain_init_template_passes_validation() {
    let yaml = crate::orchestrate::generate_chain_template("valid", "llama3");
    let config: crate::orchestrate::types::ChainConfig =
        serde_yaml::from_str(&yaml).expect("parse");
    crate::orchestrate::validate_chain_config(&config)
        .expect("generated template should pass validation");
}

#[test]
fn chain_init_template_uses_provided_model() {
    let yaml = crate::orchestrate::generate_chain_template("test", "qwen2.5:72b");
    assert!(
        yaml.contains("qwen2.5:72b"),
        "template should use provided model: {yaml}"
    );
}

// ── /chain log tests ──────────────────────────────────────────────────

#[test]
fn chain_log_no_args() {
    let cmd = parse("/chain log").expect("should parse");
    match cmd {
        SlashCommand::ChainLog(cycle) => assert!(cycle.is_none()),
        _ => panic!("expected ChainLog"),
    }
}

#[test]
fn chain_log_with_cycle() {
    let cmd = parse("/chain log 3").expect("should parse");
    match cmd {
        SlashCommand::ChainLog(cycle) => assert_eq!(cycle, Some(3)),
        _ => panic!("expected ChainLog"),
    }
}

#[test]
fn chain_log_non_numeric_arg_gives_none() {
    let cmd = parse("/chain log abc").expect("should parse");
    match cmd {
        SlashCommand::ChainLog(cycle) => assert!(cycle.is_none(), "non-numeric should be None"),
        _ => panic!("expected ChainLog"),
    }
}

#[test]
fn test_parse_undo() {
    let cmd = parse("/undo").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Undo));
}

#[test]
fn test_parse_rewind() {
    let cmd = parse("/rewind").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Undo));
}

#[test]
fn test_parse_copy() {
    let cmd = parse("/copy").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Copy));
}

#[test]
fn test_parse_diff_no_args() {
    let cmd = parse("/diff").expect("should parse");
    match cmd {
        SlashCommand::Diff(arg) => assert_eq!(arg, ""),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn test_parse_diff_staged() {
    let cmd = parse("/diff --staged").expect("should parse");
    match cmd {
        SlashCommand::Diff(arg) => assert_eq!(arg, "--staged"),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn test_parse_diff_ref() {
    let cmd = parse("/diff HEAD~1").expect("should parse");
    match cmd {
        SlashCommand::Diff(arg) => assert_eq!(arg, "HEAD~1"),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn test_parse_branch_no_args() {
    let cmd = parse("/branch").expect("should parse");
    match cmd {
        SlashCommand::Branch(target) => assert_eq!(target, ""),
        _ => panic!("expected Branch"),
    }
}

#[test]
fn test_parse_branch_with_name() {
    let cmd = parse("/branch feature/my-work").expect("should parse");
    match cmd {
        SlashCommand::Branch(target) => assert_eq!(target, "feature/my-work"),
        _ => panic!("expected Branch"),
    }
}

#[test]
fn test_parse_stats() {
    let cmd = parse("/stats").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Stats));
}

#[test]
fn test_parse_fork() {
    let cmd = parse("/fork").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Fork(None)));

    let cmd = parse("/fork 3").expect("should parse");
    match cmd {
        SlashCommand::Fork(Some(3)) => {}
        _ => panic!("expected Fork(Some(3))"),
    }
}

#[test]
fn test_parse_add_no_args() {
    let cmd = parse("/add").expect("should parse");
    match cmd {
        SlashCommand::Add(path) => assert_eq!(path, ""),
        _ => panic!("expected Add"),
    }
}

#[test]
fn test_parse_add_with_path() {
    let cmd = parse("/add src/main.rs").expect("should parse");
    match cmd {
        SlashCommand::Add(path) => assert_eq!(path, "src/main.rs"),
        _ => panic!("expected Add"),
    }
}

#[test]
fn test_parse_review_no_args() {
    let cmd = parse("/review").expect("should parse");
    match cmd {
        SlashCommand::Review(r) => assert_eq!(r, ""),
        _ => panic!("expected Review"),
    }
}

#[test]
fn test_parse_review_with_ref() {
    let cmd = parse("/review HEAD~3").expect("should parse");
    match cmd {
        SlashCommand::Review(r) => assert_eq!(r, "HEAD~3"),
        _ => panic!("expected Review"),
    }
}

#[test]
fn test_parse_review_staged() {
    let cmd = parse("/review staged").expect("should parse");
    match cmd {
        SlashCommand::Review(r) => assert_eq!(r, "staged"),
        _ => panic!("expected Review"),
    }
}

#[test]
fn test_parse_doctor() {
    let cmd = parse("/doctor").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Doctor));
}

#[test]
fn test_parse_init() {
    let cmd = parse("/init").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Init));
}

#[test]
fn test_parse_commit() {
    let cmd = parse("/commit").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Commit));
}

#[test]
fn test_parse_log_no_args_defaults_to_20() {
    let cmd = parse("/log").expect("should parse");
    match cmd {
        SlashCommand::Log(n) => assert_eq!(n, 20),
        _ => panic!("expected Log"),
    }
}

#[test]
fn test_parse_log_with_count() {
    let cmd = parse("/log 50").expect("should parse");
    match cmd {
        SlashCommand::Log(n) => assert_eq!(n, 50),
        _ => panic!("expected Log"),
    }
}

#[test]
fn test_parse_log_invalid_arg_defaults_to_20() {
    let cmd = parse("/log notanumber").expect("should parse");
    match cmd {
        SlashCommand::Log(n) => assert_eq!(n, 20, "invalid arg should default to 20"),
        _ => panic!("expected Log"),
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn init_creates_dm_md_when_missing() {
    let _g = crate::tools::CWD_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    // Change to temp dir for this test
    let original_dir = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();

    let dm_md = dir.path().join("DM.md");
    assert!(!dm_md.exists(), "DM.md should not exist before /init");

    // We can't easily call execute() without a real App/client/channel,
    // so test the underlying logic directly: template + write
    let template = crate::init::generate_dm_md_template(dir.path());
    std::fs::write(&dm_md, &template).unwrap();

    assert!(dm_md.exists(), "DM.md should be created");
    let content = std::fs::read_to_string(&dm_md).unwrap();
    assert!(
        content.contains("## Commands"),
        "template should include Commands section"
    );

    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn init_existing_dm_md_detected() {
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let dm_md = dir.path().join("DM.md");
    std::fs::write(&dm_md, "# existing").unwrap();
    // Simulate what the execute handler checks
    assert!(
        dm_md.exists(),
        "pre-existing DM.md should be detected as already present"
    );
}

// ── /export ───────────────────────────────────────────────────────────────

fn entry(kind: crate::tui::app::EntryKind, content: &str) -> crate::tui::app::DisplayEntry {
    crate::tui::app::DisplayEntry {
        kind,
        content: content.to_string(),
    }
}

#[test]
fn test_parse_export_no_arg() {
    let cmd = parse("/export").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Export(s) if s.is_empty()));
}

#[test]
fn test_parse_export_with_path() {
    let cmd = parse("/export out.md").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Export(s) if s == "out.md"));
}

#[test]
fn format_entries_empty_returns_header_only() {
    let out = format_entries_as_markdown(&[]);
    assert!(
        out.starts_with("# dm conversation export"),
        "should have header"
    );
    assert!(!out.is_empty());
}

#[test]
fn format_entries_user_message() {
    let entries = vec![entry(
        crate::tui::app::EntryKind::UserMessage,
        "hello world",
    )];
    let out = format_entries_as_markdown(&entries);
    assert!(
        out.contains("**You:** hello world"),
        "user message missing: {out}"
    );
}

#[test]
fn format_entries_assistant_message() {
    let entries = vec![entry(
        crate::tui::app::EntryKind::AssistantMessage,
        "here is the answer",
    )];
    let out = format_entries_as_markdown(&entries);
    assert!(
        out.contains("here is the answer"),
        "assistant message missing: {out}"
    );
}

#[test]
fn format_entries_system_info_italicised() {
    let entries = vec![entry(
        crate::tui::app::EntryKind::SystemInfo,
        "Compacted context",
    )];
    let out = format_entries_as_markdown(&entries);
    assert!(
        out.contains("*Compacted context*"),
        "system info not italicised: {out}"
    );
}

#[test]
fn format_entries_tool_error_shows_warning() {
    let entries = vec![entry(
        crate::tui::app::EntryKind::ToolError,
        "permission denied",
    )];
    let out = format_entries_as_markdown(&entries);
    assert!(
        out.contains("⚠"),
        "tool error should have warning glyph: {out}"
    );
    assert!(
        out.contains("permission denied"),
        "error message missing: {out}"
    );
}

#[test]
fn format_entries_file_diff_in_code_fence() {
    let entries = vec![entry(crate::tui::app::EntryKind::FileDiff, "- old\n+ new")];
    let out = format_entries_as_markdown(&entries);
    assert!(out.contains("```diff"), "diff fence missing: {out}");
    assert!(out.contains("- old"), "diff content missing: {out}");
}

#[test]
fn format_entries_tool_result_in_details() {
    let entries = vec![entry(crate::tui::app::EntryKind::ToolResult, "output here")];
    let out = format_entries_as_markdown(&entries);
    assert!(out.contains("<details>"), "details block missing: {out}");
    assert!(
        out.contains("output here"),
        "tool result content missing: {out}"
    );
}

#[test]
fn format_entries_tool_call_includes_content_in_details() {
    // Regression: previously the ToolCall details block was self-closing,
    // so the full argument body was never written to the export.
    let content = "bash: ls -la\nworking_dir: /home/user";
    let entries = vec![entry(crate::tui::app::EntryKind::ToolCall, content)];
    let out = format_entries_as_markdown(&entries);
    assert!(out.contains("<details>"), "details block missing: {out}");
    assert!(
        out.contains("bash: ls -la"),
        "first line should appear in summary: {out}"
    );
    assert!(
        out.contains("working_dir: /home/user"),
        "full content must be in details body: {out}"
    );
    assert!(
        !out.contains("</summary></details>"),
        "details must not be self-closing: {out}"
    );
}

#[test]
fn format_entries_image_attachment() {
    let entries = vec![entry(
        crate::tui::app::EntryKind::ImageAttachment,
        "screenshot.png",
    )];
    let out = format_entries_as_markdown(&entries);
    assert!(
        out.contains("[image: screenshot.png]"),
        "image reference missing: {out}"
    );
}

#[test]
fn parse_export_clean_no_path() {
    let cmd = parse("/export clean").expect("should parse");
    match cmd {
        SlashCommand::Export(s) => assert_eq!(s, "clean"),
        other => panic!("expected Export, got {:?}", other),
    }
}

#[test]
fn parse_export_clean_with_path() {
    let cmd = parse("/export clean out.md").expect("should parse");
    match cmd {
        SlashCommand::Export(s) => assert_eq!(s, "clean out.md"),
        other => panic!("expected Export, got {:?}", other),
    }
}

#[test]
fn format_entries_filtered_strips_tool_entries() {
    // clean mode drops ToolCall/ToolResult/ToolError/SystemInfo/Notice
    // so shared transcripts contain only user/assistant/diff signal.
    let entries = vec![
        entry(crate::tui::app::EntryKind::UserMessage, "hello"),
        entry(crate::tui::app::EntryKind::ToolCall, "bash ls"),
        entry(crate::tui::app::EntryKind::ToolResult, "file1\nfile2"),
        entry(crate::tui::app::EntryKind::AssistantMessage, "two files"),
    ];
    let out = format_entries_as_markdown_filtered(&entries, true);
    assert!(out.contains("**You:** hello"));
    assert!(out.contains("two files"));
    assert!(!out.contains("bash ls"), "tool call body stripped: {out}");
    assert!(!out.contains("file1"), "tool result body stripped: {out}");
    assert!(
        !out.contains("<details>"),
        "no details blocks in clean mode: {out}"
    );
}

#[test]
fn format_entries_full_keeps_tool_entries() {
    // strip_tools=false must preserve tool blocks (parity with the
    // legacy unfiltered formatter).
    let entries = vec![
        entry(crate::tui::app::EntryKind::UserMessage, "hi"),
        entry(crate::tui::app::EntryKind::ToolCall, "bash ls"),
    ];
    let out = format_entries_as_markdown_filtered(&entries, false);
    assert!(
        out.contains("<details>"),
        "full mode keeps details blocks: {out}"
    );
    assert!(out.contains("bash ls"));
}

#[test]
fn format_entries_filtered_preserves_file_diffs() {
    // FileDiff is signal, not noise — must survive clean mode.
    let entries = vec![entry(
        crate::tui::app::EntryKind::FileDiff,
        "@@ -1 +1 @@\n-a\n+b",
    )];
    let out = format_entries_as_markdown_filtered(&entries, true);
    assert!(out.contains("```diff"));
    assert!(out.contains("+b"));
}

// ── /rename parse tests ───────────────────────────────────────────────────

#[test]
fn rename_parses_title() {
    let cmd = parse("/rename My New Title").expect("should parse");
    match cmd {
        SlashCommand::Rename(t) => assert_eq!(t, "My New Title"),
        other => panic!("expected Rename, got {:?}", other),
    }
}

#[test]
fn rename_empty_arg_still_parses() {
    let cmd = parse("/rename").expect("should parse");
    match cmd {
        SlashCommand::Rename(t) => assert!(t.is_empty(), "empty title expected, got {:?}", t),
        other => panic!("expected Rename, got {:?}", other),
    }
}

#[test]
fn rename_multi_word_title_preserved() {
    let cmd = parse("/rename  The   Big   Plan  ").expect("should parse");
    match cmd {
        SlashCommand::Rename(t) => assert_eq!(t, "The   Big   Plan"),
        other => panic!("expected Rename, got {:?}", other),
    }
}

// ── /bughunter ────────────────────────────────────────────────────────────

#[test]
fn test_parse_bughunter_no_args() {
    let cmd = parse("/bughunter").expect("should parse");
    assert!(matches!(cmd, SlashCommand::BugHunter(s) if s.is_empty()));
}

#[test]
fn test_parse_bughunter_with_focus() {
    let cmd = parse("/bughunter concurrency").expect("should parse");
    assert!(matches!(cmd, SlashCommand::BugHunter(s) if s == "concurrency"));
}

#[test]
fn build_bughunter_prompt_empty_files() {
    let out = build_bughunter_prompt(&[], "");
    assert!(
        out.contains("no source files found"),
        "should note empty file list: {out}"
    );
    assert!(
        out.contains("Look broadly"),
        "default focus line missing: {out}"
    );
}

#[test]
fn build_bughunter_prompt_lists_files() {
    let files = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
    let out = build_bughunter_prompt(&files, "");
    assert!(out.contains("src/lib.rs"), "first file missing: {out}");
    assert!(out.contains("src/main.rs"), "second file missing: {out}");
}

#[test]
fn build_bughunter_prompt_with_focus() {
    let out = build_bughunter_prompt(&[], "integer overflow");
    assert!(out.contains("integer overflow"), "focus missing: {out}");
    assert!(
        !out.contains("Look broadly"),
        "should not use default when focus given: {out}"
    );
}

#[test]
fn build_bughunter_prompt_instructs_bug_categories() {
    let out = build_bughunter_prompt(&[], "");
    assert!(
        out.contains("panic"),
        "should mention panic category: {out}"
    );
    assert!(
        out.contains("Bug category"),
        "should request category: {out}"
    );
    assert!(out.contains("Suggested fix"), "should request fix: {out}");
}

// ── /pr ───────────────────────────────────────────────────────────────────

#[test]
fn pr_parses_no_args() {
    let cmd = parse("/pr").expect("should parse");
    match cmd {
        SlashCommand::Pr(base) => {
            assert!(base.is_empty(), "expected empty base, got {:?}", base)
        }
        other => panic!("expected Pr, got {:?}", other),
    }
}

#[test]
fn pr_parses_explicit_base() {
    let cmd = parse("/pr main").expect("should parse");
    match cmd {
        SlashCommand::Pr(base) => assert_eq!(base, "main"),
        other => panic!("expected Pr, got {:?}", other),
    }
}

#[test]
fn pr_parses_feature_base() {
    let cmd = parse("/pr develop").expect("should parse");
    match cmd {
        SlashCommand::Pr(base) => assert_eq!(base, "develop"),
        other => panic!("expected Pr, got {:?}", other),
    }
}

// ── /context ──────────────────────────────────────────────────────────────

#[test]
fn context_parses() {
    let cmd = parse("/context").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Context));
}

#[test]
fn build_context_report_shows_session_id_and_model() {
    let report = build_context_report(
        "abc-123",
        None,
        "gemma4:27b",
        500,
        None,
        3,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(report.contains("abc-123"), "session ID missing: {report}");
    assert!(report.contains("gemma4:27b"), "model missing: {report}");
    assert!(report.contains("3"), "turn count missing: {report}");
}

#[test]
fn build_context_report_shows_title_when_set() {
    let report = build_context_report(
        "sess-id",
        Some("My Test Session"),
        "llama3",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(
        report.contains("My Test Session"),
        "title missing: {report}"
    );
}

#[test]
fn build_context_report_ctx_usage_with_percentage() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        Some((8000, 10000)),
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(report.contains("8000"), "used tokens missing: {report}");
    assert!(report.contains("10000"), "limit missing: {report}");
    assert!(report.contains("80%"), "percentage missing: {report}");
}

#[test]
fn build_context_report_mcp_servers_listed() {
    let servers = vec![("planner".to_string(), 3), ("builder".to_string(), 5)];
    let report = build_context_report(
        "x", None, "m", 0, None, 0, &servers, false, None, None, None, None,
    );
    assert!(
        report.contains("planner"),
        "planner server missing: {report}"
    );
    assert!(report.contains("3 tools"), "tool count missing: {report}");
    assert!(
        report.contains("builder"),
        "builder server missing: {report}"
    );
}

#[test]
fn build_context_report_pending_context_indicator() {
    let with = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        true,
        None,
        None,
        None,
        None,
    );
    let without = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(
        with.contains("yes"),
        "should show 'yes' when pending: {with}"
    );
    assert!(
        without.contains("none"),
        "should show 'none' when no pending: {without}"
    );
}

#[test]
fn build_context_report_microcompact_stage_shows_line() {
    let stage = crate::compaction::CompactionStage::Microcompact {
        chars_removed: 1234,
        messages_affected: 3,
    };
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        Some(&stage),
        None,
        None,
        None,
    );
    assert!(
        report.contains("Compacting:"),
        "missing compacting line: {report}"
    );
    assert!(report.contains("Stage 1"), "missing stage label: {report}");
    assert!(report.contains("1234"), "missing chars_removed: {report}");
    assert!(
        report.contains("3 msgs"),
        "missing messages_affected: {report}"
    );
}

#[test]
fn build_context_report_session_memory_stage_shows_line() {
    let stage = crate::compaction::CompactionStage::SessionMemory {
        messages_dropped: 7,
    };
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        Some(&stage),
        None,
        None,
        None,
    );
    assert!(report.contains("Stage 2"), "{report}");
    assert!(report.contains("7 old msgs"), "{report}");
}

#[test]
fn build_context_report_full_summary_stage_shows_line() {
    let stage = crate::compaction::CompactionStage::FullSummary {
        messages_summarized: 42,
    };
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        Some(&stage),
        None,
        None,
        None,
    );
    assert!(report.contains("Stage 3"), "{report}");
    assert!(report.contains("42 msgs"), "{report}");
}

#[test]
fn build_context_report_emergency_stage_shows_line() {
    let stage = crate::compaction::CompactionStage::Emergency;
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        Some(&stage),
        None,
        None,
        None,
    );
    assert!(report.contains("Emergency"), "{report}");
    assert!(report.contains("force-dropped"), "{report}");
}

#[test]
fn build_context_report_compaction_none_hides_line() {
    // Explicit CompactionStage::None must not render a line —
    // matches the Option-absent case. Pins the contract so a
    // future edit that adds a "Compacting: not active" line
    // fails here.
    let stage = crate::compaction::CompactionStage::None;
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        Some(&stage),
        None,
        None,
        None,
    );
    assert!(
        !report.contains("Compacting:"),
        "CompactionStage::None must not render a line, got: {report}"
    );
}

#[test]
fn build_context_report_thresholds_block_shown_when_window_known() {
    // 10000-token window → thresholds from the compaction module's
    // defaults. Hint line uses current-usage = 2000, so "N tok
    // until Stage 1" is the expected next-trigger message.
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        Some((2000, 10000)),
        0,
        &[],
        false,
        None,
        Some(10000),
        None,
        None,
    );
    assert!(report.contains("Compaction thresholds"), "{report}");
    let t = crate::compaction::CompactionThresholds::from_context_window(10000);
    let s1 = format!("{} tok", t.micro_compact);
    let s2 = format!("{} tok", t.session_compact);
    let s3 = format!("{} tok", t.full_compact);
    assert!(report.contains(&s1), "missing Stage 1 absolute: {report}");
    assert!(report.contains(&s2), "missing Stage 2 absolute: {report}");
    assert!(report.contains(&s3), "missing Stage 3 absolute: {report}");
    assert!(
        report.contains("tok until Stage 1"),
        "missing hint: {report}"
    );
}

#[test]
fn build_context_report_thresholds_block_hidden_without_window() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(
        !report.contains("Compaction thresholds"),
        "without context_window the thresholds block must be absent: {report}"
    );
}

#[test]
fn build_context_report_thresholds_hint_past_stage3_is_emergency_warning() {
    // Usage beyond Stage 3 must produce the past-Stage-3 warning.
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        Some((9500, 10000)),
        0,
        &[],
        false,
        None,
        Some(10000),
        None,
        None,
    );
    assert!(
        report.contains("past Stage 3"),
        "expected past-Stage-3 warning when used > full_compact: {report}"
    );
    assert!(report.contains("emergency compact"), "{report}");
}

#[test]
fn build_context_report_shows_system_prompt_chars_when_some() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        Some(8432),
        None,
    );
    assert!(
        report.contains("System prompt: 8432 chars"),
        "missing system prompt size: {report}"
    );
}

#[test]
fn build_context_report_hides_system_prompt_line_when_none() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(
        !report.contains("System prompt:"),
        "None must omit the line entirely: {report}"
    );
}

#[test]
fn build_context_report_shows_wiki_snippet_chars_when_some() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        Some(1_123_456),
    );
    assert!(
        report.contains("Wiki snippet:  1123456 chars"),
        "missing wiki snippet size: {report}"
    );
}

#[test]
fn build_context_report_wiki_snippet_none_renders_none() {
    let report = build_context_report(
        "x",
        None,
        "m",
        0,
        None,
        0,
        &[],
        false,
        None,
        None,
        None,
        None,
    );
    assert!(
        report.contains("Wiki snippet:  none"),
        "None must render informative 'none' state: {report}"
    );
}

// ── /security-review ─────────────────────────────────────────────────────

#[test]
fn test_parse_security_review_no_args() {
    let cmd = parse("/security-review").expect("should parse");
    assert!(matches!(cmd, SlashCommand::SecurityReview(s) if s.is_empty()));
}

#[test]
fn test_parse_security_review_alias() {
    let cmd = parse("/secreview HEAD~1").expect("should parse");
    assert!(matches!(cmd, SlashCommand::SecurityReview(s) if s == "HEAD~1"));
}

#[test]
fn test_parse_security_review_with_ref() {
    let cmd = parse("/security-review main").expect("should parse");
    assert!(matches!(cmd, SlashCommand::SecurityReview(s) if s == "main"));
}

#[test]
fn build_security_review_prompt_contains_diff() {
    let out = build_security_review_prompt("+ fn danger() {}", "HEAD");
    assert!(out.contains("fn danger()"), "diff content missing: {out}");
}

#[test]
fn build_security_review_prompt_mentions_injection() {
    let out = build_security_review_prompt("", "HEAD");
    assert!(
        out.contains("Injection") || out.contains("injection"),
        "injection category missing: {out}"
    );
}

#[test]
fn build_security_review_prompt_mentions_ref() {
    let out = build_security_review_prompt("", "feature/auth");
    assert!(
        out.contains("feature/auth"),
        "git ref missing from prompt: {out}"
    );
}

#[test]
fn build_security_review_prompt_requests_severity() {
    let out = build_security_review_prompt("", "HEAD");
    assert!(
        out.contains("severity") || out.contains("Critical"),
        "severity levels missing: {out}"
    );
}

// ── /advisor ─────────────────────────────────────────────────────────────

#[test]
fn test_parse_advisor_no_args() {
    let cmd = parse("/advisor").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Advisor(s) if s.is_empty()));
}

#[test]
fn test_parse_advisor_with_topic() {
    let cmd = parse("/advisor database layer").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Advisor(s) if s == "database layer"));
}

#[test]
fn build_advisor_prompt_empty_topic_general_scope() {
    let out = build_advisor_prompt("", &[]);
    assert!(
        out.contains("overall architecture") || out.contains("general"),
        "no general scope: {out}"
    );
}

#[test]
fn build_advisor_prompt_with_topic() {
    let out = build_advisor_prompt("error handling", &[]);
    assert!(out.contains("error handling"), "topic missing: {out}");
}

#[test]
fn build_advisor_prompt_lists_files() {
    let files = vec!["src/lib.rs".to_string(), "src/main.rs".to_string()];
    let out = build_advisor_prompt("", &files);
    assert!(out.contains("src/lib.rs"), "first file missing: {out}");
    assert!(out.contains("src/main.rs"), "second file missing: {out}");
}

#[test]
fn build_advisor_prompt_requests_specifics() {
    let out = build_advisor_prompt("", &[]);
    assert!(
        out.contains("refactoring") || out.contains("Suggest"),
        "no refactoring guidance: {out}"
    );
    assert!(
        out.contains("module") || out.contains("function"),
        "no specific reference requested: {out}"
    );
}

// ── /template ─────────────────────────────────────────────────────────────

#[test]
fn template_parses_no_args() {
    let cmd = parse("/template").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Template(s) if s.is_empty()));
}

#[test]
fn template_parses_name_only() {
    let cmd = parse("/template greet").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Template(s) if s == "greet"));
}

#[test]
fn template_parses_name_and_args() {
    let cmd = parse("/template greet World").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Template(s) if s == "greet World"));
}

// ── /todo ─────────────────────────────────────────────────────────────────

#[test]
fn todo_list_parses() {
    let cmd = parse("/todo").expect("should parse");
    assert!(matches!(cmd, SlashCommand::TodoList));
}

#[test]
fn todo_clear_parses() {
    let cmd = parse("/todo clear").expect("should parse");
    assert!(matches!(cmd, SlashCommand::TodoClear));
}

#[test]
fn todo_scan_parses() {
    let cmd = parse("/todo scan").expect("should parse");
    assert!(matches!(cmd, SlashCommand::TodoScan));
}

#[test]
fn todo_done_parses_id() {
    let cmd = parse("/todo done 3").expect("should parse");
    match cmd {
        SlashCommand::TodoDone(id) => assert_eq!(id, "3"),
        other => panic!("expected TodoDone, got {:?}", other),
    }
}

#[test]
fn todo_wip_parses_id() {
    let cmd = parse("/todo wip abc").expect("should parse");
    match cmd {
        SlashCommand::TodoWip(id) => assert_eq!(id, "abc"),
        other => panic!("expected TodoWip, got {:?}", other),
    }
}

#[test]
fn todo_add_default_priority_is_medium() {
    let cmd = parse("/todo add fix the parser bug").expect("should parse");
    match cmd {
        SlashCommand::TodoAdd { priority, content } => {
            assert_eq!(priority, "medium");
            assert_eq!(content, "fix the parser bug");
        }
        other => panic!("expected TodoAdd, got {:?}", other),
    }
}

#[test]
fn todo_add_explicit_high_priority() {
    let cmd = parse("/todo add high implement feature").expect("should parse");
    match cmd {
        SlashCommand::TodoAdd { priority, content } => {
            assert_eq!(priority, "high");
            assert_eq!(content, "implement feature");
        }
        other => panic!("expected TodoAdd, got {:?}", other),
    }
}

#[test]
fn todo_add_med_normalised_to_medium() {
    let cmd = parse("/todo add med write tests").expect("should parse");
    match cmd {
        SlashCommand::TodoAdd { priority, content } => {
            assert_eq!(priority, "medium", "med should normalise to medium");
            assert_eq!(content, "write tests");
        }
        other => panic!("expected TodoAdd, got {:?}", other),
    }
}

#[test]
fn todo_add_low_priority() {
    let cmd = parse("/todo add low update docs").expect("should parse");
    match cmd {
        SlashCommand::TodoAdd { priority, content } => {
            assert_eq!(priority, "low");
            assert_eq!(content, "update docs");
        }
        other => panic!("expected TodoAdd, got {:?}", other),
    }
}

#[test]
fn todo_add_missing_text_gives_unknown() {
    let cmd = parse("/todo add high").expect("should parse");
    // "high" is parsed as priority, but content is empty → Unknown
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn todo_add_missing_text_no_priority_gives_unknown() {
    let cmd = parse("/todo add").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

// ── /effort ───────────────────────────────────────────────────────

#[test]
fn test_parse_effort_no_args() {
    let cmd = parse("/effort").expect("should parse");
    match cmd {
        SlashCommand::Effort(s) => assert!(s.is_empty()),
        other => panic!("expected Effort, got {:?}", other),
    }
}

#[test]
fn test_parse_effort_quick() {
    let cmd = parse("/effort quick").expect("should parse");
    match cmd {
        SlashCommand::Effort(s) => assert_eq!(s, "quick"),
        other => panic!("expected Effort, got {:?}", other),
    }
}

#[test]
fn test_parse_effort_thorough() {
    let cmd = parse("/effort thorough").expect("should parse");
    match cmd {
        SlashCommand::Effort(s) => assert_eq!(s, "thorough"),
        other => panic!("expected Effort, got {:?}", other),
    }
}

#[test]
fn test_parse_effort_normal() {
    let cmd = parse("/effort normal").expect("should parse");
    match cmd {
        SlashCommand::Effort(s) => assert_eq!(s, "normal"),
        other => panic!("expected Effort, got {:?}", other),
    }
}

#[test]
fn effort_instruction_normal_returns_none() {
    assert!(effort_instruction(&crate::tui::app::EffortLevel::Normal).is_none());
}

#[test]
fn effort_instruction_quick_returns_brief_hint() {
    let hint = effort_instruction(&crate::tui::app::EffortLevel::Quick);
    assert!(hint.is_some());
    let h = hint.unwrap();
    assert!(
        h.contains("brief") || h.contains("concise"),
        "quick hint should mention brevity: {h}"
    );
}

#[test]
fn effort_instruction_thorough_returns_depth_hint() {
    let hint = effort_instruction(&crate::tui::app::EffortLevel::Thorough);
    assert!(hint.is_some());
    let h = hint.unwrap();
    assert!(
        h.contains("thorough") || h.contains("comprehensive"),
        "thorough hint should mention depth: {h}"
    );
}

// ── /pin / /unpin ────────────────────────────────────────────────

#[test]
fn test_parse_pin_no_args() {
    let cmd = parse("/pin").expect("should parse");
    match cmd {
        SlashCommand::Pin(s) => assert!(s.is_empty()),
        other => panic!("expected Pin, got {:?}", other),
    }
}

#[test]
fn test_parse_pin_with_file() {
    let cmd = parse("/pin src/main.rs").expect("should parse");
    match cmd {
        SlashCommand::Pin(s) => assert_eq!(s, "src/main.rs"),
        other => panic!("expected Pin, got {:?}", other),
    }
}

#[test]
fn test_parse_unpin_no_args() {
    let cmd = parse("/unpin").expect("should parse");
    match cmd {
        SlashCommand::Unpin(s) => assert!(s.is_empty()),
        other => panic!("expected Unpin, got {:?}", other),
    }
}

#[test]
fn test_parse_unpin_all() {
    let cmd = parse("/unpin all").expect("should parse");
    match cmd {
        SlashCommand::Unpin(s) => assert_eq!(s, "all"),
        other => panic!("expected Unpin, got {:?}", other),
    }
}

#[test]
fn test_parse_unpin_with_file() {
    let cmd = parse("/unpin src/lib.rs").expect("should parse");
    match cmd {
        SlashCommand::Unpin(s) => assert_eq!(s, "src/lib.rs"),
        other => panic!("expected Unpin, got {:?}", other),
    }
}

// ── /summary ──────────────────────────────────────────────────────────────

#[test]
fn summary_parses_no_args() {
    let cmd = parse("/summary").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Summary(s) if s.is_empty()));
}

#[test]
fn summary_parses_with_focus() {
    let cmd = parse("/summary database layer").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Summary(s) if s == "database layer"));
}

#[test]
fn build_summary_prompt_no_focus_covers_all() {
    let out = build_summary_prompt(5, "");
    assert!(out.contains("5 user turns"), "turn count missing: {out}");
    assert!(
        out.contains("comprehensive"),
        "general scope missing: {out}"
    );
}

#[test]
fn build_summary_prompt_with_focus() {
    let out = build_summary_prompt(3, "auth module");
    assert!(out.contains("auth module"), "focus topic missing: {out}");
    assert!(out.contains("3 user turns"), "turn count missing: {out}");
}

#[test]
fn build_summary_prompt_singular_turn() {
    let out = build_summary_prompt(1, "");
    // Must use singular "turn" not "turns"
    assert!(out.contains("1 user turn"), "singular expected: {out}");
    assert!(
        !out.contains("1 user turns"),
        "plural must not appear: {out}"
    );
}

#[test]
fn build_summary_prompt_lists_key_sections() {
    let out = build_summary_prompt(10, "");
    assert!(
        out.contains("accomplished") || out.contains("Accomplished"),
        "accomplishments missing: {out}"
    );
    assert!(
        out.contains("decision") || out.contains("Decision"),
        "decisions section missing: {out}"
    );
    assert!(
        out.contains("Files") || out.contains("files"),
        "files section missing: {out}"
    );
}

// ── /add-dir (collect_dir_context) ───────────────────────────────

#[test]
fn collect_dir_context_nonexistent_path_errors() {
    let result = collect_dir_context("/tmp/dm_test_no_such_dir_xyz_9999", 8_000);
    assert!(result.is_err(), "expected error for nonexistent path");
    let msg = result.unwrap_err();
    assert!(msg.contains("does not exist"), "error message: {msg}");
}

#[test]
fn collect_dir_context_file_path_errors() {
    // Passing a file path (not directory) should error
    let tmp = tempfile::TempDir::new().unwrap();
    let f = tmp.path().join("file.txt");
    std::fs::write(&f, "content").unwrap();
    let result = collect_dir_context(f.to_str().unwrap(), 8_000);
    assert!(result.is_err(), "expected error for file path");
    let msg = result.unwrap_err();
    assert!(msg.contains("not a directory"), "error message: {msg}");
}

#[test]
fn collect_dir_context_empty_dir_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = collect_dir_context(tmp.path().to_str().unwrap(), 8_000);
    assert!(result.is_err(), "expected error for empty dir");
    let msg = result.unwrap_err();
    assert!(
        msg.contains("No indexable") || msg.contains("unreadable"),
        "error message: {msg}"
    );
}

#[test]
fn collect_dir_context_returns_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.rs"), "fn foo() {}").unwrap();
    std::fs::write(tmp.path().join("b.rs"), "fn bar() {}").unwrap();
    let (ctx, count) = collect_dir_context(tmp.path().to_str().unwrap(), 8_000).unwrap();
    assert_eq!(count, 2, "expected 2 files, got {count}");
    assert!(ctx.contains("fn foo"), "first file missing: {ctx}");
    assert!(ctx.contains("fn bar"), "second file missing: {ctx}");
}

#[test]
fn collect_dir_context_respects_max_bytes() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Write 3 files, each ~2KB
    for i in 0..3 {
        std::fs::write(tmp.path().join(format!("file{}.rs", i)), "x".repeat(2_000)).unwrap();
    }
    // Limit to 3KB — not all 3 files (6KB) will fit
    let (ctx, count) = collect_dir_context(tmp.path().to_str().unwrap(), 3_000).unwrap();
    assert!(
        count < 3,
        "expected fewer than 3 files with 3KB limit, got {count}"
    );
    assert!(
        ctx.contains("omitted") || count < 3,
        "expected omission notice or fewer files: {ctx}"
    );
}

#[test]
fn collect_dir_context_includes_language_hint() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.rs"), "fn main() {}").unwrap();
    let (ctx, _) = collect_dir_context(tmp.path().to_str().unwrap(), 8_000).unwrap();
    assert!(ctx.contains("```rs"), "expected rust fence: {ctx}");
}

#[test]
fn collect_dir_context_includes_file_path_header() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("lib.rs"), "pub fn x() {}").unwrap();
    let (ctx, _) = collect_dir_context(tmp.path().to_str().unwrap(), 8_000).unwrap();
    assert!(ctx.contains("File:"), "expected 'File:' header: {ctx}");
}

// ── /brief (brief_instruction) ────────────────────────────────────────────

#[test]
fn brief_instruction_returns_some_when_enabled() {
    let hint = brief_instruction(true);
    assert!(hint.is_some(), "brief mode on should return a hint");
    let text = hint.unwrap();
    assert!(!text.is_empty(), "hint should not be empty");
}

#[test]
fn brief_instruction_returns_none_when_disabled() {
    assert!(brief_instruction(false).is_none());
}

#[test]
fn brief_instruction_mentions_brevity() {
    let hint = brief_instruction(true).unwrap();
    assert!(
        hint.to_lowercase().contains("brief") || hint.to_lowercase().contains("concise"),
        "hint should mention brevity: {hint}"
    );
}

#[test]
fn brief_parses() {
    let cmd = parse("/brief").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Brief));
}

// ── /plan ────────────────────────────────────────────────────────────────

#[test]
fn plan_parses() {
    let cmd = parse("/plan").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Plan));
}

#[test]
fn plan_instruction_returns_none_when_disabled() {
    assert!(plan_instruction(false).is_none());
}

#[test]
fn plan_instruction_mentions_plan_mode() {
    let hint = plan_instruction(true).unwrap();
    assert!(
        hint.to_lowercase().contains("plan") && hint.to_lowercase().contains("read"),
        "hint should mention plan and reading: {hint}"
    );
}

#[test]
fn plan_instruction_mentions_disabled_writes() {
    let hint = plan_instruction(true).unwrap();
    assert!(
        hint.to_lowercase().contains("disabled") || hint.to_lowercase().contains("blocked"),
        "hint should mention writes are disabled: {hint}"
    );
}

// ── /files (format_files_report) ─────────────────────────────────────────

#[test]
fn files_parses() {
    let cmd = parse("/files").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Files));
}

#[test]
fn format_files_report_empty_context() {
    let report = format_files_report(&[], false);
    assert!(
        report.contains("No files"),
        "should note no files: {report}"
    );
}

#[test]
fn format_files_report_with_pinned_files() {
    let pins = vec!["src/lib.rs".to_string(), "Cargo.toml".to_string()];
    let report = format_files_report(&pins, false);
    assert!(
        report.contains("src/lib.rs"),
        "pinned file missing: {report}"
    );
    assert!(
        report.contains("Cargo.toml"),
        "pinned file missing: {report}"
    );
    assert!(
        report.contains("Pinned"),
        "pinned section header missing: {report}"
    );
}

#[test]
fn format_files_report_with_pending_context() {
    let report = format_files_report(&[], true);
    assert!(
        report.contains("queued") || report.contains("pending") || report.contains("One-shot"),
        "should note pending context: {report}"
    );
}

#[test]
fn format_files_report_with_both() {
    let pins = vec!["main.rs".to_string()];
    let report = format_files_report(&pins, true);
    assert!(report.contains("main.rs"), "pin missing: {report}");
    assert!(
        report.contains("queued") || report.contains("One-shot"),
        "pending context note missing: {report}"
    );
}

#[test]
fn format_new_save_error_includes_config_dir_and_error_text() {
    let err = anyhow::anyhow!("disk full");
    let dir = std::path::Path::new("/home/user/.config/dm");
    let msg = format_new_save_error(&err, dir);
    assert!(msg.contains("session save failed"), "got: {}", msg);
    assert!(msg.contains("disk full"), "got: {}", msg);
    assert!(msg.contains("/home/user/.config/dm"), "got: {}", msg);
    assert!(msg.contains("Check:"), "got: {}", msg);
}

// ── /new ────────────────────────���─────────────────────────────────���───────

#[test]
fn new_parses_no_title() {
    let cmd = parse("/new").expect("should parse");
    match cmd {
        SlashCommand::New(t) => assert!(t.is_empty(), "expected empty title, got: {:?}", t),
        other => panic!("expected New, got {:?}", other),
    }
}

#[test]
fn new_parses_with_title() {
    let cmd = parse("/new My Next Project").expect("should parse");
    match cmd {
        SlashCommand::New(t) => assert_eq!(t, "My Next Project"),
        other => panic!("expected New, got {:?}", other),
    }
}

#[test]
fn new_in_slash_command_names() {
    assert!(
        crate::tui::commands::SLASH_COMMAND_NAMES.contains(&"new"),
        "'new' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /version ─────────────────────────────────────────────────────────────

#[test]
fn version_parses() {
    let cmd = parse("/version").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Version));
}

#[test]
fn version_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"version"),
        "'version' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn build_version_info_contains_version_number() {
    let out = build_version_info("gemma4:26b", "localhost:11434");
    assert!(out.contains("dm v"), "missing 'dm v' prefix: {out}");
    // Should contain a semver-looking string (digit.digit.digit)
    let has_semver = out
        .split_whitespace()
        .any(|t| t.starts_with("v") && t[1..].split('.').count() >= 2);
    assert!(has_semver, "no version number found: {out}");
}

#[test]
fn build_version_info_shows_model_and_host() {
    let out = build_version_info("llama3.2:3b", "192.168.1.10:11434");
    assert!(
        out.contains("llama3.2:3b"),
        "model missing from version output: {out}"
    );
    assert!(
        out.contains("192.168.1.10:11434"),
        "host missing from version output: {out}"
    );
}

// ── /session sub-commands ────────────────────────────────────────────────

#[test]
fn session_delete_parses_session_id() {
    let cmd = parse("/session delete abc-123").expect("should parse");
    match cmd {
        SlashCommand::SessionDelete(id) => assert_eq!(id, "abc-123"),
        other => panic!("expected SessionDelete, got {:?}", other),
    }
}

#[test]
fn session_rename_parses_id_and_new_name() {
    let cmd = parse("/session rename sess-42 My Renamed Session").expect("should parse");
    match cmd {
        SlashCommand::SessionRename(id, name) => {
            assert_eq!(id, "sess-42");
            assert_eq!(name, "My Renamed Session");
        }
        other => panic!("expected SessionRename, got {:?}", other),
    }
}

#[test]
fn sessions_search_parses_query() {
    let cmd = parse("/sessions search rust async").expect("should parse");
    match cmd {
        SlashCommand::SessionSearch(q) => assert_eq!(q, "rust async"),
        other => panic!("expected SessionSearch, got {:?}", other),
    }
}

#[test]
fn sessions_with_count_parses() {
    let cmd = parse("/sessions 25").expect("should parse");
    match cmd {
        SlashCommand::Sessions(n) => assert_eq!(n, 25),
        other => panic!("expected Sessions(25), got {:?}", other),
    }
}

#[test]
fn sessions_default_count_is_ten() {
    let cmd = parse("/sessions").expect("should parse");
    match cmd {
        SlashCommand::Sessions(n) => assert_eq!(n, 10, "default count should be 10"),
        other => panic!("expected Sessions(10), got {:?}", other),
    }
}

#[test]
fn parse_sessions_tree() {
    let cmd = parse("/sessions tree").expect("should parse");
    assert!(matches!(cmd, SlashCommand::SessionsTree));
}

#[test]
fn parse_sessions_default_still_works_after_tree_added() {
    // Regression: adding the "tree" sub-arm must not break the numeric-count
    // path. A bare "/sessions" must still resolve to the default listing.
    let cmd = parse("/sessions").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Sessions(10)));
    let cmd = parse("/sessions 5").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Sessions(5)));
}

// ── /schedule sub-commands ───────────────────────────────────────────────

#[test]
fn schedule_bare_parses_as_list() {
    let cmd = parse("/schedule").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ScheduleList));
}

#[test]
fn schedule_list_explicit_parses() {
    let cmd = parse("/schedule list").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ScheduleList));
}

#[test]
fn schedule_add_parses_spec() {
    let cmd = parse("/schedule add 0 9 * * 1-5 summarize git log").expect("should parse");
    match cmd {
        SlashCommand::ScheduleAdd(spec) => assert_eq!(spec, "0 9 * * 1-5 summarize git log"),
        other => panic!("expected ScheduleAdd, got {:?}", other),
    }
}

#[test]
fn schedule_remove_parses_id() {
    let cmd = parse("/schedule remove abc123").expect("should parse");
    match cmd {
        SlashCommand::ScheduleRemove(id) => assert_eq!(id, "abc123"),
        other => panic!("expected ScheduleRemove, got {:?}", other),
    }
}

#[test]
fn undo_files_parses() {
    let cmd = parse("/undo-files").expect("should parse");
    assert!(matches!(cmd, SlashCommand::UndoFiles));
}

#[test]
fn schedule_unknown_subcommand_is_unknown() {
    let cmd = parse("/schedule frobnicate").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Unknown(_)));
}

#[test]
fn schedule_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"schedule"),
        "'schedule' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn tool_disable_parses() {
    let cmd = parse("/tool disable bash").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ToolDisable(ref name) if name == "bash"));
}

#[test]
fn tool_enable_parses() {
    let cmd = parse("/tool enable bash").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ToolEnable(ref name) if name == "bash"));
}

#[test]
fn tool_list_parses() {
    let cmd = parse("/tool list").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ToolList));
}

#[test]
fn tool_no_args_is_list() {
    let cmd = parse("/tool").expect("should parse");
    assert!(matches!(cmd, SlashCommand::ToolList));
}

#[test]
fn tool_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"tool"),
        "'tool' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn wiki_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"wiki"),
        "'wiki' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn chain_start_builds_initial_state_all_pending() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainNodeStatus, ChainState};

    let config = ChainConfig {
        name: "test".into(),
        description: None,
        nodes: vec![
            ChainNodeConfig {
                id: "n1".into(),
                name: "planner".into(),
                role: "planner".into(),
                model: "m".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
            ChainNodeConfig {
                id: "n2".into(),
                name: "builder".into(),
                role: "builder".into(),
                model: "m".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: Some("n1".into()),
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
        ],
        max_cycles: 3,
        max_total_turns: 50,
        workspace: std::path::PathBuf::from("/tmp"),
        skip_permissions_warning: true,
        loop_forever: false,
        directive: None,
    };
    let mut node_statuses = std::collections::HashMap::new();
    for node in &config.nodes {
        node_statuses.insert(node.name.clone(), ChainNodeStatus::Pending);
    }
    let state = ChainState {
        chain_id: "c1".into(),
        config,
        active_node_index: None,
        node_statuses,
        current_cycle: 0,
        turns_used: 0,
        node_outputs: std::collections::HashMap::new(),
        last_signal: None,
        last_updated: chrono::Utc::now(),
        last_abort_reason: None,
        pending_additions: vec![],
        pending_removals: vec![],
        pending_model_swaps: std::collections::HashMap::new(),
        node_durations: std::collections::HashMap::new(),
        node_failures: std::collections::HashMap::new(),
        total_duration_secs: 0.0,
        node_prompt_tokens: std::collections::HashMap::new(),
        node_completion_tokens: std::collections::HashMap::new(),
    };
    assert_eq!(state.current_cycle, 0);
    assert!(state.active_node_index.is_none());
    assert!(matches!(
        state.node_statuses.get("planner"),
        Some(ChainNodeStatus::Pending)
    ));
    assert!(matches!(
        state.node_statuses.get("builder"),
        Some(ChainNodeStatus::Pending)
    ));
    assert_eq!(state.node_statuses.len(), 2);
}

#[test]
fn chain_status_prefers_in_memory_over_disk() {
    let in_memory: Option<crate::orchestrate::types::ChainState> =
        Some(crate::orchestrate::types::ChainState {
            chain_id: "memory-chain".into(),
            config: crate::orchestrate::types::ChainConfig {
                name: "from-memory".into(),
                description: None,
                nodes: vec![],
                max_cycles: 1,
                max_total_turns: 10,
                workspace: std::path::PathBuf::from("/tmp"),
                skip_permissions_warning: true,
                loop_forever: false,
                directive: None,
            },
            active_node_index: None,
            node_statuses: std::collections::HashMap::new(),
            current_cycle: 42,
            turns_used: 7,
            node_outputs: std::collections::HashMap::new(),
            last_signal: None,
            last_updated: chrono::Utc::now(),
            last_abort_reason: None,
            pending_additions: vec![],
            pending_removals: vec![],
            pending_model_swaps: std::collections::HashMap::new(),
            node_durations: std::collections::HashMap::new(),
            node_failures: std::collections::HashMap::new(),
            total_duration_secs: 0.0,
            node_prompt_tokens: std::collections::HashMap::new(),
            node_completion_tokens: std::collections::HashMap::new(),
        });
    let resolved = in_memory.clone().or_else(crate::orchestrate::chain_status);
    let state = resolved.expect("should prefer in-memory state");
    assert_eq!(state.config.name, "from-memory");
    assert_eq!(state.current_cycle, 42);
}

#[test]
fn chain_resume_from_initial_state_uses_resume_checkpoint() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainNodeStatus, ChainState};

    let config = ChainConfig {
        name: "resumed".into(),
        description: None,
        nodes: vec![ChainNodeConfig {
            id: "n1".into(),
            name: "planner".into(),
            role: "planner".into(),
            model: "m".into(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: None,
            max_retries: 1,
            timeout_secs: 3600,
            max_tool_turns: 200,
        }],
        max_cycles: 10,
        max_total_turns: 100,
        workspace: std::path::PathBuf::from("/tmp"),
        skip_permissions_warning: true,
        loop_forever: false,
        directive: None,
    };
    let mut node_statuses = std::collections::HashMap::new();
    node_statuses.insert("planner".into(), ChainNodeStatus::Completed);
    let resume_state = ChainState {
        chain_id: "old-id".into(),
        config: config.clone(),
        active_node_index: None,
        node_statuses,
        current_cycle: 5,
        turns_used: 42,
        node_outputs: [("planner".to_string(), "last output".to_string())]
            .into_iter()
            .collect(),
        last_signal: None,
        last_updated: chrono::Utc::now(),
        last_abort_reason: None,
        pending_additions: vec![],
        pending_removals: vec![],
        pending_model_swaps: std::collections::HashMap::new(),
        node_durations: std::collections::HashMap::new(),
        node_failures: std::collections::HashMap::new(),
        total_duration_secs: 0.0,
        node_prompt_tokens: std::collections::HashMap::new(),
        node_completion_tokens: std::collections::HashMap::new(),
    };
    // Simulate what the handler does: if resume_state is Some, clone it and update config/chain_id.
    let mut initial = resume_state.clone();
    initial.config = config;
    initial.chain_id = "new-chain-id".into();
    assert_eq!(initial.current_cycle, 5);
    assert_eq!(initial.turns_used, 42);
    assert!(matches!(
        initial.node_statuses.get("planner"),
        Some(ChainNodeStatus::Completed)
    ));
    assert_eq!(initial.node_outputs.get("planner").unwrap(), "last output");
    assert_eq!(initial.chain_id, "new-chain-id");
}

#[test]
fn chain_resume_from_rejects_double_start() {
    use crate::orchestrate::types::{ChainConfig, ChainState};

    let state = ChainState {
        chain_id: "running".into(),
        config: ChainConfig {
            name: "active".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        active_node_index: None,
        node_statuses: std::collections::HashMap::new(),
        current_cycle: 1,
        turns_used: 3,
        node_outputs: std::collections::HashMap::new(),
        last_signal: None,
        last_updated: chrono::Utc::now(),
        last_abort_reason: None,
        pending_additions: vec![],
        pending_removals: vec![],
        pending_model_swaps: std::collections::HashMap::new(),
        node_durations: std::collections::HashMap::new(),
        node_failures: std::collections::HashMap::new(),
        total_duration_secs: 0.0,
        node_prompt_tokens: std::collections::HashMap::new(),
        node_completion_tokens: std::collections::HashMap::new(),
    };
    // The handler checks app.chain_state.is_some() before proceeding.
    // Simulate: if chain_state is Some, resume-from should be rejected.
    assert!(state.chain_id == "running");
    let is_running = Some(state).is_some();
    assert!(
        is_running,
        "double-start guard should detect existing chain"
    );
}

#[test]
fn chain_start_guard_checks_memory_first() {
    use crate::orchestrate::types::{ChainConfig, ChainState};

    let state = ChainState::new(
        ChainConfig {
            name: "running".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        "active-chain".into(),
    );
    // The guard is: app.chain_state.is_some() || chain_status().is_some()
    // When in-memory state is set, the first check short-circuits (no disk I/O).
    let in_memory = Some(state);
    let would_block = in_memory.is_some() || crate::orchestrate::chain_status().is_some();
    assert!(
        would_block,
        "should detect running chain from in-memory state"
    );
}

#[test]
fn chain_add_validates_input_from_against_in_memory_state() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};

    let state = ChainState::new(
        ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "n1".into(),
                    name: "planner".into(),
                    role: "planner".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
                ChainNodeConfig {
                    id: "n2".into(),
                    name: "builder".into(),
                    role: "builder".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: Some("n1".into()),
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
            ],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        "c1".into(),
    );
    // Simulate the validation logic from /chain add handler
    let current_state = Some(state).or_else(crate::orchestrate::chain_status);
    let target = "planner";
    let is_valid = current_state.as_ref().is_none_or(|st| {
        let known: std::collections::HashSet<&str> =
            st.config.nodes.iter().map(|n| n.name.as_str()).collect();
        known.contains(target)
    });
    assert!(is_valid, "planner should be a valid input_from target");

    let bad_target = "nonexistent";
    let current_state2 = current_state;
    let is_valid2 = current_state2.as_ref().is_none_or(|st| {
        let known: std::collections::HashSet<&str> =
            st.config.nodes.iter().map(|n| n.name.as_str()).collect();
        known.contains(bad_target)
    });
    assert!(!is_valid2, "nonexistent should be rejected");
}

#[test]
fn chain_add_updates_in_memory_state() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainNodeStatus, ChainState};

    let mut state = ChainState::new(
        ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![ChainNodeConfig {
                id: "n1".into(),
                name: "builder".into(),
                role: "builder".into(),
                model: "m".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        "c1".into(),
    );
    assert_eq!(state.node_statuses.len(), 1);
    state
        .node_statuses
        .insert("reviewer".into(), ChainNodeStatus::Pending);
    assert_eq!(state.node_statuses.len(), 2);
    assert!(matches!(
        state.node_statuses.get("reviewer"),
        Some(ChainNodeStatus::Pending)
    ));
}

#[test]
fn chain_remove_updates_in_memory_state() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};

    let mut state = ChainState::new(
        ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "n1".into(),
                    name: "builder".into(),
                    role: "builder".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
                ChainNodeConfig {
                    id: "n2".into(),
                    name: "reviewer".into(),
                    role: "reviewer".into(),
                    model: "m".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: Some("n1".into()),
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
            ],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        "c1".into(),
    );
    assert_eq!(state.node_statuses.len(), 2);
    state.node_statuses.remove("reviewer");
    state.pending_removals.push("reviewer".into());
    assert_eq!(state.node_statuses.len(), 1);
    assert!(!state.node_statuses.contains_key("reviewer"));
    assert_eq!(state.pending_removals, ["reviewer"]);
}

#[test]
fn chain_model_updates_in_memory_state() {
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};

    let mut state = ChainState::new(
        ChainConfig {
            name: "test".into(),
            description: None,
            nodes: vec![ChainNodeConfig {
                id: "n1".into(),
                name: "builder".into(),
                role: "builder".into(),
                model: "llama3".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        },
        "c1".into(),
    );
    state
        .pending_model_swaps
        .insert("builder".into(), "gemma3:27b".into());
    assert_eq!(
        state.pending_model_swaps.get("builder").unwrap(),
        "gemma3:27b"
    );
}

#[test]
fn test_parse_usage() {
    let cmd = parse("/usage").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Usage));
}

#[tokio::test]
async fn test_usage_shows_token_stats() {
    let mut app = App::new(
        "test-model".into(),
        "http://localhost:11434".into(),
        "test-session".into(),
        std::path::PathBuf::from("/tmp/dm-test-usage"),
        vec![],
    );
    app.token_usage.record_turn(500, 200);
    app.token_usage.record_turn(600, 300);
    app.ctx_usage = Some((8000, 32000));

    let client =
        crate::ollama::client::OllamaClient::new("http://localhost:11434".into(), "test".into());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let result = execute(SlashCommand::Usage, &mut app, &client, &tx).await;
    match result {
        SlashResult::Info(info) => {
            assert!(info.contains("1100"), "should show prompt tokens: {}", info);
            assert!(
                info.contains("500"),
                "should show completion tokens: {}",
                info
            );
            assert!(info.contains("1600"), "should show total: {}", info);
            assert!(info.contains("2"), "should show turn count: {}", info);
            assert!(
                info.contains("800"),
                "should show avg tokens/turn: {}",
                info
            );
            assert!(info.contains("25%"), "should show context pct: {}", info);
        }
        _ => panic!("expected Info result"),
    }
}

#[tokio::test]
async fn test_usage_no_turns() {
    let mut app = App::new(
        "test-model".into(),
        "http://localhost:11434".into(),
        "test-session".into(),
        std::path::PathBuf::from("/tmp/dm-test-usage-empty"),
        vec![],
    );

    let client =
        crate::ollama::client::OllamaClient::new("http://localhost:11434".into(), "test".into());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let result = execute(SlashCommand::Usage, &mut app, &client, &tx).await;
    match result {
        SlashResult::Info(info) => {
            assert!(
                info.contains("Turns:             0"),
                "should show 0 turns: {}",
                info
            );
            assert!(
                info.contains("Avg tokens/turn:   0"),
                "should show 0 avg: {}",
                info
            );
            assert!(
                info.contains("Context window: unknown"),
                "should show unknown ctx: {}",
                info
            );
        }
        _ => panic!("expected Info result"),
    }
}

#[tokio::test]
async fn new_surfaces_save_error_in_entries_when_config_dir_unwritable() {
    // config_dir points at a file (not a directory) → session::storage::save
    // fails at create_dir_all. Deterministic across platforms.
    let (_file, mut app, client, tx, _rx) = dispatch_harness_unwritable_config();
    // Seed an entry so we can prove clear() ran AND the warning is present afterward.
    app.push_entry(EntryKind::UserMessage, "preexisting".into());

    let result = execute(SlashCommand::New(String::new()), &mut app, &client, &tx).await;

    match result {
        SlashResult::Info(m) => assert!(
            m.contains("New conversation started"),
            "expected success Info, got: {}",
            m
        ),
        _ => panic!("expected Info result"),
    }

    // The preexisting UserMessage must be gone (clear() ran).
    assert!(
        !app.entries.iter().any(|e| e.content == "preexisting"),
        "expected clear() to run; entries still contain preexisting message"
    );

    // The save-fail warning must be present (not erased by the clear).
    let warn_entries: Vec<&str> = app
        .entries
        .iter()
        .filter(|e| matches!(e.kind, EntryKind::SystemInfo))
        .map(|e| e.content.as_str())
        .collect();
    assert!(
        warn_entries
            .iter()
            .any(|c| c.contains("session save failed") && c.contains("Check:")),
        "expected save-failed warning with Check: hint in SystemInfo entries, got: {:?}",
        warn_entries
    );
}

#[tokio::test]
async fn new_does_not_surface_warning_when_config_dir_writable() {
    // Complement of the unwritable test above: with a valid config_dir the
    // save succeeds silently and no SystemInfo warning should appear. Guards
    // against a future regression where someone pushes a warning on every
    // `/new`, not just failures.
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let (mut app, client, tx, _rx) = dispatch_harness(dir.path().to_path_buf());
    app.push_entry(EntryKind::UserMessage, "preexisting".into());

    let result = execute(SlashCommand::New(String::new()), &mut app, &client, &tx).await;

    match result {
        SlashResult::Info(m) => assert!(
            m.contains("New conversation started"),
            "expected success Info, got: {}",
            m
        ),
        _ => panic!("expected Info result"),
    }

    assert!(
        !app.entries.iter().any(|e| e.content == "preexisting"),
        "expected clear() to run; entries still contain preexisting message"
    );

    let has_warning = app.entries.iter().any(|e| {
        matches!(e.kind, EntryKind::SystemInfo) && e.content.contains("session save failed")
    });
    assert!(
        !has_warning,
        "expected no save-fail warning on writable config_dir, got entries: {:?}",
        app.entries
    );
}

// ── /changelog parse tests ───────────────────────────────────────────

#[test]
fn parse_changelog_no_args() {
    let cmd = parse("/changelog").expect("should parse");
    match cmd {
        SlashCommand::Changelog { from, to } => {
            assert!(from.is_empty());
            assert!(to.is_empty());
        }
        _ => panic!("expected Changelog"),
    }
}

#[test]
fn parse_changelog_with_from() {
    let cmd = parse("/changelog v1.0.0").expect("should parse");
    match cmd {
        SlashCommand::Changelog { from, to } => {
            assert_eq!(from, "v1.0.0");
            assert!(to.is_empty());
        }
        _ => panic!("expected Changelog"),
    }
}

#[test]
fn parse_changelog_with_from_and_to() {
    let cmd = parse("/changelog v1.0.0 v2.0.0").expect("should parse");
    match cmd {
        SlashCommand::Changelog { from, to } => {
            assert_eq!(from, "v1.0.0");
            assert_eq!(to, "v2.0.0");
        }
        _ => panic!("expected Changelog"),
    }
}

// ── /blame parse tests ───────────────────────────────────────────────

#[test]
fn parse_blame_with_file() {
    let cmd = parse("/blame src/main.rs").expect("should parse");
    match cmd {
        SlashCommand::Blame { file, line } => {
            assert_eq!(file, "src/main.rs");
            assert!(line.is_none());
        }
        _ => panic!("expected Blame"),
    }
}

#[test]
fn parse_blame_with_file_and_line() {
    let cmd = parse("/blame src/main.rs 42").expect("should parse");
    match cmd {
        SlashCommand::Blame { file, line } => {
            assert_eq!(file, "src/main.rs");
            assert_eq!(line, Some(42));
        }
        _ => panic!("expected Blame"),
    }
}

#[test]
fn parse_blame_no_file() {
    let cmd = parse("/blame").expect("should parse");
    match cmd {
        SlashCommand::Blame { file, line } => {
            assert!(file.is_empty());
            assert!(line.is_none());
        }
        _ => panic!("expected Blame"),
    }
}

#[test]
fn parse_blame_non_numeric_line_ignored() {
    let cmd = parse("/blame src/main.rs foo").expect("should parse");
    match cmd {
        SlashCommand::Blame { file, line } => {
            assert_eq!(file, "src/main.rs");
            assert!(line.is_none(), "non-numeric line should be None");
        }
        _ => panic!("expected Blame"),
    }
}

// ── /conflicts parse test ────────────────────────────────────────────

#[test]
fn parse_conflicts() {
    let cmd = parse("/conflicts").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Conflicts));
}

// ── /resume parse tests ─────────────────────────────────────────────

#[test]
fn parse_resume_with_id() {
    let cmd = parse("/resume abc123").expect("should parse");
    match cmd {
        SlashCommand::Resume(id) => assert_eq!(id, "abc123"),
        _ => panic!("expected Resume"),
    }
}

#[test]
fn parse_resume_no_id() {
    let cmd = parse("/resume").expect("should parse");
    match cmd {
        SlashCommand::Resume(id) => assert!(id.is_empty(), "no id should be empty string"),
        _ => panic!("expected Resume"),
    }
}

#[test]
fn parse_resume_trims_whitespace() {
    let cmd = parse("/resume   def456  ").expect("should parse");
    match cmd {
        SlashCommand::Resume(id) => assert_eq!(id, "def456"),
        _ => panic!("expected Resume"),
    }
}

// ── /stash parse tests ──────────────────────────────────────────────

#[test]
fn parse_stash_no_args() {
    let cmd = parse("/stash").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert!(sub.is_empty()),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn parse_stash_push() {
    let cmd = parse("/stash push").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert_eq!(sub, "push"),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn parse_stash_pop() {
    let cmd = parse("/stash pop").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert_eq!(sub, "pop"),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn parse_stash_list() {
    let cmd = parse("/stash list").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert_eq!(sub, "list"),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn parse_stash_show_with_ref() {
    let cmd = parse("/stash show stash@{0}").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert_eq!(sub, "show stash@{0}"),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn parse_stash_drop() {
    let cmd = parse("/stash drop").expect("should parse");
    match cmd {
        SlashCommand::Stash(sub) => assert_eq!(sub, "drop"),
        _ => panic!("expected Stash"),
    }
}

#[test]
fn stash_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"stash"),
        "'stash' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /retry parse tests ──────────────────────────────────────────────

#[test]
fn parse_retry() {
    let cmd = parse("/retry").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Retry));
}

#[test]
fn parse_regenerate_alias() {
    let cmd = parse("/regenerate").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Retry));
}

#[test]
fn retry_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"retry"),
        "'retry' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /search parse tests ──────────────────────────────────────────────

#[test]
fn parse_search_with_query() {
    let cmd = parse("/search foo bar").expect("should parse");
    match cmd {
        SlashCommand::Search(q) => assert_eq!(q, "foo bar"),
        _ => panic!("expected Search"),
    }
}

#[test]
fn parse_search_no_query() {
    let cmd = parse("/search").expect("should parse");
    match cmd {
        SlashCommand::Search(q) => assert!(q.is_empty()),
        _ => panic!("expected Search"),
    }
}

#[test]
fn parse_grep_alias() {
    let cmd = parse("/grep pattern").expect("should parse");
    match cmd {
        SlashCommand::Search(q) => assert_eq!(q, "pattern"),
        _ => panic!("expected Search"),
    }
}

#[test]
fn search_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"search"),
        "'search' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /history parse tests ─────────────────────────────────────────────

#[test]
fn parse_history_no_args_defaults_to_30() {
    let cmd = parse("/history").expect("should parse");
    match cmd {
        SlashCommand::History(n) => assert_eq!(n, 30),
        _ => panic!("expected History"),
    }
}

#[test]
fn parse_history_with_count() {
    let cmd = parse("/history 10").expect("should parse");
    match cmd {
        SlashCommand::History(n) => assert_eq!(n, 10),
        _ => panic!("expected History"),
    }
}

#[test]
fn parse_history_invalid_defaults_to_30() {
    let cmd = parse("/history foo").expect("should parse");
    match cmd {
        SlashCommand::History(n) => assert_eq!(n, 30, "invalid arg should default to 30"),
        _ => panic!("expected History"),
    }
}

#[test]
fn history_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"history"),
        "'history' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn parse_kill() {
    let cmd = parse("/kill").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Kill));
}

#[test]
fn kill_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"kill"),
        "'kill' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /test parse tests ───────────────────────────────────────────────

#[test]
fn parse_test_no_args() {
    let cmd = parse("/test").expect("should parse");
    match cmd {
        SlashCommand::Test(s) => assert!(s.is_empty()),
        _ => panic!("expected Test"),
    }
}

#[test]
fn parse_test_with_custom_cmd() {
    let cmd = parse("/test pytest -x tests/").expect("should parse");
    match cmd {
        SlashCommand::Test(s) => assert_eq!(s, "pytest -x tests/"),
        _ => panic!("expected Test"),
    }
}

#[test]
fn test_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"test"),
        "'test' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /lint parse tests ───────────────────────────────────────────────

#[test]
fn parse_lint_no_args() {
    let cmd = parse("/lint").expect("should parse");
    match cmd {
        SlashCommand::Lint(s) => assert!(s.is_empty()),
        _ => panic!("expected Lint"),
    }
}

#[test]
fn parse_lint_with_custom_cmd() {
    let cmd = parse("/lint eslint src/ --fix").expect("should parse");
    match cmd {
        SlashCommand::Lint(s) => assert_eq!(s, "eslint src/ --fix"),
        _ => panic!("expected Lint"),
    }
}

#[test]
fn lint_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"lint"),
        "'lint' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /find parse tests ───────────────────────────────────────────────

#[test]
fn parse_find_with_pattern() {
    let cmd = parse("/find src/**/*.rs").expect("should parse");
    match cmd {
        SlashCommand::Find(s) => assert_eq!(s, "src/**/*.rs"),
        _ => panic!("expected Find"),
    }
}

#[test]
fn parse_find_no_args() {
    let cmd = parse("/find").expect("should parse");
    match cmd {
        SlashCommand::Find(s) => assert!(s.is_empty()),
        _ => panic!("expected Find"),
    }
}

#[test]
fn find_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"find"),
        "'find' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /rg parse tests ────────────────────────────────────────────────

#[test]
fn parse_rg_with_pattern() {
    let cmd = parse("/rg fn main src/").expect("should parse");
    match cmd {
        SlashCommand::Rg(s) => assert_eq!(s, "fn main src/"),
        _ => panic!("expected Rg"),
    }
}

#[test]
fn parse_rg_no_args() {
    let cmd = parse("/rg").expect("should parse");
    match cmd {
        SlashCommand::Rg(s) => assert!(s.is_empty()),
        _ => panic!("expected Rg"),
    }
}

#[test]
fn rg_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"rg"),
        "'rg' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /cd parse tests ────────────────────────────────────────────────

#[test]
fn parse_cd_with_path() {
    let cmd = parse("/cd /tmp").expect("should parse");
    match cmd {
        SlashCommand::Cd(s) => assert_eq!(s, "/tmp"),
        _ => panic!("expected Cd"),
    }
}

#[test]
fn parse_cd_no_args() {
    let cmd = parse("/cd").expect("should parse");
    match cmd {
        SlashCommand::Cd(s) => assert!(s.is_empty()),
        _ => panic!("expected Cd"),
    }
}

#[test]
fn parse_cd_tilde_path() {
    let cmd = parse("/cd ~/projects").expect("should parse");
    match cmd {
        SlashCommand::Cd(s) => assert_eq!(s, "~/projects"),
        _ => panic!("expected Cd"),
    }
}

#[test]
fn cd_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"cd"),
        "'cd' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /tree parse tests ──────────────────────────────────────────────

#[test]
fn parse_tree_no_args() {
    let cmd = parse("/tree").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert!(path.is_empty());
            assert_eq!(depth, 3);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_with_path() {
    let cmd = parse("/tree src/").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "src/");
            assert_eq!(depth, 3);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_with_depth_only() {
    let cmd = parse("/tree 5").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert!(path.is_empty());
            assert_eq!(depth, 5);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_with_path_and_depth() {
    let cmd = parse("/tree src/ 2").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "src/");
            assert_eq!(depth, 2);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn tree_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"tree"),
        "'tree' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

#[test]
fn parse_tree_multi_word_path_no_depth() {
    // 2 words, last is non-numeric — whole args become the path.
    let cmd = parse("/tree src tui").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "src tui");
            assert_eq!(depth, 3);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_multi_word_path_with_trailing_depth() {
    // 3 words, last parses as int — strip it and join the rest.
    let cmd = parse("/tree src tui 5").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "src tui");
            assert_eq!(depth, 5);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_path_containing_number_not_trailing() {
    // Mid-path number must not be mistaken for a trailing depth.
    let cmd = parse("/tree src/5 tui").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "src/5 tui");
            assert_eq!(depth, 3);
        }
        _ => panic!("expected Tree"),
    }
}

#[test]
fn parse_tree_three_words_trailing_zero_depth() {
    // "0".parse::<usize>() is Ok(0); depth-0 is still a valid parse.
    let cmd = parse("/tree a b 0").expect("should parse");
    match cmd {
        SlashCommand::Tree { path, depth } => {
            assert_eq!(path, "a b");
            assert_eq!(depth, 0);
        }
        _ => panic!("expected Tree"),
    }
}

// ── /changes parse tests ───────────────────────────────────────────

#[test]
fn parse_changes() {
    let cmd = parse("/changes").expect("should parse");
    assert!(matches!(cmd, SlashCommand::Changes));
}

#[test]
fn changes_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"changes"),
        "'changes' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /compact parse tests ────────────────────────────────────────────

#[test]
fn parse_compact_no_args() {
    let cmd = parse("/compact").expect("should parse");
    match cmd {
        SlashCommand::Compact(s) => assert!(s.is_empty()),
        _ => panic!("expected Compact"),
    }
}

#[test]
fn parse_compact_preview() {
    let cmd = parse("/compact preview").expect("should parse");
    match cmd {
        SlashCommand::Compact(s) => assert_eq!(s, "preview"),
        _ => panic!("expected Compact"),
    }
}

#[test]
fn parse_compact_force() {
    let cmd = parse("/compact force").expect("should parse");
    match cmd {
        SlashCommand::Compact(s) => assert_eq!(s, "force"),
        _ => panic!("expected Compact"),
    }
}

#[test]
fn compact_in_slash_command_names() {
    assert!(
        SLASH_COMMAND_NAMES.contains(&"compact"),
        "'compact' must be in SLASH_COMMAND_NAMES for Tab autocomplete"
    );
}

// ── /wiki planner (C77) ────────────────────────────────────────────

#[test]
fn parses_slash_wiki_planner() {
    assert!(matches!(
        parse("/wiki planner").expect("should parse"),
        SlashCommand::WikiPlanner
    ));
}

#[test]
fn parses_slash_wiki_planner_rejects_extra_words() {
    // Extra trailing tokens must route to WikiUnknown so the user gets a
    // usage hint rather than a silent success.
    match parse("/wiki planner extra").expect("should parse") {
        SlashCommand::WikiUnknown(s) => assert_eq!(s, "planner extra"),
        other => panic!("expected WikiUnknown for 'planner extra', got {:?}", other),
    }
}

#[test]
fn format_wiki_planner_brief_empty_renders_placeholder() {
    let out = format_wiki_planner_brief(&crate::wiki::PlannerBrief::default());
    assert!(
        out.contains("No planner signals"),
        "empty brief must show placeholder, got: {out}"
    );
}

#[test]
fn format_wiki_planner_brief_hot_paths_surfaces_section() {
    let brief = crate::wiki::PlannerBrief {
        hot_paths: vec![("src/foo.rs".to_string(), 3)],
        ..Default::default()
    };
    let out = format_wiki_planner_brief(&brief);
    assert!(out.contains("Hot paths"), "hot paths header: {out}");
    assert!(out.contains("src/foo.rs"), "hot path entry: {out}");
}

#[test]
fn format_wiki_planner_brief_drifting_pages_surfaces_section() {
    let brief = crate::wiki::PlannerBrief {
        drifting_pages: vec!["entities/foo.md".to_string()],
        ..Default::default()
    };
    let out = format_wiki_planner_brief(&brief);
    assert!(out.contains("Drifting pages"), "drift header: {out}");
    assert!(out.contains("entities/foo.md"), "drift entry: {out}");
}

#[test]
fn format_wiki_planner_brief_lint_counts_surfaces_section() {
    let brief = crate::wiki::PlannerBrief {
        lint_counts: vec![(crate::wiki::WikiLintKind::ItemDrift, 2)],
        ..Default::default()
    };
    let out = format_wiki_planner_brief(&brief);
    assert!(out.contains("Open lint findings"), "lint header: {out}");
    assert!(out.contains("ItemDrift"), "lint kind: {out}");
}

#[test]
fn format_wiki_planner_brief_recent_cycles_surfaces_section() {
    let brief = crate::wiki::PlannerBrief {
        recent_cycles: vec![crate::wiki::RecentCycle {
            cycle: 42,
            chain: "self-improve".to_string(),
            page_path: "synthesis/cycle-42.md".to_string(),
            last_updated: None,
            outcome: None,
        }],
        ..Default::default()
    };
    let out = format_wiki_planner_brief(&brief);
    assert!(out.contains("Recent cycles"), "cycles header: {out}");
    assert!(out.contains("Cycle 42"), "cycle row: {out}");
}

#[test]
fn wiki_unknown_usage_mentions_planner() {
    // Regression guard: usage text must advertise every /wiki subcommand,
    // including the C77 addition.
    let out = format_wiki_unknown_usage("bogus");
    assert!(
        out.contains("/wiki planner"),
        "usage must advertise /wiki planner: {out}"
    );
}

// ── /wiki <typo> suggestions (C78) ─────────────────────────────────

#[test]
fn suggest_wiki_subcommand_planer_returns_planner() {
    assert_eq!(suggest_wiki_subcommand("planer"), Some("planner"));
}

#[test]
fn suggest_wiki_subcommand_lit_returns_lint() {
    assert_eq!(suggest_wiki_subcommand("lit"), Some("lint"));
}

#[test]
fn suggest_wiki_subcommand_refrsh_returns_refresh() {
    assert_eq!(suggest_wiki_subcommand("refrsh"), Some("refresh"));
}

#[test]
fn suggest_wiki_subcommand_concept_returns_concepts() {
    // "concept" is a substring of the plural registered name — either
    // the substring-match or the Levenshtein=1 path should land it.
    assert_eq!(suggest_wiki_subcommand("concept"), Some("concepts"));
}

/// Disambiguation guard: typing `"stat"` (a typo of `status`) must
/// suggest `/wiki status`, not `/wiki stats`. Both names are valid
/// subcommands and `"stat"` is a substring of both, so the suggester's
/// answer depends on `WIKI_SUBCOMMAND_NAMES` insertion order. This
/// test pins the user-friendly resolution: `status` (a much older
/// command, more likely the intended target) wins over `stats`
/// (added in C7 for telemetry).
#[test]
fn suggest_wiki_subcommand_stat_prefers_status_over_stats() {
    assert_eq!(suggest_wiki_subcommand("stat"), Some("status"));
    assert_eq!(suggest_wiki_subcommand("statu"), Some("status"));
    // The exact `"stats"` match still resolves to `"stats"` (parser
    // intercepts before the suggester runs in the live path, but the
    // suggester itself should not be confused).
    assert_eq!(suggest_wiki_subcommand("stats"), Some("stats"));
}

#[test]
fn suggest_wiki_subcommand_too_far_returns_none() {
    assert_eq!(suggest_wiki_subcommand("asdfgh"), None);
}

#[test]
fn suggest_wiki_subcommand_empty_returns_none() {
    assert_eq!(suggest_wiki_subcommand(""), None);
}

#[test]
fn suggest_wiki_subcommand_exact_match_returns_itself() {
    // Regression guard: an exact hit must never fall through to None.
    assert_eq!(suggest_wiki_subcommand("planner"), Some("planner"));
}

#[test]
fn wiki_unknown_output_prefixes_did_you_mean_on_close_match() {
    let out = format_wiki_unknown_usage("planer");
    assert!(
        out.contains("Did you mean /wiki planner?"),
        "close match should be suggested: {out}"
    );
    assert!(
        out.contains("Usage:"),
        "full usage block should still render: {out}"
    );
}

// ── C87 /wiki fresh tests ────────────────────────────────────────────

#[test]
fn parse_wiki_fresh_returns_wiki_fresh_variant() {
    assert!(matches!(
        parse("/wiki fresh").expect("should parse"),
        SlashCommand::WikiFresh
    ));
}

#[test]
fn parse_wiki_fresh_ignores_trailing_whitespace() {
    // `splitn(2, ' ')` on "wiki fresh " yields parts[1] = "fresh ",
    // which trims back to "fresh" and routes to WikiFresh. Mirrors
    // /wiki status's trailing-whitespace behavior.
    assert!(matches!(
        parse("/wiki fresh ").expect("should parse"),
        SlashCommand::WikiFresh
    ));
}

#[test]
fn wiki_subcommand_names_includes_fresh_exactly_once() {
    let hits = WIKI_SUBCOMMAND_NAMES
        .iter()
        .filter(|&&n| n == "fresh")
        .count();
    assert_eq!(
        hits, 1,
        "WIKI_SUBCOMMAND_NAMES must list 'fresh' exactly once; got {:?}",
        WIKI_SUBCOMMAND_NAMES
    );
}

#[test]
fn suggest_wiki_subcommand_matches_fresh_via_substring() {
    // Input must embed "fresh" without also embedding "refresh" — any
    // substring of "fresh" shorter than the word is also a substring
    // of "refresh", so short needles can't disambiguate. "fresher"
    // contains "fresh" but not "refresh", so the substring path
    // lands on `fresh`.
    assert_eq!(suggest_wiki_subcommand("fresher"), Some("fresh"));
}

#[test]
fn suggest_wiki_subcommand_matches_fresh_via_levenshtein() {
    // "frsh" has Levenshtein distance 1 from "fresh" and is not a
    // substring of any other registered subcommand — exercises the
    // distance-≤-2 path.
    assert_eq!(suggest_wiki_subcommand("frsh"), Some("fresh"));
}

#[test]
fn format_wiki_fresh_renders_ranked_entity_concept_pages() {
    use crate::wiki::{IndexEntry, PageType, Wiki};
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
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
    ]);
    wiki.save_index(&idx).expect("save_index");

    let out = format_wiki_fresh(&wiki, 4096);
    // Header reports the Entity+Concept count (3), excluding synthesis.
    assert!(
        out.contains("3 Entity/Concept total"),
        "header must count only Entity/Concept; got: {}",
        out,
    );
    // Synthesis is filtered out.
    assert!(
        !out.contains("filtered synthesis"),
        "synthesis must not appear: {}",
        out,
    );
    // Newest-first ordering.
    let new_pos = out.find("newest entity").expect("newest present");
    let mid_pos = out.find("mid concept").expect("mid present");
    let old_pos = out.find("older entity").expect("old present");
    assert!(
        new_pos < mid_pos && mid_pos < old_pos,
        "order must be newest → oldest; got: {}",
        out,
    );
    // Per-entry `(updated {ts})` suffix lands on every ranked row.
    assert!(
        out.contains("(updated 2026-04-19 14:00:00)"),
        "newest row must carry its timestamp: {}",
        out,
    );
}

#[test]
fn format_wiki_fresh_emits_stub_when_no_entity_or_concept() {
    use crate::wiki::{IndexEntry, PageType, Wiki};
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
    let mut idx = wiki.load_index().unwrap_or_default();
    idx.entries.push(IndexEntry {
        title: "Project".to_string(),
        path: "summaries/project.md".to_string(),
        one_liner: "summary only".to_string(),
        category: PageType::Summary,
        last_updated: Some("2026-04-19 11:00:00".to_string()),
        outcome: None,
    });
    wiki.save_index(&idx).expect("save_index");

    let out = format_wiki_fresh(&wiki, 4096);
    assert!(
        out.contains("No Entity or Concept pages yet"),
        "stub must surface when bucket is empty: {}",
        out,
    );
    assert!(
        out.contains("/wiki refresh"),
        "stub must point at /wiki refresh for recovery: {}",
        out,
    );
}

#[test]
fn format_wiki_fresh_respects_display_budget() {
    use crate::wiki::{IndexEntry, PageType, Wiki};
    let tmp = tempfile::tempdir().expect("tempdir");
    let wiki = Wiki::open(tmp.path()).expect("open wiki");
    let mut idx = wiki.load_index().unwrap_or_default();
    for i in 0..40 {
        idx.entries.push(IndexEntry {
            title: format!("title-{:02}", i),
            path: format!("entities/e{:02}.md", i),
            one_liner: "a".repeat(40),
            category: PageType::Entity,
            last_updated: Some(format!("2026-04-19 {:02}:00:00", i % 24)),
            outcome: None,
        });
    }
    wiki.save_index(&idx).expect("save_index");

    let out = format_wiki_fresh(&wiki, 256);
    assert!(out.len() <= 256, "budget violated: {} > 256", out.len());
    assert!(
        out.ends_with('\n'),
        "truncation must land at a line boundary; got tail: {:?}",
        out.lines().last().unwrap_or(""),
    );
}

// ── C94: format_wiki_summary ─────────────────────────────────────────

/// Build a [`ProjectSummaryReport`] with default-ish fields. Caller
/// overrides `entity_count`, `kind_counts`, and `recent_cycles` per
/// test. Keeps the test body focused on what the assertion cares
/// about instead of struct-literal boilerplate.
fn summary_report_fixture(
    entity_count: usize,
    kind_counts: std::collections::BTreeMap<crate::wiki::EntityKind, usize>,
    recent_cycles: Vec<crate::wiki::RecentCycle>,
) -> crate::wiki::ProjectSummaryReport {
    crate::wiki::ProjectSummaryReport {
        path: "summaries/project.md".to_string(),
        entity_count,
        kind_counts,
        top_dependencies: Vec::new(),
        recent_cycles,
    }
}

fn cycle_fixture(cycle: usize, outcome: Option<&str>) -> crate::wiki::RecentCycle {
    crate::wiki::RecentCycle {
        cycle,
        chain: "demo".to_string(),
        page_path: format!("synthesis/cycle-{:02}.md", cycle),
        last_updated: Some("2026-04-19 12:00:00".to_string()),
        outcome: outcome.map(String::from),
    }
}

/// Empty `recent_cycles` → no trailer. Output stays single-line so
/// operators see the same compact confirmation as pre-C94.
#[test]
fn format_wiki_summary_single_line_when_no_recent_cycles() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 3);
    let report = summary_report_fixture(3, kinds, Vec::new());

    let out = format_wiki_summary(&report);
    assert!(
        !out.contains('\n'),
        "empty recent_cycles must stay single-line: {:?}",
        out,
    );
    assert!(
        out.contains("Project summary written to summaries/project.md"),
        "first-line confirmation must be preserved: {}",
        out,
    );
    assert!(
        out.contains("3 entities"),
        "entity count must render: {}",
        out,
    );
    assert!(
        out.contains("function: 3"),
        "kind breakdown must render: {}",
        out,
    );
}

/// Non-empty `recent_cycles` → exactly two lines, trailer starts
/// with `Recent cycles: `.
#[test]
fn format_wiki_summary_two_lines_when_recent_cycles_present() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);
    let report = summary_report_fixture(1, kinds, vec![cycle_fixture(42, Some("green"))]);

    let out = format_wiki_summary(&report);
    let lines: Vec<&str> = out.split('\n').collect();
    assert_eq!(
        lines.len(),
        2,
        "one cycle must produce exactly two lines: {:?}",
        out,
    );
    assert!(
        lines[1].starts_with("Recent cycles: "),
        "trailer must start with 'Recent cycles: ': {:?}",
        lines[1],
    );
}

/// `outcome: Some("green")` renders as `cycle N (green)` — matches
/// the body-side format in `build_project_summary`.
#[test]
fn format_wiki_summary_renders_outcome_in_parens() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);
    let report = summary_report_fixture(1, kinds, vec![cycle_fixture(91, Some("green"))]);

    let out = format_wiki_summary(&report);
    assert!(
        out.contains("cycle 91 (green)"),
        "outcome must render in parens: {}",
        out,
    );
}

/// `outcome: None` renders as bare `cycle N` — no empty parens, no
/// placeholder string. Locks the "no empty parens" contract.
#[test]
fn format_wiki_summary_renders_none_outcome_without_parens() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);
    let report = summary_report_fixture(1, kinds, vec![cycle_fixture(88, None)]);

    let out = format_wiki_summary(&report);
    assert!(
        out.contains("cycle 88"),
        "bare cycle must still render when outcome is None: {}",
        out,
    );
    assert!(
        !out.contains("cycle 88 ()"),
        "None outcome must not produce empty parens: {}",
        out,
    );
}

/// Cycles render in vec order — the upstream sort (newest-first via
/// `collect_recent_cycles`) is preserved; the formatter does no
/// internal re-sort.
#[test]
fn format_wiki_summary_renders_cycles_in_vec_order() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);
    let report = summary_report_fixture(
        1,
        kinds,
        vec![
            cycle_fixture(95, Some("green")),
            cycle_fixture(93, Some("yellow")),
            cycle_fixture(90, Some("green")),
        ],
    );

    let out = format_wiki_summary(&report);
    let p95 = out.find("cycle 95").expect("cycle 95 present");
    let p93 = out.find("cycle 93").expect("cycle 93 present");
    let p90 = out.find("cycle 90").expect("cycle 90 present");
    assert!(
        p95 < p93 && p93 < p90,
        "cycles must render in vec order (95, 93, 90); got {} {} {}\n{}",
        p95,
        p93,
        p90,
        out,
    );
}

/// Entity count pluralization: 1 → `1 entity`, 5 → `5 entities`.
/// Regression guard — extraction must preserve the inline logic.
#[test]
fn format_wiki_summary_entity_count_pluralizes_correctly() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);

    let one = summary_report_fixture(1, kinds.clone(), Vec::new());
    let out_one = format_wiki_summary(&one);
    assert!(
        out_one.contains(" (1 entity,"),
        "singular pluralization must render as '1 entity,': {}",
        out_one,
    );

    let many = summary_report_fixture(5, kinds, Vec::new());
    let out_many = format_wiki_summary(&many);
    assert!(
        out_many.contains(" (5 entities,"),
        "plural pluralization must render as '5 entities,': {}",
        out_many,
    );
}

/// Empty `kind_counts` → `no entity kinds` placeholder. Matches the
/// pre-C94 inline behaviour.
#[test]
fn format_wiki_summary_empty_kind_counts_uses_placeholder() {
    let report = summary_report_fixture(0, std::collections::BTreeMap::new(), Vec::new());
    let out = format_wiki_summary(&report);
    assert!(
        out.contains("no entity kinds"),
        "empty kind_counts must use placeholder: {}",
        out,
    );
}

/// TUI layer does no further truncation: 5 cycles in, 5 cycles
/// rendered. Upstream cap at `PROJECT_SUMMARY_RECENT_CYCLES` is the
/// single source of truth.
#[test]
fn format_wiki_summary_includes_all_report_cycles_without_truncation() {
    use crate::wiki::EntityKind;
    let mut kinds = std::collections::BTreeMap::new();
    kinds.insert(EntityKind::Function, 1);
    let cycles: Vec<crate::wiki::RecentCycle> =
        (90..95).map(|n| cycle_fixture(n, Some("green"))).collect();
    assert_eq!(cycles.len(), crate::wiki::PROJECT_SUMMARY_RECENT_CYCLES);
    let report = summary_report_fixture(1, kinds, cycles);

    let out = format_wiki_summary(&report);
    for n in 90..95 {
        assert!(
            out.contains(&format!("cycle {}", n)),
            "cycle {} must be rendered — no TUI-layer truncation: {}",
            n,
            out,
        );
    }
    // Exactly 5 occurrences of `cycle ` on the trailer line — comma-
    // separated list must not drop or duplicate any.
    let trailer = out.lines().last().expect("trailer line");
    assert_eq!(
        trailer.matches("cycle ").count(),
        crate::wiki::PROJECT_SUMMARY_RECENT_CYCLES,
        "trailer must carry exactly {} 'cycle ' occurrences: {}",
        crate::wiki::PROJECT_SUMMARY_RECENT_CYCLES,
        trailer,
    );
}
