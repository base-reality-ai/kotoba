//! Acceptance test for Tier 3 — agents calling wiki tools mid-conversation
//! (the second-brain LOOP).
//!
//! For the second brain to compound, the persona must recall what the
//! learner has previously encountered before answering. The kernel
//! registers `wiki_search` / `wiki_lookup` by default and dm's session
//! path injects their hints into `<tool_usage>` (canonical
//! `tools::registry::system_prompt_hints_for`). That gives the model
//! tool *availability*; this test pins kotoba's tool *usage policy*:
//!
//! 1. Vocabulary written via `host_log_vocabulary` actually lands in the
//!    wiki index AND is found by canonical `Wiki::search` — proving the
//!    second-brain pipeline (host write → kernel index → wiki_search hit)
//!    works end-to-end in host mode.
//! 2. The persona system prompt kotoba builds for a session contains the
//!    recall-guidance marker AND explicitly names `wiki_search` /
//!    `wiki_lookup`, so the persona model is steered toward recall
//!    before introducing material fresh.
//! 3. The persona body (when seeded with the wiki entity page) is
//!    preserved verbatim inside the prompt so its voice/scope sections
//!    remain.
//!
//! Together these pin the structural guarantee: a kotoba session has
//! both the *capacity* (registered wiki tools) and the *instruction*
//! (recall-first guidance) for the second brain to compound.

use dark_matter::tools::Tool;
use dark_matter::wiki::Wiki;
use serde_json::json;
use std::sync::Mutex;
use tempfile::TempDir;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

static CWD_LOCK: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn pre_seeded_vocab_is_findable_by_wiki_search() {
    let _cwd_guard = CWD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let project = TempDir::new().expect("tempdir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");

    Wiki::open(project.path()).expect("open wiki");

    // Seed the wiki the way a prior session would: the persona logs
    // 学校 (がっこう) as the learner encounters it.
    let tool = host_caps::LogVocabularyTool;
    tool.call(json!({
        "kanji": "学校",
        "kana": "がっこう",
        "romaji": "gakkou",
        "meaning": "school",
        "pos": "noun",
    }))
    .await
    .expect("seed vocab");

    // Persona-side recall path: the model would call `wiki_search` with
    // the user-mentioned term. Canonical `Wiki::search` is what the tool
    // delegates to — so a hit here is what the persona will see.
    let wiki = Wiki::open(project.path()).expect("re-open wiki");
    let hits = wiki.search("school").expect("wiki_search");
    assert!(
        hits.iter().any(|h| h.path.contains("学校")),
        "wiki_search('school') should surface the seeded vocab; got: {:?}",
        hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );

    if let Some(prior) = prior_cwd {
        std::env::set_current_dir(prior).expect("restore cwd");
    }
}

#[test]
fn persona_system_prompt_includes_recall_guidance_and_wiki_tool_names() {
    let persona_body = "# Yuki\n\nA patient Japanese tutor.\n\n## Voice\n\n- Polite\n";
    let brief = "Today's focus: café ordering.";

    let prompt = host_caps::build_persona_system_prompt("Yuki", persona_body, brief);

    // Persona section preserved verbatim — voice/scope guidance survives
    // the wrapper.
    assert!(
        prompt.contains(persona_body),
        "persona body should appear in the system prompt verbatim"
    );

    // Recall guidance marker present.
    assert!(
        prompt.contains(host_caps::PERSONA_RECALL_GUIDANCE_MARKER),
        "system prompt should include the recall-guidance marker '{}'; got:\n{}",
        host_caps::PERSONA_RECALL_GUIDANCE_MARKER,
        prompt
    );

    // Both wiki tools named explicitly so the persona model knows which
    // tool to reach for.
    assert!(
        prompt.contains("wiki_search"),
        "recall guidance should name `wiki_search`; got:\n{}",
        prompt
    );
    assert!(
        prompt.contains("wiki_lookup"),
        "recall guidance should mention `wiki_lookup` as the by-path \
         alternative; got:\n{}",
        prompt
    );

    // The recall-first directive shapes when the model reaches for the
    // tool; without this phrasing the model knows the tool exists but
    // not when to use it.
    assert!(
        prompt.contains("Recall first"),
        "guidance should explicitly say 'recall first' (not just 'wiki \
         tools available'); got:\n{}",
        prompt
    );

    // Brief is the last block — planner output is the freshest state
    // the persona should weight.
    assert!(
        prompt.trim_end().ends_with(brief.trim_end()),
        "planner brief should be the closing block of the prompt; got:\n{}",
        prompt
    );
}

#[test]
fn persona_system_prompt_handles_persona_body_without_trailing_newline() {
    // Defensive: persona bodies read from the wiki may or may not end
    // with a newline. The helper must not run the persona section into
    // the recall guidance.
    let persona_body = "# Hiro\n\nCasual study buddy.";
    let brief = "Brief.";

    let prompt = host_caps::build_persona_system_prompt("Hiro", persona_body, brief);

    let recall_idx = prompt
        .find(host_caps::PERSONA_RECALL_GUIDANCE_MARKER)
        .expect("recall marker present");
    let between = &prompt[..recall_idx];
    assert!(
        between.ends_with("\n\n"),
        "persona body should be separated from recall guidance by a \
         blank line; got tail: {:?}",
        &between[between.len().saturating_sub(8)..]
    );
}
