//! Phase 1 Validation Gate — 500-turn compaction stress test.
//!
//! These two integration tests sit in `tests/` (separate binary from the
//! unit tests in `src/compaction.rs`) so they exercise the public pipeline
//! through the same boundary an end-user's session would — no `#[cfg(test)]`
//! shortcuts. `mockito` stands in for the Ollama server.
//!
//! ─── Gate criteria (test 1 — happy path) ───
//!  * Pipeline runs 500 turns without panic.
//!  * System prompt at index 0 survives every iteration.
//!  * Graceful stages fire repeatedly (Microcompact + at least one of
//!    {`SessionMemory`, `FullSummary`}) — proves compaction actually triggers
//!    under sustained growth rather than silently falling through.
//!  * `failures == 0` — every Stage 3 attempt (canned summary mock) succeeded.
//!  * Message count + estimated tokens stay bounded throughout and at the
//!    end — no unbounded growth, no runaway memory.
//!  * At least one `.dm/wiki/synthesis/compact-*.md` lands — proves the
//!    compact-to-wiki bridge fires in integration, not just unit tests.
//!
//! Note on configuration: a 2 000-token window with `keep_tail = 3` was
//! chosen because the default (100 000 tokens, `keep_tail = 10`) cannot
//! trigger Stage 3 organically — post-Stage-2 tail tokens would need to
//! exceed `80 000 × 80 % = 64 000`, which 10 tail messages of realistic
//! size never reach. The stress properties (bounded growth, system
//! preservation, wiki persistence) hold under any window size.
//!
//! ─── Gate criteria (test 2 — circuit breaker) ───
//!  * With Stage 3 returning HTTP 500, the failure counter crosses the
//!    circuit-breaker threshold (≥ 3).
//!  * The Emergency stage fires at least once.
//!  * Emergency still writes a wiki synthesis page (no LLM call needed).
//!  * System prompt still survives.
//!  * The Ollama mock receives no more than a small bounded number of
//!    hits — proof the breaker stops hammering after tripping.

use std::path::Path;
use std::sync::Mutex;

use dark_matter::compaction::{
    compact_pipeline_with_failures, estimate_tokens, CompactionStage, CompactionThresholds,
};
use dark_matter::ollama::client::OllamaClient;

/// Serialize tests that mutate the process-wide `DM_WIKI_AUTO_INGEST` env
/// var. The default behavior is "enabled", so setting `"1"` is belt-and-
/// suspenders — but the guard still restores the previous value on drop so
/// state can't leak into neighboring test binaries.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct WikiEnvGuard {
    prev: Option<String>,
}

impl WikiEnvGuard {
    fn enable() -> Self {
        let prev = std::env::var("DM_WIKI_AUTO_INGEST").ok();
        std::env::set_var("DM_WIKI_AUTO_INGEST", "1");
        Self { prev }
    }
}

impl Drop for WikiEnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var("DM_WIKI_AUTO_INGEST", v),
            None => std::env::remove_var("DM_WIKI_AUTO_INGEST"),
        }
    }
}

/// Count `compact-*.md` synthesis pages — both Stage-3 synthesis and
/// Emergency synthesis land here, so a non-zero count proves compaction
/// preserved *something* into the wiki.
fn count_synthesis_pages(wiki_synth_dir: &Path) -> usize {
    match std::fs::read_dir(wiki_synth_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
            .filter(|e| e.file_name().to_string_lossy().starts_with("compact-"))
            .count(),
        Err(_) => 0,
    }
}

fn make_system_prompt() -> serde_json::Value {
    serde_json::json!({
        "role": "system",
        "content": "STRESS_TEST_SYSTEM_PROMPT — must survive every compaction"
    })
}

fn system_prompt_intact(messages: &[serde_json::Value]) -> bool {
    matches!(
        messages.first().and_then(|m| m["role"].as_str()),
        Some("system")
    )
}

/// 7-cycle: 6 user/asst turns then 1 tool turn. Each user/asst body is
/// 3 600 chars (~900 tokens) so a `keep_tail = 3` tail of three non-tool
/// messages can push post-Stage-2 tokens over `FULL_COMPACT_PCT × 2 000
/// = 1 600`, exercising Stage 3. Tool bodies are 3 600 chars — more than
/// double the 500-char microcompact cap — so Stage 1 has real work every
/// seventh turn.
fn make_turn_message(turn: usize) -> serde_json::Value {
    if turn.is_multiple_of(7) {
        serde_json::json!({
            "role": "tool",
            "name": "bash",
            "content": format!("tool turn {} — {}", turn, "t".repeat(3600))
        })
    } else if turn % 2 == 1 {
        serde_json::json!({
            "role": "user",
            "content": format!("user turn {} — {}", turn, "u".repeat(3600))
        })
    } else {
        serde_json::json!({
            "role": "assistant",
            "content": format!("assistant turn {} — {}", turn, "a".repeat(3600))
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// TEST 1 — 500-turn happy-path stress
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn stress_500_turns_exercises_all_stages_and_preserves_coherence() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _env = WikiEnvGuard::enable();

    let project_tmp = tempfile::TempDir::new().expect("tempdir");
    let project_root = project_tmp.path().to_path_buf();

    // Canned Stage-3 summary response. The body is deliberately tiny so
    // the post-Stage-3 message list is dominated by system prompt +
    // summary + tail — exactly the coherent shape the gate asserts.
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"message":{"role":"assistant","content":"[stress-test canned summary — 500-turn happy path]"},"prompt_eval_count":0,"eval_count":0,"eval_duration":0}"#,
        )
        .expect_at_least(1)
        .create_async()
        .await;

    let client = OllamaClient::new(server.url(), "stress-model".to_string());

    // 2 000-token window with `keep_tail = 3` — small enough that tail
    // tokens can cross `FULL_COMPACT_PCT × window = 1 600` when the tail
    // happens to hold 2–3 large user/asst messages. See module docs for
    // why the planner's nominal 100k config can't reach Stage 3.
    let thresholds = CompactionThresholds::from_context_window(2000).with_keep_tail(3);

    let mut messages: Vec<serde_json::Value> = vec![make_system_prompt()];
    let mut failures: usize = 0;

    let mut micro_count = 0usize;
    let mut session_count = 0usize;
    let mut full_count = 0usize;
    let mut emergency_count = 0usize;
    let mut none_count = 0usize;

    let mut max_len_seen = messages.len();
    let mut max_tokens_seen = estimate_tokens(&messages);

    for turn in 0..500 {
        messages.push(make_turn_message(turn));

        let result = compact_pipeline_with_failures(
            &mut messages,
            &client,
            &thresholds,
            false,
            &mut failures,
            Some(&project_root),
        )
        .await;

        if result.replace_session {
            messages = result.messages;
        }

        match result.stage {
            CompactionStage::Microcompact { .. } => micro_count += 1,
            CompactionStage::SessionMemory { .. } => session_count += 1,
            CompactionStage::FullSummary { .. } => full_count += 1,
            CompactionStage::Emergency => emergency_count += 1,
            CompactionStage::None => none_count += 1,
        }

        if messages.len() > max_len_seen {
            max_len_seen = messages.len();
        }
        let toks = estimate_tokens(&messages);
        if toks > max_tokens_seen {
            max_tokens_seen = toks;
        }

        assert!(
            system_prompt_intact(&messages),
            "turn {turn}: system prompt missing from messages[0] after stage {:?}",
            result.stage,
        );
    }

    // Persist tally to disk (no `eprintln!`) so the tester / planner has
    // a baseline of how the 500-turn run actually distributes across
    // stages. Read by hand or parsed by the tester.
    let tally = format!(
        "stress_500_turns tally\n\
         micro={micro_count}\n\
         session={session_count}\n\
         full={full_count}\n\
         emergency={emergency_count}\n\
         none={none_count}\n\
         max_len_seen={max_len_seen}\n\
         max_tokens_seen={max_tokens_seen}\n\
         final_len={}\n\
         final_tokens={}\n\
         failures={failures}\n",
        messages.len(),
        estimate_tokens(&messages),
    );
    std::fs::write(project_root.join("stress_tally.txt"), &tally).ok();
    if let Ok(dest) = std::env::var("DM_STRESS_TALLY_DIR") {
        std::fs::write(Path::new(&dest).join("stress_tally.txt"), &tally).ok();
    }

    assert!(
        micro_count >= 1,
        "Microcompact never fired over 500 turns — tally: {tally}"
    );
    assert!(
        session_count + full_count >= 1,
        "Neither SessionMemory nor FullSummary fired over 500 turns — tally: {tally}"
    );
    assert_eq!(
        failures, 0,
        "Stage 3 failure counter should stay 0 on happy path (canned mock) — tally: {tally}",
    );

    assert!(
        system_prompt_intact(&messages),
        "final system prompt missing — tally: {tally}",
    );
    assert!(
        messages.len() <= 150,
        "final message count {} breached 150-msg bound — tally: {tally}",
        messages.len(),
    );
    assert!(
        max_len_seen <= 200,
        "peak message count {max_len_seen} breached 200-msg bound — tally: {tally}",
    );
    assert!(
        estimate_tokens(&messages) <= 2000,
        "final token estimate {} breached 2000-token window — tally: {tally}",
        estimate_tokens(&messages),
    );

    let synth_dir = project_root.join(".dm/wiki/synthesis");
    let pages = count_synthesis_pages(&synth_dir);
    assert!(
        pages >= 1,
        "expected at least one wiki synthesis page under {:?}, got {} — tally: {tally}",
        synth_dir,
        pages,
    );
}

// ────────────────────────────────────────────────────────────────────────
// TEST 2 — circuit breaker trip + emergency wiki synthesis
// ────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn stress_circuit_breaker_trips_and_emergency_writes_wiki_synthesis() {
    let _env_lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _env = WikiEnvGuard::enable();

    let project_tmp = tempfile::TempDir::new().expect("tempdir");
    let project_root = project_tmp.path().to_path_buf();

    // Every Stage-3 attempt must fail. We use 502 Bad Gateway (non-
    // retriable per `is_retriable` in src/ollama/client.rs) so each
    // logical failure costs exactly 1 HTTP hit — if we used 500 the
    // client's internal `MAX_RETRIES = 4` backoff loop would amplify
    // each logical failure to 5 HTTP attempts AND inject multi-second
    // exponential sleeps, making the test slow and the mock bound
    // mushy. The retry logic has its own unit tests; here we isolate
    // the circuit breaker: trip after 3 logical failures, stop hitting
    // Ollama. A ceiling of 5 hits across 10 loop iterations means the
    // breaker must have engaged — without it the loop would fire many
    // more Stage 3 attempts.
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat")
        .with_status(502)
        .with_body("simulated model failure")
        .expect_at_least(1)
        .expect_at_most(5)
        .create_async()
        .await;

    let client = OllamaClient::new(server.url(), "broken-model".to_string());
    // Small window so each iteration's refill immediately pushes over
    // `full_compact` and forces Stage 3 to be attempted (until the
    // breaker trips).
    let thresholds = CompactionThresholds::from_context_window(2000).with_keep_tail(3);

    let mut messages: Vec<serde_json::Value> = vec![make_system_prompt()];
    let mut failures: usize = 0;
    let mut emergency_count = 0usize;
    let mut stage3_attempts = 0usize;

    for iter in 0..10 {
        // Refill to guarantee pre-pipeline tokens are above `full_compact`
        // — otherwise the breaker has nothing to engage.
        while messages.len() < 10 {
            messages.push(make_turn_message(messages.len()));
        }

        let before_failures = failures;

        let result = compact_pipeline_with_failures(
            &mut messages,
            &client,
            &thresholds,
            false,
            &mut failures,
            Some(&project_root),
        )
        .await;

        if result.replace_session {
            messages = result.messages;
        }

        if failures > before_failures {
            stage3_attempts += 1;
        }

        if matches!(result.stage, CompactionStage::Emergency) {
            emergency_count += 1;
        }

        assert!(
            system_prompt_intact(&messages),
            "iter {iter}: system prompt missing after stage {:?}",
            result.stage,
        );
    }

    let tally = format!(
        "circuit_breaker tally\n\
         failures={failures}\n\
         emergency_count={emergency_count}\n\
         stage3_attempts={stage3_attempts}\n\
         final_len={}\n\
         final_tokens={}\n",
        messages.len(),
        estimate_tokens(&messages),
    );
    std::fs::write(project_root.join("breaker_tally.txt"), &tally).ok();
    if let Ok(dest) = std::env::var("DM_STRESS_TALLY_DIR") {
        std::fs::write(Path::new(&dest).join("breaker_tally.txt"), &tally).ok();
    }

    assert!(
        failures >= 3,
        "failure counter should cross the breaker threshold (3) — tally: {tally}",
    );
    assert!(
        emergency_count >= 1,
        "Emergency stage never fired — tally: {tally}",
    );
    assert!(
        system_prompt_intact(&messages),
        "final system prompt missing — tally: {tally}",
    );

    let synth_dir = project_root.join(".dm/wiki/synthesis");
    let pages = count_synthesis_pages(&synth_dir);
    assert!(
        pages >= 1,
        "emergency compact should have written at least one synthesis page under {:?}, got {} — tally: {tally}",
        synth_dir,
        pages,
    );

    // Mockito panics inside `assert_async` if hit count falls outside
    // `[expect_at_least, expect_at_most]`, giving us the direct assertion
    // that the breaker stopped hammering Ollama.
    mock.assert_async().await;
}
