//! Phase 2.5 Slice C — session-attached SSE at POST /`chat/stream/:session_id`.
//!
//! Uses the `MockOllama` server from Slice B. Proves:
//!   1. An existing file-backed session is loaded, the turn streams, and on
//!      `Done` both the user prompt and the accumulated assistant text are
//!      appended to the session file on disk.
//!   2. A missing session returns HTTP 404 (same semantics as POST /chat
//!      with `session_id`).

mod common;

use common::mock_ollama::MockOllama;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use dark_matter::api::{build_router, replay, ApiState};
use dark_matter::config::Config;
use dark_matter::session::{storage, Session};
use http_body_util::BodyExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tower::ServiceExt;

fn make_state(config_dir: std::path::PathBuf, host: String) -> ApiState {
    ApiState {
        config: Arc::new(Config {
            host,
            host_is_default: false,
            model: "mock-model".into(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".into(),
            global_config_dir: config_dir.clone(),
            config_dir,
            routing: None,
            aliases: HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        }),
        token: None,
        active_chains: Arc::new(Mutex::new(HashMap::new())),
        reconnect_buffers: replay::new_buffers(),
        wiki_snippet: None,
        wiki_fresh: None,
        wiki_cwd: None,
        wiki_summary: Arc::new(RwLock::new(None)),
        wiki_snippet_bytes_injected: 0,
    }
}

#[tokio::test]
async fn sse_stream_session_attaches_and_appends_messages() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "test-session-sse-c";

    // Seed a session on disk with a single system message.
    let now = chrono::Utc::now();
    let session = Session {
        id: session_id.into(),
        title: None,
        created_at: now,
        updated_at: now,
        cwd: "/tmp".into(),
        host_project: None,
        model: "mock-model".into(),
        messages: vec![serde_json::json!({"role": "system", "content": "seed"})],
        compact_failures: 0,
        turn_count: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        parent_id: None,
    };
    storage::save(tmp.path(), &session).expect("seed save");

    // Mock upstream emits 2 tokens + Done.
    let mock = MockOllama::new()
        .emit_token("hi")
        .emit_token(" again")
        .emit_done(5, 2)
        .spawn()
        .await;

    let state = make_state(tmp.path().to_path_buf(), mock.host());
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"hello"}"#))
                .unwrap(),
        )
        .await
        .expect("oneshot");

    assert_eq!(response.status(), StatusCode::OK);

    let body = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let text = String::from_utf8(body.to_vec()).expect("utf-8 body");

    // Wire: 2 token events + 1 done = ids 1..=3.
    let ids: Vec<&str> = text.lines().filter(|l| l.starts_with("id:")).collect();
    assert_eq!(ids, vec!["id: 1", "id: 2", "id: 3"], "wire: {}", text);
    assert_eq!(text.matches("event: token").count(), 2);
    assert_eq!(text.matches("event: done").count(), 1);

    // Session file now has seed + user + assistant.
    let reloaded = storage::load(tmp.path(), session_id).expect("reload");
    assert_eq!(
        reloaded.messages.len(),
        3,
        "seed + user + assistant; messages were: {:?}",
        reloaded.messages
    );
    assert_eq!(reloaded.messages[1]["role"], "user");
    assert_eq!(reloaded.messages[1]["content"], "hello");
    assert_eq!(reloaded.messages[2]["role"], "assistant");
    assert_eq!(reloaded.messages[2]["content"], "hi again");
}

#[tokio::test]
async fn last_event_id_replays_missed_frames_and_closes() {
    use common::mock_ollama::MockChunk;

    // Seed a session.
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "test-session-sse-replay";
    let now = chrono::Utc::now();
    let session = Session {
        id: session_id.into(),
        title: None,
        created_at: now,
        updated_at: now,
        cwd: "/tmp".into(),
        host_project: None,
        model: "mock-model".into(),
        messages: vec![serde_json::json!({"role": "system", "content": "seed"})],
        compact_failures: 0,
        turn_count: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        parent_id: None,
    };
    storage::save(tmp.path(), &session).expect("seed save");

    // Channel-driven mock: we push 4 tokens + 1 done = 5 SSE frames total
    // (ids 1..=5). The replay ring must capture all 5.
    let (mock, tx) = MockOllama::spawn_channel().await;
    tokio::spawn(async move {
        tx.send(MockChunk::Token("a".into())).await.unwrap();
        tx.send(MockChunk::Token("b".into())).await.unwrap();
        tx.send(MockChunk::Token("c".into())).await.unwrap();
        tx.send(MockChunk::Token("d".into())).await.unwrap();
        tx.send(MockChunk::Done {
            prompt_eval_count: 5,
            eval_count: 4,
        })
        .await
        .unwrap();
    });

    let state = make_state(tmp.path().to_path_buf(), mock.host());
    let app = build_router(state.clone());

    // First request: fresh turn. Drain body; tap fills the ring.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"hi"}"#))
                .unwrap(),
        )
        .await
        .expect("initial oneshot");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.expect("drain initial");

    // Verify ring now holds exactly 5 frames with ids 1..=5.
    let ring = replay::get_or_create_ring(&state.reconnect_buffers, session_id);
    {
        let g = ring.lock().expect("ring lock");
        let all = g.frames_after(0);
        assert_eq!(all.len(), 5, "ring should hold 5 frames; had: {:?}", all);
        let ids: Vec<u64> = all.iter().map(|f| f.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(all[4].event, "done");
    }

    // Reconnect with `Last-Event-ID: 2`. Build a fresh app on the SAME
    // state so the reconnect_buffers map is shared.
    let app2 = build_router(state.clone());
    let response = app2
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .header("last-event-id", "2")
                .body(Body::from(r#"{"prompt":"ignored-on-replay"}"#))
                .unwrap(),
        )
        .await
        .expect("replay oneshot");
    assert_eq!(response.status(), StatusCode::OK);
    let body = response
        .into_body()
        .collect()
        .await
        .expect("drain replay")
        .to_bytes();
    let text = String::from_utf8(body.to_vec()).expect("utf-8 body");

    // Wire: ids 3,4,5 (the missed frames) + id 6 (replay_complete marker).
    let ids: Vec<&str> = text.lines().filter(|l| l.starts_with("id:")).collect();
    assert_eq!(
        ids,
        vec!["id: 3", "id: 4", "id: 5", "id: 6"],
        "replay wire: {}",
        text
    );
    // Tail after id=2 is tokens c+d (ids 3,4) + done (id 5) = 3 frames.
    assert_eq!(
        text.matches("event: token").count(),
        2,
        "2 missed tokens; wire: {}",
        text
    );
    assert_eq!(
        text.matches("event: done").count(),
        1,
        "original done replayed; wire: {}",
        text
    );
    assert_eq!(
        text.matches("event: replay_complete").count(),
        1,
        "replay_complete marker; wire: {}",
        text
    );
    assert!(
        text.contains("\"resumed_from\":2"),
        "marker carries resumed_from; wire: {}",
        text
    );
    assert!(
        text.contains("\"replayed\":3"),
        "marker carries replayed count; wire: {}",
        text
    );
}

#[tokio::test]
async fn last_event_id_with_no_prior_turn_returns_204() {
    // Session exists on disk but has never been streamed — no ring entry
    // in state.reconnect_buffers. Reconnect with Last-Event-ID must 204
    // short-circuit, without contacting Ollama (no mock needed), without
    // loading the session, and without creating a ring.
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "test-session-no-prior-turn";
    let now = chrono::Utc::now();
    let session = Session {
        id: session_id.into(),
        title: None,
        created_at: now,
        updated_at: now,
        cwd: "/tmp".into(),
        host_project: None,
        model: "mock-model".into(),
        messages: vec![serde_json::json!({"role": "system", "content": "seed"})],
        compact_failures: 0,
        turn_count: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        parent_id: None,
    };
    storage::save(tmp.path(), &session).expect("seed save");

    // Bogus host — if the handler tries to reach Ollama, the test fails.
    let state = make_state(tmp.path().to_path_buf(), "127.0.0.1:1".into());
    let app = build_router(state.clone());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .header("last-event-id", "5")
                .body(Body::from(r#"{"prompt":"irrelevant"}"#))
                .unwrap(),
        )
        .await
        .expect("oneshot");

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    // Ring map must not have gained an entry for this session.
    let map = state.reconnect_buffers.lock().expect("lock");
    assert!(
        !map.contains_key(session_id),
        "no-prior-turn replay must not create a ring"
    );
}

#[tokio::test]
async fn last_event_id_at_high_water_mark_returns_204() {
    use common::mock_ollama::MockChunk;

    // Drive a full turn (4 tokens + done = 5 frames, ids 1..=5), then
    // reconnect with Last-Event-ID pointing at the highest id (5) and
    // also at a value beyond the high-water mark. Both must 204.
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "test-session-high-water";
    let now = chrono::Utc::now();
    let session = Session {
        id: session_id.into(),
        title: None,
        created_at: now,
        updated_at: now,
        cwd: "/tmp".into(),
        host_project: None,
        model: "mock-model".into(),
        messages: vec![serde_json::json!({"role": "system", "content": "seed"})],
        compact_failures: 0,
        turn_count: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        parent_id: None,
    };
    storage::save(tmp.path(), &session).expect("seed save");

    let (mock, tx) = MockOllama::spawn_channel().await;
    tokio::spawn(async move {
        tx.send(MockChunk::Token("a".into())).await.unwrap();
        tx.send(MockChunk::Token("b".into())).await.unwrap();
        tx.send(MockChunk::Token("c".into())).await.unwrap();
        tx.send(MockChunk::Token("d".into())).await.unwrap();
        tx.send(MockChunk::Done {
            prompt_eval_count: 5,
            eval_count: 4,
        })
        .await
        .unwrap();
    });

    let state = make_state(tmp.path().to_path_buf(), mock.host());
    let app = build_router(state.clone());

    // Initial turn drains to completion — ring now holds ids 1..=5.
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"hi"}"#))
                .unwrap(),
        )
        .await
        .expect("initial oneshot");
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.expect("drain initial");

    // Sanity: ring actually filled to 5.
    let ring = replay::get_or_create_ring(&state.reconnect_buffers, session_id);
    assert_eq!(ring.lock().expect("ring").len(), 5);

    // Reconnect at the high-water mark: nothing newer than id 5.
    let app2 = build_router(state.clone());
    let response = app2
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .header("last-event-id", "5")
                .body(Body::from(r#"{"prompt":"ignored"}"#))
                .unwrap(),
        )
        .await
        .expect("at-high-water oneshot");
    assert_eq!(
        response.status(),
        StatusCode::NO_CONTENT,
        "Last-Event-ID at high-water mark must 204"
    );

    // Reconnect well above high-water: same 204 semantics.
    let app3 = build_router(state.clone());
    let response = app3
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/chat/stream/{}", session_id))
                .header("content-type", "application/json")
                .header("last-event-id", "999")
                .body(Body::from(r#"{"prompt":"ignored"}"#))
                .unwrap(),
        )
        .await
        .expect("above-high-water oneshot");
    assert_eq!(
        response.status(),
        StatusCode::NO_CONTENT,
        "Last-Event-ID above high-water mark must 204"
    );
}

#[tokio::test]
async fn sse_stream_session_returns_404_for_missing_session() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mock = MockOllama::new().emit_done(0, 0).spawn().await;

    let state = make_state(tmp.path().to_path_buf(), mock.host());
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chat/stream/does-not-exist")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"x"}"#))
                .unwrap(),
        )
        .await
        .expect("oneshot");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// Minimal non-streaming `/api/chat` mock — returns a single JSON body
/// matching Ollama's `"stream": false` shape. `MockOllama` serves NDJSON
/// which `client.chat()` (non-streaming) can't parse.
async fn spawn_nonstream_chat_mock() -> (String, tokio::task::JoinHandle<()>) {
    use axum::routing::post;
    use axum::Router;
    use tokio::net::TcpListener;
    let app = Router::new().route(
        "/api/chat",
        post(|| async {
            axum::Json(serde_json::json!({
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [],
                },
                "done": true,
                "prompt_eval_count": 3,
                "eval_count": 1,
                "total_duration": 1_000_000u64,
                "load_duration": 0u64,
                "prompt_eval_duration": 500_000u64,
                "eval_duration": 500_000u64,
            }))
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("{}", addr), task)
}

#[tokio::test]
async fn sessions_turn_uses_cached_wiki_snippet_for_first_system_prompt() {
    // P2 regression guard: the non-streaming session-turn handler must
    // thread `ApiState.wiki_snippet` into the first system prompt it
    // builds when the session has no messages yet. A marker in the
    // snippet must appear verbatim in the persisted system message.
    let tmp = tempfile::tempdir().expect("tempdir");
    let session_id = "test-session-wiki-snippet";

    let now = chrono::Utc::now();
    let session = Session {
        id: session_id.into(),
        title: None,
        created_at: now,
        updated_at: now,
        cwd: "/tmp".into(),
        host_project: None,
        model: "mock-model".into(),
        messages: Vec::new(),
        compact_failures: 0,
        turn_count: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        parent_id: None,
    };
    storage::save(tmp.path(), &session).expect("seed save");

    let (mock_host, _mock_task) = spawn_nonstream_chat_mock().await;

    let marker = "WIKI_PROBE_SESSION_AB42";
    let snippet_body = format!("## Project Wiki\n\n- probe marker {}", marker);
    let state = ApiState {
        config: Arc::new(Config {
            host: mock_host,
            host_is_default: false,
            model: "mock-model".into(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".into(),
            config_dir: tmp.path().to_path_buf(),
            global_config_dir: tmp.path().to_path_buf(),
            routing: None,
            aliases: HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        }),
        token: None,
        active_chains: Arc::new(Mutex::new(HashMap::new())),
        reconnect_buffers: replay::new_buffers(),
        wiki_snippet: Some(Arc::from(snippet_body.as_str())),
        wiki_fresh: None,
        wiki_cwd: None,
        wiki_summary: Arc::new(RwLock::new(None)),
        wiki_snippet_bytes_injected: 0,
    };
    let app = build_router(state);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/sessions/{}/turn", session_id))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"hi"}"#))
                .unwrap(),
        )
        .await
        .expect("oneshot");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    let body_text = String::from_utf8(body.to_vec()).unwrap_or_default();
    assert_eq!(status, StatusCode::OK, "sessions_turn body: {}", body_text);

    let reloaded = storage::load(tmp.path(), session_id).expect("reload");
    // Expect: system (built with wiki), user, assistant.
    assert!(
        reloaded.messages.len() >= 3,
        "expected system+user+assistant; got {:?}",
        reloaded.messages
    );
    let sys_content = reloaded.messages[0]["content"]
        .as_str()
        .expect("system content is string");
    assert!(
        sys_content.contains(marker),
        "cached wiki snippet must be spliced into the built system prompt; sys was: {}",
        sys_content
    );
}
