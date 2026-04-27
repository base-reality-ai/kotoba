//! Phase 2.5 Slice B — SSE end-to-end integration against a mock Ollama.
//!
//! Fires a real `POST /chat/stream` through the axum router, with the
//! underlying `OllamaClient` pointed at the in-process `MockOllama`
//! server. Validates the shipped `sse_output_stream` wire format against
//! a known chunk sequence — three tokens + one done — asserting
//! monotonic `id: 1..=4` and correct event types on the wire.

mod common;

use common::mock_ollama::MockOllama;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use dark_matter::api::{build_router, replay, ApiState};
use dark_matter::config::Config;
use http_body_util::BodyExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use tower::ServiceExt;

#[tokio::test]
async fn sse_stream_against_mock_ollama_emits_monotonic_ids() {
    // 1. Spawn mock with a known token sequence.
    let mock = MockOllama::new()
        .emit_token("hi")
        .emit_token(" ")
        .emit_token("there")
        .emit_done(7, 3)
        .spawn()
        .await;

    // 2. Build ApiState pointed at the mock (no auth token → check_auth
    //    returns true without a Bearer header).
    let state = ApiState {
        config: Arc::new(Config {
            host: mock.host(),
            host_is_default: false,
            model: "test-model".into(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".into(),
            config_dir: std::path::PathBuf::from("/tmp"),
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
    };

    // 3. Build router, fire a POST /chat/stream request.
    let app = build_router(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chat/stream")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"hi"}"#))
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

    // 4. Verify wire format: 3 token events + 1 done event, ids 1..=4.
    let ids: Vec<&str> = text.lines().filter(|l| l.starts_with("id:")).collect();
    assert_eq!(
        ids,
        vec!["id: 1", "id: 2", "id: 3", "id: 4"],
        "ids must be monotonic 1..=4; wire was:\n{}",
        text
    );
    assert_eq!(
        text.matches("event: token").count(),
        3,
        "expected 3 token events; wire was:\n{}",
        text
    );
    assert_eq!(
        text.matches("event: done").count(),
        1,
        "expected 1 done event; wire was:\n{}",
        text
    );
}
