//! Phase 2.5 Slice D.1/D.2 — channel-driven `MockOllama` integration tests.
//!
//! These exercise `MockOllama::spawn_channel()` — the caller-driven
//! counterpart to `spawn()`. The driver task pushes chunks through an
//! `mpsc::Sender` one at a time; the mock's response body yields each
//! chunk to the Ollama client as it arrives. D.2 adds an end-to-end
//! disconnect-propagation test against a real TCP-bound dm server.
//!
//! Tests go through the full pipeline: axum router → Ollama client
//! → `sse_output_stream` → SSE wire bytes.

mod common;

use common::mock_ollama::{MockChunk, MockOllama};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use dark_matter::api::{build_router, replay, ApiState};
use dark_matter::config::Config;
use http_body_util::BodyExt;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tower::ServiceExt;

fn make_state(host: String) -> ApiState {
    ApiState {
        config: Arc::new(Config {
            host,
            host_is_default: false,
            model: "mock-model".into(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".into(),
            config_dir: std::path::PathBuf::from("/tmp"),
            global_config_dir: std::path::PathBuf::from("/tmp"),
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
async fn channel_mock_streams_in_order_with_midstream_pause() {
    let (mock, tx) = MockOllama::spawn_channel().await;

    // Driver task: send two tokens with a real wall-clock pause between
    // them, then Done, then drop the sender.
    let driver = tokio::spawn(async move {
        tx.send(MockChunk::Token("hel".into())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        tx.send(MockChunk::Token("lo".into())).await.unwrap();
        tx.send(MockChunk::Done {
            prompt_eval_count: 5,
            eval_count: 2,
        })
        .await
        .unwrap();
        // drop(tx) on task end closes the channel
    });

    let state = make_state(mock.host());
    let app = build_router(state);

    let start = Instant::now();
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
    let elapsed = start.elapsed();
    let text = String::from_utf8(body.to_vec()).expect("utf-8 body");

    driver.await.expect("driver");

    // The 50ms sleep between the two tokens is real — end-to-end time
    // must reflect it. If the body were buffered into a single write,
    // the response would come back in <10ms despite the sleep.
    assert!(
        elapsed >= Duration::from_millis(50),
        "mid-stream pause must be observable end-to-end; elapsed={:?}",
        elapsed
    );

    // Wire shape: 2 token events + 1 done = ids 1..=3.
    let ids: Vec<&str> = text.lines().filter(|l| l.starts_with("id:")).collect();
    assert_eq!(ids, vec!["id: 1", "id: 2", "id: 3"], "wire: {}", text);
    assert_eq!(text.matches("event: token").count(), 2, "wire: {}", text);
    assert_eq!(text.matches("event: done").count(), 1, "wire: {}", text);

    // Token content round-trips through the pipeline: "hel" + "lo" = "hello".
    let tokens: String = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .filter_map(|v| v.get("content").and_then(|t| t.as_str().map(String::from)))
        .collect();
    assert_eq!(tokens, "hello", "token content round-trip; wire: {}", text);
}

#[tokio::test]
async fn channel_mock_error_closes_stream() {
    let (mock, tx) = MockOllama::spawn_channel().await;

    let driver = tokio::spawn(async move {
        tx.send(MockChunk::Token("partial".into())).await.unwrap();
        // Give hyper a chance to flush the Ok chunk before the Err
        // arrives. Without this, both yields land back-to-back and
        // hyper may buffer the first write and convert the pending
        // body-stream error into a 500 response with no body — real
        // upstreams don't emit errors the microsecond after data, so
        // this matches realistic streaming.
        tokio::time::sleep(Duration::from_millis(20)).await;
        tx.send(MockChunk::Error("mock upstream failed".into()))
            .await
            .unwrap();
        // drop(tx) on task end — but the Err yielded above terminates
        // the body stream before this matters.
    });

    let state = make_state(mock.host());
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

    driver.await.expect("driver");

    // Body-stream Err at the mock propagates through the client as
    // StreamEvent::Error, then through sse_output_stream as an error
    // SSE frame. Wire: token (id:1) + error (id:2), then end.
    assert_eq!(text.matches("event: token").count(), 1, "wire: {}", text);
    assert_eq!(text.matches("event: error").count(), 1, "wire: {}", text);
    assert_eq!(text.matches("event: done").count(), 0, "wire: {}", text);

    let ids: Vec<&str> = text.lines().filter(|l| l.starts_with("id:")).collect();
    assert_eq!(ids, vec!["id: 1", "id: 2"], "wire: {}", text);
}

#[tokio::test]
async fn client_disconnect_propagates_to_upstream() {
    // End-to-end cancel propagation. Chain of drops:
    //   client drops Response body
    //   → dm's rx drops
    //   → dm driver's `tx.closed()` fires
    //   → driver task exits, `upstream` (reqwest stream to mock) drops
    //   → mock's axum body stream drops
    //   → mock's mpsc receiver drops
    //   → `mock_tx.closed()` future resolves
    //
    // `tokio::time::timeout(_, mock_tx.closed())` is the correctness
    // assertion. The 2s ceiling is a sanity cap, not the threshold —
    // actual propagation takes single-digit milliseconds.
    use futures_util::StreamExt;

    let (mock, mock_tx) = MockOllama::spawn_channel().await;

    // Bind a real TCP listener so we can open a streaming client against
    // the dm router — `app.oneshot` collects the whole body at once and
    // doesn't let us drop mid-stream.
    let state = make_state(mock.host());
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind dm test server");
    let dm_addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    // Push one token so the pipeline is proven live before we drop.
    mock_tx
        .send(MockChunk::Token("hi".into()))
        .await
        .expect("send initial token");

    let url = format!("http://{}/chat/stream", dm_addr);
    let resp = reqwest::Client::new()
        .post(&url)
        .header("content-type", "application/json")
        .body(r#"{"prompt":"hi"}"#)
        .send()
        .await
        .expect("post /chat/stream");
    assert_eq!(resp.status(), 200, "status should be 200");

    // Pull at least one byte chunk off the wire — proves the pipeline
    // delivered the token end-to-end before we disconnect.
    let mut byte_stream = resp.bytes_stream();
    let first = tokio::time::timeout(Duration::from_secs(2), byte_stream.next())
        .await
        .expect("first chunk within 2s")
        .expect("stream yielded Some")
        .expect("chunk not Err");
    assert!(!first.is_empty(), "first chunk should be non-empty");

    // Drop the whole stream (and thus the Response). This is the
    // client-disconnect event.
    drop(byte_stream);

    // The mock's sender observes `closed()` only after every intermediate
    // layer has dropped its receiver/body/stream. If this times out,
    // propagation is broken somewhere in the chain above.
    let closed = tokio::time::timeout(Duration::from_secs(2), mock_tx.closed()).await;
    assert!(
        closed.is_ok(),
        "mock_tx.closed() must resolve within 2s after client disconnect"
    );

    server.abort();
}
