//! In-process Ollama mock for SSE integration tests.
//!
//! Serves `POST /api/chat` returning an NDJSON stream of chunks matching
//! Ollama's `/api/chat` streaming wire format (see `src/ollama/types.rs` —
//! `ChatChunk`). Two constructors:
//!
//! * [`MockOllama::spawn`] — static body. Builder collects chunks ahead
//!   of time (`emit_thinking`/`emit_token`/`emit_done`); the server
//!   concatenates them into one response body.
//! * [`MockOllama::spawn_channel`] — caller-driven body. Returns an
//!   [`mpsc::Sender<MockChunk>`] the test task pushes chunks through
//!   one at a time; each chunk is written to the response body as it
//!   arrives. Dropping the sender closes the body cleanly (no
//!   `done=true` → client logs a "stream closed without done" warning
//!   but no panic). A [`MockChunk::Error`] value makes the body stream
//!   yield a transport error, which propagates through the client as
//!   [`crate::ollama::types::StreamEvent::Error`].
//!
//! Both paths share [`chunk_to_value`] so the JSON shape stays in sync.
//!
//! Static usage:
//! ```ignore
//! let mock = MockOllama::new()
//!     .emit_token("hello")
//!     .emit_token(" world")
//!     .emit_done(10, 3)
//!     .spawn()
//!     .await;
//! ```
//!
//! Channel usage:
//! ```ignore
//! let (mock, tx) = MockOllama::spawn_channel().await;
//! tx.send(MockChunk::Token("hel".into())).await.unwrap();
//! tokio::time::sleep(std::time::Duration::from_millis(50)).await;
//! tx.send(MockChunk::Token("lo".into())).await.unwrap();
//! tx.send(MockChunk::Done { prompt_eval_count: 5, eval_count: 2 }).await.unwrap();
//! drop(tx); // end of stream
//! ```
//!
//! The server's tokio task is tethered to the returned [`MockHandle`] —
//! when the handle drops, the task is aborted and the listener freed.
//!
//! Scope: mid-stream disconnect cancellation and ring-buffer replay
//! (`Last-Event-ID`) are tracked as Slice D.2/D.3 and will extend this
//! mock further when those cycles land.

#![allow(dead_code)]

use axum::body::Body;
use axum::http::{header, Response, StatusCode};
use axum::{routing::post, Json, Router};
use futures_util::stream;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// A single chunk the channel-driven mock can emit. Maps 1:1 to the
/// static builder's `emit_*` methods plus a transport-error variant.
#[derive(Debug, Clone)]
pub enum MockChunk {
    Thinking(String),
    Token(String),
    Done {
        prompt_eval_count: u64,
        eval_count: u64,
    },
    /// Makes the response body yield an I/O error, which the Ollama
    /// client surfaces as `StreamEvent::Error`.
    Error(String),
}

/// Render a chunk as the `ChatChunk` JSON value Ollama sends on the
/// wire. `MockChunk::Error` has no wire shape — its JSON form would be
/// skipped by the client parser, so the channel path routes it through
/// a body-stream `Err` instead; this helper panics for that variant so
/// a misuse is loud rather than silent.
fn chunk_to_value(chunk: &MockChunk) -> Value {
    match chunk {
        MockChunk::Thinking(s) => json!({
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": s,
                "tool_calls": [],
            },
            "done": false,
        }),
        MockChunk::Token(s) => json!({
            "message": {
                "role": "assistant",
                "content": s,
                "tool_calls": [],
            },
            "done": false,
        }),
        MockChunk::Done {
            prompt_eval_count,
            eval_count,
        } => json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
            },
            "done": true,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
        }),
        MockChunk::Error(_) => {
            panic!("MockChunk::Error has no JSON wire form; it is routed through a body-stream Err")
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MockOllama {
    chunks: Vec<Value>,
}

impl MockOllama {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn emit_thinking(mut self, s: &str) -> Self {
        self.chunks
            .push(chunk_to_value(&MockChunk::Thinking(s.to_string())));
        self
    }

    pub fn emit_token(mut self, s: &str) -> Self {
        self.chunks
            .push(chunk_to_value(&MockChunk::Token(s.to_string())));
        self
    }

    /// Terminal chunk. Call exactly once at the end of a success sequence.
    pub fn emit_done(mut self, prompt_eval_count: u64, eval_count: u64) -> Self {
        self.chunks.push(chunk_to_value(&MockChunk::Done {
            prompt_eval_count,
            eval_count,
        }));
        self
    }

    pub async fn spawn(self) -> MockHandle {
        let chunks = self.chunks;
        let app = Router::new().route(
            "/api/chat",
            post(move |Json(_req): Json<Value>| {
                let chunks = chunks.clone();
                async move {
                    let ndjson: String = chunks.iter().map(|c| format!("{}\n", c)).collect();
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/x-ndjson")
                        .body(Body::from(ndjson))
                        .unwrap()
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind 127.0.0.1:0");
        let addr: SocketAddr = listener.local_addr().expect("local_addr");
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        MockHandle { addr, task }
    }

    /// Spawn a mock whose response body is driven by the caller through
    /// the returned sender. Bounded capacity 16 — enough for typical
    /// token sequences, small enough to make backpressure observable in
    /// pause-oriented tests. The receiver is consumed on the first
    /// `POST /api/chat`; a second request in the same test would get
    /// `500 Internal Server Error`.
    pub async fn spawn_channel() -> (MockHandle, mpsc::Sender<MockChunk>) {
        let (tx, rx) = mpsc::channel::<MockChunk>(16);
        let rx_slot = Arc::new(tokio::sync::Mutex::new(Some(rx)));

        let app = Router::new().route(
            "/api/chat",
            post(move |Json(_req): Json<Value>| {
                let rx_slot = rx_slot.clone();
                async move {
                    let rx = rx_slot.lock().await.take();
                    let Some(rx) = rx else {
                        return Response::builder()
                            .status(StatusCode::INTERNAL_SERVER_ERROR)
                            .body(Body::from("channel already taken"))
                            .unwrap();
                    };
                    let body_stream = stream::unfold(rx, |mut rx| async move {
                        match rx.recv().await {
                            Some(MockChunk::Error(msg)) => {
                                let err = std::io::Error::other(msg);
                                Some((Err::<String, std::io::Error>(err), rx))
                            }
                            Some(chunk) => {
                                let line = format!("{}\n", chunk_to_value(&chunk));
                                Some((Ok(line), rx))
                            }
                            None => None,
                        }
                    });
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/x-ndjson")
                        .body(Body::from_stream(body_stream))
                        .unwrap()
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind 127.0.0.1:0");
        let addr: SocketAddr = listener.local_addr().expect("local_addr");
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (MockHandle { addr, task }, tx)
    }
}

pub struct MockHandle {
    addr: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl MockHandle {
    /// Returns `"127.0.0.1:<port>"` — plug into `Config.host` directly.
    pub fn host(&self) -> String {
        format!("{}", self.addr)
    }
}

impl Drop for MockHandle {
    fn drop(&mut self) {
        self.task.abort();
    }
}
