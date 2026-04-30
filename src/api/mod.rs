//! `dm web` — headless REST API (port 7422 by default).
//!
//! Routes:
//!   GET  /health                    unauthenticated health check
//!   POST /chat                      single-turn (or persistent if `session_id` given)
//!   POST /chat/stream               SSE streaming single-turn chat (ephemeral)
//!                                   (events carry monotonic `id`\: for future Last-Event-ID resume)
//!   POST /`chat/stream/:session_id`   SSE streaming turn attached to a stored session
//!                                   (loads/saves ~/.dm/sessions/`id`.json; text-only, no tool exec)
//!   GET  /sessions                  list all sessions
//!   POST /sessions                  create a new session
//!   GET  /sessions/:id              get session history
//!   POST /sessions/:id/turn         send a turn to an existing session
//!
//! Auth: every route except /health requires
//!       `Authorization: Bearer <token>`.
//!       Token comes from --web-token or is auto-generated and stored in
//!       ~/.dm/web.token on first start.

pub mod replay;

use crate::config::Config;
use crate::ollama::client::OllamaClient;
use anyhow::Result;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;

// ── Token helpers ─────────────────────────────────────────────────────────────

pub fn token_path(config_dir: &std::path::Path) -> PathBuf {
    config_dir.join("web.token")
}

/// Read the token from `~/.dm/web.token`, generating and writing it if absent.
pub fn load_or_generate_token(config_dir: &std::path::Path) -> Result<String> {
    let path = token_path(config_dir);
    if path.exists() {
        let token = std::fs::read_to_string(&path)?.trim().to_string();
        if !token.is_empty() {
            return Ok(token);
        }
    }
    // Generate a random 32-char hex token via a v4 UUID (no extra deps).
    let token = uuid::Uuid::new_v4().to_string().replace('-', "");
    std::fs::write(&path, &token)?;
    Ok(token)
}

/// Record the last port used by `dm web` so doctor can report it.
pub fn write_last_port(config_dir: &std::path::Path, port: u16) {
    let _ = std::fs::write(config_dir.join("web.last_port"), port.to_string());
}

#[derive(Clone)]
pub struct ApiState {
    pub config: Arc<Config>,
    pub token: Option<String>,
    /// Maps `chain_id` → workspace path for active chains started via the API.
    pub active_chains: Arc<Mutex<HashMap<String, PathBuf>>>,
    /// Per-session SSE ring buffers for `Last-Event-ID` replay. See
    /// [`replay`] for the replay-only semantics (no live re-attach).
    pub reconnect_buffers: replay::ReplayBuffers,
    /// Wiki context snippet, loaded once at state construction.
    /// `None` when no `.dm/wiki/index.md` exists at startup cwd.
    /// Threaded into the system prompt on every `/chat/stream` to avoid
    /// a per-request 1.1MB disk read. Index-staleness across ingest is a
    /// known gap — see `current_wiki_summary` for the summary-only hot
    /// reload and Cycle 48 notes for why index reload is deferred.
    pub wiki_snippet: Option<Arc<str>>,
    /// Top-K entity/concept pages ranked by `last_updated` (C86), loaded
    /// once at state construction and threaded into the system prompt
    /// as a `<wiki_fresh>` block. Pure in-memory read off the C85
    /// `IndexEntry` cache — zero I/O on every `/chat/stream`. `None` on
    /// wiki-less projects or indexes with no qualifying pages.
    pub wiki_fresh: Option<Arc<str>>,
    /// Project cwd (where `.dm/wiki/` lives). Stored so
    /// [`ApiState::current_wiki_summary`] can stat-check
    /// `.dm/wiki/.summary-dirty` and reopen the Wiki for lazy reload on
    /// the web hot path. `None` on wiki-less projects; the marker
    /// stat-check never runs there.
    pub wiki_cwd: Option<PathBuf>,
    /// Wiki project-summary snippet (from `summaries/project.md`),
    /// loaded lazily. Starts as the startup value from
    /// [`load_wiki_summary`]; reloaded in-place by
    /// [`ApiState::current_wiki_summary`] when the `.summary-dirty` marker
    /// (Cycle 47) appears. Wrapped in `Arc\<RwLock\>` so handlers clone
    /// `ApiState` cheaply while all clones share the latest snippet.
    pub wiki_summary: Arc<RwLock<Option<Arc<str>>>>,
    /// Telemetry: bytes of wiki content injected into the system prompt at
    /// session start (from `wiki_snippet` + `wiki_fresh`). Set once during
    /// `ApiState` construction and logged at session end.
    pub wiki_snippet_bytes_injected: usize,
}

/// Best-effort wiki-snippet loader. Returns `None` when `.dm/wiki/index.md`
/// is absent at `cwd` (so the caller never materializes a wiki tree on
/// projects without one).
pub fn load_wiki_snippet(cwd: &std::path::Path) -> Option<Arc<str>> {
    if !cwd.join(".dm/wiki/index.md").is_file() {
        return None;
    }
    crate::wiki::Wiki::open(cwd)
        .ok()
        .and_then(|w| w.context_snippet())
        .map(Arc::from)
}

/// Best-effort project-summary loader. Returns `None` when
/// `.dm/wiki/summaries/project.md` is absent at `cwd` (so wiki-less projects
/// are unaffected). Budget matches the disk-fallback ceiling in
/// `build_system_prompt_inner`.
pub fn load_wiki_summary(cwd: &std::path::Path) -> Option<Arc<str>> {
    if !cwd.join(".dm/wiki/summaries/project.md").is_file() {
        return None;
    }
    crate::wiki::Wiki::open(cwd)
        .ok()
        .and_then(|w| w.project_summary_snippet(4096))
        .map(Arc::from)
}

/// Best-effort fresh-pages snippet loader (C86). Returns `None` when
/// `.dm/wiki/index.md` is absent, the index has no entity/concept entries,
/// or the snippet formatter emits nothing — so wiki-less and
/// low-comprehension projects keep the `<wiki_fresh>` block suppressed.
/// Budget matches `WIKI_FRESH_DISK_BUDGET` in `build_system_prompt_inner`.
pub fn load_wiki_fresh(cwd: &std::path::Path) -> Option<Arc<str>> {
    if !cwd.join(".dm/wiki/index.md").is_file() {
        return None;
    }
    crate::wiki::Wiki::open(cwd)
        .ok()
        .and_then(|w| w.fresh_pages_snippet(crate::wiki::FRESH_PAGES_BUDGET_CHARS))
        .map(Arc::from)
}

impl ApiState {
    /// Return the current project-summary snippet. Fast path: one stat
    /// check of `.dm/wiki/.summary-dirty`. If the marker exists, regenerate
    /// via [`crate::wiki::Wiki::ensure_summary_current`], reload the
    /// snippet, and swap it into the cache. On any reload failure, falls
    /// back to the last-known-good cached value — a stale snippet beats no
    /// snippet at session start.
    pub fn current_wiki_summary(&self) -> Option<Arc<str>> {
        let Some(cwd) = &self.wiki_cwd else {
            return self.read_cached_summary();
        };
        let marker = cwd.join(".dm/wiki/.summary-dirty");
        if !marker.is_file() {
            return self.read_cached_summary();
        }
        let reloaded: Option<Arc<str>> = crate::wiki::Wiki::open(cwd)
            .ok()
            .and_then(|w| {
                let _ = w.ensure_summary_current();
                w.project_summary_snippet(4096)
            })
            .map(Arc::from);
        if reloaded.is_some() {
            if let Ok(mut guard) = self.wiki_summary.write() {
                guard.clone_from(&reloaded);
            }
            return reloaded;
        }
        self.read_cached_summary()
    }

    fn read_cached_summary(&self) -> Option<Arc<str>> {
        self.wiki_summary.read().ok().and_then(|g| g.clone())
    }
}

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatRequest {
    pub prompt: String,
    pub model: Option<String>,
    /// If provided, the turn is appended to this existing session (persistent).
    /// If absent, a new ephemeral session is created for this request.
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateSessionRequest {
    pub model: Option<String>,
}

#[derive(Deserialize)]
pub struct TurnRequest {
    pub prompt: String,
}

#[derive(Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: Option<String>,
    pub host_project: Option<String>,
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: usize,
}

#[derive(Deserialize)]
pub struct StartChainRequest {
    pub config: String,
}

#[derive(Deserialize)]
pub struct ChainTalkRequest {
    pub node: String,
    pub message: String,
}

// ── Auth helper ───────────────────────────────────────────────────────────────

pub fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

pub fn check_auth(state: &ApiState, headers: &HeaderMap) -> bool {
    match &state.token {
        None => true,
        Some(expected) => headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|t| constant_time_eq(t, expected)),
    }
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "missing or invalid Authorization. Try: include 'Authorization: Bearer <token>' header with a valid API token."})),
    )
        .into_response()
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn build_router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/chat", post(chat_handler))
        .route("/chat/stream", post(chat_stream_handler))
        .route(
            "/chat/stream/:session_id",
            post(chat_stream_session_handler),
        )
        .route("/sessions", get(sessions_list))
        .route("/sessions", post(sessions_create))
        .route("/sessions/:id", get(sessions_get))
        .route("/sessions/:id/turn", post(sessions_turn))
        .route("/chains", get(chains_list))
        .route("/chains", post(chains_start))
        .route("/chains/:id", get(chains_get))
        .route("/chains/:id/stop", post(chains_stop))
        .route("/chains/:id/pause", post(chains_pause))
        .route("/chains/:id/resume", post(chains_resume))
        .route("/chains/:id/talk", post(chains_talk))
        .route("/chains/:id/log", get(chains_log))
        .route("/chains/:id/log/:cycle", get(chains_log_cycle))
        .with_state(state)
}

pub async fn run(state: ApiState, port: u16) -> Result<()> {
    let _ = crate::logging::init_in_config_dir("web", &state.config.config_dir);
    let router = build_router(state);
    let addr = format!("127.0.0.1:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    crate::logging::log(&format!("[dm web] listening on http://{}", addr));
    axum::serve(listener, router).await?;
    Ok(())
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// GET /health — unauthenticated liveness check.
async fn health_handler(State(state): State<ApiState>) -> Response {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "model": state.config.model,
    }))
    .into_response()
}

async fn chat_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let model = req.model.unwrap_or_else(|| state.config.model.clone());

    // If session_id provided, route through persistent session turn.
    if let Some(ref sid) = req.session_id {
        let Ok(session) = crate::session::storage::load(&state.config.config_dir, sid) else {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": format!("session '{}' not found. Try: GET /sessions to list active sessions, or POST /sessions to create a new one.", sid)})),
            )
                .into_response();
        };
        let client = OllamaClient::new(state.config.ollama_base_url(), session.model.clone());
        let registry = crate::tools::registry::default_registry(
            sid,
            &state.config.config_dir,
            &state.config.ollama_base_url(),
            &session.model,
            &state.config.embed_model,
        );
        let wiki_summary = state.current_wiki_summary();
        return match run_session_turn(
            &req.prompt,
            &client,
            &registry,
            session,
            &state.config,
            state.wiki_snippet.as_deref(),
            wiki_summary.as_deref(),
            state.wiki_fresh.as_deref(),
            state.wiki_snippet_bytes_injected,
        )
        .await
        {
            Ok(response) => Json(json!({"response": response, "session_id": sid, "model": &model}))
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response(),
        };
    }

    // No session_id → ephemeral single-turn.
    let client = OllamaClient::new(state.config.ollama_base_url(), model.clone());
    let registry = crate::tools::registry::default_registry(
        "web-chat",
        &state.config.config_dir,
        &state.config.ollama_base_url(),
        &model,
        &state.config.embed_model,
    );

    match crate::conversation::run_conversation_capture_in_config_dir(
        &req.prompt,
        "api",
        &client,
        &registry,
        &state.config.config_dir,
    )
    .await
    {
        Ok(capture) => Json(json!({"response": capture.text, "model": model})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Build the SSE output stream from a raw Ollama `StreamEvent` stream.
///
/// Emits a monotonic `id:` starting at 1 on every event — clients use it
/// with `Last-Event-ID` to resume from the per-session replay ring. The
/// counter is per-connection; each invocation of the handler gets a fresh
/// sequence.
///
/// `tap`: when `Some`, a copy of every emitted frame (id, event, data) is
/// pushed into the ring before yielding to the wire. Ephemeral
/// `/chat/stream` calls pass `None` — there's no session id to key a ring
/// against.
pub(crate) fn sse_output_stream<S>(
    mut event_stream: S,
    model: String,
    tap: Option<Arc<std::sync::Mutex<replay::ReplayRing>>>,
) -> impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>
where
    S: futures_util::Stream<Item = crate::ollama::types::StreamEvent> + Unpin + Send + 'static,
{
    async_stream::stream! {
        let mut seq: u64 = 0;
        let mut full_content = String::new();
        while let Some(event) = futures_util::StreamExt::next(&mut event_stream).await {
            seq += 1;
            let id = seq.to_string();
            // Each arm computes `name` + `data`, pushes to the tap (if
            // present), then yields the Event. Keeping the tap inline
            // guarantees every wire frame has a matching ring entry.
            let (name, data, terminal): (&'static str, String, bool) = match event {
                crate::ollama::types::StreamEvent::Thinking(tok) => {
                    ("thinking", json!({"content": tok}).to_string(), false)
                }
                crate::ollama::types::StreamEvent::Token(tok) => {
                    full_content.push_str(&tok);
                    ("token", json!({"content": tok}).to_string(), false)
                }
                crate::ollama::types::StreamEvent::ToolCalls(calls) => {
                    let names: Vec<&str> = calls.iter().map(|c| c.function.name.as_str()).collect();
                    ("tool", json!({"tools": names}).to_string(), false)
                }
                crate::ollama::types::StreamEvent::Done { prompt_tokens, completion_tokens } => {
                    let data = json!({
                        "response": full_content,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }).to_string();
                    ("done", data, true)
                }
                crate::ollama::types::StreamEvent::Error(e) => {
                    ("error", json!({"error": e}).to_string(), true)
                }
            };
            if let Some(ref ring) = tap {
                if let Ok(mut g) = ring.lock() {
                    g.push(replay::StoredFrame {
                        id: seq,
                        event: name.to_string(),
                        data: data.clone(),
                    });
                }
            }
            yield Ok(Event::default().id(id).event(name).data(data));
            if terminal {
                break;
            }
        }
    }
}

/// Adapt an `mpsc::Receiver\<StreamEvent\>` into a `Stream`. Shared by
/// `chat_stream_handler` and `run_session_turn_stream` so the driver
/// task boundary has a single forward path into `sse_output_stream`.
fn rx_to_event_stream(
    mut rx: tokio::sync::mpsc::Receiver<crate::ollama::types::StreamEvent>,
) -> impl futures_util::Stream<Item = crate::ollama::types::StreamEvent> + Send + 'static {
    async_stream::stream! {
        while let Some(evt) = rx.recv().await {
            yield evt;
        }
    }
}

async fn chat_stream_handler(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let model = req.model.unwrap_or_else(|| state.config.model.clone());
    let client = OllamaClient::new(state.config.ollama_base_url(), model.clone());

    let wiki_summary = state.current_wiki_summary();
    let system_prompt = crate::system_prompt::build_system_prompt_with_snippets(
        &[],
        None,
        state.wiki_snippet.as_deref(),
        wiki_summary.as_deref(),
        state.wiki_fresh.as_deref(),
    )
    .await;
    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": req.prompt}),
    ];

    let tool_defs = crate::tools::registry::default_registry(
        "web-stream",
        &state.config.config_dir,
        &state.config.ollama_base_url(),
        &model,
        &state.config.embed_model,
    )
    .definitions();

    let stream_result = client.chat_stream_with_tools(&messages, &tool_defs).await;

    match stream_result {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
        Ok(mut upstream) => {
            // Put the upstream agent on the far side of a task boundary so
            // client disconnect (body dropped → rx dropped → tx.closed)
            // can be observed as a cancel signal. Without this the select
            // has no place to hook the disconnect — the whole stream
            // future just drops when the body drops.
            let (tx, rx) = tokio::sync::mpsc::channel::<crate::ollama::types::StreamEvent>(32);

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        // biased: disconnect always wins ties so we don't
                        // keep fetching events into a channel no one drains.
                        biased;
                        _ = tx.closed() => {
                            crate::logging::log(
                                "api::sse: client disconnected; aborting upstream",
                            );
                            break;
                        }
                        maybe_evt = futures_util::StreamExt::next(&mut upstream) => {
                            match maybe_evt {
                                Some(evt) => {
                                    if tx.send(evt).await.is_err() {
                                        crate::logging::log(
                                            "api::sse: client disconnected mid-send; aborting upstream",
                                        );
                                        break;
                                    }
                                }
                                None => break,
                            }
                        }
                    }
                }
                // `upstream` drops here → reqwest connection closes cleanly.
            });

            let output_stream =
                sse_output_stream(Box::pin(rx_to_event_stream(rx)), model.clone(), None);
            Sse::new(output_stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
    }
}

/// Session-attached SSE turn. Loads the session JSON, replays its messages
/// as chat history, streams the assistant response, and on `Done` appends
/// the user prompt and accumulated assistant text back to the file.
///
/// Text-only: no tool execution on the streaming path. If the upstream
/// emits `tool_calls`, the helper forwards them as informational
/// `event: tool` frames (via `sse_output_stream`) but does not execute
/// them and does not record them in the session file. A client that
/// needs tool execution should use `POST /chat` with `session_id`.
///
/// Mid-stream client disconnect: the session is not updated (no `Done`
/// seen, no save). The next turn effectively starts fresh against the
/// pre-stream history. Mid-stream resume is Slice D.
async fn run_session_turn_stream(
    prompt: &str,
    client: &OllamaClient,
    session: crate::session::Session,
    config_dir: &std::path::Path,
    tap: Option<Arc<std::sync::Mutex<replay::ReplayRing>>>,
) -> Response {
    let mut messages: Vec<Value> = session.messages.clone();
    messages.push(json!({"role": "user", "content": prompt}));

    let mut upstream = match client.chat_stream_with_tools(&messages, &[]).await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response();
        }
    };

    let session_for_save = session.clone();
    let config_dir = config_dir.to_path_buf();
    let prompt_owned = prompt.to_string();
    let model = session.model.clone();

    // Same D.2 pattern as chat_stream_handler: park the upstream on a task
    // boundary so client disconnect (body dropped → rx dropped →
    // tx.closed) is observable. The driver also owns the persist-on-Done
    // step that the removed `teed_stream` async_stream previously did.
    let (tx, rx) = tokio::sync::mpsc::channel::<crate::ollama::types::StreamEvent>(32);

    tokio::spawn(async move {
        let mut accumulated = String::new();
        loop {
            tokio::select! {
                // biased: disconnect always wins ties — we don't keep
                // draining upstream into a channel no one's reading.
                biased;
                _ = tx.closed() => {
                    crate::logging::log(
                        "api::sse: session client disconnected; aborting upstream",
                    );
                    break;
                }
                maybe_evt = futures_util::StreamExt::next(&mut upstream) => {
                    let Some(evt) = maybe_evt else {
                        break;
                    };
                    if let crate::ollama::types::StreamEvent::Token(ref t) = evt {
                        accumulated.push_str(t);
                    }
                    let is_done = matches!(evt, crate::ollama::types::StreamEvent::Done { .. });
                    let is_error = matches!(evt, crate::ollama::types::StreamEvent::Error(_));
                    if is_done {
                        // Persist BEFORE forwarding Done. The client-visible
                        // Done frame is the "save committed" signal (guarded
                        // by session_stream_updates_file_on_done). If the
                        // client dropped between the last Token and this
                        // Done, the save still commits — matches the Cycle
                        // 16 best-effort persist-on-Done contract.
                        let mut sess = session_for_save.clone();
                        sess.messages.push(json!({"role": "user", "content": prompt_owned.clone()}));
                        sess.messages.push(json!({"role": "assistant", "content": accumulated.clone()}));
                        sess.updated_at = chrono::Utc::now();
                        let _ = crate::session::storage::save(&config_dir, &sess);
                    }
                    if tx.send(evt).await.is_err() {
                        crate::logging::log(
                            "api::sse: session client disconnected mid-send; aborting upstream",
                        );
                        break;
                    }
                    if is_done || is_error {
                        break;
                    }
                }
            }
        }
        // `upstream` drops here → reqwest connection closes cleanly.
    });

    let output = sse_output_stream(Box::pin(rx_to_event_stream(rx)), model, tap);
    Sse::new(output)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn chat_stream_session_handler(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    // Replay branch: if the client supplied `Last-Event-ID`, do not load
    // the session or spawn an upstream — just drain the session's ring
    // (if any) starting after the given id and close with a terminal
    // `event: replay_complete` marker. Replay-only semantics: no live
    // re-attach, no agent resume.
    if let Some(last_id) = replay::extract_last_event_id(&headers) {
        let ring_arc = {
            let map = state
                .reconnect_buffers
                .lock()
                .expect("reconnect_buffers poisoned");
            map.get(&session_id).cloned()
        };
        let tail = match ring_arc {
            Some(ring) => ring.lock().expect("ring poisoned").frames_after(last_id),
            None => Vec::new(),
        };
        if tail.is_empty() {
            return StatusCode::NO_CONTENT.into_response();
        }
        let replayed = tail.len() as u64;
        let resumed_from = last_id;
        let highest = tail.last().map_or(resumed_from, |f| f.id);
        let stream = async_stream::stream! {
            for f in tail {
                yield Ok::<_, std::convert::Infallible>(
                    Event::default()
                        .id(f.id.to_string())
                        .event(f.event)
                        .data(f.data),
                );
            }
            let marker = json!({"resumed_from": resumed_from, "replayed": replayed}).to_string();
            yield Ok::<_, std::convert::Infallible>(
                Event::default()
                    .id((highest + 1).to_string())
                    .event("replay_complete")
                    .data(marker),
            );
        };
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let Ok(session) = crate::session::storage::load(&state.config.config_dir, &session_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("session '{}' not found. Try: GET /sessions to list active sessions, or POST /sessions to create a new one.", session_id)})),
        )
            .into_response();
    };

    // Fresh turn: clear any ring from a prior turn on this session, then
    // thread a tap so every emitted frame is captured for future replay.
    let ring = replay::get_or_create_ring(&state.reconnect_buffers, &session_id);
    ring.lock().expect("ring poisoned").clear();

    let client = OllamaClient::new(state.config.ollama_base_url(), session.model.clone());
    run_session_turn_stream(
        &req.prompt,
        &client,
        session,
        &state.config.config_dir,
        Some(ring),
    )
    .await
}

async fn sessions_list(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    match crate::session::storage::list(&state.config.config_dir) {
        Ok(sessions) => {
            let summaries: Vec<SessionSummary> = sessions
                .into_iter()
                .map(|s| SessionSummary {
                    id: s.id,
                    title: s.title,
                    host_project: s.host_project,
                    model: s.model,
                    created_at: s.created_at.to_rfc3339(),
                    updated_at: s.updated_at.to_rfc3339(),
                    message_count: s.messages.len(),
                })
                .collect();
            Json(json!(summaries)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn sessions_create(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<CreateSessionRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let model = req.model.unwrap_or_else(|| state.config.model.clone());
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let session = crate::session::Session::new(cwd, model.clone());
    match crate::session::storage::save(&state.config.config_dir, &session) {
        Ok(()) => Json(json!({
            "id": session.id,
            "model": model,
            "created_at": session.created_at.to_rfc3339(),
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn sessions_get(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    match crate::session::storage::load(&state.config.config_dir, &id) {
        Ok(session) => {
            let messages: Vec<&Value> = session
                .messages
                .iter()
                .filter(|m| m["role"].as_str() != Some("system"))
                .collect();
            Json(json!({
                "id": session.id,
                "title": session.title,
                "model": session.model,
                "created_at": session.created_at.to_rfc3339(),
                "updated_at": session.updated_at.to_rfc3339(),
                "messages": messages,
            }))
            .into_response()
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("session '{}' not found. Try: GET /sessions to list active sessions, or POST /sessions to create a new one.", id)})),
        )
            .into_response(),
    }
}

async fn sessions_turn(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<TurnRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let Ok(session) = crate::session::storage::load(&state.config.config_dir, &id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!("session '{}' not found. Try: GET /sessions to list active sessions, or POST /sessions to create a new one.", id)})),
        )
            .into_response();
    };

    let client = OllamaClient::new(state.config.ollama_base_url(), session.model.clone());
    let registry = crate::tools::registry::default_registry(
        &id,
        &state.config.config_dir,
        &state.config.ollama_base_url(),
        &session.model,
        &state.config.embed_model,
    );

    let wiki_summary = state.current_wiki_summary();
    match run_session_turn(
        &req.prompt,
        &client,
        &registry,
        session,
        &state.config,
        state.wiki_snippet.as_deref(),
        wiki_summary.as_deref(),
        state.wiki_fresh.as_deref(),
        state.wiki_snippet_bytes_injected,
    )
    .await
    {
        Ok(response) => Json(json!({"response": response, "session_id": id})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Run one turn on an existing session and persist the result.
#[allow(clippy::too_many_arguments)]
async fn run_session_turn(
    prompt: &str,
    client: &OllamaClient,
    registry: &crate::tools::registry::ToolRegistry,
    mut session: crate::session::Session,
    config: &Config,
    wiki_snippet: Option<&str>,
    wiki_summary: Option<&str>,
    wiki_fresh: Option<&str>,
    wiki_snippet_bytes: usize,
) -> anyhow::Result<String> {
    use crate::conversation::{assistant_msg, system_msg, user_msg};
    use crate::permissions::engine::PermissionEngine;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    if session.messages.is_empty() {
        let sys = crate::system_prompt::build_system_prompt_with_snippets(
            &[],
            None,
            wiki_snippet,
            wiki_summary,
            wiki_fresh,
        )
        .await;
        session.messages.push(system_msg(&sys));
    }
    session.messages.push(user_msg(prompt));

    let tool_defs = registry.definitions();
    let engine = PermissionEngine::new(true, vec![]);
    let empty_mcp: HashMap<String, Arc<Mutex<crate::mcp::client::McpClient>>> = HashMap::new();

    for _round in 0..crate::conversation::DEFAULT_MAX_TURNS {
        let response = client.chat(&session.messages, &tool_defs).await?;

        let text = response.message.content.clone();
        let tool_calls = response.message.tool_calls.clone();

        session.messages.push(assistant_msg(&text, &tool_calls));

        if tool_calls.is_empty() {
            crate::session::storage::save(&config.config_dir, &session)?;
            log_wiki_telemetry(registry, wiki_snippet_bytes);
            return Ok(text);
        }

        for tc in &tool_calls {
            let name = &tc.function.name;
            let args = tc.function.arguments.clone();
            let decision = engine.check(name, &args);
            let result = if matches!(decision, crate::permissions::Decision::Deny) {
                crate::tools::ToolResult {
                    content: "Denied".to_string(),
                    is_error: true,
                }
            } else if let Some(server) = registry.mcp_server_for(name) {
                if let Some(mc) = empty_mcp.get(server) {
                    let mut locked = mc.lock().await;
                    match locked.call_tool(name, args).await {
                        Ok(c) => crate::tools::ToolResult {
                            content: c,
                            is_error: false,
                        },
                        Err(e) => crate::tools::ToolResult {
                            content: format!("MCP error: {}", e),
                            is_error: true,
                        },
                    }
                } else {
                    crate::tools::ToolResult {
                        content: format!("MCP server '{}' not connected", server),
                        is_error: true,
                    }
                }
            } else {
                registry
                    .call(name, args)
                    .await
                    .unwrap_or_else(|e| crate::tools::ToolResult {
                        content: format!("Tool error: {}", e),
                        is_error: true,
                    })
            };

            let tool_content = crate::util::truncate_tool_output(&result.content);
            session.messages.push(json!({
                "role": "tool",
                "name": name,
                "content": tool_content,
            }));
        }
    }

    crate::session::storage::save(&config.config_dir, &session)?;
    log_wiki_telemetry(registry, wiki_snippet_bytes);
    Ok("[reached max turns without final response]".to_string())
}

/// Emit a single telemetry line summarising wiki consultation for this turn.
fn log_wiki_telemetry(registry: &crate::tools::registry::ToolRegistry, snippet_bytes: usize) {
    use std::sync::atomic::Ordering;
    let calls = registry.wiki_tool_calls.load(Ordering::Relaxed);
    let drift = registry.wiki_drift_warnings.load(Ordering::Relaxed);
    crate::logging::log(&format!(
        "wiki_telemetry: snippet_bytes={} tool_calls={} drift_warnings={}",
        snippet_bytes, calls, drift
    ));
}

// ── Chain handlers ───────────────────────────────────────────────────────────

/// GET /chains — list available chain configs and any running chain state.
async fn chains_list(State(state): State<ApiState>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let chains_dir = state.config.config_dir.join("chains");
    let mut configs = Vec::new();

    if chains_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&chains_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("yaml")
                    || path.extension().and_then(|e| e.to_str()) == Some("yml")
                {
                    if let Ok(cfg) = crate::orchestrate::parse_chain_config(&path) {
                        configs.push(json!({
                            "name": cfg.name,
                            "description": cfg.description,
                            "nodes": cfg.nodes.len(),
                            "max_cycles": cfg.max_cycles,
                            "loop_forever": cfg.loop_forever,
                            "config_path": path.to_string_lossy(),
                        }));
                    }
                }
            }
        }
    }

    let running = crate::orchestrate::chain_status().map(|s| {
        json!({
            "chain_id": s.chain_id,
            "chain_name": s.config.name,
            "current_cycle": s.current_cycle,
            "turns_used": s.turns_used,
            "active_node_index": s.active_node_index,
        })
    });

    Json(json!({
        "configs": configs,
        "running": running,
    }))
    .into_response()
}

/// POST /chains — start a chain from a config file path.
async fn chains_start(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(req): Json<StartChainRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let config_path = std::path::Path::new(&req.config);
    let config_path = if config_path.is_absolute() {
        config_path.to_path_buf()
    } else {
        state.config.config_dir.join("chains").join(&req.config)
    };

    let mut chain_config = match crate::orchestrate::load_chain_config(&config_path) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": format!(
                    "Failed to load chain config: {}. Try: confirm the chain config exists in ~/.dm/chains/ or pass an absolute path.",
                    e
                )})),
            )
                .into_response()
        }
    };

    if let Err(e) = crate::orchestrate::validate_chain_config(&chain_config) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!(
                "Invalid chain config: {}. Try: confirm node wiring is correct and required fields (name, role, model) are present.",
                e
            )})),
        )
            .into_response();
    }

    chain_config.resolve_aliases(&state.config.aliases);

    let chain_id = format!(
        "{}-{}",
        chain_config.name,
        uuid::Uuid::new_v4()
            .to_string()
            .split('-')
            .next()
            .unwrap_or("0")
    );
    let workspace = chain_config.workspace.clone();
    let model = chain_config
        .nodes
        .first()
        .map_or_else(|| state.config.model.clone(), |n| n.model.clone());
    let ollama_url = state.config.ollama_base_url();
    let config_dir = state.config.config_dir.clone();
    let embed_model = state.config.embed_model.clone();

    let orch_config = crate::orchestrate::OrchestrationConfig {
        chain: chain_config,
        chain_id: chain_id.clone(),
        retry: crate::conversation::RetrySettings::default(),
        resume_state: None,
    };

    let ws = workspace.clone();
    let spawn_chain_id = chain_id.clone();

    let handle = tokio::spawn(async move {
        let client = OllamaClient::new(ollama_url.clone(), model.clone());
        let registry = crate::tools::registry::default_registry(
            &spawn_chain_id,
            &config_dir,
            &ollama_url,
            &model,
            &embed_model,
        );
        if let Err(e) =
            crate::orchestrate::runner::run_orchestration(orch_config, client, registry, None).await
        {
            crate::logging::log_err(&format!("[dm api] chain error: {}", e));
        }
    });

    crate::orchestrate::set_chain(handle, ws.clone());

    {
        let mut chains = state.active_chains.lock().await;
        chains.insert(chain_id.clone(), ws);
    }

    (
        StatusCode::CREATED,
        Json(json!({
            "chain_id": chain_id,
            "workspace": workspace.to_string_lossy(),
            "status": "started",
        })),
    )
        .into_response()
}

/// GET /chains/:id — get chain status.
async fn chains_get(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let workspace = {
        let chains = state.active_chains.lock().await;
        chains.get(&id).cloned()
    };

    let Some(workspace) = workspace else {
        if let Some(status) = crate::orchestrate::chain_status() {
            if status.chain_id == id {
                return Json(json!(status)).into_response();
            }
        }
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!(
                "chain '{}' not found. Try: GET /chains to list active chains.",
                id
            )})),
        )
            .into_response();
    };

    let state_path = workspace.join("chain_state.json");
    match crate::orchestrate::types::ChainState::load(&state_path) {
        Ok(chain_state) => Json(json!(chain_state)).into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": format!(
                "chain state not found for '{}'. Try: confirm the chain has run at least one cycle and produced chain_state.json.",
                id
            )})),
        )
            .into_response(),
    }
}

/// Look up chain workspace by id; returns None if unknown.
async fn lookup_chain_workspace(state: &ApiState, id: &str) -> Option<PathBuf> {
    let chains = state.active_chains.lock().await;
    chains.get(id).cloned()
}

fn chain_not_found(id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": format!(
            "chain '{}' not found. Try: GET /chains to list active chains.",
            id
        )})),
    )
        .into_response()
}

/// POST /chains/:id/stop — stop a running chain and reap it from `active_chains`.
async fn chains_stop(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let workspace = lookup_chain_workspace(&state, &id).await;

    let Some(ws) = workspace else {
        return chain_not_found(&id);
    };

    let stop_file = ws.join(".dm-stop");
    if let Err(e) = std::fs::write(&stop_file, b"") {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!(
                "Failed to write stop sentinel: {}. Try: confirm the chain workspace directory is writable.",
                e
            )})),
        )
            .into_response();
    }

    {
        let mut chains = state.active_chains.lock().await;
        chains.remove(&id);
    }

    Json(json!({"status": "stopping", "chain_id": id})).into_response()
}

/// POST /chains/:id/pause — pause a running chain.
async fn chains_pause(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let Some(ws) = lookup_chain_workspace(&state, &id).await else {
        return chain_not_found(&id);
    };

    if let Err(e) = std::fs::write(ws.join(".dm-pause"), b"") {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!(
                "Failed to write pause sentinel: {}. Try: confirm the chain workspace directory is writable.",
                e
            )})),
        )
            .into_response();
    }

    Json(json!({"status": "paused", "chain_id": id})).into_response()
}

/// POST /chains/:id/resume — resume a paused chain.
async fn chains_resume(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let Some(ws) = lookup_chain_workspace(&state, &id).await else {
        return chain_not_found(&id);
    };

    let pause_file = ws.join(".dm-pause");
    if pause_file.exists() {
        if let Err(e) = std::fs::remove_file(&pause_file) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!(
                    "Failed to remove pause sentinel: {}. Try: confirm the chain workspace directory is writable.",
                    e
                )})),
            )
                .into_response();
        }
    }

    Json(json!({"status": "resumed", "chain_id": id})).into_response()
}

/// POST /chains/:id/talk — inject a message into a running chain node.
async fn chains_talk(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<ChainTalkRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    if !crate::orchestrate::is_safe_node_name(&req.node) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": format!(
                "Invalid node name '{}': contains unsafe characters. Try: use only alphanumeric characters, dashes, and underscores.",
                req.node
            )})),
        )
            .into_response();
    }

    let Some(ws) = lookup_chain_workspace(&state, &id).await else {
        return chain_not_found(&id);
    };

    let talk_file = ws.join(format!("talk-{}.md", req.node));
    if let Err(e) = std::fs::write(&talk_file, req.message.as_bytes()) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": format!(
                "Failed to write talk file: {}. Try: confirm the chain workspace directory is writable.",
                e
            )})),
        )
            .into_response();
    }

    Json(json!({"status": "injected", "node": req.node, "chain_id": id})).into_response()
}

/// GET /chains/:id/log — get all chain artifacts.
async fn chains_log(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let Some(ws) = lookup_chain_workspace(&state, &id).await else {
        return chain_not_found(&id);
    };

    match crate::orchestrate::chain_log_from_workspace(&ws, None) {
        Ok(log) => Json(json!(log)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// GET /chains/:id/log/:cycle — get chain artifacts for a specific cycle.
async fn chains_log_cycle(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Path((id, cycle)): Path<(String, usize)>,
) -> Response {
    if !check_auth(&state, &headers) {
        return unauthorized();
    }

    let Some(ws) = lookup_chain_workspace(&state, &id).await else {
        return chain_not_found(&id);
    };

    match crate::orchestrate::chain_log_from_workspace(&ws, Some(cycle)) {
        Ok(log) => Json(json!(log)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(token: Option<&str>) -> ApiState {
        make_state_with_config_dir(token, std::path::PathBuf::from("/tmp"))
    }

    fn make_state_with_config_dir(token: Option<&str>, config_dir: std::path::PathBuf) -> ApiState {
        ApiState {
            config: Arc::new(Config {
                host: "localhost:11434".into(),
                host_is_default: false,
                model: "gemma4:26b".into(),
                model_is_default: false,
                tool_model: None,
                embed_model: "nomic-embed-text".into(),
                global_config_dir: config_dir.clone(),
                config_dir,
                routing: None,
                aliases: std::collections::HashMap::new(),
                max_retries: 3,
                retry_delay_ms: 1000,
                max_retry_delay_ms: 30_000,
                fallback_model: None,
                snapshot_interval_secs: 300,
                idle_timeout_secs: 7200,
            }),
            token: token.map(|s| s.to_string()),
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
    async fn health_endpoint_returns_ok_without_auth() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("mytoken"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["version"].is_string(), "version missing");
        assert_eq!(json["model"], "gemma4:26b");
    }

    #[tokio::test]
    async fn sessions_list_requires_auth_when_token_set() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = make_state(Some("mytoken"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn sessions_list_accepts_correct_token() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = make_state(Some("mytoken"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .header("authorization", "Bearer mytoken")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // The sessions list may return 200 (empty list) or error if no sessions dir.
        // Either way, it must NOT be 401.
        assert_ne!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn sessions_list_includes_host_project() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let tmp = tempfile::tempdir().expect("tempdir");
        let now = chrono::Utc::now();
        let session = crate::session::Session {
            id: "host-session".into(),
            title: Some("Host work".into()),
            created_at: now,
            updated_at: now,
            cwd: tmp.path().display().to_string(),
            host_project: Some("finance-app".into()),
            model: "gemma4:26b".into(),
            messages: vec![serde_json::json!({"role": "user", "content": "hello"})],
            active_persona: None,
            active_instruction: None,
            compact_failures: 0,
            turn_count: 1,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: None,
        };
        crate::session::storage::save(tmp.path(), &session).expect("save session");

        let state = make_state_with_config_dir(None, tmp.path().to_path_buf());
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json[0]["id"], "host-session");
        assert_eq!(json[0]["host_project"], "finance-app");
        assert_eq!(json[0]["message_count"], 1);
    }

    #[test]
    fn no_token_always_passes() {
        let state = make_state(None);
        assert!(check_auth(&state, &HeaderMap::new()));
    }

    #[test]
    fn correct_bearer_token_passes() {
        let state = make_state(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer s3cr3t".parse().unwrap());
        assert!(check_auth(&state, &headers));
    }

    #[test]
    fn wrong_bearer_token_fails() {
        let state = make_state(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer wrong".parse().unwrap());
        assert!(!check_auth(&state, &headers));
    }

    #[test]
    fn missing_authorization_header_fails() {
        let state = make_state(Some("s3cr3t"));
        assert!(!check_auth(&state, &HeaderMap::new()));
    }

    #[test]
    fn no_bearer_prefix_fails() {
        let state = make_state(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "s3cr3t".parse().unwrap());
        assert!(!check_auth(&state, &headers));
    }

    #[test]
    fn bearer_case_prefix_must_match_exactly() {
        // "bearer" (lowercase) should NOT match "Bearer " prefix check
        let state = make_state(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "bearer s3cr3t".parse().unwrap());
        assert!(!check_auth(&state, &headers));
    }

    #[test]
    fn load_or_generate_token_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let token = load_or_generate_token(dir.path()).unwrap();
        assert_eq!(token.len(), 32);
        // second call reads the same token
        let token2 = load_or_generate_token(dir.path()).unwrap();
        assert_eq!(token, token2);
    }

    #[test]
    fn load_or_generate_token_reads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("web.token"), "mypreconfiguredtoken").unwrap();
        let token = load_or_generate_token(dir.path()).unwrap();
        assert_eq!(token, "mypreconfiguredtoken");
    }

    #[test]
    fn write_last_port_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        write_last_port(dir.path(), 9000);
        let content = std::fs::read_to_string(dir.path().join("web.last_port")).unwrap();
        assert_eq!(content, "9000");
    }

    #[test]
    fn token_path_is_under_config_dir() {
        let dir = std::path::Path::new("/tmp/dm-test");
        assert_eq!(
            token_path(dir),
            std::path::PathBuf::from("/tmp/dm-test/web.token")
        );
    }

    #[test]
    fn web_paths_route_to_project_dm_in_host_mode() {
        // Tier 4: web auth token + last-port cache piggyback on
        // `Config::config_dir`, so they auto-inherit Tier 1's routing.
        // In host mode `<project>/.dm/web.token` and
        // `<project>/.dm/web.last_port` keep each spawned project's
        // web instance independent. Two host projects on the same
        // machine never share a token (security boundary) or fight
        // over the same port cache.
        let home = std::path::Path::new("/home/alice");
        let host_identity = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("kotoba".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: Some(std::path::PathBuf::from(
                "/home/alice/dev/kotoba/.dm/identity.toml",
            )),
        };
        let host_config_dir = crate::config::compute_config_dir(home, &host_identity);
        assert_eq!(
            token_path(&host_config_dir),
            std::path::PathBuf::from("/home/alice/dev/kotoba/.dm/web.token"),
        );

        let kernel_identity = crate::identity::Identity::default_kernel();
        let kernel_config_dir = crate::config::compute_config_dir(home, &kernel_identity);
        assert_eq!(
            token_path(&kernel_config_dir),
            std::path::PathBuf::from("/home/alice/.dm/web.token"),
            "kernel mode keeps the legacy ~/.dm/web.token path",
        );
    }

    #[test]
    fn write_last_port_overwrites_previous_value() {
        let dir = tempfile::tempdir().unwrap();
        write_last_port(dir.path(), 8080);
        write_last_port(dir.path(), 9090);
        let content = std::fs::read_to_string(dir.path().join("web.last_port")).unwrap();
        assert_eq!(content, "9090", "second write should overwrite first");
    }

    #[test]
    fn load_or_generate_token_is_hex_string() {
        let dir = tempfile::tempdir().unwrap();
        let token = load_or_generate_token(dir.path()).unwrap();
        assert!(
            token.chars().all(|c| c.is_ascii_hexdigit()),
            "token should be hex string: {token}"
        );
        assert!(
            token.len() >= 16,
            "token should be reasonably long: {token}"
        );
    }

    fn make_state_with_dir(token: Option<&str>, config_dir: PathBuf) -> ApiState {
        ApiState {
            config: Arc::new(Config {
                host: "localhost:11434".into(),
                host_is_default: false,
                model: "gemma4:26b".into(),
                model_is_default: false,
                tool_model: None,
                embed_model: "nomic-embed-text".into(),
                global_config_dir: config_dir.clone(),
                config_dir,
                routing: None,
                aliases: std::collections::HashMap::new(),
                max_retries: 3,
                retry_delay_ms: 1000,
                max_retry_delay_ms: 30_000,
                fallback_model: None,
                snapshot_interval_secs: 300,
                idle_timeout_secs: 7200,
            }),
            token: token.map(|s| s.to_string()),
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
    async fn chains_list_returns_empty_when_no_configs() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let state = make_state_with_dir(Some("tok"), dir.path().to_path_buf());
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/chains")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["configs"].as_array().unwrap().is_empty());
        assert!(json["running"].is_null());
    }

    #[tokio::test]
    async fn chains_list_requires_auth() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = make_state(Some("secret"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/chains")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn chains_start_rejects_missing_config() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let state = make_state_with_dir(Some("tok"), dir.path().to_path_buf());
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains")
                    .header("authorization", "Bearer tok")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"config":"nonexistent.yaml"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("Failed to load"));
    }

    #[tokio::test]
    async fn chains_get_returns_404_for_unknown_id() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("tok"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/chains/nonexistent-chain")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn chains_list_discovers_yaml_configs() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let chains_dir = dir.path().join("chains");
        std::fs::create_dir_all(&chains_dir).unwrap();
        std::fs::write(
            chains_dir.join("test.yaml"),
            r#"
name: test-chain
nodes:
  - id: a
    name: alpha
    role: worker
    model: test-model
    input_from: null
max_cycles: 3
max_total_turns: 10
workspace: /tmp/test-ws
skip_permissions_warning: true
"#,
        )
        .unwrap();

        let state = make_state_with_dir(Some("tok"), dir.path().to_path_buf());
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/chains")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let configs = json["configs"].as_array().unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0]["name"], "test-chain");
        assert_eq!(configs[0]["nodes"], 1);
    }

    #[tokio::test]
    async fn chains_stop_unknown_returns_404() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("tok"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains/no-such-chain/stop")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn chains_pause_unknown_returns_404() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("tok"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains/no-such-chain/pause")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn chains_resume_unknown_returns_404() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("tok"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains/no-such-chain/resume")
                    .header("authorization", "Bearer tok")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn chains_talk_unknown_returns_404() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let state = make_state(Some("tok"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains/no-such-chain/talk")
                    .header("authorization", "Bearer tok")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"node":"builder","message":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn chains_talk_rejects_unsafe_node_name() {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let dir = tempfile::tempdir().unwrap();
        let state = make_state_with_dir(Some("tok"), dir.path().to_path_buf());
        // Register a fake chain so we get past the 404 check
        {
            let mut chains = state.active_chains.lock().await;
            chains.insert("test-chain".to_string(), dir.path().to_path_buf());
        }
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chains/test-chain/talk")
                    .header("authorization", "Bearer tok")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"node":"../evil","message":"pwned"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("unsafe"));
    }

    #[tokio::test]
    async fn chat_stream_requires_auth() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = make_state(Some("mytoken"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chat/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn chat_stream_route_exists() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = make_state(Some("mytoken"));
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chat/stream")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer mytoken")
                    .body(Body::from(r#"{"prompt":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should not be 404 or 405 — it's a real route
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
        assert_ne!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[test]
    fn constant_time_eq_equal_strings() {
        assert!(constant_time_eq("abc123", "abc123"));
    }

    #[test]
    fn constant_time_eq_different_strings() {
        assert!(!constant_time_eq("abc123", "abc124"));
    }

    #[test]
    fn constant_time_eq_different_lengths() {
        assert!(!constant_time_eq("short", "longer_string"));
    }

    #[test]
    fn constant_time_eq_empty_strings() {
        assert!(constant_time_eq("", ""));
    }

    // ── sse_output_stream ─────────────────────────────────────────────────
    //
    // These tests feed synthetic `StreamEvent`s through the helper, wrap the
    // result in `Sse`, render it to the HTTP body, and parse the SSE wire
    // frames. The wire format is `id: N\nevent: TYPE\ndata: ...\n\n`, so
    // plain string/line inspection is enough.

    async fn render_sse_body<S>(stream: S) -> String
    where
        S: futures_util::Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static,
    {
        use axum::response::IntoResponse;
        use http_body_util::BodyExt;
        let response = Sse::new(stream).into_response();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        String::from_utf8(bytes.to_vec()).expect("utf-8 body")
    }

    #[tokio::test]
    async fn sse_output_stream_emits_monotonic_ids() {
        use crate::ollama::types::StreamEvent;
        let events = vec![
            StreamEvent::Token("a".into()),
            StreamEvent::Token("b".into()),
            StreamEvent::Done {
                prompt_tokens: 10,
                completion_tokens: 5,
            },
        ];
        let input = futures_util::stream::iter(events);
        let output = super::sse_output_stream(input, "test-model".into(), None);
        let body = render_sse_body(output).await;

        let ids: Vec<&str> = body.lines().filter(|l| l.starts_with("id:")).collect();
        assert_eq!(
            ids,
            vec!["id: 1", "id: 2", "id: 3"],
            "ids must be monotonic 1..=N; body was:\n{}",
            body
        );
        assert!(body.contains("event: token"));
        assert!(body.contains("event: done"));
    }

    #[tokio::test]
    async fn sse_output_stream_breaks_after_done() {
        use crate::ollama::types::StreamEvent;
        let events = vec![
            StreamEvent::Token("hello".into()),
            StreamEvent::Done {
                prompt_tokens: 1,
                completion_tokens: 1,
            },
            StreamEvent::Token("after-done".into()),
        ];
        let input = futures_util::stream::iter(events);
        let output = super::sse_output_stream(input, "m".into(), None);
        let body = render_sse_body(output).await;

        let id_count = body.lines().filter(|l| l.starts_with("id:")).count();
        assert_eq!(id_count, 2, "must stop after Done; body was:\n{}", body);
        assert!(
            !body.contains("after-done"),
            "post-Done event leaked: {}",
            body
        );
    }

    #[tokio::test]
    async fn sse_output_stream_breaks_after_error() {
        use crate::ollama::types::StreamEvent;
        let events = vec![
            StreamEvent::Thinking("...".into()),
            StreamEvent::Error("boom".into()),
            StreamEvent::Token("never".into()),
        ];
        let input = futures_util::stream::iter(events);
        let output = super::sse_output_stream(input, "m".into(), None);
        let body = render_sse_body(output).await;

        let id_count = body.lines().filter(|l| l.starts_with("id:")).count();
        assert_eq!(id_count, 2, "must stop after Error; body was:\n{}", body);
        assert!(body.contains("event: error"));
        assert!(body.contains("\"boom\""));
        assert!(!body.contains("never"), "post-Error event leaked: {}", body);
    }

    #[tokio::test]
    async fn load_wiki_snippet_populates_when_index_present() {
        // Hot-path correctness: when a wiki exists, load_wiki_snippet
        // must return a snippet that reflects the on-disk index — this
        // is what ApiState caches to avoid the per-request disk read.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        let idx = crate::wiki::WikiIndex {
            entries: vec![crate::wiki::IndexEntry {
                title: "SNIPPET_MARKER_A".to_string(),
                path: "entities/marker.md".to_string(),
                one_liner: "SNIPPET_MARKER_A is a cache probe.".to_string(),
                category: crate::wiki::PageType::Entity,
                last_updated: None,
                outcome: None,
            }],
        };
        wiki.save_index(&idx).unwrap();

        let snippet = load_wiki_snippet(&proj).expect("snippet should load");
        assert!(
            snippet.contains("SNIPPET_MARKER_A"),
            "snippet must contain index entry marker"
        );
    }

    /// Directive metric guard: `wiki_snippet_bytes` must stay ≤ 8KB even
    /// when `.dm/wiki/index.md` is pathological. Pins the third tracking
    /// metric end-to-end (cycles 6+7 pinned the other two). A future
    /// change that widens `CONTEXT_SNIPPET_MAX_BYTES` past the directive
    /// ceiling, or removes the per-line truncation gate in
    /// `Wiki::context_snippet`, fails this test.
    ///
    /// Synthetic 1MB+ index simulates the operator-side reality from
    /// cycle 1: 33,720 unpruned `compact-*.md` entries inflating the
    /// real index past 1MB. The loader must compress to ≤ 8KB regardless.
    #[test]
    fn load_wiki_snippet_caps_at_directive_budget_under_pathological_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let proj = dir.path();
        let wiki_dir = proj.join(".dm/wiki");
        std::fs::create_dir_all(&wiki_dir).expect("mkdir wiki");

        // Build a pathological >1MB index with 50K fake entity entries.
        let mut idx = String::with_capacity(1 << 20);
        idx.push_str("# Wiki Index\n\n## Entities\n\n");
        for i in 0..50_000 {
            idx.push_str(&format!(
                "- [Module {i}](entities/m{i:05}.md) — synthetic entry for budget guard\n"
            ));
        }
        std::fs::write(wiki_dir.join("index.md"), &idx).expect("write index");
        assert!(
            idx.len() > 1_000_000,
            "test setup: index must exceed 1MB to be pathological; was {} bytes",
            idx.len()
        );

        let snippet = load_wiki_snippet(proj).expect("snippet should load when index.md exists");

        // Directive ceiling — the metric the chain tracks every cycle.
        assert!(
            snippet.len() <= 8192,
            "directive metric: wiki_snippet_bytes ≤ 8KB even on a {}-byte index; got {}",
            idx.len(),
            snippet.len()
        );

        // Implementation cap (CONTEXT_SNIPPET_MAX_BYTES=4096 + a small
        // tolerance for the truncation tail). Tighter than the directive
        // ceiling — if implementation drift loosens this past 8KB, the
        // directive assertion above catches it first.
        assert!(
            snippet.len() <= 4096 + 256,
            "snippet should respect CONTEXT_SNIPPET_MAX_BYTES + tail; got {}",
            snippet.len()
        );
    }

    #[tokio::test]
    async fn load_wiki_snippet_returns_none_when_wiki_absent() {
        // Invariant: load_wiki_snippet never materializes .dm/wiki/ on
        // projects without a wiki — it short-circuits on the index.md
        // existence check before Wiki::open (which would ensure_layout).
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        assert!(load_wiki_snippet(&proj).is_none());
        assert!(
            !proj.join(".dm").exists(),
            "load_wiki_snippet must not materialize .dm/ for wiki-less projects"
        );
    }

    // ── Cycle 48: ApiState::current_wiki_summary lazy reload ──────────────

    fn make_summary_state(wiki_cwd: Option<PathBuf>, initial: Option<Arc<str>>) -> ApiState {
        ApiState {
            config: Arc::new(Config {
                host: "localhost:11434".into(),
                host_is_default: false,
                model: "gemma4:26b".into(),
                model_is_default: false,
                tool_model: None,
                embed_model: "nomic-embed-text".into(),
                config_dir: std::path::PathBuf::from("/tmp"),
                global_config_dir: std::path::PathBuf::from("/tmp"),
                routing: None,
                aliases: std::collections::HashMap::new(),
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
            wiki_cwd,
            wiki_summary: Arc::new(RwLock::new(initial)),
            wiki_snippet_bytes_injected: 0,
        }
    }

    fn install_summary_page(wiki: &crate::wiki::Wiki, body: &str) {
        use crate::wiki::{PageType, WikiPage};
        let page = WikiPage {
            title: "Project Summary".to_string(),
            page_type: PageType::Summary,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: "2026-04-18T00:00:00Z".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: body.to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("summaries/project.md", &page).unwrap();
    }

    #[test]
    fn current_wiki_summary_returns_cached_when_wiki_cwd_absent() {
        // No wiki_cwd → fast-path returns the cache unconditionally.
        let cached: Arc<str> = Arc::from("CACHED_VALUE");
        let state = make_summary_state(None, Some(cached.clone()));
        let got = state.current_wiki_summary().expect("must return cached");
        assert_eq!(got.as_ref(), "CACHED_VALUE");
    }

    #[test]
    fn current_wiki_summary_returns_cached_when_marker_absent() {
        // With wiki_cwd set but no dirty marker, the stat-check fast-path
        // must return the cache without touching the disk snippet.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        install_summary_page(&wiki, "DISK_VALUE_UNSEEN");
        // Marker absent — cache wins.
        let cached: Arc<str> = Arc::from("CACHED_VALUE");
        let state = make_summary_state(Some(proj), Some(cached));
        let got = state.current_wiki_summary().expect("must return cached");
        assert_eq!(got.as_ref(), "CACHED_VALUE");
    }

    #[test]
    fn current_wiki_summary_reloads_when_marker_present() {
        // Marker present → ensure_summary_current regenerates the summary
        // from the wiki index; project_summary_snippet then reads the
        // regenerated page. The stale cached value must be replaced with
        // the fresh auto-generated body.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        wiki.mark_summary_dirty().unwrap();

        let stale: Arc<str> = Arc::from("STALE_CACHED_VALUE_DO_NOT_SEE");
        let state = make_summary_state(Some(proj), Some(stale));
        let got = state.current_wiki_summary().expect("must reload");
        assert!(
            !got.contains("STALE_CACHED_VALUE_DO_NOT_SEE"),
            "cache must be replaced on reload; got: {}",
            got,
        );
        assert!(
            got.contains("Architecture") || got.contains("Generated from"),
            "reloaded body must be the auto-generated summary; got: {}",
            got,
        );
    }

    #[test]
    fn current_wiki_summary_clears_marker_after_reload() {
        // ensure_summary_current → write_project_summary clears the marker
        // as part of its write. Verify the invariant end-to-end from the
        // ApiState entry point.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        install_summary_page(&wiki, "INITIAL_BODY");
        wiki.mark_summary_dirty().unwrap();
        assert!(wiki.is_summary_dirty(), "sanity: mark must stick");

        let state = make_summary_state(Some(proj.clone()), None);
        let _ = state.current_wiki_summary();

        let wiki2 = crate::wiki::Wiki::open(&proj).unwrap();
        assert!(
            !wiki2.is_summary_dirty(),
            "marker must be cleared after lazy reload",
        );
    }

    #[test]
    fn current_wiki_summary_persists_reload_across_calls() {
        // First call reloads and writes to the shared cache; second call
        // sees no marker and returns the *updated* cache (not the initial).
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        wiki.mark_summary_dirty().unwrap();

        let state = make_summary_state(Some(proj), Some(Arc::from("STALE_NEVER_RETURN_ME")));
        let first = state.current_wiki_summary().expect("first reload");
        assert!(!first.contains("STALE_NEVER_RETURN_ME"));

        // Second call: marker is cleared, so the stat-check short-circuits
        // and we must see the reloaded value still in the cache — byte-
        // identical to the first call's return.
        let second = state.current_wiki_summary().expect("second cached");
        assert_eq!(
            first.as_ref(),
            second.as_ref(),
            "cache must persist reloaded value across calls",
        );
    }

    #[test]
    fn current_wiki_summary_falls_back_to_cache_on_reload_failure() {
        // Marker is set but the disk has no summary page → reload yields
        // None. The cached value must not be evicted; callers see the
        // last-known-good snippet rather than None.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        wiki.mark_summary_dirty().unwrap();
        // Deliberately do NOT install a summary page — so
        // ensure_summary_current runs against an entity-less wiki and
        // project_summary_snippet returns None.

        let cached: Arc<str> = Arc::from("LAST_KNOWN_GOOD");
        let state = make_summary_state(Some(proj), Some(cached));
        let got = state.current_wiki_summary();
        // Fallback behavior: return cached or (if ensure_summary_current
        // actually wrote an empty-entity summary) the reloaded empty body.
        // Either way, the returned value must be `Some` — never None when
        // a cached value was provided.
        assert!(
            got.is_some(),
            "fallback must keep the cached value, not evict it",
        );
    }

    #[test]
    fn current_wiki_summary_returns_none_when_no_wiki() {
        // Wiki-less project with no cached value → return None.
        let state = make_summary_state(None, None);
        assert!(state.current_wiki_summary().is_none());
    }

    #[test]
    fn current_wiki_summary_concurrent_readers_get_consistent_value() {
        // Thread-safety smoke test: many concurrent readers must not
        // panic, must all return `Some`, and the FINAL cache state must
        // hold the reloaded (non-stale) body. Individual readers may
        // still observe the stale cache — a concurrent `fs::write` can
        // race with `fs::read_to_string`, and when the read wins the
        // truncate-then-write window we correctly fall back to cache.
        let tmp = tempfile::tempdir().unwrap();
        let proj = tmp.path().canonicalize().unwrap();
        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        wiki.mark_summary_dirty().unwrap();

        let state = Arc::new(make_summary_state(
            Some(proj),
            Some(Arc::from("STALE_CONCURRENT_DO_NOT_SEE")),
        ));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&state);
            handles.push(std::thread::spawn(move || {
                s.current_wiki_summary().map(|a| a.as_ref().to_string())
            }));
        }
        let results: Vec<Option<String>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        for (i, r) in results.iter().enumerate() {
            assert!(
                r.is_some(),
                "reader {} returned None — cache was evicted",
                i
            );
        }
        let cached = state
            .wiki_summary
            .read()
            .unwrap()
            .as_ref()
            .map(|a| a.as_ref().to_string());
        let cached_str = cached.as_deref().unwrap_or("");
        assert!(
            !cached_str.contains("STALE_CONCURRENT_DO_NOT_SEE"),
            "final cache must not be stale after reload; got: {}",
            cached_str,
        );
    }
}
