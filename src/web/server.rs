use crate::session::short_id;
use crate::web::{handlers::event_to_json, state::AppState};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path as AxumPath, Query, State,
    },
    http::{header, Method},
    middleware,
    response::{Html, IntoResponse},
    routing::{delete, get, post},
    Json, Router,
};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/dashboard", get(serve_dashboard))
        .route("/ws", get(ws_handler))
        .route("/chat", post(chat_handler))
        .route("/cancel", post(cancel_handler))
        .route("/share", post(share_handler))
        // Dashboard API routes
        .route("/api/sessions", get(api_sessions_list))
        .route("/api/sessions", post(api_sessions_create))
        .route("/api/sessions/:id", delete(api_sessions_delete))
        .route("/api/schedules", get(api_schedules_list))
        .route("/api/schedules", post(api_schedules_add))
        .route("/api/schedules/:id", delete(api_schedules_remove))
        .route("/api/log", get(api_log_tail))
        .route("/api/status", get(api_status))
        .route("/api/chain", get(api_chain_status))
        .route("/api/chain/start", post(api_chain_start))
        .route("/api/chain/stop", post(api_chain_stop))
        .route("/api/chain/pause", post(api_chain_pause))
        .route("/api/chain/resume", post(api_chain_resume))
        .route("/api/chain/talk", post(api_chain_talk))
        .route("/api/chain/add", post(api_chain_add))
        .route("/api/chain/remove", post(api_chain_remove))
        .route("/api/chain/model", post(api_chain_model))
        .route("/api/chain/log", get(api_chain_log))
        .route("/api/models", get(api_models))
        .route("/api/model", post(api_set_model))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
                .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE]),
        )
        .with_state(state)
}

async fn auth_middleware(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: middleware::Next,
) -> axum::response::Response {
    let Some(expected) = &state.token else {
        return next.run(req).await;
    };
    let authorized = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| crate::api::constant_time_eq(t, expected));
    if authorized {
        next.run(req).await
    } else {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "unauthorized. Try: include 'Authorization: Bearer <token>' header with a valid token."})),
        )
            .into_response()
    }
}

pub async fn run(
    state: AppState,
    event_rx: mpsc::Receiver<crate::tui::BackendEvent>,
    port: u16,
) -> anyhow::Result<()> {
    run_with_gpu(state, event_rx, port, None).await
}

pub async fn run_with_gpu(
    state: AppState,
    mut event_rx: mpsc::Receiver<crate::tui::BackendEvent>,
    port: u16,
    gpu_rx: Option<tokio::sync::watch::Receiver<Option<crate::gpu::GpuStats>>>,
) -> anyhow::Result<()> {
    let _ = crate::logging::init_in_config_dir("web", &state.config_dir);
    // Relay: agent mpsc → broadcast fan-out to all WS clients.
    // Events are serialized here once instead of per-client.
    let bus = state.event_bus.clone();
    let bus2 = bus.clone();
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            let json = event_to_json(&event).to_string();
            let _ = bus.send(json); // ok if zero subscribers
        }
    });

    // GPU relay: watch channel → broadcast GpuUpdate events to web clients
    if let Some(mut gpu_watch) = gpu_rx {
        tokio::spawn(async move {
            loop {
                if gpu_watch.changed().await.is_err() {
                    break;
                }
                let stats_opt = gpu_watch.borrow().clone();
                if let Some(g) = stats_opt {
                    use crate::tui::BackendEvent;
                    let ev = BackendEvent::GpuUpdate {
                        util_pct: g.util_pct,
                        vram_used_mb: g.vram_used_mb,
                        vram_total_mb: g.vram_total_mb,
                        temp_c: g.temp_c,
                    };
                    let json = event_to_json(&ev).to_string();
                    let _ = bus2.send(json);
                }
            }
        });
    }

    let app = build_router(state);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    crate::logging::log(&format!("[dm] Web UI: http://localhost:{}", port));

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn serve_index() -> impl IntoResponse {
    Html(include_str!("../../assets/index.html"))
}

async fn serve_dashboard() -> impl IntoResponse {
    Html(include_str!("../../assets/dashboard.html"))
}

// ── Dashboard API handlers ────────────────────────────────────────────────────

async fn api_sessions_list(State(state): State<AppState>) -> impl IntoResponse {
    let mut sessions_json = Vec::new();
    let identity = crate::identity::load_for_cwd();

    sessions_json.push(serde_json::json!({
        "session_id": state.session_id,
        "host_project": identity.host_project,
        "model": state.model(),
        "status": "running",
        "client_count": 1,
        "last_active": chrono::Utc::now().format("%Y-%m-%d %H:%M").to_string(),
    }));

    if let Ok(historical) = crate::session::storage::list(&state.config_dir) {
        let mut historical: Vec<_> = historical
            .into_iter()
            .filter(|s| s.id != state.session_id)
            .collect();
        historical.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        for s in historical {
            sessions_json.push(serde_json::json!({
                "session_id": s.id,
                "host_project": s.host_project,
                "model": s.model,
                "status": "archived",
                "client_count": 0,
                "last_active": s.updated_at.format("%Y-%m-%d %H:%M").to_string(),
            }));
        }
    }

    Json(sessions_json)
}

async fn api_sessions_create(State(_state): State<AppState>) -> impl IntoResponse {
    let session_id = uuid::Uuid::new_v4().to_string();
    Json(serde_json::json!({ "session_id": session_id }))
}

async fn api_sessions_delete(
    State(_state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    // Attempt to kill via daemon if available; otherwise no-op stub.
    if crate::daemon::daemon_socket_exists() {
        let kill_result = async {
            let mut dc = crate::daemon::DaemonClient::connect().await?;
            dc.send_request("session.end", serde_json::json!({ "session_id": id }))
                .await?;
            anyhow::Ok(())
        }
        .await;
        let _ = kill_result;
    }
    axum::http::StatusCode::NO_CONTENT
}

async fn api_schedules_list(State(state): State<AppState>) -> impl IntoResponse {
    match crate::daemon::scheduler::load_schedules(&state.config_dir) {
        Ok(tasks) => Json(serde_json::to_value(tasks).unwrap_or(serde_json::json!([]))),
        Err(_) => Json(serde_json::json!([])),
    }
}

async fn api_schedules_add(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let cron = body["cron"].as_str().unwrap_or("").to_string();
    let prompt = body["prompt"].as_str().unwrap_or("").to_string();
    if cron.is_empty() || prompt.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({"error": "cron and prompt are required. Try: include both 'cron' (schedule expression) and 'prompt' (text) in the request body."}),
            ),
        );
    }
    match crate::daemon::scheduler::load_schedules(&state.config_dir) {
        Ok(mut tasks) => {
            let id = crate::daemon::scheduler::generate_task_id();
            let next_run = (chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339();
            tasks.push(crate::daemon::scheduler::ScheduledTask {
                id: id.clone(),
                cron,
                prompt,
                model: None,
                last_run: None,
                next_run: Some(next_run),
            });
            if let Err(e) = crate::daemon::scheduler::save_schedules(&state.config_dir, &tasks) {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": e.to_string()})),
                );
            }
            (
                axum::http::StatusCode::CREATED,
                Json(serde_json::json!({"id": id})),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        ),
    }
}

async fn api_schedules_remove(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    if let Ok(mut tasks) = crate::daemon::scheduler::load_schedules(&state.config_dir) {
        tasks.retain(|t| t.id != id);
        let _ = crate::daemon::scheduler::save_schedules(&state.config_dir, &tasks);
    }
    axum::http::StatusCode::NO_CONTENT
}

async fn api_log_tail(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let n: usize = params
        .get("tail")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let lines = crate::daemon::server::tail_log(&state.config_dir, n).unwrap_or_default();
    Json(serde_json::json!(lines))
}

async fn api_status(State(state): State<AppState>) -> impl IntoResponse {
    let chain_running = crate::orchestrate::chain_status().is_some();
    Json(serde_json::json!({
        "uptime_secs": state.start_time.elapsed().as_secs(),
        "model": state.model(),
        "version": env!("CARGO_PKG_VERSION"),
        "chain_running": chain_running,
    }))
}

async fn api_models(State(state): State<AppState>) -> impl IntoResponse {
    let ollama_url = state.base_url.replace("/api", "");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    match client.get(format!("{}/api/tags", ollama_url)).send().await {
        Ok(resp) => {
            let json: serde_json::Value = resp.json().await.unwrap_or(serde_json::json!({}));
            let models = json["models"].as_array().cloned().unwrap_or_default();
            Json(serde_json::json!({
                "ok": true,
                "models": models,
                "active_model": state.model(),
            }))
            .into_response()
        }
        Err(e) => Json(serde_json::json!({
            "ok": false,
            "error": e.to_string(),
            "active_model": state.model(),
        }))
        .into_response(),
    }
}

async fn api_set_model(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let new_model = body["model"].as_str().unwrap_or("").trim().to_string();
    if new_model.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(
                serde_json::json!({"ok": false, "error": "model name is required. Try: include 'model' in the request body."}),
            ),
        );
    }
    match state.model.write() {
        Ok(mut m) => (*m).clone_from(&new_model),
        Err(_) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"ok": false, "error": "lock poisoned. Try: restart the daemon to clear poisoned shared state."}),
                ),
            );
        }
    }
    (
        axum::http::StatusCode::OK,
        Json(serde_json::json!({"ok": true, "model": new_model})),
    )
}

/// Returns the current chain orchestration state, or null if no chain is running.
async fn api_chain_status() -> impl IntoResponse {
    match crate::orchestrate::chain_status() {
        Some(state) => {
            let cycle_display = if state.config.loop_forever {
                format!("{}/∞", state.current_cycle)
            } else {
                format!("{}/{}", state.current_cycle, state.config.max_cycles)
            };
            let active_node = state
                .active_node_index
                .and_then(|i| state.config.nodes.get(i))
                .map(|n| n.name.clone());
            let nodes: Vec<serde_json::Value> = state
                .config
                .nodes
                .iter()
                .map(|n| {
                    let status = state
                        .node_statuses
                        .get(&n.name)
                        .map_or_else(|| "unknown".into(), |s| s.to_string());
                    let durations = state
                        .node_durations
                        .get(&n.name)
                        .cloned()
                        .unwrap_or_default();
                    let failures = state.node_failures.get(&n.name).copied().unwrap_or(0);
                    let prompt_tokens = state.node_prompt_tokens.get(&n.name).copied().unwrap_or(0);
                    let completion_tokens = state
                        .node_completion_tokens
                        .get(&n.name)
                        .copied()
                        .unwrap_or(0);
                    serde_json::json!({
                        "id": n.id,
                        "name": n.name,
                        "model": n.model,
                        "status": status,
                        "runs": durations.len(),
                        "avg_duration_secs": if durations.is_empty() { 0.0 } else {
                            durations.iter().sum::<f64>() / durations.len() as f64
                        },
                        "failures": failures,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    })
                })
                .collect();
            Json(serde_json::json!({
                "running": true,
                "chain_name": state.config.name,
                "chain_id": state.chain_id,
                "cycle": cycle_display,
                "active_node": active_node,
                "turns_used": state.turns_used,
                "max_total_turns": state.config.max_total_turns,
                "total_duration_secs": state.total_duration_secs,
                "nodes": nodes,
                "last_signal": state.last_signal.as_ref().map(|s| format!("{:?}", s)),
                "last_abort_reason": state.last_abort_reason,
            }))
        }
        None => Json(serde_json::json!({
            "running": false,
        })),
    }
}

async fn api_chain_start(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if crate::orchestrate::chain_status().is_some() {
        return Json(
            serde_json::json!({"ok": false, "error": "A chain is already running. Try: POST /api/chain/stop, then retry."}),
        );
    }
    let config_path = body["config_path"].as_str().unwrap_or("").to_string();
    if config_path.is_empty() {
        return Json(
            serde_json::json!({"ok": false, "error": "config_path is required. Try: include 'config_path' (path to a chain.yaml) in the request body."}),
        );
    }
    let path = std::path::PathBuf::from(&config_path);
    let chain_config = match crate::orchestrate::load_chain_config(&path) {
        Ok(c) => c,
        Err(e) => {
            return Json(serde_json::json!({"ok": false, "error": format!(
                "Failed to load config: {}. Try: confirm the YAML is well-formed and the path is correct.",
                e
            )}))
        }
    };
    let workspace = chain_config.workspace.clone();
    let chain_id = format!("chain-{}", uuid::Uuid::new_v4().as_simple());
    let orch_config = crate::orchestrate::OrchestrationConfig {
        chain: chain_config,
        chain_id: chain_id.clone(),
        retry: crate::conversation::RetrySettings::default(),
        resume_state: None,
    };
    let active_model = state.model();
    let chain_client =
        crate::ollama::client::OllamaClient::new(state.base_url.clone(), active_model.clone());
    let session_id = uuid::Uuid::new_v4().to_string();
    let chain_registry = crate::tools::registry::default_registry(
        &session_id,
        &state.config_dir,
        &state.base_url,
        &active_model,
        "",
    );
    let (chain_tx, chain_rx) = tokio::sync::broadcast::channel(64);
    let relay_rx = chain_tx.subscribe();
    let handle = tokio::task::spawn(async move {
        if let Err(e) = crate::orchestrate::runner::run_orchestration(
            orch_config,
            chain_client,
            chain_registry,
            Some(chain_tx),
        )
        .await
        {
            let _ = e;
        }
    });
    crate::orchestrate::set_chain(handle, workspace);
    if let Ok(mut rx) = state.chain_event_rx.lock() {
        *rx = Some(chain_rx);
    }
    let bus = state.event_bus;
    tokio::spawn(async move {
        let mut rx = relay_rx;
        while let Ok(event) = rx.recv().await {
            if let Some(json) = crate::web::handlers::chain_event_to_json(&event) {
                let _ = bus.send(json.to_string());
            }
        }
    });
    Json(serde_json::json!({"ok": true, "chain_id": chain_id}))
}

async fn api_chain_stop(State(state): State<AppState>) -> impl IntoResponse {
    match crate::orchestrate::stop_chain() {
        Ok(()) => {
            crate::orchestrate::clear_chain();
            if let Ok(mut rx) = state.chain_event_rx.lock() {
                *rx = None;
            }
            Json(serde_json::json!({"ok": true, "action": "stop"}))
        }
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_pause() -> impl IntoResponse {
    match crate::orchestrate::pause_chain() {
        Ok(()) => Json(serde_json::json!({"ok": true, "action": "pause"})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_resume() -> impl IntoResponse {
    match crate::orchestrate::resume_chain() {
        Ok(()) => Json(serde_json::json!({"ok": true, "action": "resume"})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_talk(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let node = body["node"].as_str().unwrap_or("").to_string();
    let message = body["message"].as_str().unwrap_or("").to_string();
    if node.is_empty() || message.is_empty() {
        return Json(
            serde_json::json!({"ok": false, "error": "node and message are required. Try: include both 'node' and 'message' in the request body."}),
        );
    }
    match crate::orchestrate::chain_talk(&node, &message) {
        Ok(()) => Json(serde_json::json!({"ok": true, "action": "talk", "node": node})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_add(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("").to_string();
    let model = body["model"].as_str().unwrap_or("").to_string();
    let role = body["role"].as_str().unwrap_or(&name).to_string();
    let input_from = body["input_from"].as_str().map(String::from);
    if name.is_empty() || model.is_empty() {
        return Json(
            serde_json::json!({"ok": false, "error": "name and model are required. Try: include both 'name' (new node ID) and 'model' in the request body."}),
        );
    }
    let node = crate::orchestrate::types::ChainNodeConfig {
        id: name.clone(),
        name: name.clone(),
        role,
        model,
        description: None,
        system_prompt_override: None,
        system_prompt_file: None,
        input_from,
        max_retries: 1,
        timeout_secs: 3600,
        max_tool_turns: 200,
    };
    match crate::orchestrate::chain_add_node(node) {
        Ok(()) => Json(serde_json::json!({"ok": true, "action": "add", "name": name})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_remove(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("").to_string();
    if name.is_empty() {
        return Json(
            serde_json::json!({"ok": false, "error": "name is required. Try: include 'name' (node ID to remove) in the request body."}),
        );
    }
    match crate::orchestrate::chain_remove_node(&name) {
        Ok(()) => Json(serde_json::json!({"ok": true, "action": "remove", "name": name})),
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_model(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    let node = body["node"].as_str().unwrap_or("").to_string();
    let model = body["model"].as_str().unwrap_or("").to_string();
    if node.is_empty() || model.is_empty() {
        return Json(
            serde_json::json!({"ok": false, "error": "node and model are required. Try: include both 'node' (target node) and 'model' (new model) in the request body."}),
        );
    }
    match crate::orchestrate::chain_model(&node, &model) {
        Ok(()) => {
            Json(serde_json::json!({"ok": true, "action": "model", "node": node, "model": model}))
        }
        Err(e) => Json(serde_json::json!({"ok": false, "error": e.to_string()})),
    }
}

async fn api_chain_log(Query(params): Query<HashMap<String, String>>) -> impl IntoResponse {
    let cycle: Option<usize> = params.get("cycle").and_then(|v| v.parse().ok());
    match crate::orchestrate::chain_log(cycle) {
        Ok(entries) => Json(serde_json::json!({
            "ok": true,
            "entries": entries,
        })),
        Err(e) => Json(serde_json::json!({
            "ok": false,
            "error": e.to_string(),
        })),
    }
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws(socket, state))
}

async fn handle_ws(mut socket: WebSocket, state: AppState) {
    let mut rx = state.event_bus.subscribe();

    // Send initial state so the UI knows which model/session it's talking to.
    let init = serde_json::json!({
        "type": "init",
        "model": state.model(),
        "session_id": state.session_id,
    })
    .to_string();
    if socket.send(Message::Text(init)).await.is_err() {
        return;
    }

    while let Ok(json_str) = rx.recv().await {
        if socket.send(Message::Text(json_str)).await.is_err() {
            break;
        }
    }
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let text = body["message"].as_str().unwrap_or("").to_string();
    if !text.is_empty() {
        let _ = state.user_tx.send(text).await;
    }
    axum::http::StatusCode::OK
}

async fn cancel_handler(State(state): State<AppState>) -> impl IntoResponse {
    let _ = state.cancel_tx.send(true);
    axum::http::StatusCode::OK
}

async fn share_handler(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let use_md = params.get("format").is_some_and(|f| f == "md");
    match crate::session::storage::load(&state.config_dir, &state.session_id) {
        Ok(sess) => {
            if use_md {
                let md = crate::share::render_session_markdown(&sess);
                let filename = format!("dm-session-{}.md", short_id(&sess.id));
                let disposition = format!("attachment; filename=\"{}\"", filename);
                (
                    axum::http::StatusCode::OK,
                    [
                        ("content-type", "text/markdown; charset=utf-8".to_string()),
                        ("content-disposition", disposition),
                    ],
                    md,
                )
                    .into_response()
            } else {
                match crate::share::render_session(&sess) {
                    Ok(html) => {
                        let filename = format!("dm-session-{}.html", short_id(&sess.id));
                        let disposition = format!("attachment; filename=\"{}\"", filename);
                        (
                            axum::http::StatusCode::OK,
                            [
                                ("content-type", "text/html; charset=utf-8".to_string()),
                                ("content-disposition", disposition),
                            ],
                            html,
                        )
                            .into_response()
                    }
                    Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                        .into_response(),
                }
            }
        }
        Err(e) => (axum::http::StatusCode::NOT_FOUND, e.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::web::state::AppState;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tokio::sync::{broadcast, mpsc, watch};
    use tower::ServiceExt;

    fn make_state() -> (AppState, mpsc::Receiver<String>) {
        let (event_bus, _) = broadcast::channel::<String>(16);
        let (user_tx, user_rx) = mpsc::channel::<String>(8);
        let (cancel_tx, _) = watch::channel(false);
        let state = AppState {
            event_bus,
            user_tx,
            cancel_tx,
            model: std::sync::Arc::new(std::sync::RwLock::new("test-model".to_string())),
            base_url: "http://localhost:11434/api".to_string(),
            session_id: "test-session-id".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            token: None,
            start_time: std::time::Instant::now(),
            chain_event_rx: std::sync::Arc::new(std::sync::Mutex::new(None)),
        };
        (state, user_rx)
    }

    #[tokio::test]
    async fn test_get_index_returns_html() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let html = String::from_utf8_lossy(&body);
        assert!(
            html.contains("<title>Dark Matter</title>"),
            "body: {}",
            &html[..200]
        );
    }

    #[tokio::test]
    async fn test_post_chat_routes_to_user_tx() {
        let (state, mut user_rx) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"message":"hello from test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let received = user_rx.recv().await.expect("should receive message");
        assert_eq!(received, "hello from test");
    }

    #[tokio::test]
    async fn test_post_cancel_returns_ok() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/cancel")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_status_returns_model_and_version() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["model"], "test-model");
        assert!(json["version"].is_string(), "version should be a string");
    }

    #[tokio::test]
    async fn test_api_sessions_list_contains_session_id() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let arr = json.as_array().expect("should be array");
        assert!(!arr.is_empty(), "sessions list should not be empty");
        assert_eq!(arr[0]["session_id"], "test-session-id");
        assert_eq!(arr[0]["model"], "test-model");
    }

    #[tokio::test]
    async fn test_api_schedules_add_missing_fields_returns_400() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/schedules")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"cron": "0 * * * *"}"#)) // missing prompt
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert!(json["error"].is_string(), "should have error field");
    }

    #[tokio::test]
    async fn test_get_dashboard_returns_html() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/dashboard")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let html = String::from_utf8_lossy(&body);
        assert!(
            html.contains("<!"),
            "should be HTML: {}",
            &html[..200.min(html.len())]
        );
    }

    #[tokio::test]
    async fn test_post_chat_empty_message_does_not_send() {
        let (state, mut user_rx) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"message":""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        // Channel should be empty — no message forwarded for blank input
        assert!(
            user_rx.try_recv().is_err(),
            "empty message should not be forwarded to user_tx"
        );
    }

    #[tokio::test]
    async fn test_api_sessions_create_returns_uuid() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/sessions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert!(
            json["session_id"].is_string(),
            "session_id should be a string UUID"
        );
        let id = json["session_id"].as_str().unwrap();
        assert!(!id.is_empty(), "session_id should not be empty");
    }

    #[tokio::test]
    async fn test_api_log_tail_returns_array() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/log?tail=10")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert!(json.is_array(), "log tail should return JSON array");
    }

    #[tokio::test]
    async fn test_api_schedules_list_returns_array() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/schedules")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert!(json.is_array(), "schedules list should return JSON array");
    }

    #[tokio::test]
    async fn test_api_schedules_add_missing_cron_returns_400() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/schedules")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "check deps"}"#)) // missing cron
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_api_chain_no_chain_returns_not_running() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/chain")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(
            json["running"], false,
            "should report not running when no chain"
        );
    }

    #[tokio::test]
    async fn test_api_chain_talk_no_chain_returns_error() {
        crate::orchestrate::clear_chain();
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/talk")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"node":"builder","message":"focus on tests"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(
            json["error"].as_str().unwrap().contains("No chain"),
            "error: {}",
            json["error"]
        );
    }

    #[tokio::test]
    async fn test_api_chain_talk_missing_fields_returns_error() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/talk")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"node":"builder"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().unwrap().contains("required"));
    }

    #[tokio::test]
    async fn test_api_chain_add_no_chain_returns_error() {
        crate::orchestrate::clear_chain();
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/add")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"reviewer","model":"llama3"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(
            json["error"].as_str().unwrap().contains("No chain"),
            "error: {}",
            json["error"]
        );
    }

    #[tokio::test]
    async fn test_api_chain_add_missing_fields_returns_error() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/add")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"reviewer"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().unwrap().contains("required"));
    }

    #[tokio::test]
    async fn test_api_chain_remove_no_chain_returns_error() {
        crate::orchestrate::clear_chain();
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/remove")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"name":"builder"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(
            json["error"].as_str().unwrap().contains("No chain"),
            "error: {}",
            json["error"]
        );
    }

    #[tokio::test]
    async fn test_api_chain_remove_missing_name_returns_error() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/remove")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().unwrap().contains("required"));
    }

    #[tokio::test]
    async fn test_api_chain_model_no_chain_returns_error() {
        crate::orchestrate::clear_chain();
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/model")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"node":"builder","model":"llama3:70b"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(
            json["error"].as_str().unwrap().contains("No chain"),
            "error: {}",
            json["error"]
        );
    }

    #[tokio::test]
    async fn test_api_chain_model_missing_fields_returns_error() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/model")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"node":"builder"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().unwrap().contains("required"));
    }

    #[tokio::test]
    async fn test_api_chain_log_no_chain_returns_error() {
        crate::orchestrate::clear_chain();
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/chain/log")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        assert_eq!(json["ok"], false);
        assert!(
            json["error"].as_str().unwrap().contains("No chain"),
            "error: {}",
            json["error"]
        );
    }

    fn make_state_with_token(token: &str) -> AppState {
        let (event_bus, _) = broadcast::channel::<String>(16);
        let (user_tx, _) = mpsc::channel::<String>(8);
        let (cancel_tx, _) = watch::channel(false);
        AppState {
            event_bus,
            user_tx,
            cancel_tx,
            model: std::sync::Arc::new(std::sync::RwLock::new("test-model".to_string())),
            base_url: "http://localhost:11434/api".to_string(),
            session_id: "test-session".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            token: Some(token.to_string()),
            start_time: std::time::Instant::now(),
            chain_event_rx: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    #[tokio::test]
    async fn web_auth_rejects_without_token() {
        let state = make_state_with_token("secret123");
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn web_auth_passes_with_correct_token() {
        let state = make_state_with_token("secret123");
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .header("authorization", "Bearer secret123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_status_uptime_non_zero() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["uptime_secs"].is_u64(),
            "uptime_secs should be a number"
        );
    }

    #[tokio::test]
    async fn web_chain_stop_clears_globals() {
        let tmp = tempfile::tempdir().unwrap();
        let handle = tokio::spawn(async { /* no-op */ });
        crate::orchestrate::set_chain(handle, tmp.path().to_path_buf());

        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chain/stop")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["action"], "stop");
    }

    #[test]
    fn web_chain_event_rx_default_is_none() {
        let (state, _) = make_state();
        assert!(state.chain_event_rx.lock().unwrap().is_none());
    }

    #[test]
    fn short_session_id_no_panic() {
        for id in ["", "a", "ab", "abc1234"] {
            let truncated = short_id(id);
            let filename = format!("dm-session-{}.md", truncated);
            assert!(filename.starts_with("dm-session-"));
            assert!(filename.ends_with(".md"));
        }
        let full = "abcdefghijklmnop";
        assert_eq!(short_id(full), "abcdefgh");
    }

    #[tokio::test]
    async fn api_models_returns_json_structure() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json.get("active_model").is_some(),
            "should include active_model"
        );
        // When Ollama is not running, ok may be false — either way we get valid JSON
        assert!(json.get("ok").is_some());
    }

    #[tokio::test]
    async fn api_set_model_validates_empty() {
        let (state, _) = make_state();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/model")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":""}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn api_set_model_switches_model() {
        let (state, _) = make_state();
        let model_ref = state.model.clone();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/model")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"llama3:8b"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["model"], "llama3:8b");
        assert_eq!(*model_ref.read().unwrap(), "llama3:8b");
    }

    #[tokio::test]
    async fn api_models_connection_failure_returns_error() {
        let (event_bus, _) = broadcast::channel::<String>(16);
        let (user_tx, _) = mpsc::channel::<String>(8);
        let (cancel_tx, _) = watch::channel(false);
        let state = AppState {
            event_bus,
            user_tx,
            cancel_tx,
            model: std::sync::Arc::new(std::sync::RwLock::new("test".to_string())),
            base_url: "http://127.0.0.1:1/api".to_string(), // unreachable
            session_id: "test".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            token: None,
            start_time: std::time::Instant::now(),
            chain_event_rx: std::sync::Arc::new(std::sync::Mutex::new(None)),
        };
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], false);
        assert!(json["error"].as_str().is_some());
        assert_eq!(json["active_model"], "test");
    }

    #[tokio::test]
    async fn cors_preflight_returns_allow_origin() {
        let (event_bus, _) = broadcast::channel::<String>(16);
        let (user_tx, _) = mpsc::channel::<String>(8);
        let (cancel_tx, _) = watch::channel(false);
        let state = AppState {
            event_bus,
            user_tx,
            cancel_tx,
            model: std::sync::Arc::new(std::sync::RwLock::new("test".to_string())),
            base_url: "http://127.0.0.1:1/api".to_string(),
            session_id: "test".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            token: None,
            start_time: std::time::Instant::now(),
            chain_event_rx: std::sync::Arc::new(std::sync::Mutex::new(None)),
        };
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/models")
                    .header("origin", "http://localhost:3000")
                    .header("access-control-request-method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response
                .headers()
                .contains_key("access-control-allow-origin"),
            "CORS preflight should include allow-origin header"
        );
    }

    #[tokio::test]
    async fn test_api_sessions_list_includes_archived_host_sessions() {
        let tmp = tempfile::tempdir().unwrap();
        let mut session = crate::session::Session::new(
            tmp.path().display().to_string(),
            "gemma4:26b".to_string(),
        );
        session.id = "archived-host-session".into();
        session.host_project = Some("finance-app".into());
        crate::session::storage::save(tmp.path(), &session).unwrap();

        let (mut state, _) = make_state();
        state.config_dir = tmp.path().to_path_buf();

        let app = build_router(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");
        let arr = json.as_array().expect("should be array");

        // One running, one archived
        assert_eq!(
            arr.len(),
            2,
            "should contain both running and archived session"
        );

        let archived = arr
            .iter()
            .find(|s| s["status"] == "archived")
            .expect("archived session");
        assert_eq!(archived["session_id"], "archived-host-session");
        assert_eq!(archived["host_project"], "finance-app");
        assert_eq!(archived["model"], "gemma4:26b");
    }
}
