use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::{broadcast, mpsc, watch};

/// Shared state passed to all axum route handlers.
///
/// Events are pre-serialized to JSON strings before broadcast so that
/// `BackendEvent` (which has non-Clone `oneshot::Sender` fields) does not
/// need to implement `Clone`.
#[derive(Clone)]
pub struct AppState {
    /// Broadcast channel carrying pre-serialized JSON event strings.
    /// A relay task in `server::run` reads from the agent mpsc and re-broadcasts here.
    pub event_bus: broadcast::Sender<String>,
    /// Forward user messages to the agent conversation loop.
    pub user_tx: mpsc::Sender<String>,
    /// Cancel signal (watch channel; send `true` to abort the current turn).
    pub cancel_tx: watch::Sender<bool>,
    /// Active model name — mutable so the web UI can switch models at runtime.
    pub model: Arc<RwLock<String>>,
    /// Ollama base URL (e.g. "<http://localhost:11434/api>") for creating chain clients.
    pub base_url: String,
    /// Session UUID (shown truncated in the status bar).
    pub session_id: String,
    /// Dark Matter config directory — used by /share to load the session.
    pub config_dir: PathBuf,
    /// Optional bearer token for API auth. None = no auth required.
    pub token: Option<String>,
    /// Server start time — used for uptime reporting.
    pub start_time: std::time::Instant,
    /// Chain event broadcast receiver — kept alive so future SSE/WS endpoints can subscribe.
    pub chain_event_rx:
        Arc<Mutex<Option<broadcast::Receiver<crate::daemon::protocol::DaemonEvent>>>>,
}

impl AppState {
    /// Read the current active model name.
    pub fn model(&self) -> String {
        self.model.read().expect("model RwLock poisoned").clone()
    }
}
