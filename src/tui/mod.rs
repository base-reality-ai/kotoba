//! Terminal User Interface (TUI).
//!
//! Provides the primary interactive surface for the `dm` kernel, managing
//! chat sessions, model output streaming, and tool execution feedback.

pub mod agent;
pub mod app;
pub mod commands;
pub mod events;
pub mod hyperlinks;
pub mod input;
pub mod markdown;
pub mod run;
pub mod ui;

use serde_json::Value;
use tokio::sync::oneshot;

/// Events sent from the agent task → TUI task
#[derive(Debug)]
pub enum BackendEvent {
    /// A streaming text token
    StreamToken(String),
    /// Reasoning/thinking content from reasoning models
    StreamThinking(String),
    /// Streaming is complete for this round
    StreamDone {
        content: String,
        tool_calls: Vec<crate::ollama::types::ToolCall>,
    },
    /// A tool is about to execute
    ToolStarted {
        name: String,
        args: Value,
        active_model: String,
    },
    /// A tool has finished
    ToolFinished {
        name: String,
        output: String,
        is_error: bool,
    },
    /// A file was edited — show a colored diff in the TUI
    FileDiff { path: String, diff: String },
    /// Agent needs a permission decision before proceeding
    PermissionRequired {
        tool_name: String,
        args: Value,
        /// Human-readable risk banner (e.g. "rm -rf on root path") rendered
        /// above the prompt when the dispatcher gated on the bash risk
        /// classifier. `None` for normal prompts.
        reason: Option<String>,
        reply: oneshot::Sender<PermissionDecision>,
    },
    /// Agent wants to ask the user a free-form question
    AskUserQuestion {
        question: String,
        options: Vec<String>,
        reply: oneshot::Sender<String>,
    },
    /// Fatal or recoverable error message
    Error(String),
    /// Informational notice — non-error status text (e.g. retry progress, reconnection). Renders dim cyan, not red.
    Notice(String),
    /// Report of current permission rules (response to /permissions)
    PermissionsReport(String),
    /// Agent has finished the current turn, with Ollama token usage
    TurnComplete {
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    /// The user cancelled the current agent turn via Escape
    Cancelled,
    /// Background task has generated a session title
    TitleGenerated(String),
    /// Context is approaching the compaction threshold — warn the user
    ContextWarning(String),
    /// GPU stats update from the background poller
    GpuUpdate {
        util_pct: u8,
        vram_used_mb: u64,
        vram_total_mb: u64,
        temp_c: Option<u8>,
    },
    /// Current context usage — emitted after every turn for status bar display
    ContextUsage {
        /// Estimated tokens currently in the conversation
        used: usize,
        /// Model's reported context window size (`num_ctx` from /api/show)
        limit: usize,
    },
    /// Tool results were pruned to reduce context size
    ContextPruned {
        chars_removed: usize,
        messages_affected: usize,
    },
    /// A line of live output from a streaming tool (bash stdout/stderr)
    ToolOutput { name: String, line: String },
    /// File writes were intercepted — user must approve before they land on disk
    StagedChangeset(Vec<crate::changeset::PendingChange>),
    /// A sub-agent was spawned by the `AgentTool`
    AgentSpawned { prompt_preview: String, depth: u8 },
    /// A sub-agent finished (paired with `AgentSpawned` by depth)
    AgentFinished { depth: u8, elapsed_ms: u64 },
    /// Live inference performance update (debounced; emitted ≤4×/s during streaming)
    PerfUpdate {
        tok_per_sec: f32,
        ttft_ms: u64,
        total_tokens: usize,
    },
    /// A user turn is about to start — TUI should snapshot display entries for undo
    TurnStarted,
    /// `/undo` succeeded — TUI should restore display entries from snapshot
    UndoComplete,
    /// `/undo` called with no snapshot available
    NothingToUndo,
    /// File changes were applied — save for `/undo-files`
    ChangesetApplied(Vec<crate::changeset::PendingChange>),
    /// Compaction is starting. Carries the stage for future stage-aware UI
    /// (currently all stages render as `[compacting...]`). Always paired with
    /// a `CompactionCompleted` — even on error — via a scoped guard at the
    /// emit site, so the status-bar banner can never get stuck on.
    CompactionStarted(crate::compaction::CompactionStage),
    /// Compaction finished (successfully or not). Clears the banner.
    CompactionCompleted,
    /// Session was switched via /resume
    SessionSwitched {
        old_id: String,
        new_id: String,
        new_full_id: String,
        title: String,
        message_count: usize,
    },
}

/// User's response to a TUI permission prompt
#[derive(Debug, Clone)]
pub enum PermissionDecision {
    AllowOnce,
    AlwaysAllow,
    DenyOnce,
    AlwaysDeny,
}
