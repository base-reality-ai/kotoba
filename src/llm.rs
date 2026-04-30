//! Provider-neutral LLM client trait.
//!
//! Canonical dm's conversation capture, TUI, and orchestration paths were
//! historically wired to a concrete `OllamaClient`. Host projects that want to
//! drive the same loops with a non-Ollama backend (Anthropic, OpenAI, Gemini,
//! a corporate gateway) had no canonical seam: model strings flowed through,
//! but the wire was always Ollama-shaped HTTP.
//!
//! `LlmClient` is the seam. Implementations supply the few primitives the
//! conversation capture loop actually needs (current model id, context window
//! lookup, chat-with-tools). `OllamaClient` implements it; host projects can
//! ship their own implementation and pass it to
//! [`crate::conversation::run_conversation_capture_with_client_in_config_dir`].
//!
//! Scoped on purpose: this trait is the minimum surface needed for canonical
//! conversation capture, not a complete model abstraction. Streaming, fallback
//! model selection, embeddings, and model listing remain Ollama-specific until
//! a host project surfaces a need for them in canonical code.

use crate::ollama::types::{ChatResponse, ToolDefinition};

/// Provider-neutral chat backend used by the host-facing conversation capture
/// entry point. `Send + Sync` so trait objects can cross await points and be
/// shared across tasks; implementations must be safe to call concurrently
/// (the Ollama implementation is — `chat` borrows immutably).
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Currently-selected model identifier — informs prompts that surface
    /// the model name (e.g. compaction summaries) and gates fallback logic.
    fn model(&self) -> &str;

    /// Best-known context window (in tokens) for `model`. Implementations
    /// that don't track per-model context return a sensible default; the
    /// only consumer is `CompactionThresholds::from_context_window`, where
    /// under-reporting just causes earlier (safe) compaction.
    async fn model_context_limit(&self, model: &str) -> usize;

    /// Send `messages` (with optional `tools`) and await a single response.
    /// Implementations are responsible for transport-level retries, rate
    /// limit handling, and timeout discipline. Returning an `Err` surfaces
    /// the failure to the conversation loop, which decides whether to retry,
    /// fall back, or propagate.
    async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ChatResponse>;
}

#[async_trait::async_trait]
impl LlmClient for crate::ollama::client::OllamaClient {
    fn model(&self) -> &str {
        crate::ollama::client::OllamaClient::model(self)
    }

    async fn model_context_limit(&self, model: &str) -> usize {
        crate::ollama::client::OllamaClient::model_context_limit(self, model).await
    }

    async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ChatResponse> {
        crate::ollama::client::OllamaClient::chat(self, messages, tools).await
    }
}
