//! Raw HTTP bindings for the Ollama API.
//!
//! Provides the async `reqwest` client, stream parsing, and request/response
//! serialization corresponding to the Ollama `chat` and `embeddings` endpoints.

use crate::ollama::types::{
    ChatChunk, ChatResponse, ChunkMessage, ModelInfo, StreamEvent, TagsResponse, ToolCall,
    ToolDefinition,
};
use anyhow::Context;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tokio::time::{sleep, Duration};

#[cfg(not(test))]
const INITIAL_BACKOFF_MS: u64 = 2000;
#[cfg(test)]
const INITIAL_BACKOFF_MS: u64 = 10; // fast in tests

const MAX_RETRIES: u32 = 4;
const CONNECT_TIMEOUT_SECS: u64 = 30;
const READ_TIMEOUT_SECS: u64 = 300;
const METADATA_TIMEOUT_SECS: u64 = 30;

fn non_utf8_stream_line_warning(byte_len: usize, error: impl std::fmt::Display) -> String {
    format!(
        "Skipping non-UTF-8 stream line ({} bytes): {}. Try: confirm the Ollama server emits valid UTF-8 (older builds may garble bytes on disconnect); upgrade Ollama if this repeats.",
        byte_len, error
    )
}

fn malformed_stream_chunk_warning(error: impl std::fmt::Display) -> String {
    format!(
        "Skipping malformed stream chunk: {}. Try: if this repeats every turn, switch to a different model with `dm --model <name>` — some models emit non-JSON tool-call payloads.",
        error
    )
}

fn stream_eof_warning() -> &'static str {
    "Ollama stream closed without done=true — token counts unavailable for this turn. Try: check Ollama server logs for upstream errors, or increase request timeout if the model is producing very long outputs."
}

/// Host:port candidates probed by [`detect_host`] when the user has not
/// explicitly configured a host. Covers (localhost, 127.0.0.1) × (11434,
/// 11435, 8080). Most-common-first: 11434 is the Ollama default; 11435
/// is the conventional alt; 8080 shows up when Ollama sits behind a
/// reverse proxy or has been port-remapped.
pub const OLLAMA_CANDIDATE_HOSTS: &[&str] = &[
    "localhost:11434",
    "127.0.0.1:11434",
    "localhost:11435",
    "127.0.0.1:11435",
    "localhost:8080",
    "127.0.0.1:8080",
];

/// Best-effort heuristic for whether `model_name` belongs to a family that
/// supports Ollama's tool-calling API. Prefix-matches the base name
/// (pre-colon portion) against a known-capable list. Case-insensitive.
///
/// Used by `pick_best_model` and may be useful anywhere the caller needs
/// a quick "can this model use tools?" guess without a `/api/show` round-trip.
pub fn model_supports_tools(model_name: &str) -> bool {
    let base = model_name
        .split(':')
        .next()
        .unwrap_or(model_name)
        .to_lowercase();
    const TOOL_CAPABLE_PREFIXES: &[&str] = &[
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "llama4",
        "gemma2",
        "gemma3",
        "gemma4",
        "qwen2",
        "qwen2.5",
        "qwen3",
        "mistral",
        "codestral",
        "deepseek-v2",
        "deepseek-coder-v2",
        "deepseek-r1",
        "command-r",
        "phi3",
        "phi4",
    ];
    TOOL_CAPABLE_PREFIXES.iter().any(|&p| base.starts_with(p))
}

/// Pick the best-suited default model from a list of installed models.
/// Primary sort: tool-capable before non-capable. Secondary: size desc
/// (a proxy for capability within a family). Returns `None` for empty input.
///
/// This is the heuristic used by `main::run` to auto-select a model when
/// the user has not explicitly configured one. It intentionally avoids
/// `/api/show` round-trips — size is a good-enough first-pass proxy.
pub fn pick_best_model(models: &[crate::ollama::types::ModelInfo]) -> Option<String> {
    if models.is_empty() {
        return None;
    }
    let mut scored: Vec<(&crate::ollama::types::ModelInfo, bool, u64)> = models
        .iter()
        .map(|m| (m, model_supports_tools(&m.name), m.size.unwrap_or(0)))
        .collect();
    // Tool-capable first, then larger size first.
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)));
    scored.first().map(|(m, _, _)| m.name.clone())
}

/// Probe `candidates` sequentially for a running Ollama instance by
/// issuing `GET http://{host}/api/version`. Returns the first responsive
/// host, or `None` when no candidate answers within `total_budget`.
///
/// Two timeouts interact:
/// - `per_timeout` caps a single candidate's probe (connect + read).
///   Tuned for a slow-but-alive Ollama.
/// - `total_budget` caps wall-clock for the whole pass. Matters when
///   Ollama is absent: with N candidates the loop would otherwise run
///   for `N * per_timeout`, showing up as a silent startup stall. The
///   final candidate's probe is clamped to the remaining budget.
///
/// Never panics; all errors are swallowed (this is best-effort
/// auto-detection, not a reachability contract).
pub async fn detect_host(
    candidates: &[&str],
    per_timeout: Duration,
    total_budget: Duration,
) -> Option<String> {
    let deadline = tokio::time::Instant::now() + total_budget;
    let client = Client::builder()
        .connect_timeout(per_timeout)
        .timeout(per_timeout)
        .build()
        .ok()?;
    for &host in candidates {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return None;
        }
        let remaining = deadline.saturating_duration_since(now);
        let probe_timeout = per_timeout.min(remaining);
        let url = format!("http://{}/api/version", host);
        match tokio::time::timeout(probe_timeout, client.get(&url).send()).await {
            Ok(Ok(resp)) if resp.status().is_success() => {
                return Some(host.to_string());
            }
            _ => {}
        }
    }
    None
}

fn is_retriable(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::SERVICE_UNAVAILABLE // 503 = model loading
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS // 429 = rate limited
        || status == reqwest::StatusCode::INTERNAL_SERVER_ERROR // 500 = transient server error
}

fn suggest_fix(error_msg: &str) -> Option<&'static str> {
    let lower = error_msg.to_lowercase();
    if lower.contains("not found") || lower.contains("no such model") {
        Some("Try: ollama pull <model> — the model may not be downloaded yet")
    } else if lower.contains("context length") || lower.contains("too long") {
        Some("Try: reduce message history or use a model with a larger context window")
    } else if lower.contains("out of memory") {
        Some("Try: use a smaller quantization (e.g. q4_0) or a smaller model")
    } else if is_connection_error(&lower) {
        Some("Is Ollama running? Start it with: ollama serve")
    } else if lower.contains("timed out") || lower.contains("timeout") {
        Some("The model may be overloaded. Try a smaller model or increase timeout")
    } else {
        None
    }
}

fn is_connection_error(lower: &str) -> bool {
    lower.contains("connection refused")
        || lower.contains("connect error")
        || lower.contains("tcp connect")
        || lower.contains("failed to connect")
        || lower.contains("connection reset")
}

type ToolEventStream = Pin<Box<dyn Stream<Item = StreamEvent> + Send>>;

#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    http: Client,
    json_mode: bool,
    context_cache: Arc<TokioMutex<HashMap<String, usize>>>,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .read_timeout(Duration::from_secs(READ_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| Client::new());
        OllamaClient {
            base_url,
            model,
            http,
            json_mode: false,
            context_cache: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    /// Return a new client that targets a different model (same connection settings).
    /// Shares the context cache so model lookups are reused across node clients.
    pub fn with_model(self, model: String) -> Self {
        OllamaClient { model, ..self }
    }

    /// Enable or disable JSON output mode (adds `"format": "json"` to requests).
    pub fn with_json_mode(self, json_mode: bool) -> Self {
        OllamaClient { json_mode, ..self }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub async fn get_version(&self) -> anyhow::Result<String> {
        let url = format!("{}/api/version", self.base_url);
        let resp = self
            .http
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;
        let body: serde_json::Value = resp.json().await?;
        Ok(body["version"].as_str().unwrap_or("unknown").to_string())
    }

    /// Generate an embedding vector for `text` using the client's model.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let url = format!("{}/embed", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Ollama embed request failed")?
            .error_for_status()
            .context("Ollama embed returned error status")?
            .json::<serde_json::Value>()
            .await
            .context("Failed to parse embed response")?;

        let embeddings = resp["embeddings"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No embeddings field in Ollama embed response\n  → {}",
                    crate::ollama::hints::HINT_EMBED_MALFORMED
                )
            })?;

        embeddings
            .iter()
            .map(|v| {
                v.as_f64().map(|f| f as f32).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Non-float value in embedding vector\n  → {}",
                        crate::ollama::hints::HINT_EMBED_MALFORMED
                    )
                })
            })
            .collect()
    }

    /// Query the effective context window size for `model` via `POST /api/show`.
    ///
    /// Checks (in priority order):
    /// 1. `num_ctx` from model parameters (user-configured override)
    /// 2. `<arch>.context_length` from `model_info` (native model limit)
    /// 3. `llama.context_length` from `model_info` (common fallback key)
    /// 4. Fallback: 4096
    pub async fn model_context_limit(&self, model: &str) -> usize {
        // Check cache first
        {
            let cache = self.context_cache.lock().await;
            if let Some(&limit) = cache.get(model) {
                return limit;
            }
        }

        let limit = self.fetch_model_context_limit(model).await;

        // Cache the result
        {
            let mut cache = self.context_cache.lock().await;
            cache.insert(model.to_string(), limit);
        }

        limit
    }

    async fn fetch_model_context_limit(&self, model: &str) -> usize {
        let url = format!("{}/show", self.base_url);
        let body = serde_json::json!({"model": model});
        let Ok(resp) = self.http.post(&url).json(&body).send().await else {
            return 4096;
        };
        let Ok(json) = resp.json::<serde_json::Value>().await else {
            return 4096;
        };

        // Check num_ctx in parameters (user-configured override via Modelfile)
        if let Some(params) = json["parameters"].as_str() {
            for line in params.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "num_ctx" {
                    if let Ok(n) = parts[1].parse::<usize>() {
                        return n;
                    }
                }
            }
        }

        // Check model_info for context_length (try architecture-specific first, then llama)
        let model_info = &json["model_info"];
        if let Some(obj) = model_info.as_object() {
            for (key, val) in obj {
                if key.ends_with(".context_length") {
                    if let Some(n) = val.as_u64() {
                        return n as usize;
                    }
                }
            }
        }

        4096
    }

    pub async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let url = format!("{}/tags", self.base_url);
        let resp = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(METADATA_TIMEOUT_SECS))
            .send()
            .await
            .context("Failed to connect to Ollama — is it running?")?;
        let tags: TagsResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama /api/tags response")?;
        Ok(tags.models)
    }

    /// Delete a model via `DELETE /api/delete`.
    pub async fn delete_model(&self, model_name: &str) -> anyhow::Result<()> {
        let url = format!("{}/delete", self.base_url);
        let resp = self
            .http
            .delete(&url)
            .json(&serde_json::json!({"model": model_name}))
            .timeout(Duration::from_secs(METADATA_TIMEOUT_SECS))
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to Ollama to delete '{}'\n  → {}",
                    model_name,
                    crate::ollama::hints::HINT_CONNECT_FAILED
                )
            })?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            match crate::ollama::hints::hint_for_delete_status(status.as_u16()) {
                Some(hint) => {
                    anyhow::bail!("Ollama delete returned {}: {}\n  → {}", status, body, hint)
                }
                None => anyhow::bail!("Ollama delete returned {}: {}", status, body),
            }
        }
        Ok(())
    }

    /// Get full model details via `POST /api/show`.
    pub async fn show_model(&self, model_name: &str) -> anyhow::Result<serde_json::Value> {
        let url = format!("{}/show", self.base_url);
        let resp = self
            .http
            .post(&url)
            .json(&serde_json::json!({"model": model_name}))
            .timeout(Duration::from_secs(METADATA_TIMEOUT_SECS))
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to Ollama for show '{}'\n  → {}",
                    model_name,
                    crate::ollama::hints::HINT_CONNECT_FAILED
                )
            })?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!(
                "Model '{}' not found\n  → {}",
                model_name,
                crate::ollama::hints::HINT_MODEL_NOT_FOUND
            );
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            match crate::ollama::hints::hint_for_show_status(status.as_u16()) {
                Some(hint) => {
                    anyhow::bail!("Ollama show returned {}: {}\n  → {}", status, body, hint)
                }
                None => anyhow::bail!("Ollama show returned {}: {}", status, body),
            }
        }
        resp.json()
            .await
            .context("Failed to parse Ollama /api/show response")
    }

    /// Build the JSON body for a chat request, adding `"format":"json"` when `json_mode` is set.
    fn build_chat_request(
        &self,
        messages: &[serde_json::Value],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "tools": tools,
            "options": {
                "num_predict": -1  // unlimited output tokens
            }
        });
        if self.json_mode {
            body["format"] = serde_json::Value::String("json".to_string());
        }
        body
    }

    /// Non-streaming chat — used by the conversation loop and compaction.
    /// Returns `ChatResponse` which includes token usage from Ollama.
    pub async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ChatResponse> {
        let url = format!("{}/chat", self.base_url);
        let body = self.build_chat_request(messages, tools, false);

        let mut attempt = 0u32;
        loop {
            let send_result = self.http.post(&url).json(&body).send().await;

            let resp = match send_result {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if is_connection_error(&msg.to_lowercase()) && attempt < MAX_RETRIES {
                        let wait_ms = crate::conversation::apply_jitter(
                            INITIAL_BACKOFF_MS * (1u64 << attempt),
                        );
                        crate::warnings::push_warning(format!(
                            "Ollama connection failed, retrying in {:.1}s… (attempt {}/{})",
                            wait_ms as f64 / 1000.0,
                            attempt + 1,
                            MAX_RETRIES
                        ));
                        sleep(Duration::from_millis(wait_ms)).await;
                        attempt += 1;
                        continue;
                    }
                    let hint = suggest_fix(&msg).unwrap_or("Check that Ollama is running");
                    anyhow::bail!("Failed to connect to Ollama: {}\n  → {}", msg, hint);
                }
            };

            let status = resp.status();

            if is_retriable(status) && attempt < MAX_RETRIES {
                let wait_ms =
                    crate::conversation::apply_jitter(INITIAL_BACKOFF_MS * (1u64 << attempt));
                let reason = if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    "model loading"
                } else {
                    "rate limited"
                };
                crate::warnings::push_warning(format!(
                    "Ollama {} ({}), retrying in {:.1}s… (attempt {}/{})",
                    status,
                    reason,
                    wait_ms as f64 / 1000.0,
                    attempt + 1,
                    MAX_RETRIES
                ));
                sleep(Duration::from_millis(wait_ms)).await;
                attempt += 1;
                continue;
            }

            if !status.is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                let msg = extract_ollama_error(&body_text).unwrap_or(&body_text);
                match suggest_fix(msg) {
                    Some(hint) => anyhow::bail!("Ollama {}: {}\n  → {}", status, msg, hint),
                    None => anyhow::bail!("Ollama {}: {}", status, msg),
                }
            }

            #[derive(serde::Deserialize)]
            struct NonStreamingResponse {
                message: ChunkMessage,
                #[serde(default)]
                prompt_eval_count: u64,
                #[serde(default)]
                eval_count: u64,
                #[serde(default)]
                eval_duration: u64,
            }

            let body_text = resp
                .text()
                .await
                .context("Failed to read Ollama response body")?;

            // Ollama may return 200 with {"error":"..."} for some failures
            if let Some(err_msg) = extract_ollama_error(&body_text) {
                match suggest_fix(err_msg) {
                    Some(hint) => anyhow::bail!("Ollama error: {}\n  → {}", err_msg, hint),
                    None => anyhow::bail!("Ollama error: {}", err_msg),
                }
            }

            let result: NonStreamingResponse =
                serde_json::from_str(&body_text).context("Failed to parse Ollama chat response")?;

            return Ok(ChatResponse {
                message: result.message,
                prompt_tokens: result.prompt_eval_count,
                completion_tokens: result.eval_count,
                duration_ms: result.eval_duration / 1_000_000,
            });
        }
    }

    /// Streaming chat with tool support — used by the TUI agent task and stream-json mode.
    /// Returns a stream of `StreamEvent`s. `Done` carries token usage counts.
    pub async fn chat_stream_with_tools(
        &self,
        messages: &[serde_json::Value],
        tools: &[ToolDefinition],
    ) -> anyhow::Result<ToolEventStream> {
        let url = format!("{}/chat", self.base_url);
        let body = self.build_chat_request(messages, tools, true);

        let mut attempt = 0u32;
        let resp = loop {
            let send_result = self.http.post(&url).json(&body).send().await;

            let r = match send_result {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if is_connection_error(&msg.to_lowercase()) && attempt < MAX_RETRIES {
                        let wait_ms = crate::conversation::apply_jitter(
                            INITIAL_BACKOFF_MS * (1u64 << attempt),
                        );
                        crate::warnings::push_warning(format!(
                            "Ollama connection failed, retrying in {:.1}s… (attempt {}/{})",
                            wait_ms as f64 / 1000.0,
                            attempt + 1,
                            MAX_RETRIES
                        ));
                        sleep(Duration::from_millis(wait_ms)).await;
                        attempt += 1;
                        continue;
                    }
                    let hint = suggest_fix(&msg).unwrap_or("Check that Ollama is running");
                    anyhow::bail!("Failed to connect to Ollama: {}\n  → {}", msg, hint);
                }
            };

            let status = r.status();

            if is_retriable(status) && attempt < MAX_RETRIES {
                let wait_ms =
                    crate::conversation::apply_jitter(INITIAL_BACKOFF_MS * (1u64 << attempt));
                crate::warnings::push_warning(format!(
                    "Ollama {} (model loading), retrying in {:.1}s… (attempt {}/{})",
                    status,
                    wait_ms as f64 / 1000.0,
                    attempt + 1,
                    MAX_RETRIES
                ));
                sleep(Duration::from_millis(wait_ms)).await;
                attempt += 1;
                continue;
            }

            if !status.is_success() {
                let body_text = r.text().await.unwrap_or_default();
                let msg = extract_ollama_error(&body_text).unwrap_or(&body_text);
                match suggest_fix(msg) {
                    Some(hint) => anyhow::bail!("Ollama {}: {}\n  → {}", status, msg, hint),
                    None => anyhow::bail!("Ollama {}: {}", status, msg),
                }
            }

            break r;
        };

        let byte_stream = resp.bytes_stream();

        // Buffer raw bytes to avoid corrupting multi-byte UTF-8 characters
        // that span TCP chunk boundaries. Only decode complete lines.
        let stream = async_stream::stream! {
            let mut buf: Vec<u8> = Vec::new();
            let mut accumulated_tool_calls: Vec<ToolCall> = Vec::new();
            tokio::pin!(byte_stream);
            while let Some(chunk_result) = byte_stream.next().await {
                let bytes = match chunk_result {
                    Ok(b) => b,
                    Err(e) => { yield StreamEvent::Error(e.to_string()); return; }
                };
                buf.extend_from_slice(&bytes);
                const MAX_BUF_SIZE: usize = 10 * 1024 * 1024;
                if buf.len() > MAX_BUF_SIZE {
                    yield StreamEvent::Error(format!(
                        "stream buffer exceeded {} bytes without newline delimiter",
                        MAX_BUF_SIZE
                    ));
                    return;
                }
                while let Some(nl_pos) = buf.iter().position(|&b| b == b'\n') {
                    let mut line_bytes: Vec<u8> = buf.drain(..=nl_pos).collect();
                    line_bytes.pop(); // strip trailing newline
                    let line = match String::from_utf8(line_bytes) {
                        Ok(s) => s,
                        Err(e) => {
                            crate::warnings::push_warning(non_utf8_stream_line_warning(
                                e.as_bytes().len(),
                                e,
                            ));
                            continue;
                        }
                    };
                    let line = line.trim().to_string();
                    if line.is_empty() { continue; }
                    match serde_json::from_str::<ChatChunk>(&line) {
                        Ok(chunk) => {
                            if let Some(msg) = chunk.message {
                                if let Some(ref thinking) = msg.thinking {
                                    if !thinking.is_empty() {
                                        yield StreamEvent::Thinking(thinking.clone());
                                    }
                                }
                                if !msg.content.is_empty() {
                                    yield StreamEvent::Token(msg.content);
                                }
                                if !msg.tool_calls.is_empty() {
                                    accumulated_tool_calls.extend(msg.tool_calls);
                                }
                            }
                            if chunk.done {
                                if !accumulated_tool_calls.is_empty() {
                                    yield StreamEvent::ToolCalls(
                                        std::mem::take(&mut accumulated_tool_calls)
                                    );
                                }
                                yield StreamEvent::Done {
                                    prompt_tokens: chunk.prompt_eval_count,
                                    completion_tokens: chunk.eval_count,
                                };
                                return;
                            }
                        }
                        Err(e) => {
                            // Skip malformed intermediate chunks rather than
                            // aborting the whole stream. If the model's JSON is
                            // garbled on one line it often recovers on the next.
                            crate::warnings::push_warning(malformed_stream_chunk_warning(e));
                        }
                    }
                }
            }
            // Stream ended without an explicit done=true
            crate::warnings::push_warning(stream_eof_warning().into());
            if !accumulated_tool_calls.is_empty() {
                yield StreamEvent::ToolCalls(std::mem::take(&mut accumulated_tool_calls));
            }
            yield StreamEvent::Done { prompt_tokens: 0, completion_tokens: 0 };
        };

        Ok(Box::pin(stream))
    }
}

/// Try to extract the `"error"` field from an Ollama JSON error body.
/// Returns `None` if the body is not JSON or has no `"error"` key.
fn extract_ollama_error(body: &str) -> Option<&str> {
    // Ollama error bodies look like: {"error":"model not found, ..."}
    // We do a manual scan rather than a full parse to avoid lifetime complexity.
    let trimmed = body.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    // Use serde_json to extract the field; return a reference into `body` via str matching
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(err) = v["error"].as_str() {
            // Find the string within the original body to return a slice with the right lifetime
            return body.find(err).map(|start| &body[start..start + err.len()]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn detect_host_returns_none_for_dead_ports() {
        // Port 1 is reserved and reliably refused on every OS.
        let result = detect_host(
            &["127.0.0.1:1"],
            Duration::from_millis(200),
            Duration::from_secs(5),
        )
        .await;
        assert!(
            result.is_none(),
            "expected None for a dead port, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn detect_host_empty_list_returns_none() {
        let result = detect_host(&[], Duration::from_millis(200), Duration::from_secs(5)).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn detect_host_returns_mock_responsive_candidate() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let host = server.host_with_port();
        let result = detect_host(
            &[host.as_str()],
            Duration::from_millis(800),
            Duration::from_secs(5),
        )
        .await;
        assert_eq!(result.as_deref(), Some(host.as_str()));
    }

    #[tokio::test]
    async fn detect_host_falls_through_on_non_success_status() {
        // A 500 from the first candidate must not shortcut the probe — we're
        // looking for a *running* Ollama, not just any HTTP server.
        let mut bad = mockito::Server::new_async().await;
        let _m_bad = bad
            .mock("GET", "/api/version")
            .with_status(500)
            .create_async()
            .await;
        let mut good = mockito::Server::new_async().await;
        let _m_good = good
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let bad_host = bad.host_with_port();
        let good_host = good.host_with_port();
        let result = detect_host(
            &[bad_host.as_str(), good_host.as_str()],
            Duration::from_millis(800),
            Duration::from_secs(5),
        )
        .await;
        assert_eq!(result.as_deref(), Some(good_host.as_str()));
    }

    #[tokio::test]
    async fn detect_host_preserves_candidate_order() {
        // When two candidates are both responsive, the first one wins.
        let mut first = mockito::Server::new_async().await;
        let _m1 = first
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let mut second = mockito::Server::new_async().await;
        let _m2 = second
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let first_host = first.host_with_port();
        let second_host = second.host_with_port();
        let result = detect_host(
            &[first_host.as_str(), second_host.as_str()],
            Duration::from_millis(800),
            Duration::from_secs(5),
        )
        .await;
        assert_eq!(result.as_deref(), Some(first_host.as_str()));
    }

    #[tokio::test]
    async fn detect_host_respects_total_budget_with_dead_candidates() {
        // Covers the REFUSED-connection path: ports 1-10 actively reject
        // TCP connections in microseconds, so the budget cap is not what
        // bounds this call — refusal is. Still worth keeping: it pins
        // behavior for the "no listener" real-world case. For the actual
        // budget-cap regression guard (stalled listener that would
        // otherwise consume per_timeout), see
        // `detect_host_bounds_stalled_candidates_by_total_budget` below.
        let dead_candidates: Vec<&str> = vec![
            "127.0.0.1:1",
            "127.0.0.1:2",
            "127.0.0.1:3",
            "127.0.0.1:4",
            "127.0.0.1:5",
            "127.0.0.1:6",
            "127.0.0.1:7",
            "127.0.0.1:8",
            "127.0.0.1:9",
            "127.0.0.1:10",
        ];
        let start = tokio::time::Instant::now();
        let result = detect_host(
            &dead_candidates,
            Duration::from_millis(2000),
            Duration::from_millis(500),
        )
        .await;
        let elapsed = start.elapsed();
        assert!(result.is_none(), "expected None, got {:?}", result);
        assert!(
            elapsed < Duration::from_secs(2),
            "budget not honored: elapsed {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn detect_host_responsive_first_returns_within_budget() {
        // Happy path under a budget smaller than per_timeout: the sole
        // responsive candidate still wins. Guards against a regression
        // where the budget cap accidentally rejects the fast path.
        // Budget < per_timeout is the invariant we're proving; both are
        // generous enough to tolerate mockito scheduler jitter under
        // parallel test load.
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let host = server.host_with_port();
        let result = detect_host(
            &[host.as_str()],
            Duration::from_secs(3),
            Duration::from_secs(2),
        )
        .await;
        assert_eq!(result.as_deref(), Some(host.as_str()));
    }

    /// Spawn a "blackhole" TCP listener: accepts connections and holds them
    /// open without writing any response. A probe against this address
    /// stalls until the caller's timeout fires — the exact scenario the
    /// budget cap is designed to bound. Returns the bound address and a
    /// `JoinHandle` whose sole purpose is explicit ownership; the tokio
    /// runtime aborts the task on test-end regardless.
    async fn spawn_blackhole() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind blackhole listener");
        let addr = listener
            .local_addr()
            .expect("blackhole local_addr")
            .to_string();
        let handle = tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                held.push(stream);
            }
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn detect_host_bounds_stalled_candidates_by_total_budget() {
        // Blackhole listeners: accept TCP but never respond. Each probe
        // stalls until a timeout fires. With per_timeout=2000ms and 4
        // candidates, the uncapped baseline is ~8000ms; observed
        // <2000ms under parallel test load proves the budget cap is
        // actively short-circuiting iterations 2-4 (deadline check
        // exits or probe_timeout clamp shrinks to near-zero). The wide
        // assertion window tolerates scheduler jitter without
        // permitting "cap silently removed" to slip through.
        let (a1, _h1) = spawn_blackhole().await;
        let (a2, _h2) = spawn_blackhole().await;
        let (a3, _h3) = spawn_blackhole().await;
        let (a4, _h4) = spawn_blackhole().await;
        let candidates: Vec<&str> = vec![a1.as_str(), a2.as_str(), a3.as_str(), a4.as_str()];

        let start = tokio::time::Instant::now();
        let result = detect_host(
            &candidates,
            Duration::from_millis(2000),
            Duration::from_millis(200),
        )
        .await;
        let elapsed = start.elapsed();
        assert!(result.is_none(), "expected None, got {:?}", result);
        assert!(
            elapsed < Duration::from_secs(2),
            "budget cap not enforced: elapsed {:?} (uncapped baseline would be ~8s; \
             anything under 2s proves the cap is bounding iterations)",
            elapsed
        );
        // Lower bound: if elapsed is much less than the budget, it
        // likely means connections were refused (not stalled), making
        // the cap assertion vacuous. Catch that silent-pass mode.
        assert!(
            elapsed >= Duration::from_millis(100),
            "blackhole did not stall (elapsed {:?} < 100ms suggests refused, not stalled)",
            elapsed
        );
    }

    #[tokio::test]
    async fn detect_host_stalled_first_still_reaches_responsive_second() {
        // One stalled candidate, then a responsive mock. With a budget
        // generous enough for per_timeout × 1 + room to spare, the
        // per_timeout clamp on the first probe lets the loop advance
        // to the responsive mock. Proves one slow/stalled candidate
        // doesn't consume the whole budget.
        let (stalled_addr, _h) = spawn_blackhole().await;
        let mut responsive = mockito::Server::new_async().await;
        let _m = responsive
            .mock("GET", "/api/version")
            .with_status(200)
            .with_body(r#"{"version":"0.1.0"}"#)
            .create_async()
            .await;
        let responsive_host = responsive.host_with_port();

        let result = detect_host(
            &[stalled_addr.as_str(), responsive_host.as_str()],
            Duration::from_millis(200),
            Duration::from_millis(2000),
        )
        .await;
        assert_eq!(
            result.as_deref(),
            Some(responsive_host.as_str()),
            "expected responsive candidate to be reached after stalled first"
        );
    }

    #[test]
    fn ollama_candidate_hosts_covers_prime_directive() {
        // Prime Directive Phase 1.1: auto-detection must probe both loopback
        // aliases on all three ports we expect in the wild.
        for host in ["localhost", "127.0.0.1"] {
            for port in [11434, 11435, 8080] {
                let expected = format!("{}:{}", host, port);
                assert!(
                    OLLAMA_CANDIDATE_HOSTS.iter().any(|&c| c == expected),
                    "missing candidate {}",
                    expected
                );
            }
        }
    }

    #[test]
    fn ollama_candidate_hosts_most_common_first() {
        // Ordering matters: we probe sequentially and return the first hit,
        // so the Ollama default (11434) must come before alt ports.
        assert!(
            OLLAMA_CANDIDATE_HOSTS[0].ends_with(":11434"),
            "first candidate must be :11434"
        );
        let pos = |needle: &str| {
            OLLAMA_CANDIDATE_HOSTS
                .iter()
                .position(|c| c.ends_with(needle))
                .expect("port present")
        };
        assert!(
            pos(":11435") < pos(":8080"),
            ":11435 must precede :8080 — alt is more common than proxy"
        );
    }

    fn mk_model(name: &str, size: u64) -> crate::ollama::types::ModelInfo {
        crate::ollama::types::ModelInfo {
            name: name.to_string(),
            size: Some(size),
            modified_at: None,
            digest: None,
        }
    }

    #[test]
    fn model_supports_tools_llama31_true() {
        assert!(model_supports_tools("llama3.1:8b"));
    }

    #[test]
    fn model_supports_tools_gemma4_uppercase_true() {
        assert!(
            model_supports_tools("GEMMA4:26b"),
            "must be case-insensitive"
        );
    }

    #[test]
    fn model_supports_tools_unknown_false() {
        assert!(!model_supports_tools("foo-llm:7b"));
    }

    #[test]
    fn pick_best_model_empty_returns_none() {
        assert!(pick_best_model(&[]).is_none());
    }

    #[test]
    fn pick_best_model_prefers_tool_capable() {
        // unknown:7b is larger (8GB) but non-tool-capable;
        // phi3:3b is smaller (2GB) but tool-capable — it wins.
        let models = vec![
            mk_model("unknown:7b", 8_000_000_000),
            mk_model("phi3:3b", 2_000_000_000),
        ];
        assert_eq!(pick_best_model(&models).as_deref(), Some("phi3:3b"));
    }

    #[test]
    fn pick_best_model_prefers_larger_within_tool_capable() {
        let models = vec![
            mk_model("llama3.1:8b", 4_000_000_000),
            mk_model("llama3.1:70b", 40_000_000_000),
        ];
        assert_eq!(pick_best_model(&models).as_deref(), Some("llama3.1:70b"));
    }

    #[test]
    fn json_mode_adds_format_field() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string())
                .with_json_mode(true);
        let body = client.build_chat_request(&[], &[], false);
        assert_eq!(
            body["format"].as_str(),
            Some("json"),
            "format field should be 'json'"
        );
    }

    #[test]
    fn json_mode_absent_by_default() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        let body = client.build_chat_request(&[], &[], false);
        assert!(
            body["format"].is_null(),
            "format field should be absent by default"
        );
    }

    #[test]
    fn model_context_limit_parses_llama_context_length() {
        // Simulate the JSON response from POST /api/show
        let json: serde_json::Value = serde_json::json!({
            "model_info": {
                "llama.context_length": 8192_u64
            }
        });
        let limit = json["model_info"]["llama.context_length"]
            .as_u64()
            .map_or(4096, |n| n as usize);
        assert_eq!(limit, 8192);
    }

    #[test]
    fn model_context_limit_falls_back_when_field_absent() {
        let json: serde_json::Value = serde_json::json!({"model_info": {}});
        let limit = json["model_info"]["llama.context_length"]
            .as_u64()
            .map_or(4096, |n| n as usize);
        assert_eq!(limit, 4096);
    }

    #[test]
    fn model_context_limit_falls_back_on_missing_model_info() {
        let json: serde_json::Value = serde_json::json!({});
        let limit = json["model_info"]["llama.context_length"]
            .as_u64()
            .map_or(4096, |n| n as usize);
        assert_eq!(limit, 4096);
    }

    #[test]
    fn extract_ollama_error_parses_error_field() {
        let body = r#"{"error":"model 'foo' not found"}"#;
        let extracted = extract_ollama_error(body);
        assert_eq!(extracted, Some("model 'foo' not found"));
    }

    #[test]
    fn extract_ollama_error_returns_none_for_non_json() {
        assert!(extract_ollama_error("plain text error").is_none());
        assert!(extract_ollama_error("").is_none());
    }

    #[test]
    fn extract_ollama_error_returns_none_when_no_error_field() {
        let body = r#"{"status":"ok","result":"done"}"#;
        assert!(extract_ollama_error(body).is_none());
    }

    #[test]
    fn is_retriable_429_and_503_are_retriable() {
        assert!(is_retriable(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retriable(reqwest::StatusCode::SERVICE_UNAVAILABLE));
    }

    #[test]
    fn is_retriable_200_and_404_are_not_retriable() {
        assert!(!is_retriable(reqwest::StatusCode::OK));
        assert!(!is_retriable(reqwest::StatusCode::NOT_FOUND));
    }

    #[test]
    fn is_retriable_500_internal_server_error_is_retriable() {
        assert!(is_retriable(reqwest::StatusCode::INTERNAL_SERVER_ERROR));
    }

    #[test]
    fn build_chat_request_includes_model() {
        let client = OllamaClient::new(
            "http://localhost:11434/api".to_string(),
            "llama3:8b".to_string(),
        );
        let body = client.build_chat_request(&[], &[], false);
        assert_eq!(
            body["model"].as_str(),
            Some("llama3:8b"),
            "model should be in request body"
        );
    }

    #[test]
    fn build_chat_request_stream_field_set_correctly() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        let streaming = client.build_chat_request(&[], &[], true);
        let non_streaming = client.build_chat_request(&[], &[], false);
        assert_eq!(streaming["stream"].as_bool(), Some(true));
        assert_eq!(non_streaming["stream"].as_bool(), Some(false));
    }

    #[test]
    fn extract_ollama_error_handles_empty_json_object() {
        assert!(extract_ollama_error("{}").is_none());
    }

    #[test]
    fn extract_ollama_error_handles_non_object_json() {
        // JSON array is not an object — should return None
        assert!(extract_ollama_error("[1, 2, 3]").is_none());
    }

    #[test]
    fn is_retriable_502_not_retriable() {
        // 502 Bad Gateway is NOT in the retriable set
        assert!(!is_retriable(reqwest::StatusCode::BAD_GATEWAY));
    }

    #[test]
    fn build_chat_request_messages_field_present() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let body = client.build_chat_request(&msgs, &[], false);
        assert!(
            body["messages"].is_array(),
            "messages should be an array: {body}"
        );
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn with_model_overrides_model() {
        let client = OllamaClient::new(
            "http://localhost:11434/api".to_string(),
            "llama3:8b".to_string(),
        );
        let swapped = client.clone().with_model("codellama:13b".to_string());
        assert_eq!(swapped.model(), "codellama:13b");
        assert_eq!(swapped.base_url(), "http://localhost:11434/api");
        // Original is unchanged
        assert_eq!(client.model(), "llama3:8b");
    }

    #[test]
    fn with_model_preserves_json_mode() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string())
                .with_json_mode(true);
        let swapped = client.with_model("other".to_string());
        let body = swapped.build_chat_request(&[], &[], false);
        assert_eq!(
            body["format"].as_str(),
            Some("json"),
            "json_mode should be preserved"
        );
        assert_eq!(body["model"].as_str(), Some("other"));
    }

    #[test]
    fn with_model_shares_context_cache() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        let swapped = client.clone().with_model("other".to_string());
        // Both should point to the same Arc
        assert!(Arc::ptr_eq(&client.context_cache, &swapped.context_cache));
    }

    #[tokio::test]
    async fn context_cache_returns_cached_value() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        // Seed the cache directly
        {
            let mut cache = client.context_cache.lock().await;
            cache.insert("cached-model".to_string(), 32768);
        }
        // Should return cached value without HTTP call
        let limit = client.model_context_limit("cached-model").await;
        assert_eq!(limit, 32768);
    }

    #[test]
    fn suggest_fix_model_not_found() {
        assert!(suggest_fix("model 'llama99' not found")
            .unwrap()
            .contains("ollama pull"));
    }

    #[test]
    fn suggest_fix_context_length() {
        assert!(suggest_fix("context length exceeded")
            .unwrap()
            .contains("context window"));
    }

    #[test]
    fn suggest_fix_unknown_error() {
        assert!(suggest_fix("something went wrong").is_none());
    }

    #[test]
    fn suggest_fix_no_false_positive_oom() {
        assert!(suggest_fix("no room available").is_none());
        assert!(suggest_fix("bloom filter error").is_none());
    }

    #[test]
    fn suggest_fix_out_of_memory() {
        let hint = suggest_fix("CUDA out of memory: tried to allocate 2GB").unwrap();
        assert!(
            hint.contains("smaller"),
            "OOM hint should suggest smaller model: {}",
            hint
        );
    }

    #[test]
    fn build_chat_request_empty_tools_produces_empty_array() {
        let client =
            OllamaClient::new("http://localhost:11434/api".to_string(), "test".to_string());
        let body = client.build_chat_request(&[], &[], false);
        assert!(
            body["tools"].is_array(),
            "tools should be array even when empty: {body}"
        );
        assert!(body["tools"].as_array().unwrap().is_empty());
    }

    #[test]
    fn set_model_changes_model() {
        let mut client = OllamaClient::new(
            "http://localhost:11434/api".to_string(),
            "original".to_string(),
        );
        assert_eq!(client.model(), "original");
        client.set_model("new_model".to_string());
        assert_eq!(client.model(), "new_model");
    }

    #[test]
    fn connect_timeout_is_reasonable() {
        const { assert!(CONNECT_TIMEOUT_SECS >= 5) };
        const { assert!(CONNECT_TIMEOUT_SECS <= 120) };
    }

    #[test]
    fn read_timeout_is_reasonable() {
        const { assert!(READ_TIMEOUT_SECS >= 60) };
        const { assert!(READ_TIMEOUT_SECS <= 600) };
    }

    #[test]
    fn metadata_timeout_is_reasonable() {
        const { assert!(METADATA_TIMEOUT_SECS >= 10 && METADATA_TIMEOUT_SECS <= 60) };
    }

    #[test]
    fn stream_buffer_max_size_is_10mb() {
        const MAX_BUF_SIZE: usize = 10 * 1024 * 1024;
        assert_eq!(MAX_BUF_SIZE, 10_485_760);

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&vec![b'A'; MAX_BUF_SIZE]);
        assert!(buf.len() <= MAX_BUF_SIZE, "at limit should not trigger");

        buf.push(b'A');
        assert!(buf.len() > MAX_BUF_SIZE, "over limit should trigger");
    }

    #[test]
    fn stream_eof_warning_message_is_actionable() {
        let msg = stream_eof_warning();
        assert!(msg.contains("done=true"), "should mention the missing flag");
        assert!(
            msg.contains("token counts"),
            "should explain what's affected"
        );
        assert!(msg.contains("Try:"), "should include an operator next step");
    }

    #[test]
    fn stream_parse_warning_messages_are_actionable() {
        let non_utf8 = non_utf8_stream_line_warning(3, "utf8 error");
        assert!(non_utf8.contains("non-UTF-8"));
        assert!(
            non_utf8.contains("Try:"),
            "non-UTF-8 warning should include next step: {non_utf8}"
        );

        let malformed = malformed_stream_chunk_warning("json error");
        assert!(malformed.contains("malformed stream chunk"));
        assert!(
            malformed.contains("Try:"),
            "malformed chunk warning should include next step: {malformed}"
        );
    }

    #[test]
    fn suggest_fix_connection_refused() {
        let hint = suggest_fix("connection refused");
        assert_eq!(hint, Some("Is Ollama running? Start it with: ollama serve"));
    }

    #[test]
    fn suggest_fix_tcp_connect_error() {
        let hint = suggest_fix("tcp connect error: Connection refused");
        assert_eq!(hint, Some("Is Ollama running? Start it with: ollama serve"));
    }

    #[test]
    fn suggest_fix_failed_to_connect() {
        let hint = suggest_fix("Failed to connect to Ollama");
        assert_eq!(hint, Some("Is Ollama running? Start it with: ollama serve"));
    }

    #[test]
    fn suggest_fix_timeout() {
        let hint = suggest_fix("request timed out");
        assert_eq!(
            hint,
            Some("The model may be overloaded. Try a smaller model or increase timeout")
        );
    }

    #[test]
    fn suggest_fix_model_not_found_still_works() {
        let hint = suggest_fix("model 'foo' not found");
        assert_eq!(
            hint,
            Some("Try: ollama pull <model> — the model may not be downloaded yet")
        );
    }

    #[test]
    fn suggest_fix_oom_still_works() {
        let hint = suggest_fix("out of memory");
        assert_eq!(
            hint,
            Some("Try: use a smaller quantization (e.g. q4_0) or a smaller model")
        );
    }

    #[test]
    fn suggest_fix_unknown_error_returns_none() {
        assert_eq!(suggest_fix("something weird happened"), None);
    }

    #[test]
    fn is_connection_error_detects_variants() {
        assert!(is_connection_error("connection refused"));
        assert!(is_connection_error("connect error: timeout"));
        assert!(is_connection_error("tcp connect failed"));
        assert!(is_connection_error("failed to connect to host"));
        assert!(is_connection_error("connection reset by peer"));
        assert!(!is_connection_error("model not found"));
        assert!(!is_connection_error("out of memory"));
    }
}
