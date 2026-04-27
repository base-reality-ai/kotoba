use crate::compaction::{self, CompactionStage, CompactionThresholds};
use crate::conversation::RetrySettings;
use crate::mcp::client::McpClient;
use crate::ollama::client::OllamaClient;
use crate::ollama::retry::next_backoff_ms;
use crate::ollama::types::{StreamEvent, ToolCall};
use crate::permissions::engine::PermissionEngine;
use crate::session::{storage as session_storage, Session};
use crate::tools::registry::ToolRegistry;
use crate::tui::BackendEvent;
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};

/// Minimum gap between `PerfUpdate` emissions (avoids flooding the event loop).
pub const PERF_DEBOUNCE_MS: u128 = 250;

/// Maximum seconds to wait between consecutive stream chunks before declaring a timeout.
const STREAM_CHUNK_TIMEOUT_SECS: u64 = 300;

use super::events::{build_assistant_msg, select_client};
use super::tool_dispatch::execute_tool_round;

/// Scoped guard that emits `BackendEvent::CompactionCompleted` on drop.
///
/// The compaction pipeline is the sole source of status-bar "[compacting...]"
/// and the banner must clear even when the pipeline panics or an early
/// `return` jumps past the explicit paired send. `try_send` is intentionally
/// used: this is metadata for the UI, not a protocol message, so dropping it
/// under channel pressure (receiver gone, queue full) is preferable to
/// panicking from a `Drop` impl.
struct CompactionGuard {
    tx: mpsc::Sender<BackendEvent>,
}

impl Drop for CompactionGuard {
    fn drop(&mut self) {
        let _ = self.tx.try_send(BackendEvent::CompactionCompleted);
    }
}

/// Loading-model retries need a longer first wait than generic connection
/// drops because the model is actively being pulled into VRAM. Floors the
/// user-configured `base_delay_ms` at 5s without introducing a new config
/// knob. The configured cap (`max_delay_ms`) still applies — if the cap is
/// below the floor, the cap wins so the backoff can never exceed the
/// user's stated patience.
pub(crate) fn effective_loading_base_ms(retry: &RetrySettings) -> u64 {
    retry.base_delay_ms.max(5_000).min(retry.max_delay_ms)
}

/// Mirror of the compaction pipeline's fast-exit predicate. Returns `true`
/// when the pipeline would actually do work, `false` when it would return a
/// no-op `CompactionResult`. Extracted so the turn loop can skip emitting
/// `CompactionStarted` / `CompactionCompleted` on every turn — otherwise the
/// `[compacting...]` status-bar banner flickers on each round even when no
/// real compaction runs.
pub(crate) fn should_run_compaction(current_tokens: usize, micro_compact_threshold: usize) -> bool {
    current_tokens >= micro_compact_threshold
}

/// Drives one user turn: streams from Ollama, executes tool rounds until the model
/// produces a final response with no tool calls.
///
/// Returns `Some((prompt_tokens, completion_tokens))` on normal completion,
/// or `None` if the turn was cancelled (Cancelled event already sent).
#[allow(clippy::too_many_arguments)]
pub async fn run_turn(
    client: &OllamaClient,
    tool_client: Option<&OllamaClient>,
    registry: &ToolRegistry,
    mcp_clients: &HashMap<String, Arc<Mutex<McpClient>>>,
    hooks_config: &crate::tools::hooks::HooksConfig,
    verbose: bool,
    max_turns: usize,
    context_limit: usize,
    staging: bool,
    changeset: &mut Vec<crate::changeset::PendingChange>,
    messages: &mut Vec<Value>,
    session: &mut Session,
    engine: &mut PermissionEngine,
    config_dir: &Path,
    event_tx: &mpsc::Sender<BackendEvent>,
    cancel_rx: &tokio::sync::watch::Receiver<bool>,
    plan_mode: bool,
    retry: &RetrySettings,
) -> Option<(u64, u64)> {
    let mut total_prompt = 0u64;
    let mut total_completion = 0u64;
    let mut warned = false;
    let mut last_round_was_tools_only = false;

    // Compute thresholds from the model's actual context window
    let thresholds = CompactionThresholds::from_context_window(context_limit);
    let mut loop_tracker = crate::conversation::ToolCallTracker::new();
    let tool_defs = registry.definitions();

    for round in 0..max_turns {
        if *cancel_rx.borrow() {
            event_tx.send(BackendEvent::Cancelled).await.ok();
            return None;
        }

        if verbose {
            event_tx
                .send(BackendEvent::StreamToken(format!(
                    "[verbose] Round {} — {} messages, ~{} tokens in context (window: {})\n",
                    round + 1,
                    messages.len(),
                    compaction::estimate_tokens(messages),
                    context_limit,
                )))
                .await
                .ok();
        }

        // Warn once when context approaches compaction thresholds
        let current_tokens = compaction::estimate_tokens(messages);
        if !warned && current_tokens >= thresholds.warning {
            warned = true;
            let pct = current_tokens * 100 / context_limit.max(1);
            event_tx
                .send(BackendEvent::ContextWarning(format!(
                    "Context at {}% ({} / {} tokens) — compaction will trigger soon",
                    pct, current_tokens, context_limit
                )))
                .await
                .ok();
        }

        // Run the 3-stage compaction pipeline. Thread session cwd so the
        // compact-to-wiki tee writes into the session's project wiki even
        // if the process cwd drifted (tool-triggered cd, daemon restart).
        // Thread `session.compact_failures` so the circuit breaker survives
        // across turns (daemon / long TUI sessions) — a per-call local would
        // reset every round and the MAX_COMPACT_FAILURES cap would never trip.
        let session_root = std::path::PathBuf::from(&session.cwd);
        let failures_before = session.compact_failures;
        // Surface the compaction banner in the status bar before blocking on
        // the pipeline. The stage is not yet known (the pipeline decides
        // based on thresholds); we send `None` as a placeholder — the UI
        // renders all stages identically as `[compacting...]` so the
        // placeholder is harmless. The guard drops at the end of the scope
        // block and emits `CompactionCompleted` even if the pipeline panics
        // or a future `?` short-circuits past an explicit send.
        //
        // Gate the emit+call on the same `< micro_compact` fast-exit the
        // pipeline uses internally: skipping here avoids the status-bar
        // flicker on every turn when context is well under threshold. The
        // synthetic `None` result is shape-identical to the pipeline's
        // no-op return, so the downstream `match &result.stage` handles it
        // as a silent no-op.
        let result = if should_run_compaction(current_tokens, thresholds.micro_compact) {
            event_tx
                .send(BackendEvent::CompactionStarted(CompactionStage::None))
                .await
                .ok();
            let _compaction_guard = CompactionGuard {
                tx: event_tx.clone(),
            };
            compaction::compact_pipeline_with_failures(
                messages,
                client,
                &thresholds,
                verbose,
                &mut session.compact_failures,
                Some(&session_root),
            )
            .await
        } else {
            compaction::CompactionResult {
                messages: Vec::new(),
                stage: CompactionStage::None,
                replace_session: false,
            }
        };
        if session.compact_failures != failures_before {
            // Persist bump or reset so the counter state survives a crash
            // between rounds. Best-effort — the turn keeps running either way.
            let _ = session_storage::save(config_dir, session);
        }

        // Report what happened
        match &result.stage {
            CompactionStage::None => {}
            CompactionStage::Microcompact {
                chars_removed,
                messages_affected,
            } => {
                event_tx
                    .send(BackendEvent::ContextPruned {
                        chars_removed: *chars_removed,
                        messages_affected: *messages_affected,
                    })
                    .await
                    .ok();
            }
            CompactionStage::SessionMemory { messages_dropped } => {
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[stage 2: dropped {} old messages to free context]\n",
                        messages_dropped
                    )))
                    .await
                    .ok();
            }
            CompactionStage::FullSummary {
                messages_summarized,
            } => {
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[stage 3: summarized {} messages into compact context]\n",
                        messages_summarized
                    )))
                    .await
                    .ok();
            }
            CompactionStage::Emergency => {
                event_tx
                    .send(BackendEvent::StreamToken(
                        "[emergency compact: force-dropped to last 10 messages]\n".to_string(),
                    ))
                    .await
                    .ok();
            }
        }

        // Update session if compaction changed messages
        let send_messages = if result.replace_session {
            messages.clone_from(&result.messages);
            session.messages.clone_from(messages);
            session_storage::save(config_dir, session).ok();
            result.messages
        } else {
            result.messages
        };

        let active_client = select_client(client, tool_client, round, last_round_was_tools_only);

        let mut stream = match active_client
            .chat_stream_with_tools(&send_messages, &tool_defs)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                let err_str = e.to_string();
                let err_lower = err_str.to_lowercase();

                if err_lower.contains("not found") || err_lower.contains("no such model") {
                    let model_name = active_client.model().to_string();
                    let base_url = active_client.base_url().to_string();

                    event_tx
                        .send(BackendEvent::StreamToken(format!(
                            "\n[Model '{}' not found — pulling from Ollama...]\n",
                            model_name
                        )))
                        .await
                        .ok();

                    let (ptx, mut prx) = tokio::sync::mpsc::channel::<String>(32);
                    let event_tx2 = event_tx.clone();
                    let progress_task = tokio::spawn(async move {
                        while let Some(msg) = prx.recv().await {
                            event_tx2
                                .send(BackendEvent::StreamToken(format!("{}\n", msg)))
                                .await
                                .ok();
                        }
                    });

                    match crate::ollama::pull::pull_with_progress(&model_name, &base_url, &ptx)
                        .await
                    {
                        Ok(()) => {
                            drop(ptx);
                            progress_task.await.ok();
                            event_tx
                                .send(BackendEvent::StreamToken(format!(
                                    "[Model '{}' ready — retrying...]\n\n",
                                    model_name
                                )))
                                .await
                                .ok();
                            match active_client
                                .chat_stream_with_tools(&send_messages, &tool_defs)
                                .await
                            {
                                Ok(s) => s,
                                Err(e2) => {
                                    event_tx
                                        .send(BackendEvent::Error(format!(
                                            "Chat failed after pulling model: {}",
                                            e2
                                        )))
                                        .await
                                        .ok();
                                    return Some((total_prompt, total_completion));
                                }
                            }
                        }
                        Err(pull_err) => {
                            drop(ptx);
                            progress_task.await.ok();
                            event_tx
                                .send(BackendEvent::Error(format!(
                                    "Model '{}' not found and pull failed: {}\n  → Try manually: ollama pull {}",
                                    model_name, pull_err, model_name
                                )))
                                .await
                                .ok();
                            return Some((total_prompt, total_completion));
                        }
                    }
                } else if err_lower.contains("connect") || err_lower.contains("connection") {
                    // Autosave once — messages already includes the user's
                    // turn, so a failed reconnect leaves /resume with the
                    // input intact.
                    session.messages.clone_from(messages);
                    session_storage::save(config_dir, session).ok();

                    let max_attempts = retry.max_retries.max(1);
                    let mut new_stream_opt = None;
                    for attempt in 0..max_attempts {
                        if *cancel_rx.borrow() {
                            event_tx.send(BackendEvent::Cancelled).await.ok();
                            return None;
                        }
                        let delay_ms = next_backoff_ms(
                            attempt as u32,
                            retry.base_delay_ms,
                            retry.max_delay_ms,
                        );
                        event_tx
                            .send(BackendEvent::Notice(format!(
                                "Ollama connection dropped — retrying in {:.1}s (attempt {}/{})",
                                delay_ms as f64 / 1000.0,
                                attempt + 1,
                                max_attempts,
                            )))
                            .await
                            .ok();
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        if *cancel_rx.borrow() {
                            event_tx.send(BackendEvent::Cancelled).await.ok();
                            return None;
                        }
                        if let Ok(new_stream) = active_client
                            .chat_stream_with_tools(&send_messages, &tool_defs)
                            .await
                        {
                            new_stream_opt = Some(new_stream);
                            break;
                        }
                    }
                    if let Some(s) = new_stream_opt {
                        event_tx
                            .send(BackendEvent::Notice("✓ Reconnected to Ollama".to_string()))
                            .await
                            .ok();
                        s
                    } else {
                        event_tx
                            .send(BackendEvent::Error(format!(
                                "Ollama unreachable after {} attempts. Session auto-saved.\n  Check: ollama serve\n  Then: /resume to continue",
                                max_attempts,
                            )))
                            .await
                            .ok();
                        return Some((total_prompt, total_completion));
                    }
                } else if err_lower.contains("503")
                    || err_lower.contains("service unavailable")
                    || err_lower.contains("model is loading")
                    || err_lower.contains("loading model")
                {
                    // Loading-model floors the base delay at 5s so the
                    // model has a realistic window to warm into VRAM.
                    let loading_base_ms = effective_loading_base_ms(retry);
                    let model_name = active_client.model().to_string();

                    session.messages.clone_from(messages);
                    session_storage::save(config_dir, session).ok();

                    let max_attempts = retry.max_retries.max(1);
                    let mut new_stream_opt = None;
                    for attempt in 0..max_attempts {
                        if *cancel_rx.borrow() {
                            event_tx.send(BackendEvent::Cancelled).await.ok();
                            return None;
                        }
                        let delay_ms =
                            next_backoff_ms(attempt as u32, loading_base_ms, retry.max_delay_ms);
                        event_tx
                            .send(BackendEvent::Notice(format!(
                                "Model loading into VRAM — retrying in {:.1}s (attempt {}/{})",
                                delay_ms as f64 / 1000.0,
                                attempt + 1,
                                max_attempts,
                            )))
                            .await
                            .ok();
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                        if *cancel_rx.borrow() {
                            event_tx.send(BackendEvent::Cancelled).await.ok();
                            return None;
                        }
                        if let Ok(new_stream) = active_client
                            .chat_stream_with_tools(&send_messages, &tool_defs)
                            .await
                        {
                            new_stream_opt = Some(new_stream);
                            break;
                        }
                    }
                    if let Some(s) = new_stream_opt {
                        event_tx
                            .send(BackendEvent::Notice("✓ Reconnected to Ollama".to_string()))
                            .await
                            .ok();
                        s
                    } else {
                        event_tx
                            .send(BackendEvent::Error(format!(
                                "Model failed to load after {} attempts. Session auto-saved.\n  Check: ollama list (is {} pulled?)\n  Try: ollama run {} to warm the model, then /resume",
                                max_attempts, model_name, model_name,
                            )))
                            .await
                            .ok();
                        return Some((total_prompt, total_completion));
                    }
                } else if compaction::is_context_overflow(&e) {
                    event_tx
                        .send(BackendEvent::StreamToken(
                            "[context overflow — compacting and retrying]\n".to_string(),
                        ))
                        .await
                        .ok();
                    let compacted = compaction::compact_for_overflow_retry(
                        &send_messages,
                        thresholds.keep_tail,
                    );
                    messages.clone_from(&compacted);
                    session.messages.clone_from(&compacted);
                    let _ = session_storage::save(config_dir, session);
                    match active_client
                        .chat_stream_with_tools(&compacted, &tool_defs)
                        .await
                    {
                        Ok(s) => s,
                        Err(e2) => {
                            event_tx
                                .send(BackendEvent::Error(format!(
                                    "Context overflow retry failed after compaction: {}",
                                    e2
                                )))
                                .await
                                .ok();
                            return Some((total_prompt, total_completion));
                        }
                    }
                } else {
                    event_tx.send(BackendEvent::Error(err_str)).await.ok();
                    return Some((total_prompt, total_completion));
                }
            }
        };

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        // Perf tracking for this stream
        let turn_start = Instant::now();
        let mut first_token_at: Option<Instant> = None;
        let mut token_count: usize = 0;
        let mut last_perf_emit = Instant::now()
            .checked_sub(std::time::Duration::from_millis(
                PERF_DEBOUNCE_MS as u64 + 1,
            ))
            .unwrap_or(Instant::now());

        loop {
            let event = match tokio::time::timeout(
                std::time::Duration::from_secs(STREAM_CHUNK_TIMEOUT_SECS),
                stream.next(),
            )
            .await
            {
                Ok(Some(event)) => event,
                Ok(None) => break,
                Err(_elapsed) => {
                    event_tx
                        .send(BackendEvent::Error(format!(
                            "Ollama stream timed out — no data received for {}s. \
                             The model may be overloaded. Try a smaller model or increase timeout.",
                            STREAM_CHUNK_TIMEOUT_SECS
                        )))
                        .await
                        .ok();
                    if !content.is_empty() {
                        let partial_msg = build_assistant_msg(&content, &[]);
                        messages.push(partial_msg);
                        session.messages.clone_from(messages);
                        session_storage::save(config_dir, session).ok();
                    }
                    return Some((total_prompt, total_completion));
                }
            };
            if *cancel_rx.borrow() {
                if !content.is_empty() {
                    let partial_msg = build_assistant_msg(&content, &[]);
                    messages.push(partial_msg);
                    session.messages.clone_from(messages);
                    session_storage::save(config_dir, session).ok();
                }
                event_tx.send(BackendEvent::Cancelled).await.ok();
                return None;
            }
            match event {
                StreamEvent::Thinking(tok) => {
                    event_tx.send(BackendEvent::StreamThinking(tok)).await.ok();
                }
                StreamEvent::Token(tok) => {
                    // TTFT: record time to first non-empty token
                    if first_token_at.is_none() && !tok.is_empty() {
                        first_token_at = Some(Instant::now());
                    }
                    // Approximate token count (1 per chunk is conservative but avoids
                    // a tokenizer dependency; Ollama's eval_count arrives only at Done)
                    token_count += 1;

                    content.push_str(&tok);
                    event_tx.send(BackendEvent::StreamToken(tok)).await.ok();

                    // Debounced perf update
                    if last_perf_emit.elapsed().as_millis() >= PERF_DEBOUNCE_MS {
                        let elapsed_secs = turn_start.elapsed().as_secs_f32();
                        let tok_per_sec = if elapsed_secs > 0.0 {
                            token_count as f32 / elapsed_secs
                        } else {
                            0.0
                        };
                        let ttft_ms = first_token_at
                            .map_or(0, |t| t.duration_since(turn_start).as_millis() as u64);
                        event_tx
                            .send(BackendEvent::PerfUpdate {
                                tok_per_sec,
                                ttft_ms,
                                total_tokens: token_count,
                            })
                            .await
                            .ok();
                        last_perf_emit = Instant::now();
                    }
                }
                StreamEvent::ToolCalls(calls) => {
                    tool_calls = calls;
                }
                StreamEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                } => {
                    total_prompt += prompt_tokens;
                    total_completion += completion_tokens;
                    // Use Ollama's eval_count for total_tokens when available
                    let final_tokens = if completion_tokens > 0 {
                        completion_tokens as usize
                    } else {
                        token_count
                    };
                    let elapsed_secs = turn_start.elapsed().as_secs_f32();
                    let tok_per_sec = if elapsed_secs > 0.0 {
                        final_tokens as f32 / elapsed_secs
                    } else {
                        0.0
                    };
                    let ttft_ms = first_token_at
                        .map_or(0, |t| t.duration_since(turn_start).as_millis() as u64);
                    // Final perf update (always emitted — not debounced)
                    event_tx
                        .send(BackendEvent::PerfUpdate {
                            tok_per_sec,
                            ttft_ms,
                            total_tokens: final_tokens,
                        })
                        .await
                        .ok();
                    break;
                }
                StreamEvent::Error(e) => {
                    let err_lower = e.to_lowercase();
                    let is_connection = err_lower.contains("connect")
                        || err_lower.contains("broken pipe")
                        || err_lower.contains("reset by peer")
                        || err_lower.contains("eof")
                        || err_lower.contains("incomplete");
                    if is_connection {
                        // Autosave partial content once before the retry loop —
                        // if every attempt fails the user still gets what the
                        // model produced before the drop.
                        if !content.is_empty() {
                            let partial_msg = build_assistant_msg(&content, &[]);
                            messages.push(partial_msg);
                            session.messages.clone_from(messages);
                            session_storage::save(config_dir, session).ok();
                        }

                        let max_attempts = retry.max_retries.max(1);
                        let mut new_stream_opt = None;
                        for attempt in 0..max_attempts {
                            if *cancel_rx.borrow() {
                                event_tx.send(BackendEvent::Cancelled).await.ok();
                                return None;
                            }
                            let delay_ms = next_backoff_ms(
                                attempt as u32,
                                retry.base_delay_ms,
                                retry.max_delay_ms,
                            );
                            event_tx
                                .send(BackendEvent::Notice(format!(
                                    "Ollama connection dropped — retrying in {:.1}s (attempt {}/{})",
                                    delay_ms as f64 / 1000.0,
                                    attempt + 1,
                                    max_attempts,
                                )))
                                .await
                                .ok();
                            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                            if *cancel_rx.borrow() {
                                event_tx.send(BackendEvent::Cancelled).await.ok();
                                return None;
                            }
                            match active_client
                                .chat_stream_with_tools(&send_messages, &tool_defs)
                                .await
                            {
                                Ok(new_stream) => {
                                    new_stream_opt = Some(new_stream);
                                    break;
                                }
                                Err(_e2) => {
                                    // Keep looping until attempts exhausted.
                                }
                            }
                        }
                        if let Some(new_stream) = new_stream_opt {
                            event_tx
                                .send(BackendEvent::Notice("✓ Reconnected to Ollama".to_string()))
                                .await
                                .ok();
                            stream = new_stream;
                            content.clear();
                            tool_calls.clear();
                            continue;
                        }
                        event_tx
                            .send(BackendEvent::Error(format!(
                                "Ollama unreachable after {} attempts. Session auto-saved.\n  Check: ollama serve\n  Then: /resume to continue",
                                max_attempts,
                            )))
                            .await
                            .ok();
                        return Some((total_prompt, total_completion));
                    }
                    if !content.is_empty() {
                        let partial_msg = build_assistant_msg(&content, &[]);
                        messages.push(partial_msg);
                        session.messages.clone_from(messages);
                        session_storage::save(config_dir, session).ok();
                    }
                    event_tx
                        .send(BackendEvent::Error(format!("Ollama error: {}", e)))
                        .await
                        .ok();
                    return Some((total_prompt, total_completion));
                }
            }
        }

        messages.push(build_assistant_msg(&content, &tool_calls));
        session.messages.clone_from(messages);
        session_storage::save(config_dir, session).ok();

        event_tx
            .send(BackendEvent::StreamDone {
                content: content.clone(),
                tool_calls: tool_calls.clone(),
            })
            .await
            .ok();

        last_round_was_tools_only = !tool_calls.is_empty();

        if tool_calls.is_empty() {
            if verbose {
                event_tx
                    .send(BackendEvent::StreamToken(format!(
                        "[verbose] Tokens: {} prompt + {} completion\n",
                        total_prompt, total_completion
                    )))
                    .await
                    .ok();
            }
            return Some((total_prompt, total_completion));
        }

        // ── Loop detection: filter out repeated identical tool calls ─────────
        let mut filtered_calls = Vec::new();
        for tc in &tool_calls {
            if let Some(warning) =
                loop_tracker.record_and_check(&tc.function.name, &tc.function.arguments)
            {
                event_tx
                    .send(BackendEvent::StreamToken(format!("[dm] {}\n", warning)))
                    .await
                    .ok();
                messages.push(crate::conversation::tool_result_msg(
                    &tc.function.name,
                    &warning,
                ));
            } else {
                filtered_calls.push(tc.clone());
            }
        }

        if filtered_calls.is_empty() {
            // All tool calls were loops — continue to next round so model gets the warnings
            continue;
        }

        // ── Tool execution ────────────────────────────────────────────────────
        let active_model_name = active_client.model().to_string();
        let cancelled = execute_tool_round(
            &filtered_calls,
            registry,
            mcp_clients,
            hooks_config,
            engine,
            staging,
            changeset,
            event_tx,
            verbose,
            &active_model_name,
            messages,
            cancel_rx,
            config_dir,
            session,
            plan_mode,
        )
        .await;

        if cancelled {
            return None;
        }

        // Track consecutive all-error rounds
        let recent_tool_msgs: Vec<_> = messages
            .iter()
            .rev()
            .take_while(|m| m["role"].as_str() == Some("tool"))
            .collect();
        let all_errors = !recent_tool_msgs.is_empty()
            && recent_tool_msgs
                .iter()
                .all(|m| m["is_error"].as_bool().unwrap_or(false));
        match loop_tracker.record_error_round(all_errors) {
            crate::conversation::ErrorRoundAction::Break(msg) => {
                event_tx
                    .send(BackendEvent::StreamToken(format!("[dm] {}\n", msg)))
                    .await
                    .ok();
                messages.push(crate::conversation::tool_result_msg("system", msg));
                break;
            }
            crate::conversation::ErrorRoundAction::Warn(msg) => {
                event_tx
                    .send(BackendEvent::StreamToken(format!("[dm] {}\n", msg)))
                    .await
                    .ok();
                messages.push(crate::conversation::tool_result_msg("system", msg));
            }
            crate::conversation::ErrorRoundAction::None => {}
        }
    }

    Some((total_prompt, total_completion))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_chunk_timeout_is_reasonable() {
        const { assert!(STREAM_CHUNK_TIMEOUT_SECS >= 60) };
        const { assert!(STREAM_CHUNK_TIMEOUT_SECS <= 600) };
    }

    #[test]
    fn perf_debounce_is_positive() {
        const { assert!(PERF_DEBOUNCE_MS > 0) };
    }

    #[test]
    fn context_warning_zero_context_limit_no_panic() {
        let current_tokens: usize = 500;
        let context_limit: usize = 0;
        let pct = current_tokens * 100 / context_limit.max(1);
        assert!(pct > 0, "should compute a percentage without panicking");
    }

    #[test]
    fn detect_model_not_found_error() {
        let err = "model \"llama3\" not found, try pulling it first";
        let lower = err.to_lowercase();
        assert!(lower.contains("not found") || lower.contains("no such model"));
    }

    #[test]
    fn detect_not_found_vs_connection() {
        let not_found = "model not found";
        let connection = "connection refused";
        assert!(not_found.to_lowercase().contains("not found"));
        assert!(!not_found.to_lowercase().contains("connect"));
        assert!(connection.to_lowercase().contains("connect"));
        assert!(!connection.to_lowercase().contains("not found"));
    }

    #[test]
    fn not_found_does_not_match_generic_errors() {
        let generic = "invalid model format";
        let lower = generic.to_lowercase();
        assert!(!lower.contains("not found"));
        assert!(!lower.contains("no such model"));
    }

    fn rs(base: u64, max: u64) -> RetrySettings {
        RetrySettings {
            max_retries: 3,
            base_delay_ms: base,
            max_delay_ms: max,
        }
    }

    #[test]
    fn effective_loading_base_ms_floor_applied() {
        // 500ms user base is below the 5s loading-model floor; the floor
        // wins because the model needs real time to warm into VRAM.
        assert_eq!(effective_loading_base_ms(&rs(500, 30_000)), 5_000);
    }

    #[test]
    fn effective_loading_base_ms_above_floor_passthrough() {
        // User already configured a longer-than-5s base — respect it.
        assert_eq!(effective_loading_base_ms(&rs(10_000, 30_000)), 10_000);
    }

    #[test]
    fn effective_loading_base_ms_equal_to_floor() {
        assert_eq!(effective_loading_base_ms(&rs(5_000, 30_000)), 5_000);
    }

    #[test]
    fn effective_loading_base_ms_respects_cap_when_floor_exceeds_max() {
        // Cap beats floor: if the user said "never wait more than 3s",
        // that wins even though the loading-model semantic wants 5s.
        assert_eq!(effective_loading_base_ms(&rs(1_000, 3_000)), 3_000);
    }

    #[test]
    fn retry_exhaustion_message_has_actionable_next_steps() {
        // The TUI exhaustion message is the user's only signal after every
        // retry failed — it must name the cause, confirm the autosave, and
        // point at concrete commands to recover.
        let max_attempts = 3usize;
        let msg = format!(
            "Ollama unreachable after {} attempts. Session auto-saved.\n  Check: ollama serve\n  Then: /resume to continue",
            max_attempts,
        );
        assert!(msg.contains("unreachable"), "names the cause: {msg}");
        assert!(msg.contains("3 attempts"), "shows attempt count: {msg}");
        assert!(msg.contains("auto-saved"), "confirms autosave: {msg}");
        assert!(
            msg.contains("ollama serve"),
            "suggests serve command: {msg}"
        );
        assert!(msg.contains("/resume"), "suggests resume command: {msg}");
    }

    #[test]
    fn retry_notice_message_shape_shows_attempt_progress() {
        // The per-attempt Notice ("retrying in Ns (attempt k/N)") is the
        // user's feedback that the TUI is actively trying, not hung. The
        // format must include the delay and the k/N progress tuple.
        let delay_ms: u64 = 1_000;
        let msg = format!(
            "Ollama connection dropped — retrying in {:.1}s (attempt {}/{})",
            delay_ms as f64 / 1000.0,
            1,
            3,
        );
        assert!(msg.contains("1.0s"), "shows delay in seconds: {msg}");
        assert!(msg.contains("attempt 1/3"), "shows k/N progress: {msg}");
        assert!(msg.contains("dropped"), "names the condition: {msg}");
    }

    #[test]
    fn should_run_compaction_below_threshold_skips() {
        // 500 < 1000 → pipeline would fast-exit, gate returns false so the
        // TUI skips the emit+guard pair and no banner is shown.
        assert!(!should_run_compaction(500, 1000));
    }

    #[test]
    fn should_run_compaction_at_threshold_enters_pipeline() {
        // 1000 >= 1000 → pipeline enters stage 1 (microcompact), gate
        // returns true so the `[compacting...]` banner is surfaced.
        assert!(should_run_compaction(1000, 1000));
    }

    #[test]
    fn should_run_compaction_empty_messages_skips() {
        // An empty-ish conversation (0 tokens) must never flicker the
        // banner: 0 < any positive threshold → gate returns false.
        assert!(!should_run_compaction(0, 1000));
        assert!(!should_run_compaction(0, 1));
    }

    #[test]
    fn should_run_compaction_gate_matches_pipeline_fast_exit() {
        // Exhaustively pin the `<` vs `>=` boundary against the pipeline's
        // own fast-exit predicate (compaction.rs:796, `current_tokens <
        // thresholds.micro_compact`). `gate_enters` must equal
        // `!pipeline_fast_exits` for every sample — any drift would cause
        // the UI to either flicker on skipped turns or suppress the banner
        // on real compaction.
        let threshold: usize = 1000;
        for current_tokens in [0usize, 500, 999, 1000, 1001, 5000] {
            let gate_enters = should_run_compaction(current_tokens, threshold);
            let pipeline_fast_exits = current_tokens < threshold;
            assert_eq!(
                gate_enters, !pipeline_fast_exits,
                "mismatch at current_tokens={current_tokens}"
            );
        }
    }

    #[tokio::test]
    async fn pull_progress_channel_receives_messages() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);
        tx.send("Pulling model...".to_string()).await.unwrap();
        tx.send("Done.".to_string()).await.unwrap();
        drop(tx);
        let mut msgs = Vec::new();
        while let Some(m) = rx.recv().await {
            msgs.push(m);
        }
        assert_eq!(msgs.len(), 2);
        assert!(msgs[0].contains("Pulling"));
    }
}
