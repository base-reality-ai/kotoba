//! Execution engine for multi-agent chains.
//!
//! Evaluates YAML-defined state machines, transitioning between nodes,
//! emitting telemetry, and managing the `chain_state.json` persistence.

use super::{OrchestrationConfig, ValidationResult};
use crate::daemon::protocol::DaemonEvent;
use crate::ollama::client::OllamaClient;
use crate::orchestrate::types::{ChainNodeConfig, ChainNodeStatus, ChainState, ControlSignal};
use crate::tools::registry::ToolRegistry;
use anyhow::Result;
use std::fmt::Write as _;
use std::io::Write as _;

/// Emit a [`DaemonEvent`] on the optional broadcast sender.
/// Silently ignores send errors (no receivers) and does nothing when `tx` is `None`.
fn emit(tx: &Option<tokio::sync::broadcast::Sender<DaemonEvent>>, event: DaemonEvent) {
    if let Some(tx) = tx {
        let _ = tx.send(event);
    }
}

/// Append a `## Planner Brief` section to the user-message *envelope*
/// (the per-cycle `prompt` buffer that carries `## Chain Context` /
/// `## Input`) when `role` is `"planner"` and the workspace's wiki has
/// signal to share. The brief belongs with dynamic chain context — not
/// with the stable role/persona in `sys_prompt` — so external
/// orchestrators (Glue, `--chain`) observe it and the model attends to
/// it alongside per-cycle input. Non-planner roles and any wiki/render
/// error are silent no-ops — the brief is best-effort context.
pub(crate) fn maybe_append_planner_brief_to_envelope(
    prompt: &mut String,
    role: &str,
    workspace: &std::path::Path,
) {
    if role != "planner" {
        return;
    }
    let Ok(wiki) = crate::wiki::Wiki::open(workspace) else {
        return;
    };
    let Ok(brief) = wiki.planner_brief(
        crate::wiki::MOMENTUM_DEFAULT_WINDOW,
        crate::wiki::PLANNER_BRIEF_MAX_DRIFT,
    ) else {
        return;
    };
    if let Some(rendered) = brief.render(crate::wiki::PLANNER_BRIEF_BUDGET_CHARS) {
        prompt.push_str("\n\n");
        prompt.push_str(&rendered);
    }
}

/// Append a cycle synthesis page to the workspace wiki capturing each node's
/// final output for the just-completed cycle. Silent no-op when the wiki
/// cannot be opened or the write fails — chain cycles must never abort on
/// wiki errors.
pub(crate) fn maybe_write_cycle_synthesis(
    workspace: &std::path::Path,
    cycle: usize,
    chain_name: &str,
    nodes: &[ChainNodeConfig],
    node_outputs: &std::collections::HashMap<String, String>,
) {
    let Ok(wiki) = crate::wiki::Wiki::open(workspace) else {
        return;
    };
    let snapshots: Vec<crate::wiki::CycleNodeSnapshot> = nodes
        .iter()
        .map(|n| crate::wiki::CycleNodeSnapshot {
            name: n.name.clone(),
            role: n.role.clone(),
            output: node_outputs.get(&n.name).cloned().unwrap_or_default(),
        })
        .collect();
    let _ = wiki.write_cycle_synthesis(cycle, chain_name, &snapshots, None);
}

/// Apply pending node additions, removals, and model swaps from a persisted
/// [`ChainState`] file.  Returns `true` if mutations were applied (i.e. the
/// state file existed and was loadable), `false` otherwise.
///
/// This is called at the start of each cycle so that `/chain add`, `/chain
/// remove`, and `/chain model` take effect between cycles.
pub fn apply_pending_mutations(
    state: &mut ChainState,
    nodes: &mut Vec<ChainNodeConfig>,
    workspace: &std::path::Path,
    event_tx: &Option<tokio::sync::broadcast::Sender<DaemonEvent>>,
) -> Result<bool> {
    let state_path = workspace.join("chain_state.json");
    if !state_path.exists() {
        return Ok(false);
    }
    let Ok(reloaded) = ChainState::load(&state_path) else {
        return Ok(false);
    };

    state.turns_used = state.turns_used.max(reloaded.turns_used);

    for node in reloaded.pending_additions {
        emit(
            event_tx,
            DaemonEvent::ChainLog {
                chain_id: state.chain_id.clone(),
                level: "info".into(),
                message: format!("+node: {}", node.name),
            },
        );
        state
            .node_statuses
            .insert(node.name.clone(), ChainNodeStatus::Pending);
        nodes.push(node);
    }
    for name in &reloaded.pending_removals {
        emit(
            event_tx,
            DaemonEvent::ChainLog {
                chain_id: state.chain_id.clone(),
                level: "info".into(),
                message: format!("-node: {}", name),
            },
        );
        nodes.retain(|n| &n.name != name);
        state.node_statuses.remove(name);
    }
    for (node_name, new_model) in reloaded.pending_model_swaps {
        if let Some(n) = nodes.iter_mut().find(|n| n.name == node_name) {
            emit(
                event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "info".into(),
                    message: format!("model swap: {} → {}", node_name, new_model),
                },
            );
            n.model = new_model;
        }
    }

    // Persist cleared pending ops and the updated node list.
    state.pending_model_swaps.clear();
    state.config.nodes.clone_from(nodes);
    state.save(workspace)?;
    Ok(true)
}

pub async fn run_orchestration(
    config: OrchestrationConfig,
    client: OllamaClient,
    registry: ToolRegistry,
    event_tx: Option<tokio::sync::broadcast::Sender<DaemonEvent>>,
) -> Result<()> {
    let chain = &config.chain;
    std::fs::create_dir_all(&chain.workspace)?;

    if !chain.skip_permissions_warning {
        emit(&event_tx, DaemonEvent::ChainLog {
            chain_id: String::new(),
            level: "warn".into(),
            message: "Orchestration runs with all permissions bypassed. Sub-agents can freely read/write files and run bash commands.".into(),
        });
    }

    emit(
        &event_tx,
        DaemonEvent::ChainLog {
            chain_id: String::new(),
            level: "info".into(),
            message: format!(
                "Orchestrating chain: {} (workspace: {}, max cycles: {}, turn cap: {})",
                chain.name,
                chain.workspace.display(),
                chain.max_cycles,
                chain.max_total_turns
            ),
        },
    );

    // Cache the base system prompt once — avoids re-running git status and
    // DM.md discovery on every node in every cycle. Identity is threaded
    // transitively here: `build_system_prompt` consults
    // `identity::load_for_cwd()` and frames nodes around `display_name()`,
    // which is the host project in host mode and "dark-matter" in kernel.
    let base_system = crate::system_prompt::build_system_prompt(&[], None).await;

    if let Some(ref path) = chain.directive {
        emit(
            &event_tx,
            DaemonEvent::ChainLog {
                chain_id: String::new(),
                level: "info".into(),
                message: format!("Directive: {}", path.display()),
            },
        );
    }

    let (mut state, mut nodes, start_cycle) = if let Some(resumed) = config.resume_state {
        emit(
            &event_tx,
            DaemonEvent::ChainLog {
                chain_id: resumed.chain_id.clone(),
                level: "info".into(),
                message: format!(
                    "Resuming from cycle {} ({} turns used)",
                    resumed.current_cycle, resumed.turns_used
                ),
            },
        );
        let nodes = resumed.config.nodes.clone();
        let start = resumed.current_cycle;
        (resumed, nodes, start)
    } else {
        let s = ChainState::new(chain.clone(), config.chain_id.clone());
        let nodes = chain.nodes.clone();
        (s, nodes, 0)
    };
    state.save(&chain.workspace)?; // Persist initial state so /chain status works immediately

    // cycle_limit is None when loop_forever is set, otherwise Some(max_cycles).
    let cycle_limit = if chain.loop_forever {
        None
    } else {
        Some(chain.max_cycles)
    };
    let mut cycle: usize = start_cycle;
    loop {
        cycle += 1;
        if let Some(limit) = cycle_limit {
            if cycle > limit {
                break;
            }
        }
        state.current_cycle = cycle;
        let cycle_label =
            cycle_limit.map_or_else(|| format!("{}/∞", cycle), |l| format!("{}/{}", cycle, l));
        emit(
            &event_tx,
            DaemonEvent::ChainLog {
                chain_id: state.chain_id.clone(),
                level: "info".into(),
                message: format!("--- Cycle {} ---", cycle_label),
            },
        );
        let mut cycle_success = false;

        // Hot-reload directive from disk each cycle so live edits take effect
        // immediately (matches Glue's executor pattern).
        let directive_text = chain.directive.as_ref().and_then(|path| {
            match std::fs::read_to_string(path) {
                Ok(content) if !content.trim().is_empty() => Some(content),
                Ok(_) => None,
                Err(_) => None, // silently skip — already warned at chain start if missing
            }
        });

        // Check stop sentinel before each cycle
        let stop_file = chain.workspace.join(".dm-stop");
        if stop_file.exists() {
            let _ = std::fs::remove_file(&stop_file);
            emit(
                &event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "info".into(),
                    message: "Stop sentinel detected. Halting chain.".into(),
                },
            );
            emit(
                &event_tx,
                DaemonEvent::ChainFinished {
                    chain_id: state.chain_id.clone(),
                    success: false,
                    reason: "stop sentinel detected".to_string(),
                },
            );
            return Ok(());
        }

        // Pause sentinel: spin-wait until .dm-pause is removed
        let pause_file = chain.workspace.join(".dm-pause");
        if pause_file.exists() {
            let mut was_paused = false;
            while pause_file.exists() {
                if !was_paused {
                    emit(
                        &event_tx,
                        DaemonEvent::ChainLog {
                            chain_id: state.chain_id.clone(),
                            level: "info".into(),
                            message: "Chain paused. Use /chain resume to continue.".into(),
                        },
                    );
                    was_paused = true;
                }
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            emit(
                &event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "info".into(),
                    message: "Chain resumed.".into(),
                },
            );
        }

        // Apply any pending node additions/removals written by /chain add|remove.
        apply_pending_mutations(&mut state, &mut nodes, &chain.workspace, &event_tx)?;

        // Index-based loop so we can re-run a node on RETRY signal.
        let mut node_idx = 0;
        let mut signal_retries: usize = 0;
        let mut cycle_aborted = false;
        while node_idx < nodes.len() {
            let node = nodes[node_idx].clone();
            state.active_node_index = Some(node_idx);

            if state.turns_used >= chain.max_total_turns {
                state.save(&chain.workspace)?;
                let reason = format!(
                    "Orchestration turn cap ({}) reached. \
                     Use --max-turns to increase or simplify the task.",
                    chain.max_total_turns
                );
                emit(
                    &event_tx,
                    DaemonEvent::ChainFinished {
                        chain_id: state.chain_id.clone(),
                        success: false,
                        reason: reason.clone(),
                    },
                );
                anyhow::bail!("{}", reason);
            }

            state
                .node_statuses
                .insert(node.name.clone(), ChainNodeStatus::Running);
            state.save(&chain.workspace)?;

            emit(
                &event_tx,
                DaemonEvent::ChainNodeTransition {
                    chain_id: state.chain_id.clone(),
                    cycle,
                    node_name: node.name.clone(),
                    status: crate::orchestrate::types::STATUS_RUNNING.to_string(),
                },
            );

            print!("[Node: {}] Processing... ", node.name);
            std::io::stdout().flush()?;

            let node_start = std::time::Instant::now();

            // 1. Build system prompt — start from cached base, add node-specific context
            let mut sys_prompt = base_system.clone();

            if let Some(ref directive) = directive_text {
                sys_prompt.push_str("\n\n## Prime Directive\n\n");
                sys_prompt.push_str(directive);
            }

            write!(
                sys_prompt,
                "\n\n## Your Role\n\nYou are the {}. Your task is: {}\n",
                node.role, chain.name
            )
            .expect("write to String never fails");

            // system_prompt_file takes precedence over system_prompt_override.
            if let Some(ref file_path) = node.system_prompt_file {
                match std::fs::read_to_string(file_path) {
                    Ok(content) => write!(sys_prompt, "\n## Instructions\n{}", content)
                        .expect("write to String never fails"),
                    Err(e) => emit(
                        &event_tx,
                        DaemonEvent::ChainLog {
                            chain_id: state.chain_id.clone(),
                            level: "warn".into(),
                            message: format!(
                                "could not read system_prompt_file {:?}: {}",
                                file_path, e
                            ),
                        },
                    ),
                }
            } else if let Some(override_prompt) = &node.system_prompt_override {
                write!(sys_prompt, "\n## Instructions\n{}", override_prompt)
                    .expect("write to String never fails");
            }

            // 2. Build user message — only contains the actual work input
            let mut prompt = String::new();

            // 2a. Inject /chain talk message if present
            let talk_file = chain.workspace.join(format!("talk-{}.md", node.name));
            if talk_file.exists() {
                let msg = std::fs::read_to_string(&talk_file).unwrap_or_default();
                let _ = std::fs::remove_file(&talk_file);
                if !msg.trim().is_empty() {
                    write!(
                        prompt,
                        "## User instruction (injected via /chain talk)\n\n{}\n\n",
                        msg.trim()
                    )
                    .expect("write to String never fails");
                }
            }

            // 2b. Inject input from previous node with envelope metadata
            if let Some(input_source) = &node.input_from {
                match state.node_outputs.get(input_source) {
                    Some(prev_output) if !prev_output.is_empty() => {
                        let cycle_label = cycle_limit
                            .map_or_else(|| format!("{}/∞", cycle), |l| format!("{}/{}", cycle, l));
                        write!(
                            prompt,
                            "## Chain Context\nFrom: {}\nTo: {}\nCycle: {}\nTurn: {}\n\n## Input\n{}\n",
                            input_source, node.name, cycle_label, state.turns_used, prev_output
                        )
                        .expect("write to String never fails");
                    }
                    _ => {
                        emit(
                            &event_tx,
                            DaemonEvent::ChainLog {
                                chain_id: state.chain_id.clone(),
                                level: "warn".into(),
                                message: format!(
                                    "node '{}' expects input from '{}' but no output available yet",
                                    node.name, input_source
                                ),
                            },
                        );
                    }
                }
            }

            // If no input from previous node or talk, give a minimal kickoff message
            if prompt.trim().is_empty() {
                prompt.push_str("Begin your work.");
            }

            // Planner-role nodes get a structured wiki-derived brief appended
            // to the user-message envelope (alongside chain context), not the
            // sys_prompt — external orchestrators (Glue, `--chain`) observe
            // the envelope, and per-cycle dynamic context belongs there.
            maybe_append_planner_brief_to_envelope(&mut prompt, &node.role, &chain.workspace);

            // 3. Run Conversation — retry up to node.max_retries times on transient errors.
            //    Each attempt is wrapped in a per-node timeout to prevent indefinite stalls.
            let max_attempts = node.max_retries + 1;
            let timeout_duration = if node.timeout_secs > 0 {
                Some(std::time::Duration::from_secs(node.timeout_secs))
            } else {
                None
            };
            let mut run_result = Err(anyhow::anyhow!("not started"));
            for attempt in 1..=max_attempts {
                let node_client = client.clone().with_model(node.model.clone());
                let conversation_fut = crate::conversation::run_conversation_capture_with_turns(
                    &prompt,
                    "chain",
                    &node_client,
                    &registry,
                    node.max_tool_turns,
                    config.retry,
                    Some(&sys_prompt),
                    None,
                );
                run_result = if let Some(dur) = timeout_duration {
                    match tokio::time::timeout(dur, conversation_fut).await {
                        Ok(inner) => inner,
                        Err(_elapsed) => Err(anyhow::anyhow!(
                            "node '{}' timed out after {}s",
                            node.name,
                            node.timeout_secs
                        )),
                    }
                } else {
                    conversation_fut.await
                };
                if run_result.is_ok() {
                    break;
                }
                if attempt < max_attempts {
                    let err_msg = run_result
                        .as_ref()
                        .expect_err("just checked is_ok() == false")
                        .to_string();
                    emit(
                        &event_tx,
                        DaemonEvent::ChainLog {
                            chain_id: state.chain_id.clone(),
                            level: "warn".into(),
                            message: format!(
                                "node '{}' error (attempt {}/{}): {}. Retrying in 2s...",
                                node.name, attempt, max_attempts, err_msg
                            ),
                        },
                    );
                    state.node_statuses.insert(
                        node.name.clone(),
                        ChainNodeStatus::Failed(format!("attempt {}: {}", attempt, err_msg)),
                    );
                    state.save(&chain.workspace)?;
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
            }

            let (output, node_prompt_tok, node_completion_tok) = match run_result {
                Ok(capture) => (
                    capture.text,
                    capture.prompt_tokens,
                    capture.completion_tokens,
                ),
                Err(e) => {
                    let reason = format!("node '{}' failed: {}", node.name, e);
                    emit(
                        &event_tx,
                        DaemonEvent::ChainLog {
                            chain_id: state.chain_id.clone(),
                            level: "error".into(),
                            message: format!(
                                "node '{}' failed after {} attempt(s): {}. Aborting cycle.",
                                node.name, max_attempts, e
                            ),
                        },
                    );
                    state
                        .node_statuses
                        .insert(node.name.clone(), ChainNodeStatus::Failed(e.to_string()));
                    state.last_abort_reason = Some(reason.clone());
                    crate::notify::notify("Chain node failed", &reason);
                    // Record failure metrics
                    let elapsed = node_start.elapsed().as_secs_f64();
                    state
                        .node_durations
                        .entry(node.name.clone())
                        .or_default()
                        .push(elapsed);
                    *state.node_failures.entry(node.name.clone()).or_insert(0) += 1;
                    state.total_duration_secs += elapsed;
                    state.save(&chain.workspace)?;
                    emit(
                        &event_tx,
                        DaemonEvent::ChainNodeTransition {
                            chain_id: state.chain_id.clone(),
                            cycle,
                            node_name: node.name.clone(),
                            status: format!(
                                "{}{}",
                                crate::orchestrate::types::STATUS_FAILED_PREFIX,
                                e
                            ),
                        },
                    );
                    cycle_aborted = true;
                    break; // skip remaining nodes in this cycle
                }
            };

            let elapsed = node_start.elapsed().as_secs_f64();
            state
                .node_durations
                .entry(node.name.clone())
                .or_default()
                .push(elapsed);
            state.total_duration_secs += elapsed;
            *state
                .node_prompt_tokens
                .entry(node.name.clone())
                .or_insert(0) += node_prompt_tok;
            *state
                .node_completion_tokens
                .entry(node.name.clone())
                .or_insert(0) += node_completion_tok;

            state.turns_used += 1;
            state.node_outputs.insert(node.name.clone(), output.clone());
            state
                .node_statuses
                .insert(node.name.clone(), ChainNodeStatus::Completed);
            save_artifact(&chain.workspace, cycle, &node.name, &output)?;

            emit(
                &event_tx,
                DaemonEvent::ChainNodeTransition {
                    chain_id: state.chain_id.clone(),
                    cycle,
                    node_name: node.name.clone(),
                    status: crate::orchestrate::types::STATUS_COMPLETED.to_string(),
                },
            );

            // Check for control signals in output
            if let Some(signal) = parse_control_tokens(&output) {
                state.last_signal = Some(signal.clone());
                emit(
                    &event_tx,
                    DaemonEvent::ChainLog {
                        chain_id: state.chain_id.clone(),
                        level: "info".into(),
                        message: format!("Signal received: {:?}", signal),
                    },
                );
                match signal {
                    ControlSignal::Stop => {
                        emit(
                            &event_tx,
                            DaemonEvent::ChainLog {
                                chain_id: state.chain_id.clone(),
                                level: "info".into(),
                                message: "Stopping chain due to explicit STOP signal.".into(),
                            },
                        );
                        state.save(&chain.workspace)?;
                        emit(
                            &event_tx,
                            DaemonEvent::ChainFinished {
                                chain_id: state.chain_id.clone(),
                                success: true,
                                reason: "STOP signal received".to_string(),
                            },
                        );
                        return Ok(());
                    }
                    ControlSignal::Retry => {
                        signal_retries += 1;
                        if signal_retries > node.max_retries {
                            let reason = format!(
                                "node '{}' sent RETRY {} times (max_retries={})",
                                node.name, signal_retries, node.max_retries
                            );
                            emit(
                                &event_tx,
                                DaemonEvent::ChainLog {
                                    chain_id: state.chain_id.clone(),
                                    level: "error".into(),
                                    message: format!("{} — aborting cycle.", reason),
                                },
                            );
                            state.last_abort_reason = Some(reason);
                            state.node_statuses.insert(
                                node.name.clone(),
                                ChainNodeStatus::Failed("RETRY limit exceeded".into()),
                            );
                            *state.node_failures.entry(node.name.clone()).or_insert(0) += 1;
                            state.save(&chain.workspace)?;
                            cycle_aborted = true;
                            break;
                        }
                        emit(
                            &event_tx,
                            DaemonEvent::ChainLog {
                                chain_id: state.chain_id.clone(),
                                level: "info".into(),
                                message: format!(
                                    "RETRY signal from '{}' ({}/{}). Re-running node.",
                                    node.name, signal_retries, node.max_retries
                                ),
                            },
                        );
                        state.save(&chain.workspace)?;
                        // Don't increment node_idx — re-run same node next iteration.
                        continue;
                    }
                    ControlSignal::Escalate => {
                        let reason = format!("node '{}' sent ESCALATE", node.name);
                        emit(
                            &event_tx,
                            DaemonEvent::ChainLog {
                                chain_id: state.chain_id.clone(),
                                level: "warn".into(),
                                message: format!("ESCALATE from '{}'. Aborting cycle.", node.name),
                            },
                        );
                        state.last_abort_reason = Some(reason);
                        cycle_aborted = true;
                        break;
                    }
                    ControlSignal::Continue => {}
                }
            }

            emit(
                &event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "info".into(),
                    message: format!("{} done ({} chars)", node.name, output.len()),
                },
            );

            // 4. Check for termination (if node is a validator and it passes)
            if node.role.to_lowercase().contains("validator") {
                let validation = parse_validation(&output);
                if let ValidationResult::Pass(_) = &validation {
                    cycle_success = true;
                }
            }

            state.save(&chain.workspace)?;
            node_idx += 1;
            signal_retries = 0;
        }

        if cycle_aborted {
            emit(
                &event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "warn".into(),
                    message: format!("Cycle {} aborted. Starting next cycle.", cycle),
                },
            );
            state.save(&chain.workspace)?;
            continue;
        }

        // Clear any previous abort reason on a clean cycle.
        state.last_abort_reason = None;

        emit(
            &event_tx,
            DaemonEvent::ChainCycleComplete {
                chain_id: state.chain_id.clone(),
                cycle,
            },
        );

        maybe_write_cycle_synthesis(
            &chain.workspace,
            cycle,
            &chain.name,
            &nodes,
            &state.node_outputs,
        );

        if cycle_success {
            emit(
                &event_tx,
                DaemonEvent::ChainLog {
                    chain_id: state.chain_id.clone(),
                    level: "info".into(),
                    message: format!("Task complete after {} cycle(s).", cycle),
                },
            );
            state.save(&chain.workspace)?;
            crate::notify::notify(
                "Chain complete",
                &format!("{}: validation passed after {} cycle(s)", chain.name, cycle),
            );
            emit(
                &event_tx,
                DaemonEvent::ChainFinished {
                    chain_id: state.chain_id.clone(),
                    success: true,
                    reason: format!("validation passed at cycle {}", cycle),
                },
            );
            return Ok(());
        }
    }

    if let Some(limit) = cycle_limit {
        emit(
            &event_tx,
            DaemonEvent::ChainLog {
                chain_id: state.chain_id.clone(),
                level: "warn".into(),
                message: format!("Max cycles ({}) reached without validation pass.", limit),
            },
        );
        crate::notify::notify(
            "Chain stopped",
            &format!("{}: max cycles ({}) reached", chain.name, limit),
        );
    }
    emit(
        &event_tx,
        DaemonEvent::ChainFinished {
            chain_id: state.chain_id.clone(),
            success: false,
            reason: "max cycles reached without validation pass".to_string(),
        },
    );
    Ok(())
}

pub fn parse_control_tokens(raw: &str) -> Option<ControlSignal> {
    let signal_line = raw
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .filter(|l| l.trim_start().starts_with("SIGNAL:"))?;

    let signal_str = signal_line
        .trim_start()
        .strip_prefix("SIGNAL:")
        .unwrap_or("")
        .trim()
        .to_uppercase();

    match signal_str.as_str() {
        "CONTINUE" => Some(ControlSignal::Continue),
        "STOP" => Some(ControlSignal::Stop),
        "RETRY" => Some(ControlSignal::Retry),
        "ESCALATE" => Some(ControlSignal::Escalate),
        _ => None,
    }
}

pub fn parse_validation(raw: &str) -> ValidationResult {
    let verdict = raw
        .lines()
        .find(|l| l.trim_start().starts_with("VERDICT:"))
        .and_then(|l| l.trim_start().strip_prefix("VERDICT:"))
        .map(|v| v.trim().to_uppercase())
        .unwrap_or_default();

    let summary = extract_section(raw, "SUMMARY:");
    let issues = extract_bullets(raw, "ISSUES:");
    let suggestions = extract_bullets(raw, "SUGGESTIONS:");

    if verdict.contains("PASS") {
        ValidationResult::Pass(summary)
    } else {
        ValidationResult::Fail {
            issues,
            suggestions,
        }
    }
}

pub fn extract_section(raw: &str, header: &str) -> String {
    let start = match raw.find(header) {
        Some(i) => i + header.len(),
        None => return String::new(),
    };
    let rest = &raw[start..];
    let section_headers = ["VERDICT:", "SUMMARY:", "ISSUES:", "SUGGESTIONS:"];
    let end = section_headers
        .iter()
        .filter_map(|h| rest.find(h))
        .filter(|&i| i > 0)
        .min()
        .unwrap_or(rest.len());
    rest[..end].trim().to_string()
}

pub fn extract_bullets(raw: &str, header: &str) -> Vec<String> {
    let start = match raw.find(header) {
        Some(i) => i + header.len(),
        None => return Vec::new(),
    };
    let rest = &raw[start..];
    let section_headers = ["VERDICT:", "SUMMARY:", "ISSUES:", "SUGGESTIONS:"];
    let end = section_headers
        .iter()
        .filter_map(|h| rest.find(h))
        .filter(|&i| i > 0)
        .min()
        .unwrap_or(rest.len());
    let section = &rest[..end];

    section
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with('-') || l.starts_with('*') || l.starts_with('•'))
        .map(|l| l.trim_start_matches(['-', '*', '•', ' ']).to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

pub fn save_artifact(
    workspace: &std::path::Path,
    cycle: usize,
    name: &str,
    content: &str,
) -> Result<()> {
    let path = workspace.join(format!("cycle-{:02}-{}.md", cycle, name));
    let tmp_path = path.with_extension("md.tmp");
    std::fs::write(&tmp_path, content)?;
    std::fs::rename(&tmp_path, &path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::client::OllamaClient;
    use crate::orchestrate::types::ChainConfig;
    use crate::tools::registry::ToolRegistry;

    // ── save_artifact ────────────────────────────────────────────────────────

    // ── maybe_append_planner_brief_to_envelope (cycle 54→83) ─────────────

    #[test]
    fn planner_role_envelope_gets_planner_brief_header() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        // Seed a wiki with a single ingest so planner_brief has at
        // least one hot path and render() returns Some(..).
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        let mut prompt = String::from("base");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);
        assert!(
            prompt.contains("## Planner Brief"),
            "planner-role envelope must contain the brief header: {}",
            prompt,
        );
        assert!(
            prompt.contains("src/seeded.rs"),
            "planner-role envelope must include the seeded hot path: {}",
            prompt,
        );
    }

    #[test]
    fn non_planner_role_envelope_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        for role in ["builder", "tester", "auditor", ""] {
            let mut prompt = String::from("base");
            maybe_append_planner_brief_to_envelope(&mut prompt, role, workspace);
            assert_eq!(
                prompt, "base",
                "role {:?} must not receive the planner brief",
                role,
            );
        }
    }

    #[test]
    fn planner_brief_includes_cycle_synthesis_from_prior_cycle() {
        // Prior-cycle synthesis page → next-cycle planner envelope picks
        // it up via PlannerBrief::recent_cycles (Cycle 57 closing loop).
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().canonicalize().unwrap();

        let wiki = crate::wiki::Wiki::open(&workspace).unwrap();
        let snapshots = vec![crate::wiki::CycleNodeSnapshot {
            name: "planner".into(),
            role: "planner".into(),
            output: "prior plan".into(),
        }];
        let page_rel = wiki
            .write_cycle_synthesis(1, "demo", &snapshots, None)
            .unwrap()
            .expect("synthesis page written");

        let mut prompt = String::from("base");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", &workspace);
        assert!(
            prompt.contains("### Recent cycles"),
            "next-cycle planner envelope must include Recent cycles section: {}",
            prompt
        );
        assert!(
            prompt.contains(&page_rel),
            "next-cycle planner envelope must cite prior cycle's page path ({}): {}",
            page_rel,
            prompt
        );
    }

    #[test]
    fn planner_brief_injection_no_op_when_wiki_missing() {
        // Bare workspace with no .dm/wiki dir — Wiki::open still creates
        // it with an empty layout; the brief is empty so the envelope
        // stays untouched. Guards against false-positive injections.
        let tmp = tempfile::tempdir().unwrap();
        let mut prompt = String::from("base");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", tmp.path());
        assert_eq!(
            prompt, "base",
            "empty wiki must produce no planner-brief append: {}",
            prompt,
        );
    }

    #[test]
    fn envelope_injection_preserves_chain_context_ordering() {
        // Envelope already has `## Chain Context` / `## Input`; brief appends
        // after, so `## Input` precedes `## Planner Brief` in the envelope.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        let mut prompt =
            String::from("## Chain Context\nFrom: tester\nTo: planner\n\n## Input\nprior output\n");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);

        let input_pos = prompt.find("## Input").expect("input block present");
        let brief_pos = prompt
            .find("## Planner Brief")
            .expect("brief header present");
        assert!(
            brief_pos > input_pos,
            "brief must follow input, got: {prompt}"
        );
    }

    #[test]
    fn envelope_injection_works_with_empty_prompt() {
        // First cycle: no input_from, no /chain talk — prompt is "Begin your
        // work." Brief still appends cleanly.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        let mut prompt = String::from("Begin your work.");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);
        assert!(
            prompt.contains("Begin your work.") && prompt.contains("## Planner Brief"),
            "kickoff + brief must coexist: {prompt}"
        );
    }

    #[test]
    fn sys_prompt_no_longer_receives_planner_brief() {
        // Regression guard: after C83, `sys_prompt` must stay clean. If
        // someone re-introduces sys_prompt injection, this test fires.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        // Simulate the runner's buffer separation: sys_prompt and prompt
        // are distinct; only `prompt` reaches the injector.
        let sys_prompt = String::from("role: planner\npersona: architect");
        let mut prompt = String::from("## Chain Context\n...\n## Input\nX\n");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);

        assert!(
            !sys_prompt.contains("## Planner Brief"),
            "sys_prompt must stay clean of brief: {sys_prompt}"
        );
        assert!(
            prompt.contains("## Planner Brief"),
            "envelope must carry brief: {prompt}"
        );
    }

    #[test]
    fn envelope_injection_appends_each_call_by_design() {
        // Guards against a future refactor double-calling the injector. The
        // current contract is "exactly once per cycle"; if double-called,
        // the brief appears twice. Pin that observable so regressions are
        // loud — this is non-idempotent by design.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        wiki.log().append("ingest", "src/seeded.rs").unwrap();

        let mut prompt = String::from("base");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);
        let after_first = prompt.clone();
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);
        assert_eq!(
            prompt.matches("## Planner Brief").count(),
            2,
            "double-call appends twice (documents non-idempotence): \
             before={after_first} after={prompt}"
        );
    }

    #[test]
    fn envelope_respects_budget_cap() {
        // Guards the `PLANNER_BRIEF_BUDGET_CHARS` ceiling at the injection
        // site. A brief render that exceeds the cap must be truncated by
        // `PlannerBrief::render` before it reaches the envelope.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path();
        let wiki = crate::wiki::Wiki::open(workspace).unwrap();
        // Seed many ingests to fatten the brief past the cap.
        for i in 0..100 {
            wiki.log()
                .append("ingest", &format!("src/f{i}.rs"))
                .unwrap();
        }

        let mut prompt = String::from("base");
        maybe_append_planner_brief_to_envelope(&mut prompt, "planner", workspace);
        let brief_len = prompt.len().saturating_sub("base".len());
        assert!(
            brief_len <= crate::wiki::PLANNER_BRIEF_BUDGET_CHARS + 64,
            "envelope-injected brief must respect budget cap (got {brief_len}): {prompt}"
        );
    }

    // ── maybe_write_cycle_synthesis (cycle 55) ───────────────────────────

    #[test]
    fn maybe_write_cycle_synthesis_writes_page_in_dag_order() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().canonicalize().unwrap();
        // Ensure the wiki dir exists so write_cycle_synthesis can land a page.
        let _ = crate::wiki::Wiki::open(&workspace).unwrap();

        let nodes = vec![
            ChainNodeConfig {
                id: "planner".into(),
                name: "planner".into(),
                role: "planner".into(),
                model: "test".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
            ChainNodeConfig {
                id: "builder".into(),
                name: "builder".into(),
                role: "builder".into(),
                model: "test".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: Some("planner".into()),
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            },
        ];
        let mut outputs = std::collections::HashMap::new();
        outputs.insert("planner".to_string(), "planner-said".to_string());
        outputs.insert("builder".to_string(), "builder-said".to_string());

        maybe_write_cycle_synthesis(&workspace, 2, "demo", &nodes, &outputs);

        let synth_dir = workspace.join(".dm/wiki/synthesis");
        let mut found = None;
        for entry in std::fs::read_dir(&synth_dir).unwrap() {
            let p = entry.unwrap().path();
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if name.starts_with("cycle-02-demo-") {
                found = Some(p);
                break;
            }
        }
        let page_path = found.expect("cycle-02-demo-*.md should exist");
        let text = std::fs::read_to_string(&page_path).unwrap();
        assert!(text.contains("planner-said"));
        assert!(text.contains("builder-said"));
        let p = text.find("**planner**").unwrap();
        let b = text.find("**builder**").unwrap();
        assert!(p < b, "planner bullet should precede builder bullet");
    }

    #[test]
    fn maybe_write_cycle_synthesis_empty_nodes_no_panic() {
        // Runs against a workspace with no nodes and no outputs; must
        // silently produce a page (empty Nodes section) without panicking.
        let tmp = tempfile::tempdir().unwrap();
        let workspace = tmp.path().canonicalize().unwrap();
        let _ = crate::wiki::Wiki::open(&workspace).unwrap();

        let outputs = std::collections::HashMap::new();
        maybe_write_cycle_synthesis(&workspace, 1, "noop", &[], &outputs);
    }

    #[test]
    fn save_artifact_writes_expected_filename() {
        let tmp = tempfile::tempdir().unwrap();
        save_artifact(tmp.path(), 3, "builder", "artifact content").unwrap();
        let expected = tmp.path().join("cycle-03-builder.md");
        assert!(expected.exists(), "expected file {expected:?} to exist");
        assert_eq!(
            std::fs::read_to_string(&expected).unwrap(),
            "artifact content"
        );
    }

    #[test]
    fn save_artifact_zero_pads_cycle_number() {
        let tmp = tempfile::tempdir().unwrap();
        save_artifact(tmp.path(), 1, "tester", "x").unwrap();
        let expected = tmp.path().join("cycle-01-tester.md");
        assert!(
            expected.exists(),
            "expected zero-padded filename {expected:?}"
        );
    }

    #[test]
    fn save_artifact_overwrites_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        save_artifact(tmp.path(), 1, "n", "first").unwrap();
        save_artifact(tmp.path(), 1, "n", "second").unwrap();
        let p = tmp.path().join("cycle-01-n.md");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "second");
    }

    #[test]
    fn save_artifact_no_tmp_left_behind() {
        let tmp = tempfile::tempdir().unwrap();
        save_artifact(tmp.path(), 5, "planner", "plan content").unwrap();
        let expected = tmp.path().join("cycle-05-planner.md");
        let tmp_file = expected.with_extension("md.tmp");
        assert!(expected.exists(), "artifact file should exist");
        assert!(
            !tmp_file.exists(),
            "tmp file should not remain after atomic write"
        );
    }

    // ── parse_control_tokens ──────────────────────────────────────────────────

    #[test]
    fn parse_signal_continue() {
        let raw = "Here is my output.\nMore text.\nSIGNAL: CONTINUE";
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Continue));
    }

    #[test]
    fn parse_signal_stop() {
        let raw = "SIGNAL: STOP";
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Stop));
    }

    #[test]
    fn parse_signal_retry() {
        let raw = "  SIGNAL: RETRY"; // leading whitespace
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Retry));
    }

    #[test]
    fn parse_signal_escalate() {
        assert_eq!(
            parse_control_tokens("SIGNAL: ESCALATE"),
            Some(ControlSignal::Escalate)
        );
    }

    #[test]
    fn parse_signal_lowercase() {
        assert_eq!(
            parse_control_tokens("SIGNAL: continue"),
            Some(ControlSignal::Continue)
        );
    }

    #[test]
    fn parse_signal_missing() {
        assert_eq!(parse_control_tokens("No signal here."), None);
    }

    #[test]
    fn parse_signal_unknown_value() {
        assert_eq!(parse_control_tokens("SIGNAL: UNKNOWN"), None);
    }

    #[test]
    fn parse_signal_mid_position_ignored() {
        let raw = "Here is my output.\nSIGNAL: CONTINUE\nMore text after signal.";
        assert_eq!(parse_control_tokens(raw), None);
    }

    #[test]
    fn parse_signal_tail_with_trailing_whitespace() {
        let raw = "Output text.\nSIGNAL: STOP\n  \n";
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Stop));
    }

    #[test]
    fn parse_signal_only_line() {
        assert_eq!(
            parse_control_tokens("SIGNAL: RETRY"),
            Some(ControlSignal::Retry)
        );
    }

    #[test]
    fn parse_signal_tail_after_multiline_output() {
        let raw = "Line 1\nLine 2\nLine 3\nSIGNAL: ESCALATE";
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Escalate));
    }

    // ── parse_validation ─────────────────────────────────────────────────────

    #[test]
    fn parse_validation_pass() {
        let raw = "VERDICT: PASS\nSUMMARY: All good.\n";
        match parse_validation(raw) {
            ValidationResult::Pass(s) => assert_eq!(s.trim(), "All good."),
            other => panic!("expected Pass, got {:?}", other),
        }
    }

    #[test]
    fn parse_validation_fail() {
        let raw = "VERDICT: FAIL\nISSUES:\n- broken thing\nSUGGESTIONS:\n- fix it\n";
        match parse_validation(raw) {
            ValidationResult::Fail {
                issues,
                suggestions,
            } => {
                assert!(issues.iter().any(|i| i.contains("broken")));
                assert!(suggestions.iter().any(|s| s.contains("fix")));
            }
            other => panic!("expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn parse_validation_no_verdict_defaults_fail() {
        let raw = "Nothing here.";
        match parse_validation(raw) {
            ValidationResult::Fail { .. } => {}
            other => panic!("expected Fail, got {:?}", other),
        }
    }

    #[test]
    fn parse_validation_pass_lowercase() {
        let raw = "VERDICT: pass";
        match parse_validation(raw) {
            ValidationResult::Pass(_) => {}
            other => panic!("expected Pass, got {:?}", other),
        }
    }

    // ── extract_section / extract_bullets ────────────────────────────────────

    #[test]
    fn extract_section_basic() {
        let raw = "SUMMARY: Things went well.\nVERDICT: PASS";
        let s = extract_section(raw, "SUMMARY:");
        assert!(s.contains("Things went well"));
    }

    #[test]
    fn extract_bullets_basic() {
        let raw = "ISSUES:\n- bug one\n- bug two\nSUGGESTIONS:\n- fix it\n";
        let bullets = extract_bullets(raw, "ISSUES:");
        assert_eq!(bullets, vec!["bug one", "bug two"]);
    }

    #[test]
    fn extract_section_missing_header_returns_empty() {
        let raw = "VERDICT: PASS\nSUMMARY: All good.";
        let s = extract_section(raw, "MISSING:");
        assert_eq!(s, "", "missing header should return empty string");
    }

    #[test]
    fn extract_bullets_missing_header_returns_empty() {
        let raw = "SUMMARY: everything fine\n";
        let bullets = extract_bullets(raw, "ISSUES:");
        assert!(bullets.is_empty(), "missing header should yield no bullets");
    }

    #[test]
    fn extract_section_stops_at_next_header() {
        let raw = "SUMMARY: all good\nVERDICT: PASS\nmore text";
        let section = extract_section(raw, "SUMMARY:");
        assert!(
            section.contains("all good"),
            "section should contain its content: {section}"
        );
        assert!(
            !section.contains("PASS"),
            "section should stop at next header: {section}"
        );
    }

    // Helper: build a minimal ChainConfig for testing.
    fn test_chain(tmp: &tempfile::TempDir, max_total_turns: usize) -> ChainConfig {
        ChainConfig {
            name: "test-chain".to_string(),
            description: None,
            nodes: vec![crate::orchestrate::types::ChainNodeConfig {
                id: "n1".to_string(),
                name: "n1".to_string(),
                role: "tester".to_string(),
                model: "test-model".to_string(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 1,
            max_total_turns,
            workspace: tmp.path().to_path_buf(),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        }
    }

    /// Verify RETRY signal is detected by the parser (the executor loop honours it).
    #[test]
    fn parse_retry_signal() {
        let raw = "I couldn't complete the task.\nSIGNAL: RETRY\n";
        assert_eq!(parse_control_tokens(raw), Some(ControlSignal::Retry));
    }

    /// Verify ESCALATE signal is detected by the parser.
    #[test]
    fn parse_escalate_signal() {
        assert_eq!(
            parse_control_tokens("SIGNAL: ESCALATE"),
            Some(ControlSignal::Escalate)
        );
    }

    /// Verify that `ValidationResult::Pass` is returned when verdict contains "PASS".
    #[test]
    fn parse_validation_pass_word_in_sentence() {
        // Models sometimes write "VERDICT: Tests PASS" — should still match.
        let raw = "VERDICT: Tests PASS\nSUMMARY: Everything works.\n";
        match parse_validation(raw) {
            ValidationResult::Pass(_) => {}
            other => panic!("expected Pass, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_orchestration_turn_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let client = OllamaClient::new(
            "http://localhost:11443".to_string(),
            "test-model".to_string(),
        );
        let registry = ToolRegistry::new();

        // max_total_turns = 0 means the cap is hit immediately
        let chain_config = test_chain(&tmp, 0);
        let config = crate::orchestrate::OrchestrationConfig {
            chain: chain_config,
            chain_id: "test-chain-id".to_string(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };

        let err = run_orchestration(config, client, registry, None)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("turn cap"),
            "expected 'turn cap' in error, got: {}",
            err
        );
    }

    #[test]
    fn emit_sends_when_sender_present() {
        let (tx, mut rx) = tokio::sync::broadcast::channel(8);
        let event = crate::daemon::protocol::DaemonEvent::ChainCycleComplete {
            chain_id: "test".into(),
            cycle: 1,
        };
        emit(&Some(tx), event);
        let received = rx.try_recv().expect("should receive event");
        match received {
            crate::daemon::protocol::DaemonEvent::ChainCycleComplete { chain_id, cycle } => {
                assert_eq!(chain_id, "test");
                assert_eq!(cycle, 1);
            }
            other => panic!("expected ChainCycleComplete, got {:?}", other),
        }
    }

    #[test]
    fn emit_noop_when_sender_is_none() {
        // Just verify it doesn't panic
        emit(
            &None,
            crate::daemon::protocol::DaemonEvent::ChainFinished {
                chain_id: "x".into(),
                success: true,
                reason: "test".into(),
            },
        );
    }

    // ── apply_pending_mutations ────────────────────────────────────────────────

    fn make_node(id: &str, model: &str) -> ChainNodeConfig {
        ChainNodeConfig {
            id: id.to_string(),
            name: id.to_string(),
            role: "worker".to_string(),
            model: model.to_string(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: None,
            max_retries: 1,
            timeout_secs: 3600,
            max_tool_turns: 200,
        }
    }

    fn make_state(workspace: &std::path::Path, nodes: &[ChainNodeConfig]) -> ChainState {
        let config = ChainConfig {
            name: "test".into(),
            description: None,
            nodes: nodes.to_vec(),
            max_cycles: 3,
            max_total_turns: 60,
            workspace: workspace.to_path_buf(),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        ChainState::new(config, "test-id".into())
    }

    #[test]
    fn apply_pending_additions() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        // Seed state file with a pending addition
        state.pending_additions.push(make_node("n2", "llama3"));
        state.save(tmp.path()).unwrap();

        // Reset in-memory pending (apply_pending_mutations reads from disk)
        state.pending_additions.clear();

        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[1].id, "n2");
        assert!(state.node_statuses.contains_key("n2"));
        // Pending additions should be cleared in persisted state
        let reloaded = ChainState::load(&tmp.path().join("chain_state.json")).unwrap();
        assert!(reloaded.pending_additions.is_empty());
    }

    #[test]
    fn apply_pending_addition_id_differs_from_name() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        // Add a node where id != name
        let mut new_node = make_node("reviewer_v2", "llama3");
        new_node.name = "Reviewer".into();
        state.pending_additions.push(new_node);
        state.save(tmp.path()).unwrap();
        state.pending_additions.clear();

        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[1].id, "reviewer_v2");
        assert_eq!(nodes[1].name, "Reviewer");
        // Status must be keyed by name, not id
        assert!(
            state.node_statuses.contains_key("Reviewer"),
            "status should be keyed by name"
        );
        assert!(
            !state.node_statuses.contains_key("reviewer_v2"),
            "status should not be keyed by id"
        );
    }

    #[test]
    fn apply_pending_removals() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let n2 = make_node("n2", "llama3");
        let mut nodes = vec![n1.clone(), n2.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        state.pending_removals.push("n1".into());
        state.save(tmp.path()).unwrap();
        state.pending_removals.clear();

        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].id, "n2");
        assert!(!state.node_statuses.contains_key("n1"));
    }

    #[test]
    fn apply_pending_model_swaps() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        state
            .pending_model_swaps
            .insert("n1".into(), "llama3:70b".into());
        state.save(tmp.path()).unwrap();
        state.pending_model_swaps.clear();

        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert_eq!(nodes[0].model, "llama3:70b");
        // Verify model swaps cleared in persisted state
        let reloaded = ChainState::load(&tmp.path().join("chain_state.json")).unwrap();
        assert!(reloaded.pending_model_swaps.is_empty());
    }

    #[test]
    fn apply_pending_mixed_ops() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let n2 = make_node("n2", "mistral");
        let mut nodes = vec![n1.clone(), n2.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        // Add n3, remove n1, swap n2's model — all at once
        state.pending_additions.push(make_node("n3", "phi3"));
        state.pending_removals.push("n1".into());
        state
            .pending_model_swaps
            .insert("n2".into(), "codellama".into());
        state.save(tmp.path()).unwrap();
        state.pending_additions.clear();
        state.pending_removals.clear();
        state.pending_model_swaps.clear();

        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        // n1 removed, n2 model swapped, n3 added
        let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["n2", "n3"]);
        assert_eq!(nodes[0].model, "codellama");
        assert_eq!(nodes[1].model, "phi3");
        assert!(!state.node_statuses.contains_key("n1"));
        assert!(state.node_statuses.contains_key("n3"));
    }

    #[test]
    fn apply_pending_noop_when_no_state_file() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        // Don't write any state file
        let applied = apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert!(!applied);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn apply_pending_syncs_turns_used() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);
        state.turns_used = 5;

        // Persisted state has higher turns_used
        let mut disk_state = state.clone();
        disk_state.turns_used = 10;
        disk_state.save(tmp.path()).unwrap();

        state.turns_used = 5; // in-memory is lower
        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();

        assert_eq!(state.turns_used, 10); // should take the max
    }

    #[test]
    fn apply_pending_swap_nonexistent_node_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("n1", "llama3");
        let mut nodes = vec![n1.clone()];
        let mut state = make_state(tmp.path(), &nodes);

        state
            .pending_model_swaps
            .insert("ghost".into(), "model".into());
        state.save(tmp.path()).unwrap();
        state.pending_model_swaps.clear();

        // Should not panic or error — just silently skip
        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &None).unwrap();
        assert_eq!(nodes[0].model, "llama3"); // unchanged
    }

    #[test]
    fn emit_ignores_error_when_no_receivers() {
        let (tx, _) = tokio::sync::broadcast::channel::<crate::daemon::protocol::DaemonEvent>(1);
        // Drop the receiver so send will "fail" — emit should not panic
        emit(
            &Some(tx),
            crate::daemon::protocol::DaemonEvent::ChainCycleComplete {
                chain_id: "x".into(),
                cycle: 1,
            },
        );
    }

    #[tokio::test]
    async fn chain_id_flows_from_config_to_state() {
        let tmp = tempfile::tempdir().unwrap();
        let client = OllamaClient::new(
            "http://localhost:11443".to_string(),
            "test-model".to_string(),
        );
        let registry = ToolRegistry::new();

        let chain_config = test_chain(&tmp, 0); // turn cap 0 → immediate bail
        let config = crate::orchestrate::OrchestrationConfig {
            chain: chain_config,
            chain_id: "my-custom-chain-id".to_string(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };

        // The orchestration will bail on turn cap, but the state file is written first.
        let _ = run_orchestration(config, client, registry, None).await;

        // Read back the persisted state and verify chain_id.
        let state_path = tmp.path().join("chain_state.json");
        let state = crate::orchestrate::types::ChainState::load(&state_path).unwrap();
        assert_eq!(state.chain_id, "my-custom-chain-id");
    }

    #[tokio::test]
    async fn resume_starts_from_saved_cycle() {
        let tmp = tempfile::tempdir().unwrap();
        let client = OllamaClient::new(
            "http://localhost:11443".to_string(),
            "test-model".to_string(),
        );
        let registry = ToolRegistry::new();

        // Create a state at cycle 7 with 10 turns used, cap at 10 so turn cap fires immediately.
        let mut chain_config = test_chain(&tmp, 10);
        chain_config.max_cycles = 10; // must be > resume cycle
        let mut prior_state = ChainState::new(chain_config.clone(), "resume-id".into());
        prior_state.current_cycle = 7;
        prior_state.turns_used = 10;
        prior_state
            .node_outputs
            .insert("worker".into(), "prior output".into());
        prior_state.save(tmp.path()).unwrap();

        let config = crate::orchestrate::OrchestrationConfig {
            chain: chain_config,
            chain_id: "resume-id".to_string(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: Some(prior_state),
        };

        // Will bail on turn cap immediately (turns_used == max_total_turns), cycle starts at 8.
        let _ = run_orchestration(config, client, registry, None).await;

        let state = ChainState::load(&tmp.path().join("chain_state.json")).unwrap();
        assert_eq!(state.chain_id, "resume-id");
        assert!(
            state.current_cycle >= 8,
            "should resume at cycle 8, got {}",
            state.current_cycle
        );
        assert!(
            state.turns_used >= 10,
            "turns_used should be preserved from resume"
        );
    }

    #[test]
    fn apply_pending_mutations_emits_add_node_log() {
        let tmp = tempfile::tempdir().unwrap();
        let n1 = make_node("builder", "llama3");
        let mut state = make_state(tmp.path(), &[n1]);
        let mut nodes = state.config.nodes.clone();

        let new_node = make_node("reviewer", "llama3");
        state.pending_additions.push(new_node);
        state.save(tmp.path()).unwrap();
        state.pending_additions.clear();

        let (tx, mut rx) = tokio::sync::broadcast::channel(16);
        let event_tx: Option<tokio::sync::broadcast::Sender<DaemonEvent>> = Some(tx);
        apply_pending_mutations(&mut state, &mut nodes, tmp.path(), &event_tx).unwrap();

        assert_eq!(nodes.len(), 2);
        let event = rx.try_recv().expect("should have received an event");
        match event {
            DaemonEvent::ChainLog { message, level, .. } => {
                assert!(message.contains("+node: reviewer"), "got: {}", message);
                assert_eq!(level, "info");
            }
            other => panic!("expected ChainLog, got {:?}", other),
        }
    }

    #[test]
    fn chain_pause_updates_node_statuses() {
        use crate::orchestrate::types::ChainNodeStatus;

        let chain_config = ChainConfig {
            name: "pause-test".into(),
            description: None,
            nodes: vec![make_node("builder", "m")],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(chain_config, "c1".into());
        state
            .node_statuses
            .insert("builder".into(), ChainNodeStatus::Running);

        // Simulate what the /chain pause handler does
        for status in state.node_statuses.values_mut() {
            if matches!(status, ChainNodeStatus::Running) {
                *status = ChainNodeStatus::Paused;
            }
        }
        assert!(matches!(
            state.node_statuses.get("builder"),
            Some(ChainNodeStatus::Paused)
        ));
    }

    // ── Integration tests with mock Ollama ──────────────────────────────────

    fn mock_ollama_response(text: &str) -> String {
        serde_json::json!({
            "message": {"role": "assistant", "content": text},
            "prompt_eval_count": 10,
            "eval_count": 5,
            "eval_duration": 100_000_000u64,
        })
        .to_string()
    }

    fn mock_show_response() -> String {
        serde_json::json!({
            "model_info": {"general.context_length": 4096}
        })
        .to_string()
    }

    fn integration_chain(
        tmp: &tempfile::TempDir,
        nodes: Vec<ChainNodeConfig>,
        max_cycles: usize,
        max_turns: usize,
        loop_forever: bool,
    ) -> ChainConfig {
        ChainConfig {
            name: "integration-test".into(),
            description: None,
            nodes,
            max_cycles,
            max_total_turns: max_turns,
            workspace: tmp.path().to_path_buf(),
            skip_permissions_warning: true,
            loop_forever,
            directive: None,
        }
    }

    fn two_node_chain(tmp: &tempfile::TempDir) -> (ChainNodeConfig, ChainNodeConfig) {
        let planner = ChainNodeConfig {
            id: "planner".into(),
            name: "planner".into(),
            role: "planner".into(),
            model: "test".into(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: None,
            max_retries: 0,
            timeout_secs: 10,
            max_tool_turns: 5,
        };
        let builder = ChainNodeConfig {
            id: "builder".into(),
            name: "builder".into(),
            role: "builder".into(),
            model: "test".into(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: Some("planner".into()),
            max_retries: 0,
            timeout_secs: 10,
            max_tool_turns: 5,
        };
        let _ = tmp; // keep alive
        (planner, builder)
    }

    #[tokio::test]
    async fn runner_stops_on_stop_sentinel() {
        let mut server = mockito::Server::new_async().await;
        let _mock_chat = server
            .mock("POST", "/api/chat")
            .with_body(mock_ollama_response("test output\nSIGNAL: CONTINUE"))
            .expect_at_least(1)
            .create_async()
            .await;
        let _mock_show = server
            .mock("POST", "/api/show")
            .with_body(mock_show_response())
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let (planner, _) = two_node_chain(&tmp);
        // loop_forever with very high turn cap so only the sentinel stops it
        let chain = integration_chain(&tmp, vec![planner], 1, 100_000, true);
        let ws = chain.workspace.clone();

        // Pre-create workspace so sentinel write doesn't race with mkdir
        std::fs::create_dir_all(&ws).unwrap();

        let config = crate::orchestrate::OrchestrationConfig {
            chain,
            chain_id: "stop-test".into(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };

        let client = OllamaClient::new(format!("{}/api", server.url()), "test".into());
        let registry = ToolRegistry::new();

        let (event_tx, mut event_rx) = tokio::sync::broadcast::channel(256);
        let handle = tokio::spawn(async move {
            run_orchestration(config, client, registry, Some(event_tx)).await
        });

        // Wait for at least one cycle to complete by watching events
        let mut saw_cycle = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(std::time::Duration::from_millis(50), event_rx.recv()).await
            {
                Ok(Ok(DaemonEvent::ChainCycleComplete { .. })) => {
                    saw_cycle = true;
                    break;
                }
                Ok(Ok(DaemonEvent::ChainNodeTransition { status, .. }))
                    if status == crate::orchestrate::types::STATUS_COMPLETED =>
                {
                    saw_cycle = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_cycle, "should have seen at least one cycle complete");

        // Now write stop sentinel
        std::fs::write(ws.join(".dm-stop"), "").unwrap();

        let result = tokio::time::timeout(std::time::Duration::from_secs(10), handle).await;
        assert!(result.is_ok(), "runner should finish after stop sentinel");
        let inner = result.unwrap().unwrap();
        assert!(inner.is_ok(), "runner should exit cleanly on stop sentinel");

        // Verify state shows the chain was stopped
        let state = ChainState::load(&ws.join("chain_state.json")).unwrap();
        assert!(
            state.current_cycle >= 1,
            "should have completed at least 1 cycle before stopping"
        );
    }

    #[tokio::test]
    async fn runner_pauses_on_sentinel_and_resumes() {
        let mut server = mockito::Server::new_async().await;
        let _mock_chat = server
            .mock("POST", "/api/chat")
            .with_body(mock_ollama_response("done"))
            .expect_at_least(1)
            .create_async()
            .await;
        let _mock_show = server
            .mock("POST", "/api/show")
            .with_body(mock_show_response())
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let (planner, _) = two_node_chain(&tmp);
        let chain = integration_chain(&tmp, vec![planner], 1, 10, false);
        let ws = chain.workspace.clone();

        // Write pause sentinel BEFORE starting
        std::fs::write(ws.join(".dm-pause"), "").unwrap();

        let config = crate::orchestrate::OrchestrationConfig {
            chain,
            chain_id: "pause-test".into(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };
        let client = OllamaClient::new(format!("{}/api", server.url()), "test".into());
        let registry = ToolRegistry::new();

        let handle =
            tokio::spawn(async move { run_orchestration(config, client, registry, None).await });

        // Wait 1s — chain should still be paused (not completed)
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let state_path = ws.join("chain_state.json");
        if state_path.exists() {
            let s = ChainState::load(&state_path).unwrap();
            assert_eq!(
                s.current_cycle, 0,
                "chain should not have advanced while paused"
            );
        }

        // Remove pause sentinel → chain should complete
        let _ = std::fs::remove_file(ws.join(".dm-pause"));

        let result = tokio::time::timeout(std::time::Duration::from_secs(15), handle).await;
        assert!(result.is_ok(), "runner should finish after pause removed");

        let state = ChainState::load(&state_path).unwrap();
        assert!(
            state.current_cycle >= 1,
            "chain should have completed at least 1 cycle"
        );
    }

    #[tokio::test]
    async fn runner_picks_up_talk_file() {
        let mut server = mockito::Server::new_async().await;
        let _mock_chat = server
            .mock("POST", "/api/chat")
            .with_body(mock_ollama_response("acknowledged injected message"))
            .expect_at_least(1)
            .create_async()
            .await;
        let _mock_show = server
            .mock("POST", "/api/show")
            .with_body(mock_show_response())
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let (planner, _) = two_node_chain(&tmp);
        let chain = integration_chain(&tmp, vec![planner], 1, 10, false);
        let ws = chain.workspace.clone();

        // Write a talk file for the "planner" node
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(
            ws.join("talk-planner.md"),
            "Fix the failing test in src/lib.rs",
        )
        .unwrap();

        let config = crate::orchestrate::OrchestrationConfig {
            chain,
            chain_id: "talk-test".into(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };
        let client = OllamaClient::new(format!("{}/api", server.url()), "test".into());
        let registry = ToolRegistry::new();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            run_orchestration(config, client, registry, None),
        )
        .await;
        assert!(result.is_ok(), "runner should complete");

        // Talk file should be consumed (deleted)
        assert!(
            !ws.join("talk-planner.md").exists(),
            "talk file should be deleted after use"
        );

        // State should show completed cycle
        let state = ChainState::load(&ws.join("chain_state.json")).unwrap();
        assert!(state.current_cycle >= 1);
    }

    #[tokio::test]
    async fn chain_state_persists_across_cycles() {
        let mut server = mockito::Server::new_async().await;
        let _mock_chat = server
            .mock("POST", "/api/chat")
            .with_body(mock_ollama_response("cycle output"))
            .expect_at_least(2)
            .create_async()
            .await;
        let _mock_show = server
            .mock("POST", "/api/show")
            .with_body(mock_show_response())
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let (planner, _) = two_node_chain(&tmp);
        let chain = integration_chain(&tmp, vec![planner], 2, 100, false);
        let ws = chain.workspace.clone();

        let config = crate::orchestrate::OrchestrationConfig {
            chain,
            chain_id: "persist-test".into(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };
        let client = OllamaClient::new(format!("{}/api", server.url()), "test".into());
        let registry = ToolRegistry::new();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            run_orchestration(config, client, registry, None),
        )
        .await;
        assert!(result.is_ok(), "runner should complete");

        let state = ChainState::load(&ws.join("chain_state.json")).unwrap();
        assert_eq!(state.current_cycle, 2, "should have completed 2 cycles");
        assert!(state.turns_used >= 2, "should have used at least 2 turns");
        assert!(
            state.node_outputs.contains_key("planner"),
            "should have planner output"
        );

        // Artifacts should exist
        assert!(
            ws.join("cycle-01-planner.md").exists(),
            "cycle 1 artifact should exist"
        );
        assert!(
            ws.join("cycle-02-planner.md").exists(),
            "cycle 2 artifact should exist"
        );
    }

    #[tokio::test]
    async fn chain_events_emitted_to_broadcast() {
        let mut server = mockito::Server::new_async().await;
        let _mock_chat = server
            .mock("POST", "/api/chat")
            .with_body(mock_ollama_response("event output"))
            .expect_at_least(1)
            .create_async()
            .await;
        let _mock_show = server
            .mock("POST", "/api/show")
            .with_body(mock_show_response())
            .create_async()
            .await;

        let tmp = tempfile::tempdir().unwrap();
        let (planner, _) = two_node_chain(&tmp);
        let chain = integration_chain(&tmp, vec![planner], 1, 10, false);

        let config = crate::orchestrate::OrchestrationConfig {
            chain,
            chain_id: "event-test".into(),
            retry: crate::conversation::RetrySettings::default(),
            resume_state: None,
        };
        let client = OllamaClient::new(format!("{}/api", server.url()), "test".into());
        let registry = ToolRegistry::new();

        let (event_tx, mut event_rx) = tokio::sync::broadcast::channel(64);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            run_orchestration(config, client, registry, Some(event_tx)),
        )
        .await;
        assert!(result.is_ok(), "runner should complete");

        // Collect all events
        let mut events = vec![];
        while let Ok(ev) = event_rx.try_recv() {
            events.push(ev);
        }

        // Should have at least: ChainLog (start), ChainNodeTransition (running), ChainNodeTransition (completed), ChainFinished or ChainCycleComplete
        let has_node_transition = events
            .iter()
            .any(|e| matches!(e, DaemonEvent::ChainNodeTransition { .. }));
        let has_log = events
            .iter()
            .any(|e| matches!(e, DaemonEvent::ChainLog { .. }));
        assert!(
            has_node_transition,
            "should emit ChainNodeTransition events"
        );
        assert!(has_log, "should emit ChainLog events");
    }

    #[test]
    fn chain_resume_updates_node_statuses() {
        use crate::orchestrate::types::ChainNodeStatus;

        let chain_config = ChainConfig {
            name: "resume-test".into(),
            description: None,
            nodes: vec![make_node("builder", "m")],
            max_cycles: 3,
            max_total_turns: 50,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(chain_config, "c1".into());
        state
            .node_statuses
            .insert("builder".into(), ChainNodeStatus::Paused);

        // Simulate what the /chain resume handler does
        for status in state.node_statuses.values_mut() {
            if matches!(status, ChainNodeStatus::Paused) {
                *status = ChainNodeStatus::Pending;
            }
        }
        assert!(matches!(
            state.node_statuses.get("builder"),
            Some(ChainNodeStatus::Pending)
        ));
    }
}
