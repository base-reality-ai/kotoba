//! `/chain`-family command support.
//!
//! Phase 1.2 module 3 (seeded C33). Houses inline chain-orchestration
//! dispatch handlers (`handle_chain_*`) extracted from
//! `commands::execute()`. Migrated: `/chain stop` (C33), `/chain talk`
//! (C34), `/chain remove` (C35), `/chain pause` (C36), `/chain model`
//! (C37), `/chain log` (C38), `/chain validate` (C39), `/chain init`
//! (C40), `/chain help` (C41), `/chain metrics` (C42), `/chain list`
//! (C43), `/chain presets` (C44), `/chain add` (C45), `/chain status`
//! (C46), `/chain resume-from` (C47), `/chain resume` (C48),
//! `/chain start` (C49). All 17 `/chain` arms migrated.

use std::fmt::Write as _;

use super::SlashResult;
use crate::ollama::client::OllamaClient;
use crate::tui::app::{App, EntryKind};

/// `/chain stop` — abort the active chain and clear chain state on `app`.
///
/// Pushes a `[chain] stopped by user` system entry, zeroes
/// `app.chain_event_rx` and `app.chain_state`, then calls
/// `orchestrate::abort_chain()`. Returns `Info("Chain stopped.")` on
/// success, `Error("Could not stop chain: {e}. Try: ...")` on abort failure.
pub(super) fn handle_chain_stop(app: &mut App) -> SlashResult {
    app.push_entry(EntryKind::SystemInfo, "[chain] stopped by user".to_string());
    app.chain_event_rx = None;
    app.chain_state = None;
    match crate::orchestrate::abort_chain() {
        Ok(()) => SlashResult::Info("Chain stopped.".into()),
        Err(e) => SlashResult::Error(format!(
            "Could not stop chain: {}. Try: /chain status to confirm whether a chain is active.",
            e
        )),
    }
}

/// `/chain talk <node> <message>` — queue `message` for delivery to chain node `node`.
///
/// Delegates to `orchestrate::chain_talk()`. Returns
/// `Info("Message queued for node '{node}'.")` on success;
/// `Error("chain talk failed: {e}. Try: ...")` on error.
pub(super) fn handle_chain_talk(node: String, message: String) -> SlashResult {
    match crate::orchestrate::chain_talk(&node, &message) {
        Ok(()) => SlashResult::Info(format!("Message queued for node '{}'.", node)),
        Err(e) => SlashResult::Error(format!(
            "chain talk failed: {}. Try: /chain status to list active node names, then re-run /chain talk <node> <message>.",
            e
        )),
    }
}

/// `/chain remove <name>` — queue node `name` for removal on the next chain cycle.
///
/// Delegates to `orchestrate::chain_remove_node()`. On success, also prunes
/// `app.chain_state.node_statuses` and appends to `pending_removals` so the TUI
/// reflects the queued removal immediately. Returns
/// `Info("Node '{name}' queued for removal (takes effect next cycle).")`
/// or `Error("Could not remove node: {e}. Try: ...")`.
pub(super) fn handle_chain_remove(name: String, app: &mut App) -> SlashResult {
    match crate::orchestrate::chain_remove_node(&name) {
        Ok(()) => {
            if let Some(ref mut cs) = app.chain_state {
                cs.node_statuses.remove(&name);
                cs.pending_removals.push(name.clone());
            }
            SlashResult::Info(format!(
                "Node '{}' queued for removal (takes effect next cycle).",
                name
            ))
        }
        Err(e) => SlashResult::Error(format!(
            "Could not remove node: {}. Try: /chain status to confirm the node name and that a chain is active.",
            e
        )),
    }
}

/// `/chain pause` — pause the active chain and mark all running nodes as Paused.
///
/// Delegates to `orchestrate::pause_chain()`. On success, walks
/// `app.chain_state.node_statuses` and transitions every `Running` node to
/// `Paused` so the TUI state tracks the pause immediately. Returns
/// `Info("Chain paused. Use /chain resume to continue.")` or
/// `Error("chain pause failed: {e}. Try: ...")`.
pub(super) fn handle_chain_pause(app: &mut App) -> SlashResult {
    match crate::orchestrate::pause_chain() {
        Ok(()) => {
            if let Some(ref mut cs) = app.chain_state {
                for status in cs.node_statuses.values_mut() {
                    if matches!(status, crate::orchestrate::types::ChainNodeStatus::Running) {
                        *status = crate::orchestrate::types::ChainNodeStatus::Paused;
                    }
                }
            }
            SlashResult::Info("Chain paused. Use /chain resume to continue.".into())
        }
        Err(e) => SlashResult::Error(format!(
            "chain pause failed: {}. Try: /chain status to confirm a chain is active and not already paused.",
            e
        )),
    }
}

/// `/chain model <node> <model>` — queue a model swap for chain node `node` to use `model`.
///
/// Delegates to `orchestrate::chain_model()`. On success, also inserts the
/// `(node, model)` pair into `app.chain_state.pending_model_swaps` so the TUI
/// tracks the queued swap immediately. Returns
/// `Info("Model swap queued: node '{node}' will use '{model}' next cycle.")`
/// or `Error("chain model failed: {e}. Try: ...")`.
pub(super) fn handle_chain_model(node: String, model: String, app: &mut App) -> SlashResult {
    match crate::orchestrate::chain_model(&node, &model) {
        Ok(()) => {
            if let Some(ref mut cs) = app.chain_state {
                cs.pending_model_swaps.insert(node.clone(), model.clone());
            }
            SlashResult::Info(format!(
                "Model swap queued: node '{}' will use '{}' next cycle.",
                node, model
            ))
        }
        Err(e) => SlashResult::Error(format!(
            "chain model failed: {}. Try: /models to list installed Ollama models, then re-run /chain model <node> <model>.",
            e
        )),
    }
}

/// `/chain log [cycle]` — dump chain artifacts from the workspace.
///
/// Delegates to `orchestrate::chain_log(cycle)`. When `cycle` is `Some(n)`,
/// returns only artifacts from that cycle; otherwise returns all. Produces:
/// - `Info("No chain artifacts found.")` when the result is empty
/// - `Info(...)` with each artifact formatted as `\n--- {filename} ---\n{content}\n`
/// - `Error("chain log failed: {e}. Try: ...")` on error
pub(super) fn handle_chain_log(cycle: Option<usize>) -> SlashResult {
    match crate::orchestrate::chain_log(cycle) {
        Ok(entries) if entries.is_empty() => SlashResult::Info("No chain artifacts found.".into()),
        Ok(entries) => {
            let mut output = String::new();
            for entry in &entries {
                write!(
                    output,
                    "\n--- {} ---\n{}\n",
                    entry.filename, entry.content
                )
                .expect("write to String never fails");
            }
            SlashResult::Info(output)
        }
        Err(e) => SlashResult::Error(format!(
            "chain log failed: {}. Try: confirm a chain has run in this workspace, or pass an explicit cycle number with /chain log <n>.",
            e
        )),
    }
}

/// `/chain validate <path-to-chain.yaml>` — load and validate a chain config file.
///
/// Reports structure validity, node-wiring consistency, and model availability
/// against `client`'s `list_models()` output. If `list_models` fails (e.g.
/// Ollama unreachable), model-availability checks are silently skipped
/// (empty model list) rather than surfaced as an error. Returns
/// `Error("Usage: /chain validate <path-to-chain.yaml>")` on empty input,
/// `Error("Failed to load config: {e}. Try: ...")` on load failure, or
/// `Info(validation.report)` with the formatted report.
pub(super) async fn handle_chain_validate(file: String, client: &OllamaClient) -> SlashResult {
    if file.is_empty() {
        return SlashResult::Error("Usage: /chain validate <path-to-chain.yaml>".into());
    }
    let path = std::path::PathBuf::from(&file);
    let chain_config = match crate::orchestrate::load_chain_config(&path) {
        Ok(cfg) => cfg,
        Err(e) => {
            return SlashResult::Error(format!(
                "Failed to load config: {}. Try: confirm the YAML is well-formed and the file path is correct.",
                e
            ));
        }
    };
    let models: Vec<String> = client
        .list_models()
        .await
        .map(|ms| ms.into_iter().map(|m| m.name).collect())
        .unwrap_or_default();
    let validation = crate::orchestrate::validate_chain_config_detailed(&chain_config, &models);
    SlashResult::Info(validation.report)
}

/// `/chain init <name>` — write a starter `<name>.chain.yaml` to the working directory.
///
/// Refuses to overwrite if the target file already exists. Pulls the default
/// model from `Config::load()`, falling back to `"llama3"` if loading fails.
/// Delegates template generation to `orchestrate::generate_chain_template()`.
/// Returns `Error("File '{filename}' already exists. Try: ...")` on collision,
/// `Info("Created chain config: {filename}\n\nEdit it, then run: /chain start {filename}")`
/// on success, or `Error("Failed to write {filename}: {e}. Try: ...")` on write failure.
pub(super) fn handle_chain_init(name: String) -> SlashResult {
    let filename = format!("{}.chain.yaml", name);
    let path = std::path::PathBuf::from(&filename);
    if path.exists() {
        return SlashResult::Error(format!(
            "File '{}' already exists. Try: pick a different name, or remove the existing file first.",
            filename
        ));
    }
    let default_model =
        crate::config::Config::load().map_or_else(|_| "llama3".to_string(), |c| c.model);
    let yaml = crate::orchestrate::generate_chain_template(&name, &default_model);
    match std::fs::write(&path, &yaml) {
        Ok(()) => SlashResult::Info(format!(
            "Created chain config: {}\n\nEdit it, then run: /chain start {}",
            filename, filename
        )),
        Err(e) => SlashResult::Error(format!(
            "Failed to write {}: {}. Try: confirm the current directory is writable.",
            filename, e
        )),
    }
}

/// `/chain help` — print the chain-subcommand help text.
///
/// Returns a static `Info(...)` with one line per `/chain` subcommand
/// (`start`, `init`, `status`, `stop`, `log`, `list`, `presets`, `add`,
/// `remove`, `talk`, `model`, `validate`, `metrics`, `pause`, `resume`,
/// `resume-from`, `help`). Pure formatting — no state reads, no I/O,
/// no async.
pub(super) fn handle_chain_help() -> SlashResult {
    SlashResult::Info(
        "Chain subcommands:\n\
         \n\
         /chain start <file>          — start a chain from a YAML config file\n\
         /chain init [name]           — scaffold a starter chain config YAML\n\
         /chain status                — show running chain status\n\
         /chain stop                  — send stop signal to running chain\n\
         /chain log [cycle]           — show chain output from a cycle\n\
         /chain list                  — list available chain config files\n\
         /chain presets               — list built-in chain presets (continuous-dev, self-improve, project-audit)\n\
         /chain add <name> <role> [model] [input_from] — add node (auto-connects to last node)\n\
         /chain remove <name>         — enqueue a node for removal next cycle\n\
         /chain talk <node> <msg>     — inject a message into a node's next prompt\n\
         /chain model <node> <model>  — swap the model of a node at runtime\n\
         /chain validate <file>       — validate config + check models without running\n\
         /chain metrics               — show per-node timing and failure stats\n\
         /chain pause                 — pause the running chain between cycles\n\
         /chain resume                — resume a paused chain\n\
         /chain resume-from <workspace> — resume chain from checkpoint\n\
         /chain help                  — show this help"
            .into(),
    )
}

/// `/chain metrics` — load chain state from disk and format per-node timing/failure stats.
///
/// Locates the chain state file by checking `~/.dm/chain_state.json` first,
/// falling back to `<cwd>/chain_state.json`. Returns
/// `Error("No chain state found. Start a chain first.")` if neither exists,
/// `Error("Failed to load chain state: {e}. Try: ...")` on load failure, or
/// `Info(orchestrate::format_chain_metrics(&state))` with the rendered report.
pub(super) fn handle_chain_metrics() -> SlashResult {
    let workspace = if let Some(home) = dirs::home_dir() {
        let dm_state = home.join(".dm").join("chain_state.json");
        if dm_state.exists() {
            home.join(".dm")
        } else {
            std::env::current_dir().unwrap_or_default()
        }
    } else {
        std::env::current_dir().unwrap_or_default()
    };
    let state_path = workspace.join("chain_state.json");
    if !state_path.exists() {
        return SlashResult::Error("No chain state found. Start a chain first.".into());
    }
    match crate::orchestrate::types::ChainState::load(&state_path) {
        Ok(state) => SlashResult::Info(crate::orchestrate::format_chain_metrics(&state)),
        Err(e) => SlashResult::Error(format!(
            "Failed to load chain state: {}. Try: /chain start <file> to begin a new run, or remove the corrupted state file.",
            e
        )),
    }
}

/// `/chain list` — enumerate available chain configs on disk.
///
/// Scans `~/.dm/chains/` and the current working directory for `.yaml` /
/// `.yml` files, attempts to parse each via `load_chain_config()`, and
/// returns a formatted `Info(...)` listing each discoverable config's
/// path, name, node count, and optional description. Silently skips
/// directories that can't be read and files that fail to parse — only
/// successfully-loaded configs appear in the output. Returns
/// `Info("No chain configs found in ~/.dm/chains/ or current directory.")`
/// when no configs are discoverable.
pub(super) fn handle_chain_list() -> SlashResult {
    let mut configs = Vec::new();
    // Scan ~/.dm/chains/
    if let Some(home) = dirs::home_dir() {
        let chains_dir = home.join(".dm").join("chains");
        if let Ok(entries) = std::fs::read_dir(&chains_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                    if let Ok(cfg) = crate::orchestrate::load_chain_config(&path) {
                        configs.push((path, cfg));
                    }
                }
            }
        }
    }
    // Scan current directory
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                if let Ok(cfg) = crate::orchestrate::load_chain_config(&path) {
                    configs.push((path, cfg));
                }
            }
        }
    }
    if configs.is_empty() {
        return SlashResult::Info(
            "No chain configs found in ~/.dm/chains/ or current directory.".into(),
        );
    }
    let mut lines = vec!["Available chain configs:".to_string()];
    for (path, cfg) in &configs {
        let desc = cfg.description.as_deref().unwrap_or("");
        let nodes = cfg.nodes.len();
        lines.push(format!(
            "  {} — \"{}\" ({} nodes) {}",
            path.display(),
            cfg.name,
            nodes,
            if desc.is_empty() {
                String::new()
            } else {
                format!("— {desc}")
            }
        ));
    }
    SlashResult::Info(lines.join("\n"))
}

/// `/chain presets` — list built-in chain presets with a short shape summary.
///
/// Iterates `orchestrate::list_chain_presets()`, resolves each via
/// `orchestrate::resolve_chain_preset()` (skipping unresolvable names), and
/// formats one row per preset with node count, cycle shape (looping vs
/// single-pass), and the first non-empty trimmed line of the preset's
/// description as a one-line summary. Appends two footer hints explaining
/// how to run a preset via `dm --chain <name>` or `/chain start <name>`.
/// Returns `Info(...)` with the assembled listing.
pub(super) fn handle_chain_presets() -> SlashResult {
    let mut lines = vec!["Built-in chain presets:".to_string()];
    for name in crate::orchestrate::list_chain_presets() {
        let Some(cfg) = crate::orchestrate::resolve_chain_preset(name) else {
            continue;
        };
        let shape = if cfg.loop_forever {
            format!("looping, max {} cycles", cfg.max_cycles)
        } else {
            format!("single-pass, {} cycle(s)", cfg.max_cycles)
        };
        let summary = cfg
            .description
            .as_deref()
            .unwrap_or("")
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("")
            .to_string();
        lines.push(format!(
            "  {} — {} nodes, {} — {}",
            name,
            cfg.nodes.len(),
            shape,
            summary
        ));
    }
    lines.push(String::new());
    lines.push("Run any preset with: dm --chain <name>".to_string());
    lines.push("Or from inside the TUI: /chain start <name>".to_string());
    SlashResult::Info(lines.join("\n"))
}

/// `/chain add <name> <role> [model] [input_from]` — enqueue a new node for addition next cycle.
///
/// Empty `model` defaults to `app.model`. Pulls `current_state` from
/// `app.chain_state` if live, falling back to `orchestrate::chain_status()`.
/// When `input_from` is provided, validates it against the running chain's
/// node names; rejects with `Error("input_from '{target}' does not match
/// any node in the running chain. Known nodes: {list}")`. If `input_from`
/// is omitted, auto-wires to the last node in the current chain.
/// Constructs a `ChainNodeConfig` with defaults (`description: None`,
/// `system_prompt_override: None`, `system_prompt_file: None`,
/// `max_retries: 1`, `timeout_secs: 3600`, `max_tool_turns: 200`) and
/// calls `orchestrate::chain_add_node()`. On success, also inserts
/// `(name, Pending)` into `app.chain_state.node_statuses` so the TUI
/// reflects the queued addition immediately. Returns
/// `Info("Node '{name}' queued for addition (takes effect next cycle).")`
/// or `Error("Could not add node: {e}. Try: ...")`.
pub(super) fn handle_chain_add(
    name: String,
    role: String,
    model: String,
    input_from: Option<String>,
    app: &mut App,
) -> SlashResult {
    let model = if model.is_empty() {
        app.model.clone()
    } else {
        model
    };
    let current_state = app
        .chain_state
        .clone()
        .or_else(crate::orchestrate::chain_status);
    if let Some(ref target) = input_from {
        if let Some(ref st) = current_state {
            let known: std::collections::HashSet<&str> =
                st.config.nodes.iter().map(|n| n.name.as_str()).collect();
            if !known.contains(target.as_str()) {
                return SlashResult::Error(format!(
                    "input_from '{}' does not match any node in the running chain. Known nodes: {}",
                    target,
                    st.config
                        .nodes
                        .iter()
                        .map(|n| n.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
    }
    let resolved_input = input_from
        .or_else(|| current_state.and_then(|st| st.config.nodes.last().map(|n| n.name.clone())));
    let node = crate::orchestrate::types::ChainNodeConfig {
        id: name.clone(),
        name: name.clone(),
        role,
        model,
        description: None,
        system_prompt_override: None,
        system_prompt_file: None,
        input_from: resolved_input,
        max_retries: 1,
        timeout_secs: 3600,
        max_tool_turns: 200,
    };
    match crate::orchestrate::chain_add_node(node) {
        Ok(()) => {
            if let Some(ref mut cs) = app.chain_state {
                cs.node_statuses.insert(
                    name.clone(),
                    crate::orchestrate::types::ChainNodeStatus::Pending,
                );
            }
            SlashResult::Info(format!(
                "Node '{}' queued for addition (takes effect next cycle).",
                name
            ))
        }
        Err(e) => SlashResult::Error(format!(
            "Could not add node: {}. Try: /chain status to confirm a chain is running, or check the node name has no spaces or special characters.",
            e
        )),
    }
}

/// `/chain status` — print a multi-line readout of the currently running (or last-completed) chain.
///
/// Reads `app.chain_state` if live, falling back to
/// `orchestrate::chain_status()` for on-disk state. Formats `Chain:`,
/// `Cycle:` (as `{n}/∞` for looping chains, `{n}/{max}` otherwise),
/// `Active:` node name, `Turns used: {n}/{max}`, then per-node rows
/// ordered by config (not `HashMap` iteration order) showing status,
/// avg timing over runs, and failure count. Appends `Total time`
/// if the chain has accumulated duration, plus `Last abort` and
/// `Last signal` lines when either is set. Returns
/// `Info("No chain is running or no state file found.")` when no
/// state is available, or `Info(<formatted>)` otherwise.
pub(super) fn handle_chain_status(app: &mut App) -> SlashResult {
    let status_from_memory = app
        .chain_state
        .clone()
        .or_else(crate::orchestrate::chain_status);
    match status_from_memory {
        None => SlashResult::Info("No chain is running or no state file found.".into()),
        Some(state) => {
            let cycle_display = if state.config.loop_forever {
                format!("{}/∞", state.current_cycle)
            } else {
                format!("{}/{}", state.current_cycle, state.config.max_cycles)
            };
            let active = state
                .active_node_index
                .and_then(|i| state.config.nodes.get(i))
                .map_or("(none)", |n| n.name.as_str());
            let mut lines = format!(
                "Chain: {}\n  Cycle:      {}\n  Active:     {}\n  Turns used: {}/{}",
                state.config.name,
                cycle_display,
                active,
                state.turns_used,
                state.config.max_total_turns,
            );
            // Show nodes in config order (not HashMap iteration order).
            if !state.config.nodes.is_empty() {
                lines.push_str("\n  Nodes:");
                for node in &state.config.nodes {
                    let status = state
                        .node_statuses
                        .get(&node.name)
                        .map_or_else(|| "unknown".into(), |s| s.to_string());
                    // Append timing info if available
                    let timing = state
                        .node_durations
                        .get(&node.name)
                        .filter(|d| !d.is_empty())
                        .map(|d| {
                            let avg = d.iter().sum::<f64>() / d.len() as f64;
                            format!(" ({} runs, avg {:.1}s)", d.len(), avg)
                        })
                        .unwrap_or_default();
                    let fails = state.node_failures.get(&node.name).copied().unwrap_or(0);
                    let fail_info = if fails > 0 {
                        format!(" [{} fail{}]", fails, if fails == 1 { "" } else { "s" })
                    } else {
                        String::new()
                    };
                    write!(
                        lines,
                        "\n    {:<16} {}{}{}",
                        node.name, status, timing, fail_info
                    )
                    .expect("write to String never fails");
                }
            }
            if state.total_duration_secs > 0.0 {
                write!(lines, "\n  Total time: {:.1}s", state.total_duration_secs)
                    .expect("write to String never fails");
            }
            if let Some(ref reason) = state.last_abort_reason {
                write!(lines, "\n  Last abort: {}", reason).expect("write to String never fails");
            }
            if let Some(ref sig) = state.last_signal {
                write!(lines, "\n  Last signal: {:?}", sig).expect("write to String never fails");
            }
            SlashResult::Info(lines)
        }
    }
}

/// `/chain resume-from <workspace>` — resume a chain from a checkpoint workspace.
///
/// Refuses to start if `app.chain_state` is already populated. Loads the
/// orchestration config from the workspace via
/// `orchestrate::resume_config_from_workspace()`. If a `resume_state`
/// checkpoint exists, clones it and overlays the freshly-loaded
/// `chain_config` and `chain_id`; otherwise constructs a fresh
/// `ChainState::new(...)`. Builds a default tool registry, opens a
/// broadcast channel (capacity 64), and spawns
/// `runner::run_orchestration()` on the tokio runtime — failures inside
/// the spawned task surface via `warnings::push_warning()`. Stores the
/// `JoinHandle` via `orchestrate::set_chain()` and seeds `app` with the
/// receiver and initial state. Returns
/// `Error("A chain is already running. Use /chain stop first.")` on
/// double-start, `Error("Could not resume: {e}. Try: ...")` on config load
/// failure, or
/// `Info("Resuming chain '{name}' from cycle {n} in background.")`
/// on success.
pub(super) fn handle_chain_resume_from(
    workspace_path: String,
    app: &mut App,
    client: &OllamaClient,
) -> SlashResult {
    if app.chain_state.is_some() {
        return SlashResult::Error("A chain is already running. Use /chain stop first.".into());
    }
    let workspace = std::path::PathBuf::from(&workspace_path);
    let orch_config = match crate::orchestrate::resume_config_from_workspace(&workspace) {
        Ok(c) => c,
        Err(e) => {
            return SlashResult::Error(format!(
                "Could not resume: {}. Try: confirm the workspace path exists and contains a chain.yaml.",
                e
            ));
        }
    };
    let chain_name = orch_config.chain.name.clone();
    let resume_cycle = orch_config
        .resume_state
        .as_ref()
        .map_or(0, |s| s.current_cycle);
    // Build initial chain state from resume checkpoint (or fresh if no resume state).
    let initial_state = if let Some(ref rs) = orch_config.resume_state {
        let mut s = rs.clone();
        s.config = orch_config.chain.clone();
        s.chain_id.clone_from(&orch_config.chain_id);
        s
    } else {
        crate::orchestrate::types::ChainState::new(
            orch_config.chain.clone(),
            orch_config.chain_id.clone(),
        )
    };
    let workspace_clone = workspace;
    let chain_client = client.clone();
    let session_id = uuid::Uuid::new_v4().to_string();
    let chain_registry = crate::tools::registry::default_registry(
        &session_id,
        &app.config_dir,
        chain_client.base_url(),
        &app.model,
        "",
    );
    let (chain_tx, chain_rx) = tokio::sync::broadcast::channel(64);
    let handle = tokio::spawn(async move {
        if let Err(e) = crate::orchestrate::runner::run_orchestration(
            orch_config,
            chain_client,
            chain_registry,
            Some(chain_tx),
        )
        .await
        {
            crate::warnings::push_warning(format!("[chain] Resume failed: {}", e));
        }
    });
    crate::orchestrate::set_chain(handle, workspace_clone);
    app.chain_event_rx = Some(chain_rx);
    app.chain_state = Some(initial_state);
    SlashResult::Info(format!(
        "Resuming chain '{}' from cycle {} in background.",
        chain_name, resume_cycle
    ))
}

/// `/chain resume` — resume a paused chain or restart the most recent chain from disk.
///
/// Two-branch dispatch:
///
/// 1. If `app.chain_state.is_some()` (a chain is loaded in memory),
///    delegates to `orchestrate::resume_chain()`. On success, walks
///    `node_statuses` and transitions every `Paused` node to `Pending`
///    so the TUI tracks the resume immediately. Returns
///    `Info("Chain resumed.")` or `Error("chain resume failed: {e}. Try: ...")`.
///
/// 2. Otherwise, attempts `orchestrate::load_last_chain_pointer()` to
///    locate the most recent chain workspace on disk. Loads its config,
///    overlays the resume checkpoint (or constructs a fresh `ChainState`),
///    builds a default tool registry, opens a broadcast channel
///    (capacity 64), and spawns `runner::run_orchestration()`. Stores
///    the `JoinHandle` via `set_chain()` and seeds `app` with the receiver
///    and initial state. Returns `Info("Resuming last chain '{name}' from cycle {n}.")`
///    on success, `Error("Could not resume last chain: {e}. Try: ...")` on config load failure.
///
/// 3. If neither branch fires (no in-memory chain AND no disk pointer), returns
///    `Error("No chain is running and no previous chain found to resume.")`.
pub(super) fn handle_chain_resume(app: &mut App, client: &OllamaClient) -> SlashResult {
    if app.chain_state.is_some() {
        match crate::orchestrate::resume_chain() {
            Ok(()) => {
                if let Some(ref mut cs) = app.chain_state {
                    for status in cs.node_statuses.values_mut() {
                        if matches!(status, crate::orchestrate::types::ChainNodeStatus::Paused) {
                            *status = crate::orchestrate::types::ChainNodeStatus::Pending;
                        }
                    }
                }
                return SlashResult::Info("Chain resumed.".into());
            }
            Err(e) => {
                return SlashResult::Error(format!(
                    "chain resume failed: {}. Try: /chain status to confirm a paused chain exists, or use /chain resume <workspace> for an explicit path.",
                    e
                ));
            }
        }
    }
    if let Some(workspace) = crate::orchestrate::load_last_chain_pointer() {
        let orch_config = match crate::orchestrate::resume_config_from_workspace(&workspace) {
            Ok(c) => c,
            Err(e) => {
                return SlashResult::Error(format!(
                    "Could not resume last chain: {}. Try: /chain start <chain.yaml> to begin a new run, or /chain resume <workspace> with an explicit path.",
                    e
                ));
            }
        };
        let chain_name = orch_config.chain.name.clone();
        let resume_cycle = orch_config
            .resume_state
            .as_ref()
            .map_or(0, |s| s.current_cycle);
        let initial_state = if let Some(ref rs) = orch_config.resume_state {
            let mut s = rs.clone();
            s.config = orch_config.chain.clone();
            s.chain_id.clone_from(&orch_config.chain_id);
            s
        } else {
            crate::orchestrate::types::ChainState::new(
                orch_config.chain.clone(),
                orch_config.chain_id.clone(),
            )
        };
        let workspace_clone = workspace;
        let chain_client = client.clone();
        let session_id = uuid::Uuid::new_v4().to_string();
        let chain_registry = crate::tools::registry::default_registry(
            &session_id,
            &app.config_dir,
            chain_client.base_url(),
            &app.model,
            "",
        );
        let (chain_tx, chain_rx) = tokio::sync::broadcast::channel(64);
        let handle = tokio::spawn(async move {
            if let Err(e) = crate::orchestrate::runner::run_orchestration(
                orch_config,
                chain_client,
                chain_registry,
                Some(chain_tx),
            )
            .await
            {
                crate::warnings::push_warning(format!("[chain] Resume failed: {}", e));
            }
        });
        crate::orchestrate::set_chain(handle, workspace_clone);
        app.chain_event_rx = Some(chain_rx);
        app.chain_state = Some(initial_state);
        return SlashResult::Info(format!(
            "Resuming last chain '{}' from cycle {}.",
            chain_name, resume_cycle
        ));
    }
    SlashResult::Error("No chain is running and no previous chain found to resume.".into())
}

/// `/chain start <path>` — load a chain config from `path` and spawn it.
///
/// Validates: non-empty `file`, no chain currently running, config loads
/// (Try: hints differ for missing path vs malformed YAML), and every node's
/// `model` exists in `app.available_models` (skipped when that list is
/// empty, e.g. during early startup). Builds an `OrchestrationConfig`,
/// spawns `runner::run_orchestration` on a tokio task, registers the
/// handle via `orchestrate::set_chain`, wires the broadcast receiver into
/// `app.chain_event_rx`, and seeds `app.chain_state`. Returns
/// `Info("Chain started.")`.
pub(super) fn handle_chain_start(
    file: String,
    app: &mut App,
    client: &OllamaClient,
) -> SlashResult {
    if file.is_empty() {
        return SlashResult::Error(
            "Usage: /chain start <path-to-chain.yaml>\nTip: use /chain list to see available configs.".into(),
        );
    }
    if app.chain_state.is_some() || crate::orchestrate::chain_status().is_some() {
        return SlashResult::Error("A chain is already running. Use /chain stop first.".into());
    }
    let path = std::path::PathBuf::from(&file);
    let chain_config = match crate::orchestrate::load_chain_config(&path) {
        Ok(c) => c,
        Err(e) => {
            let hint = if !path.exists() {
                ". Try: /chain list to see available configs."
            } else {
                ". Try: /chain validate <file> to inspect the YAML and confirm structure."
            };
            return SlashResult::Error(format!("Failed to load chain config: {}{}", e, hint));
        }
    };
    if !app.available_models.is_empty() {
        let missing: Vec<&str> = chain_config
            .nodes
            .iter()
            .map(|n| n.model.as_str())
            .filter(|m| !app.available_models.iter().any(|am| am == m))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if !missing.is_empty() {
            return SlashResult::Error(format!(
                "Unknown model(s): {}. Use /models to see available models.",
                missing.join(", ")
            ));
        }
    }
    let workspace = chain_config.workspace.clone();
    let chain_id = format!("chain-{}", uuid::Uuid::new_v4().as_simple());
    let orch_config = crate::orchestrate::OrchestrationConfig {
        chain: chain_config,
        chain_id,
        retry: crate::conversation::RetrySettings::default(),
        resume_state: None,
    };
    let initial_config = orch_config.chain.clone();
    let initial_chain_id = orch_config.chain_id.clone();
    let chain_client = client.clone();
    let session_id = uuid::Uuid::new_v4().to_string();
    let chain_registry = crate::tools::registry::default_registry(
        &session_id,
        &app.config_dir,
        chain_client.base_url(),
        &app.model,
        "",
    );
    let (chain_tx, chain_rx) = tokio::sync::broadcast::channel(64);
    let chain_tx_err = chain_tx.clone();
    let err_chain_id = orch_config.chain_id.clone();
    let handle = tokio::task::spawn(async move {
        if let Err(e) = crate::orchestrate::runner::run_orchestration(
            orch_config,
            chain_client,
            chain_registry,
            Some(chain_tx),
        )
        .await
        {
            let _ = chain_tx_err.send(crate::daemon::protocol::DaemonEvent::ChainFinished {
                chain_id: err_chain_id,
                success: false,
                reason: format!("runner error: {}", e),
            });
            crate::warnings::push_warning(format!("[chain] error: {}", e));
        }
    });
    crate::orchestrate::set_chain(handle, workspace);
    app.chain_event_rx = Some(chain_rx);
    app.chain_state = Some(crate::orchestrate::types::ChainState::new(
        initial_config,
        initial_chain_id,
    ));
    SlashResult::Info("Chain started.".into())
}
