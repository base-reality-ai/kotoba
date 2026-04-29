//! Chain orchestration and multi-agent workflows.
//!
//! Manages YAML-defined agent chains, state transitions, and node execution.

pub mod presets;
pub mod runner;
pub mod types;
use crate::orchestrate::types::{ChainConfig, ChainNodeConfig, ChainState};
use anyhow::{Context, Result};
pub use presets::{list_chain_presets, resolve_chain_preset};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokio::task::JoinHandle;

// ── Node name validation ─────────────────────────────────────────────────────

/// Returns true if `name` is safe to use in filenames and state keys.
/// Rejects empty/whitespace, path separators, `..`, null bytes.
/// Allows alphanumeric, `-`, `_`, and single `.` characters.
pub fn is_safe_node_name(name: &str) -> bool {
    if name.is_empty() || name.trim().is_empty() {
        return false;
    }
    if name.contains('\0') || name.contains('/') || name.contains('\\') || name.contains("..") {
        return false;
    }
    name.chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
}

fn require_safe_node_name(name: &str) -> Result<()> {
    if !is_safe_node_name(name) {
        anyhow::bail!("Invalid node name '{}': must be alphanumeric, '-', '_', or '.' with no path separators", name);
    }
    Ok(())
}

// ── Global chain handle ──────────────────────────────────────────────────────

static CHAIN_HANDLE: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
static CHAIN_WORKSPACE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Store the running chain's task handle and workspace path.
pub fn set_chain(handle: JoinHandle<()>, workspace: PathBuf) {
    if let Ok(mut h) = CHAIN_HANDLE.lock() {
        if let Some(old) = h.take() {
            old.abort();
        }
        *h = Some(handle);
    }
    save_last_chain_pointer(&workspace);
    if let Ok(mut w) = CHAIN_WORKSPACE.lock() {
        *w = Some(workspace);
    }
}

/// Persist the last-active chain workspace to
/// `<config_dir>/last_chain.json`. Identity-aware: in host mode, the
/// pointer lands at `<project>/.dm/last_chain.json` next to the rest of
/// the project's chain workspace state; in kernel mode, `~/.dm/`
/// (Run-31 routing).
///
/// IO failures push a warning rather than vanishing — the pointer is
/// best-effort persistence, but a silent loss leaves `dm chain attach`
/// unable to find its workspace with no diagnostic trail.
pub fn save_last_chain_pointer(workspace: &Path) {
    let Some(config_dir) = crate::config::current_project_config_dir() else {
        crate::warnings::push_warning(
            "orchestrate: home directory unresolved, cannot persist last_chain pointer".to_string(),
        );
        return;
    };
    let path = config_dir.join("last_chain.json");
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            crate::warnings::push_warning(format!(
                "orchestrate: failed to create {} for last_chain pointer: {}",
                parent.display(),
                e
            ));
            return;
        }
    }
    let pointer = serde_json::json!({
        "workspace": workspace.to_string_lossy(),
        "started_at": chrono::Utc::now().to_rfc3339(),
    });
    if let Err(e) = fs::write(&path, pointer.to_string()) {
        crate::warnings::push_warning(format!(
            "orchestrate: failed to write last_chain pointer at {}: {}",
            path.display(),
            e
        ));
    }
}

/// Load the last-active chain workspace from
/// `<config_dir>/last_chain.json`. Identity-aware (mirrors
/// `save_last_chain_pointer`).
pub fn load_last_chain_pointer() -> Option<PathBuf> {
    let path = crate::config::current_project_config_dir()?.join("last_chain.json");
    let data = fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    let ws = json["workspace"].as_str()?;
    let pb = PathBuf::from(ws);
    if pb.exists() {
        Some(pb)
    } else {
        None
    }
}

/// Remove the last-active chain pointer from `<config_dir>/last_chain.json`.
/// Identity-aware (mirrors `save_last_chain_pointer`). Removal failures
/// stay silent — a missing file is the success state.
pub fn clear_last_chain_pointer() {
    if let Some(config_dir) = crate::config::current_project_config_dir() {
        let _ = fs::remove_file(config_dir.join("last_chain.json"));
    }
}

/// Read `chain_state.json` from the stored workspace and return it, if any.
pub fn chain_status() -> Option<ChainState> {
    let workspace = CHAIN_WORKSPACE.lock().ok()?.as_ref()?.clone();
    let state_path = workspace.join("chain_state.json");
    ChainState::load(&state_path).ok()
}

/// Write a `.dm-stop` sentinel file into the workspace to signal the runner.
pub fn stop_chain() -> Result<()> {
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    let stop_file = workspace.join(".dm-stop");
    fs::write(&stop_file, b"")?;
    Ok(())
}

/// Write the stop sentinel, abort the task, and clear global state.
pub fn abort_chain() -> Result<()> {
    let _ = stop_chain();
    clear_chain();
    Ok(())
}

/// Abort the chain task (if running) and clear both global handles.
pub fn clear_chain() {
    if let Ok(mut h) = CHAIN_HANDLE.lock() {
        if let Some(handle) = h.take() {
            handle.abort();
        }
    }
    if let Ok(mut w) = CHAIN_WORKSPACE.lock() {
        *w = None;
    }
    clear_last_chain_pointer();
}

/// Write a `.dm-pause` sentinel; the runner spin-waits until it is removed.
pub fn pause_chain() -> Result<()> {
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    fs::write(workspace.join(".dm-pause"), b"")?;
    Ok(())
}

/// Remove the `.dm-pause` sentinel so the runner continues.
pub fn resume_chain() -> Result<()> {
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    let pause_file = workspace.join(".dm-pause");
    if pause_file.exists() {
        fs::remove_file(&pause_file)?;
    }
    Ok(())
}

/// Enqueue a new node for addition to the running chain.
/// The runner will pick it up and insert it at the start of the next cycle.
pub fn chain_add_node(mut node: ChainNodeConfig) -> Result<()> {
    require_safe_node_name(&node.id)?;
    require_safe_node_name(&node.name)?;
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    // Resolve model alias from the operator-global config dir (alias
    // tables stay shared across host projects on the same machine).
    let config_dir = crate::config::current_global_config_dir().unwrap_or_default();
    let aliases = crate::config::load_aliases(&config_dir);
    if let Some(resolved) = aliases.get(&node.model) {
        node.model.clone_from(resolved);
    }
    let state_path = workspace.join("chain_state.json");
    let mut state = ChainState::load(&state_path)
        .with_context(|| "Failed to load chain state for node addition")?;
    state.pending_additions.push(node);
    state.save(&workspace)?;
    Ok(())
}

/// Enqueue a model swap for a node in the running chain.
/// The runner applies it at the start of the next cycle.
pub fn chain_model(node: &str, model: &str) -> Result<()> {
    require_safe_node_name(node)?;
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    // Resolve model alias from the operator-global config dir (alias
    // tables stay shared across host projects on the same machine).
    let config_dir = crate::config::current_global_config_dir().unwrap_or_default();
    let aliases = crate::config::load_aliases(&config_dir);
    let resolved_model = aliases.get(model).map_or(model, |s| s.as_str());
    let state_path = workspace.join("chain_state.json");
    let mut state = ChainState::load(&state_path)
        .with_context(|| "Failed to load chain state for model swap")?;
    state
        .pending_model_swaps
        .insert(node.to_string(), resolved_model.to_string());
    state.save(&workspace)?;
    Ok(())
}

/// Enqueue a node removal for the running chain.
/// The runner will drop the named node at the start of the next cycle.
pub fn chain_remove_node(name: &str) -> Result<()> {
    require_safe_node_name(name)?;
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    let state_path = workspace.join("chain_state.json");
    let mut state = ChainState::load(&state_path)
        .with_context(|| "Failed to load chain state for node removal")?;
    state.pending_removals.push(name.to_string());
    state.save(&workspace)?;
    Ok(())
}

/// Write a talk-injection file so the runner prepends `message` to that
/// node's prompt on its next execution.
pub fn chain_talk(node: &str, message: &str) -> Result<()> {
    require_safe_node_name(node)?;
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    let talk_file = workspace.join(format!("talk-{}.md", node));
    fs::write(&talk_file, message.as_bytes())?;
    Ok(())
}

/// A single chain artifact entry (one node's output from one cycle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainLogEntry {
    pub cycle: usize,
    pub node: String,
    pub filename: String,
    pub content: String,
}

/// Read chain artifacts from the workspace directory.
/// If `cycle` is Some, return only artifacts from that cycle.
/// Returns artifacts sorted by filename.
pub fn chain_log(cycle: Option<usize>) -> Result<Vec<ChainLogEntry>> {
    let workspace = CHAIN_WORKSPACE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().cloned())
        .ok_or_else(|| anyhow::anyhow!("No chain is running"))?;
    chain_log_from_workspace(&workspace, cycle)
}

/// Read chain artifacts from a specific workspace path.
pub fn chain_log_from_workspace(
    workspace: &Path,
    cycle: Option<usize>,
) -> Result<Vec<ChainLogEntry>> {
    let mut entries = Vec::new();
    let dir = fs::read_dir(workspace)?;
    for entry in dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("cycle-") || !name.ends_with(".md") {
            continue;
        }

        // Parse cycle number and node name from "cycle-NN-nodename.md"
        let stem = name.strip_suffix(".md").unwrap_or(name);
        let parts: Vec<&str> = stem.splitn(3, '-').collect();
        if parts.len() < 3 {
            continue;
        }
        let Ok(artifact_cycle) = parts[1].parse::<usize>() else {
            continue;
        };
        let node_name = parts[2].to_string();

        if let Some(filter_cycle) = cycle {
            if artifact_cycle != filter_cycle {
                continue;
            }
        }

        let content = fs::read_to_string(&path).unwrap_or_default();
        // Cap at 4000 chars per artifact
        let truncated = if content.len() > 4000 {
            let mut end = 4000;
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…[truncated]", &content[..end])
        } else {
            content
        };

        entries.push(ChainLogEntry {
            cycle: artifact_cycle,
            node: node_name,
            filename: name.to_string(),
            content: truncated,
        });
    }
    entries.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(entries)
}

/// Detailed validation report for a chain config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainValidationReport {
    pub chain_name: String,
    pub node_count: usize,
    pub max_cycles: usize,
    pub loop_forever: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub model_checks: Vec<(String, String, bool)>,
    pub report: String,
}

/// Validate a chain config in detail: structure, file existence, model availability.
/// Pass available model names to check model availability, or empty vec to skip.
pub fn validate_chain_config_detailed(
    config: &ChainConfig,
    available_models: &[String],
) -> ChainValidationReport {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut model_checks = Vec::new();
    let mut report = String::new();

    match validate_chain_config(config) {
        Ok(w) => {
            report.push_str("Structure:  OK\n");
            warnings.extend(w);
        }
        Err(e) => {
            errors.push(format!("Structure: {}", e));
            writeln!(report, "Structure:  FAIL — {}", e).expect("write to String never fails");
        }
    }

    writeln!(report, "Chain:      {}", config.name).expect("write to String never fails");
    writeln!(report, "Nodes:      {}", config.nodes.len()).expect("write to String never fails");
    writeln!(report, "Max cycles: {}", config.max_cycles).expect("write to String never fails");
    if config.loop_forever {
        report.push_str("Loop:       forever\n");
    }
    report.push('\n');

    for node in &config.nodes {
        if let Some(ref spf) = node.system_prompt_file {
            if !spf.exists() {
                let msg = format!(
                    "Node '{}': system_prompt_file {:?} not found",
                    node.name, spf
                );
                warnings.push(msg.clone());
                writeln!(report, "  WARN: {}", msg).expect("write to String never fails");
            }
        }
    }

    if !available_models.is_empty() {
        report.push_str("Model checks:\n");
        for node in &config.nodes {
            let found = available_models.contains(&node.model)
                || available_models.iter().any(|m| {
                    m.strip_suffix(":latest")
                        .is_some_and(|base| base == node.model)
                })
                || available_models
                    .iter()
                    .any(|m| m.split(':').next().is_some_and(|base| base == node.model));
            model_checks.push((node.name.clone(), node.model.clone(), found));
            if found {
                writeln!(report, "  {:<16} {} — OK", node.name, node.model)
                    .expect("write to String never fails");
            } else {
                errors.push(format!(
                    "Node '{}': model '{}' not found",
                    node.name, node.model
                ));
                writeln!(report, "  {:<16} {} — NOT FOUND", node.name, node.model)
                    .expect("write to String never fails");
            }
        }
    }

    report.push('\n');
    if errors.is_empty() && warnings.is_empty() {
        report.push_str("Result: PASS — config is valid and all models available");
    } else if errors.is_empty() {
        write!(report, "Result: PASS with {} warning(s)", warnings.len())
            .expect("write to String never fails");
    } else {
        write!(
            report,
            "Result: FAIL — {} error(s), {} warning(s)",
            errors.len(),
            warnings.len()
        )
        .expect("write to String never fails");
    }

    ChainValidationReport {
        chain_name: config.name.clone(),
        node_count: config.nodes.len(),
        max_cycles: config.max_cycles,
        loop_forever: config.loop_forever,
        errors,
        warnings,
        model_checks,
        report,
    }
}

pub fn parse_chain_config(path: &Path) -> Result<ChainConfig> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read chain config at {:?}", path))?;
    let config: ChainConfig = serde_yaml::from_str(&content)
        .with_context(|| format!("Failed to parse YAML in chain config at {:?}", path))?;
    Ok(config)
}

pub fn load_chain_config(path: &Path) -> Result<ChainConfig> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("chain config not found: {}", path.display()))?;

    // Chain configs are read from the operator-global config dir so a
    // single chain definition (e.g. `~/.dm/chains/glue.yaml`) works
    // across every spawned host project. Project-local YAML files are
    // still picked up via the cwd entry.
    let global_dm = crate::config::current_global_config_dir();
    let allowed_dirs: Vec<PathBuf> = [
        global_dm.as_ref().map(|d| d.join("chains")),
        global_dm,
        std::env::current_dir().ok(),
    ]
    .into_iter()
    .flatten()
    .collect();

    let in_allowed = allowed_dirs.iter().any(|dir| {
        dir.canonicalize()
            .map(|d| canonical.starts_with(&d))
            .unwrap_or(false)
    });

    if !in_allowed {
        anyhow::bail!(
            "chain config must be in ~/.dm/chains/, ~/.dm/, or the current directory, got: {}",
            path.display()
        );
    }

    parse_chain_config(&canonical)
}

/// Load a saved `ChainState` from a workspace directory and build an
/// `OrchestrationConfig` that will resume the chain from the saved cycle.
pub fn resume_config_from_workspace(workspace: &Path) -> Result<OrchestrationConfig> {
    let state_path = workspace.join("chain_state.json");
    let state = ChainState::load(&state_path)
        .with_context(|| format!("Failed to load chain state from {:?}", state_path))?;
    let chain = state.config.clone();
    let chain_id = state.chain_id.clone();
    Ok(OrchestrationConfig {
        chain,
        chain_id,
        retry: crate::conversation::RetrySettings::default(),
        resume_state: Some(state),
    })
}

/// Validate a parsed `ChainConfig` for structural correctness.
///
/// Checks:
/// - Chain name is non-empty
/// - `max_cycles` is at least 1 (unless `loop_forever` is set)
/// - No duplicate node IDs
/// - No empty or whitespace-only node IDs
/// - Every `input_from` value refers to an existing node name
pub fn validate_chain_config(chain: &ChainConfig) -> Result<Vec<String>> {
    use std::collections::HashSet;

    // Reject empty chain name
    if chain.name.trim().is_empty() {
        anyhow::bail!("Chain name must not be empty");
    }

    // Validate workspace path
    let ws = &chain.workspace;
    if ws.as_os_str().is_empty() {
        anyhow::bail!("Chain '{}': workspace path must not be empty", chain.name);
    }
    for component in ws.components() {
        if let std::path::Component::ParentDir = component {
            anyhow::bail!(
                "Chain '{}': workspace path '{}' contains '..' components",
                chain.name,
                ws.display()
            );
        }
    }
    const DANGEROUS_ROOTS: &[&str] = &[
        "/", "/etc", "/usr", "/bin", "/sbin", "/var", "/root", "/home",
    ];
    let ws_str = ws.to_string_lossy();
    if DANGEROUS_ROOTS.iter().any(|r| ws_str == *r) {
        anyhow::bail!(
            "Chain '{}': workspace path '{}' is a dangerous system directory",
            chain.name,
            ws.display()
        );
    }

    // Reject max_cycles: 0 unless loop_forever is set
    if chain.max_cycles == 0 && !chain.loop_forever {
        anyhow::bail!(
            "Chain '{}': max_cycles must be at least 1 (or set loop_forever: true)",
            chain.name,
        );
    }

    if chain.max_total_turns == 0 {
        anyhow::bail!("Chain '{}': max_total_turns must be at least 1", chain.name,);
    }

    let mut seen: HashSet<&str> = HashSet::new();
    for node in &chain.nodes {
        // Reject empty or whitespace-only node IDs
        if node.id.trim().is_empty() {
            anyhow::bail!(
                "Chain '{}': node id must not be empty or whitespace-only",
                chain.name,
            );
        }

        if node.name.trim().is_empty() {
            anyhow::bail!(
                "Chain '{}': node '{}' has an empty name",
                chain.name,
                node.id,
            );
        }

        if !is_safe_node_name(&node.id) {
            anyhow::bail!(
                "Chain '{}': node id '{}' contains unsafe characters (path separators, '..', or non-alphanumeric)",
                chain.name,
                node.id,
            );
        }

        if !is_safe_node_name(&node.name) {
            anyhow::bail!(
                "Chain '{}': node name '{}' contains unsafe characters (path separators, '..', or non-alphanumeric)",
                chain.name,
                node.name,
            );
        }

        if node.max_tool_turns == 0 {
            anyhow::bail!(
                "Chain '{}': node '{}' max_tool_turns must be at least 1",
                chain.name,
                node.id,
            );
        }

        if !seen.insert(node.id.as_str()) {
            anyhow::bail!("Chain '{}': duplicate node id '{}'", chain.name, node.id);
        }
    }

    let mut seen_names: HashSet<&str> = HashSet::new();
    for node in &chain.nodes {
        if !seen_names.insert(node.name.as_str()) {
            anyhow::bail!(
                "Chain '{}': duplicate node name '{}'",
                chain.name,
                node.name
            );
        }
    }

    let all_names: HashSet<&str> = seen_names;
    for node in &chain.nodes {
        if let Some(ref src) = node.input_from {
            if !all_names.contains(src.as_str()) {
                anyhow::bail!(
                    "Chain '{}': node '{}' has input_from '{}' which does not match any node name",
                    chain.name,
                    node.id,
                    src
                );
            }
        }
    }

    let mut warnings = Vec::new();

    if !chain.nodes.is_empty()
        && chain.nodes.len() > 1
        && chain.nodes.iter().all(|n| n.input_from.is_some())
    {
        warnings.push(format!(
            "Chain '{}': all nodes have input_from set. The first node ('{}') will start without input on cycle 1.",
            chain.name,
            chain.nodes[0].name,
        ));
    }

    if let Some(ref directive_path) = chain.directive {
        if !directive_path.exists() {
            warnings.push(format!(
                "Chain '{}': directive {:?} does not exist",
                chain.name, directive_path
            ));
        }
    }

    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrate::types::{ChainConfig, ChainNodeConfig};

    fn make_node(id: &str, input_from: Option<&str>) -> ChainNodeConfig {
        ChainNodeConfig {
            id: id.to_string(),
            name: id.to_string(),
            role: id.to_string(),
            model: "test-model".to_string(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: input_from.map(str::to_string),
            max_retries: 1,
            timeout_secs: 3600,
            max_tool_turns: 200,
        }
    }

    fn make_chain(nodes: Vec<ChainNodeConfig>) -> ChainConfig {
        ChainConfig {
            name: "test".to_string(),
            description: None,
            nodes,
            max_cycles: 3,
            max_total_turns: 60,
            workspace: std::path::PathBuf::from("/tmp/test-chain"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        }
    }

    #[test]
    fn test_valid_chain_no_input_from_passes() {
        let chain = make_chain(vec![
            make_node("planner", None),
            make_node("builder", None),
            make_node("tester", None),
        ]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn test_cyclic_chain_all_input_from_passes() {
        // planner→builder→tester→planner — cyclic chain is valid; the runner
        // handles cycle 1 gracefully (first node starts without input).
        let chain = make_chain(vec![
            make_node("planner", Some("tester")),
            make_node("builder", Some("planner")),
            make_node("tester", Some("builder")),
        ]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn test_chain_with_entry_point_and_input_from_passes() {
        // First node has no input_from (entry point), rest do — valid
        let chain = make_chain(vec![
            make_node("planner", None),
            make_node("builder", Some("planner")),
            make_node("tester", Some("builder")),
        ]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn test_duplicate_node_ids_rejected() {
        let chain = make_chain(vec![
            make_node("planner", None),
            make_node("planner", None), // duplicate
        ]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("duplicate node id"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_unknown_input_from_rejected() {
        let chain = make_chain(vec![
            make_node("planner", None),
            make_node("builder", Some("ghost")), // "ghost" doesn't exist as a name
        ]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("does not match any node name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn input_from_resolves_by_name_not_id() {
        // Node with id="planner_v2" but name="planner" — input_from should
        // reference the name, not the id.
        let mut node_a = make_node("planner_v2", None);
        node_a.name = "Planner".into();
        let mut node_b = make_node("builder_v2", Some("Planner")); // references name
        node_b.name = "Builder".into();
        let chain = make_chain(vec![node_a, node_b]);
        assert!(
            validate_chain_config(&chain).is_ok(),
            "input_from referencing a name should pass"
        );
    }

    #[test]
    fn input_from_referencing_id_when_name_differs_rejected() {
        // input_from uses the id ("planner_v2") but only the name ("Planner") exists
        let mut node_a = make_node("planner_v2", None);
        node_a.name = "Planner".into();
        let mut node_b = make_node("builder_v2", Some("planner_v2")); // references id, not name
        node_b.name = "Builder".into();
        let chain = make_chain(vec![node_a, node_b]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("does not match any node name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_all_nodes_have_input_from_passes() {
        // Cyclic chains where every node has input_from are now valid
        let chain = make_chain(vec![
            make_node("a", Some("c")),
            make_node("b", Some("a")),
            make_node("c", Some("b")),
        ]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn validate_mixed_input_from_passes() {
        // Some nodes have input_from, some don't — valid
        let chain = make_chain(vec![
            make_node("planner", None),
            make_node("builder", Some("planner")),
            make_node("tester", Some("builder")),
        ]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn duplicate_node_names_rejected() {
        let mut node_a = make_node("id_a", None);
        node_a.name = "Builder".into();
        let mut node_b = make_node("id_b", None);
        node_b.name = "Builder".into();
        let chain = make_chain(vec![node_a, node_b]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("duplicate node name"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_node_names_different_ids_rejected() {
        let mut node_a = make_node("planner_v1", None);
        node_a.name = "Planner".into();
        let mut node_b = make_node("planner_v2", None);
        node_b.name = "Planner".into();
        let chain = make_chain(vec![node_a, node_b]);
        assert!(validate_chain_config(&chain).is_err());
    }

    #[test]
    fn validate_workspace_empty_rejected() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.workspace = std::path::PathBuf::from("");
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(err.to_string().contains("workspace"), "err: {err}");
        assert!(err.to_string().contains("empty"), "err: {err}");
    }

    #[test]
    fn validate_workspace_parent_dir_rejected() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.workspace = std::path::PathBuf::from("/tmp/chains/../../../etc/shadow");
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(err.to_string().contains(".."), "err: {err}");
    }

    #[test]
    fn validate_workspace_root_rejected() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.workspace = std::path::PathBuf::from("/");
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(err.to_string().contains("dangerous"), "err: {err}");
    }

    #[test]
    fn validate_workspace_system_dir_rejected() {
        for dir in &["/etc", "/usr", "/bin", "/sbin", "/var", "/root", "/home"] {
            let mut chain = make_chain(vec![make_node("a", None)]);
            chain.workspace = std::path::PathBuf::from(dir);
            assert!(
                validate_chain_config(&chain).is_err(),
                "'{}' should be rejected as workspace",
                dir
            );
        }
    }

    #[test]
    fn validate_workspace_normal_path_accepted() {
        let chain = make_chain(vec![make_node("a", None)]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn unique_node_names_pass() {
        let mut node_a = make_node("id_a", None);
        node_a.name = "Planner".into();
        let mut node_b = make_node("id_b", None);
        node_b.name = "Builder".into();
        let chain = make_chain(vec![node_a, node_b]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn load_chain_config_roundtrip() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cwd = std::env::current_dir().unwrap();
        let config_path = cwd.join(".dm-test-chain-roundtrip.yaml");
        let yaml = r#"
name: ci-chain
nodes:
  - id: planner
    name: Planner
    role: Plan the work
    model: llama3
  - id: builder
    name: Builder
    role: Build it
    model: llama3
    input_from: planner
max_cycles: 3
max_total_turns: 60
workspace: /tmp/ci
skip_permissions_warning: true
"#;
        std::fs::write(&config_path, yaml).unwrap();
        let result = load_chain_config(&config_path);
        let _ = std::fs::remove_file(&config_path);
        let config = result.unwrap();
        assert_eq!(config.name, "ci-chain");
        assert_eq!(config.nodes.len(), 2);
        assert_eq!(config.nodes[0].id, "planner");
        assert_eq!(config.nodes[1].input_from.as_deref(), Some("planner"));
        assert_eq!(config.max_cycles, 3);
    }

    #[test]
    fn load_chain_config_missing_file_errors() {
        let result = load_chain_config(std::path::Path::new("/tmp/dm_no_such_chain.yaml"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found"), "error: {msg}");
    }

    #[test]
    fn validate_chain_empty_nodes_is_ok() {
        let chain = make_chain(vec![]);
        assert!(
            validate_chain_config(&chain).is_ok(),
            "empty node list should pass validation"
        );
    }

    #[test]
    fn validate_single_node_no_input_from_passes() {
        let chain = make_chain(vec![make_node("solo", None)]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn validate_self_referencing_input_from_is_allowed() {
        // A node referencing itself via input_from is structurally valid
        // (the runner handles cycle semantics, not the validator)
        let chain = make_chain(vec![make_node("node", Some("node"))]);
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn load_chain_config_invalid_yaml_errors() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cwd = std::env::current_dir().unwrap();
        let path = cwd.join(".dm-test-bad.yaml");
        std::fs::write(&path, "not: valid: yaml: {{{").unwrap();
        let result = load_chain_config(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err(), "invalid YAML should produce an error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Failed to parse YAML"), "error: {msg}");
    }

    // ── Degenerate config validation tests ──────────────────────────────

    #[test]
    fn validate_empty_chain_name_rejected() {
        let chain = ChainConfig {
            name: String::new(),
            ..make_chain(vec![make_node("a", None)])
        };
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_whitespace_chain_name_rejected() {
        let chain = ChainConfig {
            name: "   ".into(),
            ..make_chain(vec![make_node("a", None)])
        };
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("name must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_zero_max_cycles_rejected() {
        let chain = ChainConfig {
            max_cycles: 0,
            ..make_chain(vec![make_node("a", None)])
        };
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("max_cycles must be at least 1"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_zero_max_cycles_ok_with_loop_forever() {
        let chain = ChainConfig {
            max_cycles: 0,
            loop_forever: true,
            directive: None,
            ..make_chain(vec![make_node("a", None)])
        };
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn validate_empty_node_id_rejected() {
        let chain = make_chain(vec![make_node("", None)]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("node id must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_whitespace_node_id_rejected() {
        let mut node = make_node("placeholder", None);
        node.id = "   ".into();
        let chain = make_chain(vec![node]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(
            err.to_string().contains("node id must not be empty"),
            "error: {err}"
        );
    }

    #[test]
    fn validate_empty_node_name_rejected() {
        let mut node = make_node("valid_id", None);
        node.name = String::new();
        let chain = make_chain(vec![node]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(err.to_string().contains("empty name"), "error: {err}");
    }

    #[test]
    fn validate_whitespace_node_name_rejected() {
        let mut node = make_node("valid_id", None);
        node.name = "   ".into();
        let chain = make_chain(vec![node]);
        let err = validate_chain_config(&chain).unwrap_err();
        assert!(err.to_string().contains("empty name"), "error: {err}");
    }

    #[test]
    fn validate_missing_directive_still_passes() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.directive = Some(std::path::PathBuf::from("/tmp/dm_nonexistent_directive.md"));
        // Validation should succeed (warning only, not error)
        assert!(validate_chain_config(&chain).is_ok());
    }

    #[test]
    fn validate_directive_none_passes() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.directive = None;
        assert!(validate_chain_config(&chain).is_ok());
    }

    // ── Chain management function tests ─────────────────────────────────
    //
    // These tests mutate the global CHAIN_WORKSPACE, so they must not
    // run concurrently with each other.  We use a shared mutex to
    // serialize access.

    static MGMT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper: set `CHAIN_WORKSPACE` to a given path for testing.
    fn set_test_workspace(path: &std::path::Path) {
        if let Ok(mut w) = CHAIN_WORKSPACE.lock() {
            *w = Some(path.to_path_buf());
        }
    }

    /// Helper: clear `CHAIN_WORKSPACE`.
    fn clear_test_workspace() {
        if let Ok(mut w) = CHAIN_WORKSPACE.lock() {
            *w = None;
        }
    }

    /// Helper: write a valid `chain_state.json` into workspace.
    fn seed_chain_state(workspace: &std::path::Path) {
        let config = make_chain(vec![make_node("n1", None), make_node("n2", Some("n1"))]);
        let state = ChainState::new(config, "test-chain-id".into());
        state.save(workspace).unwrap();
    }

    #[test]
    fn stop_chain_creates_sentinel() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        stop_chain().unwrap();
        assert!(dir.path().join(".dm-stop").exists());
        clear_test_workspace();
    }

    #[test]
    fn stop_chain_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = stop_chain();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn pause_chain_creates_sentinel() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        pause_chain().unwrap();
        assert!(dir.path().join(".dm-pause").exists());
        clear_test_workspace();
    }

    #[test]
    fn pause_chain_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = pause_chain();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn resume_chain_removes_pause_sentinel() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        std::fs::write(dir.path().join(".dm-pause"), b"").unwrap();
        assert!(dir.path().join(".dm-pause").exists());
        resume_chain().unwrap();
        assert!(!dir.path().join(".dm-pause").exists());
        clear_test_workspace();
    }

    #[test]
    fn resume_chain_ok_when_not_paused() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        assert!(resume_chain().is_ok());
        clear_test_workspace();
    }

    #[test]
    fn resume_chain_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = resume_chain();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn chain_add_node_enqueues_pending_addition() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());

        let new_node = make_node("n3", Some("n2"));
        chain_add_node(new_node).unwrap();

        let state = ChainState::load(&dir.path().join("chain_state.json")).unwrap();
        assert_eq!(state.pending_additions.len(), 1);
        assert_eq!(state.pending_additions[0].id, "n3");
        clear_test_workspace();
    }

    #[test]
    fn chain_add_node_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = chain_add_node(make_node("x", None));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn chain_remove_node_enqueues_pending_removal() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());

        chain_remove_node("n1").unwrap();

        let state = ChainState::load(&dir.path().join("chain_state.json")).unwrap();
        assert_eq!(state.pending_removals.len(), 1);
        assert_eq!(state.pending_removals[0], "n1");
        clear_test_workspace();
    }

    #[test]
    fn chain_remove_node_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = chain_remove_node("x");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn chain_model_enqueues_pending_swap() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());

        chain_model("n1", "llama3:70b").unwrap();

        let state = ChainState::load(&dir.path().join("chain_state.json")).unwrap();
        assert_eq!(
            state.pending_model_swaps.get("n1").map(String::as_str),
            Some("llama3:70b")
        );
        clear_test_workspace();
    }

    #[test]
    fn chain_model_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = chain_model("x", "model");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn chain_talk_creates_talk_file() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());

        chain_talk("builder", "Please focus on error handling").unwrap();

        let talk_path = dir.path().join("talk-builder.md");
        assert!(talk_path.exists());
        let content = std::fs::read_to_string(&talk_path).unwrap();
        assert_eq!(content, "Please focus on error handling");
        clear_test_workspace();
    }

    #[test]
    fn chain_talk_errors_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = chain_talk("x", "msg");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No chain is running"));
    }

    #[test]
    fn chain_status_returns_none_when_no_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        assert!(chain_status().is_none());
    }

    #[test]
    fn chain_status_returns_state_when_available() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());

        let status = chain_status();
        assert!(status.is_some());
        let state = status.unwrap();
        assert_eq!(state.chain_id, "test-chain-id");
        assert_eq!(state.config.nodes.len(), 2);
        clear_test_workspace();
    }

    #[test]
    fn clear_chain_resets_globals() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());
        assert!(
            chain_status().is_some(),
            "chain should be visible before clear"
        );

        clear_chain();
        assert!(
            chain_status().is_none(),
            "chain_status should be None after clear_chain()"
        );
    }

    #[test]
    fn clear_chain_is_idempotent() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        clear_chain();
        clear_chain();
        assert!(chain_status().is_none());
    }

    #[test]
    fn chain_state_save_load_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let workspace = dir.path();

        let config = make_chain(vec![make_node("n1", None), make_node("n2", Some("n1"))]);
        let mut state = crate::orchestrate::types::ChainState::new(config, "test-id".into());
        state.current_cycle = 2;
        state.turns_used = 7;
        state
            .node_outputs
            .insert("n1".into(), "output from n1".into());
        state.last_abort_reason = Some("timeout".into());

        state.save(workspace).unwrap();

        let loaded =
            crate::orchestrate::types::ChainState::load(&workspace.join("chain_state.json"))
                .unwrap();
        assert_eq!(loaded.chain_id, "test-id");
        assert_eq!(loaded.current_cycle, 2);
        assert_eq!(loaded.turns_used, 7);
        assert_eq!(
            loaded.node_outputs.get("n1").map(String::as_str),
            Some("output from n1")
        );
        assert_eq!(loaded.last_abort_reason.as_deref(), Some("timeout"));
        assert_eq!(loaded.config.nodes.len(), 2);
    }

    #[test]
    fn resume_config_from_workspace_loads_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = make_chain(vec![make_node("n1", None), make_node("n2", Some("n1"))]);
        let mut state = ChainState::new(config, "resume-test".into());
        state.current_cycle = 5;
        state.turns_used = 12;
        state
            .node_outputs
            .insert("n1".into(), "prior output".into());
        state.save(dir.path()).unwrap();

        let orch = resume_config_from_workspace(dir.path()).unwrap();
        assert_eq!(orch.chain_id, "resume-test");
        assert_eq!(orch.chain.nodes.len(), 2);
        let resumed = orch.resume_state.unwrap();
        assert_eq!(resumed.current_cycle, 5);
        assert_eq!(resumed.turns_used, 12);
        assert_eq!(
            resumed.node_outputs.get("n1").map(String::as_str),
            Some("prior output")
        );
    }

    #[test]
    fn resume_config_from_nonexistent_workspace_fails() {
        let result =
            resume_config_from_workspace(std::path::Path::new("/tmp/dm_no_such_workspace"));
        assert!(result.is_err());
    }

    // ── chain_log tests ───────────────────────────────────────────────────

    #[test]
    fn chain_log_no_workspace_returns_error() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_test_workspace();
        let result = chain_log(None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No chain"));
    }

    #[test]
    fn chain_log_empty_workspace_returns_empty() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        let entries = chain_log(None).unwrap();
        assert!(entries.is_empty());
        clear_test_workspace();
    }

    #[test]
    fn chain_log_reads_artifacts() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("cycle-01-planner.md"), "the plan").unwrap();
        std::fs::write(dir.path().join("cycle-01-builder.md"), "the build").unwrap();
        set_test_workspace(dir.path());

        let entries = chain_log(None).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].filename, "cycle-01-builder.md");
        assert_eq!(entries[0].node, "builder");
        assert_eq!(entries[0].cycle, 1);
        assert_eq!(entries[0].content, "the build");
        assert_eq!(entries[1].filename, "cycle-01-planner.md");
        assert_eq!(entries[1].content, "the plan");
        clear_test_workspace();
    }

    #[test]
    fn chain_log_filters_by_cycle() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("cycle-01-planner.md"), "plan1").unwrap();
        std::fs::write(dir.path().join("cycle-02-planner.md"), "plan2").unwrap();
        set_test_workspace(dir.path());

        let entries = chain_log(Some(2)).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].cycle, 2);
        assert_eq!(entries[0].content, "plan2");
        clear_test_workspace();
    }

    #[test]
    fn chain_log_truncates_large_content() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        let big = "x".repeat(5000);
        std::fs::write(dir.path().join("cycle-01-big.md"), &big).unwrap();
        set_test_workspace(dir.path());

        let entries = chain_log(None).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].content.len() < 4100, "should be truncated");
        assert!(entries[0].content.ends_with("…[truncated]"));
        clear_test_workspace();
    }

    #[test]
    fn chain_log_from_workspace_reads_directly() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("cycle-03-tester.md"), "test output").unwrap();
        std::fs::write(dir.path().join("not-an-artifact.txt"), "ignore").unwrap();

        let entries = chain_log_from_workspace(dir.path(), None).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].node, "tester");
        assert_eq!(entries[0].cycle, 3);
    }

    // ── validate_chain_config_detailed tests ──────────────────────────────

    #[test]
    fn validate_detailed_valid_config_passes() {
        let chain = make_chain(vec![make_node("a", None), make_node("b", Some("a"))]);
        let report = validate_chain_config_detailed(&chain, &[]);
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(report.report.contains("PASS"));
        assert_eq!(report.node_count, 2);
    }

    #[test]
    fn validate_detailed_structure_error() {
        let chain = ChainConfig {
            name: String::new(),
            ..make_chain(vec![make_node("a", None)])
        };
        let report = validate_chain_config_detailed(&chain, &[]);
        assert!(!report.errors.is_empty());
        assert!(report.report.contains("FAIL"));
    }

    #[test]
    fn validate_detailed_missing_model() {
        let chain = make_chain(vec![make_node("a", None)]);
        let models = vec!["llama3:latest".to_string()];
        let report = validate_chain_config_detailed(&chain, &models);
        assert!(!report.errors.is_empty(), "should flag missing model");
        assert!(report.report.contains("NOT FOUND"));
    }

    #[test]
    fn validate_detailed_model_with_latest_suffix() {
        let chain = make_chain(vec![make_node("a", None)]);
        // Node uses "test-model", available list has "test-model:latest"
        let models = vec!["test-model:latest".to_string()];
        let report = validate_chain_config_detailed(&chain, &models);
        assert!(
            report.errors.is_empty(),
            "test-model should match test-model:latest: {:?}",
            report.errors
        );
        assert!(report.model_checks[0].2, "model should be found");
    }

    #[test]
    fn validate_detailed_missing_system_prompt_file() {
        let mut node = make_node("a", None);
        node.system_prompt_file = Some(std::path::PathBuf::from("/tmp/dm_nonexistent_spf.md"));
        let chain = make_chain(vec![node]);
        let report = validate_chain_config_detailed(&chain, &[]);
        assert!(
            !report.warnings.is_empty(),
            "should warn about missing file"
        );
        assert!(report.report.contains("WARN"));
        // Still passes (warnings don't cause failure)
        assert!(report.errors.is_empty());
    }

    #[test]
    fn safe_node_name_rejects_path_traversal() {
        assert!(!is_safe_node_name("../foo"));
        assert!(!is_safe_node_name("foo/bar"));
        assert!(!is_safe_node_name("foo\\bar"));
        assert!(!is_safe_node_name(".."));
        assert!(!is_safe_node_name("a..b"));
    }

    #[test]
    fn safe_node_name_rejects_null_and_empty() {
        assert!(!is_safe_node_name("foo\0bar"));
        assert!(!is_safe_node_name(""));
        assert!(!is_safe_node_name("   "));
    }

    #[test]
    fn safe_node_name_accepts_valid() {
        assert!(is_safe_node_name("planner"));
        assert!(is_safe_node_name("my-node"));
        assert!(is_safe_node_name("node_1"));
        assert!(is_safe_node_name("v2.0"));
        assert!(is_safe_node_name("Builder"));
    }

    #[test]
    fn safe_node_name_rejects_special_chars() {
        assert!(!is_safe_node_name("node name"));
        assert!(!is_safe_node_name("node@1"));
        assert!(!is_safe_node_name("node!"));
    }

    #[test]
    fn chain_talk_rejects_unsafe_name() {
        let result = chain_talk("../etc/passwd", "pwned");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Invalid node name"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn validate_config_rejects_unsafe_node_id() {
        let mut node = make_node("a", None);
        node.id = "../evil".to_string();
        node.name = "safe".to_string();
        let chain = make_chain(vec![node]);
        let result = validate_chain_config(&chain);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("unsafe"), "unexpected error: {}", msg);
    }

    #[test]
    fn validate_config_rejects_unsafe_node_name() {
        let mut node = make_node("a", None);
        node.id = "safe".to_string();
        node.name = "foo/bar".to_string();
        let chain = make_chain(vec![node]);
        let result = validate_chain_config(&chain);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("unsafe"), "unexpected error: {}", msg);
    }

    #[test]
    fn chain_add_node_rejects_unsafe_id() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());
        let mut node = make_node("valid", None);
        node.id = "../evil".to_string();
        let result = chain_add_node(node);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid node name"));
        clear_test_workspace();
    }

    #[test]
    fn chain_add_node_rejects_unsafe_name() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());
        let mut node = make_node("valid_id", None);
        node.name = "foo/bar".to_string();
        let result = chain_add_node(node);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid node name"));
        clear_test_workspace();
    }

    #[test]
    fn chain_remove_node_rejects_unsafe_name() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());
        let result = chain_remove_node("../evil");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid node name"));
        clear_test_workspace();
    }

    #[test]
    fn chain_model_rejects_unsafe_name() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let dir = tempfile::TempDir::new().unwrap();
        set_test_workspace(dir.path());
        seed_chain_state(dir.path());
        let result = chain_model("../evil", "llama3");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid node name"));
        clear_test_workspace();
    }

    #[test]
    fn chain_state_load_corrupt_json_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let state_path = dir.path().join("chain_state.json");
        std::fs::write(&state_path, "not valid json {{{").unwrap();
        let result = ChainState::load(&state_path);
        assert!(result.is_err());
    }

    #[test]
    fn validate_chain_warns_all_input_from() {
        let nodes = vec![
            make_node("planner", Some("tester")),
            make_node("builder", Some("planner")),
            make_node("tester", Some("builder")),
        ];
        let chain = make_chain(nodes);
        let warnings = validate_chain_config(&chain).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("all nodes have input_from"),
            "got: {}",
            warnings[0]
        );
    }

    #[test]
    fn validate_chain_warns_missing_directive() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.directive = Some(std::path::PathBuf::from("/nonexistent/directive.md"));
        let warnings = validate_chain_config(&chain).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("directive"), "got: {}", warnings[0]);
    }

    #[test]
    fn validate_chain_no_warnings_clean_config() {
        let nodes = vec![
            make_node("planner", None),
            make_node("builder", Some("planner")),
        ];
        let chain = make_chain(nodes);
        let warnings = validate_chain_config(&chain).unwrap();
        assert!(
            warnings.is_empty(),
            "expected no warnings, got: {:?}",
            warnings
        );
    }

    #[tokio::test]
    async fn abort_chain_clears_handle_and_workspace() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        let workspace = tempfile::TempDir::new().unwrap();
        let handle = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::from_secs(300)).await;
        });
        set_chain(handle, workspace.path().to_path_buf());
        assert!(CHAIN_WORKSPACE.lock().unwrap().is_some());

        abort_chain().unwrap();

        assert!(CHAIN_HANDLE.lock().unwrap().is_none());
        assert!(CHAIN_WORKSPACE.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn clear_chain_aborts_running_task() {
        {
            let _lock = MGMT_TEST_LOCK.lock().unwrap();
            let (_tx, mut rx) = tokio::sync::oneshot::channel::<()>();
            let handle = tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(300)).await;
            });
            let workspace = tempfile::TempDir::new().unwrap();
            set_chain(handle, workspace.path().to_path_buf());

            clear_chain();

            assert!(
                rx.try_recv().is_err(),
                "task should be aborted, sender dropped"
            );
            assert!(CHAIN_HANDLE.lock().unwrap().is_none());
        }
    }

    #[test]
    fn abort_chain_when_no_chain_still_succeeds() {
        let _lock = MGMT_TEST_LOCK.lock().unwrap();
        clear_chain();
        assert!(abort_chain().is_ok());
    }

    #[test]
    fn save_and_load_last_chain_pointer_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        save_last_chain_pointer(tmp.path());
        let loaded = load_last_chain_pointer();
        assert!(loaded.is_some(), "pointer should load after save");
        assert_eq!(loaded.unwrap(), tmp.path());
        clear_last_chain_pointer();
    }

    #[test]
    fn load_missing_pointer_returns_none() {
        clear_last_chain_pointer();
        let loaded = load_last_chain_pointer();
        assert!(
            loaded.is_none(),
            "should be None when no pointer file exists"
        );
    }

    #[test]
    fn clear_removes_pointer() {
        let tmp = tempfile::TempDir::new().unwrap();
        save_last_chain_pointer(tmp.path());
        clear_last_chain_pointer();
        let loaded = load_last_chain_pointer();
        assert!(loaded.is_none(), "pointer should be gone after clear");
    }

    #[test]
    fn load_pointer_returns_none_for_nonexistent_workspace() {
        let tmp = tempfile::TempDir::new().unwrap();
        let gone_path = tmp.path().to_path_buf();
        save_last_chain_pointer(&gone_path);
        drop(tmp);
        let loaded = load_last_chain_pointer();
        assert!(
            loaded.is_none(),
            "should return None when workspace dir no longer exists"
        );
        clear_last_chain_pointer();
    }

    #[test]
    fn load_chain_config_rejects_etc_path() {
        let result = load_chain_config(std::path::Path::new("/etc/passwd"));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be in") || msg.contains("not found"),
            "should reject /etc/passwd: {msg}"
        );
    }

    #[test]
    fn load_chain_config_rejects_parent_traversal() {
        let result = load_chain_config(std::path::Path::new("../../etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_rejects_zero_max_tool_turns() {
        let mut node = make_node("a", None);
        node.max_tool_turns = 0;
        let chain = make_chain(vec![node]);
        let result = validate_chain_config(&chain);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("max_tool_turns"), "error: {msg}");
    }

    #[test]
    fn validate_rejects_zero_max_total_turns() {
        let mut chain = make_chain(vec![make_node("a", None)]);
        chain.max_total_turns = 0;
        let result = validate_chain_config(&chain);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("max_total_turns"), "error: {msg}");
    }
}

pub fn format_chain_metrics(state: &ChainState) -> String {
    let mut out = String::new();
    write!(
        out,
        "Chain: {}  |  Cycles: {}  |  Total: {:.1}s\n\n",
        state.config.name, state.current_cycle, state.total_duration_secs
    )
    .expect("write to String never fails");
    writeln!(
        out,
        "{:<16} {:>5} {:>8} {:>8} {:>8} {:>5} {:>10} {:>10}",
        "Node", "Runs", "Avg(s)", "Min(s)", "Max(s)", "Fails", "PromptTok", "CompTok"
    )
    .expect("write to String never fails");
    writeln!(out, "{}", "-".repeat(80)).expect("write to String never fails");

    let mut seen = std::collections::HashSet::new();
    let mut node_names: Vec<String> = Vec::new();
    for node in &state.config.nodes {
        node_names.push(node.name.clone());
        seen.insert(node.name.clone());
    }
    for name in state.node_durations.keys() {
        if seen.insert(name.clone()) {
            node_names.push(name.clone());
        }
    }

    for name in &node_names {
        let durations = state.node_durations.get(name);
        let failures = state.node_failures.get(name).copied().unwrap_or(0);
        let prompt_tok = state.node_prompt_tokens.get(name).copied().unwrap_or(0);
        let completion_tok = state.node_completion_tokens.get(name).copied().unwrap_or(0);
        if let Some(durs) = durations {
            if durs.is_empty() {
                writeln!(
                    out,
                    "{:<16} {:>5} {:>8} {:>8} {:>8} {:>5} {:>10} {:>10}",
                    name, 0, "-", "-", "-", failures, prompt_tok, completion_tok
                )
                .expect("write to String never fails");
                continue;
            }
            let count = durs.len();
            let sum: f64 = durs.iter().sum();
            let avg = sum / count as f64;
            let min = durs.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = durs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            writeln!(
                out,
                "{:<16} {:>5} {:>8.1} {:>8.1} {:>8.1} {:>5} {:>10} {:>10}",
                name, count, avg, min, max, failures, prompt_tok, completion_tok
            )
            .expect("write to String never fails");
        } else {
            writeln!(
                out,
                "{:<16} {:>5} {:>8} {:>8} {:>8} {:>5} {:>10} {:>10}",
                name, 0, "-", "-", "-", failures, prompt_tok, completion_tok
            )
            .expect("write to String never fails");
        }
    }

    if node_names.is_empty() {
        out.push_str("No metrics recorded yet.\n");
    }

    out
}

pub fn generate_chain_template(name: &str, model: &str) -> String {
    format!(
        "\
# Chain config for {name}
# Edit this file, then run: /chain start {name}.chain.yaml
#
# Each node runs in order. The output of one node becomes the input
# of the next via input_from. The chain loops for max_cycles iterations.

name: {name}
description: \"{name} agent pipeline\"
max_cycles: 3
max_total_turns: 60
workspace: .dm-chain-{name}
skip_permissions_warning: false
# loop_forever: false  # uncomment to run until manually stopped
# directive: ./directive.md  # shared context prepended to every node's prompt

nodes:
  - id: planner
    name: Planner
    role: >-
      You are the Planner. Read the user's request and break it into
      concrete steps. Output a numbered plan. End with SIGNAL: CONTINUE.
    model: {model}
    # timeout_secs: 3600  # max seconds per node (0 = no limit)
    # max_tool_turns: 200  # max tool rounds per node invocation
    # system_prompt_file: planner.md  # optional: load prompt from file

  - id: builder
    name: Builder
    role: >-
      You are the Builder. Implement the plan from the Planner using
      the available tools. When done, end with SIGNAL: CONTINUE.
    model: {model}
    input_from: Planner
"
    )
}

pub struct OrchestrationConfig {
    pub chain: ChainConfig,
    pub chain_id: String,
    pub retry: crate::conversation::RetrySettings,
    /// When set, resume from this saved state instead of starting fresh.
    pub resume_state: Option<ChainState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    Pass(String),
    Fail {
        issues: Vec<String>,
        suggestions: Vec<String>,
    },
}
