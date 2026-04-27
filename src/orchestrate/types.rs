use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    pub name: String,
    /// Optional human-readable description of what this chain does.
    #[serde(default)]
    pub description: Option<String>,
    pub nodes: Vec<ChainNodeConfig>,
    pub max_cycles: usize,
    pub max_total_turns: usize,
    pub workspace: PathBuf,
    pub skip_permissions_warning: bool,
    /// When true, the chain loops indefinitely until a STOP signal or `.dm-stop` sentinel.
    /// `max_cycles` is ignored when this is set.
    #[serde(default)]
    pub loop_forever: bool,
    /// Optional path to a shared directive file whose contents are prepended to every node's prompt.
    #[serde(default)]
    pub directive: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainNodeConfig {
    pub id: String,
    pub name: String,
    pub role: String,
    pub model: String,
    /// Optional human-readable description (informational; not sent to the model).
    #[serde(default)]
    pub description: Option<String>,
    pub system_prompt_override: Option<String>,
    /// Path to a markdown file whose contents are used as the node's system prompt.
    /// Takes precedence over `system_prompt_override` when both are set.
    #[serde(default)]
    pub system_prompt_file: Option<PathBuf>,
    /// The ID of the node this node receives input from.
    /// If None, it's the entry point.
    pub input_from: Option<String>,
    /// How many times to retry this node on transient failure. Defaults to 1.
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,
    /// Maximum seconds a node may run before being timed out.
    /// Defaults to 3600 (1 hour). Set to 0 to disable timeout.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    /// Maximum tool rounds per node invocation. Defaults to 200.
    /// Higher values let complex nodes (builders) use more tools per turn.
    #[serde(default = "default_max_tool_turns")]
    pub max_tool_turns: usize,
}

fn default_max_retries() -> usize {
    1
}

fn default_timeout_secs() -> u64 {
    3600
}

fn default_max_tool_turns() -> usize {
    200
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlSignal {
    Continue,
    Stop,
    Retry,
    Escalate,
}

pub const STATUS_PENDING: &str = "pending";
pub const STATUS_RUNNING: &str = "running";
pub const STATUS_COMPLETED: &str = "completed";
pub const STATUS_FAILED_PREFIX: &str = "failed: ";
pub const STATUS_PAUSED: &str = "paused";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainNodeStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Paused,
}

impl ChainNodeStatus {
    pub fn from_status_str(s: &str) -> Option<Self> {
        match s {
            STATUS_PENDING => Some(Self::Pending),
            STATUS_RUNNING => Some(Self::Running),
            STATUS_COMPLETED => Some(Self::Completed),
            STATUS_PAUSED => Some(Self::Paused),
            s if s.starts_with(STATUS_FAILED_PREFIX) => {
                Some(Self::Failed(s[STATUS_FAILED_PREFIX.len()..].to_string()))
            }
            _ => None,
        }
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, ChainNodeStatus::Failed(_))
    }
}

impl fmt::Display for ChainNodeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainNodeStatus::Pending => write!(f, "{}", STATUS_PENDING),
            ChainNodeStatus::Running => write!(f, "{}", STATUS_RUNNING),
            ChainNodeStatus::Completed => write!(f, "{}", STATUS_COMPLETED),
            ChainNodeStatus::Failed(reason) => write!(f, "{}{}", STATUS_FAILED_PREFIX, reason),
            ChainNodeStatus::Paused => write!(f, "{}", STATUS_PAUSED),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainState {
    pub chain_id: String,
    pub config: ChainConfig,
    pub active_node_index: Option<usize>,
    pub node_statuses: HashMap<String, ChainNodeStatus>,
    pub current_cycle: usize,
    pub turns_used: usize,
    pub node_outputs: HashMap<String, String>,
    pub last_signal: Option<ControlSignal>,
    pub last_updated: DateTime<Utc>,
    /// Reason the most recent cycle was aborted, if any.
    #[serde(default)]
    pub last_abort_reason: Option<String>,
    /// Nodes queued for addition by /chain add — applied at the start of the next cycle.
    #[serde(default)]
    pub pending_additions: Vec<ChainNodeConfig>,
    /// Node names queued for removal by /chain remove — applied at the start of the next cycle.
    #[serde(default)]
    pub pending_removals: Vec<String>,
    /// Pending model swaps written by /chain model — node name → new model string.
    /// Applied and cleared at the start of the next cycle.
    #[serde(default)]
    pub pending_model_swaps: std::collections::HashMap<String, String>,
    /// Per-node execution durations in seconds, accumulated across cycles.
    #[serde(default)]
    pub node_durations: HashMap<String, Vec<f64>>,
    /// Per-node failure count, accumulated across cycles.
    #[serde(default)]
    pub node_failures: HashMap<String, usize>,
    /// Total wall-clock seconds the chain has been running.
    #[serde(default)]
    pub total_duration_secs: f64,
    /// Per-node accumulated prompt tokens (input context).
    #[serde(default)]
    pub node_prompt_tokens: HashMap<String, u64>,
    /// Per-node accumulated completion tokens (model output).
    #[serde(default)]
    pub node_completion_tokens: HashMap<String, u64>,
}

impl ChainConfig {
    /// Resolve model aliases in all node configs and return a report of what changed.
    /// Each entry is `(node_name, original_model, resolved_model)`.
    /// Call this after loading a chain config to ensure node models
    /// are real Ollama model names, not alias shorthand.
    pub fn resolve_aliases_with_report(
        &mut self,
        aliases: &std::collections::HashMap<String, String>,
    ) -> Vec<(String, String, String)> {
        if aliases.is_empty() {
            return Vec::new();
        }
        let mut report = Vec::new();
        for node in &mut self.nodes {
            if let Some(resolved) = aliases.get(&node.model) {
                report.push((node.name.clone(), node.model.clone(), resolved.clone()));
                node.model = resolved.clone();
            }
        }
        report
    }

    /// Resolve model aliases in all node configs (discards the report).
    pub fn resolve_aliases(&mut self, aliases: &std::collections::HashMap<String, String>) {
        self.resolve_aliases_with_report(aliases);
    }
}

impl ChainState {
    pub fn new(chain_config: ChainConfig, chain_id: String) -> Self {
        let mut node_statuses = HashMap::new();
        for node in &chain_config.nodes {
            node_statuses.insert(node.name.clone(), ChainNodeStatus::Pending);
        }
        Self {
            chain_id,
            config: chain_config,
            active_node_index: None,
            node_statuses,
            current_cycle: 0,
            turns_used: 0,
            node_outputs: HashMap::new(),
            last_signal: None,
            last_updated: Utc::now(),
            last_abort_reason: None,
            pending_additions: Vec::new(),
            pending_removals: Vec::new(),
            pending_model_swaps: std::collections::HashMap::new(),
            node_durations: HashMap::new(),
            node_failures: HashMap::new(),
            total_duration_secs: 0.0,
            node_prompt_tokens: HashMap::new(),
            node_completion_tokens: HashMap::new(),
        }
    }

    pub fn save(&self, workspace: &std::path::Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(workspace)?;
        let state_path = workspace.join("chain_state.json");
        let tmp_path = state_path.with_extension("json.tmp");
        let content = serde_json::to_string(self)?;
        std::fs::write(&tmp_path, &content)?;
        std::fs::rename(&tmp_path, &state_path)?;
        Ok(())
    }

    pub fn load(state_path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(state_path)?;
        let state: Self = serde_json::from_str(&content)?;
        Ok(state)
    }

    /// Returns true if any node in this state is currently in a Failed status.
    pub fn has_failed_nodes(&self) -> bool {
        self.node_statuses.values().any(|s| s.is_failed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_status_display_pending() {
        assert_eq!(ChainNodeStatus::Pending.to_string(), "pending");
    }

    #[test]
    fn node_status_display_running() {
        assert_eq!(ChainNodeStatus::Running.to_string(), "running");
    }

    #[test]
    fn node_status_display_completed() {
        assert_eq!(ChainNodeStatus::Completed.to_string(), "completed");
    }

    #[test]
    fn node_status_display_failed() {
        let s = ChainNodeStatus::Failed("connection refused".into()).to_string();
        assert!(s.starts_with("failed: "));
        assert!(s.contains("connection refused"));
    }

    #[test]
    fn node_status_display_paused() {
        assert_eq!(ChainNodeStatus::Paused.to_string(), "paused");
    }

    #[test]
    fn is_failed_only_for_failed_variant() {
        assert!(!ChainNodeStatus::Pending.is_failed());
        assert!(!ChainNodeStatus::Running.is_failed());
        assert!(!ChainNodeStatus::Completed.is_failed());
        assert!(ChainNodeStatus::Failed("err".into()).is_failed());
        assert!(!ChainNodeStatus::Paused.is_failed());
    }

    #[test]
    fn node_statuses_keyed_by_name_not_id() {
        let node = ChainNodeConfig {
            id: "planner_v2".into(),
            name: "Planner".into(),
            role: "plan".into(),
            model: "m".into(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: None,
            max_retries: 1,
            timeout_secs: 3600,
            max_tool_turns: 200,
        };
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![node],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "id".into());
        // Must be keyed by name, not id
        assert!(
            state.node_statuses.contains_key("Planner"),
            "should be keyed by name"
        );
        assert!(
            !state.node_statuses.contains_key("planner_v2"),
            "should not be keyed by id"
        );
    }

    #[test]
    fn has_failed_nodes_false_when_all_ok() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        state
            .node_statuses
            .insert("n1".into(), ChainNodeStatus::Completed);
        assert!(!state.has_failed_nodes());
    }

    #[test]
    fn has_failed_nodes_true_when_one_failed() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        state
            .node_statuses
            .insert("n1".into(), ChainNodeStatus::Completed);
        state
            .node_statuses
            .insert("n2".into(), ChainNodeStatus::Failed("boom".into()));
        assert!(state.has_failed_nodes());
    }

    #[test]
    fn last_abort_reason_defaults_none() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "id".into());
        assert!(state.last_abort_reason.is_none());
    }

    #[test]
    fn chain_config_directive_serde_roundtrip() {
        // Without directive
        let yaml = "name: t\nnodes: []\nmax_cycles: 1\nmax_total_turns: 10\nworkspace: /tmp\nskip_permissions_warning: false\n";
        let cfg: ChainConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.directive.is_none());

        // With directive
        let yaml_with = "name: t\nnodes: []\nmax_cycles: 1\nmax_total_turns: 10\nworkspace: /tmp\nskip_permissions_warning: false\ndirective: ./directive.md\n";
        let cfg2: ChainConfig = serde_yaml::from_str(yaml_with).unwrap();
        assert_eq!(
            cfg2.directive.as_deref(),
            Some(std::path::Path::new("./directive.md"))
        );

        // Round-trip via YAML
        let serialized = serde_yaml::to_string(&cfg2).unwrap();
        let cfg3: ChainConfig = serde_yaml::from_str(&serialized).unwrap();
        assert_eq!(
            cfg3.directive.as_deref(),
            Some(std::path::Path::new("./directive.md"))
        );
    }

    #[test]
    fn chain_config_description_optional() {
        let yaml = r#"
name: my-chain
nodes: []
max_cycles: 5
max_total_turns: 100
workspace: /tmp
skip_permissions_warning: false
"#;
        let cfg: ChainConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.description.is_none());

        let yaml_with_desc = r#"
name: my-chain
description: "Runs planner-builder-tester loop"
nodes: []
max_cycles: 5
max_total_turns: 100
workspace: /tmp
skip_permissions_warning: false
"#;
        let cfg2: ChainConfig = serde_yaml::from_str(yaml_with_desc).unwrap();
        assert_eq!(
            cfg2.description.as_deref(),
            Some("Runs planner-builder-tester loop")
        );
    }

    #[test]
    fn save_creates_workspace_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("deeply").join("nested").join("workspace");
        // Directory doesn't exist yet
        assert!(!nested.exists());

        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: nested.clone(),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "id".into());
        state.save(&nested).unwrap();

        assert!(nested.exists(), "save should create workspace directory");
        assert!(nested.join("chain_state.json").exists());
    }

    #[test]
    fn node_timeout_defaults_to_3600() {
        let yaml = r#"
id: builder
name: Builder
role: build stuff
model: llama3
"#;
        let node: ChainNodeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            node.timeout_secs, 3600,
            "timeout_secs should default to 3600"
        );
    }

    #[test]
    fn node_timeout_zero_disables() {
        let yaml = r#"
id: builder
name: Builder
role: build stuff
model: llama3
timeout_secs: 0
"#;
        let node: ChainNodeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(
            node.timeout_secs, 0,
            "timeout_secs: 0 should disable timeout"
        );
    }

    #[test]
    fn node_timeout_custom_value() {
        let yaml = r#"
id: builder
name: Builder
role: build stuff
model: llama3
timeout_secs: 600
"#;
        let node: ChainNodeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(node.timeout_secs, 600);
    }

    #[test]
    fn node_timeout_survives_round_trip() {
        let node = ChainNodeConfig {
            id: "n".into(),
            name: "n".into(),
            role: "r".into(),
            model: "m".into(),
            description: None,
            system_prompt_override: None,
            system_prompt_file: None,
            input_from: None,
            max_retries: 1,
            timeout_secs: 120,
            max_tool_turns: 50,
        };
        let yaml = serde_yaml::to_string(&node).unwrap();
        let back: ChainNodeConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.timeout_secs, 120);
        assert_eq!(back.max_tool_turns, 50);
    }

    // ── max_tool_turns ─────────────────────────────────────────────────────

    #[test]
    fn max_tool_turns_defaults_to_200() {
        let yaml = "id: n\nname: N\nrole: r\nmodel: m\n";
        let node: ChainNodeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(node.max_tool_turns, 200);
    }

    #[test]
    fn max_tool_turns_custom_value() {
        let yaml = "id: n\nname: N\nrole: r\nmodel: m\nmax_tool_turns: 50\n";
        let node: ChainNodeConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(node.max_tool_turns, 50);
    }

    // ── metrics fields ──────────────────────────────────────────────────────

    #[test]
    fn metrics_default_to_empty() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "id".into());
        assert!(state.node_durations.is_empty());
        assert!(state.node_failures.is_empty());
        assert_eq!(state.total_duration_secs, 0.0);
    }

    #[test]
    fn metrics_survive_json_round_trip() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        state
            .node_durations
            .insert("builder".into(), vec![12.5, 15.3, 9.8]);
        state.node_failures.insert("builder".into(), 1);
        state.total_duration_secs = 37.6;

        let json = serde_json::to_string(&state).unwrap();
        let back: ChainState = serde_json::from_str(&json).unwrap();

        assert_eq!(back.node_durations["builder"], vec![12.5, 15.3, 9.8]);
        assert_eq!(back.node_failures["builder"], 1);
        assert!((back.total_duration_secs - 37.6).abs() < 0.01);
    }

    #[test]
    fn metrics_missing_in_json_default_to_empty() {
        // Simulate loading a state file from before metrics were added
        let json = r#"{
            "chain_id": "old",
            "config": {"name":"t","nodes":[],"max_cycles":1,"max_total_turns":10,"workspace":"/tmp","skip_permissions_warning":true},
            "active_node_index": null,
            "node_statuses": {},
            "current_cycle": 0,
            "turns_used": 0,
            "node_outputs": {},
            "last_signal": null,
            "last_updated": "2025-01-01T00:00:00Z"
        }"#;
        let state: ChainState = serde_json::from_str(json).unwrap();
        assert!(state.node_durations.is_empty());
        assert!(state.node_failures.is_empty());
        assert_eq!(state.total_duration_secs, 0.0);
    }

    #[test]
    fn metrics_accumulate_across_entries() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        // Simulate 3 cycles of duration recording
        state
            .node_durations
            .entry("planner".into())
            .or_default()
            .push(10.0);
        state
            .node_durations
            .entry("planner".into())
            .or_default()
            .push(12.0);
        state
            .node_durations
            .entry("builder".into())
            .or_default()
            .push(20.0);
        *state.node_failures.entry("builder".into()).or_insert(0) += 1;
        state.total_duration_secs = 42.0;

        let durs = &state.node_durations["planner"];
        assert_eq!(durs.len(), 2);
        assert_eq!(durs[0], 10.0);
        assert_eq!(durs[1], 12.0);
        assert_eq!(state.node_failures["builder"], 1);
    }

    // ── token tracking fields ─��────────────────────────────────────────────

    #[test]
    fn token_fields_default_to_empty() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "id".into());
        assert!(state.node_prompt_tokens.is_empty());
        assert!(state.node_completion_tokens.is_empty());
    }

    #[test]
    fn token_fields_survive_json_round_trip() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        state.node_prompt_tokens.insert("builder".into(), 150_000);
        state
            .node_completion_tokens
            .insert("builder".into(), 45_000);

        let json = serde_json::to_string(&state).unwrap();
        let back: ChainState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.node_prompt_tokens["builder"], 150_000);
        assert_eq!(back.node_completion_tokens["builder"], 45_000);
    }

    #[test]
    fn token_fields_missing_in_old_json_default_to_empty() {
        // Simulate loading a state file from before token tracking was added
        let json = r#"{
            "chain_id": "old",
            "config": {"name":"t","nodes":[],"max_cycles":1,"max_total_turns":10,"workspace":"/tmp","skip_permissions_warning":true},
            "active_node_index": null,
            "node_statuses": {},
            "current_cycle": 0,
            "turns_used": 0,
            "node_outputs": {},
            "last_signal": null,
            "last_updated": "2025-01-01T00:00:00Z"
        }"#;
        let state: ChainState = serde_json::from_str(json).unwrap();
        assert!(state.node_prompt_tokens.is_empty());
        assert!(state.node_completion_tokens.is_empty());
    }

    #[test]
    fn token_fields_accumulate() {
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut state = ChainState::new(config, "id".into());
        *state
            .node_prompt_tokens
            .entry("planner".into())
            .or_insert(0) += 10_000;
        *state
            .node_prompt_tokens
            .entry("planner".into())
            .or_insert(0) += 12_000;
        *state
            .node_completion_tokens
            .entry("planner".into())
            .or_insert(0) += 3_000;
        assert_eq!(state.node_prompt_tokens["planner"], 22_000);
        assert_eq!(state.node_completion_tokens["planner"], 3_000);
    }

    // ── alias resolution ──────────────────────────────────────────────────

    #[test]
    fn chain_config_resolve_aliases() {
        let mut config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "a".into(),
                    name: "A".into(),
                    role: "r".into(),
                    model: "fast".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
                ChainNodeConfig {
                    id: "b".into(),
                    name: "B".into(),
                    role: "r".into(),
                    model: "gemma4:26b".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
            ],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut aliases = HashMap::new();
        aliases.insert("fast".to_string(), "llama3.2:3b".to_string());
        config.resolve_aliases(&aliases);
        assert_eq!(config.nodes[0].model, "llama3.2:3b"); // resolved
        assert_eq!(config.nodes[1].model, "gemma4:26b"); // unchanged
    }

    #[test]
    fn resolve_aliases_with_report_returns_changes() {
        let mut config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![
                ChainNodeConfig {
                    id: "a".into(),
                    name: "Planner".into(),
                    role: "r".into(),
                    model: "fast".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
                ChainNodeConfig {
                    id: "b".into(),
                    name: "Builder".into(),
                    role: "r".into(),
                    model: "gemma4:26b".into(),
                    description: None,
                    system_prompt_override: None,
                    system_prompt_file: None,
                    input_from: None,
                    max_retries: 1,
                    timeout_secs: 3600,
                    max_tool_turns: 200,
                },
            ],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let mut aliases = HashMap::new();
        aliases.insert("fast".to_string(), "llama3.2:3b".to_string());
        let report = config.resolve_aliases_with_report(&aliases);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].0, "Planner");
        assert_eq!(report[0].1, "fast");
        assert_eq!(report[0].2, "llama3.2:3b");
        assert_eq!(config.nodes[0].model, "llama3.2:3b");
        assert_eq!(config.nodes[1].model, "gemma4:26b"); // unchanged
    }

    #[test]
    fn resolve_aliases_with_report_empty_returns_empty() {
        let mut config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![ChainNodeConfig {
                id: "a".into(),
                name: "A".into(),
                role: "r".into(),
                model: "fast".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let report = config.resolve_aliases_with_report(&HashMap::new());
        assert!(report.is_empty());
        assert_eq!(config.nodes[0].model, "fast"); // unchanged
    }

    #[test]
    fn chain_config_resolve_empty_aliases_is_noop() {
        let mut config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![ChainNodeConfig {
                id: "a".into(),
                name: "A".into(),
                role: "r".into(),
                model: "fast".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: std::path::PathBuf::from("/tmp"),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        config.resolve_aliases(&HashMap::new());
        assert_eq!(config.nodes[0].model, "fast"); // unchanged
    }

    #[test]
    fn save_is_atomic_no_tmp_left_behind() {
        let tmp = tempfile::tempdir().unwrap();
        let config = ChainConfig {
            name: "test-chain".into(),
            description: None,
            nodes: vec![ChainNodeConfig {
                id: "n1".into(),
                name: "n1".into(),
                role: "worker".into(),
                model: "llama3".into(),
                description: None,
                system_prompt_override: None,
                system_prompt_file: None,
                input_from: None,
                max_retries: 1,
                timeout_secs: 3600,
                max_tool_turns: 200,
            }],
            max_cycles: 3,
            max_total_turns: 60,
            workspace: tmp.path().to_path_buf(),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "test-id".into());
        state.save(tmp.path()).unwrap();
        let state_path = tmp.path().join("chain_state.json");
        let tmp_path = state_path.with_extension("json.tmp");
        assert!(state_path.exists(), "chain_state.json should exist");
        assert!(
            !tmp_path.exists(),
            "tmp file should not remain after atomic save"
        );
        let reloaded = ChainState::load(&state_path).unwrap();
        assert_eq!(reloaded.chain_id, "test-id");
    }

    #[test]
    fn chain_state_save_produces_compact_json() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = ChainConfig {
            name: "t".into(),
            description: None,
            nodes: vec![],
            max_cycles: 1,
            max_total_turns: 10,
            workspace: tmp.path().to_path_buf(),
            skip_permissions_warning: true,
            loop_forever: false,
            directive: None,
        };
        let state = ChainState::new(config, "compact-test".into());
        state.save(tmp.path()).unwrap();
        let content = std::fs::read_to_string(tmp.path().join("chain_state.json")).unwrap();
        assert!(
            !content.contains('\n'),
            "compact JSON should not contain newlines"
        );
    }

    #[test]
    fn from_status_str_all_variants() {
        assert!(matches!(
            ChainNodeStatus::from_status_str("pending"),
            Some(ChainNodeStatus::Pending)
        ));
        assert!(matches!(
            ChainNodeStatus::from_status_str("running"),
            Some(ChainNodeStatus::Running)
        ));
        assert!(matches!(
            ChainNodeStatus::from_status_str("completed"),
            Some(ChainNodeStatus::Completed)
        ));
        assert!(matches!(
            ChainNodeStatus::from_status_str("paused"),
            Some(ChainNodeStatus::Paused)
        ));
        match ChainNodeStatus::from_status_str("failed: timeout") {
            Some(ChainNodeStatus::Failed(r)) => assert_eq!(r, "timeout"),
            other => panic!("expected Failed, got {:?}", other),
        }
        assert!(ChainNodeStatus::from_status_str("unknown").is_none());
        assert!(ChainNodeStatus::from_status_str("").is_none());
    }

    #[test]
    fn status_constants_match_display() {
        assert_eq!(ChainNodeStatus::Pending.to_string(), STATUS_PENDING);
        assert_eq!(ChainNodeStatus::Running.to_string(), STATUS_RUNNING);
        assert_eq!(ChainNodeStatus::Completed.to_string(), STATUS_COMPLETED);
        assert_eq!(ChainNodeStatus::Paused.to_string(), STATUS_PAUSED);
        assert!(ChainNodeStatus::Failed("x".into())
            .to_string()
            .starts_with(STATUS_FAILED_PREFIX));
    }
}
