//! Process-wide configuration loaded from `~/.dm/settings.json`,
//! `~/.dm/config.toml`, environment variables, and CLI flags.
//!
//! `Config::load()` is the single entry point. It resolves model + Ollama host
//! defaults, prompt-routing rules, model aliases, retry/backoff settings, and
//! daemon snapshot/idle parameters. Most subsystems take `&Config` (or an
//! `Arc<Config>`) rather than reading the environment directly.
//!
//! # Identity-aware routing
//!
//! `Config` exposes two roots:
//!
//! - `config_dir` — the **project-scoped** config dir. In `kernel` mode this
//!   is `~/.dm` (unchanged). In `host` mode this is `<project_root>/.dm` so
//!   per-project state (sessions, daemon socket, index, chains, hooks, …)
//!   lives next to the host project's identity rather than colliding under
//!   `~/.dm/` with every other host project on the same machine.
//! - `global_config_dir` — always `~/.dm`. Operator-level config that should
//!   transcend host projects — `settings.json` (default Ollama host/model)
//!   and `config.toml` (routing rules + model aliases) — is read from here
//!   in both modes so a single laptop's operator preferences apply across
//!   every spawned project.
//!
//! See `VISION.md` § "Identity protocol" and the Run-31 directive
//! ("Identity-Aware Configuration Routing") for the rationale. The
//! per-consumer decisions (sessions/daemon/web/logs project-local;
//! settings/config.toml global) are documented in
//! `.dm/wiki/concepts/identity-config-routing.md`.

use crate::identity::{Identity, Mode};
use anyhow::Context;
use dirs::home_dir;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model routing rules loaded from `~/.dm/config.toml [routing]`.
#[derive(Debug, Clone, Default)]
pub struct RoutingConfig {
    /// Named rules: key (e.g. "code", "quick") → model name.
    pub rules: HashMap<String, String>,
    /// Fallback model when no rule matches (from the "default" key).
    pub default: String,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    /// True when `host` came from the hardcoded "localhost:11434" fallback
    /// (no `OLLAMA_HOST` env, no `settings.host`). Used by `main::run` to
    /// decide whether to auto-probe for an Ollama instance.
    pub host_is_default: bool,
    pub model: String,
    /// True when `model` came from the hardcoded "gemma4:26b-128k" fallback
    /// (no `DM_MODEL` env, no `settings.model`, no CLI override). Used by
    /// `main::run` to decide whether to auto-pick a model from the Ollama
    /// installed-models list.
    pub model_is_default: bool,
    /// If set, tool-processing rounds use this model; reasoning rounds use `model`.
    pub tool_model: Option<String>,
    /// Embedding model for semantic search (default: "nomic-embed-text").
    pub embed_model: String,
    /// Project-scoped config dir. Equals `~/.dm` in `kernel` mode and
    /// `<project_root>/.dm` in `host` mode. Owns per-project state:
    /// sessions, daemon socket/pid, semantic-search index, chains,
    /// schedules, hooks, todos, project memory, evals, plugins.
    pub config_dir: PathBuf,
    /// Operator-level config dir; always `~/.dm`. Owns global-by-design
    /// state that transcends host projects: `settings.json`,
    /// `config.toml`, the global half of `mcp_servers.json` layering.
    /// Equal to `config_dir` in `kernel` mode.
    pub global_config_dir: PathBuf,
    /// Routing rules from `<global_config_dir>/config.toml`, if present.
    pub routing: Option<RoutingConfig>,
    /// Model aliases from `<global_config_dir>/config.toml [aliases]`.
    pub aliases: HashMap<String, String>,
    /// Max Ollama retry attempts on transient failures (default 3).
    pub max_retries: usize,
    /// Base delay in ms for exponential backoff on retries (default 1000).
    pub retry_delay_ms: u64,
    /// Maximum delay cap in ms for exponential backoff (default 30000).
    pub max_retry_delay_ms: u64,
    /// Fallback model to switch to when the primary model hits a fatal error.
    pub fallback_model: Option<String>,
    /// Seconds between periodic daemon session snapshots. 0 disables the
    /// timer entirely. Clamped to [0, 86400]. Default 300 (5 minutes).
    pub snapshot_interval_secs: u64,
    /// Seconds of idleness (time since last user turn) before a daemon
    /// session is eligible for GC. 0 disables eviction entirely. Any other
    /// value is clamped to [60, 86400] — below 60 would collide with the
    /// daemon's 60s GC tick cadence. Default 7200 (2 hours).
    pub idle_timeout_secs: u64,
}

/// Schema for ~/.dm/settings.json
#[derive(Debug, Deserialize, Serialize, Default)]
struct Settings {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    host: Option<String>,
    #[serde(default)]
    tool_model: Option<String>,
    #[serde(default)]
    embed_model: Option<String>,
    /// Max Ollama retry attempts on transient failures (1-10, default 3).
    #[serde(default)]
    max_retries: Option<usize>,
    /// Base delay in ms for exponential backoff on retries (500-5000, default 1000).
    #[serde(default)]
    retry_delay_ms: Option<u64>,
    /// Maximum delay cap in ms for exponential backoff (5000-60000, default 30000).
    #[serde(default)]
    max_retry_delay_ms: Option<u64>,
    #[serde(default)]
    fallback_model: Option<String>,
    #[serde(default)]
    auto_format: Option<bool>,
    /// Seconds between periodic daemon session snapshots (0 disables,
    /// default 300, max 86400).
    #[serde(default)]
    snapshot_interval_secs: Option<u64>,
    /// Seconds of idleness before a daemon session is GC'd. 0 disables,
    /// else clamped to [60, 86400]. Default 7200 (2h).
    #[serde(default)]
    idle_timeout_secs: Option<u64>,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let home = home_dir().context("Could not determine home directory")?;
        let identity = crate::identity::load_for_cwd();
        let global_config_dir = home.join(".dm");
        let config_dir = compute_config_dir(&home, &identity);

        // Always create the global dir — settings.json/config.toml live there
        // regardless of mode.
        std::fs::create_dir_all(&global_config_dir)
            .context("Failed to create ~/.dm config directory")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(
                &global_config_dir,
                std::fs::Permissions::from_mode(0o700),
            );
        }
        // In host mode, also ensure the project-scoped dir exists. (Spawn
        // already creates `<project>/.dm/`, but defensive create_dir_all
        // covers hand-edited / pre-spawn-paradigm host projects.)
        if config_dir != global_config_dir {
            std::fs::create_dir_all(&config_dir).with_context(|| {
                format!(
                    "Failed to create project config directory at {}",
                    config_dir.display()
                )
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(&config_dir, std::fs::Permissions::from_mode(0o700));
            }
        }

        // Operator-level data (settings.json, config.toml) is read from the
        // global dir in both modes — see module docs for the routing rule.
        let settings = load_settings(&global_config_dir);

        // Priority: CLI flag (applied later in main.rs) > env var > settings.json > default
        let (raw_host, host_is_default) = derive_host(
            std::env::var("OLLAMA_HOST").ok().as_deref(),
            settings.host.as_deref(),
        );
        let host = normalize_host(&raw_host);

        let (model, model_is_default) = derive_model(
            std::env::var("DM_MODEL").ok().as_deref(),
            settings.model.as_deref(),
        );

        let routing = load_routing_config(&global_config_dir);
        let aliases = load_aliases(&global_config_dir);

        let raw_retries = settings.max_retries.unwrap_or(3);
        let max_retries = raw_retries.clamp(1, 10);
        if max_retries != raw_retries {
            crate::warnings::push_warning(format!(
                "max_retries {} clamped to {} (valid range: 1–10)",
                raw_retries, max_retries
            ));
        }
        let raw_delay = settings.retry_delay_ms.unwrap_or(1000);
        let retry_delay_ms = raw_delay.clamp(500, 5000);
        if retry_delay_ms != raw_delay {
            crate::warnings::push_warning(format!(
                "retry_delay_ms {} clamped to {} (valid range: 500–5000)",
                raw_delay, retry_delay_ms
            ));
        }
        let raw_max_delay = settings.max_retry_delay_ms.unwrap_or(30_000);
        let max_retry_delay_ms = raw_max_delay.clamp(5_000, 60_000);
        if max_retry_delay_ms != raw_max_delay {
            crate::warnings::push_warning(format!(
                "max_retry_delay_ms {} clamped to {} (valid range: 5000–60000)",
                raw_max_delay, max_retry_delay_ms
            ));
        }

        let fallback_model = std::env::var("DM_FALLBACK_MODEL")
            .ok()
            .or(settings.fallback_model);

        let raw_snap = settings.snapshot_interval_secs.unwrap_or(300);
        let snapshot_interval_secs = raw_snap.min(86_400);
        if snapshot_interval_secs != raw_snap {
            crate::warnings::push_warning(format!(
                "snapshot_interval_secs {} clamped to {} (max 86400)",
                raw_snap, snapshot_interval_secs
            ));
        }

        let raw_idle = settings.idle_timeout_secs.unwrap_or(7200);
        // 0 is a legal "disabled" sentinel and must not be clamped up to 60.
        let idle_timeout_secs = if raw_idle == 0 {
            0
        } else {
            raw_idle.clamp(60, 86_400)
        };
        if idle_timeout_secs != raw_idle {
            crate::warnings::push_warning(format!(
                "idle_timeout_secs {} clamped to {} (valid: 0 or 60–86400)",
                raw_idle, idle_timeout_secs
            ));
        }

        Ok(Config {
            host,
            host_is_default,
            model,
            model_is_default,
            tool_model: settings.tool_model,
            embed_model: settings
                .embed_model
                .unwrap_or_else(|| "nomic-embed-text".to_string()),
            config_dir,
            global_config_dir,
            routing,
            aliases,
            max_retries,
            retry_delay_ms,
            max_retry_delay_ms,
            fallback_model,
            snapshot_interval_secs,
            idle_timeout_secs,
        })
    }

    /// Resolve a model name through aliases. If `name` matches an alias key,
    /// return the target. Otherwise return `name` unchanged.
    /// No recursive resolution — one level only.
    pub fn resolve_alias(&self, name: &str) -> String {
        self.aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    pub fn ollama_base_url(&self) -> String {
        format!("http://{}/api", self.host)
    }
}

/// Identity-aware project config dir resolver.
///
/// - `kernel` mode → `<home>/.dm` (legacy, unchanged).
/// - `host` mode → `<project_root>/.dm`, where `project_root` is derived
///   from `identity.source` (`<project>/.dm/identity.toml`'s grandparent).
///   If a host identity is present without a `source` (constructed in-memory,
///   e.g. tests), falls back to `<home>/.dm` so we never invent a project
///   root from thin air.
///
/// Pure for testability — no I/O, no env access.
pub(crate) fn compute_config_dir(home: &Path, identity: &Identity) -> PathBuf {
    if identity.mode == Mode::Host {
        if let Some(project_root) = identity
            .source
            .as_deref()
            .and_then(Path::parent) // .../.dm
            .and_then(Path::parent)
        // <project_root>
        {
            return project_root.join(".dm");
        }
    }
    home.join(".dm")
}

/// Resolve the Ollama host from the available sources, returning the chosen
/// raw value and a flag indicating whether it came from the hardcoded
/// default (i.e. the user has *not* explicitly configured a host).
///
/// Priority: `OLLAMA_HOST` env > `settings.host` > `"localhost:11434"`.
pub(crate) fn derive_host(env: Option<&str>, settings: Option<&str>) -> (String, bool) {
    if let Some(v) = env {
        return (v.to_string(), false);
    }
    if let Some(v) = settings {
        return (v.to_string(), false);
    }
    ("localhost:11434".to_string(), true)
}

/// Resolve the model name from available sources. Returns `(value, is_default)`
/// — `is_default` is true only when all explicit sources were `None` and the
/// hardcoded `gemma4:26b-128k` fallback kicked in.
///
/// Priority: `DM_MODEL` env > `settings.model` > `"gemma4:26b-128k"`.
pub(crate) fn derive_model(env: Option<&str>, settings: Option<&str>) -> (String, bool) {
    if let Some(v) = env {
        return (v.to_string(), false);
    }
    if let Some(v) = settings {
        return (v.to_string(), false);
    }
    ("gemma4:26b-128k".to_string(), true)
}

/// Normalize an Ollama host string: strip `http://` or `https://` scheme
/// and any trailing root slash, leaving `hostname:port` (or
/// `hostname:port/path-prefix`) form. Called at every config-boundary
/// site that writes `Config::host` (parse in `Config::load` and CLI
/// overrides in `main.rs`) so `Config::host` is always canonical —
/// downstream `ollama_base_url()` then safely prepends exactly one
/// `http://`.
pub fn normalize_host(raw: &str) -> String {
    let without_scheme = raw
        .strip_prefix("https://")
        .or_else(|| raw.strip_prefix("http://"))
        .unwrap_or(raw);
    without_scheme.trim_end_matches('/').to_string()
}

/// Check whether auto-formatting of tool-written files is enabled.
/// Priority: `DM_AUTO_FORMAT` env var > `auto_format` in settings.json > false.
///
/// Always reads `~/.dm/settings.json` regardless of identity mode — auto-format
/// is operator-level, not project-level (`settings.json` lives in
/// `Config::global_config_dir`).
pub fn auto_format_enabled() -> bool {
    if let Ok(val) = std::env::var("DM_AUTO_FORMAT") {
        return val == "1" || val.eq_ignore_ascii_case("true");
    }
    let global_config_dir = home_dir().map(|h| h.join(".dm")).unwrap_or_default();
    let settings = load_settings(&global_config_dir);
    settings.auto_format.unwrap_or(false)
}

/// Load `~/.dm/config.toml` and extract the `[routing]` section.
/// Missing file or missing section → None (no routing).
pub fn load_routing_config(config_dir: &std::path::Path) -> Option<RoutingConfig> {
    let path = config_dir.join("config.toml");
    let content = std::fs::read_to_string(&path).ok()?;
    let table: toml::Table = match content.parse() {
        Ok(t) => t,
        Err(e) => {
            crate::warnings::push_warning(format!(
                "config.toml is malformed ({}), routing disabled",
                e
            ));
            return None;
        }
    };
    let routing = table.get("routing")?.as_table()?;

    let mut rules = HashMap::new();
    for (k, v) in routing {
        if let Some(s) = v.as_str() {
            rules.insert(k.clone(), s.to_string());
        }
    }
    let default = rules.remove("default").unwrap_or_default();
    Some(RoutingConfig { rules, default })
}

/// Load `~/.dm/config.toml` and extract the `[aliases]` section.
/// Missing file or missing section → empty `HashMap`.
pub fn load_aliases(config_dir: &std::path::Path) -> HashMap<String, String> {
    let path = config_dir.join("config.toml");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    let table: toml::Table = match content.parse() {
        Ok(t) => t,
        Err(e) => {
            crate::warnings::push_warning(format!(
                "config.toml is malformed ({}), aliases disabled",
                e
            ));
            return HashMap::new();
        }
    };
    let Some(aliases) = table.get("aliases").and_then(|v| v.as_table()) else {
        return HashMap::new();
    };
    aliases
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect()
}

fn load_settings(config_dir: &std::path::Path) -> Settings {
    let path = config_dir.join("settings.json");
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            crate::warnings::push_warning(format!(
                "{} is malformed: {}. Valid fields: model, host, tool_model, embed_model, \
                 max_retries, retry_delay_ms, max_retry_delay_ms, fallback_model, auto_format. \
                 Using defaults.",
                path.display(),
                e
            ));
            Settings::default()
        }),
        Err(_) => Settings::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Identity, Mode};
    use std::sync::Mutex as StdMutex;

    // Serializes tests that mutate DM_AUTO_FORMAT. Without this, parallel
    // runs race: `auto_format_enabled_default_false` can observe a "1"/"true"
    // value set by `auto_format_enabled_respects_env_var`.
    // Pattern mirrors src/logging.rs:260.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    struct AutoFormatEnvGuard {
        prev: Option<String>,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl AutoFormatEnvGuard {
        fn acquire() -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let prev = std::env::var("DM_AUTO_FORMAT").ok();
            Self {
                prev,
                _guard: guard,
            }
        }
    }

    impl Drop for AutoFormatEnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("DM_AUTO_FORMAT", v),
                None => std::env::remove_var("DM_AUTO_FORMAT"),
            }
        }
    }

    #[test]
    fn malformed_settings_json_falls_back_to_defaults() {
        // The fallback logic: serde parse failure → Settings::default()
        let result: Settings =
            serde_json::from_str("{not valid json}").unwrap_or_else(|_| Settings::default());
        assert!(
            result.model.is_none(),
            "malformed JSON should yield None model"
        );
        assert!(
            result.host.is_none(),
            "malformed JSON should yield None host"
        );
    }

    #[test]
    fn valid_settings_json_is_parsed() {
        let result: Settings =
            serde_json::from_str(r#"{"model":"llama3","host":"10.0.0.1:11434"}"#)
                .unwrap_or_default();
        assert_eq!(result.model.as_deref(), Some("llama3"));
        assert_eq!(result.host.as_deref(), Some("10.0.0.1:11434"));
    }

    #[test]
    fn missing_fields_in_settings_are_none() {
        let result: Settings = serde_json::from_str(r#"{}"#).unwrap_or_default();
        assert!(result.model.is_none());
        assert!(result.host.is_none());
    }

    #[test]
    fn load_routing_config_missing_file_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_routing_config(dir.path());
        assert!(result.is_none(), "missing file should return None");
    }

    #[test]
    fn load_routing_config_missing_routing_section_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), "[general]\nkey = \"val\"\n").unwrap();
        let result = load_routing_config(dir.path());
        assert!(result.is_none(), "no [routing] section should return None");
    }

    #[test]
    fn load_routing_config_parses_rules_and_default() {
        let dir = tempfile::tempdir().unwrap();
        let toml = "[routing]\ndefault = \"gemma4:26b\"\ncode = \"codellama:34b\"\n";
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let result = load_routing_config(dir.path()).expect("should parse routing");
        assert_eq!(result.default, "gemma4:26b");
        assert_eq!(
            result.rules.get("code").map(String::as_str),
            Some("codellama:34b")
        );
        // "default" key should be removed from rules (moved to `.default` field)
        assert!(!result.rules.contains_key("default"));
    }

    #[test]
    fn ollama_base_url_formats_correctly() {
        let config = Config {
            host: "10.0.0.5:11434".to_string(),
            host_is_default: false,
            model: "gemma4:26b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            global_config_dir: std::path::PathBuf::from("/tmp"),
            routing: None,
            aliases: HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        };
        assert_eq!(config.ollama_base_url(), "http://10.0.0.5:11434/api");
    }

    #[test]
    fn load_routing_config_no_default_key_gives_empty_default() {
        let dir = tempfile::tempdir().unwrap();
        let toml = "[routing]\ncode = \"codellama:34b\"\n";
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let result = load_routing_config(dir.path()).expect("should parse routing");
        assert_eq!(
            result.default, "",
            "missing 'default' key should yield empty string"
        );
        assert_eq!(
            result.rules.get("code").map(String::as_str),
            Some("codellama:34b")
        );
    }

    #[test]
    fn ollama_base_url_localhost_format() {
        let config = Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "llama3:8b".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            global_config_dir: std::path::PathBuf::from("/tmp"),
            routing: None,
            aliases: HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        };
        assert_eq!(config.ollama_base_url(), "http://localhost:11434/api");
    }

    #[test]
    fn load_routing_config_empty_routing_section_returns_empty_rules() {
        let dir = tempfile::tempdir().unwrap();
        let toml = "[routing]\n";
        std::fs::write(dir.path().join("config.toml"), toml).unwrap();
        let result =
            load_routing_config(dir.path()).expect("empty routing section should still parse");
        assert!(
            result.rules.is_empty(),
            "empty [routing] section should have no rules"
        );
        assert_eq!(result.default, "");
    }

    #[test]
    fn retry_settings_from_json() {
        let s: Settings =
            serde_json::from_str(r#"{"max_retries": 5, "retry_delay_ms": 2000}"#).unwrap();
        assert_eq!(s.max_retries, Some(5));
        assert_eq!(s.retry_delay_ms, Some(2000));
    }

    #[test]
    fn retry_settings_missing_defaults_to_none() {
        let s: Settings = serde_json::from_str(r#"{}"#).unwrap();
        assert!(s.max_retries.is_none());
        assert!(s.retry_delay_ms.is_none());
    }

    #[test]
    fn resolve_alias_returns_target() {
        let mut aliases = HashMap::new();
        aliases.insert("fast".to_string(), "llama3.2:3b".to_string());
        let config = Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "x".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            global_config_dir: std::path::PathBuf::from("/tmp"),
            routing: None,
            aliases,
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        };
        assert_eq!(config.resolve_alias("fast"), "llama3.2:3b");
    }

    #[test]
    fn resolve_alias_passthrough_unknown() {
        let config = Config {
            host: "localhost:11434".to_string(),
            host_is_default: false,
            model: "x".to_string(),
            model_is_default: false,
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            config_dir: std::path::PathBuf::from("/tmp"),
            global_config_dir: std::path::PathBuf::from("/tmp"),
            routing: None,
            aliases: HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        };
        assert_eq!(config.resolve_alias("gemma4:26b"), "gemma4:26b");
    }

    #[test]
    fn load_aliases_from_toml() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.toml"),
            "[aliases]\nfast = \"llama3.2:3b\"\nsmart = \"gemma4:26b\"\n",
        )
        .unwrap();
        let aliases = load_aliases(tmp.path());
        assert_eq!(aliases.get("fast").unwrap(), "llama3.2:3b");
        assert_eq!(aliases.get("smart").unwrap(), "gemma4:26b");
    }

    #[test]
    fn load_aliases_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let aliases = load_aliases(tmp.path());
        assert!(aliases.is_empty());
    }

    #[test]
    fn load_aliases_no_section() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.toml"),
            "[routing]\ndefault = \"x\"\n",
        )
        .unwrap();
        let aliases = load_aliases(tmp.path());
        assert!(aliases.is_empty());
    }

    #[test]
    fn fallback_model_loads_from_settings() {
        let result: Settings =
            serde_json::from_str(r#"{"fallback_model":"llama3.2:3b"}"#).unwrap_or_default();
        assert_eq!(result.fallback_model.as_deref(), Some("llama3.2:3b"));
    }

    #[test]
    fn fallback_model_defaults_to_none() {
        let result: Settings = serde_json::from_str(r#"{}"#).unwrap_or_default();
        assert!(result.fallback_model.is_none());
    }

    #[test]
    fn auto_format_enabled_respects_env_var() {
        let _guard = AutoFormatEnvGuard::acquire();
        std::env::set_var("DM_AUTO_FORMAT", "1");
        assert!(auto_format_enabled());
        std::env::set_var("DM_AUTO_FORMAT", "true");
        assert!(auto_format_enabled());
        std::env::set_var("DM_AUTO_FORMAT", "0");
        assert!(!auto_format_enabled());
    }

    #[test]
    fn auto_format_enabled_default_false() {
        let _guard = AutoFormatEnvGuard::acquire();
        std::env::remove_var("DM_AUTO_FORMAT");
        assert!(!auto_format_enabled());
    }

    #[test]
    fn normalize_host_passes_through_bare_hostport() {
        assert_eq!(normalize_host("localhost:11434"), "localhost:11434");
    }

    #[test]
    fn normalize_host_strips_http_scheme() {
        assert_eq!(normalize_host("http://myhost:1234"), "myhost:1234");
    }

    #[test]
    fn normalize_host_strips_https_scheme() {
        assert_eq!(normalize_host("https://api.example.com"), "api.example.com");
    }

    #[test]
    fn normalize_host_strips_trailing_slash() {
        assert_eq!(normalize_host("myhost:1234/"), "myhost:1234");
    }

    #[test]
    fn normalize_host_strips_scheme_and_trailing_slash() {
        assert_eq!(normalize_host("http://myhost:1234/"), "myhost:1234");
    }

    #[test]
    fn normalize_host_preserves_path_prefix() {
        // Reverse-proxy deployments mount Ollama at a sub-path — keep it.
        assert_eq!(
            normalize_host("proxy.example.com/ollama"),
            "proxy.example.com/ollama"
        );
        assert_eq!(
            normalize_host("https://proxy.example.com/ollama"),
            "proxy.example.com/ollama"
        );
    }

    #[test]
    fn auto_format_settings_json_field() {
        let result: Settings = serde_json::from_str(r#"{"auto_format": true}"#).unwrap_or_default();
        assert_eq!(result.auto_format, Some(true));
    }

    #[test]
    fn auto_format_settings_json_default_none() {
        let result: Settings = serde_json::from_str(r#"{}"#).unwrap_or_default();
        assert!(result.auto_format.is_none());
    }

    #[test]
    fn settings_unknown_field_still_parses() {
        let result: Settings =
            serde_json::from_str(r#"{"model":"llama3","unknown_key":"ignored"}"#).unwrap();
        assert_eq!(result.model.as_deref(), Some("llama3"));
    }

    #[test]
    fn settings_retries_clamped_high() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"max_retries": 99, "retry_delay_ms": 1, "max_retry_delay_ms": 999999}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let max_retries = settings.max_retries.unwrap_or(3).clamp(1, 10);
        let retry_delay_ms = settings.retry_delay_ms.unwrap_or(1000).clamp(500, 5000);
        let max_retry_delay_ms = settings
            .max_retry_delay_ms
            .unwrap_or(30_000)
            .clamp(5_000, 60_000);
        assert_eq!(max_retries, 10);
        assert_eq!(retry_delay_ms, 500);
        assert_eq!(max_retry_delay_ms, 60_000);
    }

    #[test]
    fn load_routing_config_malformed_toml_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), "not valid [[ toml !!!").unwrap();
        let result = load_routing_config(dir.path());
        assert!(result.is_none(), "malformed TOML should return None");
    }

    #[test]
    fn load_aliases_malformed_toml_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), "not valid [[ toml !!!").unwrap();
        let aliases = load_aliases(dir.path());
        assert!(
            aliases.is_empty(),
            "malformed TOML should return empty aliases"
        );
    }

    #[test]
    fn settings_retries_clamped_low() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"max_retries": 0, "retry_delay_ms": 100, "max_retry_delay_ms": 1000}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let max_retries = settings.max_retries.unwrap_or(3).clamp(1, 10);
        let retry_delay_ms = settings.retry_delay_ms.unwrap_or(1000).clamp(500, 5000);
        let max_retry_delay_ms = settings
            .max_retry_delay_ms
            .unwrap_or(30_000)
            .clamp(5_000, 60_000);
        assert_eq!(max_retries, 1);
        assert_eq!(retry_delay_ms, 500);
        assert_eq!(max_retry_delay_ms, 5_000);
    }

    #[test]
    fn config_clamping_warns_on_out_of_range() {
        let marker = "clamp_test_4e9d";
        let raw = 50u64;
        let clamped = raw.clamp(1, 10);
        if clamped != raw {
            crate::warnings::push_warning(format!(
                "{} max_retries {} clamped to {} (valid range: 1–10)",
                marker, raw, clamped
            ));
        }
        let warnings = crate::warnings::peek_warnings();
        let relevant: Vec<_> = warnings.iter().filter(|w| w.contains(marker)).collect();
        assert!(
            relevant
                .iter()
                .any(|w| w.contains("clamped") && w.contains("50")),
            "should warn when value is out of range: {:?}",
            relevant
        );
    }

    #[test]
    fn config_in_range_no_clamping_warning() {
        let marker = "clamp_test_5f0e";
        let raw = 5u64;
        let clamped = raw.clamp(1, 10);
        if clamped != raw {
            crate::warnings::push_warning(format!(
                "{} max_retries {} clamped to {} (valid range: 1–10)",
                marker, raw, clamped
            ));
        }
        let warnings = crate::warnings::peek_warnings();
        let relevant: Vec<_> = warnings.iter().filter(|w| w.contains(marker)).collect();
        assert!(
            !relevant.iter().any(|w| w.contains("clamped")),
            "should not warn when value is in range: {:?}",
            relevant
        );
    }

    #[test]
    fn snapshot_interval_default_is_300() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.snapshot_interval_secs.unwrap_or(300);
        let clamped = raw.min(86_400);
        assert_eq!(clamped, 300);
    }

    #[test]
    fn snapshot_interval_respects_settings() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"snapshot_interval_secs": 60}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.snapshot_interval_secs.unwrap_or(300);
        let clamped = raw.min(86_400);
        assert_eq!(clamped, 60);
    }

    #[test]
    fn snapshot_interval_zero_disables() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"snapshot_interval_secs": 0}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.snapshot_interval_secs.unwrap_or(300);
        let clamped = raw.min(86_400);
        assert_eq!(
            clamped, 0,
            "0 is the disable sentinel, not overridden by default"
        );
    }

    #[test]
    fn snapshot_interval_clamps_to_max() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"snapshot_interval_secs": 100000}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.snapshot_interval_secs.unwrap_or(300);
        let clamped = raw.min(86_400);
        assert_eq!(clamped, 86_400);
        assert_ne!(
            clamped, raw,
            "100000 must differ from clamped so the warning fires"
        );
    }

    // Replicates the clamp logic from Config::load so these tests don't touch
    // the user's real ~/.dm. Keep in sync with the match in load().
    fn clamp_idle_timeout(raw: u64) -> u64 {
        if raw == 0 {
            0
        } else {
            raw.clamp(60, 86_400)
        }
    }

    #[test]
    fn idle_timeout_default_is_7200() {
        let dir = tempfile::tempdir().unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.idle_timeout_secs.unwrap_or(7200);
        assert_eq!(clamp_idle_timeout(raw), 7200);
    }

    #[test]
    fn idle_timeout_respects_settings() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"idle_timeout_secs": 300}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.idle_timeout_secs.unwrap_or(7200);
        assert_eq!(clamp_idle_timeout(raw), 300);
    }

    #[test]
    fn idle_timeout_zero_disables() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"idle_timeout_secs": 0}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.idle_timeout_secs.unwrap_or(7200);
        let clamped = clamp_idle_timeout(raw);
        assert_eq!(
            clamped, 0,
            "0 is the disable sentinel, must not be clamped up to 60"
        );
        assert_eq!(clamped, raw, "0 must equal raw so no clamp warning fires");
    }

    #[test]
    fn idle_timeout_clamps_below_min() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"idle_timeout_secs": 30}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.idle_timeout_secs.unwrap_or(7200);
        let clamped = clamp_idle_timeout(raw);
        assert_eq!(clamped, 60, "30 must be clamped up to the 60s minimum");
        assert_ne!(
            clamped, raw,
            "30 must differ from clamped so the warning fires"
        );
    }

    #[test]
    fn idle_timeout_clamps_above_max() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"idle_timeout_secs": 100000}"#,
        )
        .unwrap();
        let settings = load_settings(dir.path());
        let raw = settings.idle_timeout_secs.unwrap_or(7200);
        let clamped = clamp_idle_timeout(raw);
        assert_eq!(clamped, 86_400);
        assert_ne!(
            clamped, raw,
            "100000 must differ from clamped so the warning fires"
        );
    }

    #[test]
    fn derive_host_default_when_both_none() {
        let (host, is_default) = derive_host(None, None);
        assert_eq!(host, "localhost:11434");
        assert!(is_default, "hardcoded default must be flagged as default");
    }

    #[test]
    fn derive_host_not_default_when_env_set() {
        let (host, is_default) = derive_host(Some("10.0.0.9:11434"), None);
        assert_eq!(host, "10.0.0.9:11434");
        assert!(!is_default, "env-supplied host is not the default");
    }

    #[test]
    fn derive_host_not_default_when_settings_set() {
        let (host, is_default) = derive_host(None, Some("192.168.1.7:11434"));
        assert_eq!(host, "192.168.1.7:11434");
        assert!(!is_default, "settings-supplied host is not the default");
    }

    #[test]
    fn derive_host_env_wins_over_settings() {
        let (host, is_default) = derive_host(Some("env:11434"), Some("settings:11434"));
        assert_eq!(host, "env:11434", "env var must take precedence");
        assert!(!is_default);
    }

    #[test]
    fn derive_model_default_when_both_none() {
        let (model, is_default) = derive_model(None, None);
        assert_eq!(model, "gemma4:26b-128k");
        assert!(is_default, "hardcoded default must be flagged as default");
    }

    #[test]
    fn derive_model_not_default_when_env_set() {
        let (model, is_default) = derive_model(Some("llama3.1:8b"), None);
        assert_eq!(model, "llama3.1:8b");
        assert!(!is_default);
    }

    #[test]
    fn derive_model_not_default_when_settings_set() {
        let (model, is_default) = derive_model(None, Some("qwen2.5:14b"));
        assert_eq!(model, "qwen2.5:14b");
        assert!(!is_default);
    }

    #[test]
    fn derive_model_env_wins_over_settings() {
        let (model, is_default) = derive_model(Some("env-model"), Some("settings-model"));
        assert_eq!(model, "env-model", "env var must take precedence");
        assert!(!is_default);
    }

    #[cfg(unix)]
    #[test]
    fn config_dir_gets_restricted_permissions() {
        use std::os::unix::fs::MetadataExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let config_dir = tmp.path().join(".dm");
        std::fs::create_dir_all(&config_dir).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&config_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mode = std::fs::metadata(&config_dir).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o700, "config dir should be owner-only: {:o}", mode);
    }

    // -- compute_config_dir routing ---------------------------------------
    //
    // The directive's headline change: project-scoped config_dir in host
    // mode, unchanged in kernel mode. Kernel-mode regression coverage is
    // the highest-priority test of this run — existing canonical-dm users
    // must see bit-for-bit identical paths.

    #[test]
    fn compute_config_dir_kernel_returns_home_dm() {
        let home = std::path::Path::new("/home/alice");
        let identity = Identity::default_kernel();
        assert_eq!(
            compute_config_dir(home, &identity),
            std::path::PathBuf::from("/home/alice/.dm"),
            "kernel mode must remain ~/.dm — backwards compatibility rule",
        );
    }

    #[test]
    fn compute_config_dir_kernel_with_explicit_source_still_uses_home() {
        // A kernel-mode identity loaded from a real `.dm/identity.toml`
        // (e.g. inside the canonical-dm repo) must NOT route to that
        // project root — kernel mode is always ~/.dm regardless of where
        // the identity file came from.
        let home = std::path::Path::new("/home/alice");
        let identity = Identity {
            mode: Mode::Kernel,
            host_project: None,
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: Some(std::path::PathBuf::from(
                "/home/alice/dev/dark-matter/.dm/identity.toml",
            )),
        };
        assert_eq!(
            compute_config_dir(home, &identity),
            std::path::PathBuf::from("/home/alice/.dm"),
        );
    }

    #[test]
    fn compute_config_dir_host_routes_to_project_root() {
        // The headline kotoba paradigm-gap fix: a host project's config
        // dir is `<project>/.dm`, derived from `identity.source`'s
        // grandparent. Confirms sessions / daemon / index / chains all
        // land next to the host project's identity, not in `~/.dm/`.
        let home = std::path::Path::new("/home/alice");
        let identity = Identity {
            mode: Mode::Host,
            host_project: Some("kotoba".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: Some(std::path::PathBuf::from(
                "/home/alice/dev/kotoba/.dm/identity.toml",
            )),
        };
        assert_eq!(
            compute_config_dir(home, &identity),
            std::path::PathBuf::from("/home/alice/dev/kotoba/.dm"),
        );
    }

    #[test]
    fn compute_config_dir_host_without_source_falls_back_to_home() {
        // Defense in depth: a host identity without a `source` (in-memory
        // construction, not loaded from disk) must NOT invent a project
        // root from cwd or anywhere else — it falls back to ~/.dm so we
        // never silently pollute a user's working directory.
        let home = std::path::Path::new("/home/alice");
        let identity = Identity {
            mode: Mode::Host,
            host_project: Some("synthetic".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        assert_eq!(
            compute_config_dir(home, &identity),
            std::path::PathBuf::from("/home/alice/.dm"),
            "sourceless host identity must not invent a project root",
        );
    }
}
