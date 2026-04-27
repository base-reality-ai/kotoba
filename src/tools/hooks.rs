use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A hook is a shell command that runs before or after a specific tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolHook {
    /// Shell command run before the tool executes.
    /// Env: `DM_TOOL_NAME`, `DM_TOOL_ARGS` (JSON)
    #[serde(default)]
    pub pre: Option<String>,
    /// Shell command run after the tool executes.
    /// Env: `DM_TOOL_NAME`, `DM_TOOL_ARGS` (JSON), `DM_TOOL_RESULT`
    #[serde(default)]
    pub post: Option<String>,
}

/// Map from tool name (or `"*"` for all tools) to hook definition.
/// Loaded from `~/.dm/hooks.json`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HooksConfig {
    #[serde(default)]
    pub hooks: HashMap<String, ToolHook>,
}

impl HooksConfig {
    pub fn load(config_dir: &Path) -> Self {
        let path = config_dir.join("hooks.json");
        match std::fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => HooksConfig::default(),
        }
    }

    /// Return the hook for `tool_name`, falling back to the wildcard `"*"` entry.
    pub fn hook_for(&self, tool_name: &str) -> Option<&ToolHook> {
        self.hooks.get(tool_name).or_else(|| self.hooks.get("*"))
    }
}

/// Run a hook shell command. Returns stdout (trimmed) or an error.
pub async fn run_hook(
    command: &str,
    tool_name: &str,
    tool_args: &serde_json::Value,
    tool_result: Option<&str>,
) -> Result<String> {
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c").arg(command);
    cmd.env("DM_TOOL_NAME", tool_name);
    cmd.env("DM_TOOL_ARGS", tool_args.to_string());
    if let Some(result) = tool_result {
        cmd.env("DM_TOOL_RESULT", result);
    }
    cmd.kill_on_drop(true);

    let output = cmd.output().await?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_config(hooks: HashMap<String, ToolHook>) -> HooksConfig {
        HooksConfig { hooks }
    }

    #[test]
    fn hook_for_specific_tool_returns_it() {
        let mut hooks = HashMap::new();
        hooks.insert(
            "bash".to_string(),
            ToolHook {
                pre: Some("echo pre".into()),
                post: None,
            },
        );
        let cfg = make_config(hooks);
        assert!(cfg.hook_for("bash").is_some());
        assert!(cfg.hook_for("grep").is_none());
    }

    #[test]
    fn wildcard_hook_matches_any_tool() {
        let mut hooks = HashMap::new();
        hooks.insert(
            "*".to_string(),
            ToolHook {
                pre: None,
                post: Some("echo post".into()),
            },
        );
        let cfg = make_config(hooks);
        assert!(cfg.hook_for("bash").is_some());
        assert!(cfg.hook_for("glob").is_some());
        assert!(cfg.hook_for("anything").is_some());
    }

    #[test]
    fn specific_tool_takes_precedence_over_wildcard() {
        let mut hooks = HashMap::new();
        hooks.insert(
            "bash".to_string(),
            ToolHook {
                pre: Some("specific".into()),
                post: None,
            },
        );
        hooks.insert(
            "*".to_string(),
            ToolHook {
                pre: Some("wildcard".into()),
                post: None,
            },
        );
        let cfg = make_config(hooks);
        assert_eq!(
            cfg.hook_for("bash").and_then(|h| h.pre.as_deref()),
            Some("specific")
        );
        assert_eq!(
            cfg.hook_for("grep").and_then(|h| h.pre.as_deref()),
            Some("wildcard")
        );
    }

    #[test]
    fn empty_hooks_config_returns_none() {
        let cfg = HooksConfig::default();
        assert!(cfg.hook_for("bash").is_none());
        assert!(cfg.hook_for("*").is_none());
    }

    #[test]
    fn hooks_config_deserializes_from_json() {
        let json_str = r#"{"hooks":{"bash":{"pre":"echo running bash","post":"echo bash done"}}}"#;
        let cfg: HooksConfig = serde_json::from_str(json_str).unwrap();
        let hook = cfg.hook_for("bash").unwrap();
        assert_eq!(hook.pre.as_deref(), Some("echo running bash"));
        assert_eq!(hook.post.as_deref(), Some("echo bash done"));
    }

    #[test]
    fn hooks_config_missing_file_returns_default() {
        let cfg = HooksConfig::load(std::path::Path::new("/nonexistent/path"));
        assert!(cfg.hooks.is_empty());
    }

    #[tokio::test]
    async fn run_hook_captures_stdout() {
        let out = run_hook("echo hello-hook", "bash", &json!({}), None)
            .await
            .unwrap();
        assert_eq!(out, "hello-hook");
    }

    #[tokio::test]
    async fn run_hook_receives_env_vars() {
        let args = json!({"command": "ls"});
        let out = run_hook("echo $DM_TOOL_NAME", "bash", &args, None)
            .await
            .unwrap();
        assert_eq!(out, "bash");
    }

    #[tokio::test]
    async fn run_hook_post_receives_result_env() {
        let out = run_hook(
            "echo $DM_TOOL_RESULT",
            "bash",
            &json!({}),
            Some("my-result"),
        )
        .await
        .unwrap();
        assert_eq!(out, "my-result");
    }

    #[tokio::test]
    async fn run_hook_non_zero_exit_does_not_propagate() {
        // A command that exits non-zero still returns stdout (empty here) without error
        let result = run_hook("exit 1", "bash", &json!({}), None).await;
        assert!(result.is_ok(), "non-zero exit should not be an error");
    }
}
