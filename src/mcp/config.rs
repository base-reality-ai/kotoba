use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// One entry in an MCP server config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

/// Load all MCP server configs. Returns an empty list if the file is absent or malformed.
pub fn load_configs(config_dir: &std::path::Path) -> Vec<McpServerConfig> {
    let path = config_dir.join("mcp_servers.json");
    load_configs_file(&path)
}

/// Load global MCP configs plus project-local MCP configs.
///
/// Global config lives in `config_dir/mcp_servers.json` (normally `~/.dm`).
/// Host projects may add runtime capabilities without recompiling by placing
/// `.dm/mcp_servers.json` under the project root. Project-local entries with
/// the same server name replace global entries so a host can override a
/// user-wide default for that project.
pub fn load_configs_with_project(config_dir: &Path, project_root: &Path) -> Vec<McpServerConfig> {
    let mut configs = load_configs(config_dir);
    let local = load_configs_file(&project_root.join(".dm").join("mcp_servers.json"));

    let mut by_name: HashMap<String, usize> = configs
        .iter()
        .enumerate()
        .map(|(idx, cfg)| (cfg.name.clone(), idx))
        .collect();
    for cfg in local {
        if let Some(idx) = by_name.get(&cfg.name).copied() {
            configs[idx] = cfg;
        } else {
            by_name.insert(cfg.name.clone(), configs.len());
            configs.push(cfg);
        }
    }

    configs
}

fn load_configs_file(path: &Path) -> Vec<McpServerConfig> {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_configs_missing_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let configs = load_configs(dir.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn load_configs_malformed_json_returns_empty() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("mcp_servers.json"), "not json at all").unwrap();
        let configs = load_configs(dir.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn load_configs_valid_json_parses_entries() {
        let dir = TempDir::new().unwrap();
        let json = r#"[
            {"name": "my-server", "command": "/usr/bin/my-mcp", "args": ["--port", "9000"]}
        ]"#;
        std::fs::write(dir.path().join("mcp_servers.json"), json).unwrap();
        let configs = load_configs(dir.path());
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "my-server");
        assert_eq!(configs[0].command, "/usr/bin/my-mcp");
        assert_eq!(configs[0].args, vec!["--port", "9000"]);
    }

    #[test]
    fn load_configs_multiple_entries() {
        let dir = TempDir::new().unwrap();
        let json = r#"[
            {"name": "alpha", "command": "/bin/alpha", "args": []},
            {"name": "beta",  "command": "/bin/beta",  "args": ["--verbose"]}
        ]"#;
        std::fs::write(dir.path().join("mcp_servers.json"), json).unwrap();
        let configs = load_configs(dir.path());
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].name, "alpha");
        assert_eq!(configs[1].name, "beta");
        assert!(configs[1].args.contains(&"--verbose".to_string()));
    }

    #[test]
    fn load_configs_empty_array_returns_empty() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("mcp_servers.json"), "[]").unwrap();
        let configs = load_configs(dir.path());
        assert!(configs.is_empty());
    }

    #[test]
    fn mcp_server_config_round_trips_serde() {
        let cfg = McpServerConfig {
            name: "test".to_string(),
            command: "/bin/test".to_string(),
            args: vec!["--flag".to_string()],
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let loaded: McpServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.name, cfg.name);
        assert_eq!(loaded.command, cfg.command);
        assert_eq!(loaded.args, cfg.args);
    }

    #[test]
    fn load_configs_with_project_includes_project_local_entries() {
        let global = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        std::fs::create_dir_all(project.path().join(".dm")).unwrap();
        std::fs::write(
            project.path().join(".dm/mcp_servers.json"),
            r#"[{"name":"host-mcp","command":"node","args":["server.js"]}]"#,
        )
        .unwrap();

        let configs = load_configs_with_project(global.path(), project.path());
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "host-mcp");
        assert_eq!(configs[0].command, "node");
        assert_eq!(configs[0].args, vec!["server.js"]);
    }

    #[test]
    fn load_configs_with_project_merges_global_then_local() {
        let global = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        std::fs::create_dir_all(project.path().join(".dm")).unwrap();
        std::fs::write(
            global.path().join("mcp_servers.json"),
            r#"[
                {"name":"global-only","command":"/bin/global","args":[]},
                {"name":"override-me","command":"/bin/global-old","args":[]}
            ]"#,
        )
        .unwrap();
        std::fs::write(
            project.path().join(".dm/mcp_servers.json"),
            r#"[
                {"name":"override-me","command":"/bin/local-new","args":["--host"]},
                {"name":"local-only","command":"/bin/local","args":[]}
            ]"#,
        )
        .unwrap();

        let configs = load_configs_with_project(global.path(), project.path());
        assert_eq!(configs.len(), 3);
        assert_eq!(configs[0].name, "global-only");
        assert_eq!(configs[1].name, "override-me");
        assert_eq!(configs[1].command, "/bin/local-new");
        assert_eq!(configs[1].args, vec!["--host"]);
        assert_eq!(configs[2].name, "local-only");
    }
}
