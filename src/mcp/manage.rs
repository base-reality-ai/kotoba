use crate::logging;
use anyhow::Context;
use std::collections::HashMap;
use std::path::Path;

/// One entry in `~/.dm/mcp_servers.json` (manage format with env and enabled fields)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McpServerEntry {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// Load MCP server entries from `config_dir/mcp_servers.json`.
/// Returns `Ok(vec![])` if the file does not exist.
pub fn load_mcp_config(config_dir: &Path) -> anyhow::Result<Vec<McpServerEntry>> {
    let path = config_dir.join("mcp_servers.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read mcp_servers.json: {}", path.display()))?;
    // Try parsing as array of McpServerEntry first; fall back to legacy McpServerConfig format
    if let Ok(entries) = serde_json::from_str::<Vec<McpServerEntry>>(&content) {
        return Ok(entries);
    }
    // Try legacy format (name + command + args, no env/enabled)
    #[derive(serde::Deserialize)]
    struct Legacy {
        name: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
    }
    let legacy: Vec<Legacy> =
        serde_json::from_str(&content).with_context(|| "Cannot parse mcp_servers.json")?;
    Ok(legacy
        .into_iter()
        .map(|l| McpServerEntry {
            name: l.name,
            command: l.command,
            args: l.args,
            env: HashMap::new(),
            enabled: true,
        })
        .collect())
}

/// Save MCP server entries to `config_dir/mcp_servers.json` atomically via tmp+rename.
pub fn save_mcp_config(config_dir: &Path, entries: &[McpServerEntry]) -> anyhow::Result<()> {
    let path = config_dir.join("mcp_servers.json");
    let tmp_path = path.with_extension("json.tmp");
    let json =
        serde_json::to_string_pretty(entries).context("Cannot serialize mcp_servers config")?;
    std::fs::write(&tmp_path, &json)
        .with_context(|| format!("Cannot write tmp mcp config: {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("Cannot rename mcp config: {}", path.display()))?;
    Ok(())
}

/// Add or update an MCP server entry. If a server with the same name exists, it is replaced.
pub fn add_mcp_server(config_dir: &Path, new_entry: McpServerEntry) -> anyhow::Result<()> {
    let mut entries = load_mcp_config(config_dir)?;
    if let Some(existing) = entries.iter_mut().find(|e| e.name == new_entry.name) {
        logging::log(&format!(
            "dm: updated existing MCP server '{}'",
            new_entry.name
        ));
        *existing = new_entry;
    } else {
        entries.push(new_entry);
    }
    save_mcp_config(config_dir, &entries)
}

/// Split a command string into program and arguments.
/// e.g. `"npx foo bar"` → `("npx", ["foo", "bar"])`
pub fn parse_command(cmd: &str) -> (String, Vec<String>) {
    let mut parts = cmd.split_whitespace();
    let prog = parts.next().unwrap_or("").to_string();
    let args: Vec<String> = parts.map(|s| s.to_string()).collect();
    (prog, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_config_roundtrips_json() {
        let entry = McpServerEntry {
            name: "filesystem".to_string(),
            command: "npx".to_string(),
            args: vec![
                "@modelcontextprotocol/server-filesystem".to_string(),
                "/tmp".to_string(),
            ],
            env: HashMap::new(),
            enabled: true,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        let loaded: McpServerEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.name, entry.name);
        assert_eq!(loaded.command, entry.command);
        assert_eq!(loaded.args, entry.args);
        assert_eq!(loaded.enabled, entry.enabled);
    }

    #[test]
    fn mcp_disable_sets_enabled_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entries = vec![McpServerEntry {
            name: "my-server".to_string(),
            command: "npx".to_string(),
            args: vec![],
            env: HashMap::new(),
            enabled: true,
        }];
        save_mcp_config(dir.path(), &entries).expect("save");

        let mut loaded = load_mcp_config(dir.path()).expect("load");
        if let Some(entry) = loaded.iter_mut().find(|e| e.name == "my-server") {
            entry.enabled = false;
        }
        save_mcp_config(dir.path(), &loaded).expect("save disabled");

        let reloaded = load_mcp_config(dir.path()).expect("reload");
        let found = reloaded
            .iter()
            .find(|e| e.name == "my-server")
            .expect("find");
        assert!(!found.enabled);
    }

    #[test]
    fn mcp_add_appends_to_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entries = vec![McpServerEntry {
            name: "server-one".to_string(),
            command: "npx".to_string(),
            args: vec![],
            env: HashMap::new(),
            enabled: true,
        }];
        save_mcp_config(dir.path(), &entries).expect("save");

        let mut loaded = load_mcp_config(dir.path()).expect("load");
        loaded.push(McpServerEntry {
            name: "server-two".to_string(),
            command: "node".to_string(),
            args: vec!["server.js".to_string()],
            env: HashMap::new(),
            enabled: true,
        });
        save_mcp_config(dir.path(), &loaded).expect("save with new");

        let reloaded = load_mcp_config(dir.path()).expect("reload");
        assert_eq!(reloaded.len(), 2);
    }

    #[test]
    fn parse_command_splits_correctly() {
        let (prog, args) = parse_command("npx @modelcontextprotocol/server-filesystem /tmp");
        assert_eq!(prog, "npx");
        assert_eq!(
            args,
            vec!["@modelcontextprotocol/server-filesystem", "/tmp"]
        );
    }

    #[test]
    fn load_mcp_config_returns_empty_when_file_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = load_mcp_config(dir.path());
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn parse_command_empty_string_returns_empty_prog() {
        let (prog, args) = parse_command("");
        assert_eq!(prog, "");
        assert!(args.is_empty());
    }

    #[test]
    fn parse_command_single_word_no_args() {
        let (prog, args) = parse_command("node");
        assert_eq!(prog, "node");
        assert!(args.is_empty());
    }

    #[test]
    fn parse_command_handles_extra_whitespace() {
        let (prog, args) = parse_command("  npx   server.js   --port 3000  ");
        assert_eq!(prog, "npx");
        assert_eq!(args, vec!["server.js", "--port", "3000"]);
    }

    #[test]
    fn save_mcp_config_atomic_no_tmp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entries = vec![McpServerEntry {
            name: "test".to_string(),
            command: "npx".to_string(),
            args: vec![],
            env: HashMap::new(),
            enabled: true,
        }];
        save_mcp_config(dir.path(), &entries).expect("save");
        let tmp = dir.path().join("mcp_servers.json.tmp");
        assert!(!tmp.exists(), "tmp file should be cleaned up after rename");
        let main = dir.path().join("mcp_servers.json");
        assert!(main.exists(), "main file should exist");
    }

    #[test]
    fn add_mcp_server_replaces_duplicate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entry1 = McpServerEntry {
            name: "my-server".to_string(),
            command: "npx".to_string(),
            args: vec!["old-pkg".to_string()],
            env: HashMap::new(),
            enabled: true,
        };
        add_mcp_server(dir.path(), entry1).expect("first add");

        let entry2 = McpServerEntry {
            name: "my-server".to_string(),
            command: "node".to_string(),
            args: vec!["new-server.js".to_string()],
            env: HashMap::new(),
            enabled: true,
        };
        add_mcp_server(dir.path(), entry2).expect("second add");

        let entries = load_mcp_config(dir.path()).expect("load");
        assert_eq!(
            entries.len(),
            1,
            "duplicate should be replaced, not appended"
        );
        assert_eq!(entries[0].command, "node");
        assert_eq!(entries[0].args, vec!["new-server.js"]);
    }
}
