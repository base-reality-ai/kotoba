//! Configuration and loading of named agents.
//!
//! Defines schemas for agent presets (system prompts, tool restrictions)
//! stored in the `~/.dm/agents/` registry.

use anyhow::Context;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct AgentConfig {
    pub name: String,
    pub model: Option<String>,
    pub system_extra: Option<String>,
    pub tools: Option<Vec<String>>, // None = all tools
    pub no_memory: bool,
    pub description: Option<String>,
}

impl AgentConfig {
    pub fn load(config_dir: &Path, name: &str) -> anyhow::Result<Self> {
        let path = Self::agent_path(config_dir, name);
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Cannot read agent config: {}", path.display()))?;
        let cfg: AgentConfig = serde_json::from_str(&content)
            .with_context(|| format!("Invalid JSON in agent config: {}", path.display()))?;
        Ok(cfg)
    }

    pub fn save(&self, config_dir: &Path) -> anyhow::Result<()> {
        let agents_dir = config_dir.join("agents");
        std::fs::create_dir_all(&agents_dir).context("Cannot create agents directory")?;
        let slug = Self::slugify(&self.name);
        let path = agents_dir.join(format!("{}.json", slug));
        let json = serde_json::to_string_pretty(self).context("Cannot serialize agent config")?;
        std::fs::write(&path, json)
            .with_context(|| format!("Cannot write agent config: {}", path.display()))?;
        Ok(())
    }

    #[must_use]
    pub fn list(config_dir: &Path) -> Vec<AgentConfig> {
        let agents_dir = config_dir.join("agents");
        let Ok(read_dir) = std::fs::read_dir(&agents_dir) else {
            return Vec::new();
        };
        let mut result = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(cfg) = serde_json::from_str::<AgentConfig>(&content) {
                        result.push(cfg);
                    }
                }
            }
        }
        result
    }

    #[must_use]
    pub fn agent_path(config_dir: &Path, name: &str) -> PathBuf {
        let slug = Self::slugify(name);
        config_dir.join("agents").join(format!("{}.json", slug))
    }

    #[must_use]
    pub fn slugify(name: &str) -> String {
        if name.is_empty() {
            return "unnamed".to_string();
        }
        let slug: String = name
            .to_lowercase()
            .chars()
            .map(|c| if c == ' ' { '-' } else { c })
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .collect();
        // Collapse consecutive hyphens and trim
        let slug = slug
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-");
        if slug.is_empty() {
            "unnamed".to_string()
        } else {
            slug
        }
    }

    /// Delete agent file. Returns Err if file does not exist.
    pub fn delete(config_dir: &Path, name: &str) -> anyhow::Result<()> {
        let path = Self::agent_path(config_dir, name);
        std::fs::remove_file(&path)
            .with_context(|| format!("Cannot delete agent '{}': file not found", name))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_config_roundtrips_json() {
        let cfg = AgentConfig {
            name: "test-agent".to_string(),
            model: Some("gemma4:26b".to_string()),
            system_extra: Some("You are helpful".to_string()),
            tools: Some(vec!["bash".to_string(), "read_file".to_string()]),
            no_memory: false,
            description: Some("A test agent".to_string()),
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let loaded: AgentConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.name, cfg.name);
        assert_eq!(loaded.model, cfg.model);
        assert_eq!(loaded.system_extra, cfg.system_extra);
        assert_eq!(loaded.tools, cfg.tools);
        assert_eq!(loaded.no_memory, cfg.no_memory);
        assert_eq!(loaded.description, cfg.description);
    }

    #[test]
    fn agent_list_finds_json_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg1 = AgentConfig {
            name: "agent-one".to_string(),
            ..Default::default()
        };
        let cfg2 = AgentConfig {
            name: "agent-two".to_string(),
            ..Default::default()
        };
        cfg1.save(dir.path()).expect("save cfg1");
        cfg2.save(dir.path()).expect("save cfg2");

        let listed = AgentConfig::list(dir.path());
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn agent_create_name_is_slug() {
        assert_eq!(AgentConfig::slugify("My Cool Agent!"), "my-cool-agent");
    }

    #[test]
    fn agent_tools_restriction_filters_registry() {
        let config = AgentConfig {
            tools: Some(vec!["bash".into()]),
            ..Default::default()
        };
        assert_eq!(config.tools.as_deref(), Some(&["bash".to_string()][..]));
    }

    #[test]
    fn slugify_empty_string_returns_unnamed() {
        assert_eq!(AgentConfig::slugify(""), "unnamed");
    }

    #[test]
    fn slugify_all_special_chars_returns_unnamed() {
        assert_eq!(AgentConfig::slugify("!!!"), "unnamed");
    }

    #[test]
    fn slugify_spaces_become_hyphens() {
        assert_eq!(AgentConfig::slugify("hello world"), "hello-world");
    }

    #[test]
    fn slugify_consecutive_spaces_collapse_to_single_hyphen() {
        assert_eq!(AgentConfig::slugify("hello  world"), "hello-world");
    }

    #[test]
    fn slugify_uppercase_lowercased() {
        assert_eq!(AgentConfig::slugify("MyAgent"), "myagent");
    }

    #[test]
    fn agent_path_uses_slugified_name() {
        let dir = std::path::Path::new("/tmp/dm_test_agents");
        let path = AgentConfig::agent_path(dir, "My Agent");
        assert!(path.to_str().unwrap().contains("my-agent.json"));
        assert!(path.starts_with(dir.join("agents")));
    }

    #[test]
    fn agent_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = AgentConfig {
            name: "save-load-test".to_string(),
            model: Some("gemma4:26b".to_string()),
            system_extra: Some("extra sys".to_string()),
            tools: None,
            no_memory: true,
            description: Some("roundtrip test agent".to_string()),
        };
        cfg.save(dir.path()).expect("save");
        let loaded = AgentConfig::load(dir.path(), "save-load-test").expect("load");
        assert_eq!(loaded.name, cfg.name);
        assert_eq!(loaded.model, cfg.model);
        assert_eq!(loaded.no_memory, cfg.no_memory);
        assert_eq!(loaded.description, cfg.description);
    }

    #[test]
    fn agent_delete_removes_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = AgentConfig {
            name: "to-delete".to_string(),
            ..Default::default()
        };
        cfg.save(dir.path()).expect("save");
        // File should now exist
        let path = AgentConfig::agent_path(dir.path(), "to-delete");
        assert!(path.exists(), "file should exist after save");
        // Delete it
        AgentConfig::delete(dir.path(), "to-delete").expect("delete");
        assert!(!path.exists(), "file should be gone after delete");
    }

    #[test]
    fn agent_delete_nonexistent_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = AgentConfig::delete(dir.path(), "ghost-agent");
        assert!(result.is_err(), "deleting non-existent agent should error");
    }
}
