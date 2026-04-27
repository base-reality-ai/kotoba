//! Persistent cross-session user memory.
//!
//! Manages the `memory.json` store for recording long-term facts,
//! preferences, and contextual knowledge surfaced to agents.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const MAX_MEMORY_BYTES: usize = 8 * 1024; // 8 KB

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MemoryEntry {
    pub timestamp: DateTime<Utc>,
    pub summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ProjectMemory {
    pub entries: Vec<MemoryEntry>,
}

impl ProjectMemory {
    pub fn load(config_dir: &Path, project_hash: &str) -> Result<Self> {
        let path = memory_path(config_dir, project_hash)?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn save(&self, config_dir: &Path, project_hash: &str) -> Result<()> {
        let path = memory_path(config_dir, project_hash)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = path.with_extension("json.tmp");
        std::fs::write(&tmp_path, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp_path, &path)?;
        Ok(())
    }

    pub fn append(&mut self, summary: &str) {
        self.entries.push(MemoryEntry {
            timestamp: Utc::now(),
            summary: summary.trim().to_string(),
        });
        self.trim_to_size();
    }

    fn trim_to_size(&mut self) {
        while self.serialized_size() > MAX_MEMORY_BYTES && !self.entries.is_empty() {
            self.entries.remove(0);
        }
    }

    fn serialized_size(&self) -> usize {
        serde_json::to_string(self).map(|s| s.len()).unwrap_or(0)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns a formatted system message block, or None if no entries.
    pub fn to_system_message(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let mut msg = String::from("<project_memory>\n");
        for entry in &self.entries {
            let ts = entry.timestamp.format("%Y-%m-%d");
            writeln!(msg, "[{}] {}", ts, entry.summary).expect("write to String never fails");
        }
        msg.push_str("</project_memory>");
        Some(msg)
    }

    pub fn file_path(config_dir: &Path, project_hash: &str) -> Result<PathBuf> {
        memory_path(config_dir, project_hash)
    }
}

fn validate_project_hash(hash: &str) -> Result<()> {
    if hash.is_empty() {
        anyhow::bail!("Project hash cannot be empty");
    }
    if !hash
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        anyhow::bail!(
            "Invalid project hash '{}': only alphanumeric characters, dashes, and underscores are allowed",
            hash
        );
    }
    Ok(())
}

fn memory_path(config_dir: &Path, project_hash: &str) -> Result<PathBuf> {
    validate_project_hash(project_hash)?;
    Ok(config_dir
        .join("projects")
        .join(project_hash)
        .join("memory.json"))
}

/// If the session has ≥3 user turns, ask the model to summarize and append to memory.
pub async fn maybe_update_memory(
    session: &crate::session::Session,
    client: &crate::ollama::client::OllamaClient,
    config_dir: &Path,
    project_hash: &str,
) -> Result<()> {
    let user_turns = session
        .messages
        .iter()
        .filter(|m| m["role"].as_str() == Some("user"))
        .count();

    if user_turns < 3 {
        return Ok(());
    }

    // Build a compact transcript (truncated per message)
    let mut transcript = String::new();
    for msg in &session.messages {
        let role = msg["role"].as_str().unwrap_or("unknown");
        if role == "system" {
            continue;
        }
        let content = msg["content"].as_str().unwrap_or("");
        let mut snippet_end = content.len().min(400);
        while snippet_end > 0 && !content.is_char_boundary(snippet_end) {
            snippet_end -= 1;
        }
        let snippet = &content[..snippet_end];
        write!(transcript, "{}: {}\n\n", role, snippet).expect("write to String never fails");
    }

    let summarize_prompt = format!(
        "Summarize the key decisions, discoveries, and progress from this coding session \
         in 2-3 sentences. Focus on what was built, what was learned, and what remains. \
         Be specific about file names and feature names. Do not use bullet points.\n\n\
         Session:\n{transcript}"
    );

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": summarize_prompt,
    })];

    let resp = client.chat(&messages, &[]).await?;
    let summary = resp.message.content.trim().to_string();

    if summary.is_empty() {
        return Ok(());
    }

    let mut memory = ProjectMemory::load(config_dir, project_hash)?;
    memory.append(&summary);
    memory.save(config_dir, project_hash)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_memory(entries: &[&str]) -> ProjectMemory {
        let mut m = ProjectMemory::default();
        for s in entries {
            m.entries.push(MemoryEntry {
                timestamp: Utc::now(),
                summary: s.to_string(),
            });
        }
        m
    }

    #[test]
    fn test_memory_roundtrip() {
        let tmp = tempdir().unwrap();
        let hash = "testhash";
        let mut mem = ProjectMemory::default();
        mem.append("Built the indexer module.");
        mem.save(tmp.path(), hash).unwrap();

        let loaded = ProjectMemory::load(tmp.path(), hash).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].summary, "Built the indexer module.");
    }

    #[test]
    fn test_memory_load_missing_returns_empty() {
        let tmp = tempdir().unwrap();
        let mem = ProjectMemory::load(tmp.path(), "nohash").unwrap();
        assert!(mem.entries.is_empty());
    }

    #[test]
    fn test_memory_to_system_message_empty() {
        let mem = ProjectMemory::default();
        assert!(mem.to_system_message().is_none());
    }

    #[test]
    fn test_memory_to_system_message_formats() {
        let mem = make_memory(&["Shipped GPU stats.", "Added context window."]);
        let msg = mem.to_system_message().unwrap();
        assert!(msg.contains("<project_memory>"), "should have opening tag");
        assert!(msg.contains("Shipped GPU stats."));
        assert!(msg.contains("Added context window."));
        assert!(msg.contains("</project_memory>"), "should have closing tag");
    }

    #[test]
    fn test_memory_clear() {
        let mut mem = make_memory(&["entry1", "entry2"]);
        mem.clear();
        assert!(mem.entries.is_empty());
        assert!(mem.to_system_message().is_none());
    }

    #[test]
    fn test_memory_trim_to_8kb() {
        let tmp = tempdir().unwrap();
        let hash = "trimhash";
        let mut mem = ProjectMemory::default();
        // Add many large entries to exceed 8KB
        for i in 0..50 {
            mem.append(&format!("Session {i}: {}", "x".repeat(300)));
        }
        mem.save(tmp.path(), hash).unwrap();

        let loaded = ProjectMemory::load(tmp.path(), hash).unwrap();
        // Should have been trimmed
        let size = serde_json::to_string(&loaded).unwrap().len();
        assert!(
            size <= 8 * 1024,
            "memory should be trimmed to ≤8KB, got {size}"
        );
    }

    #[test]
    fn test_maybe_update_memory_skips_short_sessions() {
        // Test the user_turns < 3 guard directly
        let now = Utc::now();
        let session = crate::session::Session {
            id: "test-session".to_string(),
            cwd: "/tmp".to_string(),
            host_project: None,
            model: "test".to_string(),
            messages: vec![
                serde_json::json!({"role": "user", "content": "hello"}),
                serde_json::json!({"role": "assistant", "content": "hi"}),
            ],
            title: None,
            created_at: now,
            updated_at: now,
            compact_failures: 0,
            turn_count: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: None,
        };
        let user_turns = session
            .messages
            .iter()
            .filter(|m| m["role"].as_str() == Some("user"))
            .count();
        assert_eq!(user_turns, 1);
        assert!(
            user_turns < 3,
            "short session should not trigger memory update"
        );
    }

    #[test]
    fn test_memory_append_trims_whitespace() {
        let mut mem = ProjectMemory::default();
        mem.append("  summary with spaces  \n");
        assert_eq!(mem.entries[0].summary, "summary with spaces");
    }

    #[test]
    fn test_memory_file_path_is_under_projects() {
        let dir = std::path::Path::new("/tmp/dm_test");
        let path = ProjectMemory::file_path(dir, "abc123").unwrap();
        assert!(path.starts_with(dir.join("projects").join("abc123")));
        assert_eq!(path.file_name().unwrap(), "memory.json");
    }

    #[test]
    fn test_memory_append_multiple_entries_ordered() {
        let mut mem = ProjectMemory::default();
        mem.append("first");
        mem.append("second");
        mem.append("third");
        assert_eq!(mem.entries.len(), 3);
        assert_eq!(mem.entries[0].summary, "first");
        assert_eq!(mem.entries[2].summary, "third");
    }

    #[test]
    fn test_memory_to_system_message_contains_date_format() {
        let mem = make_memory(&["some entry"]);
        let msg = mem.to_system_message().unwrap();
        // Should contain a date in [YYYY-MM-DD] format
        assert!(
            msg.contains('['),
            "should have opening bracket for date: {msg}"
        );
        assert!(
            msg.contains(']'),
            "should have closing bracket for date: {msg}"
        );
    }

    #[test]
    fn save_memory_atomic_no_tmp_left() {
        let tmp = tempdir().unwrap();
        let hash = "atomichash";
        let mut mem = ProjectMemory::default();
        mem.append("atomic write test");
        mem.save(tmp.path(), hash).unwrap();

        let mem_dir = tmp.path().join("projects").join(hash);
        assert!(
            mem_dir.join("memory.json").exists(),
            "memory file should exist"
        );
        assert!(
            !mem_dir.join("memory.json.tmp").exists(),
            "tmp file should be cleaned up"
        );

        let loaded = ProjectMemory::load(tmp.path(), hash).unwrap();
        assert_eq!(loaded.entries[0].summary, "atomic write test");
    }

    #[test]
    fn validate_project_hash_accepts_normal() {
        assert!(validate_project_hash("abc123").is_ok());
        assert!(validate_project_hash("my-project_v2").is_ok());
        assert!(validate_project_hash("ABC").is_ok());
    }

    #[test]
    fn validate_project_hash_rejects_traversal() {
        assert!(validate_project_hash("../../etc").is_err());
    }

    #[test]
    fn validate_project_hash_rejects_slashes() {
        assert!(validate_project_hash("foo/bar").is_err());
        assert!(validate_project_hash("a\\b").is_err());
    }

    #[test]
    fn validate_project_hash_rejects_empty() {
        assert!(validate_project_hash("").is_err());
    }

    #[test]
    fn test_memory_append_empty_trims_to_empty_string() {
        let mut mem = ProjectMemory::default();
        mem.append("   ");
        assert_eq!(
            mem.entries[0].summary, "",
            "empty whitespace-only string should trim to empty"
        );
    }
}
