//! Session record + persistence. A `Session` captures one dm conversation
//! — cwd, host identity, model, messages, compaction-failure breaker state,
//! parent lineage — so it can be paused, snapshotted, replayed, and forked.
//!
//! `storage` reads/writes session JSON under `~/.dm/sessions/`; `replay`
//! reconstructs the message stream for daemon resume and `/sessions tree`.

pub mod replay;
pub mod storage;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub cwd: String,
    /// Host project active when this session was created. `None` means kernel
    /// mode or an older session JSON written before identity tracking.
    #[serde(default)]
    pub host_project: Option<String>,
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    /// Consecutive Stage-3 (full-summary) compaction failures. The circuit
    /// breaker in `compact_pipeline_with_failures` trips at
    /// `MAX_COMPACT_FAILURES` and falls back to emergency compact. Persisting
    /// this across `run_conversation` calls is what makes the breaker actually
    /// work in daemon and multi-turn TUI use — a per-call local would reset
    /// every turn and the breaker would never trip.
    ///
    /// `#[serde(default)]` lets older session JSON files (written before this
    /// field existed) deserialize cleanly at 0.
    #[serde(default)]
    pub compact_failures: usize,
    #[serde(default)]
    pub turn_count: u32,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    /// ID of the parent session this was forked from, if any. `None` for
    /// root sessions and for sessions written by dm versions predating
    /// lineage tracking (`#[serde(default)]` loads those as `None`).
    /// Used by `/sessions tree` to reconstruct the branching graph — the
    /// chain is a linked list, so a tree walker recovers the root by
    /// following `parent_id` until it reaches `None`.
    #[serde(default)]
    pub parent_id: Option<String>,
}

/// Return the first 8 bytes of a session id (or the full id if shorter) as a
/// borrowed slice. Centralizes the "8-char short id" UX decision used in TUI
/// status lines, log messages, filenames, web server responses, and signal-
/// exit banners. Empty-safe and short-id-safe by construction.
pub fn short_id(id: &str) -> &str {
    &id[..8.min(id.len())]
}

impl Session {
    pub fn new(cwd: String, model: String) -> Self {
        let now = Utc::now();
        let host_project = crate::identity::load_at(Path::new(&cwd))
            .ok()
            .and_then(|identity| identity.host_project);
        Session {
            id: Uuid::new_v4().to_string(),
            title: None,
            created_at: now,
            updated_at: now,
            cwd,
            host_project,
            model,
            messages: Vec::new(),
            compact_failures: 0,
            turn_count: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: None,
        }
    }

    pub fn push_message(&mut self, msg: serde_json::Value) {
        self.messages.push(msg);
        self.updated_at = Utc::now();
    }

    /// Create a forked session from this one.
    /// If `at_turn` is None, copies all messages.
    /// If `at_turn` is Some(n), includes messages up through the nth user turn
    /// and any assistant response immediately following it.
    pub fn fork(&self, at_turn: Option<usize>) -> Session {
        let now = Utc::now();
        let title = Some(format!(
            "{} (fork)",
            self.title.as_deref().unwrap_or("untitled")
        ));

        let messages = match at_turn {
            None => self.messages.clone(),
            Some(n) => {
                let mut user_count = 0usize;
                let mut cut = 0usize;
                for (i, msg) in self.messages.iter().enumerate() {
                    if msg["role"].as_str() == Some("user") {
                        user_count += 1;
                        if user_count == n {
                            // Include this user message and any assistant response after it
                            cut = i + 1;
                            if let Some(next) = self.messages.get(i + 1) {
                                if next["role"].as_str() == Some("assistant") {
                                    cut = i + 2;
                                }
                            }
                            break;
                        }
                    }
                }
                if user_count < n {
                    self.messages.clone()
                } else {
                    self.messages[..cut].to_vec()
                }
            }
        };

        Session {
            id: Uuid::new_v4().to_string(),
            title,
            created_at: now,
            updated_at: now,
            cwd: self.cwd.clone(),
            host_project: self.host_project.clone(),
            model: self.model.clone(),
            messages,
            compact_failures: 0,
            turn_count: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: Some(self.id.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_new_has_no_title() {
        let s = Session::new("/tmp".to_string(), "model".to_string());
        assert!(s.title.is_none());
    }

    #[test]
    fn session_new_records_host_project_from_cwd_identity() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dm = tmp.path().join(".dm");
        std::fs::create_dir_all(&dm).unwrap();
        std::fs::write(
            dm.join(crate::identity::IDENTITY_FILENAME),
            "mode = \"host\"\nhost_project = \"finance-app\"\n",
        )
        .unwrap();

        let s = Session::new(tmp.path().display().to_string(), "model".to_string());

        assert_eq!(s.host_project.as_deref(), Some("finance-app"));
    }

    #[test]
    fn legacy_session_json_defaults_host_project_to_none() {
        let now = Utc::now().to_rfc3339();
        let json = serde_json::json!({
            "id": "legacy",
            "title": null,
            "created_at": now,
            "updated_at": now,
            "cwd": "/tmp",
            "model": "model",
            "messages": []
        });

        let loaded: Session = serde_json::from_value(json).unwrap();

        assert_eq!(loaded.host_project, None);
    }

    #[test]
    fn session_title_round_trips_via_serde() {
        let mut s = Session::new("/tmp".to_string(), "model".to_string());
        s.title = Some("My Test Session".to_string());
        let json = serde_json::to_string(&s).unwrap();
        let loaded: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.title.as_deref(), Some("My Test Session"));
    }

    fn make_session_with_messages() -> Session {
        let mut s = Session::new("/tmp".to_string(), "test-model".to_string());
        s.title = Some("test".to_string());
        s.messages
            .push(serde_json::json!({"role": "system", "content": "You are helpful."}));
        s.messages
            .push(serde_json::json!({"role": "user", "content": "Hello"}));
        s.messages
            .push(serde_json::json!({"role": "assistant", "content": "Hi there!"}));
        s.messages
            .push(serde_json::json!({"role": "user", "content": "How are you?"}));
        s.messages
            .push(serde_json::json!({"role": "assistant", "content": "Great!"}));
        s.messages
            .push(serde_json::json!({"role": "user", "content": "Bye"}));
        s
    }

    #[test]
    fn fork_copies_all_messages() {
        let s = make_session_with_messages();
        let forked = s.fork(None);
        assert_eq!(forked.messages.len(), s.messages.len());
    }

    #[test]
    fn fork_at_turn() {
        let s = make_session_with_messages();
        // Fork at turn 2 should include system + user1 + assistant1 + user2 + assistant2
        let forked = s.fork(Some(2));
        assert_eq!(forked.messages.len(), 5);
        assert_eq!(forked.messages[3]["content"], "How are you?");
        assert_eq!(forked.messages[4]["content"], "Great!");
    }

    #[test]
    fn fork_new_id() {
        let s = make_session_with_messages();
        let forked = s.fork(None);
        assert_ne!(forked.id, s.id);
    }

    #[test]
    fn fork_preserves_model_and_cwd() {
        let s = make_session_with_messages();
        let forked = s.fork(None);
        assert_eq!(forked.model, s.model);
        assert_eq!(forked.cwd, s.cwd);
        assert_eq!(forked.host_project, s.host_project);
        assert_eq!(forked.title.as_deref(), Some("test (fork)"));
    }

    #[test]
    fn fork_empty_session() {
        let s = Session::new("/tmp".to_string(), "model".to_string());
        let forked = s.fork(None);
        assert!(forked.messages.is_empty());
        assert_eq!(forked.title.as_deref(), Some("untitled (fork)"));
    }

    #[test]
    fn fork_at_turn_beyond_end() {
        let s = make_session_with_messages();
        // Only 3 user turns; fork at 10 should copy all
        let forked = s.fork(Some(10));
        assert_eq!(forked.messages.len(), s.messages.len());
    }

    #[test]
    fn old_session_without_title_deserializes_cleanly() {
        // Simulate an old session JSON without the `title` field
        let json = r#"{
            "id": "test-id",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "cwd": "/tmp",
            "model": "gemma4:26b",
            "messages": []
        }"#;
        let s: Session = serde_json::from_str(json).unwrap();
        assert!(s.title.is_none());
    }

    #[test]
    fn session_compact_failures_defaults_to_zero() {
        let s = Session::new("/tmp".to_string(), "model".to_string());
        assert_eq!(s.compact_failures, 0);
    }

    #[test]
    fn session_roundtrips_compact_failures_through_serde() {
        let mut s = Session::new("/tmp".to_string(), "model".to_string());
        s.compact_failures = 2;
        let json = serde_json::to_string(&s).unwrap();
        let loaded: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.compact_failures, 2);
    }

    #[test]
    fn session_missing_compact_failures_field_defaults_to_zero() {
        // Old session JSON written before `compact_failures` existed must
        // still load cleanly — that's the whole backwards-compat story for
        // the circuit-breaker persistence change.
        let json = r#"{
            "id": "old-id",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "cwd": "/tmp",
            "model": "gemma4:26b",
            "messages": []
        }"#;
        let s: Session = serde_json::from_str(json).unwrap();
        assert_eq!(s.compact_failures, 0);
    }

    #[test]
    fn session_fork_resets_compact_failures_counter() {
        // Fork produces a fresh session with a clean circuit-breaker state —
        // a user forking off a session that hit the breaker shouldn't inherit
        // the trip count. `Session::fork` line 98 is the only thing locking
        // this invariant and the builder added no direct coverage for it.
        let mut s = make_session_with_messages();
        s.compact_failures = MAX_COMPACT_FAILURES_FOR_TEST;
        let forked = s.fork(None);
        assert_eq!(
            forked.compact_failures, 0,
            "fork must reset failure counter regardless of parent"
        );
        // Parent counter stays untouched by fork.
        assert_eq!(s.compact_failures, MAX_COMPACT_FAILURES_FOR_TEST);
    }

    #[test]
    fn session_fork_at_turn_resets_compact_failures_counter() {
        // Same invariant, but via the `Some(n)` branch of fork — separate code
        // path at the match arm, separate initializer at line 98.
        let mut s = make_session_with_messages();
        s.compact_failures = 7;
        let forked = s.fork(Some(2));
        assert_eq!(
            forked.compact_failures, 0,
            "fork(at_turn) must reset failure counter"
        );
    }

    #[test]
    fn session_roundtrips_large_compact_failures_value() {
        // The field is `usize`, so serde must carry any in-bounds value. A
        // runaway counter (e.g. if the breaker logic regressed and failed to
        // cap) must still round-trip cleanly rather than overflow on load.
        let mut s = Session::new("/tmp".to_string(), "model".to_string());
        s.compact_failures = 1_000_000;
        let json = serde_json::to_string(&s).unwrap();
        let loaded: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.compact_failures, 1_000_000);
    }

    // Mirror of `compaction::MAX_COMPACT_FAILURES` for the fork test — keeping
    // a local const avoids a cross-module dep just for a test constant.
    const MAX_COMPACT_FAILURES_FOR_TEST: usize = 3;

    #[test]
    fn session_new_initializes_turn_and_token_counters_to_zero() {
        let s = Session::new("/tmp".to_string(), "model".to_string());
        assert_eq!(s.turn_count, 0);
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.completion_tokens, 0);
    }

    #[test]
    fn session_missing_turn_and_token_fields_default_to_zero() {
        // Old session JSON written before turn_count / prompt_tokens /
        // completion_tokens existed must still load cleanly — the /sessions
        // listing relies on `#[serde(default)]` to show old rows with a
        // fallback estimate instead of failing to parse.
        let json = r#"{
            "id": "old-id",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "cwd": "/tmp",
            "model": "gemma4:26b",
            "messages": []
        }"#;
        let s: Session = serde_json::from_str(json).unwrap();
        assert_eq!(s.turn_count, 0);
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.completion_tokens, 0);
    }

    #[test]
    fn session_roundtrips_turn_and_token_counters() {
        let mut s = Session::new("/tmp".to_string(), "model".to_string());
        s.turn_count = 17;
        s.prompt_tokens = 12_345;
        s.completion_tokens = 6_789;
        let json = serde_json::to_string(&s).unwrap();
        let loaded: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.turn_count, 17);
        assert_eq!(loaded.prompt_tokens, 12_345);
        assert_eq!(loaded.completion_tokens, 6_789);
    }

    #[test]
    fn session_fork_resets_turn_and_token_counters() {
        let mut s = make_session_with_messages();
        s.turn_count = 10;
        s.prompt_tokens = 5_000;
        s.completion_tokens = 2_000;
        let forked = s.fork(None);
        assert_eq!(forked.turn_count, 0);
        assert_eq!(forked.prompt_tokens, 0);
        assert_eq!(forked.completion_tokens, 0);
    }

    #[test]
    fn push_message_updates_timestamp() {
        let mut session = Session::new("/tmp".to_string(), "test-model".to_string());
        let before = session.updated_at;
        std::thread::sleep(std::time::Duration::from_millis(10));
        session.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        assert!(
            session.updated_at > before,
            "updated_at should advance after push_message"
        );
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn session_new_has_no_parent_id() {
        let s = Session::new("/tmp".to_string(), "model".to_string());
        assert!(s.parent_id.is_none(), "root session has no parent");
    }

    #[test]
    fn fork_sets_parent_id_to_source() {
        let s = make_session_with_messages();
        let forked = s.fork(None);
        assert_eq!(
            forked.parent_id.as_deref(),
            Some(s.id.as_str()),
            "fork must record its source in parent_id"
        );
    }

    #[test]
    fn fork_of_fork_chains_parents() {
        // Nested fork: grandchild.parent_id == child.id (NOT root.id).
        // The chain is a linked list; tree-walkers recover the root by
        // iterating until parent_id is None.
        let root = make_session_with_messages();
        let child = root.fork(None);
        let grandchild = child.fork(None);
        assert_eq!(grandchild.parent_id.as_deref(), Some(child.id.as_str()));
        assert_ne!(
            grandchild.parent_id.as_deref(),
            Some(root.id.as_str()),
            "grandchild points at its direct parent, not the root"
        );
    }

    #[test]
    fn parent_id_round_trips_via_serde() {
        let s = make_session_with_messages();
        let forked = s.fork(None);
        let json = serde_json::to_string(&forked).unwrap();
        let loaded: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.parent_id, forked.parent_id);
    }

    #[test]
    fn session_json_without_parent_id_loads_as_none() {
        // Backwards-compat: pre-parent_id session files omit the field.
        // #[serde(default)] must yield None (not error) — mirrors the
        // compact_failures precedent at the top of this module.
        let json = r#"{
            "id": "abc-123",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "cwd": "/tmp",
            "model": "test",
            "messages": []
        }"#;
        let loaded: Session = serde_json::from_str(json).expect("legacy session must load");
        assert!(
            loaded.parent_id.is_none(),
            "legacy session without field → None"
        );
        assert_eq!(loaded.id, "abc-123");
    }

    #[test]
    fn short_id_truncates_long_ids_to_8_bytes() {
        assert_eq!(super::short_id("abcdefghijklmnop"), "abcdefgh");
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        assert_eq!(super::short_id(uuid), "550e8400");
    }

    #[test]
    fn short_id_returns_full_id_when_shorter_than_8() {
        assert_eq!(super::short_id("abcd"), "abcd");
        assert_eq!(super::short_id("abcdefgh"), "abcdefgh");
    }

    #[test]
    fn short_id_handles_empty_string_without_panic() {
        assert_eq!(super::short_id(""), "");
    }
}
