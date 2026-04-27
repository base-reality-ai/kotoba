pub mod agent;
pub mod apply_diff;
pub mod ask_user_question;
pub mod bash;
pub mod chain;
pub mod file_edit;
pub mod file_read;
pub mod file_write;
pub mod fs_error;
pub mod glob;
pub mod grep;
pub mod hooks;
pub mod ls;
pub mod multi_edit;
pub mod notebook;
pub mod path_safety;
pub mod registry;
pub mod semantic_search;
pub mod todo;
pub mod web_fetch;
pub mod web_search;
pub mod wiki_lookup;
pub mod wiki_search;

use crate::ollama::types::{FunctionDefinition, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// The result of executing a tool
#[derive(Debug)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

/// Every tool implements this trait
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name as Ollama sees it (`snake_case`)
    fn name(&self) -> &'static str;

    /// Human-readable description for the LLM
    fn description(&self) -> &'static str;

    /// JSON Schema for the tool's input parameters
    fn parameters(&self) -> Value;

    /// Execute the tool with the given arguments
    async fn call(&self, args: Value) -> Result<ToolResult>;

    /// Whether this tool is read-only (safe to run concurrently with other
    /// read-only tools). Mutating tools (default) are run one at a time.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Optional usage hint injected into the system prompt so the model
    /// knows *how* to use this tool effectively. Return `None` (default)
    /// if no special guidance is needed beyond the tool description.
    fn system_prompt_hint(&self) -> Option<&'static str> {
        None
    }

    /// Produce the `ToolDefinition` to send to Ollama
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters(),
            },
        }
    }
}

// Shared lock for cwd-dependent code paths. Originally test-only for
// tests that mutate the process-global cwd via `std::env::set_current_dir`;
// promoted to always-on in cycle 15 so production code that *reads* cwd
// (`compaction::full_compact`'s None-arm fallback) can serialize against
// the same lock. Without a single cross-module lock, parallel tests in
// different tool modules can race on cwd and tear down each other's
// tempdirs before cwd is restored, and production cwd reads can race
// against test cwd writes.
//
// Production cost: uncontended (no production thread mutates cwd from
// another thread), so each acquisition is a microsecond uncontended
// mutex op — amortized over the multi-second LLM call that follows.
//
// Invariant: this is the ONLY CWD_LOCK in the codebase. Any test that
// mutates `std::env::current_dir` or reads cwd-dependent state must
// acquire this mutex. Do not introduce module-local CWD_LOCK statics
// — a second mutex defeats the cross-module serialization and allows
// tests in different modules to race on cwd (Cycle 26 identified,
// Cycle 32 closed after two race incidents).
pub(crate) static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

tokio::task_local! {
    /// Marks a tokio task whose outer scope already holds `CWD_LOCK`.
    /// Production cwd readers in `compaction::full_compact`'s and
    /// `compaction::write_emergency_synthesis`'s `None`-arm fallback
    /// check this before acquiring `CWD_LOCK` to avoid re-entrant
    /// deadlock. Set ONLY by tests that need to hold `CWD_LOCK` AND
    /// call `full_compact(.., None)` (cycle 21 unblocked tee tests
    /// 2+3). Production code never sets this — so the fast path
    /// remains "acquire briefly, read, drop" with no behavior
    /// change for production.
    pub(crate) static CWD_LOCK_HELD: ();
}

/// Read `std::env::current_dir()` while serializing against test-pool
/// cwd mutators. If the calling task is inside a `CWD_LOCK_HELD.scope()`
/// (i.e., the outer test already holds `CWD_LOCK`), skip the
/// re-acquisition that would otherwise deadlock — `std::sync::Mutex`
/// is non-reentrant. Otherwise acquire `CWD_LOCK` briefly.
pub(crate) fn read_cwd_synchronized() -> Option<std::path::PathBuf> {
    if CWD_LOCK_HELD.try_with(|_| ()).is_ok() {
        std::env::current_dir().ok()
    } else {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::current_dir().ok()
    }
}

/// After a file write succeeds, update the wiki and check for drift.
///
/// Spawns a blocking task that:
/// 1. Opens the wiki for the current working directory (project root).
/// 2. Ingests the new content for `canonical_path`.
/// 3. Checks whether the source file is newer than its entity page.
///
/// Returns `Some(warning)` when drift is detected, `None` otherwise.
/// Errors during ingest are pushed via `crate::warnings` and do not
/// block the return.
pub async fn post_write_wiki_sync(
    canonical_path: &std::path::Path,
    content: &str,
) -> Option<String> {
    let ingest_path = canonical_path.to_path_buf();
    let ingest_root = std::env::current_dir().unwrap_or_default();
    let ingest_content = content.to_string();
    tokio::task::spawn_blocking(move || {
        let mut warning = None;
        if let Ok(wiki) = crate::wiki::Wiki::open(&ingest_root) {
            if let Err(e) = wiki.ingest_file(&ingest_root, &ingest_path, &ingest_content) {
                crate::warnings::push_warning(format!(
                    "wiki ingest failed for {}: {}",
                    ingest_path.display(),
                    e
                ));
            }
            warning = wiki
                .check_source_drift(&ingest_root, &ingest_path)
                .unwrap_or(None);
        }
        warning
    })
    .await
    .unwrap_or(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn post_write_wiki_sync_detects_drift_when_auto_ingest_disabled() {
        let _guard = CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let file = tmp.path().join("src/main.rs");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "fn main() {}").unwrap();

        // Pre-ingest so entity page exists.
        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();
        wiki.ingest_file(tmp.path(), &file, "fn main() {}").unwrap();

        // Disable auto-ingest so the edit won't update the page.
        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        // Re-write the file (updates mtime) without updating the wiki page.
        fs::write(&file, "fn main() { updated }").unwrap();
        let warning = post_write_wiki_sync(&file, "fn main() { updated }").await;

        // Restore process-global state BEFORE assertions so a panic
        // cannot leak the tempdir cwd or the disabled flag.
        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(
            warning.is_some(),
            "should detect drift when auto-ingest is disabled"
        );
        let w = warning.unwrap();
        assert!(
            w.contains("[wiki-drift]"),
            "warning should carry [wiki-drift] marker: {}",
            w
        );
        assert!(
            w.contains("may be stale"),
            "warning should mention stale: {}",
            w
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn post_write_wiki_sync_creates_wiki_when_missing() {
        let _guard = CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let file = tmp.path().join("orphan.rs");
        fs::write(&file, "// no wiki here").unwrap();
        let _warning = post_write_wiki_sync(&file, "// no wiki here").await;

        std::env::set_current_dir(&orig).unwrap();
        // Wiki::open creates the layout on first use; the entity page
        // should now exist.
        assert!(
            tmp.path().join(".dm/wiki/entities/orphan_rs.md").is_file(),
            "helper should have created wiki and ingested the file"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn post_write_wiki_sync_wiki_internal_skipped() {
        let _guard = CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki_file = tmp.path().join(".dm/wiki/index.md");
        fs::create_dir_all(wiki_file.parent().unwrap()).unwrap();
        fs::write(&wiki_file, "# Index\n").unwrap();

        let warning = post_write_wiki_sync(&wiki_file, "# Index\n").await;

        std::env::set_current_dir(&orig).unwrap();
        assert!(
            warning.is_none(),
            "wiki-internal file should be silently skipped"
        );
    }

    // C30 cascade canary — pins that holding `CWD_LOCK` across a multi-second
    // operation does NOT cascade into unrelated cwd-readers. Cycle-9
    // characterized the cascade: a lock-holding test that awaits a slow op
    // (e.g., 3s Ollama probe) overlaps non-CWD_LOCK cwd-readers in
    // `tools::wiki_lookup`, `tools::wiki_search`, and `tools::tests::post_write_wiki_sync_*`,
    // making them read the test's tempdir as cwd and fail. Cycle-10 added
    // `CWD_LOCK` acquisition to those readers. This test holds the lock and
    // its tempdir-cwd for 1.5s under tokio's parallel scheduler — if the
    // C30 race surface is genuinely closed, all parallel cwd-readers either
    // queue on the lock or run cleanly. If 23+ failures cascade, the audit
    // is incomplete and a future cycle must extend the CWD_LOCK additions.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn cwd_lock_long_holder_does_not_cascade_into_unprotected_readers() {
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = TempDir::new().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        // Park here long enough that any parallel-scheduled
        // non-CWD_LOCK cwd-reader would see the wrong cwd. Mirrors the
        // 3-second Ollama probe that exposed the original cascade — 1.5s
        // is shorter to keep the suite fast but still wide enough to
        // catch overlap with the typical tools test runtime.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

        std::env::set_current_dir(&orig).unwrap();
        // No assertion needed — the test's purpose is to expose
        // cascading failures in *other* tests via the cwd-poisoning
        // race. If any sibling cascades, the suite reports them; if
        // not, the C30 surface is genuinely closed.
    }
}
