//! Cross-cutting wiki-consultation telemetry. Counters live as process-
//! global atomics (one set per `dm` invocation) so the TUI slash dispatcher,
//! the web `log_wiki_telemetry` path, and the agent's tool registry can all
//! cooperate without threading a shared `Arc` through every layer.
//! Mirrors the existing `crate::warnings` global pattern.
//!
//! ## Test isolation
//!
//! Any test that mutates the globals — directly via `record_wiki_*`, or
//! transitively via `ToolRegistry::call("wiki_*", …)` (which mirrors into
//! the globals at `registry.rs::call`'s epilogue) — MUST hold
//! `TELEMETRY_LOCK`. The simplest way is `let _g = telemetry_test_guard();`
//! at the top of the test. Forgetting this creates parallel-execution
//! flakes whose blame travels — the failing assertion is in a sibling
//! test that *was* holding the lock (C7→C8 and C11→C16 P0 regressions
//! in this run).

use std::sync::atomic::{AtomicUsize, Ordering};

static WIKI_TOOL_CALLS: AtomicUsize = AtomicUsize::new(0);
static WIKI_DRIFT_WARNINGS: AtomicUsize = AtomicUsize::new(0);

/// Increment the global `wiki_tool_calls` counter. **Test isolation:**
/// any test that exercises this path must hold `TELEMETRY_LOCK` (use
/// `telemetry_test_guard()`).
pub fn record_wiki_tool_call() {
    WIKI_TOOL_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Increment the global `wiki_drift_warnings` counter. **Test isolation:**
/// see `record_wiki_tool_call` — same `TELEMETRY_LOCK` requirement.
pub fn record_wiki_drift_warning() {
    WIKI_DRIFT_WARNINGS.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot the counters as `(tool_calls, drift_warnings)`. The two reads
/// are independent — a record between them just reflects the most recent
/// observable state, which is fine for a status panel.
pub fn snapshot() -> (usize, usize) {
    (
        WIKI_TOOL_CALLS.load(Ordering::Relaxed),
        WIKI_DRIFT_WARNINGS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
pub fn reset() {
    WIKI_TOOL_CALLS.store(0, Ordering::Relaxed);
    WIKI_DRIFT_WARNINGS.store(0, Ordering::Relaxed);
}

/// Test-isolation lock — held for the duration of any test that mutates
/// the globals. Same pattern as `crate::warnings::WARNINGS_LOCK`.
#[cfg(test)]
pub(crate) static TELEMETRY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Test-only RAII guard combining `TELEMETRY_LOCK.lock()` + `reset()`.
///
/// Use this in any test that calls `record_wiki_*` directly OR
/// dispatches a `wiki_*` tool through `ToolRegistry::call(...)` OR
/// asserts on `snapshot()`. Holding `TELEMETRY_LOCK` is the only thing
/// preventing parallel telemetry-touching tests from clobbering each
/// other's counter values (see C16 P0 regression for what happens when
/// this is forgotten).
///
/// The returned guard MUST be name-bound for the test's lifetime
/// (`let _g = telemetry_test_guard();` — `_` would drop immediately).
/// Recovers from poisoned locks (other tests panicking under the lock
/// won't cascade).
#[cfg(test)]
pub(crate) fn telemetry_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = TELEMETRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    guard
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_then_record_increments() {
        let _g = TELEMETRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        record_wiki_tool_call();
        record_wiki_tool_call();
        record_wiki_tool_call();
        let (calls, drift) = snapshot();
        assert_eq!(calls, 3);
        assert_eq!(drift, 0);
    }

    #[test]
    fn record_drift_independent_of_tool_call() {
        let _g = TELEMETRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        record_wiki_drift_warning();
        record_wiki_drift_warning();
        let (calls, drift) = snapshot();
        assert_eq!(calls, 0);
        assert_eq!(drift, 2);
    }

    #[test]
    fn snapshot_returns_both_counts() {
        let _g = TELEMETRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        record_wiki_tool_call();
        record_wiki_drift_warning();
        record_wiki_drift_warning();
        record_wiki_tool_call();
        record_wiki_tool_call();
        let (calls, drift) = snapshot();
        assert_eq!(calls, 3);
        assert_eq!(drift, 2);
    }

    #[test]
    fn telemetry_test_guard_resets_and_holds_lock() {
        // Establish a known dirty state for the pre-populate phase.
        // Other parallel tests may leave the globals non-zero; reset
        // first to make this test's pre-state deterministic.
        let _g0 = TELEMETRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        record_wiki_tool_call();
        record_wiki_drift_warning();
        assert_eq!(snapshot(), (1, 1));
        drop(_g0);

        // Acquire via the helper — must reset to (0, 0) AND keep the
        // lock held for the rest of the test scope.
        let _guard = telemetry_test_guard();
        assert_eq!(
            snapshot(),
            (0, 0),
            "helper must reset both counters on acquisition"
        );

        // While holding the guard, mutations land on the zeroed state.
        record_wiki_tool_call();
        assert_eq!(snapshot(), (1, 0));
        // _guard drops at scope end — lock released.
    }
}
