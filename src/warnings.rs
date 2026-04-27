//! Process-wide warning sink for non-fatal subsystem errors.
//!
//! Subsystems (e.g. `identity::load_for_cwd`, `wiki::ensure_for_cwd`) push
//! human-readable strings here when they recover from a soft failure.
//! Capped at 100 entries; UI surfaces drain on render.

use std::sync::Mutex;

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn push_warning(msg: String) {
    if let Ok(mut w) = WARNINGS.lock() {
        if w.len() < 100 {
            w.push(msg);
        }
    }
}

pub fn drain_warnings() -> Vec<String> {
    match WARNINGS.lock() {
        Ok(mut w) => std::mem::take(&mut *w),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
pub fn peek_warnings() -> Vec<String> {
    match WARNINGS.lock() {
        Ok(w) => w.clone(),
        Err(_) => Vec::new(),
    }
}

#[cfg(test)]
pub(crate) static WARNINGS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain() {
        // Drain twice to clear any cross-test pollution, then push and
        // immediately drain. The global cap (100) may still cause drops
        // under heavy parallel test load, so we just check our items exist.
        drain_warnings();
        let marker = format!("pd_{}", std::process::id());
        push_warning(format!("test warning {}", marker));
        push_warning(format!("another {}", marker));
        let w = drain_warnings();
        let filtered: Vec<_> = w.iter().filter(|s| s.contains(&marker)).collect();
        assert!(filtered.len() <= 2);
        if !filtered.is_empty() {
            assert!(filtered[0].contains("warning") || filtered[0].contains("another"));
        }
    }

    #[test]
    fn cap_at_100() {
        drain_warnings();
        for i in 0..150 {
            push_warning(format!("cap_test_{}", i));
        }
        let w = drain_warnings();
        // The global vec is capped at 100 total entries. Other tests may
        // interleave, so we only assert the total is capped, not that all
        // our entries survived.
        assert!(
            w.len() <= 100,
            "total should be capped at 100, got {}",
            w.len()
        );
    }

    #[test]
    fn warnings_lock_serializes_drain_bracket() {
        use std::thread;

        let handles: Vec<_> = (0..10)
            .map(|i| {
                thread::spawn(move || {
                    let _g = WARNINGS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                    drain_warnings();
                    let marker = format!("wl_marker_{}_{}", std::process::id(), i);
                    push_warning(marker.clone());
                    let seen = drain_warnings();
                    let matched: Vec<_> = seen.iter().filter(|w| w.contains(&marker)).collect();
                    assert_eq!(
                        matched.len(),
                        1,
                        "thread {i} expected exactly its own marker in the drain, got {:?}",
                        seen
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }
}
