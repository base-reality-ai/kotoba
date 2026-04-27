//! Staged file-change primitives used before edits are written to disk.
//!
//! Builds unified diffs from original/proposed content, applies all or selected
//! pending changes, and reverts only when the current file still matches the
//! proposed content so external edits are not overwritten.

use similar::TextDiff;
use std::path::PathBuf;

/// One staged file modification — computed but not yet written to disk.
#[derive(Debug, Clone)]
pub struct PendingChange {
    pub path: PathBuf,
    pub original: String,
    pub proposed: String,
    /// Unified diff text (pre-rendered).
    pub diff: String,
}

impl PendingChange {
    /// Returns `(added_lines, removed_lines)` by scanning the diff.
    pub fn lines_changed(&self) -> (usize, usize) {
        let added = self
            .diff
            .lines()
            .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
            .count();
        let removed = self
            .diff
            .lines()
            .filter(|l| l.starts_with('-') && !l.starts_with("---"))
            .count();
        (added, removed)
    }
}

/// Build a `PendingChange` from original and proposed file content.
pub fn make_change(path: PathBuf, original: &str, proposed: &str) -> PendingChange {
    let diff = TextDiff::from_lines(original, proposed);
    let diff_text = diff
        .unified_diff()
        .header(
            &format!("a/{}", path.display()),
            &format!("b/{}", path.display()),
        )
        .to_string();
    PendingChange {
        path,
        original: original.to_string(),
        proposed: proposed.to_string(),
        diff: diff_text,
    }
}

/// Write all pending changes to disk.  Returns the number of files written.
pub async fn apply_all(changes: &[PendingChange]) -> (usize, Vec<String>) {
    let mut written = 0;
    let mut errors = Vec::new();
    for change in changes {
        if let Some(parent) = change.path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.ok();
            }
        }
        match tokio::fs::write(&change.path, &change.proposed).await {
            Ok(()) => written += 1,
            Err(e) => errors.push(format!("{}: {}", change.path.display(), e)),
        }
    }
    (written, errors)
}

/// Write only selected pending changes to disk.
/// `decisions[i]` maps to `changes[i]`: `Some(true)` or `None` → apply, `Some(false)` → skip.
/// If `decisions` is shorter than `changes`, extra changes are applied (undecided = accept).
pub async fn apply_selected(
    changes: &[PendingChange],
    decisions: &[Option<bool>],
) -> (usize, usize, Vec<String>) {
    let mut written = 0;
    let mut skipped = 0;
    let mut errors = Vec::new();
    for (i, change) in changes.iter().enumerate() {
        let decision = decisions.get(i).copied().flatten();
        if decision == Some(false) {
            skipped += 1;
            continue;
        }
        if let Some(parent) = change.path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.ok();
            }
        }
        match tokio::fs::write(&change.path, &change.proposed).await {
            Ok(()) => written += 1,
            Err(e) => errors.push(format!("{}: {}", change.path.display(), e)),
        }
    }
    (written, skipped, errors)
}

/// Revert applied changes by writing back original content.
/// Only reverts files whose current content matches `proposed` — files modified
/// since apply are skipped to avoid overwriting external changes.
pub async fn revert_all(changes: &[PendingChange]) -> (usize, usize, Vec<String>) {
    let mut reverted = 0;
    let mut skipped = 0;
    let mut errors = Vec::new();
    for change in changes {
        let current = match tokio::fs::read_to_string(&change.path).await {
            Ok(c) => c,
            Err(e) => {
                errors.push(format!("{}: {}", change.path.display(), e));
                continue;
            }
        };
        if current != change.proposed {
            skipped += 1;
            errors.push(format!(
                "{}: file modified since apply, skipping revert",
                change.path.display()
            ));
            continue;
        }
        match tokio::fs::write(&change.path, &change.original).await {
            Ok(()) => reverted += 1,
            Err(e) => errors.push(format!("{}: {}", change.path.display(), e)),
        }
    }
    (reverted, skipped, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn changeset_add_generates_diff() {
        let change = make_change(
            PathBuf::from("src/lib.rs"),
            "fn old() {}\n",
            "fn new() {}\n",
        );
        assert!(
            change.diff.contains("-fn old()"),
            "diff should show removed line"
        );
        assert!(
            change.diff.contains("+fn new()"),
            "diff should show added line"
        );
    }

    #[test]
    fn lines_changed_counts_correctly() {
        let change = make_change(
            PathBuf::from("a.rs"),
            "line1\nline2\nline3\n",
            "line1\nLINE2\nLINE3\nline4\n",
        );
        let (added, removed) = change.lines_changed();
        // -line2, -line3 removed; +LINE2, +LINE3, +line4 added
        assert_eq!(removed, 2, "two lines removed");
        assert_eq!(added, 3, "three lines added");
    }

    #[tokio::test]
    async fn apply_all_writes_files() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("out.txt");
        let change = make_change(path.clone(), "", "hello\n");
        let (written, errors) = apply_all(&[change]).await;
        assert_eq!(written, 1);
        assert!(errors.is_empty());
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello\n");
    }

    #[tokio::test]
    async fn apply_all_reports_error_on_bad_path() {
        // Path with non-existent root that can't be created
        let change = PendingChange {
            path: PathBuf::from("/proc/dm_test_nonexistent/file.txt"),
            original: String::new(),
            proposed: "x".to_string(),
            diff: String::new(),
        };
        let (written, errors) = apply_all(&[change]).await;
        assert_eq!(written, 0);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn make_change_identical_content_has_no_diff_lines() {
        let change = make_change(PathBuf::from("same.rs"), "fn foo() {}\n", "fn foo() {}\n");
        let (added, removed) = change.lines_changed();
        assert_eq!(added, 0, "no additions for identical content");
        assert_eq!(removed, 0, "no removals for identical content");
    }

    #[test]
    fn lines_changed_all_removed_when_proposed_empty() {
        let change = make_change(PathBuf::from("del.rs"), "line1\nline2\n", "");
        let (added, removed) = change.lines_changed();
        assert_eq!(removed, 2, "all original lines should be removed");
        assert_eq!(added, 0, "no lines added");
    }

    #[tokio::test]
    async fn apply_all_empty_slice_is_noop() {
        let (written, errors) = apply_all(&[]).await;
        assert_eq!(written, 0);
        assert!(errors.is_empty());
    }

    #[tokio::test]
    async fn apply_selected_skips_rejected() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("a.txt");
        let p2 = tmp.path().join("b.txt");
        let p3 = tmp.path().join("c.txt");
        let c1 = make_change(p1.clone(), "", "aaa\n");
        let c2 = make_change(p2.clone(), "", "bbb\n");
        let c3 = make_change(p3.clone(), "", "ccc\n");
        let decisions = vec![Some(true), Some(false), None];
        let (written, skipped, errors) = apply_selected(&[c1, c2, c3], &decisions).await;
        assert_eq!(written, 2);
        assert_eq!(skipped, 1);
        assert!(errors.is_empty());
        assert!(p1.exists(), "accepted file should be written");
        assert!(!p2.exists(), "rejected file should NOT be written");
        assert!(p3.exists(), "undecided file should be written");
    }

    #[tokio::test]
    async fn apply_selected_all_rejected() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("a.txt");
        let p2 = tmp.path().join("b.txt");
        let c1 = make_change(p1.clone(), "", "aaa\n");
        let c2 = make_change(p2.clone(), "", "bbb\n");
        let decisions = vec![Some(false), Some(false)];
        let (written, skipped, errors) = apply_selected(&[c1, c2], &decisions).await;
        assert_eq!(written, 0);
        assert_eq!(skipped, 2);
        assert!(errors.is_empty());
        assert!(!p1.exists());
        assert!(!p2.exists());
    }

    #[tokio::test]
    async fn apply_selected_all_none_applies_all() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("a.txt");
        let p2 = tmp.path().join("b.txt");
        let c1 = make_change(p1.clone(), "", "aaa\n");
        let c2 = make_change(p2.clone(), "", "bbb\n");
        let decisions = vec![None, None];
        let (written, skipped, errors) = apply_selected(&[c1, c2], &decisions).await;
        assert_eq!(written, 2);
        assert_eq!(skipped, 0);
        assert!(errors.is_empty());
        assert!(p1.exists());
        assert!(p2.exists());
    }

    #[tokio::test]
    async fn apply_selected_mismatched_lengths() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("a.txt");
        let p2 = tmp.path().join("b.txt");
        let p3 = tmp.path().join("c.txt");
        let c1 = make_change(p1.clone(), "", "aaa\n");
        let c2 = make_change(p2.clone(), "", "bbb\n");
        let c3 = make_change(p3.clone(), "", "ccc\n");
        // decisions shorter than changes — extras treated as undecided (applied)
        let decisions = vec![Some(false)];
        let (written, skipped, errors) = apply_selected(&[c1, c2, c3], &decisions).await;
        assert_eq!(
            written, 2,
            "extra changes beyond decisions should be applied"
        );
        assert_eq!(skipped, 1);
        assert!(errors.is_empty());
        assert!(!p1.exists(), "first file rejected");
        assert!(p2.exists(), "second file applied (no decision)");
        assert!(p3.exists(), "third file applied (no decision)");
    }

    #[tokio::test]
    async fn revert_all_restores_originals() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("revert.txt");
        let change = make_change(path.clone(), "original content\n", "new content\n");
        // Apply first
        let (written, _) = apply_all(std::slice::from_ref(&change)).await;
        assert_eq!(written, 1);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content\n");
        // Revert
        let (reverted, skipped, errors) = revert_all(&[change]).await;
        assert_eq!(reverted, 1);
        assert_eq!(skipped, 0);
        assert!(errors.is_empty());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "original content\n"
        );
    }

    #[tokio::test]
    async fn revert_all_skips_modified_file() {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("modified.txt");
        let change = make_change(path.clone(), "original\n", "proposed\n");
        let (written, _) = apply_all(std::slice::from_ref(&change)).await;
        assert_eq!(written, 1);
        // Simulate external modification
        std::fs::write(&path, "user edited this\n").unwrap();
        let (reverted, skipped, errors) = revert_all(&[change]).await;
        assert_eq!(reverted, 0);
        assert_eq!(skipped, 1);
        assert!(!errors.is_empty());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "user edited this\n"
        );
    }

    /// Pin the per-iteration aggregation contract on `revert_all` when one
    /// change reverts cleanly and another is content-mismatched in the same
    /// call. Catches a future refactor that:
    ///   - returns early on the first skip (would lose the second change's revert)
    ///   - silently overwrites the modified file (would break the no-silent-overwrite contract)
    ///   - drops the canonical "modified since apply" phrase from the error
    #[tokio::test]
    async fn revert_all_mixed_success_and_skip_aggregates_correctly() {
        let tmp = tempdir().unwrap();
        let clean_path = tmp.path().join("clean.txt");
        let modified_path = tmp.path().join("modified.txt");
        let clean_change = make_change(clean_path.clone(), "orig-clean\n", "prop-clean\n");
        let modified_change = make_change(modified_path.clone(), "orig-mod\n", "prop-mod\n");

        let (written, _) = apply_all(&[clean_change.clone(), modified_change.clone()]).await;
        assert_eq!(written, 2);

        // External tampering on one of the two files.
        std::fs::write(&modified_path, "user touched this\n").unwrap();

        let (reverted, skipped, errors) = revert_all(&[clean_change, modified_change]).await;

        // Aggregation: clean revert + skip ≠ early-exit.
        assert_eq!(reverted, 1, "clean change must still revert");
        assert_eq!(skipped, 1, "modified change must skip");
        assert_eq!(
            errors.len(),
            1,
            "exactly one error (the skip-message) — clean side adds nothing"
        );

        // Canonical skip-message form pinned.
        assert!(
            errors[0].contains("modified since apply"),
            "missing canonical skip phrase: {}",
            errors[0]
        );
        assert!(
            errors[0].contains("modified.txt"),
            "error must name the offending path: {}",
            errors[0]
        );

        // On-disk: the user's tampered content is preserved verbatim.
        assert_eq!(
            std::fs::read_to_string(&modified_path).unwrap(),
            "user touched this\n",
            "modified file must NOT be silently overwritten by revert"
        );
        // On-disk: the clean file IS reverted to its original.
        assert_eq!(
            std::fs::read_to_string(&clean_path).unwrap(),
            "orig-clean\n",
            "clean change must restore its original content"
        );
    }

    #[tokio::test]
    async fn revert_all_reports_error_for_deleted_file() {
        // If the user deletes the post-apply file before reverting, the
        // read_to_string Err branch should push an error and continue —
        // never silently treat a missing file as a successful revert and
        // never recreate the original behind the user's back.
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("deleted.txt");
        let change = make_change(path.clone(), "original\n", "proposed\n");
        let (written, _) = apply_all(std::slice::from_ref(&change)).await;
        assert_eq!(written, 1);
        // Externally remove the file between apply and revert.
        std::fs::remove_file(&path).unwrap();
        let (reverted, skipped, errors) = revert_all(&[change]).await;
        assert_eq!(reverted, 0, "deleted file must not count as reverted");
        assert_eq!(
            skipped, 0,
            "read-error path is errors-only, not skipped (skipped is for content-mismatch)"
        );
        assert_eq!(errors.len(), 1, "exactly one error for the deleted file");
        assert!(
            errors[0].contains("deleted.txt"),
            "error must name the path: {}",
            errors[0]
        );
        assert!(
            !path.exists(),
            "revert must not silently recreate the deleted file"
        );
    }
}
