//! Next-step hints for filesystem tool errors.
//!
//! When `tokio::fs::*` returns an `io::Error`, the underlying OS
//! message ("Permission denied", "No such file or directory") tells
//! the user what went wrong but not what to do about it. The
//! [`hint_for`] helper inspects [`ErrorKind`] and returns a one-line
//! hint to append to the tool's user-facing error — closing the
//! "error messages include next steps" gap for the `file_read` /
//! `file_edit` / `file_write` surfaces.

use std::io::ErrorKind;

/// Filesystem operation category. Drives the chmod-flag selection
/// (`+r` for reads, `+w` for writes/edits) in `PermissionDenied` hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsOp {
    Read,
    Write,
    Edit,
}

impl FsOp {
    fn chmod_flag(self) -> &'static str {
        match self {
            FsOp::Read => "+r",
            FsOp::Write | FsOp::Edit => "+w",
        }
    }
}

/// Return a one-line next-step hint for an `io::Error`, suitable for
/// appending to a user-facing `"Cannot {op} file 'X': {err}"` message.
/// Returns `String::new()` when the error kind has no generic actionable
/// guidance — callers append verbatim either way.
///
/// Every non-empty hint starts with `"\n    "` so it renders indented
/// under the main error line in terminal output.
pub fn hint_for(err: &std::io::Error, path: &str, op: FsOp) -> String {
    match err.kind() {
        ErrorKind::PermissionDenied => format!(
            "\n    Try: chmod {} '{}' or check file ownership. \
             For writes, the parent directory must also be writable.",
            op.chmod_flag(),
            path,
        ),
        ErrorKind::NotFound if matches!(op, FsOp::Read | FsOp::Edit) => format!(
            "\n    Check: path '{}' exists and is spelled correctly. \
             Use an absolute path or one relative to the project root.",
            path,
        ),
        ErrorKind::IsADirectory => format!(
            "\n    Check: '{}' is a directory, not a file. \
             Use `glob` or `ls` to list its contents instead.",
            path,
        ),
        ErrorKind::NotADirectory => format!(
            "\n    Check: a parent component of '{}' is a file, not a \
             directory. Verify the path structure.",
            path,
        ),
        ErrorKind::ReadOnlyFilesystem => format!(
            "\n    Check: the filesystem containing '{}' is mounted \
             read-only. Remount rw or pick a different path.",
            path,
        ),
        ErrorKind::StorageFull => {
            String::from("\n    Check: disk is full (`df -h`). Free space and retry.")
        }
        _ => String::new(),
    }
}

/// Format a user-facing write-error message: `"{prefix}: {err}{hint}"`.
///
/// The hint comes from [`hint_for`] with [`FsOp::Write`]; when no hint maps
/// to the error kind the message ends at the raw error text. Callers get a
/// single-step way to produce a consistent, actionable error string — see
/// `/export` dispatcher for the original use site.
pub fn format_write_error(err: &std::io::Error, path: &str, prefix: &str) -> String {
    format!("{}: {}{}", prefix, err, hint_for(err, path, FsOp::Write))
}

/// Format a user-facing read-error message: `"{prefix}: {err}{hint}"`.
///
/// Parallel to [`format_write_error`] but uses [`FsOp::Read`] so the
/// chmod-flag hint is `+r` and `NotFound` routes to the "Check: path
/// exists" guidance (which is silent for Write). Callers at
/// `file_read.rs` and `apply_diff.rs` use this to keep the
/// `"Cannot read file 'X': {err}"` dialect with a trailing hint.
pub fn format_read_error(err: &std::io::Error, path: &str, prefix: &str) -> String {
    format!("{}: {}{}", prefix, err, hint_for(err, path, FsOp::Read))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};

    #[test]
    fn hint_for_permission_denied_read_uses_plus_r() {
        let err = Error::from(ErrorKind::PermissionDenied);
        let hint = hint_for(&err, "src/main.rs", FsOp::Read);
        assert!(hint.contains("chmod +r 'src/main.rs'"), "got: {}", hint);
        assert!(hint.contains("ownership"), "got: {}", hint);
    }

    #[test]
    fn hint_for_permission_denied_write_uses_plus_w() {
        let err = Error::from(ErrorKind::PermissionDenied);
        let hint = hint_for(&err, "out.txt", FsOp::Write);
        assert!(hint.contains("chmod +w 'out.txt'"), "got: {}", hint);
        assert!(
            hint.contains("parent directory must also be writable"),
            "got: {}",
            hint
        );
    }

    #[test]
    fn hint_for_permission_denied_edit_uses_plus_w() {
        // Edit is a write op from the user's point of view — the chmod
        // flag must match Write, not Read.
        let err = Error::from(ErrorKind::PermissionDenied);
        let hint = hint_for(&err, "src/lib.rs", FsOp::Edit);
        assert!(hint.contains("chmod +w 'src/lib.rs'"), "got: {}", hint);
    }

    #[test]
    fn hint_for_not_found_read_includes_check_hint() {
        let err = Error::from(ErrorKind::NotFound);
        let hint = hint_for(&err, "missing.rs", FsOp::Read);
        assert!(hint.contains("Check:"), "got: {}", hint);
        assert!(hint.contains("'missing.rs'"), "got: {}", hint);
        assert!(hint.contains("absolute"), "got: {}", hint);
    }

    #[test]
    fn hint_for_not_found_edit_includes_check_hint() {
        // Edits on a non-existent path are user errors (the file must
        // exist to be edited). Hint must fire for Edit, unlike Write
        // where "not found" is the normal case (we create the file).
        let err = Error::from(ErrorKind::NotFound);
        let hint = hint_for(&err, "missing.rs", FsOp::Edit);
        assert!(hint.contains("Check:"), "got: {}", hint);
    }

    #[test]
    fn hint_for_not_found_write_returns_empty() {
        // Writes on a non-existent path are the NORMAL case (create
        // the file). Appending a "path doesn't exist" hint would be
        // misleading — suppress it for Write.
        let err = Error::from(ErrorKind::NotFound);
        let hint = hint_for(&err, "new.rs", FsOp::Write);
        assert!(
            hint.is_empty(),
            "NotFound on Write must be silent; got: {}",
            hint
        );
    }

    #[test]
    fn hint_for_generic_kind_returns_empty_string() {
        // An unhandled ErrorKind must produce an empty suffix — callers
        // append verbatim, so a non-empty hint on an unknown kind would
        // confuse the user. Pin this as a contract.
        let err = Error::from(ErrorKind::TimedOut);
        let hint = hint_for(&err, "x", FsOp::Read);
        assert!(
            hint.is_empty(),
            "unhandled kind must return empty: {:?}",
            hint
        );
    }

    #[test]
    fn hint_for_always_starts_with_indent_prefix_when_nonempty() {
        // Every non-empty hint must begin with "\n    " so it renders
        // indented under the main error line in terminal output.
        let cases: Vec<(ErrorKind, FsOp)> = vec![
            (ErrorKind::PermissionDenied, FsOp::Read),
            (ErrorKind::PermissionDenied, FsOp::Write),
            (ErrorKind::PermissionDenied, FsOp::Edit),
            (ErrorKind::NotFound, FsOp::Read),
            (ErrorKind::NotFound, FsOp::Edit),
        ];
        for (kind, op) in cases {
            let err = Error::from(kind);
            let hint = hint_for(&err, "p", op);
            assert!(
                hint.starts_with("\n    "),
                "hint for {:?}/{:?} must start with \\n+4sp, got: {:?}",
                kind,
                op,
                hint
            );
        }
    }

    #[test]
    fn fsop_chmod_flag_distinct_per_category() {
        // Guardrail: Read and Write/Edit must map to different flags,
        // so PermissionDenied hints don't converge on a single useless
        // blanket suggestion.
        assert_eq!(FsOp::Read.chmod_flag(), "+r");
        assert_eq!(FsOp::Write.chmod_flag(), "+w");
        assert_eq!(FsOp::Edit.chmod_flag(), "+w");
    }

    #[test]
    fn hint_for_covers_all_error_kinds() {
        // Sweep: every non-empty arm must fire on its (kind, op) pair
        // and embed its signature phrase. Guards against silent-drop
        // regressions when new arms are added or strings edited.
        let cases = [
            (ErrorKind::PermissionDenied, FsOp::Read, "chmod"),
            (ErrorKind::PermissionDenied, FsOp::Write, "chmod"),
            (ErrorKind::PermissionDenied, FsOp::Edit, "chmod"),
            (ErrorKind::NotFound, FsOp::Read, "Check: path"),
            (ErrorKind::NotFound, FsOp::Edit, "Check: path"),
            (ErrorKind::IsADirectory, FsOp::Read, "directory"),
            (ErrorKind::NotADirectory, FsOp::Write, "not a directory"),
            (ErrorKind::ReadOnlyFilesystem, FsOp::Write, "read-only"),
            (ErrorKind::StorageFull, FsOp::Write, "disk is full"),
        ];
        for (kind, op, needle) in cases {
            let err = Error::from(kind);
            let hint = hint_for(&err, "/tmp/x", op);
            assert!(!hint.is_empty(), "{:?}+{:?} must emit hint", kind, op);
            assert!(
                hint.contains(needle),
                "{:?}+{:?}: missing {:?} in {:?}",
                kind,
                op,
                needle,
                hint
            );
        }
    }

    #[test]
    fn hint_for_notfound_on_write_is_empty() {
        // NotFound during Write means the parent dir is missing — the
        // message is handled upstream (create_dir_all path), so
        // hint_for stays silent to avoid duplicate guidance.
        let err = Error::from(ErrorKind::NotFound);
        assert!(hint_for(&err, "/x", FsOp::Write).is_empty());
    }

    #[test]
    fn hint_for_unknown_kind_is_empty() {
        // Second pin on the "unhandled kind → empty" contract, using
        // a distinct ErrorKind (Interrupted) from the existing
        // TimedOut case to guard against arm-specific drift.
        let err = Error::from(ErrorKind::Interrupted);
        assert!(hint_for(&err, "/x", FsOp::Read).is_empty());
    }

    #[test]
    fn format_write_error_appends_hint_for_permission_denied() {
        let err = Error::from(ErrorKind::PermissionDenied);
        let s = format_write_error(&err, "/root/x.md", "Export failed");
        assert!(s.starts_with("Export failed: "), "raw error preserved");
        assert!(s.contains("chmod"), "hint appended: {:?}", s);
        assert!(s.contains("/root/x.md"), "path echoed in hint: {:?}", s);
    }

    #[test]
    fn format_write_error_empty_hint_for_unmapped_kind() {
        let err = Error::from(ErrorKind::Interrupted);
        let s = format_write_error(&err, "/x", "Export failed");
        assert!(s.starts_with("Export failed: "));
        // Unmapped kind → no trailing hint, so message ends with the raw error text.
        assert!(!s.contains("Try:"), "no Try: for unmapped kind: {:?}", s);
        assert!(
            !s.contains("Check:"),
            "no Check: for unmapped kind: {:?}",
            s
        );
    }

    #[test]
    fn format_write_error_full_disk_hint() {
        let err = Error::from(ErrorKind::StorageFull);
        let s = format_write_error(&err, "/tmp/x.md", "Export failed");
        assert!(s.contains("disk is full"), "full-disk hint needle: {:?}", s);
    }

    #[test]
    fn format_read_error_appends_hint_for_permission_denied() {
        // Read path uses +r (not +w) — pin the Read/Write split at the
        // format helper boundary.
        let err = Error::from(ErrorKind::PermissionDenied);
        let s = format_read_error(&err, "src/secret.rs", "Cannot read file 'src/secret.rs'");
        assert!(s.starts_with("Cannot read file 'src/secret.rs': "));
        assert!(s.contains("chmod +r"), "read hint uses +r: {:?}", s);
        assert!(
            !s.contains("chmod +w"),
            "read hint must not suggest +w: {:?}",
            s
        );
    }

    #[test]
    fn format_read_error_not_found_includes_check_hint() {
        // Pin the NotFound-on-Read contract: a missing-path message must
        // append the "Check: path exists" hint (Write suppresses it).
        let err = Error::from(ErrorKind::NotFound);
        let s = format_read_error(&err, "missing.rs", "Cannot read file 'missing.rs'");
        assert!(s.contains("Check:"), "NotFound-on-Read hints: {:?}", s);
    }

    #[test]
    fn format_read_error_is_a_directory_hint_includes_directory_keyword() {
        // When a path points at a directory, the "is a directory" hint
        // should ride along — distinct from NotFound / PermissionDenied
        // — so the caller knows to use `glob` or `ls` instead. Pin
        // this branch against silent regression.
        let err = Error::from(ErrorKind::IsADirectory);
        let s = format_read_error(&err, "src/tools", "Cannot read 'src/tools'");
        assert!(
            s.contains("is a directory"),
            "IsADirectory needle preserved: {:?}",
            s
        );
    }

    #[test]
    fn format_write_error_not_a_directory_hint_mentions_parent_component() {
        // NotADirectory fires when a path component mid-way is a file
        // (e.g. writing to `existing_file/inside/new.txt`). The hint
        // must name the "parent component" so the user can inspect the
        // path shape — different guidance from IsADirectory.
        let err = Error::from(ErrorKind::NotADirectory);
        let s = format_write_error(&err, "foo/bar.txt", "Error writing file 'foo/bar.txt'");
        assert!(
            s.contains("parent component") || s.contains("not a\n    directory"),
            "NotADirectory needle preserved: {:?}",
            s
        );
    }
}
