use std::path::{Path, PathBuf};

use anyhow::Result;

/// Resolved path info returned by `validate_path`.
pub struct ResolvedPath {
    /// The canonical absolute path (symlinks resolved).
    pub canonical: PathBuf,
    /// Whether the original path contained a symlink.
    pub has_symlink: bool,
    /// Whether the resolved path is inside the project root.
    pub in_project: bool,
}

/// Validate and resolve a file path for tool operations.
///
/// - Expands `~` to home directory
/// - Resolves `.` and `..` components
/// - Detects symlinks in the path chain
/// - Checks if the resolved path is within the project root
///
/// Returns the resolved path info. Does NOT block operations — the caller
/// decides policy (warn on symlink, deny outside project, etc).
pub fn validate_path(raw: &str, project_root: &Path) -> Result<ResolvedPath> {
    // 1. Expand ~ to home dir
    let expanded = if let Some(rest) = raw.strip_prefix("~/") {
        match std::env::var("HOME") {
            Ok(home) => PathBuf::from(home).join(rest),
            Err(_) => PathBuf::from(raw),
        }
    } else {
        PathBuf::from(raw)
    };

    // 2. Make absolute if relative
    let absolute = if expanded.is_relative() {
        std::env::current_dir()?.join(&expanded)
    } else {
        expanded
    };

    // 3. Check for symlinks by walking the path components
    let has_symlink = path_contains_symlink(&absolute);

    // 4. Canonicalize (resolves symlinks and ..)
    // For new files that don't exist yet, canonicalize the parent
    let canonical = if absolute.exists() {
        absolute.canonicalize()?
    } else if let Some(parent) = absolute.parent() {
        if parent.exists() {
            parent
                .canonicalize()?
                .join(absolute.file_name().unwrap_or_default())
        } else {
            absolute.clone()
        }
    } else {
        absolute.clone()
    };

    // 5. Check if inside project root
    let project_canonical = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    let in_project = canonical.starts_with(&project_canonical);

    Ok(ResolvedPath {
        canonical,
        has_symlink,
        in_project,
    })
}

/// Generate informational warnings for a resolved path.
/// Used by read-only tools (`file_read`, glob, grep, ls, `notebook_read`).
pub fn read_warnings(resolved: &ResolvedPath, raw_path: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    if resolved.has_symlink {
        warnings.push(format!(
            "Note: path contains symlink: {} -> {}",
            raw_path,
            resolved.canonical.display()
        ));
    }
    if !resolved.in_project {
        warnings.push(format!(
            "Note: accessing path outside project directory: {}",
            resolved.canonical.display()
        ));
    }
    warnings
}

/// Check if a write operation should be blocked due to symlinks.
/// Returns `Some(error_message)` if blocked, `None` if safe.
/// Used by write tools (`file_edit`, `file_write`, `multi_edit`, `apply_diff`, `notebook_edit`).
pub fn check_write_blocked(resolved: &ResolvedPath, raw_path: &str, verb: &str) -> Option<String> {
    if resolved.has_symlink {
        let cap = capitalize_first(verb);
        Some(format!(
            "Refusing to {} through symlink: {} -> {}. {} the target file directly at {}",
            verb,
            raw_path,
            resolved.canonical.display(),
            cap,
            resolved.canonical.display()
        ))
    } else {
        None
    }
}

pub const MAX_EDIT_FILE_SIZE: u64 = 10 * 1024 * 1024;

pub async fn check_file_editable(path: &str, verb: &str) -> Option<super::ToolResult> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(m) => m,
        Err(e) => {
            return Some(super::ToolResult {
                content: super::fs_error::format_read_error(
                    &e,
                    path,
                    &format!("Cannot read file '{}'", path),
                ),
                is_error: true,
            });
        }
    };

    if metadata.len() > MAX_EDIT_FILE_SIZE {
        return Some(super::ToolResult {
            content: format!(
                "File is too large to {} ({:.1} MB, max {} MB): {}. Try: use bash with sed/awk for targeted edits on large files.",
                verb,
                metadata.len() as f64 / 1_048_576.0,
                MAX_EDIT_FILE_SIZE / (1024 * 1024),
                path
            ),
            is_error: true,
        });
    }

    if crate::index::chunker::is_binary(std::path::Path::new(path)) {
        return Some(super::ToolResult {
            content: format!(
                "Cannot {} binary file: {}. Try: this tool only handles text files; use bash with hexdump/xxd for binary inspection.",
                verb, path
            ),
            is_error: true,
        });
    }

    None
}

/// Check if a path points to a sensitive file that should not be written to.
/// Returns `Some(reason)` if the path is sensitive, `None` if safe.
pub fn is_sensitive_path(path: &Path) -> Option<&'static str> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let name_lower = name.to_lowercase();

    if name_lower.starts_with(".env") {
        return Some(".env files may contain secrets");
    }

    if path.components().any(|c| c.as_os_str() == ".ssh") {
        return Some("SSH directory contains private keys");
    }

    if path.components().any(|c| c.as_os_str() == ".gnupg") {
        return Some("GPG directory contains private keys");
    }

    for pattern in &["credentials", "secret", "private_key", "service_account"] {
        if name_lower.contains(pattern) {
            return Some("filename suggests credentials");
        }
    }

    let path_str = path.to_string_lossy();
    if path_str.starts_with("/etc/shadow") || path_str.starts_with("/etc/passwd") {
        return Some("system authentication file");
    }

    if name_lower == ".netrc"
        || name_lower == ".npmrc"
        || name_lower == ".pypirc"
        || name_lower == ".docker/config.json"
    {
        return Some("authentication config file");
    }

    if name_lower.ends_with(".pem") || name_lower.ends_with(".key") || name_lower.ends_with(".p12")
    {
        return Some("file extension suggests private key material");
    }

    None
}

/// Check if a write to a sensitive path should be blocked.
/// Returns `Some(error_message)` if blocked, `None` if safe.
pub fn check_sensitive_blocked(path: &Path, raw_path: &str, verb: &str) -> Option<String> {
    is_sensitive_path(path).map(|reason| {
        format!(
            "Blocked: cannot {} '{}' — {}. Use --dangerously-skip-permissions to override.",
            verb, raw_path, reason
        )
    })
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

/// Check whether any component in the path is a symlink.
fn path_contains_symlink(path: &Path) -> bool {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component);
        if current.is_symlink() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use tempfile::TempDir;

    #[test]
    fn absolute_path_in_project() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("test.txt");
        std::fs::write(&file, "hello").unwrap();
        let r = validate_path(file.to_str().unwrap(), tmp.path()).unwrap();
        assert!(r.in_project);
        assert!(!r.has_symlink);
    }

    #[test]
    fn path_outside_project() {
        let tmp = TempDir::new().unwrap();
        let r = validate_path("/etc/hosts", tmp.path()).unwrap();
        assert!(!r.in_project);
    }

    #[test]
    fn symlink_detection() {
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("real.txt");
        std::fs::write(&real, "data").unwrap();
        let link = tmp.path().join("link.txt");
        symlink(&real, &link).unwrap();
        let r = validate_path(link.to_str().unwrap(), tmp.path()).unwrap();
        assert!(r.has_symlink);
        assert_eq!(r.canonical, real.canonicalize().unwrap());
    }

    #[test]
    fn dotdot_traversal() {
        let tmp = TempDir::new().unwrap();
        let sub = tmp.path().join("a/b");
        std::fs::create_dir_all(&sub).unwrap();
        let traversal = sub.join("../../escape.txt");
        // The file doesn't exist but parent resolution should work
        let r = validate_path(traversal.to_str().unwrap(), &sub).unwrap();
        // After canonicalization, ../.. from a/b/ goes back to tmp root
        assert!(!r.in_project); // sub is the "project root", escape.txt is above it
    }

    #[test]
    fn relative_path_resolution() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let r = validate_path("src/main.rs", &std::env::current_dir().unwrap()).unwrap();
        assert!(r.canonical.is_absolute());
    }

    #[test]
    fn new_file_parent_resolution() {
        let tmp = TempDir::new().unwrap();
        let new_file = tmp.path().join("does_not_exist.txt");
        let r = validate_path(new_file.to_str().unwrap(), tmp.path()).unwrap();
        assert!(r.in_project);
        assert!(!r.has_symlink);
        assert!(r.canonical.is_absolute());
    }

    // ── read_warnings / check_write_blocked helpers ─────────────────────────

    #[test]
    fn read_warnings_symlink() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/real/path"),
            has_symlink: true,
            in_project: true,
        };
        let w = read_warnings(&resolved, "/link/path");
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("symlink"));
    }

    #[test]
    fn read_warnings_outside_project() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/etc/passwd"),
            has_symlink: false,
            in_project: false,
        };
        let w = read_warnings(&resolved, "/etc/passwd");
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("outside project"));
    }

    #[test]
    fn read_warnings_clean_path_empty() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/project/src/main.rs"),
            has_symlink: false,
            in_project: true,
        };
        assert!(read_warnings(&resolved, "src/main.rs").is_empty());
    }

    #[test]
    fn read_warnings_both_symlink_and_outside() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/etc/shadow"),
            has_symlink: true,
            in_project: false,
        };
        let w = read_warnings(&resolved, "/link/shadow");
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn check_write_blocked_symlink() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/real/file.txt"),
            has_symlink: true,
            in_project: true,
        };
        let msg = check_write_blocked(&resolved, "/link/file.txt", "edit");
        assert!(msg.is_some());
        let msg = msg.unwrap();
        assert!(msg.contains("Refusing to edit through symlink"));
        assert!(msg.contains("Edit the target file"));
    }

    #[test]
    fn check_write_blocked_no_symlink() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/project/file.txt"),
            has_symlink: false,
            in_project: true,
        };
        assert!(check_write_blocked(&resolved, "file.txt", "edit").is_none());
    }

    #[test]
    fn check_write_blocked_verb_capitalized() {
        let resolved = ResolvedPath {
            canonical: PathBuf::from("/real/f"),
            has_symlink: true,
            in_project: true,
        };
        let msg = check_write_blocked(&resolved, "link", "apply diff to").unwrap();
        assert!(
            msg.contains("Apply diff to the target"),
            "verb should be capitalized: {}",
            msg
        );
    }

    // ── check_file_editable ─────────────────────────────────────────────────

    #[tokio::test]
    async fn check_file_editable_normal_file_ok() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("normal.txt");
        std::fs::write(&file, "hello world").unwrap();
        let result = check_file_editable(file.to_str().unwrap(), "edit").await;
        assert!(result.is_none(), "normal text file should be editable");
    }

    #[tokio::test]
    async fn check_file_editable_binary_blocked() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("binary.bin");
        std::fs::write(&file, b"\x00\x01\x02\x03binary").unwrap();
        let result = check_file_editable(file.to_str().unwrap(), "edit").await;
        assert!(result.is_some());
        let err = result.unwrap();
        assert!(err.is_error);
        assert!(
            err.content.contains("binary"),
            "should mention binary: {}",
            err.content
        );
    }

    #[tokio::test]
    async fn check_file_editable_nonexistent_errors() {
        let result = check_file_editable("/nonexistent/path/file.txt", "edit").await;
        assert!(result.is_some());
        assert!(result.unwrap().is_error);
    }

    #[tokio::test]
    async fn check_file_editable_verb_in_message() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("binary.bin");
        std::fs::write(&file, b"\x00\x01\x02data").unwrap();
        let result = check_file_editable(file.to_str().unwrap(), "apply diff to").await;
        let msg = result.unwrap().content;
        assert!(
            msg.contains("apply diff to"),
            "verb should appear in message: {}",
            msg
        );
    }

    #[tokio::test]
    async fn check_file_editable_binary_error_includes_canonical_try_hint() {
        // Pins the canonical "Try: ..." phrasing on the binary-file branch so a
        // future refactor can't silently strip the actionable next-step hint.
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("binary.bin");
        std::fs::write(&file, b"\x00\x01\x02\x03binary").unwrap();
        let result = check_file_editable(file.to_str().unwrap(), "edit").await;
        let err = result.expect("binary file must produce error");
        assert!(err.is_error);
        assert!(
            err.content.contains("Cannot edit binary file"),
            "missing canonical lead: {}",
            err.content
        );
        assert!(
            err.content
                .contains("Try: this tool only handles text files"),
            "missing canonical Try: hint: {}",
            err.content
        );
        assert!(
            err.content.contains("hexdump/xxd"),
            "missing actionable alternative tool: {}",
            err.content
        );
    }

    #[test]
    fn max_edit_file_size_is_reasonable() {
        const { assert!(MAX_EDIT_FILE_SIZE >= 1_048_576) };
        const { assert!(MAX_EDIT_FILE_SIZE <= 100 * 1024 * 1024) };
    }

    /// Pins the canonical `Try: ...` phrasing on the oversize-file branch.
    /// Uses a sparse file via `set_len` (instant, zero disk blocks) so the
    /// test doesn't actually allocate `MAX_EDIT_FILE_SIZE` bytes on disk.
    /// Size check fires before binary check, so the file's content (or lack
    /// thereof) doesn't matter for this branch.
    #[tokio::test]
    async fn check_file_editable_oversize_error_includes_canonical_try_hint() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("huge.bin");
        let f = std::fs::File::create(&file).unwrap();
        f.set_len(MAX_EDIT_FILE_SIZE + 1_048_576).unwrap();
        drop(f);

        let result = check_file_editable(file.to_str().unwrap(), "edit").await;
        let err = result.expect("oversize file must produce error");
        assert!(err.is_error);
        assert!(
            err.content.contains("File is too large to edit"),
            "missing canonical lead: {}",
            err.content
        );
        assert!(
            err.content.contains("Try: use bash with sed/awk"),
            "missing canonical Try: hint: {}",
            err.content
        );
        assert!(
            err.content.contains("MB"),
            "should report sizes in MB: {}",
            err.content
        );
    }

    // ── is_sensitive_path ────────────────────────────────────────────────────

    #[test]
    fn sensitive_env_file() {
        assert!(is_sensitive_path(Path::new(".env")).is_some());
        assert!(is_sensitive_path(Path::new(".env.local")).is_some());
        assert!(is_sensitive_path(Path::new(".env.production")).is_some());
        assert!(is_sensitive_path(Path::new("/app/.env")).is_some());
    }

    #[test]
    fn sensitive_ssh_dir() {
        assert!(is_sensitive_path(Path::new("/home/user/.ssh/id_rsa")).is_some());
        assert!(is_sensitive_path(Path::new("/home/user/.ssh/authorized_keys")).is_some());
        assert!(is_sensitive_path(Path::new(".ssh/config")).is_some());
    }

    #[test]
    fn sensitive_credentials_files() {
        assert!(is_sensitive_path(Path::new("credentials.json")).is_some());
        assert!(is_sensitive_path(Path::new("client_secret.json")).is_some());
        assert!(is_sensitive_path(Path::new("private_key.pem")).is_some());
        assert!(is_sensitive_path(Path::new("service_account.json")).is_some());
    }

    #[test]
    fn sensitive_auth_config() {
        assert!(is_sensitive_path(Path::new(".npmrc")).is_some());
        assert!(is_sensitive_path(Path::new(".netrc")).is_some());
        assert!(is_sensitive_path(Path::new(".pypirc")).is_some());
    }

    #[test]
    fn sensitive_system_files() {
        assert!(is_sensitive_path(Path::new("/etc/shadow")).is_some());
        assert!(is_sensitive_path(Path::new("/etc/passwd")).is_some());
    }

    #[test]
    fn sensitive_key_extensions() {
        assert!(is_sensitive_path(Path::new("server.pem")).is_some());
        assert!(is_sensitive_path(Path::new("tls.key")).is_some());
        assert!(is_sensitive_path(Path::new("cert.p12")).is_some());
    }

    #[test]
    fn sensitive_gnupg() {
        assert!(is_sensitive_path(Path::new("/home/user/.gnupg/private-keys-v1.d/key")).is_some());
    }

    #[test]
    fn not_sensitive_normal_files() {
        assert!(is_sensitive_path(Path::new("src/main.rs")).is_none());
        assert!(is_sensitive_path(Path::new("README.md")).is_none());
        assert!(is_sensitive_path(Path::new("Cargo.toml")).is_none());
        assert!(is_sensitive_path(Path::new("package.json")).is_none());
        assert!(is_sensitive_path(Path::new(".gitignore")).is_none());
    }

    #[test]
    fn check_sensitive_blocked_returns_message() {
        let msg = check_sensitive_blocked(Path::new(".env"), ".env", "write to");
        assert!(msg.is_some());
        let msg = msg.unwrap();
        assert!(msg.contains("Blocked"), "{}", msg);
        assert!(msg.contains("secrets"), "{}", msg);
        assert!(msg.contains("--dangerously-skip-permissions"), "{}", msg);
    }

    #[test]
    fn check_sensitive_blocked_safe_path_returns_none() {
        assert!(check_sensitive_blocked(Path::new("src/lib.rs"), "src/lib.rs", "edit").is_none());
    }

    #[test]
    fn symlink_in_directory_chain() {
        let tmp = TempDir::new().unwrap();
        let real_dir = tmp.path().join("real_dir");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::write(real_dir.join("file.txt"), "content").unwrap();
        let link_dir = tmp.path().join("link_dir");
        symlink(&real_dir, &link_dir).unwrap();
        let r = validate_path(link_dir.join("file.txt").to_str().unwrap(), tmp.path()).unwrap();
        assert!(r.has_symlink);
    }
}
