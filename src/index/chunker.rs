//! File chunking heuristics for semantic indexing.
//!
//! Splits source code files into overlapping line-based blocks to optimize
//! embedding context windows without losing semantic boundaries.

use anyhow::Result;
use std::path::Path;

const CHUNK_LINES: usize = 60;
const STEP_LINES: usize = 40; // overlap = CHUNK_LINES - STEP_LINES = 20 lines

/// Maximum file size to index (1 MB).
const MAX_FILE_BYTES: u64 = 1_048_576;

/// Default ignore patterns applied when no .dmignore entry overrides them.
const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    "target/**",
    ".git/**",
    "node_modules/**",
    ".cargo/**",
    "__pycache__/**",
    ".venv/**",
    "venv/**",
    "dist/**",
    "build/**",
    ".next/**",
    ".dm/**",
    ".dm-workspace/**",
    "*.bin",
    "*.so",
    "*.dylib",
    "*.exe",
    "*.pyc",
    ".DS_Store",
];

/// Load ignore patterns from:
///   1. Built-in defaults
///   2. `<project_root>/.dmignore`  (project-level)
///   3. `<config_dir>/ignore`          (global user-level)
pub fn load_ignore_patterns(project_root: &Path, config_dir: &Path) -> Vec<glob::Pattern> {
    let mut patterns: Vec<glob::Pattern> = DEFAULT_IGNORE_PATTERNS
        .iter()
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect();

    for ignore_file in &[project_root.join(".dmignore"), config_dir.join("ignore")] {
        if let Ok(content) = std::fs::read_to_string(ignore_file) {
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    if let Ok(pat) = glob::Pattern::new(line) {
                        patterns.push(pat);
                    }
                }
            }
        }
    }

    patterns
}

/// Returns `true` if `path` matches any ignore pattern relative to `project_root`.
pub fn is_ignored(path: &Path, project_root: &Path, patterns: &[glob::Pattern]) -> bool {
    let rel = match path.strip_prefix(project_root) {
        Ok(r) => r,
        Err(_) => return false,
    };
    // Also check just the filename component against single-segment patterns like "*.pyc"
    let filename = path.file_name().unwrap_or_default();
    let filename_path = Path::new(filename);

    patterns
        .iter()
        .any(|p| p.matches_path(rel) || p.matches_path(filename_path))
}

/// Returns `true` if the first 512 bytes contain a null byte (binary file heuristic).
pub(crate) fn is_binary(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else {
        return true;
    };
    let mut buf = [0u8; 512];
    match f.read(&mut buf) {
        Ok(0) => false,
        Ok(n) => buf[..n].contains(&0),
        Err(_) => true,
    }
}

/// Collect all indexable text files under `root`, respecting `.dmignore` patterns.
pub fn collect_indexable_files(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let config_dir = dirs::home_dir().unwrap_or_default().join(".dm");
    let patterns = load_ignore_patterns(root, &config_dir);
    let mut files = Vec::new();
    visit_dir(root, root, &patterns, &mut files);
    Ok(files)
}

fn visit_dir(
    dir: &Path,
    root: &Path,
    patterns: &[glob::Pattern],
    out: &mut Vec<std::path::PathBuf>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden files/dirs
        if name_str.starts_with('.') {
            continue;
        }

        if is_ignored(&path, root, patterns) {
            continue;
        }

        if path.is_dir() {
            visit_dir(&path, root, patterns, out);
        } else if path.is_file() {
            if let Ok(meta) = path.metadata() {
                if meta.len() > MAX_FILE_BYTES {
                    continue;
                }
            }
            if is_binary(&path) {
                continue;
            }
            out.push(path);
        }
    }
}

/// Split `file` into overlapping text chunks.
/// Returns `Vec<(start_line, end_line, text)>` (1-indexed, inclusive).
pub fn chunk_file(path: &Path, project_root: &Path) -> Result<Vec<(usize, usize, String)>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    let rel = path
        .strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();
    let _ = rel; // callers use the start/end indices

    let mut chunks = Vec::new();
    let mut i = 0usize;

    while i < lines.len() {
        let end = (i + CHUNK_LINES).min(lines.len());
        let text = lines[i..end].join("\n");

        // Skip near-empty chunks (e.g. files with only blank lines)
        if text.trim().len() > 20 {
            chunks.push((i + 1, end, text));
        }

        if end == lines.len() {
            break;
        }
        i += STEP_LINES;
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── is_ignored ────────────────────────────────────────────────────────────

    fn make_patterns(pats: &[&str]) -> Vec<glob::Pattern> {
        pats.iter()
            .filter_map(|p| glob::Pattern::new(p).ok())
            .collect()
    }

    #[test]
    fn is_ignored_matches_pattern() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let target = root.join("target").join("debug").join("dm");
        let patterns = make_patterns(&["target/**"]);
        assert!(is_ignored(&target, root, &patterns));
    }

    #[test]
    fn is_ignored_does_not_match_unrelated_file() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let src = root.join("src").join("main.rs");
        let patterns = make_patterns(&["target/**"]);
        assert!(!is_ignored(&src, root, &patterns));
    }

    #[test]
    fn is_ignored_extension_pattern() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let pyc = root.join("src").join("foo.pyc");
        let patterns = make_patterns(&["*.pyc"]);
        assert!(is_ignored(&pyc, root, &patterns));
    }

    #[test]
    fn is_ignored_path_outside_root_returns_false() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        // Path that cannot be stripped of root prefix
        let outside = std::path::Path::new("/some/other/path/file.txt");
        let patterns = make_patterns(&["*.txt"]);
        // strip_prefix fails → returns false
        assert!(!is_ignored(outside, root, &patterns));
    }

    // ── chunk_file ────────────────────────────────────────────────────────────

    fn write_n_lines(dir: &TempDir, n: usize) -> std::path::PathBuf {
        let p = dir.path().join("file.txt");
        let content = (1..=n)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn chunk_file_short_file_single_chunk() {
        let dir = TempDir::new().unwrap();
        let p = write_n_lines(&dir, 10);
        let chunks = chunk_file(&p, dir.path()).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, 1);
        assert_eq!(chunks[0].1, 10);
    }

    #[test]
    fn chunk_file_long_file_produces_multiple_chunks() {
        let dir = TempDir::new().unwrap();
        // 100 lines → should produce more than one chunk (CHUNK_LINES=60, STEP_LINES=40)
        let p = write_n_lines(&dir, 100);
        let chunks = chunk_file(&p, dir.path()).unwrap();
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn chunk_file_chunks_have_correct_line_numbers() {
        let dir = TempDir::new().unwrap();
        let p = write_n_lines(&dir, 80);
        let chunks = chunk_file(&p, dir.path()).unwrap();
        // First chunk starts at line 1
        assert_eq!(chunks[0].0, 1);
        // Second chunk starts at STEP_LINES+1 = 41
        if chunks.len() > 1 {
            assert_eq!(chunks[1].0, STEP_LINES + 1);
        }
    }

    #[test]
    fn chunk_file_near_empty_file_returns_no_chunks() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("empty.txt");
        // Less than 20 non-whitespace chars — all chunks filtered
        std::fs::write(&p, "\n\n\n\n\n").unwrap();
        let chunks = chunk_file(&p, dir.path()).unwrap();
        assert!(chunks.is_empty(), "expected no chunks for near-empty file");
    }

    #[test]
    fn chunk_file_chunk_text_contains_expected_lines() {
        let dir = TempDir::new().unwrap();
        let p = write_n_lines(&dir, 10);
        let chunks = chunk_file(&p, dir.path()).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].2.contains("line 1"));
        assert!(chunks[0].2.contains("line 10"));
    }

    // ── is_binary ────────────────────────────────────────────────────────────

    #[test]
    fn is_binary_text_file_returns_false() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("text.rs");
        std::fs::write(&p, "fn main() {}\n").unwrap();
        assert!(!is_binary(&p), "text file should not be detected as binary");
    }

    #[test]
    fn is_binary_file_with_null_byte_returns_true() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("binary.bin");
        let mut content = b"some text ".to_vec();
        content.push(0u8); // null byte → binary
        std::fs::write(&p, &content).unwrap();
        assert!(
            is_binary(&p),
            "file with null byte should be detected as binary"
        );
    }

    #[test]
    fn is_binary_empty_file_returns_false() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("empty.txt");
        std::fs::write(&p, b"").unwrap();
        assert!(
            !is_binary(&p),
            "empty file should not be detected as binary"
        );
    }

    #[test]
    fn is_binary_null_beyond_512_bytes_ignored() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("long.txt");
        // 600 bytes of 'a' then a null — null is beyond the 512-byte probe window
        let mut content = vec![b'a'; 600];
        content[550] = 0u8;
        std::fs::write(&p, &content).unwrap();
        assert!(
            !is_binary(&p),
            "null beyond 512-byte window should not trigger binary detection"
        );
    }

    #[test]
    fn is_binary_nonexistent_file_returns_true() {
        assert!(
            is_binary(std::path::Path::new("/nonexistent/path/file.bin")),
            "unreadable files should be treated as binary to skip indexing"
        );
    }
}
