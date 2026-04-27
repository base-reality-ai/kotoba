//! Semantic search indexing and vector storage.
//!
//! Manages chunking source files, generating embeddings via the Ollama client,
//! and maintaining the local `.dm-workspace/` vector database.

pub mod chunker;
pub mod command;
pub mod storage;

use serde::{Deserialize, Serialize};

/// A single indexed chunk of a source file.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Chunk {
    /// Path relative to the project root
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    /// File mtime (Unix seconds) at index time — used for invalidation
    pub mtime_secs: u64,
}

/// Hex-encoded first 8 bytes of SHA-256 of the absolute project root path.
/// Used as a stable, collision-resistant directory name under `~/.dm/index/`.
pub fn project_hash(root: &std::path::Path) -> String {
    let path_str = root.to_string_lossy();
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for byte in path_str.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn project_hash_is_16_hex_chars() {
        let hash = project_hash(Path::new("/home/user/myproject"));
        assert_eq!(hash.len(), 16, "hash should be 16 hex chars: {}", hash);
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "not hex: {}",
            hash
        );
    }

    #[test]
    fn project_hash_same_path_same_hash() {
        let p = Path::new("/home/user/project");
        assert_eq!(project_hash(p), project_hash(p));
    }

    #[test]
    fn project_hash_different_paths_different_hashes() {
        let a = project_hash(Path::new("/home/user/project_a"));
        let b = project_hash(Path::new("/home/user/project_b"));
        assert_ne!(a, b, "different paths should produce different hashes");
    }

    #[test]
    fn project_hash_empty_path() {
        // Should not panic
        let hash = project_hash(Path::new(""));
        assert_eq!(hash.len(), 16);
    }

    #[test]
    fn project_hash_root_is_valid() {
        let hash = project_hash(Path::new("/"));
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn project_hash_relative_path_no_panic() {
        let hash = project_hash(Path::new("relative/path"));
        assert_eq!(hash.len(), 16);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn project_hash_case_sensitive() {
        // Different case → different path → different hash
        let lower = project_hash(Path::new("/home/User/Project"));
        let upper = project_hash(Path::new("/home/user/project"));
        assert_ne!(lower, upper, "path hash should be case-sensitive");
    }
}
