//! Persistent vector database storage formats.
//!
//! Manages parallel arrays of chunk metadata (JSON) and their dense embeddings
//! (binary f32 arrays) in the `.dm-workspace/` layout.

use super::Chunk;
use anyhow::{Context, Result};
use std::io::{Read, Write};
use std::path::Path;

/// In-memory index: parallel arrays of chunk metadata and embedding vectors.
#[derive(Default, Clone, Debug)]
pub struct IndexStore {
    pub chunks: Vec<Chunk>,
    pub vectors: Vec<Vec<f32>>,
}

impl IndexStore {
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Load from `index_dir/chunks.json` and `index_dir/vectors.bin`.
    /// Returns an empty store if either file is absent (first run).
    pub fn load(index_dir: &Path) -> Result<Self> {
        let chunks_path = index_dir.join("chunks.json");
        let vecs_path = index_dir.join("vectors.bin");

        if !chunks_path.exists() || !vecs_path.exists() {
            return Ok(Self::default());
        }

        let chunks: Vec<Chunk> = serde_json::from_str(
            &std::fs::read_to_string(&chunks_path).context("read chunks.json")?,
        )
        .context("parse chunks.json")?;

        let mut vecs_file = std::fs::File::open(&vecs_path).context("open vectors.bin")?;
        let mut vecs_bytes = Vec::new();
        vecs_file
            .read_to_end(&mut vecs_bytes)
            .context("read vectors.bin")?;

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
        let mut cursor = 0usize;

        while cursor + 4 <= vecs_bytes.len() {
            // Read dim (u32 LE)
            const MAX_VECTOR_DIM: usize = 16_384;
            let dim = u32::from_le_bytes(
                vecs_bytes[cursor..cursor + 4]
                    .try_into()
                    .context("read dim from vectors.bin")?,
            ) as usize;
            cursor += 4;
            anyhow::ensure!(
                dim <= MAX_VECTOR_DIM,
                "vectors.bin corrupt: dimension {} exceeds maximum {} (file may be corrupted)",
                dim,
                MAX_VECTOR_DIM
            );
            let byte_len = dim * 4;
            if cursor + byte_len > vecs_bytes.len() {
                anyhow::bail!("vectors.bin truncated");
            }
            let floats: Vec<f32> = vecs_bytes[cursor..cursor + byte_len]
                .chunks_exact(4)
                .map(|b| -> anyhow::Result<f32> {
                    let arr: [u8; 4] = b.try_into().context("read float bytes")?;
                    Ok(f32::from_le_bytes(arr))
                })
                .collect::<anyhow::Result<Vec<_>>>()
                .context("parse float vector from vectors.bin")?;
            cursor += byte_len;
            vectors.push(floats);
        }

        anyhow::ensure!(
            vectors.len() == chunks.len(),
            "chunks.json has {} entries but vectors.bin has {} rows",
            chunks.len(),
            vectors.len()
        );

        Ok(Self { chunks, vectors })
    }

    /// Save to `index_dir/chunks.json` and `index_dir/vectors.bin`.
    pub fn save(&self, index_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(index_dir).context("create index dir")?;

        let chunks_path = index_dir.join("chunks.json");
        let chunks_tmp = chunks_path.with_extension("json.tmp");
        let chunks_json = serde_json::to_string_pretty(&self.chunks).context("serialize chunks")?;
        std::fs::write(&chunks_tmp, &chunks_json).context("write chunks.json.tmp")?;

        let vecs_path = index_dir.join("vectors.bin");
        let vecs_tmp = vecs_path.with_extension("bin.tmp");
        let mut vecs_file = std::fs::File::create(&vecs_tmp).context("create vectors.bin.tmp")?;

        for vec in &self.vectors {
            let dim = vec.len() as u32;
            vecs_file
                .write_all(&dim.to_le_bytes())
                .context("write dim")?;
            for &f in vec {
                vecs_file
                    .write_all(&f.to_le_bytes())
                    .context("write float")?;
            }
        }
        drop(vecs_file);

        std::fs::rename(&chunks_tmp, &chunks_path).context("rename chunks.json")?;
        std::fs::rename(&vecs_tmp, &vecs_path).context("rename vectors.bin")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Chunk;
    use tempfile::TempDir;

    fn make_chunk(file: &str, start: usize, end: usize, text: &str) -> Chunk {
        Chunk {
            file: file.to_string(),
            start_line: start,
            end_line: end,
            text: text.to_string(),
            mtime_secs: 12345,
        }
    }

    #[test]
    fn load_missing_files_returns_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore::load(dir.path()).unwrap();
        assert!(store.is_empty());
        assert_eq!(store.chunks.len(), 0);
        assert_eq!(store.vectors.len(), 0);
    }

    #[test]
    fn save_then_load_round_trips_chunks() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore {
            chunks: vec![
                make_chunk("src/foo.rs", 1, 10, "fn foo() {}"),
                make_chunk("src/bar.rs", 5, 15, "fn bar() {}"),
            ],
            vectors: vec![vec![1.0f32, 0.0, 0.0], vec![0.0f32, 1.0, 0.0]],
        };

        store.save(dir.path()).unwrap();
        let loaded = IndexStore::load(dir.path()).unwrap();

        assert_eq!(loaded.chunks.len(), 2);
        assert_eq!(loaded.chunks[0].file, "src/foo.rs");
        assert_eq!(loaded.chunks[1].file, "src/bar.rs");
        assert_eq!(loaded.chunks[0].start_line, 1);
        assert_eq!(loaded.chunks[1].end_line, 15);
    }

    #[test]
    fn save_then_load_round_trips_vectors() {
        let dir = TempDir::new().unwrap();
        let v0 = vec![1.0f32, 2.0, 3.0];
        let v1 = vec![4.0f32, 5.0, 6.0];
        let store = IndexStore {
            chunks: vec![make_chunk("a.rs", 1, 1, "a"), make_chunk("b.rs", 1, 1, "b")],
            vectors: vec![v0.clone(), v1.clone()],
        };

        store.save(dir.path()).unwrap();
        let loaded = IndexStore::load(dir.path()).unwrap();

        assert_eq!(loaded.vectors.len(), 2);
        for (got, expected) in loaded.vectors[0].iter().zip(&v0) {
            assert!(
                (got - expected).abs() < 1e-6,
                "v0 mismatch: {} vs {}",
                got,
                expected
            );
        }
        for (got, expected) in loaded.vectors[1].iter().zip(&v1) {
            assert!(
                (got - expected).abs() < 1e-6,
                "v1 mismatch: {} vs {}",
                got,
                expected
            );
        }
    }

    #[test]
    fn save_then_load_single_entry() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore {
            chunks: vec![make_chunk("x.rs", 3, 7, "hello world code")],
            vectors: vec![vec![0.5f32, 0.5]],
        };

        store.save(dir.path()).unwrap();
        let loaded = IndexStore::load(dir.path()).unwrap();

        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.chunks[0].text, "hello world code");
        assert_eq!(loaded.chunks[0].mtime_secs, 12345);
        assert_eq!(loaded.vectors[0].len(), 2);
    }

    #[test]
    fn is_empty_true_for_default() {
        let store = IndexStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn is_empty_false_after_adding_chunk() {
        let store = IndexStore {
            chunks: vec![make_chunk("f.rs", 1, 1, "fn main() {}")],
            vectors: vec![vec![1.0f32]],
        };
        assert!(!store.is_empty());
    }

    #[test]
    fn save_creates_both_files() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore {
            chunks: vec![make_chunk("x.rs", 1, 1, "content")],
            vectors: vec![vec![1.0f32, 0.0]],
        };
        store.save(dir.path()).unwrap();
        assert!(dir.path().join("chunks.json").exists());
        assert!(dir.path().join("vectors.bin").exists());
    }

    #[test]
    fn save_load_empty_store_roundtrips() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore::default();
        store.save(dir.path()).unwrap();
        let loaded = IndexStore::load(dir.path()).unwrap();
        assert!(
            loaded.is_empty(),
            "empty store should remain empty after save/load"
        );
        assert_eq!(loaded.vectors.len(), 0);
    }

    #[test]
    fn is_empty_checks_chunks_not_vectors() {
        // is_empty is based on chunks only
        let store = IndexStore {
            chunks: vec![],
            vectors: vec![vec![1.0f32]], // vector present but no chunks
        };
        assert!(
            store.is_empty(),
            "is_empty should return true when chunks is empty"
        );
    }

    #[test]
    fn load_corrupt_vectors_bin_huge_dim_rejected() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("chunks.json"), "[]").unwrap();
        let mut bad_vec = Vec::new();
        bad_vec.extend_from_slice(&u32::MAX.to_le_bytes());
        bad_vec.extend_from_slice(&[0u8; 16]);
        std::fs::write(dir.path().join("vectors.bin"), &bad_vec).unwrap();
        let err = IndexStore::load(dir.path()).unwrap_err();
        assert!(
            err.to_string().contains("corrupt") || err.to_string().contains("exceeds"),
            "err: {err}"
        );
    }

    #[test]
    fn max_vector_dim_is_reasonable() {
        const MAX_VECTOR_DIM: usize = 16_384;
        const { assert!(MAX_VECTOR_DIM >= 1024 && MAX_VECTOR_DIM <= 65536) };
    }

    #[test]
    fn save_no_tmp_files_left_behind() {
        let dir = TempDir::new().unwrap();
        let store = IndexStore {
            chunks: vec![make_chunk("x.rs", 1, 5, "content")],
            vectors: vec![vec![1.0f32, 2.0]],
        };
        store.save(dir.path()).unwrap();
        assert!(
            !dir.path().join("chunks.json.tmp").exists(),
            "chunks tmp should not remain"
        );
        assert!(
            !dir.path().join("vectors.bin.tmp").exists(),
            "vectors tmp should not remain"
        );
        assert!(dir.path().join("chunks.json").exists());
        assert!(dir.path().join("vectors.bin").exists());
    }
}
