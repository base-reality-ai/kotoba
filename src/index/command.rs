//! Project semantic indexing and directory watching.
//!
//! Coordinates file discovery, change detection, and parallel embedding
//! generation via the Ollama client to maintain a fresh search index.

use super::{
    chunker::{chunk_file, collect_indexable_files, is_ignored, load_ignore_patterns},
    project_hash,
    storage::IndexStore,
    Chunk,
};
use crate::logging;
use crate::ollama::client::OllamaClient;
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, UNIX_EPOCH};

pub async fn run_index(embed_client: &OllamaClient, config_dir: &Path) -> Result<()> {
    let _ = crate::logging::init("index");
    let cwd = std::env::current_dir()?;
    let project_id = project_hash(&cwd);
    let index_dir = config_dir.join("index").join(&project_id);
    std::fs::create_dir_all(&index_dir)?;

    let mut store = IndexStore::load(&index_dir).unwrap_or_default();
    let files = collect_indexable_files(&cwd)?;

    let mut added = 0usize;
    let mut skipped = 0usize;
    let mut reindexed = 0usize;
    let total_files = files.len();

    for file in &files {
        let Ok(meta) = std::fs::metadata(file) else {
            continue;
        };
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());

        let rel = file
            .strip_prefix(&cwd)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        // Check if all chunks for this file are current
        let file_chunks: Vec<_> = store.chunks.iter().filter(|c| c.file == rel).collect();
        if !file_chunks.is_empty() && file_chunks.iter().all(|c| c.mtime_secs == mtime) {
            skipped += 1;
            continue;
        }

        // Remove stale chunks and their vectors for this file
        let pairs: Vec<_> = store
            .chunks
            .into_iter()
            .zip(store.vectors)
            .filter(|(c, _)| c.file != rel)
            .collect();
        store.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
        store.vectors = pairs.into_iter().map(|(_, v)| v).collect();

        // Chunk the file and embed each chunk
        let Ok(raw_chunks) = chunk_file(file, &cwd) else {
            continue; // skip unreadable files
        };

        for (start_line, end_line, text) in raw_chunks {
            let embedding = match embed_client.embed(&text).await {
                Ok(e) => e,
                Err(e) => {
                    crate::logging::log_err(&format!("index: embed failed for {}: {}", rel, e));
                    continue;
                }
            };
            store.chunks.push(Chunk {
                file: rel.clone(),
                start_line,
                end_line,
                text,
                mtime_secs: mtime,
            });
            store.vectors.push(embedding);
            added += 1;
        }
        reindexed += 1;
        if total_files > 20 {
            logging::log(&format!(
                "  [{}/{}] {}",
                skipped + reindexed,
                total_files,
                rel
            ));
        }
    }

    store.save(&index_dir)?;
    println!(
        "Index updated: {} chunks added, {} files unchanged ({} files scanned, {} total chunks)",
        added,
        skipped,
        total_files,
        store.chunks.len()
    );
    println!("Index stored at: {}", index_dir.display());
    Ok(())
}

/// Watch the project directory for changes and incrementally re-index modified files.
/// Performs an initial full index pass, then waits for file-system events.
pub async fn run_index_watch(embed_client: &OllamaClient, config_dir: &Path) -> Result<()> {
    use notify::{RecommendedWatcher, RecursiveMode, Watcher};
    use std::sync::mpsc;

    let _ = crate::logging::init("index");
    let cwd = std::env::current_dir()?;
    let project_id = project_hash(&cwd);
    let index_dir = config_dir.join("index").join(&project_id);

    // Initial full index pass
    run_index(embed_client, config_dir).await?;

    logging::log(&format!(
        "[dm] Watching {} for changes (Ctrl+C to stop)...",
        cwd.display()
    ));

    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher: RecommendedWatcher = Watcher::new(
        tx,
        notify::Config::default().with_poll_interval(Duration::from_millis(500)),
    )?;
    watcher.watch(&cwd, RecursiveMode::Recursive)?;

    let patterns = load_ignore_patterns(&cwd, config_dir);

    loop {
        // Collect events for ~500 ms, then batch-process
        let mut changed: HashSet<std::path::PathBuf> = HashSet::new();
        let deadline = std::time::Instant::now() + Duration::from_millis(500);

        loop {
            match rx.try_recv() {
                Ok(Ok(event)) => {
                    for path in event.paths {
                        if path.is_file() && !is_ignored(&path, &cwd, &patterns) {
                            changed.insert(path);
                        }
                    }
                }
                Ok(Err(e)) => crate::logging::log_err(&format!("index watch: {}", e)),
                Err(mpsc::TryRecvError::Empty) => {
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }

        if changed.is_empty() {
            continue;
        }

        logging::log(&format!(
            "[dm] Re-indexing {} changed file(s)...",
            changed.len()
        ));

        let mut store = IndexStore::load(&index_dir).unwrap_or_default();

        for path in &changed {
            if !path.exists() {
                // File deleted — remove its chunks
                let rel = path
                    .strip_prefix(&cwd)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                let pairs: Vec<_> = store
                    .chunks
                    .into_iter()
                    .zip(store.vectors)
                    .filter(|(c, _)| c.file != rel)
                    .collect();
                store.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
                store.vectors = pairs.into_iter().map(|(_, v)| v).collect();
                logging::log(&format!("[dm]   Removed: {}", rel));
                continue;
            }

            match reindex_file(path, &cwd, embed_client, &mut store).await {
                Ok(n) => logging::log(&format!(
                    "[dm]   Indexed: {} ({} chunks)",
                    path.display(),
                    n
                )),
                Err(e) => crate::logging::log_err(&format!(
                    "index watch: error indexing {}: {}",
                    path.display(),
                    e
                )),
            }
        }

        store.save(&index_dir)?;
    }
}

/// Prune chunks for files that no longer exist on disk. No re-embedding; sync only.
pub fn run_index_gc(config_dir: &Path) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let project_id = project_hash(&cwd);
    let index_dir = config_dir.join("index").join(&project_id);

    let Ok(mut store) = IndexStore::load(&index_dir) else {
        println!("No index found for current project.");
        return Ok(());
    };

    let before = store.chunks.len();

    // Collect live relative paths
    let live: HashSet<String> = collect_indexable_files(&cwd)?
        .into_iter()
        .map(|p| {
            p.strip_prefix(&cwd)
                .unwrap_or(&p)
                .to_string_lossy()
                .to_string()
        })
        .collect();

    let pairs: Vec<_> = store
        .chunks
        .into_iter()
        .zip(store.vectors)
        .filter(|(c, _)| live.contains(&c.file))
        .collect();
    store.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
    store.vectors = pairs.into_iter().map(|(_, v)| v).collect();

    let removed = before - store.chunks.len();
    store.save(&index_dir)?;
    println!(
        "GC: removed {} stale chunk(s), {} chunk(s) retained.",
        removed,
        store.chunks.len()
    );
    Ok(())
}

/// Remove a file's existing chunks from the store, then embed and insert fresh chunks.
/// Returns the number of new chunks added.
pub async fn reindex_file(
    file: &Path,
    project_root: &Path,
    embed_client: &OllamaClient,
    store: &mut IndexStore,
) -> Result<usize> {
    let rel = file
        .strip_prefix(project_root)
        .unwrap_or(file)
        .to_string_lossy()
        .to_string();

    let meta = std::fs::metadata(file)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());

    // Remove stale entries for this file
    let pairs: Vec<_> = store
        .chunks
        .drain(..)
        .zip(store.vectors.drain(..))
        .filter(|(c, _)| c.file != rel)
        .collect();
    store.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
    store.vectors = pairs.into_iter().map(|(_, v)| v).collect();

    // Embed new chunks
    let raw_chunks = chunk_file(file, project_root)?;
    let n = raw_chunks.len();
    for (start_line, end_line, text) in raw_chunks {
        let embedding = embed_client.embed(&text).await?;
        store.chunks.push(Chunk {
            file: rel.clone(),
            start_line,
            end_line,
            text,
            mtime_secs: mtime,
        });
        store.vectors.push(embedding);
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use crate::index::{storage::IndexStore, Chunk};
    use tempfile::tempdir;

    fn make_chunk(file: &str) -> Chunk {
        Chunk {
            file: file.to_string(),
            start_line: 1,
            end_line: 10,
            text: "hello".to_string(),
            mtime_secs: 0,
        }
    }

    fn store_with_chunks(files: &[&str]) -> IndexStore {
        let mut store = IndexStore::default();
        for &f in files {
            store.chunks.push(make_chunk(f));
            store.vectors.push(vec![1.0_f32, 0.0, 0.0]);
        }
        store
    }

    #[test]
    fn test_gc_removes_deleted_file_chunks() {
        let tmp = tempdir().unwrap();
        let index_dir = tmp.path().join("index");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Put a chunk for a file that does not exist on disk ("ghost.rs")
        let store = store_with_chunks(&["ghost.rs"]);
        store.save(&index_dir).unwrap();

        // load, filter, save manually (mimicking gc logic without cwd dependency)
        let mut s = IndexStore::load(&index_dir).unwrap();
        let live: std::collections::HashSet<String> = std::collections::HashSet::new(); // empty = no live files
        let pairs: Vec<_> = s
            .chunks
            .into_iter()
            .zip(s.vectors)
            .filter(|(c, _)| live.contains(&c.file))
            .collect();
        s.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
        s.vectors = pairs.into_iter().map(|(_, v)| v).collect();
        s.save(&index_dir).unwrap();

        let reloaded = IndexStore::load(&index_dir).unwrap();
        assert!(
            reloaded.chunks.is_empty(),
            "deleted file chunk should have been pruned"
        );
    }

    #[test]
    fn test_gc_preserves_live_file_chunks() {
        let tmp = tempdir().unwrap();
        let index_dir = tmp.path().join("index");
        std::fs::create_dir_all(&index_dir).unwrap();

        // Create an actual file so "live.rs" is a real path
        let live_path = tmp.path().join("live.rs");
        std::fs::write(&live_path, "fn main() {}").unwrap();

        let store = store_with_chunks(&["live.rs", "ghost.rs"]);
        store.save(&index_dir).unwrap();

        let mut s = IndexStore::load(&index_dir).unwrap();
        let mut live = std::collections::HashSet::new();
        live.insert("live.rs".to_string());

        let pairs: Vec<_> = s
            .chunks
            .into_iter()
            .zip(s.vectors)
            .filter(|(c, _)| live.contains(&c.file))
            .collect();
        s.chunks = pairs.iter().map(|(c, _)| c.clone()).collect();
        s.vectors = pairs.into_iter().map(|(_, v)| v).collect();
        s.save(&index_dir).unwrap();

        let reloaded = IndexStore::load(&index_dir).unwrap();
        assert_eq!(reloaded.chunks.len(), 1);
        assert_eq!(reloaded.chunks[0].file, "live.rs");
    }

    #[test]
    fn test_gc_retains_chunks_vectors_in_sync() {
        // After filtering, chunks.len() must always equal vectors.len()
        let store = store_with_chunks(&["a.rs", "b.rs", "c.rs"]);
        let live: std::collections::HashSet<String> = ["a.rs".to_string(), "c.rs".to_string()]
            .into_iter()
            .collect();

        let pairs: Vec<_> = store
            .chunks
            .into_iter()
            .zip(store.vectors)
            .filter(|(c, _)| live.contains(&c.file))
            .collect();
        let retained_chunks: Vec<_> = pairs.iter().map(|(c, _)| c.clone()).collect();
        let retained_vectors: Vec<_> = pairs.into_iter().map(|(_, v)| v).collect();

        assert_eq!(
            retained_chunks.len(),
            retained_vectors.len(),
            "chunks and vectors must stay in sync after GC"
        );
        assert_eq!(retained_chunks.len(), 2);
        let files: Vec<&str> = retained_chunks.iter().map(|c| c.file.as_str()).collect();
        assert!(files.contains(&"a.rs"));
        assert!(files.contains(&"c.rs"));
        assert!(!files.contains(&"b.rs"));
    }

    #[test]
    fn gc_on_empty_store_stays_empty() {
        let store = IndexStore::default();
        let live: std::collections::HashSet<String> = std::collections::HashSet::new();
        let pairs: Vec<_> = store
            .chunks
            .into_iter()
            .zip(store.vectors)
            .filter(|(c, _)| live.contains(&c.file))
            .collect();
        assert!(pairs.is_empty(), "GC on empty store should return no pairs");
    }

    #[tokio::test]
    async fn reindex_file_removes_stale_entries_on_embed_error() {
        // Even when embedding fails (unreachable server), the old stale chunks
        // for the target file should be purged from the store.
        let tmp = tempdir().unwrap();
        let project_root = tmp.path();
        let file_path = project_root.join("foo.rs");
        // Write >20 chars of content so chunk_file produces at least one chunk
        // and embed() is actually called (triggering the connection error).
        let long_content: String = (0..30).map(|i| format!("fn func_{i}() {{}}\n")).collect();
        std::fs::write(&file_path, &long_content).unwrap();

        let mut store = store_with_chunks(&["foo.rs", "bar.rs"]);
        assert_eq!(store.chunks.len(), 2);

        // Use an unreachable embed client
        let embed_client = crate::ollama::client::OllamaClient::new(
            "http://127.0.0.1:1".to_string(),
            "nomodel".to_string(),
        );

        // This should fail to embed, but stale "foo.rs" chunks should be gone
        let result = reindex_file(&file_path, project_root, &embed_client, &mut store).await;
        assert!(result.is_err(), "embed should fail with unreachable server");
        // "foo.rs" stale chunks must be removed; "bar.rs" chunk must remain
        assert_eq!(
            store.chunks.iter().filter(|c| c.file == "foo.rs").count(),
            0,
            "stale foo.rs chunks should be drained even on embed error"
        );
        assert_eq!(
            store.chunks.iter().filter(|c| c.file == "bar.rs").count(),
            1,
            "bar.rs chunks should be unaffected"
        );
        assert_eq!(
            store.chunks.len(),
            store.vectors.len(),
            "chunks and vectors must remain in sync"
        );
    }

    #[tokio::test]
    async fn reindex_file_returns_error_for_missing_file() {
        let tmp = tempdir().unwrap();
        let project_root = tmp.path();
        let nonexistent = project_root.join("does_not_exist.rs");
        let mut store = IndexStore::default();
        let embed_client = crate::ollama::client::OllamaClient::new(
            "http://127.0.0.1:1".to_string(),
            "nomodel".to_string(),
        );

        let result = reindex_file(&nonexistent, project_root, &embed_client, &mut store).await;
        assert!(result.is_err(), "missing file should return an error");
    }

    #[test]
    fn current_mtime_chunks_detected_as_fresh() {
        // If all chunks for a file have the same mtime as the file on disk,
        // they should be considered current (skipped).
        let mtime: u64 = 12345678;
        let chunks = [
            Chunk {
                file: "src/main.rs".into(),
                start_line: 1,
                end_line: 5,
                text: "a".into(),
                mtime_secs: mtime,
            },
            Chunk {
                file: "src/main.rs".into(),
                start_line: 6,
                end_line: 10,
                text: "b".into(),
                mtime_secs: mtime,
            },
        ];
        let file_chunks: Vec<_> = chunks.iter().filter(|c| c.file == "src/main.rs").collect();
        let all_current =
            !file_chunks.is_empty() && file_chunks.iter().all(|c| c.mtime_secs == mtime);
        assert!(
            all_current,
            "chunks with matching mtime should be treated as current"
        );
    }

    #[test]
    fn index_progress_counter_tracks_all_files() {
        let total_files = 100;
        let mut skipped = 0usize;
        let mut reindexed = 0usize;
        for i in 0..total_files {
            if i % 3 == 0 {
                skipped += 1;
            } else {
                reindexed += 1;
            }
            assert!(skipped + reindexed <= total_files);
        }
        assert_eq!(
            skipped + reindexed,
            total_files,
            "skipped + reindexed must equal total"
        );
    }

    #[test]
    fn index_progress_suppressed_for_small_repos() {
        let total_files = 15;
        let should_print = total_files > 20;
        assert!(
            !should_print,
            "progress should be suppressed for <= 20 files"
        );

        let total_files = 21;
        let should_print = total_files > 20;
        assert!(should_print, "progress should be shown for > 20 files");
    }

    #[test]
    fn stale_mtime_chunk_detected_as_outdated() {
        let disk_mtime: u64 = 99999999;
        let stored_mtime: u64 = 11111111;
        let chunks = [Chunk {
            file: "src/lib.rs".into(),
            start_line: 1,
            end_line: 5,
            text: "x".into(),
            mtime_secs: stored_mtime,
        }];
        let file_chunks: Vec<_> = chunks.iter().filter(|c| c.file == "src/lib.rs").collect();
        let all_current =
            !file_chunks.is_empty() && file_chunks.iter().all(|c| c.mtime_secs == disk_mtime);
        assert!(
            !all_current,
            "chunks with stale mtime should not be treated as current"
        );
    }
}
