use crate::index::{project_hash, storage::IndexStore, Chunk};
use crate::ollama::client::OllamaClient;
use crate::tools::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;

pub struct SemanticSearchTool {
    embed_client: OllamaClient,
    config_dir: PathBuf,
    project_root: PathBuf,
}

impl SemanticSearchTool {
    pub fn new(embed_client: OllamaClient, config_dir: PathBuf, project_root: PathBuf) -> Self {
        Self {
            embed_client,
            config_dir,
            project_root,
        }
    }

    fn index_dir(&self) -> PathBuf {
        let project_id = project_hash(&self.project_root);
        self.config_dir.join("index").join(project_id)
    }
}

#[async_trait]
impl Tool for SemanticSearchTool {
    fn name(&self) -> &'static str {
        "semantic_search"
    }

    fn description(&self) -> &'static str {
        "Search the indexed codebase by natural language or exact identifiers. \
         Supports semantic (meaning-based), keyword (exact match), and hybrid modes. \
         Requires `dm index` to have been run first."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or identifier-based search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 20)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "default": "hybrid",
                    "description": "'hybrid' (default) combines embedding similarity with keyword matching — best for most queries. 'keyword' is better for exact identifier names. 'semantic' is better for conceptual queries."
                }
            },
            "required": ["query"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use semantic_search to find code by meaning, not just text matching. \
              Good for finding implementations when you don't know the exact name.",
        )
    }

    async fn call(&self, args: serde_json::Value) -> Result<ToolResult> {
        let query = args["query"].as_str().unwrap_or("").to_string();
        if query.is_empty() {
            return Ok(ToolResult {
                content: "query parameter is required".to_string(),
                is_error: true,
            });
        }
        let top_k = (args["top_k"].as_u64().unwrap_or(5) as usize).min(20);
        let mode = args["mode"].as_str().unwrap_or("hybrid");

        let index_dir = self.index_dir();
        let store = match IndexStore::load(&index_dir) {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => {
                return Ok(ToolResult {
                    content: "Index is empty. Run `dm index` to build it.".to_string(),
                    is_error: false,
                });
            }
            Err(_) => {
                return Ok(ToolResult {
                    content: "No index found. Run `dm index` first.".to_string(),
                    is_error: false,
                });
            }
        };

        let n = store.chunks.len();

        let semantic_scores: Vec<f32> = if mode != "keyword" {
            let query_vec = self.embed_client.embed(&query).await?;
            store
                .vectors
                .iter()
                .map(|v| cosine_similarity(&query_vec, v))
                .collect()
        } else {
            vec![0.0f32; n]
        };

        let keyword_scores: Vec<f32> = if mode != "semantic" {
            keyword_scores(&query, &store.chunks)
        } else {
            vec![0.0f32; n]
        };

        let (sem_w, kw_w): (f32, f32) = match mode {
            "semantic" => (1.0, 0.0),
            "keyword" => (0.0, 1.0),
            _ => (0.7, 0.3),
        };

        let mut scored: Vec<(f32, usize)> = semantic_scores
            .iter()
            .zip(keyword_scores.iter())
            .enumerate()
            .map(|(i, (s, k))| (sem_w * s + kw_w * k, i))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<String> = scored
            .iter()
            .take(top_k)
            .filter(|(score, _)| *score > 0.01)
            .map(|(score, i)| {
                let chunk = &store.chunks[*i];
                format!(
                    "## {}:{}-{} (score: {:.3})\n{}",
                    chunk.file, chunk.start_line, chunk.end_line, score, chunk.text
                )
            })
            .collect();

        if results.is_empty() {
            return Ok(ToolResult {
                content: "No relevant results found.".to_string(),
                is_error: false,
            });
        }

        Ok(ToolResult {
            content: results.join("\n\n---\n\n"),
            is_error: false,
        })
    }
}

/// Cosine similarity between two equal-length float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

/// TF-like keyword scoring: fraction of query terms (len >= 2) found in chunk text.
/// Single-character tokens are excluded from scoring.
/// Uses word-boundary-aware matching for short terms (len <= 4) to avoid
/// false positives like "fn" matching inside "function".
pub fn keyword_scores(query: &str, chunks: &[Chunk]) -> Vec<f32> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() > 1)
        .map(|t| t.to_lowercase())
        .collect();

    if terms.is_empty() {
        return vec![0.0; chunks.len()];
    }

    chunks
        .iter()
        .map(|chunk| {
            let text_lower = chunk.text.to_lowercase();
            let matches = terms
                .iter()
                .filter(|t| {
                    if t.len() <= 4 {
                        word_boundary_match(&text_lower, t)
                    } else {
                        text_lower.contains(t.as_str())
                    }
                })
                .count();
            matches as f32 / terms.len() as f32
        })
        .collect()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn word_boundary_match(haystack: &str, needle: &str) -> bool {
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let abs = start + pos;
        // '\0' is not a word char, so when the short-circuit misses (unreachable
        // per abs==0 / after_pos>=len guards) we still produce the right result.
        let before_ok = abs == 0 || !is_word_char(haystack[..abs].chars().last().unwrap_or('\0'));
        let after_pos = abs + needle.len();
        let after_ok = after_pos >= haystack.len()
            || !is_word_char(haystack[after_pos..].chars().next().unwrap_or('\0'));
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Chunk;

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_identical_unit_vectors_is_one() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_non_unit_vectors_is_one() {
        let v = vec![3.0f32, 4.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors_is_negative_one() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_vectors_returns_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_different_length_returns_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_zero_magnitude_vector_returns_zero() {
        let zero = vec![0.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
    }

    // ── keyword_scores ────────────────────────────────────────────────────────

    fn make_chunk(text: &str) -> Chunk {
        Chunk {
            file: "test.rs".to_string(),
            start_line: 1,
            end_line: 1,
            text: text.to_string(),
            mtime_secs: 0,
        }
    }

    #[test]
    fn keyword_scores_all_terms_match() {
        let chunks = vec![make_chunk("fn foo bar baz")];
        let scores = keyword_scores("foo bar baz", &chunks);
        assert_eq!(scores.len(), 1);
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "all terms should match: {}",
            scores[0]
        );
    }

    #[test]
    fn keyword_scores_no_terms_match() {
        let chunks = vec![make_chunk("hello world")];
        let scores = keyword_scores("xyz qux", &chunks);
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn keyword_scores_partial_match() {
        let chunks = vec![make_chunk("fn foo other")];
        // query has 2 terms: "foo" and "bar"; only "foo" matches → score = 0.5
        let scores = keyword_scores("foo bar", &chunks);
        assert!((scores[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn keyword_scores_empty_query_returns_zeros() {
        let chunks = vec![make_chunk("fn foo"), make_chunk("fn bar")];
        let scores = keyword_scores("", &chunks);
        assert_eq!(scores, vec![0.0, 0.0]);
    }

    #[test]
    fn keyword_scores_single_char_terms_ignored() {
        // "a" and "b" are single-char → filtered out → empty terms → zeros
        let chunks = vec![make_chunk("a b c function")];
        let scores = keyword_scores("a b", &chunks);
        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn keyword_scores_case_insensitive() {
        let chunks = vec![make_chunk("fn MyFunction")];
        let scores = keyword_scores("myfunction", &chunks);
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keyword_scores_underscore_in_term() {
        // Underscores are kept as part of identifiers
        let chunks = vec![make_chunk("fn my_func() {}")];
        let scores = keyword_scores("my_func", &chunks);
        assert!((scores[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keyword_scores_short_term_word_boundary() {
        // "fn" should NOT match inside "function"
        let chunks = vec![make_chunk("pub function do_stuff() {}")];
        let scores = keyword_scores("fn", &chunks);
        assert_eq!(
            scores[0], 0.0,
            "\"fn\" should not match inside \"function\""
        );
    }

    #[test]
    fn keyword_scores_short_term_exact_word_matches() {
        // "fn" should match when it's a standalone word
        let chunks = vec![make_chunk("fn foo() {}")];
        let scores = keyword_scores("fn", &chunks);
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "\"fn\" should match standalone"
        );
    }

    #[test]
    fn keyword_scores_long_term_still_uses_contains() {
        // Terms longer than 4 chars still use substring matching
        let chunks = vec![make_chunk("my_functionality works")];
        let scores = keyword_scores("function", &chunks);
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "long term should use contains"
        );
    }

    #[test]
    fn word_boundary_match_at_edges() {
        assert!(word_boundary_match("fn foo", "fn"), "start of string");
        assert!(word_boundary_match("pub fn", "fn"), "end of string");
        assert!(word_boundary_match("fn", "fn"), "exact string");
        assert!(!word_boundary_match("xfn", "fn"), "no boundary before");
        assert!(!word_boundary_match("fnx", "fn"), "no boundary after");
    }

    #[test]
    fn word_boundary_match_handles_leading_multibyte_char() {
        // Space precedes needle; `chars().last()` on the prefix returns ' ',
        // which is not a word char — boundary holds. The multibyte 'é' is
        // the prior grapheme, proving `.unwrap_or('\0')` handles UTF-8.
        assert!(word_boundary_match("café fn", "fn"));
    }

    #[test]
    fn word_boundary_match_handles_trailing_multibyte_char() {
        // Space separates fn from a multibyte-containing word — boundary holds.
        assert!(word_boundary_match("fn café", "fn"));
        // No separator; 'c' is a word char — boundary breaks.
        assert!(!word_boundary_match("fncafé", "fn"));
    }

    #[test]
    fn keyword_scores_length_matches_chunk_count() {
        let chunks = vec![make_chunk("alpha"), make_chunk("beta"), make_chunk("gamma")];
        let scores = keyword_scores("alpha", &chunks);
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-6); // "alpha" matches first chunk
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[2], 0.0);
    }
}
