use std::fmt::Write as _;

use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct WikiSearchTool;

#[async_trait]
impl Tool for WikiSearchTool {
    fn name(&self) -> &'static str {
        "wiki_search"
    }

    fn description(&self) -> &'static str {
        "Search the project's wiki for pages matching a keyword or phrase. \
         Returns matching pages with titles, paths, match counts, and snippets. \
         Use to recall what the wiki knows about entities, concepts, or past decisions. \
         On a complete miss, lists the closest near-miss page titles by edit distance \
         to ease debugging."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for (case-insensitive substring match)"
                },
                "category": {
                    "type": "string",
                    "enum": ["entity", "concept", "summary", "synthesis"],
                    "description": "Optional: filter results to a single category (entity, concept, summary, or synthesis)"
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
            "Use wiki_search to query the project's living wiki before asking \
             questions the wiki may already answer. Good for: finding entity \
             pages, recalling past decisions, checking conventions, or locating \
             related work. Prefer wiki_search over file_read when you need \
             conceptual/synthetic knowledge rather than raw source.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let query = args["query"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: query"))?;

        let cwd = std::env::current_dir().unwrap_or_default();
        let identity = crate::identity::load_for_cwd();
        let wiki = match crate::wiki::Wiki::open(&cwd) {
            Ok(w) => w,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("No wiki found for this project. Error: {}", e),
                    is_error: false,
                });
            }
        };

        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Ok(ToolResult {
                content: "wiki_search: query is empty. \
                          Try: provide a keyword like 'compaction' or 'session', \
                          optionally with a category prefix ('entity:', 'concept:', \
                          'summary:', 'synthesis:') or a `category` parameter."
                    .to_string(),
                is_error: true,
            });
        }

        let hits = match wiki.search_for_identity(trimmed, &identity) {
            Ok(h) => h,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Wiki search failed: {}", e),
                    is_error: true,
                });
            }
        };

        // Validate `category` if present. Match the slash command's
        // case-insensitive contract (see wiki/commands/wiki.rs::parse_category_prefix)
        // so "model and operator see byte-identical fuzzy semantics".
        let category_filter: Option<String> = match args["category"].as_str() {
            None => None,
            Some(raw) => {
                let cat = raw.trim().to_lowercase();
                if cat.is_empty() {
                    return Ok(ToolResult {
                        content: "wiki_search: `category` is empty. \
                                  Try: omit the parameter entirely, or pass one of \
                                  'entity', 'concept', 'summary', 'synthesis'."
                            .to_string(),
                        is_error: true,
                    });
                }
                match cat.as_str() {
                    "entity" | "concept" | "summary" | "synthesis" => Some(cat),
                    _ => {
                        return Ok(ToolResult {
                            content: format!(
                                "wiki_search: unknown category '{}'. \
                                 Try: one of 'entity', 'concept', 'summary', 'synthesis' \
                                 (case-insensitive).",
                                raw
                            ),
                            is_error: true,
                        });
                    }
                }
            }
        };

        let hits: Vec<_> = if let Some(ref cat) = category_filter {
            hits.into_iter()
                .filter(|h| h.category.as_str() == cat)
                .collect()
        } else {
            hits
        };

        if hits.is_empty() {
            // Fuzzy fallback: rank index entries by Levenshtein distance and
            // surface the closest. Empty index → bare 404 (no chatter).
            // Shared logic lives in `wiki::fuzzy`.
            let category = category_filter.as_deref();
            let needle = trimmed.to_lowercase();
            let max_dist = crate::wiki::fuzzy::fuzzy_threshold(&needle);
            let ranked = match wiki.load_index() {
                Ok(idx) => crate::wiki::fuzzy::rank_entries_by_levenshtein(
                    &idx.entries,
                    &needle,
                    category,
                    3,
                ),
                Err(_) => Vec::new(),
            };

            let accepted = ranked.first().is_some_and(|(d, _, _)| *d <= max_dist);
            if accepted {
                let mut content = format!("No matches for '{}'. Closest titles:\n", trimmed);
                for (d, p, t) in ranked.iter().take_while(|(d, _, _)| *d <= max_dist) {
                    writeln!(content, "  - {} — {} (distance {})", p, t, d)
                        .expect("write to String never fails");
                }
                content
                    .push_str("\nTip: try wiki_lookup with the path above to read the full page.");
                return Ok(ToolResult {
                    content,
                    is_error: false,
                });
            }

            if !ranked.is_empty() {
                let mut content = format!(
                    "No wiki matches for '{}'. Closest titles \
                     (above similarity threshold {}):\n",
                    trimmed, max_dist
                );
                for (d, p, t) in &ranked {
                    writeln!(content, "  - {} — {} (distance {})", p, t, d)
                        .expect("write to String never fails");
                }
                content.push_str("\nTry wiki_lookup if you know the path.");
                return Ok(ToolResult {
                    content,
                    is_error: false,
                });
            }

            return Ok(ToolResult {
                content: format!("No wiki matches for '{}'.", trimmed),
                is_error: false,
            });
        }

        let mut content = format!(
            "Found {} match{} for '{}':",
            hits.len(),
            if hits.len() == 1 { "" } else { "es" },
            trimmed,
        );
        for h in &hits {
            if identity.is_host() {
                write!(
                    content,
                    "\n  [{}] [{}x] {} — {}\n        {}",
                    h.layer.as_str(),
                    h.match_count,
                    h.path,
                    h.title,
                    h.snippet,
                )
                .expect("write to String never fails");
            } else {
                write!(
                    content,
                    "\n  [{}x] {} — {}\n        {}",
                    h.match_count, h.path, h.title, h.snippet,
                )
                .expect("write to String never fails");
            }
        }

        Ok(ToolResult {
            content,
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn name_and_schema() {
        let t = WikiSearchTool;
        assert_eq!(t.name(), "wiki_search");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("query")));
        assert!(p["properties"]["query"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(WikiSearchTool.is_read_only());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn search_finds_content_in_wiki() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();

        // Create a properly-formatted entity page
        let page = crate::wiki::WikiPage {
            title: "TestEntity".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/foo.rs".to_string()],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "This entity handles authentication and session tokens.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/test-entity.md", &page).unwrap();

        // Update the index so search discovers the page
        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/test-entity.md".to_string(),
            title: "TestEntity".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "core entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        let result = WikiSearchTool
            .call(json!({"query": "authentication"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("TestEntity"),
            "should find entity page: {}",
            result.content
        );
        assert!(
            result.content.contains("authentication"),
            "should mention query: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn search_filters_by_category() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let entity_page = crate::wiki::WikiPage {
            title: "TestEntity".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/foo.rs".to_string()],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "This entity handles auth.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/test-entity.md", &entity_page)
            .unwrap();

        let concept_page = crate::wiki::WikiPage {
            title: "AuthPattern".to_string(),
            page_type: crate::wiki::PageType::Concept,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Authentication patterns in the codebase.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("concepts/auth-pattern.md", &concept_page)
            .unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/test-entity.md".to_string(),
            title: "TestEntity".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "core entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        idx.entries.push(crate::wiki::IndexEntry {
            path: "concepts/auth-pattern.md".to_string(),
            title: "AuthPattern".to_string(),
            category: crate::wiki::PageType::Concept,
            one_liner: "auth concept".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        let result = WikiSearchTool
            .call(json!({"query": "auth", "category": "entity"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("TestEntity"),
            "should find entity: {}",
            result.content
        );
        assert!(
            !result.content.contains("AuthPattern"),
            "should not find concept: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn host_mode_search_labels_layers_and_orders_host_first() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        std::fs::create_dir_all(tmp.path().join(".dm")).unwrap();
        std::fs::write(
            tmp.path().join(".dm/identity.toml"),
            "mode = \"host\"\nhost_project = \"finance-app\"\ncanonical_dm_revision = \"abc123\"\n",
        )
        .unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();
        let host_page = crate::wiki::WikiPage {
            title: "HostFinance".to_string(),
            page_type: crate::wiki::PageType::Concept,
            layer: crate::wiki::Layer::Host,
            sources: vec![],
            last_updated: "2026-04-26 00:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "auth".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("concepts/host-finance.md", &host_page)
            .unwrap();
        let kernel_page = crate::wiki::WikiPage {
            title: "KernelAuth".to_string(),
            page_type: crate::wiki::PageType::Concept,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: "2026-04-26 00:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "auth auth auth".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("concepts/kernel-auth.md", &kernel_page)
            .unwrap();
        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "concepts/kernel-auth.md".to_string(),
            title: "KernelAuth".to_string(),
            category: crate::wiki::PageType::Concept,
            one_liner: "kernel".to_string(),
            last_updated: Some("2026-04-26 00:00:00".to_string()),
            outcome: None,
        });
        idx.entries.push(crate::wiki::IndexEntry {
            path: "concepts/host-finance.md".to_string(),
            title: "HostFinance".to_string(),
            category: crate::wiki::PageType::Concept,
            one_liner: "host".to_string(),
            last_updated: Some("2026-04-26 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        let result = WikiSearchTool.call(json!({"query": "auth"})).await.unwrap();

        assert!(!result.is_error, "{}", result.content);
        let host_pos = result.content.find("[host]").expect("host label");
        let kernel_pos = result.content.find("[kernel]").expect("kernel label");
        assert!(
            host_pos < kernel_pos,
            "host-layer hit must render before kernel-layer hit: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn empty_query_returns_clear_error_with_next_step() {
        // Updated cycle 442: blank query is now `is_error: true` with a
        // "Try: …" next-step hint, matching the directive's error-message
        // contract and parity with wiki_lookup's blank-path guard.
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        // Ensure wiki exists
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiSearchTool.call(json!({"query": ""})).await.unwrap();
        assert!(result.is_error, "blank query should be flagged is_error");
        assert!(
            result.content.contains("query is empty"),
            "should explain the failure: {}",
            result.content
        );
        assert!(
            result.content.contains("Try:"),
            "should include next-step hint per directive: {}",
            result.content
        );

        // Whitespace-only query hits the same guard.
        let result = WikiSearchTool.call(json!({"query": "   "})).await.unwrap();
        assert!(result.is_error, "whitespace-only query should be flagged");

        let _ = std::env::set_current_dir("/");
    }

    /// Pins the canonical `Try: ...` phrasing on the blank-QUERY guard so a
    /// future refactor that strips the keyword example or category-prefix
    /// example doesn't go unnoticed. The hint steers the model toward both
    /// the keyword form ('compaction') and the colon-syntax form ('entity:').
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_search_blank_query_pins_canonical_try_hint() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiSearchTool.call(json!({"query": ""})).await.unwrap();
        let _ = std::env::set_current_dir("/");

        assert!(result.is_error);
        let c = &result.content;
        assert!(
            c.contains("wiki_search: query is empty"),
            "missing canonical lead: {}",
            c
        );
        assert!(c.contains("Try:"), "missing Try: hint: {}", c);
        assert!(c.contains("'compaction'"), "missing keyword example: {}", c);
        assert!(
            c.contains("'entity:'"),
            "missing colon-syntax example: {}",
            c
        );
    }

    /// Pins the canonical `Try: ...` phrasing on the empty-CATEGORY guard
    /// (distinct from blank-QUERY). The hint must list all four allowed
    /// categories — treating the list as the canonical contract so future
    /// drops or renames trigger this test.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_search_empty_category_pins_all_four_options() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        // Non-empty query (so we skip the blank-query guard) plus an empty
        // string for category to land on the empty-category branch.
        let result = WikiSearchTool
            .call(json!({"query": "anything", "category": ""}))
            .await
            .unwrap();
        let _ = std::env::set_current_dir("/");

        assert!(result.is_error);
        let c = &result.content;
        assert!(
            c.contains("`category` is empty"),
            "missing canonical lead: {}",
            c
        );
        assert!(c.contains("Try:"), "missing Try: hint: {}", c);
        // Treat the four allowed names as the canonical contract.
        for cat in ["'entity'", "'concept'", "'summary'", "'synthesis'"] {
            assert!(c.contains(cat), "category list must include {}: {}", cat, c);
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn empty_query_hint_mentions_category_options() {
        // After C6 the tool accepts an optional `category` JSON arg; the
        // empty-query hint must surface that. Drift trip: future
        // wording changes that drop the category list should fail.
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiSearchTool.call(json!({"query": ""})).await.unwrap();
        let _ = std::env::set_current_dir("/");

        assert!(result.is_error, "blank query is now is_error: true");
        let c = &result.content;
        assert!(c.contains("category"), "must name the param: {}", c);
        assert!(
            c.contains("entity") && c.contains("concept") && c.contains("synthesis"),
            "must list category options: {}",
            c
        );
    }

    /// E2E mirror of `wiki::tests::c9_c10_backfilled_concept_pages_are_search_findable`
    /// at the LLM-tool layer: dispatch through the registry, assert the
    /// rendered tool content surfaces `concepts/error-handling.md` for the
    /// same query that the engine layer hits. Closes the LLM-tool side of
    /// the C12 (preamble) + C14 (engine) discoverability win.
    ///
    /// Skips when the test runs outside the dark-matter package
    /// (sandboxed runner, workspace overlay).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn live_wiki_search_dispatch_finds_backfilled_concept_page() {
        let _telem = crate::telemetry::telemetry_test_guard();
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let project_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !project_root
            .join(".dm/wiki/concepts/error-handling.md")
            .is_file()
        {
            return;
        }
        let orig = std::env::current_dir().unwrap_or_else(|_| project_root.clone());
        std::env::set_current_dir(&project_root).unwrap();

        let mut registry = crate::tools::registry::ToolRegistry::new();
        registry.register(WikiSearchTool);

        let result = registry
            .call("wiki_search", json!({"query": "logging system"}))
            .await
            .expect("dispatch ok");

        // Restore cwd before assertions so a panic doesn't leak it.
        let _ = std::env::set_current_dir(&orig);

        assert!(
            !result.is_error,
            "dispatch must succeed: {}",
            result.content
        );
        assert!(
            result.content.contains("concepts/error-handling.md"),
            "wiki_search tool must surface error-handling.md for 'logging system': {}",
            result.content
        );
        assert!(
            result.content.contains("Found"),
            "must use the substring-hit render path (Found N matches), \
             not the fuzzy fallback: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn missing_query_errors() {
        let result = WikiSearchTool.call(json!({})).await;
        assert!(result.is_err(), "missing query should error");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn no_matches_returns_info() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiSearchTool
            .call(json!({"query": "xyzzy-not-found"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No wiki matches"),
            "should say no matches: {}",
            result.content
        );
        assert!(
            !result.content.contains("similarity threshold"),
            "empty index must not emit fuzzy chatter: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn search_fuzzy_finds_title_typo() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();
        let page = crate::wiki::WikiPage {
            title: "AuthService".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/auth.rs".to_string()],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Handles login.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/auth-service.md", &page).unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/auth-service.md".to_string(),
            title: "AuthService".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "auth entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        // Title typo: missing 'v' — substring search misses, fuzzy finds.
        let result = WikiSearchTool
            .call(json!({"query": "AuthSerice"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Closest titles:"),
            "should emit accepted-fuzzy header: {}",
            result.content
        );
        assert!(
            result.content.contains("AuthService"),
            "should surface near-miss title: {}",
            result.content
        );
        assert!(
            result.content.contains("(distance "),
            "should annotate distance: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn search_fuzzy_below_threshold_lists_rejected() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();
        let page = crate::wiki::WikiPage {
            title: "SessionManager".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/session.rs".to_string()],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Session bookkeeping.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/session-manager.md", &page)
            .unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/session-manager.md".to_string(),
            title: "SessionManager".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "session entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        let result = WikiSearchTool
            .call(json!({"query": "completely-unrelated-xyzzy"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No wiki matches"),
            "should still 404: {}",
            result.content
        );
        assert!(
            result.content.contains("above similarity threshold"),
            "should list rejected near-misses: {}",
            result.content
        );
        assert!(
            result.content.contains("(distance "),
            "rejected list should annotate distance: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn search_fuzzy_respects_category_filter() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let entity_page = crate::wiki::WikiPage {
            title: "AuthEntity".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/auth.rs".to_string()],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Auth entity.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/auth-entity.md", &entity_page)
            .unwrap();

        let concept_page = crate::wiki::WikiPage {
            title: "AuthPattern".to_string(),
            page_type: crate::wiki::PageType::Concept,
            layer: crate::wiki::Layer::Kernel,
            sources: vec![],
            last_updated: "2026-04-24 00:00:00".to_string(),
            entity_kind: None,
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Auth concept.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("concepts/auth-pattern.md", &concept_page)
            .unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/auth-entity.md".to_string(),
            title: "AuthEntity".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "auth entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        idx.entries.push(crate::wiki::IndexEntry {
            path: "concepts/auth-pattern.md".to_string(),
            title: "AuthPattern".to_string(),
            category: crate::wiki::PageType::Concept,
            one_liner: "auth concept".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        // Title typo + category=concept → only AuthPattern surfaces.
        let result = WikiSearchTool
            .call(json!({"query": "AuthSerice", "category": "concept"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("AuthPattern"),
            "concept fuzzy hit should surface: {}",
            result.content
        );
        assert!(
            !result.content.contains("AuthEntity"),
            "entity must be filtered out of fuzzy fallback: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn category_validation_rejects_blank_and_unknown_with_next_step() {
        // Cycle 443: category arg is now validated. Blank string and
        // unknown values used to silently filter to zero hits with no
        // hint that the category was the cause. They now return
        // is_error: true with a "Try: …" next step listing valid options.
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        // Blank string: rejected with "category is empty" message.
        let result = WikiSearchTool
            .call(json!({"query": "auth", "category": ""}))
            .await
            .unwrap();
        assert!(result.is_error, "blank category must be flagged");
        assert!(
            result.content.contains("`category` is empty"),
            "should explain failure: {}",
            result.content
        );
        assert!(
            result.content.contains("Try:"),
            "should include next step: {}",
            result.content
        );

        // Unknown value (plural typo): rejected with "unknown category".
        let result = WikiSearchTool
            .call(json!({"query": "auth", "category": "entities"}))
            .await
            .unwrap();
        assert!(result.is_error, "unknown category must be flagged");
        assert!(
            result.content.contains("unknown category 'entities'"),
            "should echo the bad value: {}",
            result.content
        );
        assert!(
            result.content.contains("entity") && result.content.contains("synthesis"),
            "should list valid options: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn category_validation_accepts_case_insensitively() {
        // Parity with the `/wiki search` slash command's
        // `parse_category_prefix`, which is case-insensitive. Tool
        // and slash must see byte-identical fuzzy semantics.
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let wiki = crate::wiki::Wiki::open(tmp.path()).unwrap();
        let page = crate::wiki::WikiPage {
            title: "AuthEntity".to_string(),
            page_type: crate::wiki::PageType::Entity,
            layer: crate::wiki::Layer::Kernel,
            sources: vec!["src/auth.rs".to_string()],
            last_updated: "2026-04-25 00:00:00".to_string(),
            entity_kind: Some(crate::wiki::EntityKind::Unknown),
            purpose: None,
            key_exports: vec![],
            dependencies: vec![],
            outcome: None,
            scope: vec![],
            body: "Auth entity body.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/auth-entity.md", &page).unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/auth-entity.md".to_string(),
            title: "AuthEntity".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "auth entity".to_string(),
            last_updated: Some("2026-04-25 00:00:00".to_string()),
            outcome: None,
        });
        std::fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        // Mixed-case "Entity" should match same as "entity".
        let result = WikiSearchTool
            .call(json!({"query": "AuthEntity", "category": "Entity"}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "valid case-insensitive value should pass: {}",
            result.content
        );
        assert!(
            result.content.contains("AuthEntity"),
            "should still find the page: {}",
            result.content
        );

        // Whitespace + uppercase: trimmed and lowered.
        let result = WikiSearchTool
            .call(json!({"query": "AuthEntity", "category": "  ENTITY  "}))
            .await
            .unwrap();
        assert!(
            !result.is_error,
            "trim+lower should pass: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }
}
