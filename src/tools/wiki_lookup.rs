use std::fmt::Write as _;

use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct WikiLookupTool;

#[async_trait]
impl Tool for WikiLookupTool {
    fn name(&self) -> &'static str {
        "wiki_lookup"
    }

    fn description(&self) -> &'static str {
        "Read a specific wiki page by path. Returns the full page content \
         (title, body, sources, last-updated). Use when you know the page \
         you want (e.g. after wiki_search surfaced it). If the path is not \
         found, falls back to substring search across page titles and \
         paths, then to fuzzy match for typos. On a complete miss, lists \
         the closest near-misses with their edit distance to ease debugging."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Wiki-relative path to the page, e.g. 'entities/foo.md' or 'concepts/auth.md'"
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use wiki_lookup after wiki_search to read the full content of \
             a specific page. Provide the wiki-relative path (e.g. \
             'entities/ToolRegistry.md') — not the on-disk absolute path.",
        )
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: path"))?;

        // Guard: blank path would substring-match every index entry
        // (since String::contains("") is always true) and dump the whole
        // wiki as "did you mean…". Return a clear next-step error instead.
        if path.trim().is_empty() {
            return Ok(ToolResult {
                content: "wiki_lookup: path is empty. \
                          Try: provide a wiki-relative path like \
                          'entities/ToolRegistry.md', or use wiki_search \
                          for keyword-based discovery."
                    .to_string(),
                is_error: true,
            });
        }

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

        // Try direct path lookup first.
        match wiki.read_page(path) {
            Ok(page) => {
                let content = if identity.is_host() {
                    format!("Layer: {}\n\n{}", page.layer.as_str(), page.to_markdown())
                } else {
                    page.to_markdown()
                };
                return Ok(ToolResult {
                    content,
                    is_error: false,
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Path not found — fall back to title substring search.
            }
            Err(e) => {
                return Ok(ToolResult {
                    content: format!("Cannot read page '{}': {}", path, e),
                    is_error: true,
                });
            }
        }

        // Title fallback: search index entries for a case-insensitive substring match.
        let idx = match wiki.load_index() {
            Ok(i) => i,
            Err(e) => {
                return Ok(ToolResult {
                    content: format!(
                        "Page '{}' not found. Could not load index for title fallback: {}",
                        path, e
                    ),
                    is_error: true,
                });
            }
        };

        let needle = path.to_lowercase();
        let mut matches: Vec<(String, String, crate::wiki::Layer)> = Vec::new();
        for entry in &idx.entries {
            if entry.title.to_lowercase().contains(&needle)
                || entry.path.to_lowercase().contains(&needle)
            {
                let layer = wiki
                    .read_page(&entry.path)
                    .map(|page| page.layer)
                    .unwrap_or(crate::wiki::Layer::Kernel);
                matches.push((entry.path.clone(), entry.title.clone(), layer));
            }
        }
        if identity.is_host() {
            matches.sort_by(|a, b| {
                layer_rank(a.2)
                    .cmp(&layer_rank(b.2))
                    .then_with(|| a.0.cmp(&b.0))
            });
        }

        if matches.is_empty() {
            // Fuzzy fallback: rank index entries by Levenshtein distance
            // and surface the closest near-misses — turning a 404 into a
            // debugging hint when nothing clears the threshold.
            let max_dist = crate::wiki::fuzzy::fuzzy_threshold(&needle);
            let ranked =
                crate::wiki::fuzzy::rank_entries_by_levenshtein(&idx.entries, &needle, None, 3);

            let accepted = ranked.first().is_some_and(|(d, _, _)| *d <= max_dist);
            if accepted {
                let mut content = format!("No exact match for '{}'. Closest:\n", path);
                for (d, p, t) in ranked.iter().take_while(|(d, _, _)| *d <= max_dist) {
                    writeln!(content, "  - {} — {} (distance {})", p, t, d)
                        .expect("write to String never fails");
                }
                content.push_str("\nTip: try wiki_search for keyword-based discovery.");
                return Ok(ToolResult {
                    content,
                    is_error: false,
                });
            }

            if !ranked.is_empty() {
                let mut content = format!(
                    "No wiki page found for '{}'. Closest pages \
                     (above similarity threshold {}):\n",
                    path, max_dist
                );
                for (d, p, t) in &ranked {
                    writeln!(content, "  - {} — {} (distance {})", p, t, d)
                        .expect("write to String never fails");
                }
                content.push_str("\nTry wiki_search to discover pages by keyword.");
                return Ok(ToolResult {
                    content,
                    is_error: false,
                });
            }

            return Ok(ToolResult {
                content: format!(
                    "No wiki page found for '{}'. \
                     Try wiki_search to discover pages by keyword.",
                    path
                ),
                is_error: false,
            });
        }

        let mut content = format!("Page '{}' not found. Did you mean one of these?\n", path);
        for (p, t, layer) in &matches {
            if identity.is_host() {
                writeln!(content, "  - [{}] {} — {}", layer.as_str(), p, t)
                    .expect("write to String never fails");
            } else {
                writeln!(content, "  - {} — {}", p, t).expect("write to String never fails");
            }
        }

        Ok(ToolResult {
            content,
            is_error: false,
        })
    }
}

fn layer_rank(layer: crate::wiki::Layer) -> u8 {
    match layer {
        crate::wiki::Layer::Host => 0,
        crate::wiki::Layer::Kernel => 1,
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
        let t = WikiLookupTool;
        assert_eq!(t.name(), "wiki_lookup");
        let p = t.parameters();
        let required = p["required"].as_array().unwrap();
        assert!(required.contains(&json!("path")));
        assert!(p["properties"]["path"].is_object());
    }

    #[test]
    fn is_read_only_true() {
        assert!(WikiLookupTool.is_read_only());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_reads_page_by_path() {
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
            body: "Handles login and session management.".to_string(),
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

        let result = WikiLookupTool
            .call(json!({"path": "entities/auth-service.md"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("AuthService"),
            "should contain title: {}",
            result.content
        );
        assert!(
            result.content.contains("Handles login"),
            "should contain body: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn host_mode_lookup_direct_page_labels_layer() {
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
        let page = crate::wiki::WikiPage {
            title: "HostBudget".to_string(),
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
            body: "Host-domain budget concept.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("concepts/host-budget.md", &page).unwrap();

        let result = WikiLookupTool
            .call(json!({"path": "concepts/host-budget.md"}))
            .await
            .unwrap();

        assert!(!result.is_error, "{}", result.content);
        assert!(
            result.content.starts_with("Layer: host\n\n"),
            "host-mode direct lookup must label the page layer: {}",
            result.content
        );
        assert!(result.content.contains("layer: host"));

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_title_fallback_finds_by_substring() {
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
            body: "Manages user sessions.".to_string(),
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

        // Query by title substring — should fall back and suggest the page.
        let result = WikiLookupTool
            .call(json!({"path": "Session"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Did you mean"),
            "should suggest matches: {}",
            result.content
        );
        assert!(
            result.content.contains("session-manager.md"),
            "should list path: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_missing_page_graceful() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiLookupTool
            .call(json!({"path": "entities/nonexistent.md"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No wiki page found"),
            "should be graceful: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    async fn missing_path_errors() {
        let result = WikiLookupTool.call(json!({})).await;
        assert!(result.is_err(), "missing path should error");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_fuzzy_finds_path_typo() {
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
            body: "Handles login and session management.".to_string(),
            extras: ::std::collections::BTreeMap::new(),
        };
        wiki.write_page("entities/auth_service.md", &page).unwrap();

        let mut idx = wiki.load_index().unwrap();
        idx.entries.push(crate::wiki::IndexEntry {
            path: "entities/auth_service.md".to_string(),
            title: "AuthService".to_string(),
            category: crate::wiki::PageType::Entity,
            one_liner: "auth entity".to_string(),
            last_updated: Some("2026-04-24 00:00:00".to_string()),
            outcome: None,
        });
        fs::write(wiki.root().join("index.md"), idx.to_markdown()).unwrap();

        // Typo: "entites" instead of "entities".
        let result = WikiLookupTool
            .call(json!({"path": "entites/auth_service.md"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Closest:"),
            "should announce fuzzy match: {}",
            result.content
        );
        assert!(
            result.content.contains("auth_service"),
            "should suggest the typo'd target: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_fuzzy_finds_title_typo() {
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

        // Title typo: missing 'v' in "AuthService".
        let result = WikiLookupTool
            .call(json!({"path": "AuthSerice"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("AuthService"),
            "should surface the close title: {}",
            result.content
        );
        assert!(
            result.content.contains("distance"),
            "should annotate distance: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn lookup_fuzzy_below_threshold_returns_not_found() {
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

        let result = WikiLookupTool
            .call(json!({"path": "completely-unrelated-xyzzy"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No wiki page found"),
            "fuzzy should not fire below threshold: {}",
            result.content
        );
        assert!(
            result.content.contains("above similarity threshold"),
            "below-threshold 404 should still list rejected near-misses: {}",
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
    async fn lookup_empty_index_returns_clean_404() {
        let _guard = crate::tools::CWD_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        std::env::set_current_dir(tmp.path()).unwrap();

        // Open the wiki but don't push any IndexEntry rows.
        let _ = crate::wiki::Wiki::open(tmp.path()).unwrap();

        let result = WikiLookupTool
            .call(json!({"path": "entities/anything.md"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("No wiki page found"),
            "empty index should still 404 cleanly: {}",
            result.content
        );
        assert!(
            !result.content.contains("above similarity threshold"),
            "no rejected list when index has nothing to reject: {}",
            result.content
        );

        let _ = std::env::set_current_dir("/");
    }

    #[tokio::test]
    async fn lookup_blank_path_is_clear_error_with_next_step() {
        // No CWD lock or wiki setup needed — blank-path guard returns
        // before any filesystem access.
        let result = WikiLookupTool.call(json!({"path": ""})).await.unwrap();
        assert!(result.is_error, "blank path should be flagged is_error");
        assert!(
            result.content.contains("path is empty"),
            "should explain the failure: {}",
            result.content
        );
        assert!(
            result.content.contains("Try:"),
            "should include next-step hint per directive: {}",
            result.content
        );

        // Whitespace-only path is also blank.
        let result = WikiLookupTool.call(json!({"path": "   "})).await.unwrap();
        assert!(result.is_error, "whitespace-only path should be flagged");
        assert!(
            result.content.contains("path is empty"),
            "whitespace-only path should hit same guard: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn lookup_blank_path_error_includes_canonical_try_hint() {
        // Pins the canonical "Try: ..." phrasing on the blank-path guard so
        // a future refactor can't silently drop the example path or the
        // wiki_search cross-reference, weakening the actionable hint.
        let result = WikiLookupTool.call(json!({"path": ""})).await.unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("Try: provide a wiki-relative path"),
            "missing canonical Try: hint lead: {}",
            result.content
        );
        assert!(
            result.content.contains("entities/"),
            "missing example path format: {}",
            result.content
        );
        assert!(
            result.content.contains("wiki_search"),
            "missing cross-reference to alternate tool: {}",
            result.content
        );
    }
}
