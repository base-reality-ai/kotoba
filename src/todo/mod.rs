//! Codebase TODO scanning and AI-assisted resolution.
//!
//! Extracts TODO comments from source files and runs an agent loop
//! to autonomously implement their requested changes.

pub mod scanner;

use crate::logging;
use crate::ollama::client::OllamaClient;
use std::path::Path;

/// Format a list of TODO items as a numbered, human-readable string.
pub fn format_todo_list(items: &[scanner::TodoItem]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            format!(
                "{}. [{}] {}:{} \u{2014} {}",
                i + 1,
                item.tag,
                item.file,
                item.line,
                item.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Scan for TODO-style comments and optionally fix them with AI.
pub async fn run_todo(
    globs: &[String],
    fix: bool,
    client: &OllamaClient,
    config_dir: &Path,
) -> anyhow::Result<()> {
    // Expand globs into paths
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let effective_globs: Vec<String> = if globs.is_empty() {
        vec!["src/**/*.rs".to_string()]
    } else {
        globs.to_vec()
    };

    for pattern in &effective_globs {
        match glob::glob(pattern) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if entry.is_file() {
                        paths.push(entry);
                    }
                }
            }
            Err(e) => {
                logging::log_err(&format!(
                    "dm --todo: invalid glob pattern '{}': {}",
                    pattern, e
                ));
            }
        }
    }

    let items = scanner::scan_todos(&paths);

    if items.is_empty() {
        println!("No TODO comments found.");
        return Ok(());
    }

    let list = format_todo_list(&items);
    println!("{}", list);
    println!();

    // Ask model to prioritize
    let prompt = format!(
        "Prioritize these TODO items from highest to lowest urgency. \
        For each, give a one-line rationale.\n\n{}",
        list
    );

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": prompt,
    })];

    let response = client.chat(&messages, &[]).await?;
    let reply = response.message.content;
    println!("{}", reply);

    if fix {
        use crate::permissions;
        use crate::session::Session;
        use crate::tools;
        use std::collections::HashMap;

        for item in &items {
            let fix_prompt = format!(
                "Fix this TODO comment: {} at {}:{} — {}. \
                Apply the fix using file_edit or other tools.",
                item.tag, item.file, item.line, item.text
            );

            let cwd = std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let mut sess = Session::new(cwd, client.model().to_string());
            let session_id = sess.id.clone();

            let registry = tools::registry::default_registry(
                &session_id,
                config_dir,
                client.base_url(),
                client.model(),
                "nomic-embed-text",
            );

            let mut engine = permissions::engine::PermissionEngine::new(true, vec![]);
            let mcp_clients: HashMap<
                String,
                std::sync::Arc<tokio::sync::Mutex<crate::mcp::client::McpClient>>,
            > = HashMap::new();

            let system_prompt = crate::system_prompt::build_system_prompt(&[], None).await;

            crate::conversation::run_conversation(
                &fix_prompt,
                "todo",
                client,
                None,
                &registry,
                &mcp_clients,
                system_prompt,
                &mut engine,
                &mut sess,
                config_dir,
                false,
                "text",
                10,
                false,
                None,
            )
            .await?;

            // Run cargo test to verify
            let test_output = tokio::process::Command::new("cargo")
                .arg("test")
                .output()
                .await?;

            if !test_output.status.success() {
                let stderr = String::from_utf8_lossy(&test_output.stderr);
                logging::log_err(&format!(
                    "dm --todo-fix: tests failed after fixing [{}] at {}:{}\n{}",
                    item.tag, item.file, item.line, stderr
                ));
                return Err(anyhow::anyhow!(
                    "Tests failed after fixing [{}] at {}:{}",
                    item.tag,
                    item.file,
                    item.line
                ));
            }

            println!(
                "Fixed and verified: [{}] {}:{}",
                item.tag, item.file, item.line
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scanner::TodoItem;

    fn item(tag: &str, file: &str, line: usize, text: &str) -> TodoItem {
        TodoItem {
            tag: tag.to_string(),
            file: file.to_string(),
            line,
            text: text.to_string(),
        }
    }

    #[test]
    fn format_empty_list_returns_empty_string() {
        assert_eq!(format_todo_list(&[]), "");
    }

    #[test]
    fn format_single_item() {
        let items = vec![item("TODO", "src/lib.rs", 42, "fix this")];
        let out = format_todo_list(&items);
        assert_eq!(out, "1. [TODO] src/lib.rs:42 \u{2014} fix this");
    }

    #[test]
    fn format_multiple_items_numbered_from_one() {
        let items = vec![
            item("TODO", "a.rs", 1, "first"),
            item("FIXME", "b.rs", 10, "second"),
            item("HACK", "c.rs", 99, "third"),
        ];
        let out = format_todo_list(&items);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("1. "), "first line: {}", lines[0]);
        assert!(lines[1].starts_with("2. "), "second line: {}", lines[1]);
        assert!(lines[2].starts_with("3. "), "third line: {}", lines[2]);
    }

    #[test]
    fn format_includes_tag_file_line_text() {
        let items = vec![item("FIXME", "src/main.rs", 7, "handle error")];
        let out = format_todo_list(&items);
        assert!(out.contains("[FIXME]"), "tag missing: {out}");
        assert!(out.contains("src/main.rs"), "file missing: {out}");
        assert!(out.contains(":7"), "line missing: {out}");
        assert!(out.contains("handle error"), "text missing: {out}");
    }

    #[test]
    fn format_uses_em_dash_separator() {
        let items = vec![item("TODO", "x.rs", 1, "desc")];
        let out = format_todo_list(&items);
        assert!(out.contains('\u{2014}'), "em dash missing: {out}");
    }

    #[test]
    fn format_items_joined_by_newlines() {
        let items = vec![item("TODO", "a.rs", 1, "x"), item("TODO", "b.rs", 2, "y")];
        let out = format_todo_list(&items);
        assert_eq!(out.lines().count(), 2);
        assert!(!out.ends_with('\n'), "should not have trailing newline");
    }

    #[test]
    fn format_items_numbered_beyond_nine() {
        let items: Vec<TodoItem> = (0..12).map(|i| item("TODO", "x.rs", i, "text")).collect();
        let out = format_todo_list(&items);
        assert!(out.contains("10."), "should have item 10");
        assert!(out.contains("12."), "should have item 12");
    }

    #[test]
    fn format_item_line_number_zero() {
        // Line 0 is unusual but should not panic
        let items = vec![item("NOTE", "z.rs", 0, "at top")];
        let out = format_todo_list(&items);
        assert!(out.contains(":0"), "should include line 0: {out}");
    }
}
