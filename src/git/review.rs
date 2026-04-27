use super::run_git;
use crate::logging;
use crate::ollama::client::OllamaClient;

pub async fn run_review(client: &OllamaClient, git_ref: &str, verbose: bool) -> anyhow::Result<()> {
    if git_ref != "staged" {
        if let Err(e) = run_git(&["rev-parse", "--verify", git_ref]) {
            anyhow::bail!(
                "Invalid git ref '{}': {}. Use 'staged' for staged changes, or a valid ref like HEAD~1, main, or a commit hash.",
                git_ref, e
            );
        }
    }

    let diff = if git_ref == "staged" {
        run_git(&["diff", "--cached"])?
    } else {
        run_git(&["diff", git_ref])?
    };

    if diff.trim().is_empty() {
        crate::warnings::push_warning(format!(
            "dm review: no changes to review (diff against '{}' is empty).",
            git_ref
        ));
        return Ok(());
    }

    let sections = split_diff_by_file(&diff);
    println!(
        "Reviewing {} file(s) against '{}'...\n",
        sections.len(),
        git_ref
    );

    let mut tasks = Vec::new();
    for (filename, file_diff) in sections {
        if file_diff.contains("Binary files") && file_diff.lines().count() < 5 {
            let fname = filename.clone();
            tasks.push(tokio::spawn(async move {
                (fname, "(binary file, skipped)".to_string())
            }));
            continue;
        }

        if verbose {
            logging::log(&format!("[dm] Reviewing {}...", filename));
        }

        let trunc = if file_diff.len() > 6000 {
            format!(
                "{}\n[... truncated ...]",
                super::safe_truncate(&file_diff, 6000)
            )
        } else {
            file_diff.clone()
        };

        let lang = language_from_filename(&filename).to_string();
        let client = client.clone();
        let fname = filename.clone();
        tasks.push(tokio::spawn(async move {
            let messages = vec![
                serde_json::json!({
                    "role": "system",
                    "content": "You are a senior code reviewer. Be concise and specific. \
                                 Focus on bugs, security issues, and logic errors over style nits. \
                                 Output a bulleted list only — no preamble or sign-off."
                }),
                serde_json::json!({
                    "role": "user",
                    "content": format!(
                        "Review this {} diff for '{}':\n\n```diff\n{}\n```",
                        lang, fname, trunc
                    )
                }),
            ];
            match client.chat(&messages, &[]).await {
                Ok(resp) => (fname, resp.message.content.trim().to_string()),
                Err(e) => (fname, format!("Error: {}", e)),
            }
        }));
    }

    let results = futures_util::future::join_all(tasks).await;
    for result in results {
        let (filename, review) =
            result.unwrap_or_else(|e| ("unknown".to_string(), format!("Task error: {}", e)));
        println!("### {}", filename);
        println!("{}\n", review);
    }

    Ok(())
}

fn extract_filename_from_diff_header(line: &str) -> String {
    if let Some(pos) = line.rfind(" b/") {
        line[pos + 3..].to_string()
    } else {
        line.strip_prefix("diff --git ")
            .and_then(|rest| rest.split_once(' '))
            .map_or_else(
                || "unknown".to_string(),
                |(_, b_part)| b_part.trim_start_matches("b/").to_string(),
            )
    }
}

fn language_from_filename(filename: &str) -> &'static str {
    match filename.rsplit('.').next().unwrap_or("") {
        "rs" => "Rust",
        "py" => "Python",
        "js" => "JavaScript",
        "ts" => "TypeScript",
        "tsx" => "TypeScript (React)",
        "jsx" => "JavaScript (React)",
        "go" => "Go",
        "java" => "Java",
        "rb" => "Ruby",
        "c" | "h" => "C",
        "cpp" | "cc" | "cxx" | "hpp" => "C++",
        "sh" | "bash" => "Shell",
        "yaml" | "yml" => "YAML",
        "toml" => "TOML",
        "json" => "JSON",
        "html" => "HTML",
        "css" => "CSS",
        "sql" => "SQL",
        "md" => "Markdown",
        _ => "code",
    }
}

/// Split a unified diff into `(filename, diff_text)` pairs.
pub fn split_diff_by_file(diff: &str) -> Vec<(String, String)> {
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_file = String::new();
    let mut current_lines: Vec<&str> = Vec::new();

    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            if !current_file.is_empty() {
                sections.push((current_file.clone(), current_lines.join("\n")));
            }
            current_file = extract_filename_from_diff_header(line);
            current_lines = vec![line];
        } else {
            current_lines.push(line);
        }
    }
    if !current_file.is_empty() {
        sections.push((current_file, current_lines.join("\n")));
    }

    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_single_file_diff() {
        let diff = "\
diff --git a/src/foo.rs b/src/foo.rs
index abc..def 100644
--- a/src/foo.rs
+++ b/src/foo.rs
@@ -1 +1 @@
-old
+new
";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0, "src/foo.rs");
        assert!(sections[0].1.contains("+new"));
    }

    #[test]
    fn split_two_file_diff() {
        let diff = "\
diff --git a/a.rs b/a.rs
--- a/a.rs
+++ b/a.rs
@@ -1 +1 @@
-old_a
+new_a
diff --git a/b.rs b/b.rs
--- a/b.rs
+++ b/b.rs
@@ -1 +1 @@
-old_b
+new_b
";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].0, "a.rs");
        assert_eq!(sections[1].0, "b.rs");
        assert!(sections[0].1.contains("old_a"));
        assert!(sections[1].1.contains("old_b"));
    }

    #[test]
    fn split_empty_diff_returns_empty() {
        let sections = split_diff_by_file("");
        assert!(sections.is_empty());
    }

    #[test]
    fn split_diff_with_b_slash_extracts_filename() {
        let diff = "diff --git a/src/main.rs b/src/main.rs\n+added line\n";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0, "src/main.rs");
    }

    #[test]
    fn split_diff_no_git_header_returns_empty() {
        // A diff without any "diff --git" header lines yields no sections
        let diff = "--- a/foo.rs\n+++ b/foo.rs\n@@ -1 +1 @@\n-old\n+new\n";
        let sections = split_diff_by_file(diff);
        assert!(
            sections.is_empty(),
            "no diff --git headers → no sections, got: {:?}",
            sections.iter().map(|(f, _)| f).collect::<Vec<_>>()
        );
    }

    #[test]
    fn split_diff_three_files() {
        let diff = "\
diff --git a/a.rs b/a.rs
+a
diff --git a/b.rs b/b.rs
+b
diff --git a/c.rs b/c.rs
+c
";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 3);
        assert_eq!(sections[0].0, "a.rs");
        assert_eq!(sections[1].0, "b.rs");
        assert_eq!(sections[2].0, "c.rs");
    }

    #[test]
    fn split_diff_each_section_contains_its_own_lines() {
        let diff = "\
diff --git a/x.rs b/x.rs
-removed from x
+added to x
diff --git a/y.rs b/y.rs
-removed from y
+added to y
";
        let sections = split_diff_by_file(diff);
        assert!(sections[0].1.contains("added to x"));
        assert!(!sections[0].1.contains("added to y"));
        assert!(sections[1].1.contains("added to y"));
        assert!(!sections[1].1.contains("added to x"));
    }

    #[test]
    fn split_diff_section_includes_git_header_line() {
        let diff = "diff --git a/foo.rs b/foo.rs\n+added\n";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 1);
        // The "diff --git" header line is the first line of the section content
        assert!(
            sections[0].1.starts_with("diff --git"),
            "section should start with git diff header: {}",
            &sections[0].1[..sections[0].1.len().min(50)]
        );
    }

    #[test]
    fn split_diff_filename_with_path_separator() {
        // Deep nested path like "src/tools/bash.rs"
        let diff = "diff --git a/src/tools/bash.rs b/src/tools/bash.rs\n+x\n";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections[0].0, "src/tools/bash.rs");
    }

    #[test]
    fn extract_filename_standard() {
        let line = "diff --git a/src/main.rs b/src/main.rs";
        assert_eq!(extract_filename_from_diff_header(line), "src/main.rs");
    }

    #[test]
    fn extract_filename_with_spaces() {
        let line = "diff --git a/path with spaces/file.rs b/path with spaces/file.rs";
        assert_eq!(
            extract_filename_from_diff_header(line),
            "path with spaces/file.rs"
        );
    }

    #[test]
    fn extract_filename_rename() {
        let line = "diff --git a/old_name.rs b/new_name.rs";
        assert_eq!(extract_filename_from_diff_header(line), "new_name.rs");
    }

    #[test]
    fn split_diff_binary_file() {
        let diff = "\
diff --git a/image.png b/image.png
Binary files /dev/null and b/image.png differ
";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0, "image.png");
        assert!(sections[0].1.contains("Binary files"));
    }

    #[test]
    fn split_diff_mixed_text_and_binary() {
        let diff = "\
diff --git a/code.rs b/code.rs
+new code
diff --git a/logo.png b/logo.png
Binary files differ
";
        let sections = split_diff_by_file(diff);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].0, "code.rs");
        assert_eq!(sections[1].0, "logo.png");
    }

    #[test]
    fn language_from_filename_rust() {
        assert_eq!(language_from_filename("src/main.rs"), "Rust");
        assert_eq!(language_from_filename("lib.py"), "Python");
        assert_eq!(language_from_filename("app.tsx"), "TypeScript (React)");
    }

    #[test]
    fn language_from_filename_unknown() {
        assert_eq!(language_from_filename("data.xyz"), "code");
        assert_eq!(language_from_filename("noext"), "code");
    }
}
