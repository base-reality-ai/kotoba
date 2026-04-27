use super::run_git;
use crate::logging;
use crate::ollama::client::OllamaClient;
use std::io::Write as _;

/// Truncate a diff string to at most `max_bytes`, appending a marker if truncated.
pub fn truncate_diff(diff: &str, max_bytes: usize) -> String {
    if diff.len() <= max_bytes {
        diff.to_string()
    } else {
        format!(
            "{}\n\n[... diff truncated at {} chars ...]",
            super::safe_truncate(diff, max_bytes),
            max_bytes
        )
    }
}

/// Build the LLM prompt used to generate a commit message from a (possibly truncated) diff.
pub fn build_commit_prompt(stat: &str, diff_truncated: &str) -> String {
    format!(
        "Generate a concise git commit message for the following staged changes.\n\
         Follow conventional commits format (type: description) where appropriate.\n\
         Respond with ONLY the commit message — no explanation, no quotes, no markdown.\n\n\
         Files changed:\n{}\n\n\
         ```diff\n{}\n```",
        stat, diff_truncated
    )
}

pub async fn run_commit(client: &OllamaClient, verbose: bool) -> anyhow::Result<()> {
    // 1. Check for staged changes
    let staged_stat = run_git(&["diff", "--cached", "--stat"])?;
    if staged_stat.trim().is_empty() {
        logging::log_err("dm commit: nothing staged. Use `git add` first.");
        std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
    }

    // 2. Get the full staged diff (cap at 8000 chars)
    let diff = run_git(&["diff", "--cached"])?;
    let diff_truncated = truncate_diff(&diff, 8000);

    // 3. Ask model to generate a commit message
    if verbose {
        logging::log("[dm] Generating commit message...");
    }
    let prompt = build_commit_prompt(&staged_stat, &diff_truncated);

    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let resp = client.chat(&messages, &[]).await?;
    let message = resp.message.content.trim().to_string();

    // 4. Show message and prompt for confirmation
    println!("\nProposed commit message:\n");
    println!("  {}\n", message);

    print!("Commit with this message? [y/n/e(dit)] ");
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let choice = input.trim().to_lowercase();

    let final_message = match choice.as_str() {
        "y" | "yes" | "" => message,
        "e" | "edit" => {
            let tmp = tempfile_path();
            std::fs::write(&tmp, &message)?;
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            std::process::Command::new(&editor).arg(&tmp).status()?;
            let edited = std::fs::read_to_string(&tmp)?.trim().to_string();
            let _ = std::fs::remove_file(&tmp);
            edited
        }
        _ => {
            println!("Aborted.");
            return Ok(());
        }
    };

    if final_message.is_empty() {
        logging::log_err("dm commit: empty message, aborting.");
        std::process::exit(crate::exit_codes::ExitCode::AgentError.as_i32());
    }

    // 5. Run git commit
    let output = std::process::Command::new("git")
        .args(["commit", "-m", &final_message])
        .output()?;

    std::io::stdout().write_all(&output.stdout)?;
    std::io::stderr().write_all(&output.stderr)?;

    if !output.status.success() {
        std::process::exit(
            output
                .status
                .code()
                .unwrap_or(crate::exit_codes::ExitCode::AgentError.as_i32()),
        );
    }

    Ok(())
}

fn tempfile_path() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut path = std::env::temp_dir();
    path.push(format!("dm-commit-{}-{}.txt", std::process::id(), nanos));
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate_diff ────────────────────────────────────────────────────────

    #[test]
    fn truncate_diff_short_diff_unchanged() {
        let diff = "- old line\n+ new line\n";
        let out = truncate_diff(diff, 8000);
        assert_eq!(out, diff, "short diff should pass through unchanged");
    }

    #[test]
    fn truncate_diff_exact_limit_unchanged() {
        let diff = "a".repeat(8000);
        let out = truncate_diff(&diff, 8000);
        assert_eq!(out.len(), 8000, "diff exactly at limit should be unchanged");
        assert!(!out.contains("truncated"), "no truncation marker expected");
    }

    #[test]
    fn truncate_diff_long_diff_appends_marker() {
        let diff = "b".repeat(9000);
        let out = truncate_diff(&diff, 8000);
        assert!(
            out.contains("[... diff truncated at 8000 chars ...]"),
            "truncation marker missing: {out}"
        );
    }

    #[test]
    fn truncate_diff_truncated_body_is_at_most_max_bytes() {
        let diff = "x".repeat(10_000);
        let out = truncate_diff(&diff, 8000);
        // The body before the marker should be ≤ 8000 bytes.
        let body = out.split("\n\n[... diff truncated").next().unwrap_or("");
        assert!(
            body.len() <= 8000,
            "body before marker is {} bytes, expected ≤ 8000",
            body.len()
        );
    }

    // ── build_commit_prompt ──────────────────────────────────────────────────

    #[test]
    fn build_commit_prompt_contains_diff() {
        let diff = "- removed\n+ added\n";
        let prompt = build_commit_prompt("", diff);
        assert!(prompt.contains(diff), "prompt should embed the diff");
    }

    #[test]
    fn build_commit_prompt_mentions_conventional_commits() {
        let prompt = build_commit_prompt("", "diff content");
        assert!(
            prompt.contains("conventional commits"),
            "prompt should reference conventional commits format"
        );
    }

    #[test]
    fn build_commit_prompt_instructs_no_explanation() {
        let prompt = build_commit_prompt("", "diff");
        assert!(
            prompt.contains("ONLY the commit message"),
            "prompt should say to respond with only the message"
        );
    }

    #[test]
    fn build_commit_prompt_wraps_diff_in_code_fence() {
        let prompt = build_commit_prompt("", "my diff");
        assert!(
            prompt.contains("```diff"),
            "prompt should use diff code fence"
        );
        assert!(prompt.contains("```"), "prompt should close code fence");
    }

    #[test]
    fn build_commit_prompt_includes_stat() {
        let stat = " src/main.rs | 3 ++-\n 1 file changed, 2 insertions(+), 1 deletion(-)";
        let diff = "+new line";
        let prompt = build_commit_prompt(stat, diff);
        assert!(
            prompt.contains("Files changed:"),
            "prompt should have files changed section"
        );
        assert!(
            prompt.contains("src/main.rs"),
            "prompt should include stat output"
        );
        assert!(prompt.contains("+new line"), "prompt should include diff");
    }

    #[test]
    fn truncate_diff_empty_diff_unchanged() {
        let out = truncate_diff("", 8000);
        assert_eq!(out, "", "empty diff should be returned as-is");
    }

    #[test]
    fn tempfile_path_unique() {
        let a = tempfile_path();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = tempfile_path();
        assert_ne!(a, b, "successive calls should produce different paths");
    }

    #[test]
    fn tempfile_path_is_in_temp_dir() {
        let p = tempfile_path();
        let tmp = std::env::temp_dir();
        assert!(
            p.starts_with(&tmp),
            "tempfile should be in temp dir: {:?} not in {:?}",
            p,
            tmp
        );
    }

    #[test]
    fn truncate_diff_multibyte_does_not_panic() {
        // 🦀 is 4 bytes. Put enough to exceed limit and ensure no panic on boundary cut.
        let diff: String = "🦀".repeat(3000); // ~12000 bytes
        let out = truncate_diff(&diff, 8000);
        assert!(
            out.contains("truncated"),
            "should truncate long multibyte diff"
        );
        // Ensure result is valid UTF-8 (i.e., didn't cut mid-codepoint)
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
    }
}
