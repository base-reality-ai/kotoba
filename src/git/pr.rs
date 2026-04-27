use super::run_git;
use crate::config::RoutingConfig;
use crate::logging;
use crate::ollama::client::OllamaClient;
use crate::routing;

/// `client` is pre-built from config.model (used as fallback when no routing is configured).
pub async fn run_pr(
    client: &OllamaClient,
    routing: &Option<RoutingConfig>,
    base: Option<&str>,
    push: bool,
    open: bool,
    draft: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    // 1. Detect current branch
    let branch = run_git(&["rev-parse", "--abbrev-ref", "HEAD"])?
        .trim()
        .to_string();
    if branch == "HEAD" {
        anyhow::bail!("dm pr: detached HEAD — check out a branch first.");
    }

    // 2. Detect base branch if not specified
    let base = match base {
        Some(b) if !b.is_empty() => b.to_string(),
        _ => detect_base_branch()?,
    };

    if verbose {
        crate::warnings::push_warning(format!("[pr] branch: {}, base: {}", branch, base));
    }

    // 3. Gather context
    let commits = run_git(&["log", &format!("{}..HEAD", base), "--oneline"])?;
    if commits.trim().is_empty() {
        logging::log(&format!(
            "dm pr: no commits ahead of '{}'. Nothing to PR.",
            base
        ));
        return Ok(());
    }

    let diff = run_git(&["diff", &format!("{}...HEAD", base)])?;
    let diff_truncated = super::commit::truncate_diff(&diff, 12_000);

    // 4a. Diff summary — route to "code" model
    let (code_model, _) = routing::resolve_model("review this diff", routing, client.model());
    let code_client = if code_model != client.model() {
        std::borrow::Cow::Owned(OllamaClient::new(
            client.base_url().to_string(),
            code_model.clone(),
        ))
    } else {
        std::borrow::Cow::Borrowed(client)
    };

    if verbose {
        logging::log(&format!("[dm pr] diff summary model: {}", code_model));
    }

    let summary_prompt = format!(
        "Summarise the key changes in this diff in 3-5 bullet points. Be concise.\n\n\
         Commits:\n{commits}\n\nDiff:\n```diff\n{diff}\n```",
        commits = commits.trim(),
        diff = diff_truncated,
    );
    let summary_msgs = vec![serde_json::json!({"role": "user", "content": summary_prompt})];
    let summary_resp = code_client.chat(&summary_msgs, &[]).await?;
    let diff_summary = summary_resp.message.content.trim().to_string();

    // 4b. PR title + body — route to "default" model
    let (default_model, _) = routing::resolve_model("write documentation", routing, client.model());
    let default_client = if default_model != client.model() {
        std::borrow::Cow::Owned(OllamaClient::new(
            client.base_url().to_string(),
            default_model.clone(),
        ))
    } else {
        std::borrow::Cow::Borrowed(client)
    };

    if verbose {
        logging::log(&format!("[dm pr] PR draft model: {}", default_model));
    }

    let draft_prompt = format!(
        "Generate a GitHub pull request title and body.\n\
         Branch: {branch}\nBase: {base}\n\n\
         Change summary:\n{summary}\n\n\
         Respond with exactly this format (no extra text):\n\
         TITLE: <one-line PR title>\n\
         BODY:\n<markdown body with ## Summary, ## Changes, ## Testing sections>",
        branch = branch,
        base = base,
        summary = diff_summary,
    );
    let draft_msgs = vec![serde_json::json!({"role": "user", "content": draft_prompt})];
    let draft_resp = default_client.chat(&draft_msgs, &[]).await?;
    let raw = draft_resp.message.content.trim().to_string();

    // 5. Parse title and body
    let (title, body) = parse_pr_response(&raw);

    // 6. Print formatted output
    println!("## Pull Request: {}\n", branch);
    println!("**Title:** {}\n", title);
    println!("{}", body);

    // 7. Optional push
    if push {
        logging::log("[dm pr] pushing branch to origin…");
        let status = std::process::Command::new("git")
            .args(["push", "-u", "origin", "HEAD"])
            .status()?;
        if !status.success() {
            anyhow::bail!("dm pr: `git push` failed.");
        }
    }

    // 8. Optional gh pr create
    if open {
        if !which_gh() {
            anyhow::bail!(
                "dm pr --open: `gh` CLI not found. Install it from https://cli.github.com/"
            );
        }
        let mut gh_args = vec![
            "pr".to_string(),
            "create".to_string(),
            "--base".to_string(),
            base.clone(),
            "--title".to_string(),
            title,
            "--body".to_string(),
            body,
        ];
        if draft {
            gh_args.push("--draft".to_string());
        }
        logging::log("[dm pr] running `gh pr create`…");
        let status = std::process::Command::new("gh").args(&gh_args).status()?;
        if !status.success() {
            anyhow::bail!("dm pr: `gh pr create` failed.");
        }
    } else if which_gh() && !push {
        // Hint when gh is available but --open not passed
        println!("\n---");
        println!("Tip: run with --push --open to push and create the PR automatically.");
    }

    Ok(())
}

pub fn detect_base_branch() -> anyhow::Result<String> {
    // Try origin/HEAD symbolic ref first (most reliable)
    if let Ok(out) = run_git(&["rev-parse", "--abbrev-ref", "origin/HEAD"]) {
        let trimmed = out.trim().trim_start_matches("origin/");
        if !trimmed.is_empty() && trimmed != "HEAD" {
            return Ok(trimmed.to_string());
        }
    }
    // Check remote tracking branches (more reliable than local refs)
    for candidate in &["main", "master", "develop"] {
        let remote_ref = format!("origin/{}", candidate);
        if run_git(&["rev-parse", "--verify", &remote_ref]).is_ok() {
            return Ok(candidate.to_string());
        }
    }
    // Last resort: check local branches
    for candidate in &["main", "master"] {
        if run_git(&["rev-parse", "--verify", candidate]).is_ok() {
            return Ok(candidate.to_string());
        }
    }
    anyhow::bail!("Cannot detect base branch. Pass it explicitly: dm --pr main");
}

pub fn parse_pr_response(raw: &str) -> (String, String) {
    let title = raw
        .lines()
        .find(|l| l.starts_with("TITLE:"))
        .map(|l| l.trim_start_matches("TITLE:").trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| "Update".to_string());

    let body = if let Some(idx) = raw.find("BODY:") {
        raw[idx + 5..].trim().to_string()
    } else {
        raw.to_string()
    };

    (title, body)
}

fn which_gh() -> bool {
    std::process::Command::new("gh")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_diff_empty_exits_clean() {
        // parse_pr_response with empty input should not panic; falls back gracefully
        let (title, body) = parse_pr_response("");
        assert_eq!(title, "Update");
        assert_eq!(body, "");
    }

    #[test]
    fn pr_draft_formats_title_and_body() {
        let raw = "TITLE: Add routing support\nBODY:\n## Summary\nAdds model routing.";
        let (title, body) = parse_pr_response(raw);
        assert_eq!(title, "Add routing support");
        assert!(body.contains("## Summary"));
        assert!(body.contains("routing"));
    }

    #[test]
    fn pr_draft_body_fallback_when_no_body_marker() {
        let raw = "Here is the PR description without markers.";
        let (title, body) = parse_pr_response(raw);
        assert_eq!(title, "Update"); // no TITLE: line
        assert_eq!(body, raw);
    }

    #[test]
    fn pr_draft_trims_body_whitespace() {
        let raw = "TITLE: Fix bug\nBODY:\n\n## Summary\nFixed it.\n";
        let (_, body) = parse_pr_response(raw);
        assert!(body.starts_with("## Summary"));
    }

    #[test]
    fn parse_pr_response_title_with_leading_spaces() {
        let raw = "TITLE:   Spaced title   \nBODY:\nBody here.";
        let (title, _) = parse_pr_response(raw);
        assert_eq!(title, "Spaced title");
    }

    #[test]
    fn parse_pr_response_empty_title_line_falls_back() {
        // TITLE: with nothing after it → should fall back to "Update"
        let raw = "TITLE:\nBODY:\nSome body.";
        let (title, _) = parse_pr_response(raw);
        assert_eq!(title, "Update");
    }

    #[test]
    fn parse_pr_response_body_content_after_body_marker() {
        let raw = "TITLE: My feature\nBODY:\n## Testing\nRan the tests.";
        let (_, body) = parse_pr_response(raw);
        assert!(
            body.contains("## Testing"),
            "body should have Testing section: {body}"
        );
        assert!(
            body.contains("Ran the tests."),
            "body should have testing text: {body}"
        );
    }

    #[test]
    fn parse_pr_response_title_only_no_body_marker() {
        // Only TITLE: present — body falls back to entire raw string
        let raw = "TITLE: Just a title";
        let (title, body) = parse_pr_response(raw);
        assert_eq!(title, "Just a title");
        assert_eq!(body, raw);
    }

    #[test]
    fn detect_base_branch_returns_something() {
        // In a real git repo this should succeed
        if let Ok(base) = detect_base_branch() {
            assert!(!base.is_empty(), "base branch should not be empty");
            assert!(
                !base.contains("origin/"),
                "should not contain origin/ prefix"
            );
        }
    }

    #[test]
    fn pr_diff_truncation_uses_shared_logic() {
        let long_diff = "x".repeat(15_000);
        let truncated = crate::git::commit::truncate_diff(&long_diff, 12_000);
        assert!(truncated.contains("[... diff truncated at 12000 chars ...]"));
    }

    #[test]
    fn parse_pr_response_multiple_title_lines_takes_first() {
        let raw = "TITLE: First title\nTITLE: Second title\nBODY:\nBody text.";
        let (title, _) = parse_pr_response(raw);
        assert_eq!(title, "First title");
    }
}
