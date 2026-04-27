//! `/git`-family command support.
//!
//! Phase 1.2 module 2 (seeded C23, fattening C24+). Houses inline git
//! dispatch handlers (`handle_*`) extracted from `commands::execute()`,
//! plus helpers they share (e.g. `build_security_review_prompt`). Migrated:
//! `/review` (C24), `/commit` (C25), `/security-review` (C26),
//! `/branch` (C27), `/changelog` (C28), `/pr` (C29), `/blame` (C30),
//! `/conflicts` (C31), `/stash` (C32 — sub-dispatched via private
//! `stash_push`/`stash_pop`/`stash_list`/`stash_drop`/`stash_show` helpers).
//! `/git` family closed.
//!
//! Scope note: `/conflicts` and `/stash` are git-family by behavior but were
//! not in the Prime Directive's illustrative `{commit, pr, review, blame,
//! branch}` list — migrated here anyway since the module principle is
//! "split by command family", not strict directive-listed scope.

use std::fmt::Write as _;

use super::SlashResult;

/// Build the LLM prompt for `/security-review`.
/// `diff` is the raw git diff text; `git_ref` is the reference used (for display only).
pub(super) fn build_security_review_prompt(diff: &str, git_ref: &str) -> String {
    format!(
        "You are a security engineer performing a focused security review of this diff (against `{}`).\n\n\
         Look specifically for:\n\
         - **Injection** — SQL, shell command, path traversal, template injection\n\
         - **Authentication/authorisation** — missing auth checks, privilege escalation, \
           insecure session handling\n\
         - **Secrets** — hardcoded API keys, passwords, tokens, or cryptographic material\n\
         - **Input validation** — missing bounds checks, untrusted data used unsafely\n\
         - **Error handling** — errors silently swallowed, sensitive info in error messages\n\
         - **Cryptography** — weak algorithms, improper nonce/IV reuse, timing attacks\n\
         - **Concurrency** — data races, TOCTOU, deadlocks that could be exploited\n\
         - **Dependencies** — obvious use of deprecated or known-vulnerable APIs\n\n\
         For each finding, report: **severity** (Critical/High/Medium/Low/Info), \
         **location** (file + line if visible), and **recommended remediation**.\n\
         If no security issues are found, say so explicitly.\n\n\
         ```diff\n{}\n```",
        git_ref, diff
    )
}

/// `/review [<ref>]` — stage a review-the-diff prompt against `<ref>` (default `HEAD`).
///
/// Runs `git diff <ref>` with a 10s timeout, truncates to 12k chars via
/// `crate::git::safe_truncate`, and parks the prompt in `app.pending_context`
/// so the next user Enter submits the review request.
pub(super) async fn handle_review(git_ref: String, app: &mut crate::tui::app::App) -> SlashResult {
    let ref_arg = if git_ref.is_empty() {
        "HEAD".to_string()
    } else {
        git_ref.clone()
    };

    let mut cmd = tokio::process::Command::new("git");
    cmd.arg("diff").arg(&ref_arg);

    let output = match tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available or not a git repository ({}). Try: confirm git is installed and the cwd is a git repo.",
                e
            ));
        }
        Err(_) => {
            return SlashResult::Error(
                "git diff timed out. Try: re-run, or check repo health with `git status`.".into(),
            );
        }
    };

    if !output.status.success() && output.stdout.is_empty() {
        return SlashResult::Error(
            "not a git repository or git diff failed. Try: cd into a git repo, or run `git status` to diagnose.".into(),
        );
    }

    let diff = String::from_utf8_lossy(&output.stdout);
    if diff.trim().is_empty() {
        return SlashResult::Info(format!(
            "No changes to review (diff against '{}' is empty).",
            ref_arg
        ));
    }

    let truncated = crate::git::safe_truncate(&diff, 12_000);
    let file_count = truncated
        .lines()
        .filter(|l| l.starts_with("diff --git "))
        .count();

    let review_prompt = format!(
        "Please review this code change. Identify bugs, security issues, logic errors, \
         style violations, and improvement suggestions. Format your response as a concise \
         bulleted list per file. If a change looks good, say so briefly.\n\n\
         ```diff\n{}\n```",
        truncated
    );
    app.pending_context = Some(review_prompt);
    SlashResult::Info(format!(
        "Review context ready ({} file{}) — press Enter (or type a question) to start review",
        file_count,
        if file_count == 1 { "" } else { "s" }
    ))
}

/// `/commit` — stage a generate-commit-message prompt for currently staged changes.
///
/// Runs `git diff --cached` with a 10s timeout, truncates to 8k chars via
/// `crate::git::safe_truncate`, and parks the prompt in `app.pending_context`
/// so the next user Enter asks the model for a commit message.
pub(super) async fn handle_commit(app: &mut crate::tui::app::App) -> SlashResult {
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .args(["diff", "--cached"])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ));
        }
        Err(_) => {
            return SlashResult::Error(
                "git diff --cached timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            );
        }
    };

    if !output.status.success() && output.stdout.is_empty() {
        return SlashResult::Error("Not a git repository or git diff failed".into());
    }

    let diff = String::from_utf8_lossy(&output.stdout);
    if diff.trim().is_empty() {
        return SlashResult::Info("Nothing staged. Run `git add <files>` first.".into());
    }

    let truncated = crate::git::safe_truncate(&diff, 8_000);
    let line_count = truncated.lines().count();

    let commit_prompt = format!(
        "Generate a concise git commit message for these staged changes. \
         Follow conventional commits format (type: description) where appropriate. \
         Respond with ONLY the commit message — no explanation, no quotes, no markdown.\n\n\
         ```diff\n{}\n```",
        truncated
    );
    app.pending_context = Some(commit_prompt);
    SlashResult::Info(format!(
        "Staged diff ready ({} lines) — press Enter to generate commit message",
        line_count
    ))
}

/// `/security-review [<ref>]` — stage a security-audit prompt for the diff
/// against `<ref>` (default `HEAD`).
///
/// Runs `git diff <ref>` with a 10s timeout, truncates to 12k chars, and parks
/// the model prompt (built by `build_security_review_prompt`) in
/// `app.pending_context` for the next Enter to submit.
pub(super) async fn handle_security_review(
    git_ref: String,
    app: &mut crate::tui::app::App,
) -> SlashResult {
    let ref_arg = if git_ref.is_empty() {
        "HEAD".to_string()
    } else {
        git_ref.clone()
    };
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .arg("diff")
            .arg(&ref_arg)
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git diff timed out. Try: re-run, or check repo health with `git status`.".into(),
            )
        }
    };
    let diff = String::from_utf8_lossy(&output.stdout);
    if diff.trim().is_empty() {
        return SlashResult::Info(format!(
            "No changes to review (diff against '{}' is empty).",
            ref_arg
        ));
    }
    let truncated = crate::git::safe_truncate(&diff, 12_000);
    let file_count = truncated
        .lines()
        .filter(|l| l.starts_with("diff --git "))
        .count();
    let prompt = build_security_review_prompt(truncated, &ref_arg);
    app.pending_context = Some(prompt);
    SlashResult::Info(format!(
        "Security review ready ({} file{}) — press Enter to start",
        file_count,
        if file_count == 1 { "" } else { "s" }
    ))
}

/// `/branch [<name>]` — list branches (no arg) or checkout/create `<name>`.
///
/// With no argument: runs `git branch --list --sort=-committerdate` (5s timeout)
/// and returns the formatted list. With `<name>`: tries `git checkout <name>`
/// first, falling back to `git checkout -b <name>` if the branch doesn't exist
/// (each step has its own 10s timeout).
pub(super) async fn handle_branch(target: String) -> SlashResult {
    if target.is_empty() {
        let output = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::process::Command::new("git")
                .args(["branch", "--list", "--sort=-committerdate"])
                .output(),
        )
        .await
        {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return SlashResult::Error(format!(
                    "git not available: {}. Try: confirm git is installed and on $PATH.",
                    e
                ));
            }
            Err(_) => {
                return SlashResult::Error(
                    "git branch timed out. Try: re-run, or check repo health with `git status`."
                        .into(),
                );
            }
        };

        if !output.status.success() {
            return SlashResult::Error("Not a git repository or git branch failed".into());
        }

        let text = String::from_utf8_lossy(&output.stdout);
        if text.trim().is_empty() {
            return SlashResult::Info("No branches found.".into());
        }
        SlashResult::Info(format!("Branches:\n{}", text.trim_end()))
    } else {
        let checkout = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            tokio::process::Command::new("git")
                .args(["checkout", &target])
                .output(),
        )
        .await;

        match checkout {
            Ok(Ok(out)) if out.status.success() => {
                let msg = String::from_utf8_lossy(&out.stderr);
                let display = if msg.trim().is_empty() {
                    format!("Switched to branch '{}'", target)
                } else {
                    msg.trim().to_string()
                };
                SlashResult::Info(display)
            }
            Ok(Ok(_)) => {
                let create = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    tokio::process::Command::new("git")
                        .args(["checkout", "-b", &target])
                        .output(),
                )
                .await;
                match create {
                    Ok(Ok(out)) if out.status.success() => SlashResult::Info(format!(
                        "Created and switched to new branch '{}'",
                        target
                    )),
                    Ok(Ok(out)) => {
                        let err = String::from_utf8_lossy(&out.stderr);
                        SlashResult::Error(format!(
                            "Could not create branch '{}': {}",
                            target,
                            err.trim()
                        ))
                    }
                    Ok(Err(e)) => SlashResult::Error(format!(
                        "git error: {}. Try: confirm you're in a git repository and the branch name is valid.",
                        e
                    )),
                    Err(_) => SlashResult::Error("git checkout timed out. Try: re-run, or check repo health with `git status`.".into()),
                }
            }
            Ok(Err(e)) => SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            )),
            Err(_) => SlashResult::Error(
                "git checkout timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            ),
        }
    }
}

/// `/changelog [<from>] [<to>]` — stage a changelog-generation prompt for the
/// commit range `<from>..<to>`.
///
/// `from` defaults to the most recent tag (falling back to the root commit if
/// no tags exist). `to` defaults to `HEAD`. Runs `git log --format=%h %aI %s`
/// over the range, truncates to 8k chars, and parks the prompt in
/// `app.pending_context` for the next user Enter to submit.
pub(super) async fn handle_changelog(
    from: String,
    to: String,
    app: &mut crate::tui::app::App,
) -> SlashResult {
    let from_ref = if from.is_empty() {
        // Try last tag, fall back to first commit
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::process::Command::new("git")
                .args(["describe", "--tags", "--abbrev=0"])
                .output(),
        )
        .await
        {
            Ok(Ok(out)) if out.status.success() => {
                String::from_utf8_lossy(&out.stdout).trim().to_string()
            }
            _ => {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    tokio::process::Command::new("git")
                        .args(["rev-list", "--max-parents=0", "HEAD"])
                        .output(),
                )
                .await
                {
                    Ok(Ok(out)) if out.status.success() => {
                        String::from_utf8_lossy(&out.stdout).trim().to_string()
                    }
                    _ => {
                        return SlashResult::Error(
                            "Cannot determine starting ref (no tags, no commits)".into(),
                        );
                    }
                }
            }
        }
    } else {
        from
    };

    let to_ref = if to.is_empty() {
        "HEAD".to_string()
    } else {
        to
    };
    let range = format!("{}..{}", from_ref, to_ref);

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .args(["log", "--format=%h %aI %s", &range])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git log timed out. Try: re-run, or check repo health with `git status`.".into(),
            )
        }
    };

    let log_text = String::from_utf8_lossy(&output.stdout);
    if log_text.trim().is_empty() {
        return SlashResult::Info(format!("No changes since {}.", from_ref));
    }

    let commit_count = log_text.lines().count();
    let truncated = crate::git::safe_truncate(&log_text, 8_000);

    let changelog_prompt = format!(
        "Generate a changelog for the following git commits ({range}).\n\
         Format using the Conventional Commits specification, grouping by type \
         (feat, fix, docs, etc.).\n\
         Respond with ONLY the changelog content — no preamble, no explanation.\n\n\
         Commits:\n{truncated}"
    );
    app.pending_context = Some(changelog_prompt);
    SlashResult::Info(format!(
        "Changelog context ready ({} commit{}, {}) — press Enter to generate",
        commit_count,
        if commit_count == 1 { "" } else { "s" },
        range,
    ))
}

/// `/pr [<base>]` — stage a draft-pull-request prompt for the commits/diff
/// ahead of `<base>`.
///
/// `base` defaults to the repo's `origin/HEAD` symbolic-ref, falling back to
/// `main` then `master` when `origin/HEAD` is unset. Runs `git log <base>..HEAD`
/// (10s timeout) + `git diff <base>...HEAD` (15s, 12k char truncation) and
/// parks a TITLE/BODY-structured prompt in `app.pending_context` for the next
/// user Enter to submit.
pub(super) async fn handle_pr(base_arg: String, app: &mut crate::tui::app::App) -> SlashResult {
    // Resolve base branch: use explicit arg, otherwise auto-detect.
    let base = if !base_arg.is_empty() {
        base_arg.clone()
    } else {
        // Try origin/HEAD symbolic-ref, then main, then master.
        let origin_head = tokio::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "origin/HEAD"])
            .output()
            .await
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                let stripped = s.trim_start_matches("origin/").to_string();
                if stripped.is_empty() || stripped == "HEAD" {
                    None
                } else {
                    Some(stripped)
                }
            });

        if let Some(b) = origin_head {
            b
        } else {
            // Fall back to main or master.
            let has_main = tokio::process::Command::new("git")
                .args(["rev-parse", "--verify", "main"])
                .output()
                .await
                .map(|o| o.status.success())
                .unwrap_or(false);
            if has_main {
                "main".to_string()
            } else {
                "master".to_string()
            }
        }
    };

    // Collect commits ahead of base.
    let log_output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .args(["log", &format!("{}..HEAD", base), "--oneline"])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ));
        }
        Err(_) => {
            return SlashResult::Error(
                "git log timed out. Try: re-run, or check repo health with `git status`.".into(),
            );
        }
    };

    let commits = String::from_utf8_lossy(&log_output.stdout)
        .trim()
        .to_string();
    if commits.is_empty() {
        return SlashResult::Info(format!("No commits ahead of '{}'. Nothing to PR.", base));
    }
    let commit_count = commits.lines().count();

    // Collect diff against base.
    let diff_output = match tokio::time::timeout(
        std::time::Duration::from_secs(15),
        tokio::process::Command::new("git")
            .args(["diff", &format!("{}...HEAD", base)])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git diff failed: {}. Try: confirm the base branch exists locally (e.g., main, master).",
                e
            ));
        }
        Err(_) => {
            return SlashResult::Error(
                "git diff timed out. Try: re-run, or check repo health with `git status`.".into(),
            );
        }
    };

    let diff_raw = String::from_utf8_lossy(&diff_output.stdout);
    let diff = crate::git::safe_truncate(&diff_raw, 12_000);

    let prompt = format!(
        "Please draft a GitHub pull request for these changes.\n\n\
         Base branch: {base}\n\
         Commits ({commit_count}):\n\
         {commits}\n\n\
         Diff:\n\
         ```diff\n{diff}\n```\n\n\
         Respond with exactly this format:\n\
         TITLE: <one-line PR title>\n\
         BODY:\n\
         ## Summary\n<what changed and why>\n\n\
         ## Changes\n<bullet list of key changes>\n\n\
         ## Testing\n<how to verify>",
        base = base,
        commit_count = commit_count,
        commits = commits,
        diff = diff,
    );

    app.pending_context = Some(prompt);
    SlashResult::Info(format!(
        "PR context ready ({} commit{} ahead of '{}') — press Enter to draft PR",
        commit_count,
        if commit_count == 1 { "" } else { "s" },
        base
    ))
}

/// `/blame <file> [line]` — stage a git-blame analysis prompt for `<file>`.
///
/// With an optional `line`, narrows the blame to a ±10-line window around it
/// via `-L<start>,<end>`. Runs `git blame` with a 10s timeout, truncates output
/// to 8k chars, and parks the prompt in `app.pending_context` for the next
/// user Enter to submit.
pub(super) async fn handle_blame(
    file: String,
    line: Option<usize>,
    app: &mut crate::tui::app::App,
) -> SlashResult {
    if file.is_empty() {
        return SlashResult::Error("Usage: /blame <file> [line]".into());
    }

    let mut args = vec!["blame".to_string()];
    if let Some(l) = line {
        let start = l.saturating_sub(10);
        let end = l + 10;
        args.push(format!("-L{},{}", start.max(1), end));
    }
    args.push("--".to_string());
    args.push(file.clone());

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git").args(&args).output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git blame timed out. Try: re-run, or check repo health with `git status`.".into(),
            )
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return SlashResult::Error(format!(
            "git blame failed: {}. Try: confirm the file is tracked by git and the path is correct.",
            stderr.trim()
        ));
    }

    let blame_text = String::from_utf8_lossy(&output.stdout);
    if blame_text.trim().is_empty() {
        return SlashResult::Info(format!("No blame output for '{}'.", file));
    }

    let truncated = crate::git::safe_truncate(&blame_text, 8_000);
    let line_count = truncated.lines().count();

    let blame_prompt = format!(
        "Here is the git blame output for `{}`{}. \
         Analyze the authorship and change history. \
         Identify which commits touched which sections and summarize the evolution.\n\n\
         ```\n{}\n```",
        file,
        if let Some(l) = line {
            format!(" around line {}", l)
        } else {
            String::new()
        },
        truncated
    );
    app.pending_context = Some(blame_prompt);
    SlashResult::Info(format!(
        "Blame context ready ({} lines) — type your question to analyze",
        line_count
    ))
}

/// `/conflicts` — stage a merge-conflict-resolution prompt for every file
/// currently in the unmerged-paths state.
///
/// Runs `git diff --name-only --diff-filter=U` (10s timeout) to enumerate
/// conflicted files, then collects each file's diff via per-file `git diff`
/// (5s timeout each, 2k char truncation per file). Partial per-file failures
/// are silently skipped. Parks the aggregated prompt in `app.pending_context`.
pub(super) async fn handle_conflicts(app: &mut crate::tui::app::App) -> SlashResult {
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .args(["diff", "--name-only", "--diff-filter=U"])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git diff timed out. Try: re-run, or check repo health with `git status`.".into(),
            )
        }
    };

    let files_text = String::from_utf8_lossy(&output.stdout);
    let conflict_files: Vec<&str> = files_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();

    if conflict_files.is_empty() {
        return SlashResult::Info("No unresolved merge conflicts detected.".into());
    }

    let mut conflict_details = String::new();
    for cf in &conflict_files {
        if let Ok(Ok(content_out)) = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::process::Command::new("git")
                .args(["diff", "--", cf])
                .output(),
        )
        .await
        {
            let diff = String::from_utf8_lossy(&content_out.stdout);
            let truncated = crate::git::safe_truncate(&diff, 2_000);
            write!(
                conflict_details,
                "### {}\n```diff\n{}\n```\n\n",
                cf, truncated
            )
            .expect("write to String never fails");
        }
    }

    let conflict_prompt = format!(
        "The following {} file{} {} unresolved merge conflicts. \
         Help me resolve them by analyzing both sides and suggesting the best resolution.\n\n{}",
        conflict_files.len(),
        if conflict_files.len() == 1 { "" } else { "s" },
        if conflict_files.len() == 1 {
            "has"
        } else {
            "have"
        },
        conflict_details
    );
    app.pending_context = Some(conflict_prompt);
    SlashResult::Info(format!(
        "{} conflict{} found: {} — press Enter for resolution help",
        conflict_files.len(),
        if conflict_files.len() == 1 { "" } else { "s" },
        conflict_files.join(", ")
    ))
}

/// `/stash` — sub-dispatcher for `git stash` operations.
///
/// Forwards to the matching helper based on the first whitespace-separated
/// token of `sub`. Empty subcommand and `push` both go to `stash_push`;
/// unknown subcommands return a usage hint.
pub(super) async fn handle_stash(sub: String, app: &mut crate::tui::app::App) -> SlashResult {
    let args: Vec<&str> = sub.split_whitespace().collect();
    let subcmd = args.first().copied().unwrap_or("");
    let trailing: &[&str] = if args.is_empty() { &[] } else { &args[1..] };
    match subcmd {
        "" | "push" => stash_push(trailing, subcmd).await,
        "pop" => stash_pop().await,
        "list" => stash_list().await,
        "drop" => stash_drop(trailing).await,
        "show" => stash_show(trailing, app).await,
        other => SlashResult::Error(format!(
            "Unknown stash subcommand '{}'. Use: /stash [push|pop|list|drop|show]",
            other
        )),
    }
}

async fn stash_push(trailing: &[&str], subcmd: &str) -> SlashResult {
    let msg_str = if subcmd == "push" && !trailing.is_empty() {
        trailing.join(" ")
    } else {
        String::new()
    };
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(["stash", "push"]);
    if !msg_str.is_empty() {
        cmd.args(["-m", &msg_str]);
    }
    let output = match tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git stash timed out. Try: re-run, or check repo health with `git status`.".into(),
            )
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return SlashResult::Error(format!(
            "git stash failed: {}. Try: run `git status` to see the working tree state.",
            stderr.trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let msg = text.trim();
    if msg.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let s = stderr.trim();
        if s.is_empty() {
            return SlashResult::Info("No local changes to stash.".into());
        }
        return SlashResult::Info(format!("Stashed: {}", s));
    }
    SlashResult::Info(format!("Stashed: {}", msg))
}

async fn stash_pop() -> SlashResult {
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::process::Command::new("git")
            .args(["stash", "pop"])
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return SlashResult::Error(format!(
                "git not available: {}. Try: confirm git is installed and on $PATH.",
                e
            ))
        }
        Err(_) => {
            return SlashResult::Error(
                "git stash pop timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            )
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return SlashResult::Error(format!(
            "git stash pop failed: {}. Try: /stash list to see saved stashes, or resolve conflicts before popping.",
            stderr.trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    SlashResult::Info(format!("Popped stash:\n{}", text.trim()))
}

async fn stash_list() -> SlashResult {
    let output =
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::process::Command::new("git")
                .args(["stash", "list"])
                .output(),
        )
        .await
        {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return SlashResult::Error(format!(
                    "git not available: {}. Try: confirm git is installed and on $PATH.",
                    e
                ))
            }
            Err(_) => return SlashResult::Error(
                "git stash list timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            ),
        };
    let text = String::from_utf8_lossy(&output.stdout);
    if text.trim().is_empty() {
        return SlashResult::Info("No stashes saved.".into());
    }
    SlashResult::Info(format!("Stash list:\n{}", text.trim()))
}

async fn stash_drop(trailing: &[&str]) -> SlashResult {
    let stash_ref = trailing.first().copied().unwrap_or("");
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(["stash", "drop"]);
    if !stash_ref.is_empty() {
        cmd.arg(stash_ref);
    }
    let output =
        match tokio::time::timeout(std::time::Duration::from_secs(5), cmd.output()).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return SlashResult::Error(format!(
                    "git not available: {}. Try: confirm git is installed and on $PATH.",
                    e
                ))
            }
            Err(_) => return SlashResult::Error(
                "git stash drop timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            ),
        };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return SlashResult::Error(format!(
            "git stash drop failed: {}. Try: /stash list to see saved stashes and pick a valid stash@{{n}}.",
            stderr.trim()
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    SlashResult::Info(format!("Dropped: {}", text.trim()))
}

async fn stash_show(trailing: &[&str], app: &mut crate::tui::app::App) -> SlashResult {
    let stash_ref = trailing.first().copied().unwrap_or("");
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(["stash", "show", "-p"]);
    if !stash_ref.is_empty() {
        cmd.arg(stash_ref);
    }
    let output =
        match tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await {
            Ok(Ok(out)) => out,
            Ok(Err(e)) => {
                return SlashResult::Error(format!(
                    "git not available: {}. Try: confirm git is installed and on $PATH.",
                    e
                ))
            }
            Err(_) => return SlashResult::Error(
                "git stash show timed out. Try: re-run, or check repo health with `git status`."
                    .into(),
            ),
        };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return SlashResult::Error(format!(
            "git stash show failed: {}. Try: /stash list to see saved stashes and pick a valid stash@{{n}}.",
            stderr.trim()
        ));
    }
    let diff = String::from_utf8_lossy(&output.stdout);
    if diff.trim().is_empty() {
        return SlashResult::Info("Stash is empty.".into());
    }
    let truncated = crate::git::safe_truncate(&diff, 8_000);
    app.pending_context = Some(format!(
        "Here is the diff from a git stash. Analyze the changes:\n\n```diff\n{}\n```",
        truncated
    ));
    SlashResult::Info(format!(
        "Stash diff ready ({} lines) — press Enter to analyze",
        truncated.lines().count()
    ))
}
