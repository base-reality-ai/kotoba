//! Git integration and higher-level repository workflows.
//!
//! Provides utilities for generating commits, changelogs, and pull requests,
//! as well as code review functionality.

pub mod changelog;
pub mod commit;
pub mod pr;
pub mod review;

pub(crate) use crate::util::safe_truncate;
pub use changelog::run_changelog;

/// Run a git command and return its stdout, or bail on non-zero exit.
pub fn run_git(args: &[&str]) -> anyhow::Result<String> {
    let output = std::process::Command::new("git").args(args).output()?;
    if !output.status.success() {
        anyhow::bail!(
            "git {}: {}",
            args.first().copied().unwrap_or("(no args)"),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_git_version_succeeds() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let output = run_git(&["--version"]).unwrap();
        assert!(
            output.contains("git version"),
            "should return git version: {}",
            output
        );
    }

    #[test]
    fn run_git_invalid_command_errors() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let result = run_git(&["not-a-real-command"]);
        assert!(result.is_err(), "invalid git command should error");
    }

    #[test]
    fn run_git_error_message_includes_subcommand() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let err = run_git(&["not-a-real-command"]).unwrap_err();
        assert!(
            err.to_string().contains("not-a-real-command"),
            "error should include the subcommand: {}",
            err
        );
    }

    #[test]
    fn run_git_status_works_in_repo() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let output = run_git(&["status", "--porcelain"]);
        assert!(output.is_ok(), "git status should work in repo");
    }

    #[test]
    fn run_git_empty_args_no_panic() {
        let _cwd_guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let result = run_git(&[]);
        // git with no args exits non-zero; we just must not panic
        if let Err(e) = result {
            assert!(e.to_string().contains("(no args)"));
        }
    }
}
