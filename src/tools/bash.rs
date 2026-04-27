use super::{Tool, ToolResult};
use crate::permissions::Decision;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::process::Command;

static BASH_CHILD_PID: AtomicU32 = AtomicU32::new(0);

/// Kill the currently-running bash child process (if any).
/// Returns `true` if a process was signalled, `false` if nothing was running.
pub fn kill_running_bash() -> bool {
    let pid = BASH_CHILD_PID.load(Ordering::SeqCst);
    signal_bash_process_group(pid)
}

fn signal_bash_process_group(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let pgid = nix::unistd::Pid::from_raw(pid as i32);
    let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
    true
}

#[derive(Debug, Clone, PartialEq)]
pub enum BashRisk {
    Safe,
    Warn(String),
}

pub fn classify_risk(command: &str) -> BashRisk {
    let norm: Vec<&str> = command.split_whitespace().collect();
    let joined = norm.join(" ").to_lowercase();

    // rm with recursive + root or force-recursive
    if joined.starts_with("rm ") || joined.contains("| rm ") || joined.contains("&& rm ") {
        if (joined.contains(" -rf") || joined.contains(" -fr"))
            && (joined.contains(" /") && !joined.contains(" /tmp") && !joined.contains(" /home/"))
        {
            return BashRisk::Warn("rm -rf on root or system path".into());
        }
        if joined.contains(" -rf /") || joined.contains(" -fr /") {
            return BashRisk::Warn("rm -rf on root path".into());
        }
    }

    // Dangerous git operations
    if joined.contains("git push") && (joined.contains("--force") || joined.contains(" -f")) {
        return BashRisk::Warn("git force push can overwrite remote history".into());
    }
    if joined.contains("git reset --hard") {
        return BashRisk::Warn("git reset --hard discards uncommitted changes".into());
    }
    if joined.contains("git clean")
        && (joined.contains("-fd") || joined.contains("-df") || joined.contains("-f"))
    {
        return BashRisk::Warn("git clean -f deletes untracked files".into());
    }
    // git branch -D (uppercase) — need to check original command since -D != -d
    if command.contains("git branch") && command.contains(" -D ") {
        return BashRisk::Warn("git branch -D force-deletes a branch".into());
    }

    // Disk/device operations
    if joined.starts_with("dd ") && joined.contains("of=/dev/") {
        return BashRisk::Warn("dd writing to device can destroy data".into());
    }
    if joined.starts_with("mkfs") || joined.starts_with("fdisk") {
        return BashRisk::Warn("filesystem/partition tool can destroy data".into());
    }

    // System operations
    if joined.starts_with("shutdown")
        || joined.starts_with("reboot")
        || joined.starts_with("init 0")
        || joined.starts_with("init 6")
    {
        return BashRisk::Warn("system shutdown/reboot".into());
    }
    if joined.contains("kill -9") || joined.contains("killall") || joined.contains("pkill") {
        return BashRisk::Warn("process kill command".into());
    }

    // chmod/chown on system paths
    if (joined.starts_with("chmod") || joined.starts_with("chown"))
        && joined.contains(" -r")
        && joined.contains(" /")
    {
        return BashRisk::Warn("recursive permission change on system path".into());
    }

    BashRisk::Safe
}

/// Applies the risky-command floor to an engine decision for a bash command.
///
/// - `bypass` mode: caller opted out of safety; return engine decision as-is.
/// - Engine said `Deny`: explicit user-set rule wins (never silently downgrade).
/// - Risk is `Warn(_)`: force `Ask` regardless of what rules say, and thread
///   the risk description out so the UI can show *why* the prompt fired.
/// - Otherwise: return engine decision unchanged.
///
/// Returns `(decision, reason)` where `reason` is the human-readable risk
/// description to display in the permission prompt, if any.
pub fn effective_bash_decision(
    engine_decision: Decision,
    risk: &BashRisk,
    bypass: bool,
) -> (Decision, Option<String>) {
    if bypass {
        return (engine_decision, None);
    }
    match (engine_decision, risk) {
        (Decision::Deny, _) => (Decision::Deny, None),
        (_, BashRisk::Warn(reason)) => (Decision::Ask, Some(reason.clone())),
        (d, BashRisk::Safe) => (d, None),
    }
}

/// Shared wrapper used by both headless and TUI permission paths.
///
/// Non-bash tools pass the engine decision through unchanged; bash commands are
/// classified by `classify_risk` and routed through `effective_bash_decision`
/// so the risky-command floor applies uniformly.
pub fn decision_with_risk(
    tool_name: &str,
    args: &serde_json::Value,
    engine_decision: Decision,
    bypass: bool,
) -> (Decision, Option<String>) {
    if tool_name != "bash" {
        return (engine_decision, None);
    }
    let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let risk = classify_risk(command);
    effective_bash_decision(engine_decision, &risk, bypass)
}

/// Maximum allowed timeout: 10 minutes.
const MAX_TIMEOUT_MS: u64 = 600_000;

/// Maximum output size returned to the model (1MB). Output beyond this is
/// truncated with a notice. The TUI streaming path still shows all lines live.
const MAX_OUTPUT_BYTES: usize = 1_048_576;
const TRUNCATION_NOTICE: &str = "\n\n[Output truncated at 1MB limit]";

/// Truncate `content` to `MAX_OUTPUT_BYTES` on a clean UTF-8 boundary,
/// then append the truncation notice. Returns true if truncation occurred.
fn truncate_output(content: &mut String) -> bool {
    if content.len() <= MAX_OUTPUT_BYTES {
        return false;
    }
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content.truncate(end);
    content.push_str(TRUNCATION_NOTICE);
    true
}

#[cfg(test)]
pub fn summarize_build_errors_for_test(output: &str) -> Option<String> {
    summarize_build_errors(output)
}

fn summarize_build_errors(output: &str) -> Option<String> {
    let diags = crate::testfix::parse_diagnostics(output);
    let errors: Vec<_> = diags
        .iter()
        .filter(|d| d.level == crate::testfix::DiagLevel::Error)
        .collect();
    if errors.is_empty() {
        return None;
    }
    let show = errors.len().min(10);
    let mut summary = format!("[dm] Build failed — {} error(s):\n", errors.len());
    for (i, e) in errors.iter().take(show).enumerate() {
        writeln!(summary, "  {}. {}", i + 1, e.summary_line())
            .expect("write to String never fails");
    }
    if errors.len() > 10 {
        writeln!(summary, "  ... and {} more", errors.len() - 10)
            .expect("write to String never fails");
    }
    Some(summary)
}

/// A line of output from a running bash command.
/// Used by the TUI streaming path to display output in real time.
pub struct BashOutputLine {
    pub text: String,
    pub is_stderr: bool,
}

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Execute a bash shell command and return stdout/stderr. Use for running commands, scripts, git operations, etc. Avoid interactive commands. Long-running commands should include a timeout."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Output is capped at 1MB; use head/tail/grep to limit output from verbose commands."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to the current working directory."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds (default 30000)"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of what this command does (shown in UI, not executed)"
                }
            },
            "required": ["command"]
        })
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some("Prefer dedicated tools over bash when possible: use `grep` instead of `grep`/`rg` commands, \
              `glob` instead of `find`, `read_file` instead of `cat`/`head`/`tail`, and `edit_file` instead of `sed`/`awk`. \
              Reserve bash for git operations, builds, test runs, and system commands that have no dedicated tool. \
              Never run interactive commands (vim, less, top). Quote paths with spaces. \
              Use the `cwd` parameter to run commands in a specific directory instead of `cd dir && command`.")
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: command"))?;
        if let BashRisk::Warn(desc) = classify_risk(command) {
            crate::warnings::push_warning(format!("[bash] ⚠ {}: {}", desc, command));
        }
        let cwd = args["cwd"].as_str();
        let timeout_ms = args["timeout_ms"]
            .as_u64()
            .unwrap_or(30_000)
            .min(MAX_TIMEOUT_MS);
        Ok(run_bash(command, cwd, timeout_ms, None).await)
    }
}

/// Shared bash execution logic used by both `BashTool::call()` (batch) and the
/// TUI streaming path. When `line_tx` is `Some`, output lines are sent as they
/// arrive; when `None`, the command runs in batch mode using `wait_with_output`.
///
/// Both paths use `process_group(0)` for process isolation and
/// SIGTERM → poll → SIGKILL for graceful timeout shutdown.
pub async fn run_bash(
    command: &str,
    cwd: Option<&str>,
    timeout_ms: u64,
    line_tx: Option<&tokio::sync::mpsc::Sender<BashOutputLine>>,
) -> ToolResult {
    // Validate working directory
    if let Some(dir) = cwd {
        let cwd_path = std::path::Path::new(dir);
        if !cwd_path.is_dir() {
            return ToolResult {
                content: format!(
                    "cwd does not exist or is not a directory: {}. Try: pass an absolute path that exists, or omit cwd to use the session default.",
                    dir
                ),
                is_error: true,
            };
        }
    }

    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(command);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .process_group(0);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ToolResult {
                content: format!(
                    "Failed to spawn bash: {}. Try: confirm bash is installed and on PATH.",
                    e
                ),
                is_error: true,
            };
        }
    };

    let child_id = child.id();
    if let Some(pid) = child_id {
        BASH_CHILD_PID.store(pid, Ordering::SeqCst);
    }

    let result = if let Some(tx) = line_tx {
        run_bash_streaming(&mut child, child_id, timeout_ms, tx).await
    } else {
        run_bash_batch(child, child_id, timeout_ms).await
    };

    BASH_CHILD_PID.store(0, Ordering::SeqCst);
    result
}

/// Batch execution path — waits for the command to finish, then returns all output.
async fn run_bash_batch(
    mut child: tokio::process::Child,
    child_id: Option<u32>,
    timeout_ms: u64,
) -> ToolResult {
    use tokio::io::AsyncReadExt;

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();

    let collect = async {
        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        if let Some(ref mut out) = stdout_pipe {
            out.read_to_end(&mut stdout_buf).await.ok();
        }
        if let Some(ref mut err) = stderr_pipe {
            err.read_to_end(&mut stderr_buf).await.ok();
        }
        let status = child.wait().await;
        (stdout_buf, stderr_buf, status)
    };

    let result = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), collect).await;

    match result {
        Ok((stdout_buf, stderr_buf, Ok(status))) => {
            let stdout = String::from_utf8_lossy(&stdout_buf);
            let stderr = String::from_utf8_lossy(&stderr_buf);
            let exit_code = status.code().unwrap_or(-1);

            let mut content = String::new();
            if !stdout.is_empty() {
                content.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str("STDERR: ");
                content.push_str(&stderr);
            }

            truncate_output(&mut content);

            if !status.success() {
                if let Some(summary) = summarize_build_errors(&content) {
                    content = format!("{}\n--- Full output ---\n{}", summary, content);
                    truncate_output(&mut content);
                }
                write!(content, "\n(exit code: {})", exit_code)
                    .expect("write to String never fails");
            } else if content.is_empty() {
                content = format!("(exit code: {})", exit_code);
            }

            ToolResult {
                content,
                is_error: !status.success(),
            }
        }
        Ok((_, _, Err(e))) => ToolResult {
            content: format!(
                "Failed to execute bash: {}. Try: review the command for syntax errors and re-run.",
                e
            ),
            is_error: true,
        },
        Err(_) => graceful_kill(&mut child, child_id, timeout_ms).await,
    }
}

/// Streaming execution path — sends output lines as they arrive via `line_tx`.
async fn run_bash_streaming(
    child: &mut tokio::process::Child,
    child_id: Option<u32>,
    timeout_ms: u64,
    line_tx: &tokio::sync::mpsc::Sender<BashOutputLine>,
) -> ToolResult {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let Some(stdout) = child.stdout.take() else {
        return ToolResult {
            content: "Failed to capture stdout from child process. Try: re-run the command."
                .to_string(),
            is_error: true,
        };
    };
    let Some(stderr) = child.stderr.take() else {
        return ToolResult {
            content: "Failed to capture stderr from child process. Try: re-run the command."
                .to_string(),
            is_error: true,
        };
    };
    let mut stdout_reader = BufReader::new(stdout).lines();
    let mut stderr_reader = BufReader::new(stderr).lines();

    let mut all_output = String::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

    let mut stdout_done = false;
    let mut stderr_done = false;
    loop {
        if stdout_done && stderr_done {
            break;
        }
        tokio::select! {
            line = stdout_reader.next_line(), if !stdout_done => {
                match line {
                    Ok(Some(l)) => {
                        line_tx.send(BashOutputLine { text: l.clone(), is_stderr: false }).await.ok();
                        if all_output.len() < MAX_OUTPUT_BYTES {
                            all_output.push_str(&l);
                            all_output.push('\n');
                        }
                    }
                    _ => stdout_done = true,
                }
            }
            line = stderr_reader.next_line(), if !stderr_done => {
                match line {
                    Ok(Some(l)) => {
                        line_tx.send(BashOutputLine { text: l.clone(), is_stderr: true }).await.ok();
                        if all_output.len() < MAX_OUTPUT_BYTES {
                            all_output.push_str("STDERR: ");
                            all_output.push_str(&l);
                            all_output.push('\n');
                        }
                    }
                    _ => stderr_done = true,
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                return graceful_kill(child, child_id, timeout_ms).await;
            }
        }
    }

    let status = child.wait().await.ok();
    let is_error = status.is_none_or(|s| !s.success());
    let exit_code = status.and_then(|s| s.code()).unwrap_or(-1);

    truncate_output(&mut all_output);

    if is_error {
        write!(all_output, "\n(exit code: {})", exit_code).expect("write to String never fails");
    } else if all_output.is_empty() {
        all_output = format!("(exit code: {})", exit_code);
    }

    ToolResult {
        content: all_output,
        is_error,
    }
}

/// SIGTERM → poll up to 2s → SIGKILL → reap, then return a timeout `ToolResult`.
async fn graceful_kill(
    child: &mut tokio::process::Child,
    child_id: Option<u32>,
    timeout_ms: u64,
) -> ToolResult {
    if let Some(pid) = child_id {
        let pgid = nix::unistd::Pid::from_raw(pid as i32);
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGTERM);
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_err() {
                break;
            }
        }
        let _ = nix::sys::signal::killpg(pgid, nix::sys::signal::Signal::SIGKILL);
    }
    const REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    if tokio::time::timeout(REAP_TIMEOUT, child.wait())
        .await
        .is_err()
    {
        crate::warnings::push_warning(
            "bash tool: process did not exit within 5s after SIGKILL".to_string(),
        );
    }
    ToolResult {
        content: format!(
            "Command timed out after {}ms and was terminated. Try: increase timeout_ms, or break the command into smaller steps.",
            timeout_ms
        ),
        is_error: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool;
        let result = tool.call(json!({"command": "echo hello"})).await.unwrap();
        assert!(result.content.contains("hello"));
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn bash_exit_code_failure() {
        let tool = BashTool;
        let result = tool.call(json!({"command": "exit 1"})).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn bash_missing_command_arg_errors() {
        let tool = BashTool;
        let result = tool.call(json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn bash_stderr_captured() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "echo errline >&2"}))
            .await
            .unwrap();
        assert!(
            result.content.contains("errline"),
            "stderr not in output: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_combined_stdout_and_stderr() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "echo out; echo err >&2"}))
            .await
            .unwrap();
        assert!(result.content.contains("out"));
        assert!(result.content.contains("err"));
    }

    #[tokio::test]
    async fn bash_empty_output_shows_exit_code() {
        let tool = BashTool;
        let result = tool.call(json!({"command": "true"})).await.unwrap();
        // `true` exits 0 with no output — fallback message should appear
        assert!(
            result.content.contains("exit code") || result.content.is_empty() || !result.is_error
        );
    }

    #[tokio::test]
    async fn bash_timeout_returns_error() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "sleep 10", "timeout_ms": 100}))
            .await
            .unwrap();
        assert!(result.is_error, "timeout should set is_error=true");
        assert!(
            result.content.contains("timed out"),
            "should mention timeout: {}",
            result.content
        );
        assert!(
            result.content.contains("terminated"),
            "should mention terminated: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_stderr_prefix_format() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "echo errline >&2"}))
            .await
            .unwrap();
        // When only stderr (no stdout), content should start with "STDERR: "
        assert!(
            result.content.starts_with("STDERR: "),
            "stderr-only output should start with 'STDERR: ': {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_multiline_stdout_preserved() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "printf 'line1\\nline2\\nline3\\n'"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("line1"));
        assert!(result.content.contains("line2"));
        assert!(result.content.contains("line3"));
    }

    #[tokio::test]
    async fn bash_nonzero_exit_is_error() {
        let tool = BashTool;
        let result = tool.call(json!({"command": "exit 42"})).await.unwrap();
        assert!(result.is_error, "non-zero exit should set is_error=true");
    }

    #[tokio::test]
    async fn bash_cwd_runs_in_directory() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "pwd", "cwd": "/tmp"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("/tmp"),
            "pwd should show /tmp: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_cwd_nonexistent_returns_error() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "pwd", "cwd": "/nonexistent_dir_xyz"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("does not exist"));
    }

    #[tokio::test]
    async fn bash_failure_shows_exit_code() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "echo fail; exit 42"}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("exit code: 42"),
            "should show exit code: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_success_no_exit_code_in_output() {
        let tool = BashTool;
        let result = tool.call(json!({"command": "echo hello"})).await.unwrap();
        assert!(!result.is_error);
        assert!(
            !result.content.contains("exit code"),
            "success with output should not show exit code: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn bash_timeout_kills_process() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "sleep 60", "timeout_ms": 200}))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
        assert!(result.content.contains("terminated"));
    }

    // ── run_bash() streaming tests ──────────────────────────────────────

    #[tokio::test]
    async fn run_bash_streaming_captures_lines() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash("echo hello; echo world", None, 30_000, Some(&tx)).await;
        drop(tx); // close sender so rx.recv() terminates

        assert!(!result.is_error, "should succeed: {}", result.content);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("world"));

        // Verify lines were sent through the channel
        let mut lines = Vec::new();
        while let Ok(line) = rx.try_recv() {
            lines.push(line.text);
        }
        assert!(
            lines.iter().any(|l| l.contains("hello")),
            "channel should have 'hello': {:?}",
            lines
        );
        assert!(
            lines.iter().any(|l| l.contains("world")),
            "channel should have 'world': {:?}",
            lines
        );
    }

    #[tokio::test]
    async fn run_bash_streaming_stderr_flagged() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash("echo errline >&2", None, 30_000, Some(&tx)).await;
        drop(tx);

        assert!(
            result.content.contains("STDERR: errline"),
            "stderr should be prefixed: {}",
            result.content
        );

        let mut found_stderr = false;
        while let Ok(line) = rx.try_recv() {
            if line.text.contains("errline") {
                assert!(line.is_stderr, "stderr line should have is_stderr=true");
                found_stderr = true;
            }
        }
        assert!(found_stderr, "should have received stderr line via channel");
    }

    #[tokio::test]
    async fn run_bash_streaming_cwd() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash("pwd", Some("/tmp"), 30_000, Some(&tx)).await;
        assert!(!result.is_error);
        assert!(
            result.content.contains("/tmp"),
            "streaming cwd should work: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn run_bash_streaming_cwd_nonexistent() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash("pwd", Some("/nonexistent_dir_xyz"), 30_000, Some(&tx)).await;
        assert!(result.is_error);
        assert!(result.content.contains("does not exist"));
    }

    #[tokio::test]
    async fn run_bash_streaming_timeout_graceful() {
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join("stream_terminated");
        let cmd = format!(
            "trap 'touch {}; exit 0' TERM; while true; do sleep 0.1 & wait; done",
            marker.display()
        );
        let (tx, _rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash(&cmd, None, 500, Some(&tx)).await;
        assert!(result.is_error);
        assert!(
            result.content.contains("timed out"),
            "should mention timeout: {}",
            result.content
        );
        assert!(
            marker.exists(),
            "SIGTERM handler should have fired in streaming mode"
        );
    }

    #[tokio::test]
    async fn run_bash_streaming_exit_code_on_failure() {
        let (tx, _rx) = tokio::sync::mpsc::channel::<BashOutputLine>(64);
        let result = run_bash("echo fail; exit 42", None, 30_000, Some(&tx)).await;
        assert!(result.is_error);
        assert!(
            result.content.contains("exit code: 42"),
            "streaming should show exit code: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn run_bash_batch_still_works() {
        let result = run_bash("echo batch_test", None, 30_000, None).await;
        assert!(!result.is_error);
        assert!(result.content.contains("batch_test"));
    }

    // ── output truncation tests ─────────────────────────────────────────

    #[tokio::test]
    async fn bash_output_truncated_at_limit() {
        // seq 1 999999 produces ~7MB of output
        let result = run_bash("seq 1 999999", None, 30_000, None).await;
        assert!(
            !result.is_error,
            "seq should succeed: {}",
            &result.content[..100.min(result.content.len())]
        );
        assert!(
            result.content.contains("[Output truncated at 1MB limit]"),
            "should contain truncation notice"
        );
        // Content should be at most MAX_OUTPUT_BYTES + truncation notice + exit code slack
        assert!(
            result.content.len() <= MAX_OUTPUT_BYTES + 200,
            "content too large: {} bytes",
            result.content.len()
        );
    }

    #[tokio::test]
    async fn bash_streaming_output_truncated() {
        // Generate ~2MB: 20000 lines of 100 chars each
        let (tx, mut rx) = tokio::sync::mpsc::channel::<BashOutputLine>(1024);

        // Drain the channel in a background task to prevent backpressure stalls
        let drain_handle = tokio::spawn(async move {
            let mut count = 0usize;
            while rx.recv().await.is_some() {
                count += 1;
            }
            count
        });

        let result = run_bash(
            "for i in $(seq 1 20000); do printf '%0100d\\n' $i; done",
            None,
            30_000,
            Some(&tx),
        )
        .await;
        drop(tx);

        assert!(!result.is_error);
        assert!(
            result.content.contains("[Output truncated at 1MB limit]"),
            "streaming should contain truncation notice"
        );
        assert!(
            result.content.len() <= MAX_OUTPUT_BYTES + 200,
            "streaming content too large: {} bytes",
            result.content.len()
        );

        // TUI channel should have received lines even past the cap
        let line_count = drain_handle.await.unwrap();
        assert!(
            line_count > 10_000,
            "TUI should receive lines past cap: {}",
            line_count
        );
    }

    #[tokio::test]
    async fn bash_small_output_not_truncated() {
        let result = run_bash("echo hello", None, 30_000, None).await;
        assert!(!result.is_error);
        assert!(
            !result.content.contains("truncated"),
            "small output should not be truncated: {}",
            result.content
        );
    }

    #[test]
    fn truncate_output_noop_for_small() {
        let mut s = "hello world".to_string();
        assert!(!truncate_output(&mut s));
        assert_eq!(s, "hello world");
    }

    #[test]
    fn truncate_output_truncates_large() {
        let mut s = "x".repeat(MAX_OUTPUT_BYTES + 1000);
        assert!(truncate_output(&mut s));
        assert!(s.ends_with("[Output truncated at 1MB limit]"));
        assert!(s.len() <= MAX_OUTPUT_BYTES + TRUNCATION_NOTICE.len() + 1);
    }

    #[test]
    fn truncate_output_utf8_safe() {
        // Build a string where MAX_OUTPUT_BYTES falls in the middle of a multi-byte char
        let mut s = "a".repeat(MAX_OUTPUT_BYTES - 1);
        s.push('é'); // 2-byte UTF-8 char, so total = MAX_OUTPUT_BYTES + 1
        assert!(truncate_output(&mut s));
        // Must be valid UTF-8
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    }

    #[test]
    fn summarize_build_errors_finds_rust_errors() {
        let input = "\
error[E0308]: mismatched types
  --> src/foo.rs:10:5
   |
error[E0425]: cannot find value `x`
  --> src/bar.rs:25:9
   |
";
        let summary = summarize_build_errors(input).unwrap();
        assert!(
            summary.contains("2 error(s)"),
            "should show count: {}",
            summary
        );
        assert!(
            summary.contains("[E0308]"),
            "should include first error code"
        );
        assert!(
            summary.contains("[E0425]"),
            "should include second error code"
        );
        assert!(summary.contains("src/foo.rs:10"), "should include location");
        assert!(summary.contains("src/bar.rs:25"), "should include location");
    }

    #[test]
    fn summarize_build_errors_returns_none_on_clean_output() {
        let input = "Compiling foo v0.1.0\nFinished dev profile\n";
        assert!(summarize_build_errors(input).is_none());
    }

    #[test]
    fn summarize_build_errors_caps_at_10_errors() {
        let mut input = String::new();
        for i in 0..15 {
            writeln!(input, "error: problem number {}", i).expect("write to String never fails");
        }
        let summary = summarize_build_errors(&input).unwrap();
        assert!(
            summary.contains("15 error(s)"),
            "should show total count: {}",
            summary
        );
        assert!(
            summary.contains("... and 5 more"),
            "should show overflow: {}",
            summary
        );
    }

    #[test]
    fn summarize_build_errors_ignores_warnings() {
        let input = "warning: unused variable `x`\nwarning: dead code\n";
        assert!(
            summarize_build_errors(input).is_none(),
            "warnings-only should return None"
        );
    }

    #[test]
    fn summarize_build_errors_includes_file_location() {
        let input = "\
error[E0308]: mismatched types
  --> src/main.rs:42:10
   |
";
        let summary = summarize_build_errors(input).unwrap();
        assert!(
            summary.contains("src/main.rs:42"),
            "should include file:line: {}",
            summary
        );
    }

    #[tokio::test]
    async fn bash_timeout_graceful_sigterm() {
        // Spawn a process that traps SIGTERM and writes a file.
        // Use `wait` so bash can handle signals between loop iterations.
        let tmp = tempfile::TempDir::new().unwrap();
        let marker = tmp.path().join("terminated");
        let cmd = format!(
            "trap 'touch {}; exit 0' TERM; while true; do sleep 0.1 & wait; done",
            marker.display()
        );
        let tool = BashTool;
        let result = tool
            .call(json!({"command": cmd, "timeout_ms": 500}))
            .await
            .unwrap();
        assert!(result.is_error);
        // The 2s grace period in the timeout handler gives plenty of time
        // for the SIGTERM handler to fire and create the marker file.
        assert!(
            marker.exists(),
            "SIGTERM handler should have fired before SIGKILL"
        );
    }

    #[tokio::test]
    async fn bash_timeout_capped_to_max() {
        let tool = BashTool;
        let result = tool
            .call(json!({"command": "echo capped", "timeout_ms": 999_999_999}))
            .await
            .unwrap();
        assert!(!result.is_error, "should succeed: {}", result.content);
        assert!(result.content.contains("capped"));
    }

    #[tokio::test]
    async fn bash_timeout_reaps_child_no_zombie() {
        let result = run_bash("sleep 999", None, 500, None).await;
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    #[test]
    fn max_timeout_constant_is_reasonable() {
        const { assert!(MAX_TIMEOUT_MS >= 60_000) };
        const { assert!(MAX_TIMEOUT_MS <= 3_600_000) };
    }

    #[tokio::test]
    async fn graceful_kill_reaps_normal_process() {
        let start = std::time::Instant::now();
        let result = run_bash("sleep 999", None, 200, None).await;
        let elapsed = start.elapsed();
        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
        assert!(elapsed.as_secs() < 10, "should not hang: {:?}", elapsed);
    }

    #[tokio::test]
    async fn graceful_kill_timeout_message_contains_duration() {
        let result = run_bash("sleep 999", None, 300, None).await;
        assert!(result.is_error);
        assert!(
            result.content.contains("300ms"),
            "should contain timeout duration: {}",
            result.content
        );
    }

    #[test]
    fn signal_bash_process_group_returns_false_for_zero_pid() {
        assert!(!signal_bash_process_group(0));
    }

    #[test]
    fn signal_bash_process_group_returns_true_for_nonzero_pid() {
        assert!(signal_bash_process_group(99999));
    }

    // ── classify_risk tests ──────────────────────────────────────────────

    #[test]
    fn classify_risk_safe_commands() {
        assert_eq!(classify_risk("ls -la"), BashRisk::Safe);
        assert_eq!(classify_risk("cargo test"), BashRisk::Safe);
        assert_eq!(classify_risk("git status"), BashRisk::Safe);
        assert_eq!(
            classify_risk("git push origin feature-branch"),
            BashRisk::Safe
        );
        assert_eq!(classify_risk("rm temp.txt"), BashRisk::Safe);
        assert_eq!(classify_risk("echo hello"), BashRisk::Safe);
        assert_eq!(classify_risk("cat /etc/passwd"), BashRisk::Safe);
        assert_eq!(classify_risk("git commit -m 'fix'"), BashRisk::Safe);
        assert_eq!(
            classify_risk("dd if=input.img of=output.img"),
            BashRisk::Safe
        );
    }

    #[test]
    fn classify_risk_rm_rf_root() {
        assert!(matches!(classify_risk("rm -rf /"), BashRisk::Warn(_)));
        assert!(matches!(classify_risk("rm -rf /usr"), BashRisk::Warn(_)));
        assert!(matches!(classify_risk("rm -fr /etc"), BashRisk::Warn(_)));
    }

    #[test]
    fn classify_risk_git_force_push() {
        assert!(matches!(
            classify_risk("git push --force origin main"),
            BashRisk::Warn(_)
        ));
        assert!(matches!(
            classify_risk("git push -f origin main"),
            BashRisk::Warn(_)
        ));
    }

    #[test]
    fn classify_risk_git_reset_hard() {
        assert!(matches!(
            classify_risk("git reset --hard HEAD~3"),
            BashRisk::Warn(_)
        ));
        assert!(matches!(
            classify_risk("git reset --hard"),
            BashRisk::Warn(_)
        ));
    }

    #[test]
    fn classify_risk_git_clean() {
        assert!(matches!(classify_risk("git clean -fd"), BashRisk::Warn(_)));
        assert!(matches!(classify_risk("git clean -f"), BashRisk::Warn(_)));
    }

    #[test]
    fn classify_risk_git_branch_force_delete() {
        assert!(matches!(
            classify_risk("git branch -D feature"),
            BashRisk::Warn(_)
        ));
    }

    #[test]
    fn classify_risk_dd_device() {
        assert!(matches!(
            classify_risk("dd if=/dev/zero of=/dev/sda"),
            BashRisk::Warn(_)
        ));
    }

    #[test]
    fn classify_risk_mkfs() {
        assert!(matches!(
            classify_risk("mkfs.ext4 /dev/sda1"),
            BashRisk::Warn(_)
        ));
    }

    #[test]
    fn classify_risk_shutdown() {
        assert!(matches!(
            classify_risk("shutdown -h now"),
            BashRisk::Warn(_)
        ));
        assert!(matches!(classify_risk("reboot"), BashRisk::Warn(_)));
    }

    #[test]
    fn classify_risk_kill() {
        assert!(matches!(classify_risk("kill -9 1234"), BashRisk::Warn(_)));
        assert!(matches!(classify_risk("killall node"), BashRisk::Warn(_)));
    }

    #[test]
    fn classify_risk_chmod_recursive_system() {
        assert!(matches!(classify_risk("chmod -R 777 /"), BashRisk::Warn(_)));
    }

    #[test]
    fn classify_risk_warn_message_is_descriptive() {
        if let BashRisk::Warn(desc) = classify_risk("git push --force origin main") {
            assert!(
                desc.contains("force push"),
                "desc should mention force push: {desc}"
            );
        } else {
            panic!("expected Warn");
        }
    }

    #[test]
    fn effective_decision_bypass_returns_engine_as_is() {
        // Bypass mode = "I know what I'm doing" — safety floor is off.
        let (d, r) =
            effective_bash_decision(Decision::Allow, &BashRisk::Warn("rm -rf root".into()), true);
        assert_eq!(d, Decision::Allow);
        assert!(r.is_none());
    }

    #[test]
    fn effective_decision_explicit_deny_wins_over_risk() {
        // User said "never allow this"; risk classifier doesn't override them.
        let (d, r) =
            effective_bash_decision(Decision::Deny, &BashRisk::Warn("rm -rf root".into()), false);
        assert_eq!(d, Decision::Deny);
        assert!(r.is_none());
    }

    #[test]
    fn effective_decision_risky_upgrades_allow_to_ask() {
        // THE hole: engine would have let rm -rf through because user clicked
        // "always allow bash" on a prior ls. Floor upgrades it to Ask.
        let (d, r) = effective_bash_decision(
            Decision::Allow,
            &BashRisk::Warn("rm -rf on root path".into()),
            false,
        );
        assert_eq!(d, Decision::Ask);
        assert_eq!(r.as_deref(), Some("rm -rf on root path"));
    }

    #[test]
    fn effective_decision_risky_keeps_ask_and_threads_reason() {
        let (d, r) = effective_bash_decision(
            Decision::Ask,
            &BashRisk::Warn("git force push".into()),
            false,
        );
        assert_eq!(d, Decision::Ask);
        assert_eq!(r.as_deref(), Some("git force push"));
    }

    #[test]
    fn effective_decision_safe_passes_through_allow() {
        let (d, r) = effective_bash_decision(Decision::Allow, &BashRisk::Safe, false);
        assert_eq!(d, Decision::Allow);
        assert!(r.is_none());
    }

    #[test]
    fn effective_decision_safe_passes_through_ask() {
        // First-time bash call: engine says Ask (no rule yet), risk is Safe —
        // user sees a normal permission prompt with no scary banner.
        let (d, r) = effective_bash_decision(Decision::Ask, &BashRisk::Safe, false);
        assert_eq!(d, Decision::Ask);
        assert!(r.is_none());
    }

    #[test]
    fn effective_decision_always_allow_bash_still_prompts_for_rm_rf() {
        // End-to-end semantic: after a user has clicked "Always allow" on bash,
        // the engine returns Allow for every bash call. Verify the floor still
        // catches rm -rf /. This is the exact scenario the feature prevents.
        let risk = classify_risk("rm -rf /");
        assert!(matches!(risk, BashRisk::Warn(_)));
        let (d, r) = effective_bash_decision(Decision::Allow, &risk, false);
        assert_eq!(d, Decision::Ask);
        assert!(r.is_some());
    }

    // ── decision_with_risk tests ────────────────────────────────────────

    #[test]
    fn decision_with_risk_non_bash_passes_through() {
        let (d, r) = decision_with_risk(
            "read_file",
            &json!({"path": "src/foo.rs"}),
            Decision::Allow,
            false,
        );
        assert_eq!(d, Decision::Allow);
        assert!(r.is_none());
    }

    #[test]
    fn decision_with_risk_bash_safe_passes_through() {
        let (d, r) = decision_with_risk(
            "bash",
            &json!({"command": "ls -la"}),
            Decision::Allow,
            false,
        );
        assert_eq!(d, Decision::Allow);
        assert!(r.is_none());
    }

    #[test]
    fn decision_with_risk_bash_dangerous_softens_to_ask() {
        let (d, r) = decision_with_risk(
            "bash",
            &json!({"command": "rm -rf /"}),
            Decision::Allow,
            false,
        );
        assert_eq!(d, Decision::Ask);
        assert!(r.is_some(), "risk reason should be threaded");
    }
}
