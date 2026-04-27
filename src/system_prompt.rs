//! Generates the core system prompt that frames the LLM context.
//!
//! Includes identity handling, workspace structure injection, and wiki awareness.

use chrono::Local;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use tokio::process::Command;

pub async fn build_system_prompt(add_dirs: &[PathBuf], extra: Option<&str>) -> String {
    build_system_prompt_with_context(add_dirs, extra, &[]).await
}

/// Fast-path variant used by the web API: callers that already hold the
/// wiki snippets (cached at state construction) pass them in, avoiding a
/// per-call disk read + parse of `.dm/wiki/index.md` and
/// `.dm/wiki/summaries/project.md`. `None` for either field suppresses
/// the corresponding `<wiki_...>` block; the disk fallback is NEVER
/// invoked here — that's the whole point of the fast path.
pub async fn build_system_prompt_with_snippets(
    add_dirs: &[PathBuf],
    extra: Option<&str>,
    wiki_snippet: Option<&str>,
    wiki_summary: Option<&str>,
    wiki_fresh: Option<&str>,
) -> String {
    build_system_prompt_inner(
        add_dirs,
        extra,
        &[],
        true,
        false,
        "",
        wiki_snippet,
        wiki_summary,
        wiki_fresh,
        false,
    )
    .await
}

/// Full version with context globs.
pub async fn build_system_prompt_with_context(
    add_dirs: &[PathBuf],
    extra: Option<&str>,
    context_globs: &[String],
) -> String {
    build_system_prompt_full(add_dirs, extra, context_globs, true, false).await
}

/// Full version with context globs and workspace-context control.
/// `workspace_context`: if true, inject README.md from cwd when no project DM.md is present.
/// `no_dm_md`: if true, skip DM.md inheritance walk entirely.
/// `tool_hints`: if non-empty, injected as a `<tool_usage>` section with per-tool guidelines.
pub async fn build_system_prompt_full(
    add_dirs: &[PathBuf],
    extra: Option<&str>,
    context_globs: &[String],
    workspace_context: bool,
    no_dm_md: bool,
) -> String {
    build_system_prompt_full_with_tools(
        add_dirs,
        extra,
        context_globs,
        workspace_context,
        no_dm_md,
        "",
    )
    .await
}

/// Inner builder that accepts optional tool hints.
pub async fn build_system_prompt_full_with_tools(
    add_dirs: &[PathBuf],
    extra: Option<&str>,
    context_globs: &[String],
    workspace_context: bool,
    no_dm_md: bool,
    tool_hints: &str,
) -> String {
    build_system_prompt_inner(
        add_dirs,
        extra,
        context_globs,
        workspace_context,
        no_dm_md,
        tool_hints,
        None,
        None,
        None,
        true,
    )
    .await
}

/// Actual implementation. `preloaded_wiki_snippet = Some(_)` splices the
/// caller-provided snippet. `preloaded_wiki_summary = Some(_)` splices the
/// project-summary narrative (from `summaries/project.md`). `wiki_disk_fallback`
/// controls whether, in the absence of preloaded values, the builder falls
/// back to reading `.dm/wiki/index.md` and `.dm/wiki/summaries/project.md`
/// off disk. The sibling `_with_snippet(s)` entry points pass `false` so the
/// hot path NEVER re-reads those files, even if the caller intentionally
/// passed `None` (no wiki).
#[allow(clippy::too_many_arguments)]
async fn build_system_prompt_inner(
    add_dirs: &[PathBuf],
    extra: Option<&str>,
    context_globs: &[String],
    workspace_context: bool,
    no_dm_md: bool,
    tool_hints: &str,
    preloaded_wiki_snippet: Option<&str>,
    preloaded_wiki_summary: Option<&str>,
    preloaded_wiki_fresh: Option<&str>,
    wiki_disk_fallback: bool,
) -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cwd_path = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let cwd = cwd_path.to_string_lossy().to_string();
    let now = Local::now().format("%Y-%m-%d %H:%M:%S %Z").to_string();
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|s| s.rsplit('/').next().map(String::from))
        .unwrap_or_else(|| "unknown".to_string());
    let (is_git, default_branch) = get_git_info().await;

    let git_status = get_git_status().await;
    let dm_md = if no_dm_md { vec![] } else { load_dm_md().await };

    let identity = crate::identity::load_for_cwd();
    let project_name = identity.display_name();

    // Determine if a project-level (cwd) DM.md was found
    let has_project_dm_md = dm_md.iter().any(|(path, _)| {
        let p = PathBuf::from(path);
        // A project-level DM.md sits inside the cwd subtree (not in ~/.dm/)
        if let Ok(abs) = std::fs::canonicalize(&p) {
            abs.starts_with(&cwd_path)
        } else {
            false
        }
    });

    let git_repo_str = if is_git { "Yes" } else { "No" };
    let branch_line = if is_git {
        format!("Default branch: {default_branch}\n")
    } else {
        String::new()
    };

    let mut prompt = format!(
        "You are Dark Matter (dm), an AI coding assistant running locally via Ollama.\n\
        You are currently framing the project: {project_name}.\n\n\
        <system_info>\n\
        OS: {os} ({arch})\n\
        Shell: {shell}\n\
        Date: {now}\n\
        Working directory: {cwd}\n\
        Is git repo: {git_repo_str}\n\
        {branch_line}\
        </system_info>\n\n\
        <guidelines>\n\
        - Read files before modifying them. Understand existing code before suggesting changes.\n\
        - Don't create files unless necessary. Prefer editing existing files over creating new ones.\n\
        - Don't add features, refactor, or make improvements beyond what was asked.\n\
        - Don't add error handling for scenarios that can't happen. Only validate at system boundaries.\n\
        - Don't create abstractions for one-time operations. Three similar lines beat a premature abstraction.\n\
        - If an approach fails, diagnose why before switching tactics. Read the error and check your assumptions.\n\
        - Be careful not to introduce security vulnerabilities (command injection, XSS, SQL injection).\n\
        - When reporting results, be accurate — don't claim tests pass if they didn't, and don't hedge confirmed results.\n\
        - Use tools efficiently: prefer grep/glob for searching over bash, use file_edit for modifications over file_write.\n\
        - For multi-step tasks, work incrementally and verify each step before proceeding.\n\
        </guidelines>\n"
    );

    if !tool_hints.is_empty() {
        prompt.push_str("\n<tool_usage>\n");
        prompt.push_str(tool_hints);
        prompt.push_str("</tool_usage>\n");
    }

    if let Some(status) = git_status {
        write!(prompt, "\n<git_status>\n{status}\n</git_status>\n")
            .expect("write to String never fails");
    }

    if !dm_md.is_empty() {
        prompt.push_str("\n<dm_md>\n");
        for (path, content) in &dm_md {
            writeln!(prompt, "<!-- From: {} -->\n{}", path, content)
                .expect("write to String never fails");
        }
        prompt.push_str("</dm_md>\n");
    }

    // Workspace context: inject README.md when no project DM.md and flag is on
    if workspace_context && !has_project_dm_md {
        if let Some(readme) = load_workspace_readme_at(&cwd_path).await {
            write!(
                prompt,
                "\n<workspace_context source=\"README.md\">\n{}\n</workspace_context>\n",
                readme
            )
            .expect("write to String never fails");
        }
    }

    if !add_dirs.is_empty() {
        prompt.push_str("\n## Additional Context Directories\n");
        for dir in add_dirs {
            writeln!(prompt, "- {}", dir.display()).expect("write to String never fails");
        }
    }

    // Inject files from context globs
    if !context_globs.is_empty() {
        const CHAR_CAP: usize = 50_000;
        const SIZE_CAP: u64 = 100 * 1024; // 100 KB

        let mut injected_section = String::new();
        let mut total_chars: usize = 0;
        let mut omitted: usize = 0;
        let mut capped = false;

        for glob_pattern in context_globs {
            let Ok(paths) = glob::glob(glob_pattern) else {
                continue;
            };
            for path_result in paths {
                let Ok(path) = path_result else {
                    continue;
                };

                if capped {
                    omitted += 1;
                    continue;
                }

                // Skip files >100KB
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta.len() > SIZE_CAP {
                        omitted += 1;
                        continue;
                    }
                } else {
                    continue;
                }

                let Ok(contents) = std::fs::read_to_string(&path) else {
                    omitted += 1;
                    continue;
                };

                if total_chars + contents.len() > CHAR_CAP {
                    omitted += 1;
                    capped = true;
                    continue;
                }

                // Detect language for code fence
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let lang = ext_to_lang(ext);

                let display_path = path.to_string_lossy();
                write!(
                    injected_section,
                    "\n### {}\n```{}\n{}\n```\n",
                    display_path, lang, contents
                )
                .expect("write to String never fails");
                total_chars += contents.len();
            }
        }

        if !injected_section.is_empty() || omitted > 0 {
            prompt.push_str("\n## Injected Context\n");
            prompt.push_str(&injected_section);
            if omitted > 0 {
                writeln!(
                    prompt,
                    "\n[context truncated — {} file{} omitted]",
                    omitted,
                    if omitted == 1 { "" } else { "s" }
                )
                .expect("write to String never fails");
            }
        }
    }

    // Wiki context — read-only, best-effort. Narrative summary first,
    // TOC index second: the model sees "what the project is" before the
    // catalog of pages. Callers on the hot path (web API) pre-load both
    // snippets at state construction. Otherwise, fall back to gating on
    // file existence so the system prompt never materializes a wiki tree
    // on projects that don't have one (Wiki::open → ensure_layout would).
    const WIKI_SUMMARY_DISK_BUDGET: usize = 4096;
    if let Some(summary) = preloaded_wiki_summary {
        prompt.push_str("\n<wiki_summary>\n");
        prompt.push_str(summary);
        prompt.push_str("\n</wiki_summary>\n");
    } else if wiki_disk_fallback {
        // Gate on summary file OR dirty-marker presence: ingest may have
        // marked staleness before a summary ever existed. Both only appear
        // inside `.dm/wiki/`, so wiki-less projects stay untouched.
        let summary_path = cwd_path.join(".dm/wiki/summaries/project.md");
        let dirty_marker = cwd_path.join(".dm/wiki/.summary-dirty");
        if summary_path.is_file() || dirty_marker.is_file() {
            if let Ok(wiki) = crate::wiki::Wiki::open(&cwd_path) {
                // Lazy-regenerate if a prior ingest marked the summary
                // stale. Best-effort: on regen failure fall through to
                // the existing (stale) snippet — still better than none.
                let _ = wiki.ensure_summary_current();
                if let Some(summary) = wiki.project_summary_snippet(WIKI_SUMMARY_DISK_BUDGET) {
                    prompt.push_str("\n<wiki_summary>\n");
                    prompt.push_str(&summary);
                    prompt.push_str("\n</wiki_summary>\n");
                }
            }
        }
    }
    // Fresh-pages ranking (C86): top-K entity+concept pages by
    // `last_updated` desc, drawn from the C85 index cache (zero I/O on
    // modern indexes). Emits between `<wiki_summary>` (narrative) and
    // `<wiki_index>` (catalog) so the model sees "what the project is",
    // "what's live right now", then "where to find more" in that order.
    const WIKI_FRESH_DISK_BUDGET: usize = 1024;
    if let Some(fresh) = preloaded_wiki_fresh {
        prompt.push_str("\n<wiki_fresh>\n");
        prompt.push_str(fresh);
        prompt.push_str("\n</wiki_fresh>\n");
    } else if wiki_disk_fallback && cwd_path.join(".dm/wiki/index.md").is_file() {
        if let Ok(wiki) = crate::wiki::Wiki::open(&cwd_path) {
            if let Some(fresh) = wiki.fresh_pages_snippet(WIKI_FRESH_DISK_BUDGET) {
                prompt.push_str("\n<wiki_fresh>\n");
                prompt.push_str(&fresh);
                prompt.push_str("\n</wiki_fresh>\n");
            }
        }
    }
    if let Some(snippet) = preloaded_wiki_snippet {
        prompt.push_str("\n<wiki_index>\n");
        prompt.push_str(snippet);
        prompt.push_str("\n</wiki_index>\n");
    } else if wiki_disk_fallback && cwd_path.join(".dm/wiki/index.md").is_file() {
        if let Ok(wiki) = crate::wiki::Wiki::open(&cwd_path) {
            if let Some(snippet) = wiki.context_snippet() {
                prompt.push_str("\n<wiki_index>\n");
                prompt.push_str(&snippet);
                prompt.push_str("\n</wiki_index>\n");
            }
        }
    }

    if let Some(extra) = extra {
        prompt.push_str("\n\n## Additional Instructions\n\n");
        prompt.push_str(extra);
    }

    prompt
}

/// Insert model identity into an existing system prompt's `<system_info>` block.
/// If no `</system_info>` tag is found, the info is appended at the end.
pub fn append_model_info(prompt: &mut String, model: &str, tool_model: Option<&str>) {
    let mut info = format!("Model: {}\n", model);
    if let Some(tm) = tool_model {
        writeln!(info, "Tool model: {}", tm).expect("write to String never fails");
    }
    if let Some(pos) = prompt.find("</system_info>") {
        prompt.insert_str(pos, &info);
    } else {
        prompt.push_str(&info);
    }
    if let Some(hint) = crate::ollama::model_hints::model_hint(model) {
        prompt.push_str("\n\n## Model-Specific Guidance\n");
        prompt.push_str(hint);
        prompt.push('\n');
    }
}

fn ext_to_lang(ext: &str) -> &str {
    match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" => "typescript",
        "go" => "go",
        "c" => "c",
        "cpp" | "cc" | "cxx" => "cpp",
        "java" => "java",
        "sh" | "bash" => "bash",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "json" => "json",
        "md" => "markdown",
        "html" => "html",
        "css" => "css",
        _ => "",
    }
}

async fn get_git_info() -> (bool, String) {
    let is_repo = Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !is_repo {
        return (false, String::new());
    }
    let branch = Command::new("git")
        .args(["symbolic-ref", "refs/remotes/origin/HEAD", "--short"])
        .output()
        .await
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                s.strip_prefix("origin/").map(String::from)
            } else {
                None
            }
        })
        .unwrap_or_else(|| "main".to_string());
    (true, branch)
}

async fn get_git_status() -> Option<String> {
    let output = Command::new("git")
        .args(["status", "--short", "--branch"])
        .output()
        .await
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Load README.md from `dir` (if present, ≤32KB).
/// Returns `None` when no README exists or it can't be read.
async fn load_workspace_readme_at(dir: &Path) -> Option<String> {
    const MAX_BYTES: usize = 32 * 1024;
    for name in &["README.md", "README.txt", "README"] {
        let path = dir.join(name);
        if let Ok(content) = tokio::fs::read_to_string(&path).await {
            let trimmed = if content.len() > MAX_BYTES {
                let cut = crate::util::safe_truncate(&content, MAX_BYTES);
                format!("{}\n[...truncated]", cut)
            } else {
                content
            };
            return Some(trimmed);
        }
    }
    None
}

/// Returns the README file name found in cwd (for doctor), or None.
pub async fn workspace_context_source() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    for name in &["README.md", "README.txt", "README"] {
        if cwd.join(name).exists() {
            return Some((*name).to_string());
        }
    }
    None
}

async fn load_dm_md() -> Vec<(String, String)> {
    let mut results = Vec::new();

    // User-level DM.md
    if let Some(home) = dirs::home_dir() {
        let user_claude = home.join(".dm").join("DM.md");
        if let Ok(content) = tokio::fs::read_to_string(&user_claude).await {
            results.push((user_claude.to_string_lossy().to_string(), content));
        }
    }

    // Project-level: walk up from cwd
    if let Ok(cwd) = std::env::current_dir() {
        let candidates = collect_dm_md_candidates(&cwd);
        for path in candidates.into_iter().rev() {
            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                results.push((path.to_string_lossy().to_string(), content));
            }
        }
    }

    results
}

pub const DM_MD_WALK_CAP: usize = 10;

/// Public wrapper for use by `dm doctor`.
pub fn collect_dm_md_candidates_for_doctor(start: &Path) -> Vec<PathBuf> {
    collect_dm_md_candidates(start)
}

fn collect_dm_md_candidates(start: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut current = start.to_path_buf();
    let mut levels = 0usize;
    loop {
        if levels >= DM_MD_WALK_CAP {
            break;
        }
        for candidate in &["DM.md", ".dm/DM.md"] {
            let p = current.join(candidate);
            if p.exists() {
                found.push(p);
            }
        }
        levels += 1;
        if !current.pop() {
            break;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn context_glob_injects_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("test.rs");
        std::fs::write(&file, "fn hello() {}").unwrap();
        let pattern = file.to_str().unwrap().to_string();
        let prompt = build_system_prompt_with_context(&[], None, &[pattern]).await;
        assert!(prompt.contains("fn hello()"), "should inject file content");
    }

    #[tokio::test]
    async fn context_glob_skips_missing_pattern() {
        let prompt =
            build_system_prompt_with_context(&[], None, &["/nonexistent/**/*.rs".to_string()])
                .await;
        // Should not panic/error, just produce a normal prompt
        assert!(
            prompt.contains("You are Dark Matter"),
            "prompt should still be valid"
        );
    }

    #[tokio::test]
    async fn context_glob_caps_at_50k_chars() {
        let dir = tempfile::tempdir().unwrap();
        // Write 3 files of ~20k chars each (total 60k, over the 50k cap)
        let content = "x".repeat(20_000);
        for i in 0..3 {
            let path = dir.path().join(format!("large{}.txt", i));
            std::fs::write(&path, &content).unwrap();
        }
        let pattern = dir.path().join("*.txt").to_string_lossy().to_string();
        let prompt = build_system_prompt_with_context(&[], None, &[pattern]).await;
        // Find the injected section length
        let injected_start = prompt.find("## Injected Context").unwrap_or(0);
        let injected_section = &prompt[injected_start..];
        assert!(
            injected_section.len() <= 55_000,
            "injected section should be bounded near 50k chars, got {}",
            injected_section.len()
        );
        assert!(
            prompt.contains("omitted"),
            "should note omitted files when cap is hit"
        );
    }

    #[tokio::test]
    async fn load_workspace_readme_at_finds_readme_md() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "# My Project\nHello world.").unwrap();
        let result = load_workspace_readme_at(dir.path()).await;
        assert!(result.is_some(), "should find README.md");
        assert!(result.unwrap().contains("My Project"));
    }

    #[tokio::test]
    async fn load_workspace_readme_at_returns_none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_workspace_readme_at(dir.path()).await;
        assert!(result.is_none(), "should return None when no README");
    }

    #[tokio::test]
    async fn load_workspace_readme_at_truncates_large_files() {
        let dir = tempfile::tempdir().unwrap();
        let big = "x".repeat(33 * 1024); // > 32KB
        std::fs::write(dir.path().join("README.md"), &big).unwrap();
        let result = load_workspace_readme_at(dir.path()).await.unwrap();
        assert!(
            result.contains("[...truncated]"),
            "large README should be truncated"
        );
        assert!(result.len() <= 32 * 1024 + 50); // small margin for the truncation suffix
    }

    #[tokio::test]
    async fn workspace_context_source_finds_readme() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "content").unwrap();
        let result = load_workspace_readme_at(dir.path()).await;
        assert!(result.is_some(), "README.md should be loadable");
    }

    #[test]
    fn dm_md_walk_merges_parent() {
        // parent/DM.md + parent/child/DM.md — both should appear,
        // child appears later (deepest wins on conflict in concatenated output).
        let root = tempfile::tempdir().unwrap();
        let parent = root.path();
        let child = parent.join("child");
        std::fs::create_dir_all(&child).unwrap();
        std::fs::write(parent.join("DM.md"), "# Parent rules").unwrap();
        std::fs::write(child.join("DM.md"), "# Child rules").unwrap();

        let candidates = collect_dm_md_candidates(&child);
        // child/DM.md and parent/DM.md should both appear
        assert!(candidates.iter().any(|p| p.ends_with("child/DM.md")));
        assert!(candidates
            .iter()
            .any(|p| p.ends_with("DM.md") && !p.ends_with("child/DM.md")));
    }

    #[test]
    fn dm_md_walk_stops_at_root() {
        // Walk from a deep path — should always terminate, never panic.
        let candidates = collect_dm_md_candidates(std::path::Path::new("/"));
        // Just assert it runs without panic; may find 0 or more files.
        let _ = candidates;
    }

    #[test]
    fn dm_md_cap_at_10_levels() {
        // Build a 12-level deep directory tree; walk must stop at 10.
        let root = tempfile::tempdir().unwrap();
        let mut current = root.path().to_path_buf();
        for i in 0..12 {
            current = current.join(format!("level{}", i));
            std::fs::create_dir_all(&current).unwrap();
            std::fs::write(current.join("DM.md"), format!("# Level {}", i)).unwrap();
        }
        // collect from the deepest level
        let candidates = collect_dm_md_candidates(&current);
        assert!(
            candidates.len() <= DM_MD_WALK_CAP,
            "should collect at most {} files, got {}",
            DM_MD_WALK_CAP,
            candidates.len()
        );
    }

    #[test]
    fn ext_to_lang_maps_common_extensions() {
        assert_eq!(ext_to_lang("rs"), "rust");
        assert_eq!(ext_to_lang("py"), "python");
        assert_eq!(ext_to_lang("js"), "javascript");
        assert_eq!(ext_to_lang("mjs"), "javascript");
        assert_eq!(ext_to_lang("cjs"), "javascript");
        assert_eq!(ext_to_lang("ts"), "typescript");
        assert_eq!(ext_to_lang("go"), "go");
        assert_eq!(ext_to_lang("c"), "c");
        assert_eq!(ext_to_lang("cpp"), "cpp");
        assert_eq!(ext_to_lang("cc"), "cpp");
        assert_eq!(ext_to_lang("cxx"), "cpp");
        assert_eq!(ext_to_lang("java"), "java");
        assert_eq!(ext_to_lang("sh"), "bash");
        assert_eq!(ext_to_lang("bash"), "bash");
        assert_eq!(ext_to_lang("toml"), "toml");
        assert_eq!(ext_to_lang("yaml"), "yaml");
        assert_eq!(ext_to_lang("yml"), "yaml");
        assert_eq!(ext_to_lang("json"), "json");
        assert_eq!(ext_to_lang("md"), "markdown");
        assert_eq!(ext_to_lang("html"), "html");
        assert_eq!(ext_to_lang("css"), "css");
    }

    #[test]
    fn ext_to_lang_unknown_returns_empty() {
        assert_eq!(ext_to_lang("xyz"), "");
        assert_eq!(ext_to_lang(""), "");
        assert_eq!(ext_to_lang("txt"), "");
    }

    #[tokio::test]
    async fn extra_instructions_appear_in_prompt() {
        let prompt = build_system_prompt(&[], Some("Always respond in haiku format.")).await;
        assert!(
            prompt.contains("Always respond in haiku format."),
            "extra instructions should appear in prompt"
        );
        assert!(
            prompt.contains("Additional Instructions"),
            "should have an 'Additional Instructions' section"
        );
    }

    #[tokio::test]
    async fn build_system_prompt_returns_valid_base_prompt() {
        let prompt = build_system_prompt(&[], None).await;
        assert!(
            prompt.contains("You are Dark Matter"),
            "should contain identity"
        );
        assert!(
            prompt.contains("system_info"),
            "should contain system_info section"
        );
        assert!(
            prompt.contains("Working directory"),
            "should contain working directory"
        );
        assert!(!prompt.is_empty(), "prompt must not be empty");
    }

    #[tokio::test]
    async fn build_system_prompt_full_no_dm_md_skips_dm_md() {
        let prompt = build_system_prompt_full(&[], None, &[], false, true).await;
        assert!(
            prompt.contains("You are Dark Matter"),
            "should contain identity"
        );
        // With no_dm_md=true, dm_md section should not appear
        assert!(
            !prompt.contains("<dm_md>"),
            "should skip dm_md when no_dm_md is true"
        );
    }

    #[tokio::test]
    async fn build_system_prompt_full_no_workspace_context() {
        let prompt = build_system_prompt_full(&[], None, &[], false, false).await;
        assert!(
            prompt.contains("You are Dark Matter"),
            "should contain identity"
        );
        // workspace_context=false means no README injection
        assert!(
            !prompt.contains("<workspace_context"),
            "should skip workspace context"
        );
    }

    #[tokio::test]
    async fn add_dirs_appear_in_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let prompt = build_system_prompt(std::slice::from_ref(&path), None).await;
        assert!(
            prompt.contains(&path.to_string_lossy().to_string()),
            "additional context directory path should appear in prompt"
        );
    }

    #[tokio::test]
    async fn system_prompt_contains_guidelines() {
        let prompt = build_system_prompt(&[], None).await;
        assert!(
            prompt.contains("<guidelines>"),
            "should contain guidelines section"
        );
        assert!(
            prompt.contains("</guidelines>"),
            "should close guidelines section"
        );
        assert!(
            prompt.contains("Read files before modifying"),
            "should include read-first guideline"
        );
        assert!(
            prompt.contains("security vulnerabilities"),
            "should include security guideline"
        );
    }

    #[tokio::test]
    async fn system_prompt_readme_truncate_multibyte_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        // Build a README that lands a multi-byte char right at the 32KB boundary
        let base = "x".repeat(32 * 1024 - 2);
        let content = format!("{}漢字", base); // 漢 is 3 bytes, straddles boundary
        std::fs::write(dir.path().join("README.md"), &content).unwrap();
        let result = load_workspace_readme_at(dir.path()).await;
        assert!(
            result.is_some(),
            "should not panic on multibyte at boundary"
        );
        let text = result.unwrap();
        assert!(text.contains("[...truncated]"));
    }

    #[tokio::test]
    async fn system_prompt_guidelines_include_tool_efficiency() {
        let prompt = build_system_prompt(&[], None).await;
        assert!(
            prompt.contains("grep/glob"),
            "guidelines should mention preferred search tools"
        );
        assert!(
            prompt.contains("file_edit"),
            "guidelines should mention preferred edit tool"
        );
    }

    #[tokio::test]
    async fn context_glob_omitted_count_accurate() {
        let dir = tempfile::tempdir().unwrap();
        let content = "x".repeat(15_000);
        for i in 0..5 {
            let path = dir.path().join(format!("file{}.txt", i));
            std::fs::write(&path, &content).unwrap();
        }
        let pattern = dir.path().join("*.txt").to_string_lossy().to_string();
        let prompt = build_system_prompt_with_context(&[], None, &[pattern]).await;
        let tail = &prompt[prompt.len().saturating_sub(300)..];
        assert!(
            prompt.contains("2 files omitted"),
            "should count all omitted files, not just the one that triggered the cap. Tail: {}",
            tail
        );
    }

    #[tokio::test]
    async fn context_glob_omitted_count_across_globs() {
        let dir = tempfile::tempdir().unwrap();
        let sub1 = dir.path().join("a");
        std::fs::create_dir(&sub1).unwrap();
        for i in 0..3 {
            std::fs::write(sub1.join(format!("f{}.txt", i)), "y".repeat(18_000)).unwrap();
        }
        let sub2 = dir.path().join("b");
        std::fs::create_dir(&sub2).unwrap();
        for i in 0..2 {
            std::fs::write(sub2.join(format!("g{}.txt", i)), "z".repeat(1_000)).unwrap();
        }
        let patterns = vec![
            sub1.join("*.txt").to_string_lossy().to_string(),
            sub2.join("*.txt").to_string_lossy().to_string(),
        ];
        let prompt = build_system_prompt_with_context(&[], None, &patterns).await;
        let tail = &prompt[prompt.len().saturating_sub(300)..];
        assert!(
            prompt.contains("3 files omitted"),
            "should count omitted from both glob patterns. Tail: {}",
            tail
        );
    }

    #[tokio::test]
    async fn system_prompt_contains_shell_info() {
        let prompt = build_system_prompt(&[], None).await;
        assert!(
            prompt.contains("Shell:"),
            "should include shell type in system_info"
        );
    }

    #[tokio::test]
    async fn system_prompt_contains_git_repo_info() {
        let prompt = build_system_prompt(&[], None).await;
        assert!(
            prompt.contains("Is git repo: Yes") || prompt.contains("Is git repo: No"),
            "should include git repo status"
        );
    }

    #[test]
    fn append_model_info_inserts_before_system_info_close() {
        let mut prompt = "OS: linux\n</system_info>\nrest".to_string();
        append_model_info(&mut prompt, "llama3.3:70b", None);
        assert!(prompt.contains("Model: llama3.3:70b\n</system_info>"));
        assert!(!prompt.contains("Tool model:"));
    }

    #[test]
    fn append_model_info_includes_tool_model_when_set() {
        let mut prompt = "OS: linux\n</system_info>\nrest".to_string();
        append_model_info(&mut prompt, "qwen3:32b", Some("qwen3:8b"));
        assert!(prompt.contains("Model: qwen3:32b"));
        assert!(prompt.contains("Tool model: qwen3:8b"));
    }

    #[test]
    fn append_model_info_appends_when_no_system_info_tag() {
        let mut prompt = "No tags here.".to_string();
        append_model_info(&mut prompt, "llama3", None);
        assert!(prompt.contains("Model: llama3\n"));
    }

    #[test]
    fn append_model_info_includes_hint_for_known_model() {
        let mut prompt = "OS: linux\n</system_info>\nrest".to_string();
        append_model_info(&mut prompt, "gemma4:26b", None);
        assert!(
            prompt.contains("## Model-Specific Guidance"),
            "should inject hint section"
        );
        assert!(prompt.contains("context"), "should contain gemma hint");
    }

    #[test]
    fn append_model_info_no_hint_for_unknown_model() {
        let mut prompt = "OS: linux\n</system_info>\nrest".to_string();
        append_model_info(&mut prompt, "unknown-model:7b", None);
        assert!(
            !prompt.contains("## Model-Specific Guidance"),
            "should not inject hint for unknown model"
        );
    }

    #[tokio::test]
    async fn system_prompt_contains_default_branch() {
        let prompt = build_system_prompt(&[], None).await;
        if prompt.contains("Is git repo: Yes") {
            assert!(
                prompt.contains("Default branch:"),
                "should show default branch in git repo"
            );
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_index_is_injected_when_present() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let wiki = crate::wiki::Wiki::open(&proj).unwrap();
        let idx = crate::wiki::WikiIndex {
            entries: vec![crate::wiki::IndexEntry {
                title: "WIKI_MARKER_42".to_string(),
                path: "entities/marker.md".to_string(),
                one_liner: "WIKI_MARKER_42 is a test entry.".to_string(),
                category: crate::wiki::PageType::Entity,
                last_updated: None,
                outcome: None,
            }],
        };
        wiki.save_index(&idx).unwrap();

        let prompt = build_system_prompt(&[], None).await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(
            prompt.contains("<wiki_index>"),
            "prompt missing wiki_index tag"
        );
        assert!(
            prompt.contains("WIKI_MARKER_42"),
            "marker should appear in injected snippet"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_index_absent_leaves_prompt_unchanged() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let prompt = build_system_prompt(&[], None).await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(
            !prompt.contains("<wiki_index>"),
            "wiki_index must not be injected when .dm/wiki/ is absent"
        );
        assert!(
            !proj.join(".dm").exists(),
            "build_system_prompt must not materialize .dm/ for projects without a wiki"
        );
    }

    #[tokio::test]
    async fn build_system_prompt_with_snippet_uses_provided_snippet() {
        // Hot-path fast case: when a caller pre-loads the snippet (as the
        // web API does at ApiState construction), the builder must splice
        // it verbatim and never touch `.dm/wiki/` on disk — this is the
        // whole point of the sibling entry point.
        let snippet = "## Project Wiki\n\n- precomputed marker RX_9142";
        let prompt = build_system_prompt_with_snippets(&[], None, Some(snippet), None, None).await;
        assert!(
            prompt.contains("<wiki_index>"),
            "wiki_index tag must appear"
        );
        assert!(
            prompt.contains("RX_9142"),
            "pre-loaded snippet content must be spliced verbatim"
        );
    }

    #[tokio::test]
    async fn wiki_summary_block_injects_when_preloaded() {
        // The summary snippet is spliced into its own `<wiki_summary>`
        // block — narrative that the model reads on turn zero, separate
        // from the TOC block.
        let summary = "# Project\n\n## Purpose\n- sentinel PROJ_5812";
        let prompt = build_system_prompt_with_snippets(&[], None, None, Some(summary), None).await;
        assert!(
            prompt.contains("<wiki_summary>"),
            "wiki_summary open tag must appear"
        );
        assert!(
            prompt.contains("</wiki_summary>"),
            "wiki_summary close tag must appear"
        );
        assert!(
            prompt.contains("PROJ_5812"),
            "summary content must be spliced verbatim"
        );
    }

    #[tokio::test]
    async fn wiki_summary_block_omitted_when_none() {
        // No summary passed, no block emitted. A missing summary must
        // not produce an empty `<wiki_summary></wiki_summary>` shell.
        let prompt = build_system_prompt_with_snippets(&[], None, None, None, None).await;
        assert!(
            !prompt.contains("<wiki_summary>"),
            "wiki_summary block must be absent when no summary provided"
        );
    }

    #[tokio::test]
    async fn wiki_summary_precedes_wiki_index() {
        // Ordering canary: narrative summary appears before the TOC.
        // The model reads top-to-bottom so summary first = "what this
        // project is" before the catalog of pages.
        let idx = "## Project Wiki\n\n- entry";
        let sum = "# Project\n\n## Purpose\n- thing";
        let prompt = build_system_prompt_with_snippets(&[], None, Some(idx), Some(sum), None).await;
        let sum_pos = prompt.find("<wiki_summary>").expect("summary tag");
        let idx_pos = prompt.find("<wiki_index>").expect("index tag");
        assert!(
            sum_pos < idx_pos,
            "wiki_summary must precede wiki_index (summary at {}, index at {})",
            sum_pos,
            idx_pos
        );
    }

    #[tokio::test]
    async fn wiki_summary_and_index_coexist() {
        // Both blocks render together when both are provided — they're
        // independent channels, not mutually exclusive.
        let idx = "## Project Wiki\n\n- X_IDX_MARKER";
        let sum = "# Project\n\n- Y_SUM_MARKER";
        let prompt = build_system_prompt_with_snippets(&[], None, Some(idx), Some(sum), None).await;
        assert!(prompt.contains("X_IDX_MARKER"));
        assert!(prompt.contains("Y_SUM_MARKER"));
        assert!(prompt.contains("<wiki_summary>"));
        assert!(prompt.contains("<wiki_index>"));
    }

    // ── C86 <wiki_fresh> integration tests ───────────────────────────────

    #[tokio::test]
    async fn build_system_prompt_inner_emits_wiki_fresh_between_summary_and_index() {
        // Ordering canary for the C86 block. The model reads top-to-bottom,
        // so the sequence must be: narrative summary → what's live now →
        // catalog of pages. A reordering would bury the freshness signal
        // under the TOC.
        let sum = "# Project\n\n## Purpose\n- P86_SUMMARY_MARKER";
        let fresh = "## Fresh wiki pages\n- F86_FRESH_MARKER — live\n";
        let idx = "## Project Wiki\n\n- I86_INDEX_MARKER";
        let prompt =
            build_system_prompt_with_snippets(&[], None, Some(idx), Some(sum), Some(fresh)).await;
        let sum_pos = prompt.find("<wiki_summary>").expect("summary tag present");
        let fresh_pos = prompt.find("<wiki_fresh>").expect("fresh tag present");
        let idx_pos = prompt.find("<wiki_index>").expect("index tag present");
        assert!(
            sum_pos < fresh_pos,
            "wiki_summary must precede wiki_fresh: summary={}, fresh={}",
            sum_pos,
            fresh_pos,
        );
        assert!(
            fresh_pos < idx_pos,
            "wiki_fresh must precede wiki_index: fresh={}, index={}",
            fresh_pos,
            idx_pos,
        );
        assert!(
            prompt.contains("F86_FRESH_MARKER"),
            "fresh content must be spliced verbatim",
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn build_system_prompt_with_snippets_forwards_preloaded_wiki_fresh() {
        // API fast-path contract: `build_system_prompt_with_snippets`
        // passes `wiki_disk_fallback=false` so the disk-fallback arm is
        // genuinely unreachable. A preloaded value must render into the
        // prompt and be suppressed when None is passed — no disk touch
        // either way. Proven by running from a cwd with no `.dm/wiki/`.
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        // Preloaded: block emits, content spliced.
        let fresh = "## Fresh wiki pages\n- CARRIED_MARKER_X86 — here\n";
        let with = build_system_prompt_with_snippets(&[], None, None, None, Some(fresh)).await;

        // Not preloaded: block suppressed even though disk fallback is off.
        let without = build_system_prompt_with_snippets(&[], None, None, None, None).await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(
            with.contains("<wiki_fresh>"),
            "preloaded fresh snippet must render: {}",
            with,
        );
        assert!(
            with.contains("CARRIED_MARKER_X86"),
            "preloaded content must be spliced verbatim",
        );
        assert!(
            !without.contains("<wiki_fresh>"),
            "None + fast-path must suppress the block",
        );
        assert!(
            !proj.join(".dm").exists(),
            "fast-path must not materialize `.dm/` on wiki-less cwds",
        );
    }

    // Identity threading: the system prompt must frame the project around the
    // identity at cwd. Kernel mode (no `.dm/identity.toml`) renders the
    // canonical name; host mode renders the spawned host_project. Locks down
    // the contract so future refactors of `build_system_prompt` can't silently
    // drop the identity wiring.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn build_system_prompt_frames_kernel_when_no_identity_file() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let prompt = build_system_prompt(&[], None).await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(
            prompt.contains("framing the project: dark-matter"),
            "kernel default must frame as canonical dm. Prompt head: {}",
            &prompt[..prompt.len().min(400)],
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn build_system_prompt_frames_host_project_when_identity_file_present() {
        let _g = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(proj.join(".dm")).unwrap();
        std::fs::write(
            proj.join(".dm").join("identity.toml"),
            "mode = \"host\"\nhost_project = \"finance-app\"\n",
        )
        .unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        let prompt = build_system_prompt(&[], None).await;

        std::env::set_current_dir(&orig).unwrap();

        assert!(
            prompt.contains("framing the project: finance-app"),
            "host identity must frame around host_project. Prompt head: {}",
            &prompt[..prompt.len().min(400)],
        );
        assert!(
            !prompt.contains("framing the project: dark-matter"),
            "host mode must not also render the canonical name in the same line",
        );
    }
}
