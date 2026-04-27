//! LLM prompt builders and user-facing instruction strings for TUI slash
//! commands. Pure functions over plain values — no App/Session reach-ins.
//! Extracted from `commands.rs` to keep the dispatch file focused on
//! parsing and routing.

use crate::tui::app::EffortLevel;

/// Return the effort hint string to prepend to user messages, or `None` for normal (no hint).
pub fn effort_instruction(level: &EffortLevel) -> Option<&'static str> {
    match level {
        EffortLevel::Quick => Some(
            "Respond briefly and directly. Prefer concise answers over exhaustive explanations. \
             Skip preamble and summary; give the essential information only.",
        ),
        EffortLevel::Normal => None,
        EffortLevel::Thorough => Some(
            "Be thorough and comprehensive. Check all relevant edge cases, explain your \
             reasoning step-by-step, and surface any potential issues even if not directly \
             asked. Prefer depth over brevity.",
        ),
    }
}

/// Return the brief-mode hint string, or `None` when brief mode is off.
/// Distinct from effort level: brief mode is a binary on/off toggle for response format.
pub fn brief_instruction(brief_mode: bool) -> Option<&'static str> {
    if brief_mode {
        Some(
            "Be brief. Use bullet points or very short prose. \
             No preamble, no summary, no filler. Get to the point immediately.",
        )
    } else {
        None
    }
}

/// Return the plan-mode hint string, or `None` when plan mode is off.
pub fn plan_instruction(plan_mode: bool) -> Option<&'static str> {
    if plan_mode {
        Some(
            "Plan mode is ACTIVE. Write tools are disabled. \
             Focus on reading the codebase, understanding the problem, and designing your approach. \
             Use read_file, glob, grep, ls, and web_search to explore. \
             Describe your plan to the user. When ready, the user will exit plan mode with /plan."
        )
    } else {
        None
    }
}

/// Build the LLM prompt for `/bughunter`.
/// `files` is a list of source file paths; `focus` narrows the analysis (empty = general).
pub fn build_bughunter_prompt(files: &[String], focus: &str) -> String {
    let file_list = if files.is_empty() {
        "  (no source files found)".to_string()
    } else {
        files
            .iter()
            .map(|f| format!("  - {}", f))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let focus_line = if focus.is_empty() {
        "Look broadly for all categories of bugs.".to_string()
    } else {
        format!("Focus specifically on: **{}**.", focus)
    };

    format!(
        "You are a senior code reviewer performing a bug-hunt on this Rust codebase.\n\
         {focus_line}\n\n\
         Source files in scope:\n\
         {file_list}\n\n\
         For each potential bug you find, report:\n\
         1. **File and line** (if known)\n\
         2. **Bug category** (panic, logic error, resource leak, race condition, \
            integer overflow, error ignored, off-by-one, etc.)\n\
         3. **Description** of the problem\n\
         4. **Suggested fix**\n\n\
         Use the available file-read tools to inspect files before reporting. \
         Be precise — only report real issues, not style preferences.",
        focus_line = focus_line,
        file_list = file_list,
    )
}

/// Build a human-readable context report for `/context`.
///
/// All parameters are plain values so the function is pure and testable.
#[allow(clippy::too_many_arguments)]
pub fn build_context_report(
    session_id: &str,
    session_title: Option<&str>,
    model: &str,
    total_tokens: usize,
    ctx_usage: Option<(usize, usize)>,
    turns: usize,
    mcp_servers: &[(String, usize)],
    has_pending_context: bool,
    compaction_stage: Option<&crate::compaction::CompactionStage>,
    context_window: Option<usize>,
    system_prompt_chars: Option<usize>,
    wiki_snippet_chars: Option<usize>,
) -> String {
    let title_line = session_title
        .map(|t| format!("  Title:         {}\n", t))
        .unwrap_or_default();

    let ctx_line = if let Some((used, limit)) = ctx_usage {
        let pct = if limit > 0 { used * 100 / limit } else { 0 };
        format!("  Context window: {}/{} tokens ({}%)\n", used, limit, pct)
    } else {
        "  Context window: unknown\n".to_string()
    };

    // None = caller didn't measure; omit the line rather than pollute the report.
    let system_prompt_line = match system_prompt_chars {
        Some(n) => format!("  System prompt: {} chars\n", n),
        None => String::new(),
    };

    // None / Some(0) = no wiki index or empty index; surface as informative state.
    let wiki_snippet_line = match wiki_snippet_chars {
        Some(0) | None => "  Wiki snippet:  none\n".to_string(),
        Some(n) => format!("  Wiki snippet:  {} chars\n", n),
    };

    let mcp_line = if mcp_servers.is_empty() {
        "  MCP servers:    none\n".to_string()
    } else {
        let names: Vec<String> = mcp_servers
            .iter()
            .map(|(name, tool_count)| format!("{} ({} tools)", name, tool_count))
            .collect();
        format!("  MCP servers:    {}\n", names.join(", "))
    };

    let pending_line = if has_pending_context {
        "  Pending context: yes — will be prepended to next message\n"
    } else {
        "  Pending context: none\n"
    };

    // Compaction status — show only when a stage is actively running.
    // `CompactionStage::None` is treated as "not running" and produces
    // no line, matching the Option<CompactionStage> semantics in
    // `app.rs` where the outer None means no active compaction.
    let compaction_line = match compaction_stage {
        Some(crate::compaction::CompactionStage::Microcompact {
            chars_removed,
            messages_affected,
        }) => format!(
            "  Compacting:     Stage 1 microcompact — trimming tool results \
             ({} chars from {} msgs)\n",
            chars_removed, messages_affected
        ),
        Some(crate::compaction::CompactionStage::SessionMemory { messages_dropped }) => format!(
            "  Compacting:     Stage 2 session memory — pruning {} old msgs\n",
            messages_dropped
        ),
        Some(crate::compaction::CompactionStage::FullSummary {
            messages_summarized,
        }) => format!(
            "  Compacting:     Stage 3 full summary — summarizing {} msgs\n",
            messages_summarized
        ),
        Some(crate::compaction::CompactionStage::Emergency) => {
            "  Compacting:     Emergency — force-dropped oldest messages\n".to_string()
        }
        Some(crate::compaction::CompactionStage::None) | None => String::new(),
    };

    // Compaction thresholds — only rendered when the context window is
    // known. Answers "why did compaction fire?" / "when will it fire
    // next?" by showing each stage's absolute token trigger plus a
    // headroom hint from current usage to the next stage.
    let thresholds_block = if let Some(window) = context_window {
        if window == 0 {
            String::new()
        } else {
            let t = crate::compaction::CompactionThresholds::from_context_window(window);
            // Defaults to 0 when ctx_usage is absent — in practice
            // callers that pass `context_window=Some(N)` also pass
            // ctx_usage, so the hint line remains accurate. The
            // unwrap_or(0) is a safety default, not a supported combo.
            let used = ctx_usage.map_or(0, |(u, _)| u);
            let next_line = next_threshold_hint(used, &t);
            format!(
                "  Compaction thresholds (auto-fires at % of {} tok window):\n    \
                   Stage 1 (microcompact):   {:>3}% = {:>6} tok\n    \
                   Stage 2 (session memory): {:>3}% = {:>6} tok\n    \
                   Stage 3 (full summary):   {:>3}% = {:>6} tok\n\
                 {}",
                window,
                t.micro_compact * 100 / window,
                t.micro_compact,
                t.session_compact * 100 / window,
                t.session_compact,
                t.full_compact * 100 / window,
                t.full_compact,
                next_line,
            )
        }
    } else {
        String::new()
    };

    format!(
        "Session context:\n\
         {}  Session ID:     {}\n\
           Model:          {}\n\
           Turns:          {}\n\
           Total tokens:   {}\n\
         {}{}{}{}{}{}{}\
         ",
        title_line,
        session_id,
        model,
        turns,
        total_tokens,
        ctx_line,
        system_prompt_line,
        wiki_snippet_line,
        mcp_line,
        pending_line,
        compaction_line,
        thresholds_block,
    )
}

/// Compute a "N tok until next compaction trigger" hint for the
/// `/context` output. Past Stage 3 produces an emergency-warning line
/// instead; below Stage 1, points at Stage 1. Returns a fully-formed
/// line including leading indentation and trailing newline so the
/// caller can concatenate directly.
fn next_threshold_hint(used: usize, t: &crate::compaction::CompactionThresholds) -> String {
    let next = if used < t.micro_compact {
        Some(("Stage 1", t.micro_compact))
    } else if used < t.session_compact {
        Some(("Stage 2", t.session_compact))
    } else if used < t.full_compact {
        Some(("Stage 3", t.full_compact))
    } else {
        None
    };
    match next {
        Some((name, boundary)) => format!(
            "    Current: {} tok — {} tok until {}.\n",
            used,
            boundary.saturating_sub(used),
            name
        ),
        None => format!(
            "    Current: {} tok — past Stage 3 threshold ({} tok); \
             emergency compact may fire.\n",
            used, t.full_compact
        ),
    }
}

/// Build the LLM prompt for `/advisor`.
/// `topic` narrows the scope (empty = general architecture review).
/// `files` is the list of source files visible to the model.
pub fn build_advisor_prompt(topic: &str, files: &[String]) -> String {
    let file_list = if files.is_empty() {
        "  (no source files provided)".to_string()
    } else {
        files
            .iter()
            .map(|f| format!("  - {}", f))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let scope = if topic.is_empty() {
        "Review the overall architecture and design of the codebase.".to_string()
    } else {
        format!("Focus on: **{}**.", topic)
    };

    format!(
        "You are a senior software architect providing design and architecture advice.\n\
         {scope}\n\n\
         Source files in scope:\n\
         {file_list}\n\n\
         Please:\n\
         1. Identify architectural strengths worth preserving\n\
         2. Point out structural issues, coupling problems, or abstraction leaks\n\
         3. Suggest concrete refactoring steps with expected benefits\n\
         4. Note any scalability, maintainability, or testability concerns\n\n\
         Use the available file-read tools to inspect any file before commenting on it. \
         Be specific — reference actual module/function names.",
        scope = scope,
        file_list = file_list,
    )
}

/// Build the LLM prompt for `/summary`.
///
/// `turns` is the number of user turns so far; `focus` optionally narrows the summary.
/// Pure and testable: no side effects.
pub fn build_summary_prompt(turns: usize, focus: &str) -> String {
    let scope = if focus.is_empty() {
        "Provide a comprehensive summary of the entire conversation.".to_string()
    } else {
        format!("Summarize the conversation, focusing on: **{}**.", focus)
    };

    format!(
        "You are summarizing a coding session that spanned {} user turn{}.\n\n\
         {scope}\n\n\
         Your summary should cover:\n\
         1. **What was accomplished** — features built, bugs fixed, or problems solved\n\
         2. **Key decisions made** — important design or implementation choices\n\
         3. **Files changed** — which files were modified and why\n\
         4. **Remaining work** — TODOs, known issues, or next steps mentioned\n\n\
         Be concise but complete. Use bullet points for clarity. \
         If specific files or functions were central to the work, name them.",
        turns,
        if turns == 1 { "" } else { "s" },
        scope = scope,
    )
}

/// Build the version info string shown by `/version`.
pub fn build_version_info(model: &str, host: &str) -> String {
    let version = env!("CARGO_PKG_VERSION");
    format!(
        "dm v{version}\n  model : {model}\n  host  : {host}",
        version = version,
        model = model,
        host = host,
    )
}
