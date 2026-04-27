use std::fmt::Write as _;

use crate::ollama::client::OllamaClient;
use crate::session::short_id;
use crate::session::storage as session_storage;
use crate::tui::app::{App, EffortLevel, EntryKind};

mod chain;
mod format;
mod git;
mod prompts;
mod session;
mod suggest;
mod tools;
mod wiki;

use chain::{
    handle_chain_add, handle_chain_help, handle_chain_init, handle_chain_list, handle_chain_log,
    handle_chain_metrics, handle_chain_model, handle_chain_pause, handle_chain_presets,
    handle_chain_remove, handle_chain_resume, handle_chain_resume_from, handle_chain_start,
    handle_chain_status, handle_chain_stop, handle_chain_talk, handle_chain_validate,
};
use git::{
    handle_blame, handle_branch, handle_changelog, handle_commit, handle_conflicts, handle_pr,
    handle_review, handle_security_review, handle_stash,
};
use session::{handle_export, handle_fork, handle_resume, handle_sessions, handle_sessions_tree};
use tools::{
    handle_mcp, handle_permissions, handle_tool_disable, handle_tool_enable, handle_tool_list,
};
use wiki::{
    handle_wiki_concepts, handle_wiki_fresh, handle_wiki_lint, handle_wiki_momentum,
    handle_wiki_planner, handle_wiki_prune, handle_wiki_refresh, handle_wiki_search,
    handle_wiki_seed, handle_wiki_stats, handle_wiki_status, handle_wiki_summary,
    handle_wiki_unknown,
};

use self::format::*;
pub use self::prompts::{brief_instruction, effort_instruction, plan_instruction};
use self::prompts::{
    build_advisor_prompt, build_bughunter_prompt, build_context_report, build_summary_prompt,
    build_version_info,
};
use self::suggest::suggest_slash_command;

/// Format the warning shown in-TUI when `/new`'s best-effort session save
/// fails. Mirrors the `main.rs` SIGTERM message so users see one idiom for
/// "save failed, here's what to check" regardless of where the save was
/// triggered.
pub(crate) fn format_new_save_error(err: &anyhow::Error, config_dir: &std::path::Path) -> String {
    format!(
        "Warning: {}",
        crate::session::storage::format_save_failure_tail(err, config_dir, ". ")
    )
}

#[derive(Debug)]
pub enum SlashCommand {
    Help,
    Clear,
    Compact(String),
    Edit,
    Model(String),
    Models,
    Sessions(usize),
    SessionDelete(String),
    SessionRename(String, String),
    SessionSearch(String),
    /// /sessions tree — render saved sessions as an ASCII fork tree
    SessionsTree,
    /// /resume \<id\> — switch to a different session by ID (prefix match)
    Resume(String),
    Permissions,
    Attach(String),
    Bug,
    Mcp,
    Config(String),
    Share(String),
    Quit,
    /// /memory — list all memory entries
    MemoryList,
    /// /memory add \<text\> — add a new memory entry
    MemoryAdd(String),
    /// /memory forget \<N\> — remove entry at 1-based index N
    MemoryForget(usize),
    /// /memory clear — clear all memory entries
    MemoryClear,
    /// /agent list
    AgentList,
    /// /agent \<name\>
    AgentRun(String),
    /// /eval [suite-name-or-path] — list suites when empty, run one when given
    Eval(String),
    /// /chain start \<file\> — start a chain from a YAML config file
    ChainStart(String),
    /// /chain status — show running chain status
    ChainStatus,
    /// /chain stop — send stop signal to running chain
    ChainStop,
    /// /chain add \<name\> \<role\> \[model\] \[`input_from`\] — enqueue a node for addition next cycle
    ChainAdd {
        name: String,
        role: String,
        model: String,
        input_from: Option<String>,
    },
    /// /chain remove \<name\> — enqueue a node for removal next cycle
    ChainRemove(String),
    /// /chain talk \<node\> <message...> — inject a message into a node's next prompt
    ChainTalk {
        node: String,
        message: String,
    },
    /// /chain model \<node\> \<model\> — swap the model of a node at runtime
    ChainModel {
        node: String,
        model: String,
    },
    /// /chain list — list available chain config files
    ChainList,
    /// /chain presets — list built-in chain presets
    ChainPresets,
    /// /chain pause — pause the running chain between cycles
    ChainPause,
    /// /chain resume — resume a paused chain
    ChainResume,
    /// /chain resume-from \<workspace\> — resume a chain from a saved checkpoint
    ChainResumeFrom(String),
    /// /chain init \[name\] — scaffold a starter chain config YAML
    ChainInit(String),
    /// /chain log \[cycle\] — show chain output from a cycle
    ChainLog(Option<usize>),
    /// /chain validate \<file\> — validate a chain config without running it
    ChainValidate(String),
    /// /chain metrics — show per-node timing and failure stats
    ChainMetrics,
    /// /chain (no args) or /chain help — show available chain subcommands
    ChainHelp,
    /// /fork \[N\] — fork conversation into a new session (optionally at turn N)
    Fork(Option<usize>),
    /// /undo — revert the last user turn
    Undo,
    /// /undo-files — revert the most recent batch of applied file changes
    UndoFiles,
    /// /retry — regenerate the last assistant response
    Retry,
    /// /copy — copy the last assistant response to clipboard
    Copy,
    /// /diff \[args\] — capture git diff as pending context for next message
    Diff(String),
    /// /add \<file\> — add file contents to pending context for next message
    Add(String),
    /// /doctor — run system diagnostics
    Doctor,
    /// /branch — list all branches, or /branch \<name\> to checkout/create
    Branch(String),
    /// /stats — show session statistics
    Stats,
    /// /review \[ref\] — capture git diff as a code-review prompt for the next message
    Review(String),
    /// /init — create DM.md in the current project directory
    Init,
    /// /commit — capture staged diff as an AI commit-message prompt
    Commit,
    /// /log \[N\] — capture recent git log as context (default 20 commits)
    Log(usize),
    /// /export \[path\] — export conversation to markdown (full, with tool details)
    /// /export clean \[path\] — export conversation (user/assistant/diffs only)
    Export(String),
    /// /bughunter \[focus\] — scan codebase for bugs; optional focus area
    BugHunter(String),
    /// /rename \<title\> — rename the current session
    Rename(String),
    /// /context — show current session context composition
    Context,
    /// /pr \[base\] — draft a PR description from commits ahead of base (default: main/master)
    Pr(String),
    /// /security-review \[ref\] — security-focused code review of a diff
    SecurityReview(String),
    /// /changelog [from \[to\]] — generate changelog from git history as pending context
    Changelog {
        from: String,
        to: String,
    },
    /// /blame \<file\> \[line\] — capture git blame output as context
    Blame {
        file: String,
        line: Option<usize>,
    },
    /// /conflicts — detect unresolved merge conflicts in working tree
    Conflicts,
    /// /stash [push|pop|list|drop|show] — git stash operations
    Stash(String),
    /// /advisor \[topic\] — get architectural/design advice; optional topic narrows scope
    Advisor(String),
    /// /template — list available templates
    /// /template \<name\> [args...] — load template and set as pending context
    Template(String),
    /// /todo — list session todos
    TodoList,
    /// /todo add [high|med|low] \<text\> — add a new todo (default priority: medium)
    TodoAdd {
        priority: String,
        content: String,
    },
    /// /todo done \<id\> — mark a todo as completed
    TodoDone(String),
    /// /todo wip \<id\> — mark a todo as in-progress
    TodoWip(String),
    /// /todo clear — remove all completed todos
    TodoClear,
    /// /todo scan — scan codebase for TODO/FIXME/HACK comments
    TodoScan,
    /// /effort [quick|normal|thorough] — set response depth (no arg = show current)
    Effort(String),
    /// /add-dir \<path\> — add all text files under a directory to context for the next message
    AddDir(String),
    /// /pin \<file\> — prepend file contents to every subsequent message
    /// /pin        — list currently pinned files
    Pin(String),
    /// /unpin \<file\> — remove a pinned file
    /// /unpin all  — clear all pinned files
    Unpin(String),
    /// /summary \[focus\] — queue a session summary prompt; optional focus narrows scope
    Summary(String),
    /// /brief — toggle brief output mode (concise, bullet-point responses)
    Brief,
    /// /plan — toggle plan mode (read-only exploration, write tools blocked)
    Plan,
    /// /files — list files currently in the conversation context (pinned + pending)
    Files,
    /// /new \[title\] — start a fresh conversation (clear history, new session); optional title
    New(String),
    /// /usage — show session token usage summary
    Usage,
    /// /version — show dm version, model, and host info
    Version,
    /// /tool list — list all tools and their status
    ToolList,
    /// /tool disable \<name\> — disable a tool for this session
    ToolDisable(String),
    /// /tool enable \<name\> — re-enable a disabled tool
    ToolEnable(String),
    /// /schedule — list all scheduled tasks
    ScheduleList,
    /// /schedule add \<cron5\> \<prompt\> — add a scheduled task
    ScheduleAdd(String),
    /// /schedule remove \<id\> — remove a scheduled task by ID
    ScheduleRemove(String),
    /// /search \<query\> — search conversation history for a string
    Search(String),
    /// /history \[N\] — show session activity log (default last 30 entries)
    History(usize),
    /// /kill — terminate the currently-running bash process without aborting the turn
    Kill,
    /// /test \[cmd\] — run tests, parse failures, and set as context for AI fixing
    Test(String),
    /// /lint \[cmd\] — run linter, parse warnings, and set as context for AI fixing
    Lint(String),
    /// /find \<glob\> — find files matching a glob pattern
    Find(String),
    /// /rg \<pattern\> \[path\] — search file contents with ripgrep
    Rg(String),
    /// /cd \<path\> — change working directory (no arg = show current)
    Cd(String),
    /// /changes — list files modified by AI in this session
    Changes,
    /// /tree \[path\] \[depth\] — show directory tree as context
    Tree {
        path: String,
        depth: usize,
    },
    /// /wiki status — summarize the project wiki
    WikiStatus,
    /// /wiki search \<query\> — case-insensitive substring search across
    /// wiki pages; empty query shows usage.
    WikiSearch(String),
    /// /wiki lint — report static-consistency findings for the wiki
    WikiLint,
    /// /wiki refresh — re-ingest entity pages whose source changed or
    /// whose `entity_kind` is missing for a `.rs` source.
    WikiRefresh,
    /// /wiki summary — regenerate `summaries/project.md` from entity pages.
    WikiSummary,
    /// /wiki concepts — detect shared dependencies across entity pages and
    /// generate `concepts/dep-*.md` pages for each dep with ≥ 3 consumers.
    WikiConcepts,
    /// /wiki momentum — surface hot paths and modules from the most
    /// recent ingest activity in `log.md`.
    WikiMomentum,
    /// /wiki fresh — show top-K entity/concept pages ranked by
    /// `last_updated` (mirrors the `\<wiki_fresh\>` block the model sees).
    WikiFresh,
    /// /wiki planner — render the orchestration planner brief so operators
    /// can inspect what chain planners see at cycle start.
    WikiPlanner,
    /// /wiki stats — show this-session wiki tool-call + drift-warning
    /// counters and the system-prompt wiki-snippet byte budget.
    WikiStats,
    /// /wiki prune \[\<N\>\] — drop oldest `synthesis/compact-*.md` pages
    /// beyond the cap. Default cap = `wiki::DEFAULT_COMPACT_KEEP`. `N=0`
    /// clears every compact-synthesis page.
    WikiPrune(usize),
    /// /wiki seed \[\<dir\>\] — recursively ingest `.rs` files under
    /// `<dir>` (default `src`) into entity pages. Operator-deliberate
    /// bulk seed; reuses `Wiki::ingest_file` so behavior matches the
    /// auto-ingest hook. `None` = default dir.
    WikiSeed(Option<String>),
    /// /wiki \<other\> — unknown subcommand.
    WikiUnknown(String),
    Unknown(String),
}

/// Result returned from executing a slash command (Info variant for memory ops).
pub enum SlashResult {
    /// Command completed normally; nothing special for the caller to do.
    Done,
    /// Display informational text to the user.
    Info(String),
    /// The last assistant response was written to a file for editing.
    EditInEditor { path: std::path::PathBuf },
    /// An error message to display to the user.
    Error(String),
}

/// All slash command names (without leading '/'), sorted alphabetically.
/// Used by the Tab autocomplete in the TUI input handler.
pub const SLASH_COMMAND_NAMES: &[&str] = &[
    "add",
    "add-dir",
    "advisor",
    "agent",
    "attach",
    "blame",
    "branch",
    "brief",
    "bug",
    "bughunter",
    "cd",
    "chain",
    "changelog",
    "changes",
    "clear",
    "commit",
    "compact",
    "config",
    "conflicts",
    "context",
    "copy",
    "diff",
    "doctor",
    "edit",
    "effort",
    "eval",
    "export",
    "files",
    "find",
    "help",
    "init",
    "log",
    "history",
    "kill",
    "lint",
    "mcp",
    "memory",
    "model",
    "models",
    "new",
    "permissions",
    "pin",
    "plan",
    "pr",
    "quit",
    "rename",
    "resume",
    "retry",
    "review",
    "rg",
    "schedule",
    "search",
    "security-review",
    "sessions",
    "share",
    "stash",
    "stats",
    "summary",
    "template",
    "test",
    "todo",
    "tool",
    "tree",
    "undo",
    "undo-files",
    "unpin",
    "usage",
    "version",
    "wiki",
];

pub const CHAIN_SUBCOMMAND_NAMES: &[&str] = &[
    "add",
    "help",
    "init",
    "list",
    "log",
    "metrics",
    "model",
    "pause",
    "remove",
    "resume",
    "resume-from",
    "start",
    "status",
    "stop",
    "talk",
    "validate",
];

pub fn parse(input: &str) -> Option<SlashCommand> {
    if !input.starts_with('/') {
        return None;
    }
    let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
    Some(match parts[0].trim() {
        "help" => SlashCommand::Help,
        "clear" => SlashCommand::Clear,
        "compact" => SlashCommand::Compact(parts.get(1).unwrap_or(&"").trim().to_string()),
        "edit" => SlashCommand::Edit,
        "model" => SlashCommand::Model(parts.get(1).unwrap_or(&"").trim().to_string()),
        "models" => SlashCommand::Models,
        "sessions" => {
            let remainder = parts.get(1).unwrap_or(&"").trim();
            if let Some(query) = remainder.strip_prefix("search ") {
                SlashCommand::SessionSearch(query.trim().to_string())
            } else if remainder == "tree" {
                SlashCommand::SessionsTree
            } else {
                let n = remainder.parse().unwrap_or(10);
                SlashCommand::Sessions(n)
            }
        }
        "session" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if let Some(rest) = sub.strip_prefix("delete ") {
                SlashCommand::SessionDelete(rest.trim().to_string())
            } else if let Some(rest) = sub.strip_prefix("rename ") {
                let mut parts2 = rest.splitn(2, ' ');
                let id = parts2.next().unwrap_or("").trim().to_string();
                let name = parts2.next().unwrap_or("").trim().to_string();
                SlashCommand::SessionRename(id, name)
            } else {
                SlashCommand::Unknown("session".to_string())
            }
        }
        "resume" => SlashCommand::Resume(parts.get(1).unwrap_or(&"").trim().to_string()),
        "permissions" => SlashCommand::Permissions,
        "attach" => SlashCommand::Attach(parts.get(1).unwrap_or(&"").trim().to_string()),
        "bug" => SlashCommand::Bug,
        "mcp" => SlashCommand::Mcp,
        "config" => SlashCommand::Config(parts.get(1).unwrap_or(&"").trim().to_string()),
        "share" => SlashCommand::Share(parts.get(1).unwrap_or(&"").trim().to_string()),
        "quit" | "exit" => SlashCommand::Quit,
        "memory" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if sub.is_empty() {
                SlashCommand::MemoryList
            } else if sub == "clear" {
                SlashCommand::MemoryClear
            } else if let Some(text) = sub.strip_prefix("add ") {
                SlashCommand::MemoryAdd(text.trim().to_string())
            } else if let Some(rest) = sub.strip_prefix("forget ") {
                let n: usize = rest.trim().parse().unwrap_or(0);
                if n == 0 {
                    SlashCommand::Unknown("memory forget (index must be ≥1)".to_string())
                } else {
                    SlashCommand::MemoryForget(n)
                }
            } else {
                SlashCommand::Unknown(format!("memory {}", sub))
            }
        }
        "wiki" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if sub.is_empty() || sub == "status" {
                SlashCommand::WikiStatus
            } else if sub == "lint" {
                SlashCommand::WikiLint
            } else if sub == "refresh" {
                SlashCommand::WikiRefresh
            } else if sub == "summary" {
                SlashCommand::WikiSummary
            } else if sub == "concepts" {
                SlashCommand::WikiConcepts
            } else if sub == "momentum" {
                SlashCommand::WikiMomentum
            } else if sub == "fresh" {
                SlashCommand::WikiFresh
            } else if sub == "planner" {
                SlashCommand::WikiPlanner
            } else if sub == "stats" {
                SlashCommand::WikiStats
            } else if sub == "search" || sub.starts_with("search ") || sub.starts_with("search\t") {
                let q = sub["search".len()..].trim_start();
                SlashCommand::WikiSearch(q.to_string())
            } else if sub == "prune" {
                SlashCommand::WikiPrune(crate::wiki::DEFAULT_COMPACT_KEEP)
            } else if let Some(rest) = sub.strip_prefix("prune ") {
                match rest.trim().parse::<usize>() {
                    Ok(n) => SlashCommand::WikiPrune(n),
                    Err(_) => SlashCommand::WikiUnknown(sub.to_string()),
                }
            } else if sub == "seed" {
                SlashCommand::WikiSeed(None)
            } else if let Some(rest) = sub.strip_prefix("seed ") {
                let dir = rest.trim();
                if dir.is_empty() {
                    // Treat trailing-only whitespace as default-dir, mirroring
                    // the behavior of `/wiki  ` → `WikiStatus`.
                    SlashCommand::WikiSeed(None)
                } else {
                    SlashCommand::WikiSeed(Some(dir.to_string()))
                }
            } else {
                SlashCommand::WikiUnknown(sub.to_string())
            }
        }
        "agent" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if sub == "list" || sub.is_empty() {
                SlashCommand::AgentList
            } else {
                SlashCommand::AgentRun(sub.to_string())
            }
        }
        "eval" => SlashCommand::Eval(parts.get(1).unwrap_or(&"").trim().to_string()),
        "chain" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            let mut sub_parts = sub.splitn(2, ' ');
            match sub_parts.next().unwrap_or("") {
                "start" => {
                    let file = sub_parts.next().unwrap_or("").trim().to_string();
                    SlashCommand::ChainStart(file)
                }
                "status" => SlashCommand::ChainStatus,
                "stop" => SlashCommand::ChainStop,
                "list" | "ls" => SlashCommand::ChainList,
                "presets" => SlashCommand::ChainPresets,
                "add" => {
                    let rest = sub_parts.next().unwrap_or("").trim();
                    let mut p = rest.splitn(4, ' ');
                    let name = p.next().unwrap_or("").trim().to_string();
                    let role = p.next().unwrap_or("").trim().to_string();
                    let model = p.next().unwrap_or("").trim().to_string();
                    let input_from = p
                        .next()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    if name.is_empty() || role.is_empty() {
                        SlashCommand::Unknown(
                            "chain add <name> <role> [model] [input_from]".to_string(),
                        )
                    } else {
                        SlashCommand::ChainAdd {
                            name,
                            role,
                            model,
                            input_from,
                        }
                    }
                }
                "remove" => {
                    let name = sub_parts.next().unwrap_or("").trim().to_string();
                    if name.is_empty() {
                        SlashCommand::Unknown("chain remove <name>".to_string())
                    } else {
                        SlashCommand::ChainRemove(name)
                    }
                }
                "talk" => {
                    let rest = sub_parts.next().unwrap_or("").trim();
                    let mut talk_parts = rest.splitn(2, ' ');
                    let node = talk_parts.next().unwrap_or("").trim().to_string();
                    let message = talk_parts.next().unwrap_or("").trim().to_string();
                    if node.is_empty() || message.is_empty() {
                        SlashCommand::Unknown("chain talk <node> <message>".to_string())
                    } else {
                        SlashCommand::ChainTalk { node, message }
                    }
                }
                "model" => {
                    let rest = sub_parts.next().unwrap_or("").trim();
                    let mut p = rest.splitn(2, ' ');
                    let node = p.next().unwrap_or("").trim().to_string();
                    let model = p.next().unwrap_or("").trim().to_string();
                    if node.is_empty() || model.is_empty() {
                        SlashCommand::Unknown("chain model <node> <model>".to_string())
                    } else {
                        SlashCommand::ChainModel { node, model }
                    }
                }
                "init" => {
                    let name = sub_parts.next().unwrap_or("").trim().to_string();
                    let name = if name.is_empty() {
                        "my-chain".to_string()
                    } else {
                        name
                    };
                    SlashCommand::ChainInit(name)
                }
                "log" => {
                    let arg = sub_parts.next().unwrap_or("").trim().to_string();
                    let cycle = if arg.is_empty() {
                        None
                    } else {
                        arg.parse::<usize>().ok()
                    };
                    SlashCommand::ChainLog(cycle)
                }
                "validate" | "check" => {
                    let file = sub_parts.next().unwrap_or("").trim().to_string();
                    SlashCommand::ChainValidate(file)
                }
                "metrics" => SlashCommand::ChainMetrics,
                "pause" => SlashCommand::ChainPause,
                "resume" => SlashCommand::ChainResume,
                "resume-from" => {
                    let workspace = sub_parts.next().unwrap_or("").trim().to_string();
                    if workspace.is_empty() {
                        SlashCommand::Unknown("chain resume-from <workspace>".to_string())
                    } else {
                        SlashCommand::ChainResumeFrom(workspace)
                    }
                }
                "" | "help" => SlashCommand::ChainHelp,
                _ => SlashCommand::Unknown(format!("chain {}", sub)),
            }
        }
        "undo" | "rewind" => SlashCommand::Undo,
        "undo-files" => SlashCommand::UndoFiles,
        "retry" | "regenerate" => SlashCommand::Retry,
        "fork" => {
            let turn = parts.get(1).and_then(|p| p.trim().parse::<usize>().ok());
            SlashCommand::Fork(turn)
        }
        "copy" => SlashCommand::Copy,
        "diff" => SlashCommand::Diff(parts[1..].join(" ").trim().to_string()),
        "add" => SlashCommand::Add(parts[1..].join(" ").trim().to_string()),
        "doctor" => SlashCommand::Doctor,
        "branch" => SlashCommand::Branch(parts.get(1).unwrap_or(&"").trim().to_string()),
        "stats" => SlashCommand::Stats,
        "review" => SlashCommand::Review(parts.get(1).unwrap_or(&"").trim().to_string()),
        "init" => SlashCommand::Init,
        "commit" => SlashCommand::Commit,
        "log" => {
            let n = parts
                .get(1)
                .and_then(|p| p.trim().parse::<usize>().ok())
                .unwrap_or(20);
            SlashCommand::Log(n)
        }
        "export" => SlashCommand::Export(parts.get(1).unwrap_or(&"").trim().to_string()),
        "bughunter" => SlashCommand::BugHunter(parts.get(1).unwrap_or(&"").trim().to_string()),
        "rename" => SlashCommand::Rename(parts[1..].join(" ").trim().to_string()),
        "context" => SlashCommand::Context,
        "pr" => SlashCommand::Pr(parts.get(1).unwrap_or(&"").trim().to_string()),
        "security-review" | "secreview" => {
            SlashCommand::SecurityReview(parts.get(1).unwrap_or(&"").trim().to_string())
        }
        "changelog" => {
            let args: Vec<&str> = parts.get(1).unwrap_or(&"").split_whitespace().collect();
            let from = args.first().unwrap_or(&"").to_string();
            let to = args.get(1).unwrap_or(&"").to_string();
            SlashCommand::Changelog { from, to }
        }
        "blame" => {
            let args: Vec<&str> = parts.get(1).unwrap_or(&"").split_whitespace().collect();
            let file = args.first().unwrap_or(&"").to_string();
            let line = args.get(1).and_then(|p| p.parse::<usize>().ok());
            SlashCommand::Blame { file, line }
        }
        "conflicts" => SlashCommand::Conflicts,
        "stash" => SlashCommand::Stash(parts[1..].join(" ").trim().to_string()),
        "advisor" => SlashCommand::Advisor(parts.get(1).unwrap_or(&"").trim().to_string()),
        "effort" => SlashCommand::Effort(parts.get(1).unwrap_or(&"").trim().to_lowercase()),
        "add-dir" => SlashCommand::AddDir(parts.get(1).unwrap_or(&"").trim().to_string()),
        "pin" => SlashCommand::Pin(parts.get(1).unwrap_or(&"").trim().to_string()),
        "unpin" => SlashCommand::Unpin(parts.get(1).unwrap_or(&"").trim().to_string()),
        "template" => SlashCommand::Template(parts.get(1).unwrap_or(&"").trim().to_string()),
        "todo" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if sub.is_empty() {
                SlashCommand::TodoList
            } else if sub == "clear" {
                SlashCommand::TodoClear
            } else if sub == "scan" {
                SlashCommand::TodoScan
            } else if let Some(rest) = sub.strip_prefix("done ") {
                SlashCommand::TodoDone(rest.trim().to_string())
            } else if let Some(rest) = sub.strip_prefix("wip ") {
                SlashCommand::TodoWip(rest.trim().to_string())
            } else if let Some(rest) = sub.strip_prefix("add ") {
                // Optionally starts with a priority keyword: high, med, medium, low
                let rest = rest.trim();
                let mut words = rest.splitn(2, ' ');
                let first = words.next().unwrap_or("").to_lowercase();
                let (priority, content) = match first.as_str() {
                    "high" | "med" | "medium" | "low" => {
                        let p = if first == "med" {
                            "medium".to_string()
                        } else {
                            first
                        };
                        let c = words.next().unwrap_or("").trim().to_string();
                        (p, c)
                    }
                    _ => ("medium".to_string(), rest.to_string()),
                };
                if content.is_empty() {
                    SlashCommand::Unknown("todo add: text is required".to_string())
                } else {
                    SlashCommand::TodoAdd { priority, content }
                }
            } else {
                SlashCommand::Unknown(format!("todo {}", sub))
            }
        }
        "summary" => SlashCommand::Summary(parts.get(1).unwrap_or(&"").trim().to_string()),
        "brief" => SlashCommand::Brief,
        "plan" => SlashCommand::Plan,
        "files" => SlashCommand::Files,
        "new" => SlashCommand::New(parts.get(1).unwrap_or(&"").trim().to_string()),
        "usage" => SlashCommand::Usage,
        "version" => SlashCommand::Version,
        "search" | "grep" => SlashCommand::Search(parts.get(1).unwrap_or(&"").trim().to_string()),
        "history" => {
            let n = parts
                .get(1)
                .and_then(|p| p.trim().parse::<usize>().ok())
                .unwrap_or(30);
            SlashCommand::History(n)
        }
        "kill" => SlashCommand::Kill,
        "test" => SlashCommand::Test(parts.get(1).unwrap_or(&"").trim().to_string()),
        "lint" => SlashCommand::Lint(parts.get(1).unwrap_or(&"").trim().to_string()),
        "find" => SlashCommand::Find(parts.get(1).unwrap_or(&"").trim().to_string()),
        "rg" => SlashCommand::Rg(parts.get(1).unwrap_or(&"").trim().to_string()),
        "cd" => SlashCommand::Cd(parts.get(1).unwrap_or(&"").trim().to_string()),
        "changes" => SlashCommand::Changes,
        "tree" => {
            let args = parts.get(1).unwrap_or(&"").trim();
            let words: Vec<&str> = args.split_whitespace().collect();
            let (path, depth) = match words.len() {
                0 => (String::new(), 3),
                1 => {
                    if let Ok(d) = words[0].parse::<usize>() {
                        (String::new(), d)
                    } else {
                        (words[0].to_string(), 3)
                    }
                }
                _ => {
                    // Reached only when words.len() >= 2, so `last` is always
                    // present — but we read it without panicking regardless.
                    let last = words.last().copied().unwrap_or("");
                    if let Ok(d) = last.parse::<usize>() {
                        (words[..words.len() - 1].join(" "), d)
                    } else {
                        (args.to_string(), 3)
                    }
                }
            };
            SlashCommand::Tree { path, depth }
        }
        "tool" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if let Some(name) = sub.strip_prefix("disable ") {
                SlashCommand::ToolDisable(name.trim().to_string())
            } else if let Some(name) = sub.strip_prefix("enable ") {
                SlashCommand::ToolEnable(name.trim().to_string())
            } else if sub == "list" || sub.is_empty() {
                SlashCommand::ToolList
            } else {
                SlashCommand::Unknown(format!("tool {}", sub))
            }
        }
        "schedule" => {
            let sub = parts.get(1).unwrap_or(&"").trim();
            if sub.is_empty() {
                SlashCommand::ScheduleList
            } else if let Some(rest) = sub.strip_prefix("add ") {
                SlashCommand::ScheduleAdd(rest.trim().to_string())
            } else if let Some(rest) = sub.strip_prefix("remove ") {
                SlashCommand::ScheduleRemove(rest.trim().to_string())
            } else if sub == "list" {
                SlashCommand::ScheduleList
            } else {
                SlashCommand::Unknown(format!("schedule {sub}"))
            }
        }
        other => SlashCommand::Unknown(other.to_string()),
    })
}

/// Execute a slash command, mutating App state and/or signalling the agent.
/// Always returns after execution — the caller should NOT send the text to the agent.
pub async fn execute(
    cmd: SlashCommand,
    app: &mut App,
    client: &OllamaClient,
    user_tx: &tokio::sync::mpsc::Sender<String>,
) -> SlashResult {
    match cmd {
        SlashCommand::Help => {
            app.mode = crate::tui::app::Mode::HelpOverlay;
        }
        SlashCommand::Clear => {
            app.entries.clear();
            app.streaming_partial = None;
            app.push_entry(EntryKind::SystemInfo, "Conversation cleared.".to_string());
            user_tx.send("/clear".to_string()).await.ok();
        }
        SlashCommand::Compact(mode) => {
            let turn_count: usize = app
                .entries
                .iter()
                .filter(|e| matches!(e.kind, EntryKind::UserMessage))
                .count();

            let ctx_pct: usize = if let Some((used, limit)) = app.ctx_usage {
                if limit == 0 {
                    0
                } else {
                    used * 100 / limit
                }
            } else {
                0
            };

            if mode == "preview" {
                let total_entries = app.entries.len();
                let keep_tail = 8usize;
                let to_remove = total_entries.saturating_sub(keep_tail + 1);
                let chars_estimate: usize = app
                    .entries
                    .iter()
                    .take(to_remove)
                    .map(|e| e.content.len())
                    .sum();
                let token_estimate = chars_estimate / 4;
                return SlashResult::Info(format!(
                    "Compact preview:\n  Context:     {}% full\n  Turns:       {}\n  Will remove: {} messages (~{} tokens)\n  Will keep:   last {} messages + system prompt\n\nRun /compact to execute, or /compact force to skip threshold checks.",
                    ctx_pct, turn_count, to_remove, token_estimate, keep_tail
                ));
            }

            let force = mode == "force";
            if !force {
                if turn_count < 5 {
                    return SlashResult::Info(format!(
                        "Nothing to compact — context is only {}% full ({} turns). Use /compact force to override.",
                        ctx_pct, turn_count
                    ));
                }
                if ctx_pct < 30 {
                    return SlashResult::Info(format!(
                        "Nothing to compact — context is only {}% full ({} turns). Use /compact force to override.",
                        ctx_pct, turn_count
                    ));
                }
            }

            app.push_entry(EntryKind::SystemInfo, "Compacting context…".to_string());
            user_tx.send("/compact/smart".to_string()).await.ok();
        }
        SlashCommand::Model(name) if !name.is_empty() => {
            app.model.clone_from(&name);
            app.push_entry(
                EntryKind::SystemInfo,
                format!("Switched to model: {}", name),
            );
            user_tx.send(format!("/model {}", name)).await.ok();
        }
        SlashCommand::Model(_) => {
            app.push_entry(EntryKind::SystemInfo, "Usage: /model <name>".to_string());
        }
        SlashCommand::Models => match client.list_models().await {
            Ok(models) if models.is_empty() => {
                app.push_entry(
                    EntryKind::SystemInfo,
                    "No models found. Try: ollama pull gemma4:26b".to_string(),
                );
            }
            Ok(models) => {
                app.available_models = models.iter().map(|m| m.name.clone()).collect();
                let list = models
                    .iter()
                    .map(|m| format!("  • {}", m.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                app.push_entry(
                    EntryKind::SystemInfo,
                    format!("Available models:\n{}", list),
                );
            }
            Err(e) => {
                app.push_entry(
                    EntryKind::SystemInfo,
                    format!("Error listing models: {}", e),
                );
            }
        },
        SlashCommand::Sessions(limit) => return handle_sessions(limit, app),
        SlashCommand::SessionSearch(query) => {
            match session_storage::search_sessions(&app.config_dir, &query) {
                Ok(matches) if matches.is_empty() => {
                    app.push_entry(
                        EntryKind::SystemInfo,
                        format!("No sessions matching \"{}\".", query),
                    );
                }
                Ok(matches) => {
                    let header = format!(
                        "{} session{} matching \"{}\":",
                        matches.len(),
                        if matches.len() == 1 { "" } else { "s" },
                        query
                    );
                    let lines: Vec<String> = matches
                        .iter()
                        .map(|m| {
                            let short = short_id(&m.session.id);
                            let date = m.session.updated_at.format("%Y-%m-%d").to_string();
                            let title = m
                                .session
                                .title
                                .as_deref()
                                .filter(|t| !t.is_empty())
                                .unwrap_or("(no title)");
                            format!("  [{}] \"{}\" — {}  …{}…", short, title, date, m.snippet)
                        })
                        .collect();
                    app.push_entry(
                        EntryKind::SystemInfo,
                        format!("{}\n{}", header, lines.join("\n")),
                    );
                }
                Err(e) => {
                    app.push_entry(
                        EntryKind::SystemInfo,
                        format!("Error searching sessions: {}", e),
                    );
                }
            }
        }
        SlashCommand::SessionsTree => return handle_sessions_tree(app),
        SlashCommand::SessionDelete(id) => match session_storage::delete(&app.config_dir, &id) {
            Ok(()) => app.push_entry(EntryKind::SystemInfo, format!("Session {} deleted.", id)),
            Err(e) => app.push_entry(EntryKind::SystemInfo, format!("Error: {}", e)),
        },
        SlashCommand::SessionRename(id, name) => {
            match session_storage::update_title(&app.config_dir, &id, &name) {
                Ok(()) => app.push_entry(
                    EntryKind::SystemInfo,
                    format!("Renamed session {} to \"{}\"", short_id(&id), name),
                ),
                Err(e) => app.push_entry(EntryKind::SystemInfo, format!("Error: {}", e)),
            }
        }
        SlashCommand::Resume(id) => return handle_resume(id, app, user_tx).await,
        SlashCommand::Permissions => return handle_permissions(user_tx).await,
        SlashCommand::Attach(path) => {
            if path.is_empty() {
                app.push_entry(
                    EntryKind::SystemInfo,
                    "Usage: /attach <path>  (png, jpg, jpeg, gif, webp)".to_string(),
                );
            } else {
                match load_image_base64(&path) {
                    Ok((filename, data)) => {
                        app.pending_images.push((filename.clone(), data));
                        app.push_entry(
                            EntryKind::SystemInfo,
                            format!(
                                "Attached: {} ({} image(s) pending)",
                                filename,
                                app.pending_images.len()
                            ),
                        );
                    }
                    Err(e) => {
                        app.push_entry(
                            EntryKind::SystemInfo,
                            format!("Error attaching '{}': {}", path, e),
                        );
                    }
                }
            }
        }
        SlashCommand::Bug => {
            let version = env!("CARGO_PKG_VERSION");
            let os = std::env::consts::OS;
            let title = urlencoding::encode(&format!("Bug report: dm v{}", version)).into_owned();
            let body = urlencoding::encode(&format!(
                "## Environment\n\
                - dm version: {}\n\
                - OS: {}\n\
                - Model: {}\n\
                - Session: {}\n\n\
                ## What happened\n\n\
                <!-- Describe the bug -->\n\n\
                ## Steps to reproduce\n\n\
                1. \n\
                2. \n\n\
                ## Expected behavior\n\n\
                <!-- What should have happened -->",
                version,
                os,
                app.model,
                short_id(&app.session_id)
            ))
            .into_owned();
            let url = format!(
                "https://github.com/base-reality-ai/dark-matter/issues/new?title={}&body={}",
                title, body
            );
            if open_url(&url) {
                app.push_entry(
                    EntryKind::SystemInfo,
                    "Opened bug report in browser.".to_string(),
                );
            } else {
                app.push_entry(EntryKind::SystemInfo, format!("File a bug at:\n{}", url));
            }
        }
        SlashCommand::Mcp => return handle_mcp(app),
        SlashCommand::Config(arg) => {
            let parts: Vec<&str> = arg.splitn(3, ' ').filter(|s| !s.is_empty()).collect();
            if parts.is_empty() {
                // /config — show current settings.json
                let settings_path = app.config_dir.join("settings.json");
                let content = std::fs::read_to_string(&settings_path)
                    .unwrap_or_else(|_| "(no settings.json found)".to_string());
                app.push_entry(
                    EntryKind::SystemInfo,
                    format!("Config: {}\n{}", settings_path.display(), content),
                );
            } else if parts[0] == "bell" {
                if parts.len() < 2 {
                    app.push_entry(
                        EntryKind::SystemInfo,
                        format!(
                            "bell = {} (use /config bell true|false)",
                            app.bell_on_complete
                        ),
                    );
                } else {
                    match parts[1] {
                        "true" | "on" | "1" => {
                            app.bell_on_complete = true;
                            app.push_entry(EntryKind::SystemInfo, "Bell enabled.".into());
                        }
                        "false" | "off" | "0" => {
                            app.bell_on_complete = false;
                            app.push_entry(EntryKind::SystemInfo, "Bell disabled.".into());
                        }
                        _ => {
                            app.push_entry(
                                EntryKind::SystemInfo,
                                "Usage: /config bell true|false".into(),
                            );
                        }
                    }
                }
            } else if parts[0] == "set" && parts.len() == 3 {
                // /config set <key> <value>
                let key = parts[1];
                let value = parts[2];
                let settings_path = app.config_dir.join("settings.json");
                let mut settings: serde_json::Value = std::fs::read_to_string(&settings_path)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_else(|| serde_json::json!({}));
                settings[key] = serde_json::Value::String(value.to_string());
                match std::fs::write(
                    &settings_path,
                    serde_json::to_string_pretty(&settings).unwrap_or_default(),
                ) {
                    Ok(()) => app.push_entry(
                        EntryKind::SystemInfo,
                        format!("Set {} = {} (saved to settings.json)", key, value),
                    ),
                    Err(e) => app.push_entry(
                        EntryKind::SystemInfo,
                        format!("Error saving settings: {}", e),
                    ),
                }
            } else {
                app.push_entry(
                    EntryKind::SystemInfo,
                    "Usage: /config  OR  /config set <key> <value>".to_string(),
                );
            }
        }
        SlashCommand::Share(path) => {
            match crate::session::storage::load(&app.config_dir, &app.session_id) {
                Ok(sess) => {
                    let use_md = path.ends_with(".md");
                    let (content_result, default_ext) = if use_md {
                        (Ok(crate::share::render_session_markdown(&sess)), "md")
                    } else {
                        (crate::share::render_session(&sess), "html")
                    };
                    match content_result {
                        Ok(content) => {
                            let dest = if path.is_empty() {
                                format!("dm-session-{}.{}", short_id(&app.session_id), default_ext)
                            } else {
                                path
                            };
                            match std::fs::write(&dest, &content) {
                                Ok(_) => app.push_entry(
                                    EntryKind::SystemInfo,
                                    format!("Exported: {}", dest),
                                ),
                                Err(e) => app.push_entry(
                                    EntryKind::SystemInfo,
                                    format!("Export failed: {}", e),
                                ),
                            }
                        }
                        Err(e) => {
                            app.push_entry(EntryKind::SystemInfo, format!("Render failed: {}", e))
                        }
                    }
                }
                Err(e) => {
                    app.push_entry(EntryKind::SystemInfo, format!("Session load failed: {}", e))
                }
            }
        }
        SlashCommand::Edit => {
            // Find the last assistant text entry
            let last_assistant = app
                .entries
                .iter()
                .rev()
                .find(|e| e.kind == EntryKind::AssistantMessage)
                .map(|e| e.content.clone());

            match last_assistant {
                None => {
                    return SlashResult::Error("No assistant response to edit.".into());
                }
                Some(content) => {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let dm_dir = app.config_dir.clone();
                    let filename = format!("edit_{}.md", ts);
                    let temp_path = dm_dir.join(&filename);
                    match std::fs::write(&temp_path, &content) {
                        Ok(()) => {
                            return SlashResult::EditInEditor { path: temp_path };
                        }
                        Err(e) => {
                            return SlashResult::Error(format!("Failed to write edit file: {}", e));
                        }
                    }
                }
            }
        }
        SlashCommand::Undo => {
            user_tx.send("/undo".to_string()).await.ok();
        }
        SlashCommand::Retry => {
            let last_user_msg = app
                .entries
                .iter()
                .rev()
                .find(|e| matches!(e.kind, EntryKind::UserMessage))
                .map(|e| e.content.clone());
            match last_user_msg {
                Some(msg) => {
                    user_tx.send("/undo".to_string()).await.ok();
                    user_tx.send(msg).await.ok();
                    return SlashResult::Info("Regenerating last response...".into());
                }
                None => {
                    return SlashResult::Error("No previous message to retry.".into());
                }
            }
        }
        SlashCommand::UndoFiles => match app.file_undo_history.pop() {
            Some(changes) => {
                let paths: Vec<String> = changes
                    .iter()
                    .map(|c| c.path.display().to_string())
                    .collect();
                let (reverted, skipped, errors) = crate::changeset::revert_all(&changes).await;
                if errors.is_empty() {
                    return SlashResult::Info(format!(
                        "Reverted {} file{}: {}",
                        reverted,
                        if reverted == 1 { "" } else { "s" },
                        paths.join(", ")
                    ));
                } else if skipped > 0 && reverted == 0 {
                    return SlashResult::Error(format!(
                        "All files were modified externally, nothing reverted: {}",
                        errors.join("; ")
                    ));
                }
                return SlashResult::Error(format!(
                    "Reverted {}/{} files. Errors: {}",
                    reverted,
                    changes.len(),
                    errors.join("; ")
                ));
            }
            None => {
                return SlashResult::Info("No file changes to undo.".to_string());
            }
        },
        SlashCommand::Diff(arg) => {
            let mut cmd = tokio::process::Command::new("git");
            cmd.arg("diff");
            if arg.is_empty() {
                cmd.arg("HEAD");
            } else {
                cmd.arg(&arg);
            }
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                cmd.output(),
            )
            .await
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
                        "git diff timed out. Try: re-run, or check repo health with `git status`."
                            .into(),
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
                return SlashResult::Info("No changes to show.".to_string());
            }

            let truncated = crate::git::safe_truncate(&diff, 12_000);
            let line_count = truncated.lines().count();
            app.pending_context = Some(format!("```diff\n{}\n```", truncated));
            return SlashResult::Info(format!(
                "Git diff ready ({} lines) — type your question to analyze it",
                line_count
            ));
        }
        SlashCommand::Add(path) => {
            if path.is_empty() {
                return SlashResult::Info(
                    "Usage: /add <file> — add file contents to context".to_string(),
                );
            }
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    return SlashResult::Info(format!("Error reading '{}': {}", path, e));
                }
            };
            let truncated = crate::git::safe_truncate(&content, 8_000).to_string();
            let line_count = truncated.lines().count();
            let lang = std::path::Path::new(&path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let formatted = format!("File: {}\n```{}\n{}\n```", path, lang, truncated);
            app.pending_context = Some(match app.pending_context.take() {
                Some(existing) => format!("{}\n\n{}", existing, formatted),
                None => formatted,
            });
            return SlashResult::Info(format!(
                "Added '{}' ({} lines) to context — type your question",
                path, line_count
            ));
        }
        SlashCommand::AddDir(dir_path) => {
            if dir_path.is_empty() {
                return SlashResult::Info(
                    "Usage: /add-dir <directory> — add all text files in a directory to context"
                        .to_string(),
                );
            }
            const MAX_DIR_BYTES: usize = 24_000;
            match collect_dir_context(&dir_path, MAX_DIR_BYTES) {
                Err(msg) => return SlashResult::Error(msg),
                Ok((context, file_count)) => {
                    let header = format!("## Directory context: {}\n\n", dir_path);
                    let full = format!("{}{}", header, context);
                    let line_count = full.lines().count();
                    app.pending_context = Some(match app.pending_context.take() {
                        Some(existing) => format!("{}\n\n{}", existing, full),
                        None => full,
                    });
                    return SlashResult::Info(format!(
                        "Added {} file(s) from '{}' ({} lines) to context — type your question",
                        file_count, dir_path, line_count
                    ));
                }
            }
        }
        SlashCommand::Doctor => {
            let config = crate::config::Config {
                host: app.host.clone(),
                host_is_default: false,
                model: app.model.clone(),
                model_is_default: false,
                tool_model: None,
                embed_model: "nomic-embed-text".to_string(),
                config_dir: app.config_dir.clone(),
                routing: None,
                aliases: std::collections::HashMap::new(),
                max_retries: 3,
                retry_delay_ms: 1000,
                max_retry_delay_ms: 30_000,
                fallback_model: None,
                snapshot_interval_secs: 300,
                idle_timeout_secs: 7200,
            };
            let identity = crate::identity::load_for_cwd();
            let output = crate::doctor::run_doctor_capture(client, &config, &identity).await;
            return SlashResult::Info(output);
        }
        SlashCommand::Copy => {
            app.copy_last_response();
            return SlashResult::Info("Copied to clipboard".to_string());
        }
        SlashCommand::Quit => {
            app.should_quit = true;
        }
        SlashCommand::MemoryList => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let phash = crate::index::project_hash(&cwd);
            let mem_path = match crate::memory::ProjectMemory::file_path(&app.config_dir, &phash) {
                Ok(p) => p,
                Err(e) => return SlashResult::Error(format!("Invalid project hash: {}", e)),
            };
            match crate::memory::ProjectMemory::load(&app.config_dir, &phash) {
                Ok(mem) => {
                    if mem.entries.is_empty() {
                        return SlashResult::Info(format!(
                            "Project memory (0 entries, {}):\n(empty)",
                            mem_path.display()
                        ));
                    }
                    let mut lines = format!(
                        "Project memory ({} entries, {}):",
                        mem.entries.len(),
                        mem_path.display()
                    );
                    for (i, entry) in mem.entries.iter().enumerate() {
                        let ts = entry.timestamp.format("%Y-%m-%d");
                        write!(lines, "\n[{}] {} — {}", i + 1, ts, entry.summary)
                            .expect("write to String never fails");
                    }
                    return SlashResult::Info(lines);
                }
                Err(e) => {
                    return SlashResult::Error(format!("Error loading memory: {}", e));
                }
            }
        }
        SlashCommand::MemoryAdd(text) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let phash = crate::index::project_hash(&cwd);
            match crate::memory::ProjectMemory::load(&app.config_dir, &phash) {
                Ok(mut mem) => {
                    mem.append(&text);
                    match mem.save(&app.config_dir, &phash) {
                        Ok(()) => return SlashResult::Info("Added entry.".to_string()),
                        Err(e) => return SlashResult::Error(format!("Error saving memory: {}", e)),
                    }
                }
                Err(e) => return SlashResult::Error(format!("Error loading memory: {}", e)),
            }
        }
        SlashCommand::MemoryForget(n) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let phash = crate::index::project_hash(&cwd);
            match crate::memory::ProjectMemory::load(&app.config_dir, &phash) {
                Ok(mut mem) => {
                    if n == 0 || n > mem.entries.len() {
                        return SlashResult::Error("No entry at that index.".into());
                    }
                    mem.entries.remove(n - 1);
                    match mem.save(&app.config_dir, &phash) {
                        Ok(()) => return SlashResult::Info(format!("Removed entry {}.", n)),
                        Err(e) => return SlashResult::Error(format!("Error saving memory: {}", e)),
                    }
                }
                Err(e) => return SlashResult::Error(format!("Error loading memory: {}", e)),
            }
        }
        SlashCommand::MemoryClear => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let phash = crate::index::project_hash(&cwd);
            match crate::memory::ProjectMemory::load(&app.config_dir, &phash) {
                Ok(mut mem) => {
                    mem.clear();
                    match mem.save(&app.config_dir, &phash) {
                        Ok(()) => return SlashResult::Info("Memory cleared.".to_string()),
                        Err(e) => return SlashResult::Error(format!("Error saving memory: {}", e)),
                    }
                }
                Err(e) => return SlashResult::Error(format!("Error loading memory: {}", e)),
            }
        }
        SlashCommand::AgentList => {
            let agents = crate::agents::AgentConfig::list(&app.config_dir);
            if agents.is_empty() {
                return SlashResult::Info(
                    "No agents configured. Use `dm --agent-create <name>` to create one."
                        .to_string(),
                );
            }
            let mut lines = format!("{:<20} {:<20} {}", "NAME", "MODEL", "DESCRIPTION");
            for a in &agents {
                let model = a.model.as_deref().unwrap_or("(default)");
                let desc = a.description.as_deref().unwrap_or("");
                write!(lines, "\n{:<20} {:<20} {}", a.name, model, desc)
                    .expect("write to String never fails");
            }
            return SlashResult::Info(lines);
        }
        SlashCommand::AgentRun(name) => {
            match crate::agents::AgentConfig::load(&app.config_dir, &name) {
                Ok(agent) => {
                    let model = agent.model.as_deref().unwrap_or("(session default)");
                    let desc = agent.description.as_deref().unwrap_or("(no description)");
                    return SlashResult::Info(format!(
                        "Agent '{}': model={}, description={}\nTo use this agent, restart with: dm --agent-run {} -p \"your prompt\"",
                        name, model, desc, name
                    ));
                }
                Err(e) => {
                    return SlashResult::Error(format!("Agent '{}' not found: {}", name, e));
                }
            }
        }
        SlashCommand::Eval(arg) => {
            if arg.is_empty() {
                // List available suites
                match crate::eval::runner::list_evals() {
                    Ok(paths) if paths.is_empty() => {
                        return SlashResult::Info(
                            "No eval suites found in ~/.dm/evals/\n\
                             Create a YAML suite and run: /eval <name>"
                                .to_string(),
                        );
                    }
                    Ok(paths) => {
                        let mut lines = "Eval suites:".to_string();
                        for p in &paths {
                            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                            write!(lines, "\n  {}", name).expect("write to String never fails");
                        }
                        lines.push_str("\n\nRun with: /eval <name>");
                        return SlashResult::Info(lines);
                    }
                    Err(e) => return SlashResult::Error(format!("Error listing evals: {}", e)),
                }
            }

            // Resolve suite path: try as-is, then ~/.dm/evals/<arg>.yaml
            let suite_path = {
                let direct = std::path::PathBuf::from(&arg);
                if direct.exists() {
                    direct
                } else {
                    let named = app.config_dir.join("evals").join(format!("{}.yaml", arg));
                    if named.exists() {
                        named
                    } else {
                        let named_yml = app.config_dir.join("evals").join(format!("{}.yml", arg));
                        if named_yml.exists() {
                            named_yml
                        } else {
                            return SlashResult::Error(format!(
                                "Eval suite '{}' not found (tried {}.yaml in ~/.dm/evals/)",
                                arg, arg
                            ));
                        }
                    }
                }
            };

            let yaml_text = match std::fs::read_to_string(&suite_path) {
                Ok(t) => t,
                Err(e) => return SlashResult::Error(format!("Cannot read suite: {}", e)),
            };
            let suite: crate::eval::EvalSuite = match serde_yaml::from_str(&yaml_text) {
                Ok(s) => s,
                Err(e) => return SlashResult::Error(format!("Cannot parse suite YAML: {}", e)),
            };

            let total = suite.cases.len();
            app.push_entry(
                EntryKind::SystemInfo,
                format!("Running eval: {} ({} cases)…", suite.name, total),
            );
            app.agent_busy = true;
            app.turn_start = Some(std::time::Instant::now());

            let eval_client = OllamaClient::new(
                client.base_url().to_string(),
                suite
                    .model
                    .clone()
                    .unwrap_or_else(|| client.model().to_string()),
            );

            let eval_opts = crate::eval::runner::EvalOptions {
                quiet: true,
                ..Default::default()
            };
            match crate::eval::runner::run_eval(&suite, None, &eval_client, &eval_opts).await {
                Ok(result) => {
                    app.agent_busy = false;
                    app.turn_start = None;
                    let passed = result.cases.iter().filter(|c| c.passed).count();
                    let failed = total - passed;

                    let mut out =
                        format!("── eval: {} ─────────────────────────\n", result.suite_name);
                    for case in &result.cases {
                        let icon = if case.passed { "✓" } else { "✗" };
                        write!(out, "  {} {}", icon, case.id).expect("write to String never fails");
                        for cr in &case.check_results {
                            if !cr.passed {
                                if let Some(ref msg) = cr.message {
                                    write!(out, "\n    ✗ {}: {}", cr.check_desc, msg)
                                        .expect("write to String never fails");
                                }
                            }
                        }
                        out.push('\n');
                    }
                    write!(
                        out,
                        "── {}/{} passed  {:.0}% ──────────────────",
                        passed, total, result.score_pct
                    )
                    .expect("write to String never fails");
                    if failed > 0 {
                        return SlashResult::Error(out);
                    }
                    return SlashResult::Info(out);
                }
                Err(e) => {
                    app.agent_busy = false;
                    app.turn_start = None;
                    return SlashResult::Error(format!("Eval failed: {}", e));
                }
            }
        }
        SlashCommand::ChainStart(file) => return handle_chain_start(file, app, client),
        SlashCommand::ChainStatus => return handle_chain_status(app),
        SlashCommand::ChainStop => return handle_chain_stop(app),
        SlashCommand::ChainAdd {
            name,
            role,
            model,
            input_from,
        } => return handle_chain_add(name, role, model, input_from, app),
        SlashCommand::ChainRemove(name) => return handle_chain_remove(name, app),
        SlashCommand::ChainTalk { node, message } => return handle_chain_talk(node, message),
        SlashCommand::ChainModel { node, model } => return handle_chain_model(node, model, app),
        SlashCommand::ChainPause => return handle_chain_pause(app),
        SlashCommand::ChainResume => return handle_chain_resume(app, client),
        SlashCommand::ChainResumeFrom(workspace_path) => {
            return handle_chain_resume_from(workspace_path, app, client)
        }
        SlashCommand::ChainInit(name) => return handle_chain_init(name),
        SlashCommand::ChainLog(cycle) => return handle_chain_log(cycle),
        SlashCommand::ChainValidate(file) => return handle_chain_validate(file, client).await,
        SlashCommand::ChainMetrics => return handle_chain_metrics(),
        SlashCommand::ChainHelp => return handle_chain_help(),
        SlashCommand::ChainList => return handle_chain_list(),
        SlashCommand::ChainPresets => return handle_chain_presets(),
        SlashCommand::Commit => return handle_commit(app).await,
        SlashCommand::Log(n) => {
            let count = n.min(200); // cap to avoid enormous output
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(5),
                tokio::process::Command::new("git")
                    .args(["log", "--oneline", &format!("-{}", count)])
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
                        "git log timed out. Try: re-run with a smaller -<n> count, or check repo health.".into(),
                    );
                }
            };

            if !output.status.success() && output.stdout.is_empty() {
                return SlashResult::Error(
                    "Not a git repository or git log failed. Try: cd into a git repo, or run `git status` to diagnose.".into(),
                );
            }

            let log_text = String::from_utf8_lossy(&output.stdout);
            if log_text.trim().is_empty() {
                return SlashResult::Info("No commits found in this repository.".into());
            }

            let commit_count = log_text.lines().count();
            app.pending_context = Some(format!(
                "Recent git history ({} commit{}):\n```\n{}\n```",
                commit_count,
                if commit_count == 1 { "" } else { "s" },
                log_text.trim_end()
            ));
            return SlashResult::Info(format!(
                "Git log ready ({} commit{}) — type your question to analyze history",
                commit_count,
                if commit_count == 1 { "" } else { "s" }
            ));
        }
        SlashCommand::Init => {
            let cwd = match std::env::current_dir() {
                Ok(d) => d,
                Err(e) => {
                    return SlashResult::Error(format!("Cannot determine current directory: {}", e))
                }
            };
            let dm_md_path = cwd.join("DM.md");
            if dm_md_path.exists() {
                return SlashResult::Info(format!(
                    "DM.md already exists at {} — edit it to update project instructions.",
                    dm_md_path.display()
                ));
            }
            let template = crate::init::generate_dm_md_template(&cwd);
            match std::fs::write(&dm_md_path, &template) {
                Ok(()) => {
                    return SlashResult::Info(format!(
                        "Created DM.md at {} — edit it to configure dm for this project.",
                        dm_md_path.display()
                    ))
                }
                Err(e) => return SlashResult::Error(format!("Failed to write DM.md: {}", e)),
            }
        }
        SlashCommand::Review(r) => return handle_review(r, app).await,
        SlashCommand::Branch(target) => return handle_branch(target).await,
        SlashCommand::Changelog { from, to } => return handle_changelog(from, to, app).await,
        SlashCommand::Blame { file, line } => return handle_blame(file, line, app).await,
        SlashCommand::Conflicts => return handle_conflicts(app).await,
        SlashCommand::Stash(sub) => return handle_stash(sub, app).await,
        SlashCommand::Stats => {
            let turns = app
                .entries
                .iter()
                .filter(|e| matches!(e.kind, EntryKind::UserMessage))
                .count();

            let ctx_line = if let Some((used, limit)) = app.ctx_usage {
                let pct = if limit > 0 { used * 100 / limit } else { 0 };
                format!("  Context:       {}/{} tokens ({}%)", used, limit, pct)
            } else {
                "  Context:       unknown".to_string()
            };

            let perf_line = if let Some(ref p) = app.perf {
                format!(
                    "  Speed:         {:.1} tok/s  (TTFT {}ms)",
                    p.tok_per_sec, p.ttft_ms
                )
            } else {
                "  Speed:         —".to_string()
            };

            let title_line = if let Some(ref t) = app.session_title {
                format!("  Title:         {}", t)
            } else {
                "  Title:         (untitled)".to_string()
            };

            let usage = &app.token_usage;
            let token_line = if usage.total() > 0 {
                format!(
                    "  Token usage:   {}p + {}c = {} total ({} turns)",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total(),
                    usage.turn_count
                )
            } else {
                "  Token usage:   —".to_string()
            };

            let info = format!(
                "Session statistics:\n  Session ID:    {}\n{}\n  Model:         {}\n  Turns:         {}\n  Total tokens:  {}\n{}\n{}\n{}",
                app.session_id,
                title_line,
                app.model,
                turns,
                app.total_tokens,
                token_line,
                ctx_line,
                perf_line,
            );
            return SlashResult::Info(info);
        }
        SlashCommand::Usage => {
            let usage = &app.token_usage;
            let avg = if usage.turn_count > 0 {
                usage.total() / usage.turn_count as u64
            } else {
                0
            };
            let ctx_line = if let Some((used, limit)) = app.ctx_usage {
                let pct = if limit > 0 { used * 100 / limit } else { 0 };
                format!("  Context window: {}/{} tokens ({}%)", used, limit, pct)
            } else {
                "  Context window: unknown".to_string()
            };
            let info = format!(
                "Token usage:\n  Prompt tokens:     {}\n  Completion tokens: {}\n  Total tokens:      {}\n  Turns:             {}\n  Avg tokens/turn:   {}\n{}",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total(),
                usage.turn_count,
                avg,
                ctx_line,
            );
            return SlashResult::Info(info);
        }
        SlashCommand::Fork(at_turn) => return handle_fork(at_turn, app),
        SlashCommand::Export(arg) => return handle_export(arg, app),
        SlashCommand::BugHunter(focus) => {
            // Collect Rust source files from the current directory.
            let cwd = std::env::current_dir().unwrap_or_default();
            let pattern = cwd.join("src/**/*.rs").to_string_lossy().to_string();
            let files: Vec<String> = glob::glob(&pattern)
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
                .filter(|p| p.is_file())
                .map(|p| {
                    p.strip_prefix(&cwd).map_or_else(
                        |_| p.to_string_lossy().to_string(),
                        |r| r.to_string_lossy().to_string(),
                    )
                })
                .collect();

            let file_count = files.len();
            let prompt = build_bughunter_prompt(&files, &focus);
            app.pending_context = Some(prompt);
            return SlashResult::Info(format!(
                "Bug hunt ready ({} files in scope) — press Enter to start analysis",
                file_count
            ));
        }
        SlashCommand::SecurityReview(r) => return handle_security_review(r, app).await,
        SlashCommand::Advisor(topic) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let pattern = cwd.join("src/**/*.rs").to_string_lossy().to_string();
            let files: Vec<String> = glob::glob(&pattern)
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
                .filter(|p| p.is_file())
                .map(|p| {
                    p.strip_prefix(&cwd).map_or_else(
                        |_| p.to_string_lossy().to_string(),
                        |r| r.to_string_lossy().to_string(),
                    )
                })
                .collect();
            let file_count = files.len();
            let prompt = build_advisor_prompt(&topic, &files);
            app.pending_context = Some(prompt);
            return SlashResult::Info(format!(
                "Advisor context ready ({} files in scope) — press Enter for analysis",
                file_count
            ));
        }
        SlashCommand::Effort(level) => {
            if level.is_empty() {
                let current = match app.effort_level {
                    EffortLevel::Quick => "quick",
                    EffortLevel::Normal => "normal (default)",
                    EffortLevel::Thorough => "thorough",
                };
                return SlashResult::Info(format!(
                    "Current effort level: {}\nUsage: /effort [quick|normal|thorough]",
                    current
                ));
            }
            let new_level = match level.as_str() {
                "quick" | "q" => EffortLevel::Quick,
                "normal" | "n" | "default" => EffortLevel::Normal,
                "thorough" | "t" | "full" | "deep" => EffortLevel::Thorough,
                other => {
                    return SlashResult::Error(format!(
                        "Unknown effort level '{}'. Use: quick, normal, or thorough.",
                        other
                    ));
                }
            };
            let name = match &new_level {
                EffortLevel::Quick => "quick",
                EffortLevel::Normal => "normal",
                EffortLevel::Thorough => "thorough",
            };
            app.effort_level = new_level;
            return SlashResult::Info(format!("Effort level set to: {}", name));
        }
        SlashCommand::Pin(path) => {
            if path.is_empty() {
                if app.pinned_files.is_empty() {
                    return SlashResult::Info("No pinned files. Use: /pin <file>".to_string());
                }
                let list = app
                    .pinned_files
                    .iter()
                    .enumerate()
                    .map(|(i, f)| format!("  {}. {}", i + 1, f))
                    .collect::<Vec<_>>()
                    .join("\n");
                return SlashResult::Info(format!("Pinned files:\n{}", list));
            }
            if app.pinned_files.contains(&path) {
                return SlashResult::Info(format!("'{}' is already pinned.", path));
            }
            if !std::path::Path::new(&path).exists() {
                return SlashResult::Error(format!(
                    "File not found: '{}'. Check the path and try again.",
                    path
                ));
            }
            app.pinned_files.push(path.clone());
            return SlashResult::Info(format!(
                "Pinned '{}'. Its contents will be prepended to every message.",
                path
            ));
        }
        SlashCommand::Unpin(arg) => {
            if arg.is_empty() || arg == "all" {
                let count = app.pinned_files.len();
                app.pinned_files.clear();
                return SlashResult::Info(if count == 0 {
                    "No pinned files to remove.".to_string()
                } else {
                    format!("Unpinned all {} file(s).", count)
                });
            }
            if let Some(pos) = app.pinned_files.iter().position(|f| f == &arg) {
                app.pinned_files.remove(pos);
                return SlashResult::Info(format!("Unpinned '{}'.", arg));
            }
            return SlashResult::Error(format!(
                "'{}' is not in the pinned list. Use /pin to see pinned files.",
                arg
            ));
        }
        SlashCommand::Rename(new_title) => {
            if new_title.is_empty() {
                return SlashResult::Info(
                    "Usage: /rename <new title>  — renames the current session".to_string(),
                );
            }
            app.session_title = Some(new_title.clone());
            // Persist: update the session file if a session ID is known.
            let config_dir = dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                .join(".dm");
            if let Ok(mut sess) = session_storage::load(&config_dir, &app.session_id) {
                sess.title = Some(new_title.clone());
                let _ = session_storage::save(&config_dir, &sess);
            }
            return SlashResult::Info(format!("Session renamed to \"{}\"", new_title));
        }
        SlashCommand::Context => {
            let turns = app
                .entries
                .iter()
                .filter(|e| matches!(e.kind, EntryKind::UserMessage))
                .count();
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            let wiki_snippet_chars = crate::api::load_wiki_snippet(&cwd).map(|s| s.len());
            let system_prompt_chars = Some(
                crate::system_prompt::build_system_prompt(&[], None)
                    .await
                    .len(),
            );
            let mut report = build_context_report(
                &app.session_id,
                app.session_title.as_deref(),
                &app.model,
                app.total_tokens,
                app.ctx_usage,
                turns,
                &app.mcp_servers,
                app.pending_context.is_some(),
                app.is_compacting.as_ref(),
                app.ctx_usage.map(|(_, limit)| limit),
                system_prompt_chars,
                wiki_snippet_chars,
            );
            if !app.pinned_files.is_empty() {
                let pin_list = app
                    .pinned_files
                    .iter()
                    .map(|f| format!("    • {}", f))
                    .collect::<Vec<_>>()
                    .join("\n");
                writeln!(report, "  Pinned files:\n{}", pin_list)
                    .expect("write to String never fails");
            }
            return SlashResult::Info(report);
        }
        SlashCommand::Pr(base_arg) => return handle_pr(base_arg, app).await,
        SlashCommand::Template(arg) => {
            let config_dir = dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                .join(".dm");
            if arg.is_empty() {
                // List available templates
                let templates = crate::templates::list_templates(&config_dir);
                if templates.is_empty() {
                    return SlashResult::Info(format!(
                        "No templates found in {}/templates/\nCreate a template: {}/templates/<name>.md",
                        config_dir.display(),
                        config_dir.display()
                    ));
                }
                let mut lines = vec!["Available templates:".to_string()];
                for t in &templates {
                    if t.description.is_empty() {
                        lines.push(format!("  /template {}", t.name));
                    } else {
                        lines.push(format!("  /template {}  — {}", t.name, t.description));
                    }
                }
                return SlashResult::Info(lines.join("\n"));
            }
            // Load named template with remaining args
            let mut iter = arg.splitn(2, ' ');
            let name = iter.next().unwrap_or("").trim();
            let rest = iter.next().unwrap_or("").trim();
            let args: Vec<String> = if rest.is_empty() {
                vec![]
            } else {
                rest.split_whitespace().map(|s| s.to_string()).collect()
            };
            match crate::templates::load_template(&config_dir, name, &args) {
                Ok(body) => {
                    app.pending_context = Some(body);
                    return SlashResult::Info(format!(
                        "Template '{}' loaded — press Enter to send",
                        name
                    ));
                }
                Err(e) => {
                    return SlashResult::Error(format!("Template error: {}", e));
                }
            }
        }
        SlashCommand::TodoList => {
            let todos = crate::tools::todo::read_todos(&app.config_dir, &app.session_id);
            return SlashResult::Info(crate::tools::todo::format_todos(&todos));
        }
        SlashCommand::TodoAdd { priority, content } => {
            let mut todos = crate::tools::todo::read_todos(&app.config_dir, &app.session_id);
            let id = format!("{}", todos.len() + 1);
            let prio = match priority.as_str() {
                "high" => crate::tools::todo::TodoPriority::High,
                "low" => crate::tools::todo::TodoPriority::Low,
                _ => crate::tools::todo::TodoPriority::Medium,
            };
            todos.push(crate::tools::todo::Todo {
                id: id.clone(),
                content: content.clone(),
                status: crate::tools::todo::TodoStatus::Pending,
                priority: prio,
            });
            match crate::tools::todo::write_todos(&app.config_dir, &app.session_id, &todos) {
                Ok(()) => {
                    return SlashResult::Info(format!(
                        "Added todo #{}: {} [{}]",
                        id, content, priority
                    ));
                }
                Err(e) => return SlashResult::Error(format!("Failed to save todo: {}", e)),
            }
        }
        SlashCommand::TodoDone(id) => {
            let mut todos = crate::tools::todo::read_todos(&app.config_dir, &app.session_id);
            let found = todos
                .iter_mut()
                .find(|t| t.id == id || t.id.starts_with(&id));
            match found {
                Some(t) => {
                    let tid = t.id.clone();
                    t.status = crate::tools::todo::TodoStatus::Completed;
                    match crate::tools::todo::write_todos(&app.config_dir, &app.session_id, &todos)
                    {
                        Ok(()) => return SlashResult::Info(format!("Marked #{} as done.", tid)),
                        Err(e) => return SlashResult::Error(format!("Failed to save: {}", e)),
                    }
                }
                None => return SlashResult::Error(format!("No todo with id '{}'.", id)),
            }
        }
        SlashCommand::TodoWip(id) => {
            let mut todos = crate::tools::todo::read_todos(&app.config_dir, &app.session_id);
            let found = todos
                .iter_mut()
                .find(|t| t.id == id || t.id.starts_with(&id));
            match found {
                Some(t) => {
                    let tid = t.id.clone();
                    t.status = crate::tools::todo::TodoStatus::InProgress;
                    match crate::tools::todo::write_todos(&app.config_dir, &app.session_id, &todos)
                    {
                        Ok(()) => {
                            return SlashResult::Info(format!("Marked #{} as in-progress.", tid))
                        }
                        Err(e) => return SlashResult::Error(format!("Failed to save: {}", e)),
                    }
                }
                None => return SlashResult::Error(format!("No todo with id '{}'.", id)),
            }
        }
        SlashCommand::TodoClear => {
            let todos = crate::tools::todo::read_todos(&app.config_dir, &app.session_id);
            let before = todos.len();
            let kept: Vec<_> = todos
                .into_iter()
                .filter(|t| !matches!(t.status, crate::tools::todo::TodoStatus::Completed))
                .collect();
            let removed = before - kept.len();
            match crate::tools::todo::write_todos(&app.config_dir, &app.session_id, &kept) {
                Ok(()) => {
                    return SlashResult::Info(format!(
                        "Removed {} completed todo{}.",
                        removed,
                        if removed == 1 { "" } else { "s" }
                    ));
                }
                Err(e) => return SlashResult::Error(format!("Failed to save: {}", e)),
            }
        }
        SlashCommand::TodoScan => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let pattern = cwd.join("src/**/*.rs").to_string_lossy().to_string();
            let paths: Vec<std::path::PathBuf> = glob::glob(&pattern)
                .into_iter()
                .flatten()
                .filter_map(|r| r.ok())
                .filter(|p| p.is_file())
                .collect();
            let items = crate::todo::scanner::scan_todos(&paths);
            if items.is_empty() {
                return SlashResult::Info("No TODO/FIXME/HACK comments found in src/.".to_string());
            }
            let list = crate::todo::format_todo_list(&items);
            return SlashResult::Info(format!(
                "{} comment{} found:\n\n{}",
                items.len(),
                if items.len() == 1 { "" } else { "s" },
                list
            ));
        }
        SlashCommand::Summary(focus) => {
            let turns = app
                .entries
                .iter()
                .filter(|e| matches!(e.kind, EntryKind::UserMessage))
                .count();
            if turns == 0 {
                return SlashResult::Info(
                    "Nothing to summarize yet — start a conversation first.".to_string(),
                );
            }
            let prompt = build_summary_prompt(turns, &focus);
            app.pending_context = Some(prompt);
            return SlashResult::Info(
                "Summary prompt ready — press Enter to generate the session summary".to_string(),
            );
        }
        SlashCommand::Brief => {
            app.brief_mode = !app.brief_mode;
            let status = if app.brief_mode { "on" } else { "off" };
            return SlashResult::Info(format!(
                "Brief mode {}. Responses will {}be formatted as concise bullet points.",
                status,
                if app.brief_mode { "" } else { "not " }
            ));
        }
        SlashCommand::Plan => {
            app.plan_mode = !app.plan_mode;
            let signal = if app.plan_mode {
                "__plan_mode__:on"
            } else {
                "__plan_mode__:off"
            };
            user_tx.send(signal.to_string()).await.ok();
            let status = if app.plan_mode { "on" } else { "off" };
            return SlashResult::Info(format!(
                "Plan mode {}. {}",
                status,
                if app.plan_mode {
                    "Write tools are blocked. Explore the codebase and design your approach."
                } else {
                    "Write tools are enabled. You can now make changes."
                }
            ));
        }
        SlashCommand::Files => {
            let report = format_files_report(&app.pinned_files, app.pending_context.is_some());
            return SlashResult::Info(report);
        }
        SlashCommand::New(title) => {
            // Save the current session to disk before wiping it (best-effort).
            // Read-then-update so on-disk fields we don't mirror in App state
            // (parent_id, turn_count, tokens, compact_failures, messages)
            // survive. Without this, forked sessions lose their lineage
            // across /new and /sessions tree shows orphaned branches.
            let cwd = std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            // Compute warning BEFORE clearing entries: pushing a warning here
            // and then calling entries.clear() below would silently erase it.
            // Consume the Err into a String so the clear can't destroy it.
            let save_warning = crate::session::storage::update_or_create_stub(
                &app.config_dir,
                &app.session_id,
                app.session_title.clone(),
                &app.model,
                &cwd,
            )
            .err()
            .map(|e| format_new_save_error(&e, &app.config_dir));

            // Reset conversation display state.
            app.entries.clear();
            app.streaming_partial = None;
            app.pending_context = None;
            app.session_title = if title.is_empty() {
                None
            } else {
                Some(title.clone())
            };

            // Assign a fresh session ID.
            app.session_id = uuid::Uuid::new_v4().to_string();

            // Signal the backend runner to clear its conversation history.
            user_tx.send("/clear".to_string()).await.ok();

            // Surface the save failure (if any) AFTER the clear so it renders.
            if let Some(warning) = save_warning {
                app.push_entry(EntryKind::SystemInfo, warning);
            }

            let msg = if title.is_empty() {
                "New conversation started. History cleared.".to_string()
            } else {
                format!("New conversation '{}' started. History cleared.", title)
            };
            return SlashResult::Info(msg);
        }
        SlashCommand::Kill => {
            if !app.agent_busy {
                return SlashResult::Info("No agent turn is running.".into());
            }
            if crate::tools::bash::kill_running_bash() {
                app.push_entry(
                    EntryKind::SystemInfo,
                    "Sent SIGTERM to running bash process.".to_string(),
                );
            } else {
                return SlashResult::Info("No bash process is currently running.".into());
            }
        }
        SlashCommand::Test(custom_cmd) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let cmd = if custom_cmd.is_empty() {
                crate::testfix::detect::detect_test_cmd(&cwd)
            } else {
                custom_cmd.clone()
            };
            app.push_entry(EntryKind::SystemInfo, format!("Running: {}", cmd));
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(120),
                tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(&cmd)
                    .output(),
            )
            .await
            {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => return SlashResult::Error(format!("Failed to run tests: {}", e)),
                Err(_) => return SlashResult::Error("Test command timed out after 120s".into()),
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}\n{}", stdout, stderr);
            if output.status.success() {
                return SlashResult::Info("All tests pass.".into());
            }
            let failures = crate::testfix::extract_failures(&combined);
            let diag_section = if failures.is_empty() {
                let tail_start = combined.len().saturating_sub(2000);
                combined[tail_start..].to_string()
            } else {
                crate::git::safe_truncate(&failures, 8000).to_string()
            };
            let prompt = format!(
                "The following test command failed: `{}`\n\n\
                Diagnostics:\n```\n{}\n```\n\n\
                For each failure, read the relevant source file, understand the issue, \
                and apply a targeted fix. Do not refactor unrelated code.",
                cmd, diag_section
            );
            app.pending_context = Some(prompt);
            let fail_count = diag_section
                .lines()
                .filter(|l| l.contains("FAILED") || l.contains("error["))
                .count();
            return SlashResult::Info(format!(
                "Tests failed ({} diagnostic lines) — press Enter to send failures to AI for fixing",
                if fail_count > 0 { fail_count } else { diag_section.lines().count() }
            ));
        }
        SlashCommand::Lint(custom_cmd) => {
            let cwd = std::env::current_dir().unwrap_or_default();
            let cmd = if custom_cmd.is_empty() {
                crate::testfix::detect::detect_lint_cmd(&cwd)
            } else {
                custom_cmd.clone()
            };
            app.push_entry(EntryKind::SystemInfo, format!("Running: {}", cmd));
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(120),
                tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(&cmd)
                    .output(),
            )
            .await
            {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => return SlashResult::Error(format!("Failed to run linter: {}", e)),
                Err(_) => return SlashResult::Error("Lint command timed out after 120s".into()),
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}\n{}", stdout, stderr);
            if output.status.success() && !combined.contains("warning[") {
                return SlashResult::Info("No lint issues found.".into());
            }
            let warnings: Vec<&str> = combined
                .lines()
                .filter(|l| {
                    l.contains("warning[")
                        || l.contains("error[")
                        || l.starts_with("warning:")
                        || l.starts_with("error:")
                })
                .collect();
            let diag_section = if warnings.is_empty() {
                let tail_start = combined.len().saturating_sub(2000);
                combined[tail_start..].to_string()
            } else {
                crate::git::safe_truncate(&warnings.join("\n"), 8000).to_string()
            };
            let prompt = format!(
                "The following lint command reported issues: `{}`\n\n\
                Warnings/errors:\n```\n{}\n```\n\n\
                Fix each warning. Apply targeted changes only — do not refactor unrelated code.",
                cmd, diag_section
            );
            app.pending_context = Some(prompt);
            return SlashResult::Info(format!(
                "Lint issues found ({} warnings) — press Enter to send to AI for fixing",
                warnings.len()
            ));
        }
        SlashCommand::Find(pattern) => {
            if pattern.is_empty() {
                return SlashResult::Error(
                    "Usage: /find <glob-pattern>  (e.g. /find src/**/*.rs)".into(),
                );
            }
            let cwd = std::env::current_dir().unwrap_or_default();
            let full_pattern = if pattern.starts_with('/') {
                pattern.clone()
            } else {
                cwd.join(&pattern).to_string_lossy().to_string()
            };
            let mut matches: Vec<String> = Vec::new();
            if let Ok(paths) = glob::glob(&full_pattern) {
                for path_result in paths.flatten() {
                    let display = path_result.strip_prefix(&cwd).map_or_else(
                        |_| path_result.to_string_lossy().to_string(),
                        |r| r.to_string_lossy().to_string(),
                    );
                    matches.push(display);
                    if matches.len() >= 100 {
                        break;
                    }
                }
            }
            if matches.is_empty() {
                return SlashResult::Info(format!("No files matching '{}'", pattern));
            }
            let count = matches.len();
            let listing = matches.join("\n");
            app.pending_context = Some(format!(
                "Files matching '{}' ({} results):\n{}\n\nWork with these files as needed.",
                pattern, count, listing
            ));
            return SlashResult::Info(format!(
                "Found {} file{} matching '{}' — press Enter to include in context",
                count,
                if count == 1 { "" } else { "s" },
                pattern
            ));
        }
        SlashCommand::Rg(args) => {
            if args.is_empty() {
                return SlashResult::Error(
                    "Usage: /rg <pattern> [path]  (e.g. /rg \"fn main\" src/)".into(),
                );
            }
            let mut rg_args = vec![
                "rg",
                "--no-heading",
                "-n",
                "--max-count",
                "5",
                "--max-filesize",
                "1M",
            ];
            let user_parts: Vec<&str> = args.split_whitespace().collect();
            for part in &user_parts {
                rg_args.push(part);
            }
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(30),
                tokio::process::Command::new(rg_args[0])
                    .args(&rg_args[1..])
                    .output(),
            )
            .await
            {
                Ok(Ok(out)) => out,
                Ok(Err(_)) => {
                    let grep_output = match tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        tokio::process::Command::new("grep")
                            .args(["-rn", "--max-count=5", &args])
                            .output(),
                    )
                    .await
                    {
                        Ok(Ok(out)) => out,
                        Ok(Err(e)) => {
                            return SlashResult::Error(format!(
                                "Neither rg nor grep available: {}",
                                e
                            ))
                        }
                        Err(_) => return SlashResult::Error("Search timed out".into()),
                    };
                    grep_output
                }
                Err(_) => return SlashResult::Error("Search timed out after 30s".into()),
            };
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.trim().is_empty() {
                return SlashResult::Info(format!("No matches for: {}", args));
            }
            let truncated = crate::git::safe_truncate(&stdout, 8000);
            let match_count = truncated.lines().count();
            app.pending_context = Some(format!(
                "Search results for `{}`:\n```\n{}\n```\n\nAnalyze or work with these matches as needed.",
                args, truncated
            ));
            return SlashResult::Info(format!(
                "Found {} match lines — press Enter to include in context",
                match_count
            ));
        }
        SlashCommand::Cd(path) => {
            if path.is_empty() {
                let cwd = std::env::current_dir().unwrap_or_default();
                return SlashResult::Info(format!("Working directory: {}", cwd.display()));
            }
            let expanded = if let Some(rest) = path.strip_prefix("~/") {
                if let Some(home) = dirs::home_dir() {
                    home.join(rest)
                } else {
                    std::path::PathBuf::from(&path)
                }
            } else if path == "~" {
                dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from(&path))
            } else {
                std::path::PathBuf::from(&path)
            };
            match std::env::set_current_dir(&expanded) {
                Ok(()) => {
                    let cwd = std::env::current_dir().unwrap_or(expanded);
                    app.push_entry(EntryKind::SystemInfo, format!("cd {}", cwd.display()));
                }
                Err(e) => return SlashResult::Error(format!("cd: {}: {}", path, e)),
            }
        }
        SlashCommand::Changes => {
            let file_edits: Vec<&crate::tui::app::ActivityEntry> = app
                .activity_log
                .iter()
                .filter(|e| matches!(e.kind, crate::tui::app::ActivityKind::FileEdit))
                .collect();
            if file_edits.is_empty() {
                return SlashResult::Info("No file changes in this session.".into());
            }
            let mut seen = std::collections::HashSet::new();
            let mut lines = Vec::new();
            let session_start = app
                .activity_log
                .first()
                .map_or_else(std::time::Instant::now, |a| a.timestamp);
            for entry in &file_edits {
                let file_path = entry
                    .detail
                    .split(':')
                    .nth(1)
                    .unwrap_or(&entry.detail)
                    .trim();
                let elapsed = entry.timestamp.duration_since(session_start).as_secs();
                let mins = elapsed / 60;
                let secs = elapsed % 60;
                if seen.insert(file_path.to_string()) {
                    lines.push(format!("  {:02}:{:02}  {}", mins, secs, file_path));
                }
            }
            return SlashResult::Info(format!(
                "Files modified by AI ({} unique):\n{}",
                seen.len(),
                lines.join("\n")
            ));
        }
        SlashCommand::Tree { path, depth } => {
            let root = if path.is_empty() {
                std::env::current_dir().unwrap_or_default()
            } else {
                std::path::PathBuf::from(&path)
            };
            if !root.is_dir() {
                return SlashResult::Error(format!("Not a directory: {}", root.display()));
            }
            fn walk_tree(
                dir: &std::path::Path,
                prefix: &str,
                depth: usize,
                max_depth: usize,
                out: &mut String,
                count: &mut usize,
            ) {
                if depth > max_depth || *count > 500 {
                    return;
                }
                let mut entries: Vec<_> = match std::fs::read_dir(dir) {
                    Ok(rd) => rd.filter_map(|e| e.ok()).collect(),
                    Err(_) => return,
                };
                entries.sort_by_key(|e| e.file_name());
                let total = entries.len();
                for (i, entry) in entries.iter().enumerate() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with('.')
                        || name_str == "target"
                        || name_str == "node_modules"
                    {
                        continue;
                    }
                    let is_last = i == total - 1;
                    let connector = if is_last { "└── " } else { "├── " };
                    let child_prefix = if is_last { "    " } else { "│   " };
                    let is_dir = entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false);
                    writeln!(
                        out,
                        "{}{}{}{}",
                        prefix,
                        connector,
                        name_str,
                        if is_dir { "/" } else { "" }
                    )
                    .expect("write to String never fails");
                    *count += 1;
                    if is_dir && depth < max_depth {
                        walk_tree(
                            &entry.path(),
                            &format!("{}{}", prefix, child_prefix),
                            depth + 1,
                            max_depth,
                            out,
                            count,
                        );
                    }
                }
            }
            let mut tree = format!("{}/\n", root.display());
            let mut count = 0usize;
            walk_tree(&root, "", 1, depth, &mut tree, &mut count);
            if count > 500 {
                tree.push_str("... (truncated at 500 entries)\n");
            }
            app.pending_context = Some(format!(
                "Directory structure ({} entries, depth {}):\n```\n{}\n```\nUse this structure to orient file references.",
                count, depth, tree
            ));
            return SlashResult::Info(format!(
                "Tree ready ({} entries) — press Enter to include in context",
                count
            ));
        }
        SlashCommand::History(limit) => {
            if app.activity_log.is_empty() {
                return SlashResult::Info("No activity yet in this session.".into());
            }
            let session_start = app
                .activity_log
                .first()
                .map_or_else(std::time::Instant::now, |a| a.timestamp);
            let entries: Vec<String> = app
                .activity_log
                .iter()
                .rev()
                .take(limit)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|entry| {
                    let elapsed = entry.timestamp.duration_since(session_start).as_secs();
                    let mins = elapsed / 60;
                    let secs = elapsed % 60;
                    let kind_icon = match entry.kind {
                        crate::tui::app::ActivityKind::BashCommand => "$",
                        crate::tui::app::ActivityKind::FileEdit => "~",
                        crate::tui::app::ActivityKind::ToolCall => ">",
                        crate::tui::app::ActivityKind::TurnComplete => "*",
                    };
                    let detail = if entry.detail.len() > 100 {
                        let mut end = 100usize.min(entry.detail.len());
                        while end > 0 && !entry.detail.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}…", &entry.detail[..end])
                    } else {
                        entry.detail.clone()
                    };
                    format!("  {:02}:{:02} {} {}", mins, secs, kind_icon, detail)
                })
                .collect();
            return SlashResult::Info(format!(
                "Activity log ({} entries, showing last {}):\n{}",
                app.activity_log.len(),
                entries.len(),
                entries.join("\n")
            ));
        }
        SlashCommand::Search(query) => {
            if query.is_empty() {
                return SlashResult::Error("Usage: /search <query>".into());
            }
            let lower_query = query.to_lowercase();
            let mut results: Vec<String> = Vec::new();
            for (i, entry) in app.entries.iter().enumerate() {
                if entry.content.to_lowercase().contains(&lower_query) {
                    let kind_label = match entry.kind {
                        EntryKind::UserMessage => "user",
                        EntryKind::AssistantMessage => "assistant",
                        EntryKind::SystemInfo => "system",
                        EntryKind::ToolCall => "tool",
                        EntryKind::ToolResult => "tool-result",
                        _ => "other",
                    };
                    let preview = if entry.content.len() > 120 {
                        let mut end = 120usize.min(entry.content.len());
                        while end > 0 && !entry.content.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}…", &entry.content[..end])
                    } else {
                        entry.content.clone()
                    };
                    let preview = preview.replace('\n', " ");
                    results.push(format!("  #{} [{}] {}", i + 1, kind_label, preview));
                    if results.len() >= 20 {
                        break;
                    }
                }
            }
            if results.is_empty() {
                return SlashResult::Info(format!("No matches for '{}'.", query));
            }
            return SlashResult::Info(format!(
                "Found {} match{} for '{}':\n{}",
                results.len(),
                if results.len() == 1 { "" } else { "es" },
                query,
                results.join("\n")
            ));
        }
        SlashCommand::Version => {
            let info = build_version_info(&app.model, &app.host);
            return SlashResult::Info(info);
        }
        SlashCommand::WikiStatus => return handle_wiki_status(),
        SlashCommand::WikiSearch(query) => return handle_wiki_search(query),
        SlashCommand::WikiLint => return handle_wiki_lint(),
        SlashCommand::WikiRefresh => return handle_wiki_refresh(),
        SlashCommand::WikiSummary => return handle_wiki_summary(),
        SlashCommand::WikiConcepts => return handle_wiki_concepts(),
        SlashCommand::WikiMomentum => return handle_wiki_momentum(),
        SlashCommand::WikiFresh => return handle_wiki_fresh(),
        SlashCommand::WikiPlanner => return handle_wiki_planner(),
        SlashCommand::WikiStats => return handle_wiki_stats(),
        SlashCommand::WikiPrune(n) => return handle_wiki_prune(n),
        SlashCommand::WikiSeed(arg) => return handle_wiki_seed(arg),
        SlashCommand::WikiUnknown(sub) => return handle_wiki_unknown(sub),
        SlashCommand::ScheduleList => {
            match crate::daemon::scheduler::load_schedules(&app.config_dir) {
                Ok(tasks) if tasks.is_empty() => {
                    return SlashResult::Info("No scheduled tasks.".to_string());
                }
                Ok(tasks) => {
                    let mut lines = format!("Scheduled tasks ({}):", tasks.len());
                    for t in &tasks {
                        let next = t.next_run.as_deref().unwrap_or("—");
                        write!(
                            lines,
                            "\n  [{}] cron={} next={} — {}",
                            t.id, t.cron, next, t.prompt
                        )
                        .expect("write to String never fails");
                    }
                    return SlashResult::Info(lines);
                }
                Err(e) => return SlashResult::Error(format!("Error loading schedules: {}", e)),
            }
        }
        SlashCommand::ScheduleAdd(spec) => {
            match crate::daemon::scheduler::parse_schedule_add(&spec) {
                Err(e) => return SlashResult::Error(format!("{}", e)),
                Ok((cron, prompt)) => {
                    match crate::daemon::scheduler::load_schedules(&app.config_dir) {
                        Err(e) => {
                            return SlashResult::Error(format!("Error loading schedules: {}", e))
                        }
                        Ok(mut tasks) => {
                            let id = crate::daemon::scheduler::generate_task_id();
                            tasks.push(crate::daemon::scheduler::ScheduledTask {
                                id: id.clone(),
                                cron,
                                prompt,
                                model: None,
                                last_run: None,
                                next_run: None,
                            });
                            match crate::daemon::scheduler::save_schedules(&app.config_dir, &tasks)
                            {
                                Ok(()) => {
                                    return SlashResult::Info(format!(
                                        "Scheduled task added (id={}).",
                                        id
                                    ))
                                }
                                Err(e) => {
                                    return SlashResult::Error(format!(
                                        "Error saving schedule: {}",
                                        e
                                    ))
                                }
                            }
                        }
                    }
                }
            }
        }
        SlashCommand::ScheduleRemove(id) => {
            if id.is_empty() {
                return SlashResult::Error("Usage: /schedule remove <id>".to_string());
            }
            match crate::daemon::scheduler::load_schedules(&app.config_dir) {
                Err(e) => return SlashResult::Error(format!("Error loading schedules: {}", e)),
                Ok(mut tasks) => {
                    let before = tasks.len();
                    tasks.retain(|t| t.id != id);
                    if tasks.len() == before {
                        return SlashResult::Error(format!("No task with id '{}'.", id));
                    }
                    match crate::daemon::scheduler::save_schedules(&app.config_dir, &tasks) {
                        Ok(()) => return SlashResult::Info(format!("Removed task {}.", id)),
                        Err(e) => {
                            return SlashResult::Error(format!("Error saving schedules: {}", e))
                        }
                    }
                }
            }
        }
        SlashCommand::ToolDisable(name) => return handle_tool_disable(name, user_tx).await,
        SlashCommand::ToolEnable(name) => return handle_tool_enable(name, user_tx).await,
        SlashCommand::ToolList => return handle_tool_list(user_tx).await,
        SlashCommand::Unknown(name) => {
            let msg = match suggest_slash_command(&name) {
                Some(sugg) => format!(
                    "Unknown command: /{}. Did you mean /{}? (Type /help for all commands.)",
                    name, sugg
                ),
                None => format!("Unknown command: /{}. Type /help for commands.", name),
            };
            app.push_entry(EntryKind::SystemInfo, msg);
        }
    }
    SlashResult::Done
}

fn open_url(url: &str) -> bool {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "cmd"
    } else {
        "xdg-open"
    };
    let mut cmd = std::process::Command::new(opener);
    if cfg!(target_os = "windows") {
        cmd.args(["/c", "start", url]);
    } else {
        cmd.arg(url);
    }
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|mut c| {
            let _ = c.wait();
            true
        })
        .unwrap_or(false)
}

fn load_image_base64(path: &str) -> anyhow::Result<(String, String)> {
    use base64::Engine;
    use std::path::Path;

    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" => {}
        other => anyhow::bail!(
            "Unsupported image format '{}'. Supported: png, jpg, jpeg, gif, webp",
            other
        ),
    }
    let bytes = std::fs::read(p)?;
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let filename = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string();
    Ok((filename, data))
}

#[cfg(test)]
mod tests;
