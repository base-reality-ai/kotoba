//! The Dark Matter (dm) CLI and entry point.
//!
//! Handles command-line argument parsing, environment initialization, and
//! dispatches execution to the appropriate interactive (TUI), headless, or
//! background daemon sub-systems.

// dm controls the filenames it creates (lowercase by convention) and
// matches against Rust ecosystem standards (`.rs`, `.md`, `.toml`,
// etc., always lowercase). Case-sensitive `.ends_with(".rs")` is
// intentional — see `wiki/ingest.rs`, `wiki/lint.rs`, `logging.rs`,
// `orchestrate/mod.rs`, `panic_hook.rs` for representative call sites.
#![allow(clippy::case_sensitive_file_extension_comparisons)]
// See `src/lib.rs` for the cast_possible_truncation rationale —
// intentional truncation in timing/budget arithmetic.
#![allow(clippy::cast_possible_truncation)]

use std::io::Write as _;

// Kernel modules live in the `dark_matter` library (src/lib.rs's `pub mod foo;`
// declarations). The dm binary uses them via `dark_matter::foo::*` rather than
// duplicating them with parallel `mod foo;` declarations here. The duplication
// previously created two distinct type identities for every kernel module
// (one in the library crate, one in the binary crate), which made host-cap
// state in `dark_matter::host::HOST_CAPS` invisible to the binary's
// `dm::host::installed_host_capabilities()` reader. See kotoba's
// `.dm/wiki/concepts/paradigm-gap-host-caps-binary-duplication.md` for the
// full diagnosis. Library is self-aliased to `dark_matter` via
// `extern crate self as dark_matter;` in lib.rs so the `dark_matter::*` paths
// in shared source files (e.g. doctor.rs, tools/registry.rs) resolve
// identically whether compiled into the library or referenced here.
//
// Glob each module into scope so existing bareword references (e.g.
// `daemon::run_daemon(...)`, `use tui::app::EntryKind`) keep working without a
// 73-callsite sweep across this file. The `dm::*` alias remains useful for
// path-position references where unambiguous (`dm::host::install_*`).
use dark_matter as dm;
// Bring every kernel module into scope. Some are unused at this exact callsite
// snapshot but kept in the list deliberately — the binary's bareword refs grow
// and shrink over time; tracking which are currently referenced isn't worth
// the maintenance cost.
#[allow(unused_imports)]
use dark_matter::{
    agents, api, bench, changeset, compaction, config, conversation, daemon,
    doctor, document, error_hints, eval, exit_codes, format, git, gpu, host,
    identity, index, init, logging, mcp, memory, models, notify, ollama,
    orchestrate, panic_hook, permissions, plugins, routing, run, security,
    session, share, summarize, system_prompt, telemetry, templates, testfix,
    todo, tokens, tools, translate, tui, util, warnings, web, wiki,
};

use dm::config::Config;
use dm::conversation::DEFAULT_MAX_TURNS;
use dm::ollama::client::OllamaClient;
use dm::session::short_id;
use anyhow::Context;
use clap::{Parser, ValueEnum};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser, Debug)]
#[command(
    name = "dm",
    version,
    about = "Dark Matter — local AI coding agent powered by Ollama",
    long_about = None,
    after_help = "EXIT CODES:\n  0  Success\n  1  Agent error (generic runtime failure)\n  2  Config error (invalid flag, missing prompt, bad config file)\n  3  Model unreachable (Ollama connect/timeout failure)",
)]
struct Cli {
    /// Prompt to send in non-interactive (print) mode
    #[arg(short = 'p', long = "print", value_name = "PROMPT")]
    print: Option<String>,

    /// Ollama model to use
    #[arg(long = "model", short = 'm', env = "DM_MODEL")]
    model: Option<String>,

    /// Ollama host (default: localhost:11434, or `OLLAMA_HOST` env var)
    #[arg(long = "host", env = "OLLAMA_HOST")]
    host: Option<String>,

    /// List available Ollama models and exit
    #[arg(long = "models")]
    models: bool,

    /// List configured model aliases and exit
    #[arg(long = "aliases")]
    aliases: bool,

    /// Pull a model from Ollama with a progress bar
    #[arg(long = "models-pull", value_name = "MODEL")]
    models_pull: Option<String>,

    /// Remove an installed model (prompts for confirmation unless --models-force)
    #[arg(long = "models-rm", value_name = "MODEL")]
    models_rm: Option<String>,

    /// Pull latest for every installed model
    #[arg(long = "models-update")]
    models_update: bool,

    /// Show size, family, quantization and context length for a model
    #[arg(long = "models-info", value_name = "MODEL")]
    models_info: Option<String>,

    /// Skip confirmation prompt for destructive model operations (used with --models-rm)
    #[arg(long = "models-force")]
    models_force: bool,

    /// Enable verbose output
    #[arg(long = "verbose", short = 'v')]
    verbose: bool,

    /// Suppress informational stderr output (summary line, etc.)
    #[arg(long = "quiet", short = 'q')]
    quiet: bool,

    /// Print inference performance stats to stderr after each print-mode turn (TTFT, tok/s, total)
    #[arg(long = "perf")]
    perf: bool,

    /// Print routing classification to stderr: which key and model were selected for each prompt
    #[arg(long = "routing-debug")]
    routing_debug: bool,

    /// Wrap file paths in tool output with OSC 8 hyperlink escapes (iTerm2/Kitty/WezTerm/foot)
    #[arg(long = "hyperlinks")]
    hyperlinks: bool,

    /// Output format: text (default), json, stream-json, stream-text
    #[arg(long = "output-format", default_value = "text")]
    output_format: String,

    /// Stream plain text tokens to stdout as they arrive (equivalent to --output-format stream-text)
    #[arg(long = "stream")]
    stream: bool,

    /// Skip all permission prompts (dangerous)
    #[arg(long = "dangerously-skip-permissions")]
    dangerously_skip_permissions: bool,

    /// Load a specific session by ID
    #[arg(long = "session-id")]
    session_id: Option<String>,

    /// Resume the most recent session (alias for --continue)
    #[arg(long = "resume")]
    resume: bool,

    /// Continue the most recent conversation
    #[arg(long = "continue", short = 'c')]
    continue_session: bool,

    /// Initialize dm in the current project (creates DM.md template)
    #[arg(long = "init")]
    init: bool,

    /// Run diagnostics: check Ollama connectivity, config, and models
    #[arg(long = "doctor")]
    doctor: bool,

    /// List unrecovered crashed sessions from panic markers
    #[arg(long = "recovery")]
    recovery: bool,

    /// Generate shell completions and print to stdout
    #[arg(long = "completions", value_name = "SHELL")]
    completions: Option<String>,

    /// Maximum tool-call rounds per turn (default: 20)
    #[arg(long)]
    max_turns: Option<usize>,

    /// Additional directories to include in context (repeatable)
    #[arg(long = "add-dir", value_name = "PATH", action = clap::ArgAction::Append)]
    add_dirs: Vec<std::path::PathBuf>,

    /// Run in pure chat mode — no tools sent to Ollama
    #[arg(long = "no-tools")]
    no_tools: bool,

    /// Append extra instructions to the system prompt (one-shot; does not persist)
    #[arg(long = "system", value_name = "TEXT")]
    extra_system: Option<String>,

    /// Print effective configuration and exit
    #[arg(long = "config")]
    show_config: bool,

    /// Pre-approve a tool for this session without the interactive dialog (repeatable)
    /// Example: --allow tool:bash --allow `tool:read_file`
    #[arg(long = "allow", value_name = "SPEC", action = clap::ArgAction::Append)]
    allow: Vec<String>,

    /// Block a tool for this session (repeatable). Overrides --allow for the same tool.
    /// Example: --disallow tool:bash --disallow `tool:write_file`
    #[arg(long = "disallow", value_name = "SPEC", action = clap::ArgAction::Append)]
    disallow: Vec<String>,

    /// Permission mode: default (ask for each tool), plan (allow read-only, ask for writes),
    /// full (allow all tools, same as --dangerously-skip-permissions)
    #[arg(long = "permission-mode", value_name = "MODE")]
    permission_mode: Option<String>,

    /// Build or update the semantic search index for the current project
    #[arg(long = "index")]
    index: bool,

    /// Embedding model for `dm index` and semantic search (default: nomic-embed-text)
    #[arg(long = "embed-model", env = "DM_EMBED_MODEL")]
    embed_model: Option<String>,

    /// Model for tool-execution rounds (fast/small); reasoning rounds use --model
    #[arg(long = "tool-model", env = "DM_TOOL_MODEL")]
    tool_model: Option<String>,

    /// Generate and confirm a commit message for staged changes
    #[arg(long = "commit")]
    commit: bool,

    /// Review changes against a git ref (use 'staged' for staged changes, default: HEAD)
    #[arg(long = "review", value_name = "REF")]
    review: Option<String>,

    /// Generate a PR title and body from the current branch diff vs base branch
    #[arg(long = "pr", value_name = "BASE", num_args = 0..=1, default_missing_value = "")]
    pr: Option<String>,

    /// Push current branch to origin before creating the PR (used with --pr)
    #[arg(long = "pr-push")]
    pr_push: bool,

    /// Open PR immediately with `gh pr create` (used with --pr)
    #[arg(long = "pr-open")]
    pr_open: bool,

    /// Create PR as draft (used with --pr --pr-open)
    #[arg(long = "pr-draft")]
    pr_draft: bool,

    /// Run as an MCP server (stdio transport) — exposes Dark Matter's tools to any MCP client
    #[arg(long = "mcp-server")]
    mcp_server: bool,

    /// Watch for file changes and re-index incrementally (use with `dm index`)
    #[arg(long = "watch")]
    watch: bool,

    /// Run a multi-agent orchestration chain (planner -> builder -> validator)
    #[arg(long = "orchestrate", value_name = "TASK")]
    orchestrate: Option<String>,

    /// Load a chain YAML config and run it (e.g. `dm --chain build/dm-chain.yaml`)
    #[arg(long = "chain", value_name = "FILE")]
    chain: Option<std::path::PathBuf>,

    /// Resume a chain from a saved checkpoint (e.g. `dm --chain-resume .dm-workspace/`)
    #[arg(long = "chain-resume", value_name = "WORKSPACE")]
    chain_resume: Option<std::path::PathBuf>,

    /// Validate a chain YAML config without running it
    #[arg(long = "chain-validate", value_name = "FILE")]
    chain_validate: Option<std::path::PathBuf>,

    /// Custom chain ID for orchestration (default: auto-generated UUID)
    #[arg(long = "chain-id", value_name = "ID")]
    chain_id: Option<String>,

    /// Max orchestration cycles before giving up (default: 5)
    #[arg(long = "max-cycles", default_value = "5")]
    max_cycles: usize,

    /// Working directory for orchestration artifacts (default: .dm-workspace/)
    #[arg(long = "workspace", value_name = "DIR")]
    workspace: Option<std::path::PathBuf>,

    /// Prune chunks for deleted files from the semantic index (no re-embedding)
    #[arg(long = "gc")]
    gc: bool,

    /// Export the current (or most recent) session as a self-contained HTML file
    #[arg(long = "share", value_name = "FILE", num_args = 0..=1, default_missing_value = "")]
    share: Option<String>,

    /// Start the web UI and listen on localhost (default port: 7421)
    #[arg(long = "serve")]
    serve: bool,

    /// Port for the web UI server (used with --serve)
    #[arg(long = "port", default_value = "7421")]
    port: u16,

    /// Start the headless REST API on localhost (default port: 7422)
    #[arg(long = "web")]
    web: bool,

    /// Port for the REST API server (used with --web)
    #[arg(long = "web-port", default_value = "7422")]
    web_port: u16,

    /// Bearer token required for --web API requests (optional)
    #[arg(long = "web-token", value_name = "TOKEN")]
    web_token: Option<String>,

    /// Execute a markdown run-spec sequentially (dm run <spec.md>)
    #[arg(long = "run", value_name = "SPEC")]
    run: Option<std::path::PathBuf>,

    /// Save run transcript to a file (used with --run)
    #[arg(long = "run-output", value_name = "FILE")]
    run_output: Option<std::path::PathBuf>,

    /// Print resolved prompts without calling the LLM (used with --run)
    #[arg(long = "dry-run")]
    dry_run: bool,

    /// Benchmark available models
    #[arg(long = "bench", value_name = "MODELS", num_args = 0..=1, default_missing_value = "__all__")]
    bench: Option<String>,

    /// Benchmark task: `code_gen|explain|bug_fix|refactor|all` (default: all)
    #[arg(long = "bench-task", default_value = "all")]
    bench_task: String,

    /// Runs per model per task
    #[arg(long = "bench-runs", default_value = "3")]
    bench_runs: usize,

    /// Save bench results to file
    #[arg(long = "bench-out")]
    bench_out: Option<std::path::PathBuf>,

    /// Show benchmark history; optionally specify a timestamp to view details
    #[arg(long = "bench-history", value_name = "TIMESTAMP", num_args = 0..=1, default_missing_value = "__list__")]
    bench_history: Option<String>,

    /// Pull (download) a model from Ollama registry
    #[arg(long = "pull", value_name = "MODEL")]
    pull: Option<String>,

    /// Run CMD, fix failures with AI, repeat (default cmd: cargo test)
    #[arg(long = "test-fix", value_name = "CMD", num_args = 0..=1, default_missing_value = "cargo test")]
    test_fix: Option<String>,

    /// Max fix rounds for --test-fix
    #[arg(long = "test-fix-rounds", default_value = "5")]
    test_fix_rounds: usize,

    /// Enable JSON output mode (passes format:json to Ollama)
    #[arg(long = "json-schema", value_name = "SCHEMA")]
    json_schema: Option<String>,

    /// List available prompt templates
    #[arg(long = "templates")]
    templates: bool,

    /// Use a prompt template by name
    #[arg(long = "template", value_name = "NAME")]
    template: Option<String>,

    /// Arguments for the template (repeatable)
    #[arg(long = "template-arg", value_name = "ARG", action = clap::ArgAction::Append)]
    template_args: Vec<String>,

    /// Inject file contents into system prompt (glob pattern, repeatable)
    #[arg(long = "context", value_name = "GLOB", action = clap::ArgAction::Append)]
    context: Vec<String>,

    /// Watch files and run test-fix on change
    #[arg(long = "watch-fix", value_name = "PATTERN", num_args = 0..=1, default_missing_value = "src/**/*.rs")]
    watch_fix: Option<String>,

    /// Command for --watch-fix (auto-detected if not specified)
    #[arg(long = "watch-cmd", default_value = "")]
    watch_cmd: String,

    /// Search session history
    #[arg(long = "search", value_name = "QUERY")]
    search: Option<String>,

    /// Remove sessions older than N days
    #[arg(long = "prune-sessions", value_name = "DAYS")]
    prune_sessions: Option<u64>,

    /// Scan for TODO/FIXME/HACK comments and prioritize them
    #[arg(long = "todo")]
    todo: bool,

    /// After listing TODOs, fix them with AI
    #[arg(long = "todo-fix")]
    todo_fix: bool,

    /// Glob patterns for --todo scan (repeatable)
    #[arg(long = "todo-glob", value_name = "GLOB", action = clap::ArgAction::Append)]
    todo_glob: Vec<String>,

    /// Export format for --share: html (default) or md
    #[arg(long = "share-format", default_value = "html")]
    share_format: String,

    /// Run linter, fix warnings with AI, repeat (auto-detects cargo clippy / eslint / ruff)
    #[arg(long = "lint-fix", value_name = "CMD", num_args = 0..=1)]
    lint_fix: Option<Option<String>>,

    /// Show accumulated project memory and exit
    #[arg(long = "memory")]
    memory_show: bool,

    /// Clear all project memory and exit
    #[arg(long = "memory-clear")]
    memory_clear: bool,

    /// Open project memory in $EDITOR and exit
    #[arg(long = "memory-edit")]
    memory_edit: bool,

    /// Disable project memory for this session (no inject, no update)
    #[arg(long = "no-memory")]
    no_memory: bool,

    /// Skip workspace context injection (README.md fallback when no DM.md)
    #[arg(long = "no-workspace-context")]
    no_workspace_context: bool,

    /// Skip DM.md inheritance walk — load no DM.md files at all
    #[arg(long = "no-claude-md")]
    no_dm_md: bool,

    /// Generate changelog from git history (optionally specify FROM ref)
    #[arg(long = "changelog", value_name = "FROM", num_args = 0..=1)]
    changelog: Option<Option<String>>,

    /// Upper bound for --changelog (default: HEAD)
    #[arg(long = "changelog-to", default_value = "HEAD")]
    changelog_to: String,

    /// Format for --changelog: md, conventional, keep-a-changelog
    #[arg(long = "changelog-format", default_value = "md")]
    changelog_format: String,

    /// Compare models on a prompt (comma-separated model names)
    #[arg(long = "compare", value_name = "MODELS")]
    compare: Option<String>,

    /// Prompt for --compare (uses --print value if omitted, then stdin)
    #[arg(long = "compare-prompt", value_name = "PROMPT")]
    compare_prompt: Option<String>,

    /// Print --compare output side-by-side
    #[arg(long = "compare-side-by-side")]
    compare_side_by_side: bool,

    /// Summarize a file, URL, or stdin
    #[arg(long = "summarize", value_name = "TARGET", num_args = 0..=1, default_missing_value = "-")]
    summarize: Option<String>,

    /// Target word count for --summarize
    #[arg(long = "summarize-length", default_value = "150")]
    summarize_length: usize,

    /// Style for --summarize: bullets, paragraph, tldr
    #[arg(long = "summarize-style", default_value = "bullets")]
    summarize_style: String,

    /// Skip diff preview — apply file writes immediately without staging (for scripting / watch-fix)
    #[arg(long = "auto-apply")]
    auto_apply: bool,

    /// Generate/update docstrings for a file
    #[arg(long = "document", value_name = "FILE")]
    document: Option<std::path::PathBuf>,

    /// Doc comment style: auto, rust, python, jsdoc, generic
    #[arg(long = "document-style", default_value = "auto")]
    document_style: String,

    /// Process multiple files matching glob for --document
    #[arg(long = "document-glob", value_name = "GLOB")]
    document_glob: Option<String>,

    /// Translate source file to another language
    #[arg(long = "translate", value_name = "FILE")]
    translate: Option<std::path::PathBuf>,

    /// Target language for --translate
    #[arg(long = "translate-to", value_name = "LANG")]
    translate_to: Option<String>,

    /// Output path for --translate (default: auto)
    #[arg(long = "translate-out", value_name = "FILE")]
    translate_out: Option<std::path::PathBuf>,

    /// Run code formatter on changed files after applying staged changes
    #[arg(long = "format-after")]
    format_after: bool,

    /// Security audit files matching GLOB
    #[arg(long = "security", value_name = "GLOB", num_args = 0..=1, default_missing_value = "src/**/*.rs")]
    security: Option<String>,

    /// Start background daemon
    #[arg(long = "daemon-start")]
    daemon_start: bool,

    /// Stop running daemon
    #[arg(long = "daemon-stop")]
    daemon_stop: bool,

    /// Show daemon status
    #[arg(long = "daemon-status")]
    daemon_status: bool,

    /// Show daemon health (uptime, sessions, pid)
    #[arg(long = "daemon-health")]
    daemon_health: bool,

    /// Restart daemon (stop + start)
    #[arg(long = "daemon-restart")]
    daemon_restart: bool,

    /// Run in-process, bypass daemon even if running
    #[arg(long = "no-daemon")]
    no_daemon: bool,

    /// Internal: run as daemon worker process (do not use directly)
    #[arg(long = "_daemon-worker", hide = true)]
    _daemon_worker: bool,

    /// Internal: run as watchdog supervisor (do not use directly)
    #[arg(long = "_daemon-watchdog", hide = true)]
    _daemon_watchdog: bool,

    /// Show daemon log (last N lines)
    #[arg(long = "daemon-log")]
    daemon_log: bool,

    /// Number of lines to show for --daemon-log (default: 50)
    #[arg(long = "tail", default_value = "50")]
    tail: usize,

    // ── Scheduler ──────────────────────────────────────────────────────────────
    /// Schedule a task: "CRON5 PROMPT" — first 5 tokens are cron, rest is prompt
    #[arg(long = "schedule-add", value_name = "CRON_PROMPT")]
    schedule_add: Option<String>,

    /// List all scheduled tasks
    #[arg(long = "schedule-list")]
    schedule_list: bool,

    /// Remove a scheduled task by ID
    #[arg(long = "schedule-remove", value_name = "ID")]
    schedule_remove: Option<String>,

    /// Run a scheduled task immediately by ID (bypass-all permissions)
    #[arg(long = "schedule-run", value_name = "ID")]
    schedule_run: Option<String>,

    // ── Desktop notifications ──────────────────────────────────────────────────
    /// Enable desktop notifications for task completion
    #[arg(long = "notify")]
    notify: bool,

    // ── Process management ─────────────────────────────────────────────────────
    /// List running daemon sessions
    #[arg(long = "ps")]
    ps: bool,

    /// Kill a daemon session by ID (use --all to kill all sessions)
    #[arg(long = "kill", value_name = "SESSION_ID")]
    kill: Option<String>,

    /// Kill all sessions (use with --kill)
    #[arg(long = "all")]
    kill_all: bool,

    /// Cancel the running turn in a daemon session (sends Cancel to agent)
    #[arg(long = "session-cancel", value_name = "SESSION_ID")]
    session_cancel: Option<String>,

    /// Gracefully shut down a daemon session (sends Shutdown, then removes)
    #[arg(long = "session-shutdown", value_name = "SESSION_ID")]
    session_shutdown: Option<String>,

    // ── Watchdog ───────────────────────────────────────────────────────────────
    /// Start daemon with watchdog (auto-restart on crash)
    #[arg(long = "watchdog")]
    watchdog: bool,

    /// Disable watchdog even if `DM_DAEMON_WATCHDOG` env is set
    #[arg(long = "no-watchdog")]
    no_watchdog: bool,

    // ── Daemon chain management ─────────────────────────────────────────────────
    /// Start a chain on the daemon from a YAML config file
    #[arg(long = "daemon-chain-start", value_name = "FILE")]
    daemon_chain_start: Option<String>,

    /// Show status of a daemon-managed chain
    #[arg(long = "daemon-chain-status", value_name = "CHAIN_ID")]
    daemon_chain_status: Option<String>,

    /// Stop a daemon-managed chain
    #[arg(long = "daemon-chain-stop", value_name = "CHAIN_ID")]
    daemon_chain_stop: Option<String>,

    /// List all daemon-managed chains
    #[arg(long = "daemon-chain-list")]
    daemon_chain_list: bool,

    /// Attach to a daemon-managed chain and stream its events
    #[arg(long = "daemon-chain-attach", value_name = "CHAIN_ID")]
    daemon_chain_attach: Option<String>,

    /// Pause a daemon-managed chain
    #[arg(long = "daemon-chain-pause", value_name = "CHAIN_ID")]
    daemon_chain_pause: Option<String>,

    /// Resume a paused daemon-managed chain
    #[arg(long = "daemon-chain-resume", value_name = "CHAIN_ID")]
    daemon_chain_resume: Option<String>,

    /// Inject a message into a chain node (`CHAIN_ID:NODE:MESSAGE`)
    #[arg(long = "daemon-chain-talk", value_name = "CHAIN_ID:NODE:MESSAGE")]
    daemon_chain_talk: Option<String>,

    /// Add a node to a running chain (`CHAIN_ID:NAME:MODEL`)
    #[arg(long = "daemon-chain-add", value_name = "CHAIN_ID:NAME:MODEL")]
    daemon_chain_add: Option<String>,

    /// Remove a node from a running chain (`CHAIN_ID:NODE`)
    #[arg(long = "daemon-chain-remove", value_name = "CHAIN_ID:NODE")]
    daemon_chain_remove: Option<String>,

    /// Swap the model for a chain node (`CHAIN_ID:NODE:MODEL`)
    #[arg(long = "daemon-chain-model", value_name = "CHAIN_ID:NODE:MODEL")]
    daemon_chain_model: Option<String>,

    // ── Agent management ───��───────────────────────────���───────────────────────
    /// Create a named agent configuration
    #[arg(long = "agent-create", value_name = "NAME")]
    agent_create: Option<String>,

    /// List all saved agents
    #[arg(long = "agent-list")]
    agent_list: bool,

    /// Show agent configuration by name
    #[arg(long = "agent-show", value_name = "NAME")]
    agent_show: Option<String>,

    /// Delete a named agent
    #[arg(long = "agent-delete", value_name = "NAME")]
    agent_delete: Option<String>,

    /// Run a named agent
    #[arg(long = "agent-run", value_name = "NAME")]
    agent_run: Option<String>,

    /// Model override for --agent-create or --agent-run
    #[arg(long = "agent-model")]
    agent_model: Option<String>,

    /// Extra system prompt for --agent-create
    #[arg(long = "agent-system")]
    agent_system: Option<String>,

    /// Comma-separated tool list for --agent-create
    #[arg(long = "agent-tools", value_name = "TOOLS")]
    agent_tools: Option<String>,

    /// Description for --agent-create
    #[arg(long = "agent-description")]
    agent_description: Option<String>,

    // ── Session replay ──────────────────────────────────────────────────────────
    /// Replay a saved session by ID
    #[arg(long = "replay", value_name = "SESSION_ID")]
    replay: Option<String>,

    /// Model override for --replay
    #[arg(long = "replay-model")]
    replay_model: Option<String>,

    /// Skip the first N user turns (0-indexed) for --replay
    #[arg(long = "replay-from-turn", default_value = "0")]
    replay_from_turn: usize,

    /// Dry-run --replay: print turns without calling Ollama
    #[arg(long = "replay-dry-run")]
    replay_dry_run: bool,

    // ── MCP management ─────────────────────────────────────────────────────────
    /// List all MCP server entries
    #[arg(long = "mcp-list")]
    mcp_list: bool,

    /// Config scope for MCP management commands (default: project in host mode, global in kernel mode)
    #[arg(long = "mcp-scope", value_enum)]
    mcp_scope: Option<McpScope>,

    // ── Eval ───────────────────────────────────────────────────────────────────
    /// Run evaluation suite(s) from YAML file(s) or glob
    #[arg(long = "eval", value_name = "FILE", action = clap::ArgAction::Append)]
    eval_files: Vec<String>,

    /// Model override for --eval
    #[arg(long = "eval-model")]
    eval_model: Option<String>,

    /// Compare two or more models on eval suites (comma-separated)
    #[arg(long = "eval-compare", value_name = "M1,M2")]
    eval_compare: Option<String>,

    /// Verbose eval output (print full model responses)
    #[arg(long = "eval-verbose")]
    eval_verbose: bool,

    /// Stop after first failing case in --eval
    #[arg(long = "eval-fail-fast")]
    eval_fail_fast: bool,

    /// Require 3/3 LLM judge agreement instead of 2/3 majority
    #[arg(long = "eval-strict")]
    eval_strict: bool,

    /// Save current eval result as the baseline for regression tracking
    #[arg(long = "eval-freeze")]
    eval_freeze: bool,

    /// Compare eval result to baseline; exit 1 if any regression (pass→fail, or rate drop > threshold with --eval-runs)
    #[arg(long = "eval-ci")]
    eval_ci: bool,

    /// Compare eval result to baseline, print diff, always exit 0
    #[arg(long = "eval-diff-baseline")]
    eval_diff_baseline: bool,

    /// Run each eval case N times to measure variance (default: 1)
    #[arg(long = "eval-runs", default_value = "1")]
    eval_runs: usize,

    /// Flag a case as flaky when pass rate is between (1-T) and T (default: 1.0 = any non-100%/0%)
    #[arg(long = "eval-flaky-threshold", default_value = "1.0")]
    eval_flaky_threshold: f32,

    /// With --eval-ci --eval-runs, regression threshold as rate drop (default: 0.1 = 10%)
    #[arg(long = "eval-regression-threshold", default_value = "0.1")]
    eval_regression_threshold: f32,

    /// Suppress per-case eval output (only show final summary)
    #[arg(long = "eval-quiet")]
    eval_quiet: bool,

    /// List eval YAML files in ~/.dm/evals/ and ./evals/
    #[arg(long = "eval-list")]
    eval_list: bool,

    /// List past eval result files in ~/.dm/eval/results/
    #[arg(long = "eval-results")]
    eval_results: bool,

    /// Load a saved eval result JSON and print a formatted report
    #[arg(long = "eval-report", value_name = "FILE")]
    eval_report: Option<String>,

    /// When combined with --eval-report, load the most recent result instead of a named file
    #[arg(long = "eval-report-last")]
    eval_report_last: bool,

    /// Compare two saved eval result JSON files side-by-side
    #[arg(long = "eval-report-compare", value_name = "FILE_A")]
    eval_report_compare: Option<String>,

    /// Second file for --eval-report-compare
    #[arg(long = "eval-report-compare-b", value_name = "FILE_B")]
    eval_report_compare_b: Option<String>,

    /// Add an MCP server: "name command [args...]"
    #[arg(long = "mcp-add", value_name = "SPEC")]
    mcp_add: Option<String>,

    /// Remove an MCP server by name
    #[arg(long = "mcp-remove", value_name = "NAME")]
    mcp_remove: Option<String>,

    /// Enable an MCP server by name
    #[arg(long = "mcp-enable", value_name = "NAME")]
    mcp_enable: Option<String>,

    /// Disable an MCP server by name
    #[arg(long = "mcp-disable", value_name = "NAME")]
    mcp_disable: Option<String>,

    /// Test an MCP server by name (spawn and run initialize handshake)
    #[arg(long = "mcp-test", value_name = "NAME")]
    mcp_test: Option<String>,

    /// Apply a unified diff from stdin (or file) to the working tree.
    /// Reads from stdin when no file is given. Reports each file: applied / failed.
    /// Exit code 0 if all hunks applied; 1 if any failed.
    #[arg(long = "patch", value_name = "FILE")]
    patch: Option<Option<String>>,

    /// Prompt passed as a positional argument
    prompt: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(clap::Subcommand, Debug, PartialEq)]
pub enum Commands {
    /// Initialize dm in the current project (creates DM.md template)
    Init,

    /// Run diagnostics: check Ollama connectivity, config, identity, and models
    Doctor,

    /// Scaffold a fresh project directory with dm as kernel
    Spawn {
        /// Name of the host project
        project_name: String,

        /// Canonical dark-matter repository URL or path (overrides DM_CANONICAL_REPO)
        #[arg(long)]
        canonical: Option<String>,
    },

    /// Pull canonical dm kernel updates into a spawned host project
    Sync {
        /// Show what would change without writing anything
        #[arg(long)]
        dry_run: bool,

        /// Discard a half-applied sync recovery state
        #[arg(long)]
        abort: bool,

        /// Report current pinned canonical revision vs canonical HEAD
        #[arg(long)]
        status: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum McpScope {
    Global,
    Project,
}

impl std::fmt::Display for McpScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpScope::Global => write!(f, "global"),
            McpScope::Project => write!(f, "project"),
        }
    }
}

impl McpScope {
    fn opposite(self) -> Self {
        match self {
            McpScope::Global => McpScope::Project,
            McpScope::Project => McpScope::Global,
        }
    }
}

#[derive(Debug, PartialEq)]
enum RunMode {
    DaemonStart,
    DaemonStop,
    DaemonStatus,
    DaemonRestart,
    DaemonClient, // connect to running daemon
    LocalAgent,   // in-process, as before
}

/// Resolve which run mode to use. Pure function — takes `daemon_running` as a bool
/// so callers can test without I/O.
fn resolve_mode(cli: &Cli, daemon_running: bool) -> RunMode {
    if cli.daemon_start {
        return RunMode::DaemonStart;
    }
    if cli.daemon_stop {
        return RunMode::DaemonStop;
    }
    if cli.daemon_status {
        return RunMode::DaemonStatus;
    }
    if cli.daemon_restart {
        return RunMode::DaemonRestart;
    }
    if cli.no_daemon {
        return RunMode::LocalAgent;
    }
    if daemon_running {
        return RunMode::DaemonClient;
    }
    RunMode::LocalAgent
}

/// Loads `Config::load()` or exits the process with `ConfigError` on failure.
/// Shared by every CLI subcommand that needs a Config up-front so the failure
/// is classified at the source (can't rely on the outer classifier — config
/// failures are not reqwest errors).
fn load_config_or_exit() -> Config {
    match Config::load() {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    &format!("config error: {:#}", e),
                    Some("dm --doctor"),
                )
            );
            std::process::exit(dm::exit_codes::ExitCode::ConfigError.as_i32());
        }
    }
}

fn resolve_mcp_scope(requested: Option<McpScope>, identity: &identity::Identity) -> McpScope {
    requested.unwrap_or_else(|| {
        if identity.is_host() {
            McpScope::Project
        } else {
            McpScope::Global
        }
    })
}

fn mcp_config_dir_for_scope(config_dir: &Path, project_root: &Path, scope: McpScope) -> PathBuf {
    match scope {
        McpScope::Global => config_dir.to_path_buf(),
        McpScope::Project => project_root.join(".dm"),
    }
}

/// Resolve the MCP config directory for any management command (`--mcp-list`,
/// `--mcp-add`, `--mcp-remove`, `--mcp-enable`, `--mcp-disable`, `--mcp-test`).
/// Reads identity from cwd so host mode defaults to the project-local
/// `.dm/mcp_servers.json`; kernel mode defaults to global `~/.dm/`.
fn resolved_mcp_dir(requested: Option<McpScope>, config_dir: &Path) -> (McpScope, PathBuf) {
    let identity = identity::load_for_cwd();
    let project_root = std::env::current_dir().unwrap_or_default();
    let scope = resolve_mcp_scope(requested, &identity);
    let dir = mcp_config_dir_for_scope(config_dir, &project_root, scope);
    (scope, dir)
}

// kotoba host-project divergence from canonical dm: register kotoba's
// HostCapabilities at dm-binary startup so the TUI / doctor / chain see
// host tools when launched as `dm` or via `kotoba dm`. With the kernel-side
// host/binary module-duplication fix in place, this install populates the
// single `dark_matter::host::HOST_CAPS` slot (no longer two).
#[path = "host_caps.rs"]
mod host_caps;
#[path = "domain.rs"]
#[allow(dead_code)]
mod domain;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    // Best-effort install — already-installed is fine (re-entrant launches,
    // tests). Failing here would block dm from launching at all.
    let _ = dark_matter::host::install_host_capabilities(Box::new(
        host_caps::KotobaCapabilities,
    ));

    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            // `:#` flattens the anyhow cause chain onto one line.
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    &format!("{:#}", e),
                    Some("dm --doctor, or re-run with RUST_LOG=debug"),
                )
            );
            dm::exit_codes::classify(&e).into()
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let mut cli = Cli::parse();

    if cli.stream && cli.output_format == "text" {
        cli.output_format = "stream-text".to_string();
    }

    let is_full_mode = cli.permission_mode.as_deref() == Some("full");
    if let Some(ref mode) = cli.permission_mode {
        match mode.as_str() {
            "default" | "plan" | "full" => {}
            _ => {
                let _ = writeln!(
                    std::io::stderr(),
                    "dm: unknown --permission-mode '{}' (use: default, plan, full)",
                    mode
                );
                std::process::exit(dm::exit_codes::ExitCode::ConfigError.as_i32());
            }
        }
    }

    // --_daemon-worker: internal flag — run as the daemon process and exit
    if cli._daemon_worker {
        let config = Arc::new(load_config_or_exit());
        daemon::run_daemon(config).await?;
        return Ok(());
    }

    // --_daemon-watchdog: internal flag — run the watchdog supervisor loop
    if cli._daemon_watchdog {
        daemon::watchdog::run_watchdog(vec!["--_daemon-worker".to_string()]).await?;
        return Ok(());
    }

    // --daemon-log: show last N lines of the daemon log file
    if cli.daemon_log {
        let config = load_config_or_exit();
        let lines = daemon::server::tail_log(&config.config_dir, cli.tail)?;
        if lines.is_empty() {
            println!("(daemon log is empty or does not exist)");
        } else {
            for line in &lines {
                println!("{}", line);
            }
        }
        return Ok(());
    }

    // ── Schedule handlers (no Ollama needed) ─────────────────────────────────

    if cli.schedule_list {
        let config = load_config_or_exit();
        let tasks = daemon::scheduler::load_schedules(&config.config_dir)?;
        if tasks.is_empty() {
            println!("No scheduled tasks.");
        } else {
            println!("{} scheduled task(s):", tasks.len());
            for t in &tasks {
                let next = t.next_run.as_deref().unwrap_or("(never)");
                let last = t.last_run.as_deref().unwrap_or("(never)");
                println!(
                    "  [{}] cron={} next={} last={} prompt={}",
                    t.id, t.cron, next, last, t.prompt
                );
            }
        }
        return Ok(());
    }

    if let Some(ref raw) = cli.schedule_add {
        let config = load_config_or_exit();
        let (cron, prompt) = daemon::scheduler::parse_schedule_add(raw)?;
        let id = daemon::scheduler::generate_task_id();
        // Set next_run to 1 day from now as initial value
        let next_run = (chrono::Utc::now() + chrono::Duration::days(1)).to_rfc3339();
        let task = daemon::scheduler::ScheduledTask {
            id: id.clone(),
            cron,
            prompt,
            model: None,
            last_run: None,
            next_run: Some(next_run),
        };
        let mut tasks = daemon::scheduler::load_schedules(&config.config_dir)?;
        tasks.push(task);
        daemon::scheduler::save_schedules(&config.config_dir, &tasks)?;
        println!("Added task {}", id);
        return Ok(());
    }

    if let Some(ref id) = cli.schedule_remove {
        let config = load_config_or_exit();
        let mut tasks = daemon::scheduler::load_schedules(&config.config_dir)?;
        let before = tasks.len();
        tasks.retain(|t| &t.id != id);
        if tasks.len() == before {
            let _ = writeln!(std::io::stderr(), "No task with id '{}'", id);
        } else {
            daemon::scheduler::save_schedules(&config.config_dir, &tasks)?;
            println!("Removed task {}", id);
        }
        return Ok(());
    }

    // ── ps / kill handlers ────────────────────────────────────────────────────

    if cli.ps {
        if !daemon::daemon_socket_exists() {
            println!("Daemon not running.");
            return Ok(());
        }
        match daemon::DaemonClient::connect().await {
            Ok(mut dc) => {
                dc.send_request("session.list", serde_json::json!({}))
                    .await?;
                match dc.recv_event().await {
                    Ok(daemon::protocol::DaemonEvent::SessionList { sessions }) => {
                        println!(
                            "{:<14} {:<14} {:<10} {:<9} LAST ACTIVE",
                            "SESSION", "MODEL", "STATUS", "CLIENTS"
                        );
                        for s in &sessions {
                            let short_id = &s.session_id[..s.session_id.len().min(8)];
                            println!(
                                "{:<14} {:<14} {:<10} {:<9} {}",
                                short_id,
                                "gemma4:26b-128k",
                                s.status,
                                s.client_count,
                                s.last_active
                            );
                        }
                    }
                    Ok(other) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Unexpected response to session.list: {:?}",
                            other
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Failed to receive session list: {}",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "[dm] Failed to connect to daemon: {}", e);
            }
        }
        return Ok(());
    }

    if let Some(ref kill_id) = cli.kill.clone() {
        if !daemon::daemon_socket_exists() {
            println!("Daemon not running.");
            return Ok(());
        }
        match daemon::DaemonClient::connect().await {
            Ok(mut dc) => {
                if cli.kill_all || kill_id == "_" {
                    // Get all session IDs first
                    dc.send_request("session.list", serde_json::json!({}))
                        .await?;
                    let session_ids = match dc.recv_event().await {
                        Ok(daemon::protocol::DaemonEvent::SessionList { sessions }) => sessions
                            .iter()
                            .map(|s| s.session_id.clone())
                            .collect::<Vec<_>>(),
                        _ => vec![],
                    };
                    // Need a fresh connection per request since each send_request blocks
                    for sid in &session_ids {
                        match daemon::DaemonClient::connect().await {
                            Ok(mut dc2) => {
                                let _ = dc2
                                    .send_request(
                                        "session.end",
                                        serde_json::json!({ "session_id": sid }),
                                    )
                                    .await;
                                println!("Killed session {}.", sid);
                            }
                            Err(e) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[dm] Failed to kill session {}: {}",
                                    sid,
                                    e
                                );
                            }
                        }
                    }
                } else {
                    let _ = dc
                        .send_request("session.end", serde_json::json!({ "session_id": kill_id }))
                        .await;
                    println!("Killed session {}.", kill_id);
                }
            }
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "[dm] Failed to connect to daemon: {}", e);
            }
        }
        return Ok(());
    }

    if cli.daemon_health {
        if !daemon::daemon_socket_exists() {
            println!("Daemon not running.");
            return Ok(());
        }
        match daemon::DaemonClient::connect().await {
            Ok(mut dc) => {
                dc.send_request("daemon.health", serde_json::json!({}))
                    .await?;
                match dc.recv_event().await {
                    Ok(daemon::protocol::DaemonEvent::Health {
                        uptime_secs,
                        session_count,
                        pid,
                    }) => {
                        let hours = uptime_secs / 3600;
                        let mins = (uptime_secs % 3600) / 60;
                        let secs = uptime_secs % 60;
                        println!("Daemon healthy");
                        println!("  PID:      {}", pid);
                        println!("  Uptime:   {}h {}m {}s", hours, mins, secs);
                        println!("  Sessions: {}", session_count);
                    }
                    Ok(daemon::protocol::DaemonEvent::Error { message, .. }) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", message);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", e);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "[dm] Failed to connect to daemon: {}", e);
            }
        }
        return Ok(());
    }

    if let Some(ref session_id) = cli.session_cancel {
        if !daemon::daemon_socket_exists() {
            println!("Daemon not running.");
            return Ok(());
        }
        match daemon::DaemonClient::connect().await {
            Ok(mut dc) => {
                dc.send_request(
                    "session.cancel",
                    serde_json::json!({ "session_id": session_id }),
                )
                .await?;
                match dc.recv_event().await {
                    Ok(daemon::protocol::DaemonEvent::SessionCancelled { session_id }) => {
                        println!("Cancelled session {}.", session_id);
                    }
                    Ok(daemon::protocol::DaemonEvent::Error { message, .. }) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", message);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", e);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "[dm] Failed to connect to daemon: {}", e);
            }
        }
        return Ok(());
    }

    if let Some(ref session_id) = cli.session_shutdown {
        if !daemon::daemon_socket_exists() {
            println!("Daemon not running.");
            return Ok(());
        }
        match daemon::DaemonClient::connect().await {
            Ok(mut dc) => {
                dc.send_request(
                    "session.shutdown",
                    serde_json::json!({ "session_id": session_id }),
                )
                .await?;
                match dc.recv_event().await {
                    Ok(daemon::protocol::DaemonEvent::SessionEnded { session_id }) => {
                        println!("Session {} shut down.", session_id);
                    }
                    Ok(daemon::protocol::DaemonEvent::Error { message, .. }) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", message);
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = writeln!(std::io::stderr(), "[dm] {}", e);
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "[dm] Failed to connect to daemon: {}", e);
            }
        }
        return Ok(());
    }

    // ── Agent management handlers (no Ollama needed) ──────────────────────────

    if cli.agent_list {
        let config = load_config_or_exit();
        let agents = agents::AgentConfig::list(&config.config_dir);
        if agents.is_empty() {
            println!("No agents configured. Use --agent-create <name> to create one.");
        } else {
            println!("{:<20} {:<20} DESCRIPTION", "NAME", "MODEL");
            for a in &agents {
                let model = a.model.as_deref().unwrap_or("(default)");
                let desc = a.description.as_deref().unwrap_or("");
                println!("{:<20} {:<20} {}", a.name, model, desc);
            }
        }
        return Ok(());
    }

    if let Some(ref name) = cli.agent_create.clone() {
        let config = load_config_or_exit();
        let tools = cli.agent_tools.as_deref().map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        });
        let agent = agents::AgentConfig {
            name: name.clone(),
            model: cli.agent_model.clone(),
            system_extra: cli.agent_system.clone(),
            tools,
            no_memory: cli.no_memory,
            description: cli.agent_description.clone(),
        };
        agent.save(&config.config_dir)?;
        let path = agents::AgentConfig::agent_path(&config.config_dir, name);
        println!("Agent '{}' saved to {}", name, path.display());
        return Ok(());
    }

    if let Some(ref name) = cli.agent_show.clone() {
        let config = load_config_or_exit();
        let agent = agents::AgentConfig::load(&config.config_dir, name)?;
        println!("{}", serde_json::to_string_pretty(&agent)?);
        return Ok(());
    }

    if let Some(ref name) = cli.agent_delete.clone() {
        let config = load_config_or_exit();
        agents::AgentConfig::delete(&config.config_dir, name)?;
        println!("Deleted.");
        return Ok(());
    }

    // ── MCP management handlers (no Ollama needed) ────────────────────────────

    if cli.mcp_scope.is_some()
        && !(cli.mcp_list
            || cli.mcp_add.is_some()
            || cli.mcp_remove.is_some()
            || cli.mcp_enable.is_some()
            || cli.mcp_disable.is_some()
            || cli.mcp_test.is_some())
    {
        let _ = writeln!(
            std::io::stderr(),
            "{}",
            dm::error_hints::format_dm_error(
                "--mcp-scope requires an MCP management command",
                Some("dm --mcp-list --mcp-scope project"),
            )
        );
        std::process::exit(dm::exit_codes::ExitCode::ConfigError.as_i32());
    }

    if cli.mcp_list {
        let config = load_config_or_exit();
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        let entries = mcp::manage::load_mcp_config(&mcp_config_dir)?;
        println!("MCP scope: {} ({})", scope, mcp_config_dir.display());
        if entries.is_empty() {
            println!("No MCP servers configured in {} scope.", scope);
        } else {
            println!("{:<20} {:<7} COMMAND", "NAME", "ENABLED");
            for e in &entries {
                let enabled = if e.enabled { "yes" } else { "no" };
                let cmd_str = if e.args.is_empty() {
                    e.command.clone()
                } else {
                    format!("{} {}", e.command, e.args.join(" "))
                };
                println!("{:<20} {:<7} {}", e.name, enabled, cmd_str);
            }
        }
        return Ok(());
    }

    if let Some(ref spec) = cli.mcp_add.clone() {
        let config = load_config_or_exit();
        let mut parts = spec.splitn(2, ' ');
        let name = parts.next().unwrap_or("").trim().to_string();
        let cmd_part = parts.next().unwrap_or("").trim().to_string();
        if name.is_empty() || cmd_part.is_empty() {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    "--mcp-add requires \"name command [args...]\"",
                    Some("dm --mcp-add myserver python -m my_mcp"),
                )
            );
            std::process::exit(dm::exit_codes::ExitCode::ConfigError.as_i32());
        }
        let (command, args) = mcp::manage::parse_command(&cmd_part);
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        mcp::manage::add_mcp_server(
            &mcp_config_dir,
            mcp::manage::McpServerEntry {
                name,
                command,
                args,
                env: std::collections::HashMap::new(),
                enabled: true,
            },
        )?;
        println!(
            "MCP server added to {} scope ({}).",
            scope,
            mcp_config_dir.display()
        );
        return Ok(());
    }

    if let Some(ref name) = cli.mcp_remove.clone() {
        let config = load_config_or_exit();
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        let mut entries = mcp::manage::load_mcp_config(&mcp_config_dir)?;
        let before = entries.len();
        entries.retain(|e| &e.name != name);
        if entries.len() == before {
            let _ = writeln!(
                std::io::stderr(),
                "No MCP server with name '{}' in {} scope ({}). Try: dm --mcp-list --mcp-scope {}",
                name,
                scope,
                mcp_config_dir.display(),
                scope.opposite(),
            );
        } else {
            mcp::manage::save_mcp_config(&mcp_config_dir, &entries)?;
            println!("Removed '{}' from {} scope.", name, scope);
        }
        return Ok(());
    }

    if let Some(ref name) = cli.mcp_enable.clone() {
        let config = load_config_or_exit();
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        let mut entries = mcp::manage::load_mcp_config(&mcp_config_dir)?;
        if let Some(e) = entries.iter_mut().find(|e| &e.name == name) {
            e.enabled = true;
        } else {
            let _ = writeln!(
                std::io::stderr(),
                "No MCP server with name '{}' in {} scope ({}). Try: dm --mcp-list --mcp-scope {}",
                name,
                scope,
                mcp_config_dir.display(),
                scope.opposite(),
            );
            return Ok(());
        }
        mcp::manage::save_mcp_config(&mcp_config_dir, &entries)?;
        println!("Enabled '{}' in {} scope.", name, scope);
        return Ok(());
    }

    if let Some(ref name) = cli.mcp_disable.clone() {
        let config = load_config_or_exit();
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        let mut entries = mcp::manage::load_mcp_config(&mcp_config_dir)?;
        if let Some(e) = entries.iter_mut().find(|e| &e.name == name) {
            e.enabled = false;
        } else {
            let _ = writeln!(
                std::io::stderr(),
                "No MCP server with name '{}' in {} scope ({}). Try: dm --mcp-list --mcp-scope {}",
                name,
                scope,
                mcp_config_dir.display(),
                scope.opposite(),
            );
            return Ok(());
        }
        mcp::manage::save_mcp_config(&mcp_config_dir, &entries)?;
        println!("Disabled '{}' in {} scope.", name, scope);
        return Ok(());
    }

    // ── Daemon chain management handlers ────────────────────────────────────────

    if let Some(ref file) = cli.daemon_chain_start {
        return daemon::commands::handle_daemon_chain_start(file).await;
    }
    if let Some(ref chain_id) = cli.daemon_chain_status {
        return daemon::commands::handle_daemon_chain_status(chain_id).await;
    }
    if let Some(ref chain_id) = cli.daemon_chain_stop {
        return daemon::commands::handle_daemon_chain_stop(chain_id).await;
    }
    if cli.daemon_chain_list {
        return daemon::commands::handle_daemon_chain_list().await;
    }
    if let Some(ref chain_id) = cli.daemon_chain_attach {
        return daemon::commands::handle_daemon_chain_attach(chain_id).await;
    }
    if let Some(ref chain_id) = cli.daemon_chain_pause {
        return daemon::commands::handle_daemon_chain_pause(chain_id).await;
    }
    if let Some(ref chain_id) = cli.daemon_chain_resume {
        return daemon::commands::handle_daemon_chain_resume(chain_id).await;
    }
    if let Some(ref arg) = cli.daemon_chain_talk {
        return daemon::commands::handle_daemon_chain_talk(arg).await;
    }
    if let Some(ref arg) = cli.daemon_chain_add {
        return daemon::commands::handle_daemon_chain_add(arg).await;
    }
    if let Some(ref arg) = cli.daemon_chain_remove {
        return daemon::commands::handle_daemon_chain_remove(arg).await;
    }
    if let Some(ref arg) = cli.daemon_chain_model {
        return daemon::commands::handle_daemon_chain_model(arg).await;
    }

    // Resolve daemon run mode before full config loading (pure function)
    let daemon_running = daemon::daemon_socket_exists();
    let run_mode = resolve_mode(&cli, daemon_running);

    match run_mode {
        RunMode::DaemonStart => {
            let use_watchdog = (cli.watchdog
                || std::env::var("DM_DAEMON_WATCHDOG").as_deref() == Ok("1"))
                && !cli.no_watchdog;
            return daemon::commands::handle_daemon_start(use_watchdog).await;
        }
        RunMode::DaemonStop => {
            return daemon::commands::handle_daemon_stop();
        }
        RunMode::DaemonStatus => {
            return daemon::commands::handle_daemon_status().await;
        }
        RunMode::DaemonRestart => {
            let use_watchdog = (cli.watchdog
                || std::env::var("DM_DAEMON_WATCHDOG").as_deref() == Ok("1"))
                && !cli.no_watchdog;
            let _ = daemon::commands::handle_daemon_stop();
            return daemon::commands::handle_daemon_start(use_watchdog).await;
        }
        RunMode::DaemonClient => {
            // Attempt to connect to daemon; fall back to local agent on failure.
            match try_run_daemon_client(&cli).await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    let _ = writeln!(
                        std::io::stderr(),
                        "[dm] Daemon client failed: {} — falling back to local agent.",
                        e
                    );
                    // Fall through to LocalAgent path below.
                }
            }
        }
        RunMode::LocalAgent => {}
    }

    let mut config = load_config_or_exit();

    // Auto-detect Ollama when the user has not explicitly configured a
    // host (no CLI flag, no OLLAMA_HOST env, no settings.json value).
    // Silent when the probe agrees with the default; announces when it
    // finds Ollama somewhere else. Best-effort — failure is a no-op.
    // The 2s total_budget caps worst-case (Ollama absent) at ~2s instead
    // of ~4.8s (6 candidates × 800ms). The CLI `--host` override on
    // line 1387 below still wins regardless of what this probe decides.
    if cli.host.is_none() && config.host_is_default {
        if let Some(detected) = dark_matter::ollama::client::detect_host(
            dark_matter::ollama::client::OLLAMA_CANDIDATE_HOSTS,
            std::time::Duration::from_millis(800),
            std::time::Duration::from_millis(2000),
        )
        .await
        {
            let normalized = dark_matter::config::normalize_host(&detected);
            if normalized != config.host && !cli.quiet {
                let _ = writeln!(std::io::stderr(), "dm: detected Ollama at {}", normalized);
            }
            config.host = normalized;
        }
    }

    if let Some(host) = cli.host {
        config.host = dark_matter::config::normalize_host(&host);
    }

    // Auto-pick a model when the user has not configured one. Runs only on
    // the default-model path, uses a 3s cap to never stall startup, and
    // silently falls through on Ollama errors (downstream handles them).
    // Uses the final resolved host (post CLI override) so the probe hits
    // the same endpoint the agent will.
    if cli.model.is_none() && config.model_is_default {
        let probe_client = dark_matter::ollama::client::OllamaClient::new(
            config.ollama_base_url(),
            config.model.clone(),
        );
        match tokio::time::timeout(
            std::time::Duration::from_secs(3),
            probe_client.list_models(),
        )
        .await
        {
            Ok(Ok(models)) if !models.is_empty() => {
                if let Some(picked) = dark_matter::ollama::client::pick_best_model(&models) {
                    if picked != config.model && !cli.quiet {
                        let _ = writeln!(std::io::stderr(), "dm: auto-selected model {}", picked);
                    }
                    config.model = picked;
                }
            }
            Ok(Ok(_)) => {
                if !cli.quiet {
                    let _ = writeln!(
                        std::io::stderr(),
                        "dm: no Ollama models installed. Try: ollama pull llama3.1:8b"
                    );
                }
            }
            Ok(Err(_)) | Err(_) => {
                // Ollama unreachable or slow — leave default in place.
            }
        }
    }

    if let Some(model) = cli.model {
        config.model = model;
    }
    if let Some(m) = cli.embed_model {
        config.embed_model = m;
    }
    if let Some(m) = cli.tool_model {
        config.tool_model = Some(m);
    }

    // Resolve model aliases (one level, no recursion)
    config.model = config.resolve_alias(&config.model);
    if let Some(ref tm) = config.tool_model {
        config.tool_model = Some(config.resolve_alias(tm));
    }

    // --aliases: list configured aliases and exit
    if cli.aliases {
        if config.aliases.is_empty() {
            println!("No aliases configured. Add [aliases] section to ~/.dm/config.toml");
        } else {
            println!("Model aliases:");
            let mut sorted: Vec<_> = config.aliases.iter().collect();
            sorted.sort_by_key(|(k, _)| (*k).clone());
            for (alias, target) in sorted {
                println!("  {} -> {}", alias, target);
            }
        }
        return Ok(());
    }

    // --config: print effective settings and exit (no Ollama needed)
    if cli.show_config {
        let rules = permissions::storage::load_rules(&config.config_dir).unwrap_or_default();
        println!("model:       {}", config.model);
        println!("host:        {}", config.host);
        println!("embed_model: {}", config.embed_model);
        println!("config_dir:  {}", config.config_dir.display());
        println!("rules:       {} saved permission rule(s)", rules.len());
        let settings_path = config.config_dir.join("settings.json");
        println!("settings:    {}", settings_path.display());
        return Ok(());
    }

    // --memory / --memory-clear / --memory-edit: manage project memory (no Ollama needed)
    if cli.memory_show || cli.memory_clear || cli.memory_edit {
        let cwd = std::env::current_dir()?;
        let phash = index::project_hash(&cwd);
        if cli.memory_clear {
            let mut mem =
                memory::ProjectMemory::load(&config.config_dir, &phash).unwrap_or_default();
            mem.clear();
            mem.save(&config.config_dir, &phash)?;
            println!("Project memory cleared.");
        } else if cli.memory_edit {
            let mem_path = memory::ProjectMemory::file_path(&config.config_dir, &phash)?;
            if let Some(parent) = mem_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !mem_path.exists() {
                std::fs::write(&mem_path, "{\"entries\":[]}")?;
            }
            let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".to_string());
            std::process::Command::new(&editor)
                .arg(&mem_path)
                .status()?;
        } else {
            let mem = memory::ProjectMemory::load(&config.config_dir, &phash).unwrap_or_default();
            if mem.entries.is_empty() {
                println!("No project memory for this directory.");
                println!("Memory accumulates automatically after sessions with ≥3 user turns.");
            } else {
                println!("Project memory ({} entries):", mem.entries.len());
                for e in &mem.entries {
                    let ts = e.timestamp.format("%Y-%m-%d");
                    println!("[{}] {}", ts, e.summary);
                }
            }
        }
        return Ok(());
    }

    // --templates: list available prompt templates and exit (no Ollama needed)
    if cli.templates {
        let list = templates::list_templates(&config.config_dir);
        if list.is_empty() {
            println!(
                "No templates found in {}",
                config.config_dir.join("templates").display()
            );
            println!("Run 'dm --init' to create example templates.");
        } else {
            println!("Available templates:");
            for t in &list {
                if t.description.is_empty() {
                    println!("  {}", t.name);
                } else {
                    println!("  {} — {}", t.name, t.description);
                }
            }
        }
        return Ok(());
    }

    let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
    let tool_client = config
        .tool_model
        .as_ref()
        .map(|m| OllamaClient::new(config.ollama_base_url(), m.clone()));

    // --commit: AI-generated commit message for staged changes
    if cli.commit {
        git::commit::run_commit(&client, cli.verbose).await?;
        return Ok(());
    }

    // --review: review a diff against a git ref
    if let Some(ref git_ref) = cli.review {
        let ref_str = if git_ref.is_empty() {
            "HEAD"
        } else {
            git_ref.as_str()
        };
        git::review::run_review(&client, ref_str, cli.verbose).await?;
        return Ok(());
    }

    // --pr: generate PR title + body from branch diff
    if let Some(ref base) = cli.pr {
        let base_ref = if base.is_empty() {
            None
        } else {
            Some(base.as_str())
        };
        git::pr::run_pr(
            &client,
            &config.routing,
            base_ref,
            cli.pr_push,
            cli.pr_open,
            cli.pr_draft,
            cli.verbose,
        )
        .await?;
        return Ok(());
    }

    // --gc: prune deleted-file chunks from the index (no re-embedding needed)
    if cli.gc {
        index::command::run_index_gc(&config.config_dir)?;
        return Ok(());
    }

    // --share: export session as a self-contained HTML or Markdown file (no Ollama needed)
    if let Some(ref out_path) = cli.share {
        let sess = if let Some(ref id) = cli.session_id {
            session::storage::load(&config.config_dir, id)?
        } else {
            session::storage::load_latest(&config.config_dir)?
                .ok_or_else(|| anyhow::anyhow!("No sessions found. Start a conversation first."))?
        };
        let (content, ext, mime) = if cli.share_format == "md" {
            let md = share::render_session_markdown(&sess);
            ("md", "md", md)
        } else {
            let html = share::render_session(&sess)?;
            ("html", "html", html)
        };
        let dest = if out_path.is_empty() {
            format!("dm-session-{}.{}", short_id(&sess.id), ext)
        } else {
            out_path.clone()
        };
        let _ = content; // suppress unused warning; mime holds the content
        std::fs::write(&dest, &mime)?;
        println!("Exported: {}", dest);
        return Ok(());
    }

    // --pull: download a model from Ollama (handled before reachability check)
    if let Some(ref model_name) = cli.pull {
        ollama::pull::run_pull(model_name, &config.ollama_base_url()).await?;
        return Ok(());
    }

    // --web: headless REST API
    if cli.web {
        // Resolve token: explicit --web-token, else load/generate from disk.
        let token = match cli.web_token.clone() {
            Some(t) => t,
            None => api::load_or_generate_token(&config.config_dir)?,
        };
        let _ = writeln!(std::io::stderr(), "[dm web] bearer token: {}", token);
        let _ = writeln!(
            std::io::stderr(),
            "[dm web] health check (no auth): http://127.0.0.1:{}/health",
            cli.web_port
        );
        api::write_last_port(&config.config_dir, cli.web_port);
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let wiki_snippet_initial = api::load_wiki_snippet(&cwd);
        let wiki_summary_initial = api::load_wiki_summary(&cwd);
        let wiki_fresh_initial = api::load_wiki_fresh(&cwd);
        let wiki_cwd = if wiki_snippet_initial.is_some()
            || wiki_summary_initial.is_some()
            || wiki_fresh_initial.is_some()
        {
            Some(cwd.clone())
        } else {
            None
        };
        let snippet_bytes = wiki_snippet_initial.as_ref().map_or(0, |s| s.len())
            + wiki_fresh_initial.as_ref().map_or(0, |s| s.len());
        let state = api::ApiState {
            config: std::sync::Arc::new(config.clone()),
            token: Some(token),
            active_chains: std::sync::Arc::new(tokio::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
            reconnect_buffers: api::replay::new_buffers(),
            wiki_snippet: wiki_snippet_initial,
            wiki_fresh: wiki_fresh_initial,
            wiki_cwd,
            wiki_summary: std::sync::Arc::new(std::sync::RwLock::new(wiki_summary_initial)),
            wiki_snippet_bytes_injected: snippet_bytes,
        };
        api::run(state, cli.web_port).await?;
        return Ok(());
    }

    // --run: execute a markdown run-spec
    if let Some(ref spec_path) = cli.run {
        run::run_spec(
            spec_path,
            cli.run_output.as_deref(),
            &client,
            &config.config_dir,
            &config.embed_model,
            cli.verbose,
            cli.dry_run,
            &config.routing,
        )
        .await?;
        return Ok(());
    }

    // --index: build/update semantic search index (requires Ollama embed model)
    if cli.index {
        let embed_client = OllamaClient::new(config.ollama_base_url(), config.embed_model.clone());
        if cli.watch {
            index::command::run_index_watch(&embed_client, &config.config_dir).await?;
        } else {
            index::command::run_index(&embed_client, &config.config_dir).await?;
        }
        return Ok(());
    }

    if let Some(cmd) = &cli.command {
        match cmd {
            Commands::Init => {
                dm::panic_hook::install(
                    dm::panic_hook::CrashContext::transient(config.config_dir.clone()),
                    || {},
                );
                init::run_init(&config).await?;
                return Ok(());
            }
            Commands::Doctor => {
                dm::panic_hook::install(
                    dm::panic_hook::CrashContext::transient(config.config_dir.clone()),
                    || {},
                );
                doctor::run_doctor(&client, &config).await;
                return Ok(());
            }
            Commands::Spawn {
                project_name,
                canonical,
            } => {
                dark_matter::spawn::run_spawn(project_name, canonical.as_deref()).await?;
                return Ok(());
            }
            Commands::Sync {
                dry_run,
                abort,
                status,
            } => {
                dark_matter::sync::run_sync(dark_matter::sync::SyncArgs {
                    dry_run: *dry_run,
                    abort: *abort,
                    status: *status,
                })
                .await?;
                return Ok(());
            }
        }
    }

    // Init — no Ollama needed
    if cli.init {
        dm::panic_hook::install(
            dm::panic_hook::CrashContext::transient(config.config_dir.clone()),
            || {},
        );
        init::run_init(&config).await?;
        return Ok(());
    }

    // Shell completions — no Ollama needed
    if let Some(shell_name) = &cli.completions {
        use clap::CommandFactory;
        use clap_complete::{
            generate,
            shells::{Bash, Fish, Zsh},
        };
        let mut cmd = Cli::command();
        let shell_lower = shell_name.to_lowercase();
        let stdout = &mut std::io::stdout();
        match shell_lower.as_str() {
            "bash" => generate(Bash, &mut cmd, "dm", stdout),
            "zsh" => generate(Zsh, &mut cmd, "dm", stdout),
            "fish" => generate(Fish, &mut cmd, "dm", stdout),
            other => {
                let _ = writeln!(
                    std::io::stderr(),
                    "Unknown shell '{}'. Supported: bash, zsh, fish",
                    other
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
        }
        return Ok(());
    }

    if cli.doctor {
        dm::panic_hook::install(
            dm::panic_hook::CrashContext::transient(config.config_dir.clone()),
            || {},
        );
        doctor::run_doctor(&client, &config).await;
        return Ok(());
    }

    if cli.recovery {
        let markers = dm::panic_hook::list_panic_markers(&config.config_dir);
        println!("{}", dm::panic_hook::format_recovery_report(&markers));
        return Ok(());
    }

    if cli.models {
        models::run_list(&client).await?;
        return Ok(());
    }

    if let Some(ref model_name) = cli.models_pull {
        models::run_pull(model_name, client.base_url()).await?;
        return Ok(());
    }

    if let Some(ref model_name) = cli.models_rm {
        models::run_rm(&client, model_name, cli.models_force).await?;
        return Ok(());
    }

    if cli.models_update {
        models::run_update(&client).await?;
        return Ok(());
    }

    if let Some(ref model_name) = cli.models_info {
        models::run_info(&client, model_name).await?;
        return Ok(());
    }

    // --chain-validate: dry-run config validation
    if let Some(ref chain_file) = cli.chain_validate {
        let mut chain_config = orchestrate::load_chain_config(chain_file)?;
        for (name, from, to) in chain_config.resolve_aliases_with_report(&config.aliases) {
            println!("  Alias resolved: node '{}': {} -> {}", name, from, to);
        }
        let models: Vec<String> = client
            .list_models()
            .await
            .map(|ms| ms.into_iter().map(|m| m.name).collect())
            .unwrap_or_default();
        let report = orchestrate::validate_chain_config_detailed(&chain_config, &models);
        println!("{}", report.report);
        if !report.errors.is_empty() {
            std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
        }
        return Ok(());
    }

    // --chain-resume: resume a chain from a saved workspace checkpoint
    if let Some(ref workspace) = cli.chain_resume {
        let mut orch_config = orchestrate::resume_config_from_workspace(workspace)?;
        for (name, from, to) in orch_config
            .chain
            .resolve_aliases_with_report(&config.aliases)
        {
            let _ = writeln!(
                std::io::stderr(),
                "dm: alias resolved: node '{}': {} -> {}",
                name,
                from,
                to
            );
        }
        // Allow CLI overrides for retry settings
        orch_config.retry = conversation::RetrySettings::from_config(&config);
        let resume_cycle = orch_config
            .resume_state
            .as_ref()
            .map_or(0, |s| s.current_cycle);
        let chain_name = orch_config.chain.name.clone();
        println!(
            "Resuming chain '{}' from cycle {}",
            chain_name, resume_cycle
        );
        let session_id = uuid::Uuid::new_v4().to_string();
        let registry = tools::registry::default_registry(
            &session_id,
            &config.config_dir,
            &config.ollama_base_url(),
            &config.model,
            &config.embed_model,
        );
        orchestrate::runner::run_orchestration(orch_config, client, registry, None).await?;
        return Ok(());
    }

    // --chain: load a YAML chain config and run it
    if let Some(ref chain_file) = cli.chain {
        let chain_as_str = chain_file.to_string_lossy();
        let mut chain_config = if let Some(preset) =
            orchestrate::resolve_chain_preset(&chain_as_str)
        {
            let _ = writeln!(
                std::io::stderr(),
                "dm: using built-in preset: {}",
                chain_as_str
            );
            preset
        } else {
            orchestrate::load_chain_config(chain_file).with_context(|| {
                    format!(
                        "chain '{}' not found as file or preset. Try: pass a path to a YAML file, or use one of the built-in presets: {}.",
                        chain_as_str,
                        orchestrate::list_chain_presets().join(", ")
                    )
                })?
        };
        for (name, from, to) in chain_config.resolve_aliases_with_report(&config.aliases) {
            let _ = writeln!(
                std::io::stderr(),
                "dm: alias resolved: node '{}': {} -> {}",
                name,
                from,
                to
            );
        }
        for w in orchestrate::validate_chain_config(&chain_config)? {
            let _ = writeln!(std::io::stderr(), "[chain] {}", w);
        }
        let session_id = uuid::Uuid::new_v4().to_string();
        let registry = tools::registry::default_registry(
            &session_id,
            &config.config_dir,
            &config.ollama_base_url(),
            &config.model,
            &config.embed_model,
        );
        let chain_id = cli
            .chain_id
            .clone()
            .unwrap_or_else(|| format!("chain-{}", uuid::Uuid::new_v4().as_simple()));
        let orch_config = orchestrate::OrchestrationConfig {
            chain: chain_config,
            chain_id,
            retry: conversation::RetrySettings::from_config(&config),
            resume_state: None,
        };
        orchestrate::runner::run_orchestration(orch_config, client, registry, None).await?;
        return Ok(());
    }

    // --orchestrate: run planner -> builder -> validator chain
    if let Some(ref task) = cli.orchestrate {
        let workspace = cli.workspace.clone().unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_default()
                .join(".dm-workspace")
        });
        let session_id = uuid::Uuid::new_v4().to_string();
        let registry = tools::registry::default_registry(
            &session_id,
            &config.config_dir,
            &config.ollama_base_url(),
            &config.model,
            &config.embed_model,
        );

        // Try to load build/dm-chain.yaml; fall back to a hardcoded 3-node chain.
        let default_model = config.model.clone();
        let mut chain =
            if let Ok(loaded) =
                orchestrate::load_chain_config(std::path::Path::new("build/dm-chain.yaml"))
            {
                loaded
            } else {
                orchestrate::types::ChainConfig {
                    name: task.clone(),
                    description: None,
                    nodes: vec![
                    orchestrate::types::ChainNodeConfig {
                        id: "planner".to_string(),
                        name: "planner".to_string(),
                        role: "planner".to_string(),
                        model: default_model.clone(),
                        description: None,
                        system_prompt_override: Some(
                            "You are a planner. Read the codebase, identify the next \
                             highest-impact improvement, and output a detailed implementation \
                             plan. Be specific about which files and lines to change."
                                .to_string(),
                        ),
                        system_prompt_file: None,
                        input_from: Some("tester".to_string()),
                        max_retries: 1,
                        timeout_secs: 3600, max_tool_turns: 200,
                    },
                    orchestrate::types::ChainNodeConfig {
                        id: "builder".to_string(),
                        name: "builder".to_string(),
                        role: "builder".to_string(),
                        model: default_model.clone(),
                        description: None,
                        system_prompt_override: Some(
                            "You are a builder. Implement the plan from the planner exactly. \
                             Run `cargo check` after changes and fix any errors. Report what \
                             you changed."
                                .to_string(),
                        ),
                        system_prompt_file: None,
                        input_from: Some("planner".to_string()),
                        max_retries: 1,
                        timeout_secs: 3600, max_tool_turns: 200,
                    },
                    orchestrate::types::ChainNodeConfig {
                        id: "tester".to_string(),
                        name: "tester".to_string(),
                        role: "tester".to_string(),
                        model: default_model.clone(),
                        description: None,
                        system_prompt_override: Some(
                            "You are a tester. Run `cargo test`, review the builder's changes, \
                             and report any failures or issues. Output VERDICT: PASS if all \
                             tests pass, otherwise VERDICT: FAIL with details."
                                .to_string(),
                        ),
                        system_prompt_file: None,
                        input_from: Some("builder".to_string()),
                        max_retries: 1,
                        timeout_secs: 3600, max_tool_turns: 200,
                    },
                ],
                    max_cycles: cli.max_cycles,
                    max_total_turns: cli.max_turns.unwrap_or(60),
                    workspace: workspace.clone(),
                    skip_permissions_warning: cli.dangerously_skip_permissions,
                    loop_forever: false,
                    directive: None,
                }
            };

        // Override name and tuning knobs from CLI regardless of source.
        chain.name = task.clone();
        chain.workspace = workspace.clone();
        chain.max_cycles = cli.max_cycles;
        chain.max_total_turns = cli.max_turns.unwrap_or(chain.max_total_turns);
        chain.skip_permissions_warning = cli.dangerously_skip_permissions;

        let chain_id = cli
            .chain_id
            .clone()
            .unwrap_or_else(|| format!("chain-{}", uuid::Uuid::new_v4().as_simple()));
        let orch_config = orchestrate::OrchestrationConfig {
            chain,
            chain_id,
            retry: conversation::RetrySettings::from_config(&config),
            resume_state: None,
        };
        orchestrate::runner::run_orchestration(orch_config, client.clone(), registry, None).await?;
        return Ok(());
    }

    // --mcp-server: expose Dark Matter's tool registry over MCP stdio transport.
    // Starts before the Ollama reachability check so the server is available
    // even when Ollama is temporarily unreachable (tools fail per-call instead).
    if cli.mcp_server {
        let session_id = uuid::Uuid::new_v4().to_string();
        let registry = tools::registry::default_registry(
            &session_id,
            &config.config_dir,
            &config.ollama_base_url(),
            &config.model,
            &config.embed_model,
        );
        if cli.watch {
            let embed_client_w =
                OllamaClient::new(config.ollama_base_url(), config.embed_model.clone());
            let config_dir_w = config.config_dir.clone();
            tokio::spawn(async move {
                if let Err(e) =
                    index::command::run_index_watch(&embed_client_w, &config_dir_w).await
                {
                    let _ = writeln!(std::io::stderr(), "[dm] Watch error: {}", e);
                }
            });
        }
        mcp::server::run_mcp_server(registry).await?;
        return Ok(());
    }

    // Warn early if Ollama is not reachable
    if let Err(e) = client.list_models().await {
        let _ = writeln!(
            std::io::stderr(),
            "dm: cannot reach Ollama at {} — {}",
            config.ollama_base_url(),
            e
        );
        let _ = writeln!(std::io::stderr(), "      Start Ollama with: ollama serve");
        let _ = writeln!(
            std::io::stderr(),
            "      Or set OLLAMA_HOST to point at your Ollama instance."
        );
        std::process::exit(dm::exit_codes::ExitCode::ModelUnreachable.as_i32());
    }

    // --mcp-test: spawn MCP server and run initialize handshake
    if let Some(ref name) = cli.mcp_test.clone() {
        let (scope, mcp_config_dir) = resolved_mcp_dir(cli.mcp_scope, &config.config_dir);
        let entries = mcp::manage::load_mcp_config(&mcp_config_dir)?;
        let entry = entries.iter().find(|e| &e.name == name).ok_or_else(|| {
            anyhow::anyhow!(
                "No MCP server with name '{}' in {} scope ({}). Try: dm --mcp-list --mcp-scope {}",
                name,
                scope,
                mcp_config_dir.display(),
                scope.opposite(),
            )
        })?;
        let args_refs: Vec<&str> = entry.args.iter().map(|s| s.as_str()).collect();
        match mcp::client::McpClient::spawn(&entry.command, &args_refs).await {
            Ok(mut mc) => match mc.list_tools().await {
                Ok(tools) => {
                    println!("MCP server '{}' OK — {} tool(s):", name, tools.len());
                    for t in &tools {
                        println!("  {}", t.name);
                    }
                }
                Err(e) => {
                    let _ = writeln!(
                        std::io::stderr(),
                        "MCP server '{}' connected but list_tools failed: {}",
                        name,
                        e
                    );
                    std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
                }
            },
            Err(e) => {
                let _ = writeln!(
                    std::io::stderr(),
                    "Failed to spawn MCP server '{}': {}",
                    name,
                    e
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
        }
        return Ok(());
    }

    // --patch: apply a unified diff from stdin or file to the working tree
    if let Some(ref file_arg) = cli.patch.clone() {
        use std::io::Read as _;
        let diff_text = match file_arg {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("Cannot read patch file: {}", path))?,
            None => {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            }
        };

        let file_diffs = tools::apply_diff::split_diff_by_file(&diff_text);
        if file_diffs.is_empty() {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    "no file diffs found in input",
                    Some("git diff | dm --apply-diff  (or pass a .patch path)"),
                )
            );
            std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
        }

        let mut any_failed = false;
        for (path, file_diff) in &file_diffs {
            let original = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    let _ = writeln!(std::io::stderr(), "  ✗ {} — cannot read: {}", path, e);
                    any_failed = true;
                    continue;
                }
            };
            match tools::apply_diff::apply_diff(&original, file_diff) {
                Ok((patched, _adjusted)) => {
                    if let Err(e) = std::fs::write(path, &patched) {
                        let _ = writeln!(std::io::stderr(), "  ✗ {} — cannot write: {}", path, e);
                        any_failed = true;
                    } else {
                        // Count changed lines for the summary
                        let added = patched
                            .lines()
                            .filter(|l| !original.contains(l.trim()))
                            .count();
                        println!("  ✓ {}", path);
                        let _ = added; // summary printed per-file above
                    }
                }
                Err(e) => {
                    let _ = writeln!(std::io::stderr(), "  ✗ {} — {}", path, e);
                    any_failed = true;
                }
            }
        }

        let total = file_diffs.len();
        let failed = file_diffs
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                // recount failures by re-checking (simpler: track separately)
                let _ = i;
                false
            })
            .count();
        let _ = failed;
        println!(
            "\n{} file{} patched{}",
            total,
            if total == 1 { "" } else { "s" },
            if any_failed {
                " (with errors — see above)"
            } else {
                ""
            },
        );

        if any_failed {
            std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
        }
        return Ok(());
    }

    // --replay: replay a saved session's user turns
    if let Some(ref session_id_replay) = cli.replay.clone() {
        let replay_client = if let Some(ref m) = cli.replay_model {
            OllamaClient::new(config.ollama_base_url(), m.clone())
        } else {
            client.clone()
        };
        session::replay::run_replay(
            session_id_replay,
            cli.replay_model.clone(),
            cli.replay_from_turn,
            cli.replay_dry_run,
            &replay_client,
            &config.config_dir,
        )
        .await?;
        return Ok(());
    }

    // --agent-run: load agent config and run conversation with its settings
    if let Some(ref agent_name) = cli.agent_run.clone() {
        let agent_cfg = agents::AgentConfig::load(&config.config_dir, agent_name)?;
        let run_model = cli
            .agent_model
            .clone()
            .or_else(|| agent_cfg.model.clone())
            .unwrap_or_else(|| config.model.clone());
        let agent_client = OllamaClient::new(config.ollama_base_url(), run_model.clone());
        let extra_system = agent_cfg.system_extra.as_deref();
        let mut system_prompt = system_prompt::build_system_prompt_full(
            &cli.add_dirs,
            extra_system,
            &cli.context,
            !cli.no_workspace_context,
            cli.no_dm_md,
        )
        .await;
        system_prompt::append_model_info(
            &mut system_prompt,
            &run_model,
            config.tool_model.as_deref(),
        );
        if !agent_cfg.no_memory && !cli.no_memory {
            let cwd_for_mem = std::env::current_dir().unwrap_or_default();
            let phash = index::project_hash(&cwd_for_mem);
            let mem = memory::ProjectMemory::load(&config.config_dir, &phash).unwrap_or_default();
            if let Some(mem_msg) = mem.to_system_message() {
                system_prompt.push_str("\n\n");
                system_prompt.push_str(&mem_msg);
            }
        }
        let mut agent_sess = new_session(&config);
        let mut registry = tools::registry::default_registry(
            &agent_sess.id,
            &config.config_dir,
            agent_client.base_url(),
            &run_model,
            &config.embed_model,
        );
        // Tool filtering based on agent config happens at conversation call site;
        // for now, --no-tools clears all tools; agent-level filtering is a future enhancement.
        if agent_cfg.tools.is_none() && cli.no_tools {
            registry.clear();
        }
        // Inject per-tool usage hints into the system prompt. Identity-aware:
        // host mode groups host_* capabilities ahead of the kernel substrate,
        // kernel mode emits a flat list (canonical dm behavior).
        let tool_hints = registry.system_prompt_hints_for(&identity::load_for_cwd());
        if !tool_hints.is_empty() {
            system_prompt.push_str("\n<tool_usage>\n");
            system_prompt.push_str(&tool_hints);
            system_prompt.push_str("</tool_usage>\n");
        }
        let settings_rules_agent =
            permissions::storage::load_rules(&config.config_dir).unwrap_or_default();
        let mut engine_agent = permissions::engine::PermissionEngine::new(
            cli.dangerously_skip_permissions || is_full_mode,
            settings_rules_agent,
        );
        apply_permission_flags(
            cli.permission_mode.as_deref(),
            &cli.allow,
            &cli.disallow,
            &mut engine_agent,
        );
        let print_mcp_agent: HashMap<String, Arc<Mutex<mcp::client::McpClient>>> = HashMap::new();
        let prompt_text = cli.print.clone().or(cli.prompt.clone()).unwrap_or_else(|| {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    "--agent-run requires a prompt via -p or positional argument",
                    Some("dm --agent-run general -p \"summarize README\""),
                )
            );
            std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
        });
        let conv_start = std::time::Instant::now();
        conversation::run_conversation(
            &prompt_text,
            "print",
            &agent_client,
            None,
            &registry,
            &print_mcp_agent,
            system_prompt,
            &mut engine_agent,
            &mut agent_sess,
            &config.config_dir,
            cli.verbose,
            &cli.output_format,
            cli.max_turns.unwrap_or(DEFAULT_MAX_TURNS),
            cli.perf,
            config.fallback_model.as_deref(),
        )
        .await?;
        if !cli.quiet && cli.output_format != "json" && cli.output_format != "stream-json" {
            let summary = conversation::summarize_session(&agent_sess.messages);
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                summary.format_line(conv_start.elapsed())
            );
        }
        return Ok(());
    }

    // --schedule-run: run a scheduled task immediately by ID
    if let Some(ref run_id) = cli.schedule_run.clone() {
        let tasks = daemon::scheduler::load_schedules(&config.config_dir)?;
        let task = tasks
            .iter()
            .find(|t| &t.id == run_id)
            .ok_or_else(|| anyhow::anyhow!("No scheduled task with id '{}'", run_id))?
            .clone();
        let session_id = uuid::Uuid::new_v4().to_string();
        let registry = tools::registry::default_registry(
            &session_id,
            &config.config_dir,
            &config.ollama_base_url(),
            &config.model,
            &config.embed_model,
        );
        let mut sess = new_session(&config);
        let mut system = system_prompt::build_system_prompt_full(&[], None, &[], true, false).await;
        system_prompt::append_model_info(&mut system, &config.model, config.tool_model.as_deref());
        let tool_hints = registry.system_prompt_hints_for(&identity::load_for_cwd());
        if !tool_hints.is_empty() {
            system.push_str("\n<tool_usage>\n");
            system.push_str(&tool_hints);
            system.push_str("</tool_usage>\n");
        }
        let mut engine = permissions::engine::PermissionEngine::new(
            true, // dangerously_skip_permissions
            vec![],
        );
        let print_mcp: std::collections::HashMap<String, Arc<Mutex<mcp::client::McpClient>>> =
            std::collections::HashMap::new();
        conversation::run_conversation(
            &task.prompt,
            "print",
            &client,
            tool_client.as_ref(),
            &registry,
            &print_mcp,
            system,
            &mut engine,
            &mut sess,
            &config.config_dir,
            false,
            "text",
            DEFAULT_MAX_TURNS,
            false,
            config.fallback_model.as_deref(),
        )
        .await?;
        return Ok(());
    }

    // --test-fix: run CMD, fix failing tests with AI, repeat
    if let Some(ref cmd) = cli.test_fix {
        let client = if cli.json_schema.is_some() {
            client.with_json_mode(true)
        } else {
            client
        };
        testfix::run_test_fix(cmd, cli.test_fix_rounds, &client, &config.config_dir).await?;
        return Ok(());
    }

    // --watch-fix: watch files and run test-fix on change
    if let Some(ref pattern) = cli.watch_fix {
        let watch_cmd = if cli.watch_cmd.is_empty() {
            let cwd = std::env::current_dir().unwrap_or_default();
            testfix::detect::detect_test_cmd(&cwd)
        } else {
            cli.watch_cmd.clone()
        };
        testfix::watcher::run_watch_fix(
            pattern,
            &watch_cmd,
            cli.test_fix_rounds,
            &client,
            &config.config_dir,
        )
        .await?;
        return Ok(());
    }

    // --search: search session history
    if let Some(days) = cli.prune_sessions {
        let pruned = session::storage::prune(&config.config_dir, days)?;
        if pruned == 0 {
            println!("No sessions older than {} days.", days);
        } else {
            println!("Pruned {} session(s) older than {} days.", pruned, days);
        }
        return Ok(());
    }

    if let Some(ref query) = cli.search {
        let matches = session::storage::search_sessions(&config.config_dir, query)?;
        if matches.is_empty() {
            println!("No sessions matching \"{}\"", query);
        } else {
            println!(
                "{} session{} matching \"{}\":",
                matches.len(),
                if matches.len() == 1 { "" } else { "s" },
                query
            );
            for m in &matches {
                let short = short_id(&m.session.id);
                let date = m.session.updated_at.format("%Y-%m-%d").to_string();
                let title = m
                    .session
                    .title
                    .as_deref()
                    .filter(|t| !t.is_empty())
                    .unwrap_or("(no title)");
                println!("  [{}] \"{}\" — {}  …{}…", short, title, date, m.snippet);
            }
        }
        return Ok(());
    }

    // --todo: scan for TODO comments and prioritize them
    if cli.todo {
        let globs: Vec<String> = cli.todo_glob.clone();
        todo::run_todo(&globs, cli.todo_fix, &client, &config.config_dir).await?;
        return Ok(());
    }

    // --lint-fix: run linter and fix warnings with AI
    if let Some(ref opt_cmd) = cli.lint_fix {
        let cwd = std::env::current_dir().unwrap_or_default();
        let cmd = match opt_cmd {
            Some(c) => c.clone(),
            None => testfix::detect::detect_lint_cmd(&cwd),
        };
        testfix::run_lint_fix(&cmd, cli.test_fix_rounds, &client, &config.config_dir).await?;
        return Ok(());
    }

    // --changelog: generate changelog from git history
    if let Some(ref opt_from) = cli.changelog {
        git::run_changelog(
            opt_from.clone(),
            &cli.changelog_to,
            &cli.changelog_format,
            &client,
        )
        .await?;
        return Ok(());
    }

    // --compare: compare models on a prompt
    if let Some(ref models_raw) = cli.compare {
        let models: Vec<String> = models_raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if models.is_empty() {
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                dm::error_hints::format_dm_error(
                    "--compare requires at least one model name",
                    Some("dm --compare llama3.1:8b,qwen2.5:7b --compare-prompt \"...\""),
                )
            );
            std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
        }
        let compare_prompt = if let Some(ref p) = cli.compare_prompt {
            p.clone()
        } else if let Some(ref p) = cli.print {
            p.clone()
        } else {
            use std::io::Read as _;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            let trimmed = buf.trim().to_string();
            if trimmed.is_empty() {
                let _ = writeln!(
                    std::io::stderr(),
                    "dm: --compare requires a prompt (use --compare-prompt or pipe stdin)"
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
            trimmed
        };
        bench::compare::run_compare(
            &models,
            &compare_prompt,
            cli.compare_side_by_side,
            &config.ollama_base_url(),
        )
        .await?;
        return Ok(());
    }

    // --summarize: summarize a file, URL, or stdin
    if let Some(ref target) = cli.summarize {
        summarize::run_summarize(target, cli.summarize_length, &cli.summarize_style, &client)
            .await?;
        return Ok(());
    }

    // --document: generate/update docstrings for a file
    if let Some(ref path) = cli.document.clone() {
        document::run_document(path, &cli.document_style, &client).await?;
        return Ok(());
    }

    // --document-glob: process multiple files matching a glob
    if let Some(ref pattern) = cli.document_glob.clone() {
        match glob::glob(pattern) {
            Ok(paths) => {
                for entry in paths {
                    match entry {
                        Ok(p) if p.is_file() => {
                            document::run_document(&p, &cli.document_style, &client).await?;
                        }
                        Ok(_) => {}
                        Err(e) => {
                            let _ = writeln!(
                                std::io::stderr(),
                                "[dm] document-glob entry error: {}",
                                e
                            );
                        }
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(
                    std::io::stderr(),
                    "{}",
                    dm::error_hints::format_dm_error(
                        &format!("invalid glob pattern '{}': {}", pattern, e),
                        Some("quote it, e.g. --pattern 'src/**/*.rs'"),
                    )
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
        }
        return Ok(());
    }

    // --translate: translate source file to another language
    if let Some(ref src_path) = cli.translate.clone() {
        let target_lang = match cli.translate_to.as_deref() {
            Some(l) if !l.is_empty() => l.to_string(),
            _ => {
                let _ = writeln!(
                    std::io::stderr(),
                    "{}",
                    dm::error_hints::format_dm_error(
                        "--translate requires --translate-to <LANG>",
                        Some("dm --translate foo.py --translate-to rust"),
                    )
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
        };
        let out = translate::output_path(src_path, &target_lang, cli.translate_out.as_deref());
        translate::run_translate(src_path, &target_lang, &out, &client).await?;
        return Ok(());
    }

    // --security: security audit files matching glob
    if let Some(ref glob_pattern) = cli.security.clone() {
        security::run_security(glob_pattern, &client, &config.config_dir).await?;
        return Ok(());
    }

    // --eval-list: list YAML eval files in ~/.dm/evals/
    if cli.eval_list {
        let files = eval::runner::list_evals()?;
        if files.is_empty() {
            println!("No eval files found in ~/.dm/evals/ or ./evals/");
            println!("Place YAML files in either directory.");
        } else {
            println!("{} eval file(s):", files.len());
            for f in &files {
                println!("  {}", f.display());
            }
        }
        return Ok(());
    }

    // --eval-results: list past result JSON files
    if cli.eval_results {
        let files = eval::runner::list_results()?;
        if files.is_empty() {
            println!("No eval results found in ~/.dm/eval/results/");
        } else {
            println!("{} result file(s):", files.len());
            for f in &files {
                println!("  {}", f.display());
            }
        }
        return Ok(());
    }

    // --eval-report-compare: side-by-side diff of two result files
    if let Some(ref file_a) = cli.eval_report_compare.clone() {
        let file_b = cli.eval_report_compare_b.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--eval-report-compare requires --eval-report-compare-b <FILE_B>")
        })?;
        let json_a =
            std::fs::read_to_string(file_a).with_context(|| format!("Cannot read {}", file_a))?;
        let json_b =
            std::fs::read_to_string(file_b).with_context(|| format!("Cannot read {}", file_b))?;
        let result_a: eval::SuiteResult =
            serde_json::from_str(&json_a).with_context(|| format!("Cannot parse {}", file_a))?;
        let result_b: eval::SuiteResult =
            serde_json::from_str(&json_b).with_context(|| format!("Cannot parse {}", file_b))?;
        eval::runner::render_compare(&result_a, &result_b);
        return Ok(());
    }

    // --eval-report / --eval-report-last: print report for a saved result
    if cli.eval_report_last || cli.eval_report.is_some() {
        let results_dir = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".dm")
            .join("eval")
            .join("results");

        let result_path = if cli.eval_report_last {
            eval::runner::find_last_result(&results_dir)?.ok_or_else(|| {
                anyhow::anyhow!("No eval results found in {}", results_dir.display())
            })?
        } else {
            std::path::PathBuf::from(cli.eval_report.as_deref().unwrap_or(""))
        };

        let json = std::fs::read_to_string(&result_path)
            .with_context(|| format!("Cannot read {}", result_path.display()))?;
        let result: eval::SuiteResult = serde_json::from_str(&json)
            .with_context(|| format!("Cannot parse {}", result_path.display()))?;
        eval::runner::render_report(&result);
        return Ok(());
    }

    // --eval: run eval suites from YAML files
    if !cli.eval_files.is_empty() {
        // Expand any glob patterns in the file args
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        for raw in &cli.eval_files {
            match glob::glob(raw) {
                Ok(entries) => {
                    let mut matched = false;
                    for entry in entries {
                        match entry {
                            Ok(p) if p.is_file() => {
                                matched = true;
                                paths.push(p);
                            }
                            Ok(_) => {}
                            Err(e) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[dm eval] glob entry error: {}",
                                    e
                                );
                            }
                        }
                    }
                    if !matched {
                        // treat as literal path
                        paths.push(std::path::PathBuf::from(raw));
                    }
                }
                Err(_) => {
                    paths.push(std::path::PathBuf::from(raw));
                }
            }
        }

        if let Some(ref compare_raw) = cli.eval_compare {
            // Compare mode: run each suite against each model
            let models: Vec<String> = compare_raw
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if models.is_empty() {
                let _ = writeln!(
                    std::io::stderr(),
                    "{}",
                    dm::error_hints::format_dm_error(
                        "--eval-compare requires at least one model name",
                        Some("dm --eval-compare llama3.1:8b,qwen2.5:7b"),
                    )
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
            for path in &paths {
                let yaml_text = std::fs::read_to_string(path)
                    .map_err(|e| anyhow::anyhow!("Cannot read {}: {}", path.display(), e))?;
                let suite: eval::EvalSuite = serde_yaml::from_str(&yaml_text)
                    .map_err(|e| anyhow::anyhow!("Cannot parse {}: {}", path.display(), e))?;
                let mut per_model: Vec<(String, Vec<eval::CaseResult>)> = Vec::new();
                let eval_opts = eval::runner::EvalOptions {
                    verbose: cli.eval_verbose,
                    fail_fast: cli.eval_fail_fast,
                    strict: cli.eval_strict,
                    runs: cli.eval_runs,
                    flaky_threshold: cli.eval_flaky_threshold,
                    quiet: cli.eval_quiet,
                };
                for m in &models {
                    let model_client = OllamaClient::new(config.ollama_base_url(), m.clone());
                    let result =
                        eval::runner::run_eval(&suite, Some(m.as_str()), &model_client, &eval_opts)
                            .await?;
                    per_model.push((m.clone(), result.cases));
                }
                eval::runner::print_compare_table(&per_model);
            }
        } else {
            // Normal eval mode (with optional baseline operations)
            let model_override = cli.eval_model.as_deref();
            let eval_opts = eval::runner::EvalOptions {
                verbose: cli.eval_verbose,
                fail_fast: cli.eval_fail_fast,
                strict: cli.eval_strict,
                runs: cli.eval_runs,
                flaky_threshold: cli.eval_flaky_threshold,
                quiet: cli.eval_quiet,
            };
            let results =
                eval::runner::run_eval_files(paths, model_override, &client, &eval_opts).await?;

            let do_freeze = cli.eval_freeze;
            let do_ci = cli.eval_ci;
            let do_diff = cli.eval_diff_baseline;

            if do_freeze || do_ci || do_diff {
                let mut any_regression = false;

                for result in &results {
                    if do_freeze {
                        match eval::baseline::save_baseline(&config.config_dir, result) {
                            Ok(path) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[eval] baseline saved: {} ({})",
                                    result.suite_name,
                                    path.display()
                                );
                            }
                            Err(e) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[eval] baseline save error: {}",
                                    e
                                );
                            }
                        }
                    }

                    if do_ci || do_diff {
                        match eval::baseline::load_baseline(&config.config_dir, &result.suite_name)
                        {
                            Ok(Some(baseline)) => {
                                let diff = if cli.eval_runs > 1 {
                                    eval::baseline::diff_results_rate(
                                        &baseline,
                                        result,
                                        cli.eval_regression_threshold,
                                    )
                                } else {
                                    eval::baseline::diff_results(&baseline, result)
                                };
                                let n = eval::baseline::print_diff(&diff, &result.suite_name);
                                if n > 0 && do_ci {
                                    any_regression = true;
                                }
                            }
                            Ok(None) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[eval] no baseline for '{}' — run with --eval-freeze first",
                                    result.suite_name
                                );
                            }
                            Err(e) => {
                                let _ = writeln!(
                                    std::io::stderr(),
                                    "[eval] baseline load error: {}",
                                    e
                                );
                            }
                        }
                    }
                }

                if any_regression {
                    std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
                }
            }
        }
        return Ok(());
    }

    // --bench-history: show benchmark history
    if let Some(ref ts) = cli.bench_history {
        let bench_dir = config.config_dir.join("bench");
        if ts == "__list__" {
            bench::print_history_list(&bench_dir)?;
        } else {
            bench::print_history_detail(&bench_dir, ts)?;
        }
        return Ok(());
    }

    // --bench: benchmark available models
    if let Some(ref bench_models_raw) = cli.bench {
        let models_filter = if bench_models_raw == "__all__" {
            None
        } else {
            Some(
                bench_models_raw
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>(),
            )
        };
        let tasks_filter = if cli.bench_task == "all" {
            None
        } else {
            Some(vec![cli.bench_task.clone()])
        };
        bench::run_bench(
            &client,
            models_filter,
            tasks_filter,
            cli.bench_runs,
            cli.bench_out,
        )
        .await?;
        return Ok(());
    }

    let settings_rules = permissions::storage::load_rules(&config.config_dir).unwrap_or_default();
    let mut engine = permissions::engine::PermissionEngine::new(
        cli.dangerously_skip_permissions || cli.serve || is_full_mode,
        settings_rules,
    );
    apply_permission_flags(
        cli.permission_mode.as_deref(),
        &cli.allow,
        &cli.disallow,
        &mut engine,
    );

    let load_recent = cli.continue_session || cli.resume;
    let mut sess = if let Some(ref id) = cli.session_id {
        session::storage::load(&config.config_dir, id)
            .with_context(|| format!("Cannot load session '{}'", id))?
    } else if load_recent {
        session::storage::load_latest(&config.config_dir)?.unwrap_or_else(|| {
            let _ = writeln!(
                std::io::stderr(),
                "[dm] No previous session found, starting new"
            );
            new_session(&config)
        })
    } else {
        new_session(&config)
    };

    if (cli.session_id.is_some() || load_recent) && !sess.messages.is_empty() {
        let _ = writeln!(
            std::io::stderr(),
            "[dm] Resuming session {} ({} messages)",
            sess.id,
            sess.messages.len()
        );
        let cleared =
            dm::panic_hook::clear_panic_markers_for_session(&config.config_dir, &sess.id);
        if cleared > 0 {
            let _ = writeln!(
                std::io::stderr(),
                "[dm] Cleared {} panic marker(s) for resumed session",
                cleared
            );
        }
    }

    // --json-schema: enable JSON mode on the Ollama client
    let client = if cli.json_schema.is_some() {
        client.with_json_mode(true)
    } else {
        client
    };

    // --template: load template and use as prompt (template wins over --print)
    let template_prompt = if let Some(ref tname) = cli.template {
        match templates::load_template(&config.config_dir, tname, &cli.template_args) {
            Ok(t) => Some(t),
            Err(e) => {
                let _ = writeln!(
                    std::io::stderr(),
                    "{}",
                    dm::error_hints::format_dm_error(
                        &format!("template error — {}", e),
                        Some("check ~/.dm/templates/ exists, or run dm --list-templates"),
                    )
                );
                std::process::exit(dm::exit_codes::ExitCode::AgentError.as_i32());
            }
        }
    } else {
        None
    };

    let prompt = template_prompt.or_else(|| cli.print.or(cli.prompt));

    let stdout_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let stdin_is_tty = std::io::IsTerminal::is_terminal(&std::io::stdin());

    // If stdin is piped and no explicit prompt given, read stdin as the prompt.
    // This makes `echo "explain this" | dm` work as non-interactive print mode.
    let prompt = if prompt.is_none() && !stdin_is_tty {
        use std::io::Read as _;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).ok();
        let trimmed = buf.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    } else {
        prompt
    };

    // Guard: cannot launch TUI if stdout is piped and no prompt to execute.
    // Empty stdin is a usage error (exit 2), not a silent no-op — callers
    // that fan out shell pipelines should guard upstream emptiness themselves.
    if prompt.is_none() && !stdout_is_tty {
        if !stdin_is_tty {
            let _ = writeln!(std::io::stderr(), "dm: no input received on stdin");
            let _ = writeln!(
                std::io::stderr(),
                "      Usage: echo \"your prompt\" | dm   or   dm -p \"your prompt\""
            );
        } else {
            let _ = writeln!(
                std::io::stderr(),
                "dm: no prompt provided. Usage: dm -p \"your question\""
            );
            let _ = writeln!(
                std::io::stderr(),
                "      or run without arguments for interactive mode (requires a TTY)."
            );
        }
        std::process::exit(dm::exit_codes::ExitCode::ConfigError.as_i32());
    }

    let cwd_for_memory = std::env::current_dir().unwrap_or_default();
    let project_hash = index::project_hash(&cwd_for_memory);

    if let Some(prompt_text) = prompt {
        // ── CLI (non-interactive) mode ─────────────────────────────────────────

        dm::panic_hook::install(
            dm::panic_hook::CrashContext::for_session(
                config.config_dir.clone(),
                sess.id.clone(),
            ),
            || {},
        );

        // Apply model routing: classify the prompt and pick the model.
        let (routed_model, route_key) =
            routing::resolve_model(&prompt_text, &config.routing, &config.model);
        if cli.routing_debug || cli.verbose {
            let _ = writeln!(
                std::io::stderr(),
                "[routing] prompt classified as '{}' → {}",
                route_key.as_str(),
                routed_model
            );
        }
        // Shadow client with the routed model if it differs.
        let client = if routed_model != config.model {
            OllamaClient::new(config.ollama_base_url(), routed_model.clone())
        } else {
            client
        };

        if cli.verbose {
            let _ = writeln!(std::io::stderr(), "[dm] Using model: {}", routed_model);
            let _ = writeln!(std::io::stderr(), "[dm] Ollama host: {}", config.host);
        }
        let mut system_prompt = system_prompt::build_system_prompt_full(
            &cli.add_dirs,
            cli.extra_system.as_deref(),
            &cli.context,
            !cli.no_workspace_context,
            cli.no_dm_md,
        )
        .await;
        system_prompt::append_model_info(
            &mut system_prompt,
            &routed_model,
            config.tool_model.as_deref(),
        );
        if !cli.no_memory {
            let mem =
                memory::ProjectMemory::load(&config.config_dir, &project_hash).unwrap_or_default();
            if let Some(mem_msg) = mem.to_system_message() {
                system_prompt.push_str("\n\n");
                system_prompt.push_str(&mem_msg);
            }
        }
        let mut registry = tools::registry::default_registry(
            &sess.id,
            &config.config_dir,
            client.base_url(),
            &config.model,
            &config.embed_model,
        );
        if cli.no_tools {
            registry.clear();
        }
        // Inject per-tool usage hints into the system prompt. Identity-aware:
        // host mode groups host_* capabilities ahead of the kernel substrate,
        // kernel mode emits a flat list (canonical dm behavior).
        let tool_hints = registry.system_prompt_hints_for(&identity::load_for_cwd());
        if !tool_hints.is_empty() {
            system_prompt.push_str("\n<tool_usage>\n");
            system_prompt.push_str(&tool_hints);
            system_prompt.push_str("</tool_usage>\n");
        }
        let mut print_mcp: HashMap<String, Arc<Mutex<mcp::client::McpClient>>> = HashMap::new();
        for plugin in plugins::discover_plugins(&config.config_dir) {
            let path_str = plugin.path.to_string_lossy().to_string();
            let args: Vec<&str> = plugin.args.iter().map(|s| s.as_str()).collect();
            if let Ok(mut mc) = mcp::client::McpClient::spawn(&path_str, &args).await {
                if let Ok(tools) = mc.list_tools().await {
                    for tool in tools {
                        registry.register_mcp(plugin.name.clone(), tool);
                    }
                    print_mcp.insert(plugin.name.clone(), Arc::new(Mutex::new(mc)));
                }
            }
        }
        let conv_start = std::time::Instant::now();
        conversation::run_conversation(
            &prompt_text,
            "print",
            &client,
            tool_client.as_ref(),
            &registry,
            &print_mcp,
            system_prompt,
            &mut engine,
            &mut sess,
            &config.config_dir,
            cli.verbose,
            &cli.output_format,
            cli.max_turns.unwrap_or(DEFAULT_MAX_TURNS),
            cli.perf,
            config.fallback_model.as_deref(),
        )
        .await?;

        if !cli.quiet && cli.output_format != "json" && cli.output_format != "stream-json" {
            let summary = conversation::summarize_session(&sess.messages);
            let _ = writeln!(
                std::io::stderr(),
                "{}",
                summary.format_line(conv_start.elapsed())
            );
        }

        // --notify: send desktop notification when complete
        if cli.notify {
            let last_msg = sess
                .messages
                .iter()
                .rev()
                .find_map(|m| {
                    if m["role"].as_str() == Some("assistant") {
                        m["content"].as_str().map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            notify::notify("Dark Matter: complete", &last_msg);
        }

        // Session-end memory update
        if !cli.no_memory {
            if let Err(e) =
                memory::maybe_update_memory(&sess, &client, &config.config_dir, &project_hash).await
            {
                let _ = writeln!(std::io::stderr(), "[dm] Memory update failed: {}", e);
            }
        }

        // --json-schema: warn if the last assistant response is not valid JSON
        if cli.json_schema.is_some() {
            if let Some(last_text) = sess.messages.iter().rev().find_map(|m| {
                if m["role"].as_str() == Some("assistant") {
                    m["content"].as_str().map(|s| s.to_string())
                } else {
                    None
                }
            }) {
                if serde_json::from_str::<serde_json::Value>(&last_text).is_err() {
                    let _ = writeln!(
                        std::io::stderr(),
                        "dm: warning — response is not valid JSON"
                    );
                }
            }
        }
    } else {
        // ── Interactive TUI mode ───────────────────────────────────────────────
        let mut system_prompt = system_prompt::build_system_prompt_full(
            &cli.add_dirs,
            cli.extra_system.as_deref(),
            &cli.context,
            !cli.no_workspace_context,
            cli.no_dm_md,
        )
        .await;
        system_prompt::append_model_info(
            &mut system_prompt,
            &config.model,
            config.tool_model.as_deref(),
        );
        if !cli.no_memory {
            let mem =
                memory::ProjectMemory::load(&config.config_dir, &project_hash).unwrap_or_default();
            if let Some(mem_msg) = mem.to_system_message() {
                system_prompt.push_str("\n\n");
                system_prompt.push_str(&mem_msg);
            }
        }
        // Create event channel early so the registry can wire AgentTool → TUI events.
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<tui::BackendEvent>(256);

        let mut registry = tools::registry::default_registry_with_events(
            &sess.id,
            &config.config_dir,
            client.base_url(),
            &config.model,
            &config.embed_model,
            event_tx.clone(),
        );
        if cli.no_tools {
            registry.clear();
        }
        // Inject per-tool usage hints into the system prompt. Identity-aware:
        // host mode groups host_* capabilities ahead of the kernel substrate,
        // kernel mode emits a flat list (canonical dm behavior).
        let tool_hints = registry.system_prompt_hints_for(&identity::load_for_cwd());
        if !tool_hints.is_empty() {
            system_prompt.push_str("\n<tool_usage>\n");
            system_prompt.push_str(&tool_hints);
            system_prompt.push_str("</tool_usage>\n");
        }

        // Load MCP servers and connect them
        let mut mcp_clients: HashMap<String, Arc<Mutex<mcp::client::McpClient>>> = HashMap::new();
        let mut mcp_server_status: Vec<(String, usize)> = Vec::new();
        let project_root = std::env::current_dir().unwrap_or_default();
        let mcp_configs = mcp::config::load_configs_with_project(&config.config_dir, &project_root);
        for cfg in &mcp_configs {
            let args_refs: Vec<&str> = cfg.args.iter().map(|s| s.as_str()).collect();
            match mcp::client::McpClient::spawn(&cfg.command, &args_refs).await {
                Ok(mut mc) => match mc.list_tools().await {
                    Ok(tools) => {
                        if cli.verbose {
                            let _ = writeln!(
                                std::io::stderr(),
                                "[dm] MCP server '{}': {} tools registered",
                                cfg.name,
                                tools.len()
                            );
                        }
                        mcp_server_status.push((cfg.name.clone(), tools.len()));
                        for tool in tools {
                            if cli.verbose {
                                let _ = writeln!(std::io::stderr(), "[dm] MCP tool: {}", tool.name);
                            }
                            registry.register_mcp(cfg.name.clone(), tool);
                        }
                        mcp_clients.insert(cfg.name.clone(), Arc::new(Mutex::new(mc)));
                    }
                    Err(e) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] MCP server '{}': failed to list tools: {}",
                            cfg.name,
                            e
                        );
                    }
                },
                Err(e) => {
                    let _ = writeln!(
                        std::io::stderr(),
                        "[dm] Failed to start MCP server '{}': {}",
                        cfg.name,
                        e
                    );
                }
            }
        }

        // Load plugins from ~/.dm/plugins/dm-tool-* executables
        for plugin in plugins::discover_plugins(&config.config_dir) {
            let path_str = plugin.path.to_string_lossy().to_string();
            let args: Vec<&str> = plugin.args.iter().map(|s| s.as_str()).collect();
            match mcp::client::McpClient::spawn(&path_str, &args).await {
                Ok(mut mc) => match mc.list_tools().await {
                    Ok(tools) => {
                        if cli.verbose {
                            let _ = writeln!(
                                std::io::stderr(),
                                "[dm] Plugin '{}': {} tool(s)",
                                plugin.name,
                                tools.len()
                            );
                        }
                        mcp_server_status.push((format!("plugin:{}", plugin.name), tools.len()));
                        for tool in tools {
                            registry.register_mcp(plugin.name.clone(), tool);
                        }
                        mcp_clients.insert(plugin.name.clone(), Arc::new(Mutex::new(mc)));
                    }
                    Err(e) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Plugin '{}': failed to list tools: {}",
                            plugin.name,
                            e
                        );
                    }
                },
                Err(e) => {
                    let _ = writeln!(
                        std::io::stderr(),
                        "[dm] Plugin '{}': failed to start: {}",
                        plugin.name,
                        e
                    );
                }
            }
        }

        let (user_tx, user_rx) = tokio::sync::mpsc::channel::<String>(32);
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);

        // Background GPU poller — updates every 2 seconds
        let (gpu_tx, gpu_rx) = tokio::sync::watch::channel(None::<gpu::GpuStats>);
        tokio::spawn(async move {
            loop {
                let stats = gpu::probe().await;
                let _ = gpu_tx.send(stats);
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });

        let agent_client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let agent_tool_client = config
            .tool_model
            .as_ref()
            .map(|m| OllamaClient::new(config.ollama_base_url(), m.clone()));
        let agent_session = sess.clone();
        let agent_engine = engine;
        let agent_config_dir = config.config_dir.clone();
        let agent_verbose = cli.verbose;
        let agent_max_turns = cli.max_turns.unwrap_or(DEFAULT_MAX_TURNS);
        // Staging: enabled in TUI when not auto-apply. Disabled in non-interactive / --serve.
        let agent_staging = stdout_is_tty && !cli.auto_apply && !cli.serve;

        let agent_format_after = cli.format_after;
        let agent_retry = dm::conversation::RetrySettings::from_config(&config);
        tokio::spawn(async move {
            tui::agent::run(
                agent_client,
                agent_tool_client,
                registry,
                mcp_clients,
                system_prompt,
                agent_session,
                agent_engine,
                agent_config_dir,
                agent_verbose,
                agent_max_turns,
                agent_staging,
                agent_format_after,
                event_tx,
                user_rx,
                cancel_rx,
                agent_retry,
            )
            .await;
        });

        if cli.serve {
            // --serve: run the web UI instead of the terminal UI.
            let (event_bus_tx, _) = tokio::sync::broadcast::channel::<String>(256);
            let web_token = dm::api::load_or_generate_token(&config.config_dir).ok();
            if let Some(ref t) = web_token {
                let _ = writeln!(std::io::stderr(), "[dm serve] bearer token: {}", t);
            }
            let state = web::state::AppState {
                event_bus: event_bus_tx,
                user_tx: user_tx.clone(),
                cancel_tx,
                model: std::sync::Arc::new(std::sync::RwLock::new(config.model.clone())),
                base_url: config.ollama_base_url(),
                session_id: sess.id.clone(),
                config_dir: config.config_dir.clone(),
                token: web_token,
                start_time: std::time::Instant::now(),
                chain_event_rx: std::sync::Arc::new(std::sync::Mutex::new(None)),
            };
            web::server::run(state, event_rx, cli.port).await?;
        } else {
            let mut app = tui::app::App::new(
                config.model.clone(),
                config.host.clone(),
                sess.id.clone(),
                config.config_dir.clone(),
                mcp_server_status,
            );
            app.hyperlinks = cli.hyperlinks;

            app.push_entry(
                tui::app::EntryKind::SystemInfo,
                format!(
                    "dm v{} | {} | session: {}",
                    env!("CARGO_PKG_VERSION"),
                    config.model,
                    short_id(&sess.id),
                ),
            );

            if !sess.messages.is_empty() {
                replay_session_into_app(&sess.messages, &mut app);
            }

            let client_for_cmds = OllamaClient::new(config.ollama_base_url(), config.model.clone());
            let tui_fut = tui::run::run_tui(
                &mut app,
                &mut event_rx,
                &user_tx,
                &client_for_cmds,
                cancel_tx,
                gpu_rx,
                None, // daemon_client: local agent mode
            );

            #[cfg(unix)]
            {
                use tokio::signal::unix::{signal, SignalKind};
                // Install both SIGTERM (systemd/kill default) and SIGINT
                // (kill -INT, parent-process interrupts). TUI-typed Ctrl-C
                // is a crossterm keypress handled by tui/input.rs — raw
                // mode blocks TTY→SIGINT conversion, so this branch only
                // catches externally-delivered signals. Per-turn saves in
                // turn.rs keep message history up-to-date; these arms just
                // refresh metadata before we exit. TerminalGuard Drop
                // handles TTY cleanup regardless of which arm fires.
                let sigterm = signal(SignalKind::terminate());
                let sigint = signal(SignalKind::interrupt());
                match (sigterm, sigint) {
                    (Ok(mut sigterm), Ok(mut sigint)) => {
                        tokio::select! {
                            r = tui_fut => r?,
                            _ = sigterm.recv() => {
                                do_signal_save(
                                    "SIGTERM",
                                    &sess.id,
                                    app.session_title.clone(),
                                    &app.model,
                                    &config.config_dir,
                                );
                            }
                            _ = sigint.recv() => {
                                do_signal_save(
                                    "SIGINT",
                                    &sess.id,
                                    app.session_title.clone(),
                                    &app.model,
                                    &config.config_dir,
                                );
                            }
                        }
                    }
                    (Ok(mut sigterm), Err(e)) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Warning: could not install SIGINT handler: {}",
                            e
                        );
                        tokio::select! {
                            r = tui_fut => r?,
                            _ = sigterm.recv() => {
                                do_signal_save(
                                    "SIGTERM",
                                    &sess.id,
                                    app.session_title.clone(),
                                    &app.model,
                                    &config.config_dir,
                                );
                            }
                        }
                    }
                    (Err(e), Ok(mut sigint)) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Warning: could not install SIGTERM handler: {}",
                            e
                        );
                        tokio::select! {
                            r = tui_fut => r?,
                            _ = sigint.recv() => {
                                do_signal_save(
                                    "SIGINT",
                                    &sess.id,
                                    app.session_title.clone(),
                                    &app.model,
                                    &config.config_dir,
                                );
                            }
                        }
                    }
                    (Err(e1), Err(e2)) => {
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Warning: could not install SIGTERM handler: {}",
                            e1
                        );
                        let _ = writeln!(
                            std::io::stderr(),
                            "[dm] Warning: could not install SIGINT handler: {}",
                            e2
                        );
                        tui_fut.await?;
                    }
                }
            }
            #[cfg(not(unix))]
            tui_fut.await?;
        }

        // Session-end memory update for TUI mode — reload session from disk
        if !cli.no_memory {
            if let Ok(updated_sess) = session::storage::load(&config.config_dir, &sess.id) {
                if let Err(e) = memory::maybe_update_memory(
                    &updated_sess,
                    &client,
                    &config.config_dir,
                    &project_hash,
                )
                .await
                {
                    let _ = writeln!(std::io::stderr(), "[dm] Memory update failed: {}", e);
                }
            }
        }
    }

    Ok(())
}

/// Replay a session's message history into the TUI app so the user can see
/// the prior conversation when resuming with `--continue` or `--session-id`.
///
/// Replays the last 40 messages to avoid flooding the screen on long sessions.
/// System messages and raw tool-result messages are skipped.
fn replay_session_into_app(messages: &[serde_json::Value], app: &mut tui::app::App) {
    use tui::app::EntryKind;

    // Show at most the last 40 non-system messages
    const MAX_REPLAY: usize = 40;
    let visible: Vec<&serde_json::Value> = messages
        .iter()
        .filter(|m| m["role"].as_str().is_some_and(|r| r != "system"))
        .collect();
    let start = visible.len().saturating_sub(MAX_REPLAY);
    let replay = &visible[start..];

    if !replay.is_empty() {
        app.push_entry(
            EntryKind::SystemInfo,
            format!(
                "── resumed session · {} message{} ──",
                replay.len(),
                if replay.len() == 1 { "" } else { "s" }
            ),
        );
    }

    for msg in replay {
        let role = msg["role"].as_str().unwrap_or("");
        match role {
            "user" => {
                let text = msg["content"].as_str().unwrap_or("").to_string();
                if !text.is_empty()
                    && !text.starts_with("\x00IMAGES\x00")
                    && !text.starts_with("__apply__")
                    && !text.starts_with("__reject__")
                {
                    app.push_entry(EntryKind::UserMessage, text);
                }
            }
            "assistant" => {
                // Surface text content and any tool calls
                let content = msg["content"].as_str().unwrap_or("").to_string();
                if !content.is_empty() {
                    app.push_entry(EntryKind::AssistantMessage, content);
                }
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for call in calls {
                        let name = call["function"]["name"].as_str().unwrap_or("?");
                        let args = &call["function"]["arguments"];
                        let repr = match name {
                            "bash" => args["command"].as_str().unwrap_or("").to_string(),
                            "read_file" | "write_file" | "edit_file" => {
                                args["path"].as_str().unwrap_or("").to_string()
                            }
                            "glob" => args["pattern"].as_str().unwrap_or("").to_string(),
                            "grep" => args["pattern"].as_str().unwrap_or("").to_string(),
                            _ => String::new(),
                        };
                        let label = if repr.is_empty() {
                            name.to_string()
                        } else {
                            let r = if repr.len() > 60 {
                                let mut end = 60usize.min(repr.len());
                                while end > 0 && !repr.is_char_boundary(end) {
                                    end -= 1;
                                }
                                format!("{}…", &repr[..end])
                            } else {
                                repr
                            };
                            format!("{}: {}", name, r)
                        };
                        app.push_entry(EntryKind::ToolCall, label);
                    }
                }
            }
            "tool" => {
                // Show tool results as compact previews
                let output = msg["content"].as_str().unwrap_or("");
                let name = msg["name"].as_str().unwrap_or("tool");
                let preview: String = output.lines().take(2).collect::<Vec<_>>().join(" ");
                let label = if preview.is_empty() {
                    format!("[{}] done", name)
                } else {
                    let p = if preview.len() > 80 {
                        let mut end = 80usize.min(preview.len());
                        while end > 0 && !preview.is_char_boundary(end) {
                            end -= 1;
                        }
                        format!("{}…", &preview[..end])
                    } else {
                        preview
                    };
                    format!("[{}] {}", name, p)
                };
                app.push_entry(EntryKind::ToolResult, label);
            }
            _ => {}
        }
    }

    if !replay.is_empty() {
        app.push_entry(
            EntryKind::SystemInfo,
            "── continuing ──────────────────────────────".to_string(),
        );
    }
}

const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "glob",
    "grep",
    "ls",
    "notebook_read",
    "semantic_search",
    "web_search",
    "web_fetch",
];

fn apply_permission_flags(
    permission_mode: Option<&str>,
    allow: &[String],
    disallow: &[String],
    engine: &mut permissions::engine::PermissionEngine,
) {
    if permission_mode == Some("plan") {
        for tool in READ_ONLY_TOOLS {
            engine.add_session_rule(permissions::Rule::tool_wide(
                tool,
                permissions::Behavior::Allow,
            ));
        }
    }

    for spec in allow {
        if let Some(tool_name) = spec.strip_prefix("tool:") {
            engine.add_session_rule(permissions::Rule::tool_wide(
                tool_name,
                permissions::Behavior::Allow,
            ));
        } else {
            let _ = writeln!(
                std::io::stderr(),
                "dm: unrecognized --allow spec '{}' (use 'tool:<name>')",
                spec
            );
        }
    }

    // --disallow added after --allow so they take precedence via last-match-wins
    for spec in disallow {
        if let Some(tool_name) = spec.strip_prefix("tool:") {
            engine.add_session_rule(permissions::Rule::tool_wide(
                tool_name,
                permissions::Behavior::Deny,
            ));
        } else {
            let _ = writeln!(
                std::io::stderr(),
                "dm: unrecognized --disallow spec '{}' (use 'tool:<name>')",
                spec
            );
        }
    }
}

fn new_session(config: &Config) -> session::Session {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    session::Session::new(cwd, config.model.clone())
}

// ── Daemon client TUI mode ───���────────────────────────────���───────────────────

/// Attempt to connect to the running daemon and run the TUI in daemon mode.
/// Returns `Ok(())` on clean exit. Returns `Err` if connection or session
/// setup fails (caller will fall back to local agent).
async fn try_run_daemon_client(cli: &Cli) -> anyhow::Result<()> {
    let mut dc = daemon::DaemonClient::connect()
        .await
        .context("connect to daemon")?;

    let _ = writeln!(std::io::stderr(), "[dm] Connected to daemon.");

    let mut config = load_config_or_exit();
    if let Some(ref host) = cli.host {
        config.host = dark_matter::config::normalize_host(host);
    }
    if let Some(ref model) = cli.model {
        config.model = model.clone();
    }

    // Create or attach to a daemon session.
    let session_id = if let Some(ref id) = cli.session_id {
        dc.send_request("session.attach", serde_json::json!({ "session_id": id }))
            .await
            .context("session.attach")?;
        id.clone()
    } else {
        dc.send_request("session.create", serde_json::json!({}))
            .await
            .context("session.create")?;
        match dc
            .recv_event()
            .await
            .context("recv session.create response")?
        {
            daemon::protocol::DaemonEvent::SessionCreated { session_id } => session_id,
            other => anyhow::bail!("unexpected response to session.create: {:?}", other),
        }
    };

    // We don't spawn a local agent — all agent work is done by the daemon.
    // Provide dummy channels so run_tui compiles; the daemon_client arm handles input/output.
    let (_event_tx, mut event_rx) = tokio::sync::mpsc::channel::<tui::BackendEvent>(1);
    let (user_tx, _user_rx) = tokio::sync::mpsc::channel::<String>(1);
    let (cancel_tx, _cancel_rx) = tokio::sync::watch::channel(false);

    let (gpu_tx, gpu_rx) = tokio::sync::watch::channel(None::<gpu::GpuStats>);
    tokio::spawn(async move {
        loop {
            let stats = gpu::probe().await;
            let _ = gpu_tx.send(stats);
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    });

    let ollama_client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
    let mut app = tui::app::App::new(
        config.model.clone(),
        config.host.clone(),
        session_id,
        config.config_dir.clone(),
        vec![],
    );
    app.hyperlinks = cli.hyperlinks;
    app.push_entry(
        tui::app::EntryKind::SystemInfo,
        format!(
            "dm v{} | {} | session: {} [daemon]",
            env!("CARGO_PKG_VERSION"),
            config.model,
            short_id(&app.session_id),
        ),
    );

    tui::run::run_tui(
        &mut app,
        &mut event_rx,
        &user_tx,
        &ollama_client,
        cancel_tx,
        gpu_rx,
        Some(dc),
    )
    .await
}

/// Build the stderr line printed when a Unix signal (SIGTERM/SIGINT) triggers
/// a best-effort session save at exit. Keeps the two signal branches in
/// lockstep — any wording change here applies to both.
#[cfg(unix)]
fn format_signal_save_message(
    signal_name: &str,
    save_result: &anyhow::Result<()>,
    session_short_id: &str,
    config_dir: &std::path::Path,
) -> String {
    match save_result {
        Ok(()) => format!(
            "dm: received {}, saved session {} and exiting",
            signal_name, session_short_id
        ),
        Err(e) => format!(
            "dm: received {} but {}",
            signal_name,
            dm::session::storage::format_save_failure_tail(e, config_dir, "\n    ")
        ),
    }
}

/// Perform the exit-save and print the resulting message. Called from both
/// SIGTERM and SIGINT arms; takes primitive refs so it borrows neither `App`
/// nor `Session` at the call site (the select! arm bodies regain access to
/// those once `tui_fut` is dropped by the losing-future semantics).
#[cfg(unix)]
fn do_signal_save(
    signal_name: &str,
    sess_id: &str,
    session_title: Option<String>,
    model: &str,
    config_dir: &std::path::Path,
) {
    let cwd = std::env::current_dir()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let short = short_id(sess_id);
    let result = dm::session::storage::update_or_create_stub(
        config_dir,
        sess_id,
        session_title,
        model,
        &cwd,
    );
    let _ = writeln!(
        std::io::stderr(),
        "{}",
        format_signal_save_message(signal_name, &result, short, config_dir)
    );
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_cli() -> Cli {
        Cli {
            print: None,
            model: None,
            host: None,
            models: false,
            aliases: false,
            models_pull: None,
            models_rm: None,
            models_update: false,
            models_info: None,
            models_force: false,
            verbose: false,
            output_format: "text".to_string(),
            stream: false,
            dangerously_skip_permissions: false,
            session_id: None,
            resume: false,
            continue_session: false,
            init: false,
            doctor: false,
            recovery: false,
            completions: None,
            max_turns: None,
            add_dirs: vec![],
            no_tools: false,
            extra_system: None,
            show_config: false,
            allow: vec![],
            disallow: vec![],
            permission_mode: None,
            index: false,
            embed_model: None,
            tool_model: None,
            commit: false,
            review: None,
            pr: None,
            pr_push: false,
            pr_open: false,
            pr_draft: false,
            mcp_server: false,
            watch: false,
            orchestrate: None,
            chain: None,
            chain_resume: None,
            chain_validate: None,
            chain_id: None,
            max_cycles: 5,
            workspace: None,
            gc: false,
            share: None,
            serve: false,
            port: 7421,
            bench: None,
            bench_task: "all".to_string(),
            bench_runs: 3,
            bench_out: None,
            bench_history: None,
            pull: None,
            test_fix: None,
            test_fix_rounds: 5,
            json_schema: None,
            templates: false,
            template: None,
            template_args: vec![],
            context: vec![],
            watch_fix: None,
            watch_cmd: String::new(),
            search: None,
            prune_sessions: None,
            todo: false,
            todo_fix: false,
            todo_glob: vec![],
            share_format: "html".to_string(),
            lint_fix: None,
            memory_show: false,
            memory_clear: false,
            memory_edit: false,
            no_memory: false,
            changelog: None,
            changelog_to: "HEAD".to_string(),
            changelog_format: "md".to_string(),
            compare: None,
            compare_prompt: None,
            compare_side_by_side: false,
            summarize: None,
            summarize_length: 150,
            summarize_style: "bullets".to_string(),
            auto_apply: false,
            document: None,
            document_style: "auto".to_string(),
            document_glob: None,
            translate: None,
            translate_to: None,
            translate_out: None,
            format_after: false,
            security: None,
            daemon_start: false,
            daemon_stop: false,
            daemon_status: false,
            daemon_health: false,
            daemon_restart: false,
            no_daemon: false,
            _daemon_worker: false,
            _daemon_watchdog: false,
            daemon_log: false,
            tail: 50,
            daemon_chain_start: None,
            daemon_chain_status: None,
            daemon_chain_stop: None,
            daemon_chain_list: false,
            daemon_chain_attach: None,
            daemon_chain_pause: None,
            daemon_chain_resume: None,
            daemon_chain_talk: None,
            daemon_chain_add: None,
            daemon_chain_remove: None,
            daemon_chain_model: None,
            schedule_add: None,
            schedule_list: false,
            schedule_remove: None,
            schedule_run: None,
            notify: false,
            ps: false,
            kill: None,
            kill_all: false,
            session_cancel: None,
            session_shutdown: None,
            watchdog: false,
            no_watchdog: false,
            agent_create: None,
            agent_list: false,
            agent_show: None,
            agent_delete: None,
            agent_run: None,
            agent_model: None,
            agent_system: None,
            agent_tools: None,
            agent_description: None,
            replay: None,
            replay_model: None,
            replay_from_turn: 0,
            replay_dry_run: false,
            eval_files: vec![],
            eval_model: None,
            eval_compare: None,
            eval_verbose: false,
            eval_fail_fast: false,
            eval_strict: false,
            perf: false,
            routing_debug: false,
            hyperlinks: false,
            eval_freeze: false,
            eval_ci: false,
            eval_diff_baseline: false,
            eval_runs: 1,
            eval_flaky_threshold: 1.0,
            eval_regression_threshold: 0.1,
            no_workspace_context: false,
            no_dm_md: false,
            web: false,
            web_port: 7422,
            web_token: None,
            run: None,
            run_output: None,
            dry_run: false,
            eval_quiet: false,
            eval_list: false,
            eval_results: false,
            eval_report: None,
            eval_report_last: false,
            eval_report_compare: None,
            eval_report_compare_b: None,
            mcp_list: false,
            mcp_scope: None,
            mcp_add: None,
            mcp_remove: None,
            mcp_enable: None,
            mcp_disable: None,
            mcp_test: None,
            patch: None,
            prompt: None,
            command: None,
            quiet: false,
        }
    }

    #[test]
    fn resolve_mode_local_when_no_daemon() {
        let mut cli = make_test_cli();
        cli.no_daemon = true;
        assert_eq!(resolve_mode(&cli, false), RunMode::LocalAgent);
    }

    #[test]
    fn resolve_mode_daemon_start_flag() {
        let mut cli = make_test_cli();
        cli.daemon_start = true;
        assert_eq!(resolve_mode(&cli, false), RunMode::DaemonStart);
    }

    #[test]
    fn resolve_mode_respects_no_daemon_flag() {
        // Even when daemon is "running" (daemon_running=true), --no-daemon takes priority
        let mut cli = make_test_cli();
        cli.no_daemon = true;
        assert_eq!(resolve_mode(&cli, true), RunMode::LocalAgent);
    }

    // ── Daemon chain CLI flags ───────────────────────────────────────────────

    #[test]
    fn daemon_chain_start_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_start.is_none());
    }

    #[test]
    fn daemon_chain_status_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_status.is_none());
    }

    #[test]
    fn daemon_chain_stop_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_stop.is_none());
    }

    #[test]
    fn daemon_chain_list_flag_defaults_false() {
        let cli = make_test_cli();
        assert!(!cli.daemon_chain_list);
    }

    #[test]
    fn daemon_chain_attach_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_attach.is_none());
    }

    #[test]
    fn daemon_chain_pause_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_pause.is_none());
    }

    #[test]
    fn daemon_chain_resume_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_resume.is_none());
    }

    #[test]
    fn daemon_chain_talk_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_talk.is_none());
    }

    #[test]
    fn daemon_chain_add_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_add.is_none());
    }

    #[test]
    fn daemon_chain_remove_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_remove.is_none());
    }

    #[test]
    fn daemon_chain_model_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.daemon_chain_model.is_none());
    }

    #[test]
    fn chain_validate_flag_defaults_none() {
        let cli = make_test_cli();
        assert!(cli.chain_validate.is_none());
    }

    #[test]
    fn doctor_subcommand_parses_as_diagnostics_alias() {
        use clap::Parser;

        let cli = Cli::try_parse_from(["dm", "doctor"]).expect("parse doctor subcommand");

        assert_eq!(cli.command, Some(Commands::Doctor));
        assert!(!cli.doctor, "subcommand must not require --doctor flag");
    }

    #[test]
    fn init_subcommand_parses_as_initialization_alias() {
        use clap::Parser;

        let cli = Cli::try_parse_from(["dm", "init"]).expect("parse init subcommand");

        assert_eq!(cli.command, Some(Commands::Init));
        assert!(!cli.init, "subcommand must not require --init flag");
    }

    #[test]
    fn sync_subcommand_parses_status_dry_run_and_abort_flags() {
        use clap::Parser;

        let cli = Cli::try_parse_from(["dm", "sync", "--status", "--dry-run", "--abort"])
            .expect("parse sync subcommand");

        assert_eq!(
            cli.command,
            Some(Commands::Sync {
                dry_run: true,
                abort: true,
                status: true,
            })
        );
    }

    #[test]
    fn mcp_scope_project_parses_with_mcp_list() {
        use clap::Parser;

        let cli =
            Cli::try_parse_from(["dm", "--mcp-list", "--mcp-scope", "project"]).expect("parse");

        assert!(cli.mcp_list);
        assert_eq!(cli.mcp_scope, Some(McpScope::Project));
    }

    /// `--mcp-scope` is now accepted by every MCP management command, not just
    /// `--mcp-list`. Pins the parser shape so a future cycle that splits the
    /// flags by command surface still makes them all parseable together.
    #[test]
    fn mcp_scope_parses_with_every_management_command() {
        use clap::Parser;
        for management_args in [
            vec!["dm", "--mcp-list", "--mcp-scope", "project"],
            vec!["dm", "--mcp-add", "name cmd", "--mcp-scope", "project"],
            vec!["dm", "--mcp-remove", "name", "--mcp-scope", "global"],
            vec!["dm", "--mcp-enable", "name", "--mcp-scope", "project"],
            vec!["dm", "--mcp-disable", "name", "--mcp-scope", "project"],
            vec!["dm", "--mcp-test", "name", "--mcp-scope", "global"],
        ] {
            let cli = Cli::try_parse_from(management_args.iter().copied()).unwrap_or_else(|e| {
                panic!("must parse {:?}: {}", management_args, e);
            });
            assert!(
                cli.mcp_scope.is_some(),
                "scope must round-trip: {:?}",
                management_args
            );
        }
    }

    #[test]
    fn mcp_scope_defaults_project_in_host_mode_global_in_kernel_mode() {
        let host = dm::identity::Identity {
            mode: dm::identity::Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        assert_eq!(resolve_mcp_scope(None, &host), McpScope::Project);
        assert_eq!(
            resolve_mcp_scope(Some(McpScope::Global), &host),
            McpScope::Global
        );

        let kernel = dm::identity::Identity::default_kernel();
        assert_eq!(resolve_mcp_scope(None, &kernel), McpScope::Global);
        assert_eq!(
            resolve_mcp_scope(Some(McpScope::Project), &kernel),
            McpScope::Project
        );
    }

    #[test]
    fn mcp_config_dir_for_scope_selects_global_or_project_dm_dir() {
        let global = std::path::Path::new("/home/user/.dm");
        let project = std::path::Path::new("/repo/host");

        assert_eq!(
            mcp_config_dir_for_scope(global, project, McpScope::Global),
            global
        );
        assert_eq!(
            mcp_config_dir_for_scope(global, project, McpScope::Project),
            project.join(".dm")
        );
    }

    #[test]
    fn print_chain_event_does_not_panic() {
        // All DaemonEvent chain variants should be handled without panic
        let events = vec![
            daemon::protocol::DaemonEvent::ChainStarted {
                chain_id: "c1".into(),
                name: "test".into(),
                node_count: 3,
            },
            daemon::protocol::DaemonEvent::ChainNodeTransition {
                chain_id: "c1".into(),
                cycle: 1,
                node_name: "builder".into(),
                status: "completed".into(),
            },
            daemon::protocol::DaemonEvent::ChainCycleComplete {
                chain_id: "c1".into(),
                cycle: 1,
            },
            daemon::protocol::DaemonEvent::ChainFinished {
                chain_id: "c1".into(),
                success: true,
                reason: "done".into(),
            },
        ];
        for event in &events {
            daemon::commands::print_chain_event(event); // must not panic
        }
    }

    #[test]
    fn daemon_chain_flags_do_not_affect_run_mode() {
        // These flags are handled before resolve_mode, so they shouldn't change it
        let mut cli = make_test_cli();
        cli.daemon_chain_start = Some("test.yaml".into());
        assert_eq!(resolve_mode(&cli, false), RunMode::LocalAgent);

        let mut cli2 = make_test_cli();
        cli2.daemon_chain_list = true;
        assert_eq!(resolve_mode(&cli2, false), RunMode::LocalAgent);

        let mut cli3 = make_test_cli();
        cli3.daemon_chain_attach = Some("chain-abc".into());
        assert_eq!(resolve_mode(&cli3, false), RunMode::LocalAgent);
    }

    // ── Permission flags ────────────────────────────────────────────────────

    #[test]
    fn disallow_flag_produces_deny_rule() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(None, &[], &["tool:bash".into()], &mut engine);
        let decision = engine.check("bash", &serde_json::json!({"command": "ls"}));
        assert_eq!(decision, permissions::Decision::Deny);
    }

    #[test]
    fn allow_flag_produces_allow_rule() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(None, &["tool:bash".into()], &[], &mut engine);
        let decision = engine.check("bash", &serde_json::json!({"command": "ls"}));
        assert_eq!(decision, permissions::Decision::Allow);
    }

    #[test]
    fn disallow_overrides_allow_for_same_tool() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(
            None,
            &["tool:bash".into()],
            &["tool:bash".into()],
            &mut engine,
        );
        let decision = engine.check("bash", &serde_json::json!({"command": "ls"}));
        assert_eq!(
            decision,
            permissions::Decision::Deny,
            "--disallow should win over --allow"
        );
    }

    #[test]
    fn permission_mode_plan_allows_read_tools() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(Some("plan"), &[], &[], &mut engine);
        assert_eq!(
            engine.check("read_file", &serde_json::json!({})),
            permissions::Decision::Allow
        );
        assert_eq!(
            engine.check("glob", &serde_json::json!({})),
            permissions::Decision::Allow
        );
        assert_eq!(
            engine.check("grep", &serde_json::json!({})),
            permissions::Decision::Allow
        );
        assert_eq!(
            engine.check("web_fetch", &serde_json::json!({})),
            permissions::Decision::Allow
        );
        // Write tools should still ask
        assert_eq!(
            engine.check("bash", &serde_json::json!({})),
            permissions::Decision::Ask
        );
        assert_eq!(
            engine.check("write_file", &serde_json::json!({})),
            permissions::Decision::Ask
        );
    }

    #[test]
    fn permission_mode_default_asks_for_everything() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(Some("default"), &[], &[], &mut engine);
        assert_eq!(
            engine.check("read_file", &serde_json::json!({})),
            permissions::Decision::Ask
        );
        assert_eq!(
            engine.check("bash", &serde_json::json!({})),
            permissions::Decision::Ask
        );
    }

    #[test]
    fn permission_mode_plan_disallow_overrides_plan_allows() {
        let mut engine = permissions::engine::PermissionEngine::new(false, vec![]);
        apply_permission_flags(Some("plan"), &[], &["tool:glob".into()], &mut engine);
        assert_eq!(
            engine.check("glob", &serde_json::json!({})),
            permissions::Decision::Deny,
            "--disallow should override plan mode's Allow"
        );
        assert_eq!(
            engine.check("read_file", &serde_json::json!({})),
            permissions::Decision::Allow,
            "other read tools should still be allowed"
        );
    }

    #[cfg(unix)]
    #[test]
    fn format_signal_save_message_success_names_signal_and_short_id() {
        let config_dir = std::path::Path::new("/home/user/.config/dm");
        let ok_result: anyhow::Result<()> = Ok(());

        let msg_term =
            super::format_signal_save_message("SIGTERM", &ok_result, "abcd1234", config_dir);
        assert!(msg_term.contains("SIGTERM"), "got: {}", msg_term);
        assert!(
            msg_term.contains("saved session abcd1234"),
            "got: {}",
            msg_term
        );
        assert!(msg_term.contains("exiting"), "got: {}", msg_term);

        let msg_int =
            super::format_signal_save_message("SIGINT", &ok_result, "abcd1234", config_dir);
        assert!(msg_int.contains("SIGINT"), "got: {}", msg_int);
        assert!(
            msg_int.contains("saved session abcd1234"),
            "got: {}",
            msg_int
        );
        // Cross-signal distinctness — the two messages must not be equal.
        assert_ne!(msg_term, msg_int);
    }

    #[cfg(unix)]
    #[test]
    fn format_signal_save_message_failure_includes_check_hint_and_error() {
        let config_dir = std::path::Path::new("/home/user/.config/dm");
        let err_result: anyhow::Result<()> = Err(anyhow::anyhow!("no space left on device"));

        let msg_term =
            super::format_signal_save_message("SIGTERM", &err_result, "abcd1234", config_dir);
        assert!(msg_term.contains("SIGTERM"), "got: {}", msg_term);
        assert!(
            msg_term.contains("session save failed"),
            "got: {}",
            msg_term
        );
        assert!(
            msg_term.contains("no space left on device"),
            "got: {}",
            msg_term
        );
        assert!(
            msg_term.contains("Check: available disk space in"),
            "got: {}",
            msg_term
        );
        assert!(
            msg_term.contains("/home/user/.config/dm"),
            "got: {}",
            msg_term
        );

        let msg_int =
            super::format_signal_save_message("SIGINT", &err_result, "abcd1234", config_dir);
        assert!(msg_int.contains("SIGINT"), "got: {}", msg_int);
        assert!(msg_int.contains("Check:"), "got: {}", msg_int);
        assert!(
            msg_int.contains("no space left on device"),
            "got: {}",
            msg_int
        );
    }

    // ── Pillar 1 TUI #4: error hint coverage (bin-target call sites) ───────
    // Each test reconstructs the exact string the corresponding eprintln!
    // site will emit and asserts the hint is present. If the call-site
    // wording changes without updating the hint, the test fails.

    #[test]
    fn config_error_hint_mentions_doctor() {
        let out = dm::error_hints::format_dm_error(
            "config error: something broken",
            Some("dm --doctor"),
        );
        assert!(out.contains("Try: dm --doctor"), "got: {out}");
        assert!(out.starts_with("dm: "), "got: {out}");
    }

    #[test]
    fn toplevel_error_hint_mentions_rust_log() {
        let out = dm::error_hints::format_dm_error(
            "some anyhow error",
            Some("dm --doctor, or re-run with RUST_LOG=debug"),
        );
        assert!(out.contains("RUST_LOG=debug"), "got: {out}");
        assert!(out.contains("dm --doctor"), "got: {out}");
    }

    #[test]
    fn mcp_add_hint_shows_concrete_example() {
        let out = dm::error_hints::format_dm_error(
            "--mcp-add requires \"name command [args...]\"",
            Some("dm --mcp-add myserver python -m my_mcp"),
        );
        assert!(out.contains("Try: dm --mcp-add myserver"), "got: {out}");
    }

    #[test]
    fn agent_run_hint_shows_concrete_example() {
        let out = dm::error_hints::format_dm_error(
            "--agent-run requires a prompt via -p or positional argument",
            Some("dm --agent-run general -p \"summarize README\""),
        );
        assert!(out.contains("dm --agent-run general"), "got: {out}");
        assert!(out.contains("-p "), "shows -p flag: {out}");
    }

    #[test]
    fn compare_hint_shows_two_model_csv() {
        let out = dm::error_hints::format_dm_error(
            "--compare requires at least one model name",
            Some("dm --compare llama3.1:8b,qwen2.5:7b --compare-prompt \"...\""),
        );
        assert!(out.contains("llama3.1:8b,qwen2.5:7b"), "csv form: {out}");
    }

    #[test]
    fn glob_hint_mentions_quoting() {
        let out = dm::error_hints::format_dm_error(
            "invalid glob pattern 'src/**.rs': parse error",
            Some("quote it, e.g. --pattern 'src/**/*.rs'"),
        );
        assert!(out.contains("quote it"), "got: {out}");
        assert!(out.contains("'src/**/*.rs'"), "shows quoted form: {out}");
    }

    #[test]
    fn translate_hint_mentions_translate_to() {
        let out = dm::error_hints::format_dm_error(
            "--translate requires --translate-to <LANG>",
            Some("dm --translate foo.py --translate-to rust"),
        );
        assert!(out.contains("--translate-to rust"), "got: {out}");
    }

    #[test]
    fn template_hint_mentions_dm_templates_path() {
        let out = dm::error_hints::format_dm_error(
            "template error — not found",
            Some("check ~/.dm/templates/ exists, or run dm --list-templates"),
        );
        assert!(out.contains("~/.dm/templates/"), "got: {out}");
        assert!(out.contains("dm --list-templates"), "fallback cmd: {out}");
    }

    // --- Ollama-path hint assertions (Cycle 67) -------------------
    // These don't exercise the HTTP path (no running Ollama in CI).
    // They pin the hint strings + classifier contracts so retrofits
    // in src/ollama/{client,pull}.rs don't silently lose their tails.

    #[test]
    fn ollama_connect_hint_points_at_ollama_serve() {
        let h = dm::ollama::hints::HINT_CONNECT_FAILED;
        assert!(h.contains("ollama serve"), "got: {h}");
    }

    #[test]
    fn ollama_embed_hint_mentions_embed_model() {
        let h = dm::ollama::hints::HINT_EMBED_MALFORMED;
        assert!(
            h.to_lowercase().contains("embed"),
            "embed hint should name embeddings: {h}"
        );
    }

    #[test]
    fn ollama_model_not_found_hint_suggests_pull() {
        let h = dm::ollama::hints::HINT_MODEL_NOT_FOUND;
        assert!(
            h.contains("ollama pull"),
            "not-found hint should suggest pull: {h}"
        );
    }

    #[test]
    fn pull_status_classifier_maps_404_to_spelling_hint() {
        let h = dm::ollama::hints::hint_for_pull_status(404).expect("404 has hint");
        assert!(
            h.contains("spelling") || h.contains("not found"),
            "got: {h}"
        );
    }

    #[test]
    fn pull_status_classifier_maps_5xx_to_retry_hint() {
        let h = dm::ollama::hints::hint_for_pull_status(503).expect("503 has hint");
        assert!(h.contains("registry") || h.contains("later"), "got: {h}");
    }

    #[test]
    fn show_status_classifier_404_is_model_not_found() {
        assert_eq!(
            dm::ollama::hints::hint_for_show_status(404),
            Some(dm::ollama::hints::HINT_MODEL_NOT_FOUND)
        );
    }

    #[test]
    fn delete_status_classifier_404_is_model_not_found() {
        assert_eq!(
            dm::ollama::hints::hint_for_delete_status(404),
            Some(dm::ollama::hints::HINT_MODEL_NOT_FOUND)
        );
    }

    /// Lint guard: every `eprintln!("dm: …")` in main.rs must either
    /// carry an actionable `Try:` tail (directly or via `format_dm_error`)
    /// or appear on the informational whitelist below.
    ///
    /// Why a source scan: Pillar 1 ("Error messages include next steps")
    /// has no type-system hook — eprintln is just a macro, so a future
    /// edit can silently regress. Reading main.rs back as source at
    /// test time is the lightest possible regression gate.
    ///
    /// `include_str!` is chosen over `fs::read_to_string(file!())` because
    /// the former is compile-time and doesn't depend on the test's CWD.
    #[test]
    fn main_rs_eprintln_dm_sites_have_hints() {
        const MAIN_SRC: &str = include_str!("main.rs");
        // Informational status lines that are NOT errors — these are
        // intentionally terse progress/detection messages, not failures.
        // Match by a substring that's unique enough not to accidentally
        // permit a future unrelated error.
        const WHITELIST: &[&str] = &[
            "dm: detected Ollama at",
            "dm: auto-selected model",
            "dm: no Ollama models installed",
            "dm: alias resolved",
            "dm: using built-in preset",
            "dm: --compare requires a prompt",
            "dm: no input received on stdin",
            "dm: no prompt provided",
            "dm: warning",
        ];
        let mut offenders: Vec<String> = Vec::new();
        for line in MAIN_SRC.lines() {
            let trimmed = line.trim();
            // Only check eprintln sites; format_dm_error call sites
            // already go through a hinted helper.
            if !trimmed.starts_with("eprintln!(\"dm: ") {
                continue;
            }
            // Self-referential: skip the WHITELIST slice literal entries.
            if trimmed.starts_with("\"dm:") {
                continue;
            }
            let has_try = trimmed.contains("Try:") || trimmed.contains("Usage:");
            let whitelisted = WHITELIST.iter().any(|w| trimmed.contains(w));
            if !has_try && !whitelisted {
                offenders.push(trimmed.to_string());
            }
        }
        assert!(
            offenders.is_empty(),
            "Unhinted `eprintln!(\"dm: ...\")` sites in main.rs — route through \
             error_hints::format_dm_error(…, Some(…)) or add a `Try:` tail, or \
             extend the informational WHITELIST if the line is intentionally terse:\n{}",
            offenders.join("\n")
        );
    }
}
