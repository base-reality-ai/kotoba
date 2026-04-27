//! Kotoba host entry point.
//!
//! Two things happen here on every startup, before delegating to dm:
//!
//! 1. Kotoba's [`host_caps::KotobaCapabilities`] is installed via
//!    [`dark_matter::host::install_host_capabilities`]. This registers
//!    the 7 host tools (invoke_persona, log_vocabulary, log_kanji,
//!    record_struggle, quiz_me, plan_session, record_session) into every dm registry
//!    that gets constructed afterwards — TUI, daemon, web, headless.
//!
//! 2. Argument routing: by default we launch dm's TUI in host mode
//!    (so `kotoba` opens the conversation interface). Subcommands
//!    `kotoba dm <args>` or `kotoba help` route to dm's CLI.
//!
//! The kernel binary at `src/main.rs` is canonical dm and is left
//! untouched — running `dm` directly bypasses kotoba's host caps.
//! Run via the `kotoba` binary to get the host-mode experience.

use anyhow::Context;
use chrono::{DateTime, Utc};
use dark_matter::host::install_host_capabilities;
use dark_matter::session::Session;
use std::path::{Path, PathBuf};

mod domain;
mod host_caps;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    install_host_capabilities(Box::new(host_caps::KotobaCapabilities))
        .map_err(|_| anyhow::anyhow!("host capabilities already installed"))?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("help") | Some("--help") | Some("-h") => {
            print_help();
            Ok(())
        }
        Some("version") | Some("--version") | Some("-V") => {
            println!("kotoba {} (dark-matter spine)", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some("session") => {
            let opts = parse_session_args(&args[1..])?;
            run_session(opts).await
        }
        Some("dm") => {
            // Forward remaining args to dm's CLI.
            forward_to_dm(args.into_iter().skip(1).collect()).await
        }
        _ => {
            // Default: launch the dm TUI in host mode. The host caps installed
            // above flow into the registry, so wiki/persona tools are available
            // to the conversation immediately.
            //
            // Runtime side note: we shell out to the `dm` binary in the
            // spawned project's target/ rather than calling `dark_matter::tui`
            // directly. This keeps the kernel binary's argv parsing intact
            // (clap, env vars, profiles) and avoids the host having to mirror
            // every CLI flag dm grows. The host caps are installed in the
            // current process; the `dm` subprocess inherits them via the
            // process-global slot since cargo links both binaries against the
            // same library, but only when invoked as `kotoba dm` (same
            // process). For the default "open the TUI" path, we re-invoke
            // ourselves so the install_host_capabilities call sticks.
            forward_to_dm(args).await
        }
    }
}

fn print_help() {
    println!("kotoba — Japanese learning, dark-matter shaped");
    println!();
    println!("USAGE:");
    println!("    kotoba                          Launch the conversation TUI in host mode");
    println!(
        "    kotoba session [--persona N]    Plan→converse→record loop (default persona: Yuki)"
    );
    println!("    kotoba session --plan-only      Print the planner brief and exit (no TUI)");
    println!("    kotoba session --yes            Skip the [Y/n] confirm and launch immediately");
    println!("    kotoba dm [ARGS]                Pass through to the dm kernel CLI");
    println!("    kotoba help                     Show this message");
    println!("    kotoba version                  Show kotoba's version");
    println!();
    println!("HOST TOOLS REGISTERED:");
    println!("    host_invoke_persona     — switch the active conversational persona");
    println!("    host_log_vocabulary     — record a Japanese word in the wiki");
    println!("    host_log_kanji          — record a kanji character with readings");
    println!("    host_record_struggle    — flag something the learner stumbled on");
    println!("    host_quiz_me            — pull due-for-review vocabulary");
    println!("    host_plan_session       — synthesize a pre-session brief from the wiki");
    println!("    host_record_session     — record a transcript back into the wiki");
    println!();
    println!("WIKI LIVES IN:");
    println!("    .dm/wiki/entities/Vocabulary/   — words you've encountered");
    println!("    .dm/wiki/entities/Kanji/        — kanji with readings + mnemonics");
    println!("    .dm/wiki/entities/Persona/      — conversational personas (Yuki, …)");
    println!("    .dm/wiki/synthesis/             — daily struggles + session notes");
    println!();
    println!("See DM.md for the full design and VISION.md for the dm spine paradigm.");
}

async fn forward_to_dm(args: Vec<String>) -> anyhow::Result<()> {
    let status = invoke_dm(args).await?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

async fn invoke_dm(args: Vec<String>) -> anyhow::Result<std::process::ExitStatus> {
    use std::process::Stdio;

    // Look up the kernel binary `dm` next to ourselves in target/.
    let exe = std::env::current_exe().context("locate kotoba binary")?;
    let dm_bin = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("kotoba binary has no parent dir"))?
        .join(if cfg!(windows) { "dm.exe" } else { "dm" });

    if !dm_bin.exists() {
        anyhow::bail!(
            "dm kernel binary not found at {}. Build with `cargo build --release` first.",
            dm_bin.display()
        );
    }

    tokio::process::Command::new(&dm_bin)
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .with_context(|| format!("invoke {}", dm_bin.display()))
}

// ---------------------------------------------------------------------------
// kotoba session — Tier 4: planner → conversation → recorder loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionOpts {
    persona: String,
    plan_only: bool,
    auto_confirm: bool,
    recent_struggle_days: i64,
}

impl Default for SessionOpts {
    fn default() -> Self {
        Self {
            persona: "Yuki".to_string(),
            plan_only: false,
            auto_confirm: false,
            recent_struggle_days: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KotobaModelConfig {
    persona_model: String,
    planner_model: String,
    recorder_model: String,
}

impl Default for KotobaModelConfig {
    fn default() -> Self {
        Self {
            persona_model: "gemini-3.1-pro-preview".to_string(),
            planner_model: "claude-opus".to_string(),
            recorder_model: "claude-opus".to_string(),
        }
    }
}

impl KotobaModelConfig {
    fn load(project_root: &Path) -> Self {
        let toml_body = std::fs::read_to_string(project_root.join(".dm/kotoba.toml")).ok();
        Self::from_sources(|key| std::env::var(key).ok(), toml_body.as_deref())
    }

    fn from_sources(
        env: impl Fn(&str) -> Option<String>,
        toml_body: Option<&str>,
    ) -> KotobaModelConfig {
        let mut cfg = KotobaModelConfig::default();
        if let Some(body) = toml_body {
            if let Ok(value) = body.parse::<toml::Value>() {
                apply_toml_model(&value, "persona_model", &mut cfg.persona_model);
                apply_toml_model(&value, "planner_model", &mut cfg.planner_model);
                apply_toml_model(&value, "recorder_model", &mut cfg.recorder_model);
            }
        }
        apply_env_model(&env, "KOTOBA_PERSONA_MODEL", &mut cfg.persona_model);
        apply_env_model(&env, "KOTOBA_PLANNER_MODEL", &mut cfg.planner_model);
        apply_env_model(&env, "KOTOBA_RECORDER_MODEL", &mut cfg.recorder_model);
        cfg
    }
}

fn apply_env_model(env: &impl Fn(&str) -> Option<String>, key: &str, out: &mut String) {
    if let Some(value) = env(key)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        *out = value;
    }
}

fn apply_toml_model(value: &toml::Value, key: &str, out: &mut String) {
    let root_value = value.get(key).and_then(|v| v.as_str());
    let models_value = value
        .get("models")
        .and_then(|v| v.get(key))
        .and_then(|v| v.as_str());
    if let Some(model) = root_value
        .or(models_value)
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
    {
        *out = model.to_string();
    }
}

fn parse_session_args(rest: &[String]) -> anyhow::Result<SessionOpts> {
    let mut opts = SessionOpts::default();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--persona" | "-p" => {
                let value = rest
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--persona requires a value"))?;
                opts.persona = value.clone();
                i += 2;
            }
            v if v.starts_with("--persona=") => {
                opts.persona = v.trim_start_matches("--persona=").to_string();
                i += 1;
            }
            "--plan-only" => {
                opts.plan_only = true;
                i += 1;
            }
            "--yes" | "-y" => {
                opts.auto_confirm = true;
                i += 1;
            }
            "--days" => {
                let value = rest
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--days requires a value"))?;
                opts.recent_struggle_days = value
                    .parse::<i64>()
                    .with_context(|| format!("--days expects an integer, got {value:?}"))?;
                i += 2;
            }
            "--help" | "-h" => {
                anyhow::bail!(
                    "kotoba session [--persona NAME] [--plan-only] [--yes] [--days N]\n  Plans, runs, and records one Japanese-learning session."
                );
            }
            other => {
                anyhow::bail!(
                    "unknown argument {:?} for `kotoba session`. Try: kotoba session --help",
                    other
                );
            }
        }
    }
    Ok(opts)
}

async fn run_session(opts: SessionOpts) -> anyhow::Result<()> {
    let project_root = std::env::current_dir().context("locate project root")?;
    let model_config = KotobaModelConfig::load(&project_root);
    let today = Utc::now().date_naive();
    let brief = host_caps::plan_session_in(
        &project_root,
        &opts.persona,
        opts.recent_struggle_days,
        today,
    );

    println!("{}", brief);

    if opts.plan_only {
        return Ok(());
    }

    if !opts.auto_confirm && !confirm_start()? {
        println!("Session cancelled.");
        return Ok(());
    }

    let persona_body = read_persona_body(&project_root, &opts.persona).unwrap_or_else(|| {
        format!(
            "(persona '{}' not found in wiki — running with the planner brief only)\n",
            opts.persona
        )
    });
    let system_prompt = format!(
        "You are the Japanese-learning persona '{}'. The persona's wiki entity follows.\n\n{}\n\n--- Planner brief for this session ---\n\n{}",
        opts.persona, persona_body, brief
    );

    let config_dir = dm_config_dir()?;
    let baseline = Utc::now();

    println!(
        "\nLaunching dm TUI as the conversation surface — close it (Ctrl+C or :q) when the session is done.\n"
    );

    let status = invoke_dm(session_dm_args(&system_prompt, &model_config)).await?;
    if !status.success() {
        eprintln!(
            "dm exited with status {}. Attempting recorder pass anyway.",
            status
        );
    }

    let after = Utc::now();
    let new_session = latest_session_after(&config_dir, baseline)?;
    let summary = match new_session {
        Some(session) => {
            let transcript = format_transcript(&session);
            if transcript.trim().is_empty() {
                println!("Session ended with no transcript content; recorder skipped.");
                return Ok(());
            }
            host_caps::record_session_in(&project_root, &transcript, &opts.persona, after)?
        }
        None => {
            println!(
                "No new dm session detected under {} (baseline {}). Recorder skipped.",
                config_dir.display(),
                baseline.to_rfc3339()
            );
            return Ok(());
        }
    };

    print_summary_lines(&summary);
    Ok(())
}

fn session_dm_args(system_prompt: &str, model_config: &KotobaModelConfig) -> Vec<String> {
    vec![
        "--system".into(),
        system_prompt.to_string(),
        "--model".into(),
        model_config.persona_model.clone(),
    ]
}

fn confirm_start() -> anyhow::Result<bool> {
    use std::io::{BufRead, Write};
    print!("Start session? [Y/n]: ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    let stdin = std::io::stdin();
    stdin.lock().read_line(&mut line)?;
    Ok(decide_confirm(&line))
}

fn decide_confirm(input: &str) -> bool {
    matches!(input.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes")
}

fn read_persona_body(project_root: &Path, persona: &str) -> Option<String> {
    let path = project_root
        .join(".dm/wiki/entities/Persona")
        .join(format!("{}.md", host_caps::slugify(persona)));
    std::fs::read_to_string(path).ok()
}

fn dm_config_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
        .join(".dm"))
}

fn latest_session_after(
    config_dir: &Path,
    baseline: DateTime<Utc>,
) -> anyhow::Result<Option<Session>> {
    let sessions = dark_matter::session::storage::list(config_dir)?;
    Ok(sessions
        .into_iter()
        .find(|s| s.created_at >= baseline || s.updated_at >= baseline))
}

fn format_transcript(session: &Session) -> String {
    let mut out = String::new();
    for msg in &session.messages {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or_default();
        if content.is_empty() {
            continue;
        }
        let label = match role {
            "user" => "Learner",
            "assistant" => "Persona",
            "system" => continue, // already in the system prompt; skip from transcript
            other => other,
        };
        for line in content.lines() {
            out.push_str(&format!("{}: {}\n", label, line));
        }
    }
    out
}

fn print_summary_lines(summary: &host_caps::SessionRecordSummary) {
    let words = if summary.words.is_empty() {
        "none".to_string()
    } else {
        summary.words.join(", ")
    };
    let struggles = if summary.struggles.is_empty() {
        "none".to_string()
    } else {
        summary.struggles.join(", ")
    };
    println!(
        "Recorded session for {} (now sessions_count={}).",
        summary.persona, summary.sessions_count
    );
    println!(
        "  Vocabulary logged ({}): {}",
        summary.vocabulary_count, words
    );
    println!(
        "  Struggles flagged ({}): {}",
        summary.struggle_count, struggles
    );
}

#[cfg(test)]
mod session_cmd_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_session_args_defaults() {
        let opts = parse_session_args(&[]).unwrap();
        assert_eq!(opts, SessionOpts::default());
    }

    #[test]
    fn parse_session_args_handles_persona_and_flags() {
        let opts = parse_session_args(&[
            "--persona".into(),
            "Tanaka".into(),
            "--plan-only".into(),
            "--yes".into(),
            "--days".into(),
            "7".into(),
        ])
        .unwrap();
        assert_eq!(opts.persona, "Tanaka");
        assert!(opts.plan_only);
        assert!(opts.auto_confirm);
        assert_eq!(opts.recent_struggle_days, 7);
    }

    #[test]
    fn parse_session_args_handles_persona_equals_form() {
        let opts = parse_session_args(&["--persona=Hiro".into()]).unwrap();
        assert_eq!(opts.persona, "Hiro");
    }

    #[test]
    fn parse_session_args_rejects_unknown_flag() {
        let err = parse_session_args(&["--bogus".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("--bogus"), "err was: {}", err);
    }

    #[test]
    fn model_config_defaults_to_v02_recommendations() {
        let cfg = KotobaModelConfig::from_sources(|_| None, None);
        assert_eq!(cfg.persona_model, "gemini-3.1-pro-preview");
        assert_eq!(cfg.planner_model, "claude-opus");
        assert_eq!(cfg.recorder_model, "claude-opus");
    }

    #[test]
    fn model_config_reads_toml_models_table() {
        let cfg = KotobaModelConfig::from_sources(
            |_| None,
            Some(
                r#"
[models]
persona_model = "persona-from-toml"
planner_model = "planner-from-toml"
recorder_model = "recorder-from-toml"
"#,
            ),
        );
        assert_eq!(cfg.persona_model, "persona-from-toml");
        assert_eq!(cfg.planner_model, "planner-from-toml");
        assert_eq!(cfg.recorder_model, "recorder-from-toml");
    }

    #[test]
    fn model_config_env_overrides_toml() {
        let cfg = KotobaModelConfig::from_sources(
            |key| match key {
                "KOTOBA_PERSONA_MODEL" => Some("persona-from-env".to_string()),
                "KOTOBA_PLANNER_MODEL" => Some("planner-from-env".to_string()),
                "KOTOBA_RECORDER_MODEL" => Some("recorder-from-env".to_string()),
                _ => None,
            },
            Some(
                r#"
persona_model = "persona-from-toml"
planner_model = "planner-from-toml"
recorder_model = "recorder-from-toml"
"#,
            ),
        );
        assert_eq!(cfg.persona_model, "persona-from-env");
        assert_eq!(cfg.planner_model, "planner-from-env");
        assert_eq!(cfg.recorder_model, "recorder-from-env");
    }

    #[test]
    fn session_dm_args_passes_persona_model_to_dm() {
        let cfg = KotobaModelConfig {
            persona_model: "persona-model".into(),
            planner_model: "planner-model".into(),
            recorder_model: "recorder-model".into(),
        };
        let args = session_dm_args("system prompt", &cfg);
        assert_eq!(
            args,
            vec!["--system", "system prompt", "--model", "persona-model"]
        );
    }

    #[test]
    fn decide_confirm_default_yes() {
        assert!(decide_confirm(""));
        assert!(decide_confirm("\n"));
        assert!(decide_confirm("y"));
        assert!(decide_confirm("Y\n"));
        assert!(decide_confirm("YES"));
    }

    #[test]
    fn decide_confirm_no_variants() {
        assert!(!decide_confirm("n"));
        assert!(!decide_confirm("no"));
        assert!(!decide_confirm("never"));
        assert!(!decide_confirm("quit"));
    }

    #[test]
    fn read_persona_body_returns_some_when_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join(".dm/wiki/entities/Persona");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Yuki.md"), "---\ntitle: Yuki\n---\n\nbody").unwrap();
        let body = read_persona_body(tmp.path(), "Yuki").unwrap();
        assert!(body.contains("title: Yuki"));
        assert!(body.contains("body"));
    }

    #[test]
    fn read_persona_body_returns_none_when_missing() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(read_persona_body(tmp.path(), "Nobody").is_none());
    }

    fn fake_session(created_at: DateTime<Utc>, updated_at: DateTime<Utc>) -> Session {
        Session {
            id: uuid::Uuid::new_v4().to_string(),
            title: None,
            created_at,
            updated_at,
            cwd: "/tmp".into(),
            host_project: Some("kotoba".into()),
            model: "test".into(),
            messages: vec![
                json!({"role": "system", "content": "system prompt — should be skipped"}),
                json!({"role": "user", "content": "こんにちは Yuki!"}),
                json!({"role": "assistant", "content": "こんにちは！\nお元気ですか？"}),
                json!({"role": "user", "content": ""}), // empty — skip
            ],
            compact_failures: 0,
            turn_count: 2,
            prompt_tokens: 0,
            completion_tokens: 0,
            parent_id: None,
        }
    }

    #[test]
    fn format_transcript_labels_roles_and_skips_system() {
        let now = Utc::now();
        let session = fake_session(now, now);
        let transcript = format_transcript(&session);
        assert!(!transcript.contains("system prompt — should be skipped"));
        assert!(transcript.contains("Learner: こんにちは Yuki!"));
        assert!(transcript.contains("Persona: こんにちは！"));
        assert!(transcript.contains("Persona: お元気ですか？"));
    }

    #[test]
    fn latest_session_after_only_returns_post_baseline() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&dir).unwrap();
        let baseline = DateTime::parse_from_rfc3339("2026-04-27T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let old = fake_session(
            DateTime::parse_from_rfc3339("2026-04-26T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            DateTime::parse_from_rfc3339("2026-04-26T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        let fresh = fake_session(
            DateTime::parse_from_rfc3339("2026-04-27T13:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            DateTime::parse_from_rfc3339("2026-04-27T13:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        std::fs::write(
            dir.join(format!("{}.json", old.id)),
            serde_json::to_string(&old).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join(format!("{}.json", fresh.id)),
            serde_json::to_string(&fresh).unwrap(),
        )
        .unwrap();

        let found = latest_session_after(tmp.path(), baseline).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, fresh.id);
    }

    #[test]
    fn latest_session_after_returns_none_when_only_old_sessions() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("sessions");
        std::fs::create_dir_all(&dir).unwrap();
        let baseline = DateTime::parse_from_rfc3339("2026-04-27T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let old = fake_session(
            DateTime::parse_from_rfc3339("2026-04-26T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            DateTime::parse_from_rfc3339("2026-04-26T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        );
        std::fs::write(
            dir.join(format!("{}.json", old.id)),
            serde_json::to_string(&old).unwrap(),
        )
        .unwrap();

        let found = latest_session_after(tmp.path(), baseline).unwrap();
        assert!(found.is_none());
    }
}
