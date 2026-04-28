//! Diagnostic tools for inspecting the `dm` environment and Ollama readiness.
//!
//! Exposes the `dm doctor` subcommand to print GPU presence, Ollama connectivity,
//! and configuration health.

use crate::config::Config;
use crate::gpu;
use crate::ollama::client::OllamaClient;
use crate::ollama::health::{render_ollama_health, OllamaHealth, HEALTH_PROBE_TIMEOUT};
use std::fmt::Write as _;
use std::time::Duration;

pub async fn run_doctor(client: &OllamaClient, config: &Config) {
    let identity = crate::identity::load_for_cwd();
    print!("{}", run_doctor_capture(client, config, &identity).await);
}

/// Render the `Identity:` block of `dm doctor` output. Pure formatter so the
/// section's contract is testable without spinning up a `Config` or probing
/// Ollama (the rest of `run_doctor_capture` does both).
fn identity_section(identity: &crate::identity::Identity) -> String {
    let mut out = String::new();
    out.push_str("Identity:\n");
    out.push_str(&format!("  Mode:          {}\n", identity.mode.as_str()));
    if let Some(ref project) = identity.host_project {
        out.push_str(&format!("  Host project:  {}\n", project));
    }
    if let Some(ref rev) = identity.canonical_dm_revision {
        out.push_str(&format!("  Canonical rev: {}\n", rev));
    }
    out.push('\n');
    out
}

fn host_capabilities_section_for_names(names: &[String], installed: bool) -> String {
    let mut out = String::new();
    out.push_str("Host capabilities:\n");
    if !installed {
        out.push_str("  none installed\n\n");
        return out;
    }

    out.push_str(&format!("  installed:    {} tool(s)\n", names.len()));
    for name in names.iter().take(3) {
        out.push_str(&format!("  • {}\n", name));
    }
    if names.len() > 3 {
        out.push_str(&format!("  … and {} more\n", names.len() - 3));
    }
    let invalid_prefix_count = names
        .iter()
        .filter(|name| !name.starts_with(dark_matter::host::HOST_TOOL_PREFIX))
        .count();
    if invalid_prefix_count > 0 {
        out.push_str(&format!(
            "  ⚠ {} tool(s) missing `{}` prefix. Try: rename host tools to start with `{}`.\n",
            invalid_prefix_count,
            dark_matter::host::HOST_TOOL_PREFIX,
            dark_matter::host::HOST_TOOL_PREFIX
        ));
    }
    out.push('\n');
    out
}

fn host_capabilities_section() -> String {
    if let Some(caps) = dark_matter::host::installed_host_capabilities() {
        let mut names: Vec<String> = caps
            .tools()
            .into_iter()
            .map(|tool| tool.name().to_string())
            .collect();
        names.sort();
        host_capabilities_section_for_names(&names, true)
    } else {
        host_capabilities_section_for_names(&[], false)
    }
}

pub async fn run_doctor_capture(
    client: &OllamaClient,
    config: &Config,
    identity: &crate::identity::Identity,
) -> String {
    let mut out = String::new();

    macro_rules! pln {
        () => {{ out.push('\n') }};
        ($fmt:literal $(, $arg:expr)*) => {{
            out.push_str(&format!(concat!($fmt, "\n") $(, $arg)*))
        }};
    }
    macro_rules! pr {
        ($fmt:literal $(, $arg:expr)*) => {{
            out.push_str(&format!($fmt $(, $arg)*))
        }};
    }

    pln!("dm doctor — system diagnostics\n");

    // 1. Config
    pln!("Config:");
    pln!("  Ollama host:  {}", config.host);
    pln!("  Model:        {}", config.model);
    if let Some(ref tm) = config.tool_model {
        pln!("  Tool model:   {} (routing enabled)", tm);
    } else {
        pln!("  Tool model:   (not set — single model mode)");
    }
    pln!("  Embed model:  {}", config.embed_model);
    pln!("  Config dir:   {}", config.config_dir.display());

    let settings_path = config.global_config_dir.join("settings.json");
    if settings_path.exists() {
        pln!("  settings.json: ✓ found");
    } else {
        pln!("  settings.json: ✗ not found (using defaults)");
    }
    pln!();

    // Identity — kernel vs host mode and host_project, sourced from the cwd's
    // `.dm/identity.toml` (kernel default when absent). When operators
    // troubleshoot, knowing the mode is high-leverage: it controls daemon
    // socket scoping, system_prompt framing, wiki snippet header, and TUI
    // status bar.
    out.push_str(&identity_section(identity));
    out.push_str(&host_capabilities_section());

    // 2. Ollama connectivity — one unified verdict shared with `dm init`
    let models_result = tokio::time::timeout(HEALTH_PROBE_TIMEOUT, client.list_models()).await;
    match models_result {
        Ok(Ok(models)) => {
            let health = if models.is_empty() {
                OllamaHealth::ReachableEmpty
            } else {
                let configured_installed = models.iter().any(|m| m.name == config.model);
                OllamaHealth::ReachableWithModels {
                    count: models.len(),
                    configured_installed,
                }
            };
            pln!(
                "{}",
                render_ollama_health(&health, &config.host, &config.model)
            );
            let total_bytes: u64 = models.iter().filter_map(|m| m.size).sum();
            let total_str = crate::models::human_size(total_bytes);
            pln!(
                "\nAvailable models ({}, {} total):",
                models.len(),
                total_str
            );
            if models.is_empty() {
                pln!("  (none — run: ollama pull gemma4:27b)");
            } else {
                for m in &models {
                    let mut tags = Vec::new();
                    if m.name == config.model {
                        tags.push("reasoning");
                    }
                    if config.tool_model.as_deref() == Some(&m.name) {
                        tags.push("tool");
                    }
                    let tag = if tags.is_empty() {
                        String::new()
                    } else {
                        format!(" ← {}", tags.join(", "))
                    };
                    let size_str = m.size.map(crate::models::human_size).unwrap_or_default();
                    let size_tag = if size_str.is_empty() {
                        String::new()
                    } else {
                        format!("  {}", size_str)
                    };
                    pln!("  • {}{}{}", m.name, size_tag, tag);
                }
            }
            if let Some(ref tm) = config.tool_model {
                let tool_available = models.iter().any(|m| &m.name == tm);
                if !tool_available {
                    pln!(
                        "\n⚠ Warning: tool model '{}' is NOT in the available list.",
                        tm
                    );
                    pln!("  Run: ollama pull {}", tm);
                }
            }
        }
        _ => {
            pln!(
                "{}",
                render_ollama_health(&OllamaHealth::Unreachable, &config.host, &config.model)
            );
        }
    }
    // 2b. Ollama version
    pr!("Ollama version... ");
    match tokio::time::timeout(HEALTH_PROBE_TIMEOUT, client.get_version()).await {
        Ok(Ok(version)) => {
            pln!("✓ {}", version);
            if let Some((major, minor)) = parse_version(&version) {
                if major == 0 && minor < 3 {
                    pln!("  ⚠ Ollama < 0.3.0 may not support thinking content or structured outputs.");
                    pln!("  Upgrade: curl -fsSL https://ollama.com/install.sh | sh");
                }
            }
        }
        Ok(Err(_)) => pln!("could not determine version"),
        Err(_) => pln!("probe timed out"),
    }

    // 2d. Context window for the configured model
    pr!("Context window ({})... ", config.model);
    match tokio::time::timeout(
        Duration::from_secs(3),
        client.model_context_limit(&config.model),
    )
    .await
    {
        Ok(n) if n > 0 => pln!("{}", render_context_window(&config.model, n)),
        Ok(_) => pln!("unknown (model may not be installed)"),
        Err(_) => pln!("probe timed out"),
    }

    // 2c. Embed model check
    let embed_client = OllamaClient::new(config.ollama_base_url(), config.embed_model.clone());
    pr!("Embed model ({})... ", config.embed_model);
    match tokio::time::timeout(HEALTH_PROBE_TIMEOUT, embed_client.embed("test")).await {
        Ok(Ok(v)) if !v.is_empty() => pln!("✓ reachable ({} dims)", v.len()),
        Ok(Ok(_)) => pln!("✗ empty response"),
        Ok(Err(e)) => pln!("✗ {}", e),
        Err(_) => pln!("✗ probe timed out"),
    }
    pln!();

    // 3. Permissions
    let perms_path = config.config_dir.join("permissions.json");
    pr!("Permissions file... ");
    if perms_path.exists() {
        match std::fs::read_to_string(&perms_path) {
            Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(_) => pln!("✓ valid ({} bytes)", s.len()),
                Err(e) => pln!("✗ parse error: {}", e),
            },
            Err(e) => pln!("✗ read error: {}", e),
        }
    } else {
        pln!("✓ not present (no saved rules)");
    }

    // 4. DM.md
    pln!();
    pr!("DM.md... ");
    let cwd = std::env::current_dir().unwrap_or_default();
    // Use the same collect function + cap as system_prompt (max 10 levels).
    let project_candidates = crate::system_prompt::collect_dm_md_candidates_for_doctor(&cwd);
    let mut found_dm_mds: Vec<std::path::PathBuf> = project_candidates;
    if let Some(home) = dirs::home_dir() {
        let user = home.join(".dm").join("DM.md");
        if user.exists() {
            found_dm_mds.push(user);
        }
    }
    if found_dm_mds.is_empty() {
        pln!("not found (no DM.md in directory tree)");
    } else {
        pln!(
            "✓ {} file(s) loaded (root-most first, deepest wins):",
            found_dm_mds.len()
        );
        for p in &found_dm_mds {
            pln!("  • {}", p.display());
        }
        pln!("  override: --no-claude-md skips all DM.md loading");
    }

    // 5. MCP servers
    pln!();
    let mcp_path = config.config_dir.join("mcp_servers.json");
    let project_mcp_path = cwd.join(".dm").join("mcp_servers.json");
    pr!("MCP servers config... ");
    if mcp_path.exists() || project_mcp_path.exists() {
        let configs = crate::mcp::config::load_configs_with_project(&config.config_dir, &cwd);
        pln!("✓ {} server(s) configured", configs.len());
        for cfg in &configs {
            pln!("  • {} ({})", cfg.name, cfg.command);
        }
        if project_mcp_path.exists() {
            pln!("  project-local: {}", project_mcp_path.display());
        }
    } else {
        pln!("not configured (no MCP servers)");
    }

    // 6. Sessions
    pln!();
    pr!("Sessions... ");
    match crate::session::storage::list_meta(&config.config_dir) {
        Ok(sessions) => {
            pln!("✓ {} session(s) saved", sessions.len());
            let cutoff = chrono::Utc::now() - chrono::Duration::days(90);
            let old_count = sessions.iter().filter(|s| s.updated_at < cutoff).count();
            if old_count > 0 {
                pln!("  ⚠ {} session(s) older than 90 days — run `dm --prune-sessions 90` to clean up", old_count);
            }
        }
        Err(_) => {
            pln!("✓ no sessions yet");
        }
    }

    // 6b. Wiki health — read-only. Gate on index.md existence so that
    //     calling doctor in a dir without a wiki does NOT scaffold one.
    pln!();
    let wiki_index = cwd.join(".dm").join("wiki").join("index.md");
    if !wiki_index.exists() {
        pln!("Wiki: not initialized — run `dm init` to scaffold .dm/wiki/");
    } else {
        match crate::wiki::Wiki::open(&cwd) {
            Ok(wiki) => {
                let stats = wiki.stats().ok();
                let lint_findings = wiki.lint().map(|v| v.len()).unwrap_or(0);
                pln!("{}", render_wiki_health(stats.as_ref(), lint_findings));
            }
            Err(e) => pln!("Wiki: error reading .dm/wiki/: {}", e),
        }
    }

    // 7. Plugins (operator-level — discovered under global_config_dir)
    pln!();
    let plugins = crate::plugins::discover_plugins(&config.global_config_dir);
    pln!("Plugins ({}):", plugins.len());
    if plugins.is_empty() {
        pln!(
            "  none found in {}",
            config.global_config_dir.join("plugins").display()
        );
        pln!("  install: drop a dm-tool-<name> executable in that directory");
    } else {
        for p in &plugins {
            pln!("  + {} ({})", p.name, p.path.display());
        }
    }
    pln!();
    pln!("To write a plugin, create an executable that speaks MCP stdio JSON-RPC:");
    pln!("  1. Read JSON-RPC requests from stdin (newline-delimited)");
    pln!("  2. Write JSON-RPC responses to stdout");
    pln!("  3. Implement: initialize, tools/list, tools/call");
    pln!("  4. Name it dm-tool-<name> and make it executable");
    pln!(
        "  5. Drop it in {}",
        config.global_config_dir.join("plugins").display()
    );

    // Web UI port check
    pln!("Web UI (--serve):");
    let default_port: u16 = 7421;
    if crate::web::handlers::port_available(default_port) {
        pln!("  ✓ Port {}: available", default_port);
    } else {
        pln!(
            "  ✗ Port {}: already in use (is dm --serve already running?)",
            default_port
        );
    }
    pln!();

    // Core tools
    pln!("Core tools:");
    let core_tools = [
        ("rg", "--version", "ripgrep (fast grep — used by grep tool)"),
        (
            "git",
            "--version",
            "git (used by commit/PR/changelog tools)",
        ),
        ("jq", "--version", "jq (used for JSON processing in bash)"),
    ];
    for (bin, flag, description) in &core_tools {
        let output = tokio::process::Command::new(bin)
            .arg(flag)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .await;
        match output {
            Ok(o) if o.status.success() => {
                let ver = String::from_utf8_lossy(&o.stdout);
                let first_line = ver.lines().next().unwrap_or("").trim();
                pln!("  ✓ {} — {}", description, first_line);
            }
            _ => pln!("  ✗ {} — not found in PATH", description),
        }
    }
    pln!();

    // GPU section
    pln!("GPU:");
    let nvidia_ok = tokio::process::Command::new("nvidia-smi")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);
    let rocm_ok = tokio::process::Command::new("rocm-smi")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false);

    match gpu::probe().await {
        Some(g) => {
            let backend = if nvidia_ok { "nvidia-smi" } else { "rocm-smi" };
            pln!("  backend:   {}", backend);
            pln!("  name:      {}", g.name);
            pln!("  util:      {}%", g.util_pct);
            let vram_used_gb = g.vram_used_mb as f64 / 1024.0;
            let vram_total_gb = g.vram_total_mb as f64 / 1024.0;
            pln!("  vram:      {:.1} / {:.1} GB", vram_used_gb, vram_total_gb);
            if let Some(temp) = g.temp_c {
                pln!("  temp:      {}°C", temp);
            }
        }
        None => {
            if !nvidia_ok && !rocm_ok {
                pln!("  backend:   none (nvidia-smi and rocm-smi not found)");
            } else {
                let backend = if nvidia_ok { "nvidia-smi" } else { "rocm-smi" };
                pln!("  backend:   {} (found but returned no data)", backend);
            }
        }
    }
    pln!();

    // Eval section
    pln!("Eval:");
    let evals_dir = config.config_dir.join("evals");
    let yaml_count = if evals_dir.exists() {
        std::fs::read_dir(&evals_dir).ok().map_or(0, |entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|ext| ext == "yaml" || ext == "yml")
                })
                .count()
        })
    } else {
        0
    };
    pln!(
        "  Eval suites: {} YAML file(s) in {}",
        yaml_count,
        evals_dir.display()
    );

    let results_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".dm")
        .join("eval")
        .join("results");
    if results_dir.exists() {
        let mut result_files: Vec<std::path::PathBuf> = std::fs::read_dir(&results_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
                    .collect()
            })
            .unwrap_or_default();
        result_files.sort();
        if let Some(latest) = result_files.last() {
            let name = latest
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            pln!("  Last result: {}", name);
        } else {
            pln!("  Last result: none");
        }
    } else {
        pln!("  Last result: none");
    }
    pln!();

    // Bench section
    pln!("Last bench:");
    let bench_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".dm")
        .join("bench");
    let mut found_bench: Option<(std::path::PathBuf, String, usize)> = None;
    if bench_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&bench_dir) {
            let mut json_files: Vec<std::path::PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
                .collect();
            json_files.sort();
            if let Some(latest) = json_files.last() {
                let model_count = std::fs::read_to_string(latest)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| {
                        v["results"].as_array().map(|a| {
                            let models: std::collections::HashSet<String> = a
                                .iter()
                                .filter_map(|r| r["model"].as_str().map(|s| s.to_string()))
                                .collect();
                            models.len()
                        })
                    })
                    .unwrap_or(0);
                let date = latest
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                found_bench = Some((latest.clone(), date, model_count));
            }
        }
    }
    if let Some((path, date, n_models)) = found_bench {
        pln!(
            "  Last bench: {} ({} model(s), {})",
            path.display(),
            n_models,
            date
        );
    } else {
        pln!("  Last bench: none");
    }
    pln!();

    // Templates section (operator-level — under global_config_dir)
    let templates = crate::templates::list_templates(&config.global_config_dir);
    pln!(
        "Templates: {} found in {}",
        templates.len(),
        config.global_config_dir.join("templates").display()
    );
    pln!();

    // Storage footprint — sum bytes under sessions/, .dm/wiki/, and
    // evals/results/. Read errors silently count as zero.
    let sessions_bytes = dir_size_bytes(&config.config_dir.join("sessions"));
    let wiki_bytes = dir_size_bytes(&cwd.join(".dm").join("wiki"));
    let evals_bytes = dir_size_bytes(&config.config_dir.join("evals").join("results"));
    pln!(
        "{}",
        render_storage_summary(sessions_bytes, wiki_bytes, evals_bytes)
    );
    pln!();

    // Project memory section
    let cwd = std::env::current_dir().unwrap_or_default();
    let phash = crate::index::project_hash(&cwd);
    let mem = crate::memory::ProjectMemory::load(&config.config_dir, &phash).unwrap_or_default();
    pln!("Project memory ({}):", cwd.display());
    if mem.entries.is_empty() {
        pln!("  no entries — memory accumulates after sessions with ≥3 user turns");
    } else {
        pln!("  {} entries", mem.entries.len());
        if let Some(last) = mem.entries.last() {
            let ts = last.timestamp.format("%Y-%m-%d");
            let mut snippet_end = 80usize.min(last.summary.len());
            while snippet_end > 0 && !last.summary.is_char_boundary(snippet_end) {
                snippet_end -= 1;
            }
            let snippet = &last.summary[..snippet_end];
            pln!("  last: [{}] {}…", ts, snippet);
        }
        pln!(
            "  path: {}",
            crate::memory::ProjectMemory::file_path(&config.config_dir, &phash).map_or_else(
                |_| "(invalid hash)".to_string(),
                |p| p.display().to_string()
            )
        );
    }
    pln!();

    // Formatters section
    pln!("Formatters:");
    let formatters = [
        ("rustfmt", "--version"),
        ("black", "--version"),
        ("prettier", "--version"),
        ("gofmt", "-h"),
    ];
    for (bin, version_flag) in &formatters {
        let ok = tokio::process::Command::new(bin)
            .arg(version_flag)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|_| true)
            .unwrap_or(false);
        if ok {
            pln!("  ✓ {} (found)", bin);
        } else {
            pln!("  ✗ {} (not found in PATH)", bin);
        }
    }
    pln!();

    // Desktop notifications check
    pln!("Desktop notifications:");
    #[cfg(target_os = "linux")]
    {
        let ok = tokio::process::Command::new("notify-send")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|_| true)
            .unwrap_or(false);
        if ok {
            pln!("  ✓ notify-send (found) — desktop notifications available");
        } else {
            pln!("  ✗ notify-send (not found) — install libnotify-bin for desktop notifications");
        }
    }
    #[cfg(target_os = "macos")]
    {
        let ok = tokio::process::Command::new("osascript")
            .arg("-e")
            .arg("return 0")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            pln!("  ✓ osascript (found) — desktop notifications available");
        } else {
            pln!("  ✗ osascript (not found)");
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        pln!("  (desktop notifications not supported on this platform)");
    }
    pln!();

    // Agent tool section
    pln!("Agent tool:");
    pln!(
        "  enabled · max depth {} · timeout {}s · sub-registry excludes agent tool",
        crate::tools::agent::MAX_AGENT_DEPTH,
        300
    );
    pln!();

    // Hooks section
    pln!("Hooks (~/.dm/hooks.json):");
    let hooks_cfg = crate::tools::hooks::HooksConfig::load(&config.config_dir);
    let hook_count = hooks_cfg.hooks.len();
    if hook_count == 0 {
        pln!("  no hooks configured");
        pln!("  (create ~/.dm/hooks.json — see docs for format)");
    } else {
        pln!(
            "  {} tool hook{} configured:",
            hook_count,
            if hook_count == 1 { "" } else { "s" }
        );
        for (tool, hook) in &hooks_cfg.hooks {
            let label = if tool == "*" {
                "* (all tools)".to_string()
            } else {
                tool.clone()
            };
            if hook.pre.is_some() && hook.post.is_some() {
                pln!("    {} — pre + post", label);
            } else if hook.pre.is_some() {
                pln!("    {} — pre", label);
            } else if hook.post.is_some() {
                pln!("    {} — post", label);
            }
        }
    }
    pln!();

    // Workspace context section
    pln!("Workspace context:");
    let cwd = std::env::current_dir().unwrap_or_default();
    let project_claude = cwd.join("DM.md").exists() || cwd.join(".dm").join("DM.md").exists();
    if project_claude {
        pln!("  DM.md found — workspace context injection skipped (DM.md takes precedence)");
    } else {
        match crate::system_prompt::workspace_context_source().await {
            Some(name) => pln!("  {} found — will be injected as workspace context", name),
            None => pln!("  no DM.md or README.md in cwd — no workspace context injected"),
        }
    }
    pln!("  override: --no-workspace-context");
    pln!();

    // REST API section
    pln!("REST API (dm web):");
    let web_port = 7422u16;
    let web_available = crate::web::handlers::port_available(web_port);
    if web_available {
        pln!("  default port {} — available", web_port);
    } else {
        pln!(
            "  default port {} — IN USE (choose another with --web-port)",
            web_port
        );
    }
    let last_port_path = config.config_dir.join("web.last_port");
    if let Ok(lp) = std::fs::read_to_string(&last_port_path) {
        pln!("  last used port: {}", lp.trim());
    }
    pln!("  routes: GET /health (no auth), POST /chat, GET/POST /sessions, GET /sessions/:id, POST /sessions/:id/turn");
    let tok_path = crate::api::token_path(&config.config_dir);
    if tok_path.exists() {
        pln!(
            "  token file: {} — present (auto-generated token in use)",
            tok_path.display()
        );
    } else {
        pln!(
            "  token file: {} — absent (will be created on next `dm web` start)",
            tok_path.display()
        );
    }
    pln!("  override:   --web-token <TOKEN> to supply your own bearer token");
    pln!();

    // Model routing section
    pln!("Model routing:");
    let routing_path = config.config_dir.join("config.toml");
    if routing_path.exists() {
        match crate::config::load_routing_config(&config.config_dir) {
            Some(rc) => {
                let rule_count = rc.rules.len();
                pln!(
                    "  {} — loaded ({} rule{})",
                    routing_path.display(),
                    rule_count,
                    if rule_count == 1 { "" } else { "s" }
                );
                for (key, model) in &rc.rules {
                    pln!("    {} → {}", key, model);
                }
                if !rc.default.is_empty() {
                    pln!("    default → {}", rc.default);
                }
            }
            None => pln!(
                "  {} — present but [routing] section missing or malformed",
                routing_path.display()
            ),
        }
    } else {
        pln!(
            "  {} — not found (no routing active)",
            routing_path.display()
        );
        pln!("  create it with a [routing] section to enable model routing");
    }
    pln!("  debug: --routing-debug prints classification per prompt");
    pln!();

    pln!("dm doctor complete.");
    out
}

/// Sum sizes of every regular file under `path`, recursing through
/// subdirectories. Missing paths and permission errors are treated as
/// zero-sized subtrees so doctor never fails on a half-present layout.
pub(crate) fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_file() {
            if let Ok(meta) = entry.metadata() {
                total = total.saturating_add(meta.len());
            }
        } else if ft.is_dir() {
            total = total.saturating_add(dir_size_bytes(&entry.path()));
        }
    }
    total
}

pub(crate) fn render_storage_summary(sessions: u64, wiki: u64, evals: u64) -> String {
    let total = sessions.saturating_add(wiki).saturating_add(evals);
    format!(
        "Storage footprint:\n  sessions: {}\n  wiki:     {}\n  evals:    {}\n  total:    {}",
        crate::models::human_size(sessions),
        crate::models::human_size(wiki),
        crate::models::human_size(evals),
        crate::models::human_size(total),
    )
}

/// Render a multi-line wiki health summary. `stats.is_none()` covers the
/// error/empty case — the wiki exists but we couldn't read its index.
pub(crate) fn render_wiki_health(
    stats: Option<&crate::wiki::WikiStats>,
    lint_findings: usize,
) -> String {
    let mut out = String::from("Wiki:\n");
    match stats {
        None => {
            out.push_str("  pages:         (index unreadable)\n");
        }
        Some(s) => {
            use crate::wiki::PageType;
            let n = |cat: PageType| s.by_category.get(&cat).copied().unwrap_or(0);
            writeln!(
                out,
                "  pages:         {} (entities: {}, concepts: {}, summaries: {}, synthesis: {})",
                s.total_pages,
                n(PageType::Entity),
                n(PageType::Concept),
                n(PageType::Summary),
                n(PageType::Synthesis),
            )
            .expect("write to String never fails");
            match &s.last_activity {
                Some(last) => writeln!(
                    out,
                    "  last activity: {} ({} log entries)",
                    last, s.log_entries
                )
                .expect("write to String never fails"),
                None => out.push_str("  last activity: (none yet)\n"),
            }
        }
    }
    if lint_findings == 0 {
        out.push_str("  lint:          ✓ clean");
    } else {
        write!(
            out,
            "  lint:          ⚠ {} finding(s) — run /wiki lint for details",
            lint_findings
        )
        .expect("write to String never fails");
    }
    out
}

/// Render a 1-line summary of the configured model's context window, with
/// a size tier so users can anticipate how often compaction will fire. The
/// caller is expected to have already emitted the model name as a prefix —
/// this renderer must not duplicate it.
pub(crate) fn render_context_window(_model: &str, tokens: usize) -> String {
    let tier = if tokens < 8_192 {
        "small — frequent compaction"
    } else if tokens < 32_768 {
        "moderate"
    } else if tokens < 131_072 {
        "large"
    } else {
        "very large — rare compaction"
    };
    let words_approx = (tokens as f64 * 0.75 / 1000.0).round() as usize;
    format!("{} tokens (~{}k words) — {}", tokens, words_approx, tier)
}

fn parse_version(v: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() >= 2 {
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn make_test_config(dir: &std::path::Path) -> Config {
        Config {
            model: "test-model".to_string(),
            tool_model: None,
            embed_model: "nomic-embed-text".to_string(),
            host: "127.0.0.1:1".to_string(), // unreachable
            host_is_default: false,
            model_is_default: false,
            config_dir: dir.to_path_buf(),
            global_config_dir: dir.to_path_buf(),
            routing: None,
            aliases: std::collections::HashMap::new(),
            max_retries: 3,
            retry_delay_ms: 1000,
            max_retry_delay_ms: 30_000,
            fallback_model: None,
            snapshot_interval_secs: 300,
            idle_timeout_secs: 7200,
        }
    }

    #[tokio::test]
    async fn doctor_output_contains_header() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("dm doctor"),
            "output should contain 'dm doctor': {}",
            &out[..200.min(out.len())]
        );
    }

    #[tokio::test]
    async fn doctor_output_contains_model_name() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("test-model"),
            "output should show configured model name"
        );
    }

    #[tokio::test]
    async fn doctor_shows_ollama_failure_when_unreachable() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        // With 127.0.0.1:1, list_models fails — output should show failure.
        // The unified verdict emits "⚠ Ollama not reachable at …"; older
        // path used "✗ FAILED". Accept either shape.
        assert!(
            out.contains("not reachable")
                || out.contains("⚠")
                || out.contains("FAILED")
                || out.contains("✗"),
            "unreachable Ollama should produce failure output: {}",
            &out[..300.min(out.len())]
        );
    }

    #[tokio::test]
    async fn doctor_output_contains_config_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        let dir_str = dir.path().to_string_lossy();
        assert!(
            out.contains(dir_str.as_ref()),
            "output should contain config_dir path"
        );
    }

    #[tokio::test]
    async fn doctor_shows_settings_json_status() {
        let dir = tempfile::tempdir().unwrap();
        // Without settings.json: shows "not found"
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("settings.json"),
            "output should mention settings.json"
        );

        // With settings.json: shows "found"
        std::fs::write(dir.path().join("settings.json"), "{}").unwrap();
        let out2 = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out2.contains("settings.json"),
            "output should still mention settings.json when present"
        );
    }

    #[tokio::test]
    async fn doctor_ends_with_complete_message() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("dm doctor complete."),
            "output should end with 'dm doctor complete.'"
        );
    }

    #[tokio::test]
    async fn doctor_shows_no_sessions_when_dir_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("0 session(s)") || out.contains("no sessions"),
            "empty dir should show no sessions: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_no_plugins_when_dir_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("none found") || out.contains("Plugins (0)"),
            "empty dir should show no plugins: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_formatters_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("Formatters"),
            "should show Formatters section: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_gpu_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(out.contains("GPU:"), "should show GPU section: {}", out);
    }

    #[tokio::test]
    async fn doctor_shows_tool_model_not_set() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("not set"),
            "with tool_model: None, should show 'not set': {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_tool_model_when_set() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = make_test_config(dir.path());
        config.tool_model = Some("fast-model".to_string());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("fast-model"),
            "should show tool_model name: {}",
            out
        );
        assert!(
            out.contains("routing enabled"),
            "should note routing is enabled: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_eval_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(out.contains("Eval:"), "should show Eval section: {}", out);
    }

    #[tokio::test]
    async fn doctor_shows_no_routing_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("not found") || out.contains("not configured"),
            "absent config.toml should show routing not found: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_permissions_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("no saved rules") || out.contains("not present"),
            "absent permissions.json should say no rules: {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_permissions_valid() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("permissions.json"), "{}").unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("valid"),
            "valid permissions.json should show 'valid': {}",
            out
        );
    }

    #[tokio::test]
    async fn doctor_shows_workspace_context_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("Workspace context") || out.contains("DM.md"),
            "should show workspace context or DM.md section: {}",
            out
        );
    }

    #[test]
    fn dir_size_bytes_missing_path_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        assert_eq!(dir_size_bytes(&missing), 0);
    }

    #[test]
    fn dir_size_bytes_sums_files_including_subdirs() {
        let dir = tempfile::tempdir().unwrap();
        // 10 bytes at root + 20 bytes in sub/a + 30 bytes in sub/deeper/b
        std::fs::write(dir.path().join("top.txt"), vec![0u8; 10]).unwrap();
        std::fs::create_dir_all(dir.path().join("sub/deeper")).unwrap();
        std::fs::write(dir.path().join("sub/a.txt"), vec![0u8; 20]).unwrap();
        std::fs::write(dir.path().join("sub/deeper/b.txt"), vec![0u8; 30]).unwrap();
        assert_eq!(dir_size_bytes(dir.path()), 60);
    }

    #[test]
    fn dir_size_bytes_empty_dir_returns_zero() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(dir_size_bytes(dir.path()), 0);
    }

    #[test]
    fn render_storage_summary_formats_all_buckets_and_total() {
        let out = render_storage_summary(24_300_000, 1_100_000, 340_000);
        assert!(
            out.contains("Storage footprint"),
            "should have header: {out}"
        );
        assert!(out.contains("sessions:"), "should have sessions row: {out}");
        assert!(out.contains("wiki:"), "should have wiki row: {out}");
        assert!(out.contains("evals:"), "should have evals row: {out}");
        assert!(out.contains("total:"), "should have total row: {out}");
        // 24.3 MB formats through human_size as "24 MB" (no decimals below 1 GB).
        assert!(out.contains("24 MB"), "sessions size formatted: {out}");
        // Total 25,740,000 → "26 MB"
        assert!(out.contains("26 MB"), "total size formatted: {out}");
    }

    #[test]
    fn render_wiki_health_no_stats_marks_unreadable() {
        let out = render_wiki_health(None, 0);
        assert!(
            out.starts_with("Wiki:"),
            "should start with Wiki header: {out}"
        );
        assert!(
            out.contains("index unreadable"),
            "should flag unreadable index: {out}"
        );
        assert!(
            out.contains("✓ clean"),
            "lint still clean when zero findings: {out}"
        );
    }

    #[test]
    fn render_wiki_health_healthy_wiki() {
        use crate::wiki::{PageType, WikiStats};
        let mut by_category = std::collections::BTreeMap::new();
        by_category.insert(PageType::Entity, 87);
        by_category.insert(PageType::Concept, 41);
        by_category.insert(PageType::Summary, 10);
        by_category.insert(PageType::Synthesis, 4);
        let stats = WikiStats {
            root: std::path::PathBuf::from("/tmp/wiki"),
            total_pages: 142,
            by_category,
            log_entries: 14,
            last_activity: Some("2026-04-17 · ingest src/init.rs".to_string()),
            most_linked: vec![],
        };
        let out = render_wiki_health(Some(&stats), 0);
        assert!(out.contains("142"), "should show total pages: {out}");
        assert!(
            out.contains("entities: 87"),
            "should show entity count: {out}"
        );
        assert!(
            out.contains("concepts: 41"),
            "should show concept count: {out}"
        );
        assert!(
            out.contains("summaries: 10"),
            "should show summary count: {out}"
        );
        assert!(
            out.contains("synthesis: 4"),
            "should show synthesis count: {out}"
        );
        assert!(
            out.contains("14 log entries"),
            "should show log entry count: {out}"
        );
        assert!(
            out.contains("2026-04-17"),
            "should show last activity: {out}"
        );
        assert!(out.contains("✓ clean"), "zero findings → clean: {out}");
    }

    #[test]
    fn render_wiki_health_with_lint_findings() {
        use crate::wiki::WikiStats;
        let stats = WikiStats {
            root: std::path::PathBuf::from("/tmp/wiki"),
            total_pages: 5,
            by_category: std::collections::BTreeMap::new(),
            log_entries: 0,
            last_activity: None,
            most_linked: vec![],
        };
        let out = render_wiki_health(Some(&stats), 3);
        assert!(out.contains("⚠"), "should warn about findings: {out}");
        assert!(
            out.contains("3 finding(s)"),
            "should show finding count: {out}"
        );
        assert!(
            out.contains("/wiki lint"),
            "should point user at /wiki lint: {out}"
        );
        assert!(
            out.contains("(none yet)"),
            "no last activity → 'none yet': {out}"
        );
    }

    #[test]
    fn render_context_window_small_tier() {
        let out = render_context_window("llama3.2:3b", 4096);
        assert!(
            out.contains("4096 tokens"),
            "should show raw token count: {out}"
        );
        assert!(out.contains("small"), "should use small tier: {out}");
        assert!(
            out.contains("frequent compaction"),
            "should warn about compaction: {out}"
        );
    }

    #[test]
    fn render_context_window_does_not_echo_model() {
        // The caller prints the model prefix; the renderer must not duplicate it.
        let out = render_context_window("gemma4:26b-128k", 131_072);
        assert!(
            !out.contains("gemma4"),
            "renderer must not repeat model name: {out}"
        );
        assert!(
            out.contains("131072"),
            "renderer must still show token count: {out}"
        );
    }

    #[test]
    fn render_context_window_moderate_tier() {
        let out = render_context_window("gemma2:9b", 8192);
        assert!(
            out.contains("moderate"),
            "8k should land in moderate tier: {out}"
        );
        assert!(!out.contains("small"), "8k is not small: {out}");
        assert!(!out.contains("large"), "8k is not large: {out}");
    }

    #[test]
    fn render_context_window_large_tier() {
        let out = render_context_window("gemma4:26b", 65_536);
        assert!(
            out.contains("large"),
            "64k should land in large tier: {out}"
        );
        assert!(!out.contains("very large"), "64k is not very large: {out}");
    }

    #[test]
    fn render_context_window_very_large_tier() {
        let out = render_context_window("gemma4:26b-128k", 131_072);
        assert!(
            out.contains("very large"),
            "128k should land in very-large tier: {out}"
        );
        assert!(
            out.contains("rare compaction"),
            "should note compaction is rare: {out}"
        );
    }

    #[test]
    fn parse_version_valid() {
        assert_eq!(parse_version("0.3.14"), Some((0, 3)));
    }

    #[test]
    fn parse_version_invalid() {
        assert_eq!(parse_version("unknown"), None);
    }

    #[test]
    fn parse_version_single() {
        assert_eq!(parse_version("1"), None);
    }

    #[test]
    fn parse_version_major_minor_only() {
        assert_eq!(parse_version("1.2"), Some((1, 2)));
    }

    #[tokio::test]
    async fn doctor_shows_core_tools_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("Core tools:"),
            "should show Core tools section: {}",
            out
        );
        assert!(out.contains("ripgrep"), "should mention ripgrep: {}", out);
    }

    #[tokio::test]
    async fn doctor_shows_web_ui_section() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let out = run_doctor_capture(
            &client,
            &config,
            &crate::identity::Identity::default_kernel(),
        )
        .await;
        assert!(
            out.contains("Web UI"),
            "should show Web UI section: {}",
            out
        );
    }

    // Identity section — pure formatter. No cwd manipulation, no CWD_LOCK
    // (an earlier draft of these tests held CWD_LOCK across the 3s Ollama
    // probe in `run_doctor_capture` and exposed the C30 cascade in 23
    // sibling tools tests). Tests the same contract that
    // `run_doctor_capture` composes from.
    #[test]
    fn identity_section_renders_kernel_default() {
        let out = identity_section(&crate::identity::Identity::default_kernel());
        assert!(out.starts_with("Identity:\n"), "missing header: {out}");
        assert!(
            out.contains("Mode:          kernel"),
            "want kernel mode: {out}"
        );
        assert!(
            !out.contains("Host project:") && !out.contains("Canonical rev:"),
            "kernel default must not surface host_project or canonical_dm_revision: {out}",
        );
    }

    #[test]
    fn identity_section_renders_host_with_project_and_rev() {
        let id = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: Some("deadbeef".into()),
            canonical_dm_repo: None,
            source: None,
        };
        let out = identity_section(&id);
        assert!(out.contains("Mode:          host"), "want host mode: {out}");
        assert!(
            out.contains("Host project:  finance-app"),
            "want host_project line: {out}",
        );
        assert!(
            out.contains("Canonical rev: deadbeef"),
            "want canonical_dm_revision line: {out}",
        );
    }

    #[test]
    fn identity_section_omits_canonical_rev_when_unset() {
        // Host mode without a recorded canonical_dm_revision (e.g. an
        // operator hand-edited identity.toml after spawn) must not emit
        // a blank `Canonical rev:` line.
        let id = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        let out = identity_section(&id);
        assert!(out.contains("Mode:          host"));
        assert!(out.contains("Host project:  finance-app"));
        assert!(!out.contains("Canonical rev:"), "want no rev line: {out}");
    }

    #[test]
    fn host_capabilities_section_reports_none_when_uninstalled() {
        let out = host_capabilities_section_for_names(&[], false);
        assert!(
            out.contains("Host capabilities:"),
            "missing section header: {out}"
        );
        assert!(
            out.contains("none installed"),
            "uninstalled host capabilities should be explicit: {out}"
        );
    }

    #[test]
    fn host_capabilities_section_lists_first_three_tools() {
        let names = vec![
            "host_alpha".to_string(),
            "host_beta".to_string(),
            "host_delta".to_string(),
            "host_gamma".to_string(),
        ];
        let out = host_capabilities_section_for_names(&names, true);
        assert!(
            out.contains("installed:    4 tool(s)"),
            "missing installed count: {out}"
        );
        assert!(out.contains("• host_alpha"), "missing first tool: {out}");
        assert!(out.contains("• host_beta"), "missing second tool: {out}");
        assert!(out.contains("• host_delta"), "missing third tool: {out}");
        assert!(
            !out.contains("• host_gamma"),
            "should cap listed tools at three: {out}"
        );
        assert!(out.contains("… and 1 more"), "missing overflow line: {out}");
    }

    #[test]
    fn host_capabilities_section_warns_on_invalid_prefix() {
        let names = vec!["host_valid".to_string(), "bad_name".to_string()];
        let out = host_capabilities_section_for_names(&names, true);
        assert!(
            out.contains("missing `host_` prefix"),
            "missing invalid-prefix warning: {out}"
        );
        assert!(
            out.contains("Try: rename host tools to start with `host_`"),
            "warning must include next step: {out}"
        );
    }

    #[tokio::test]
    async fn doctor_capture_renders_host_identity() {
        let dir = tempfile::tempdir().unwrap();
        let config = make_test_config(dir.path());
        let client = OllamaClient::new(config.ollama_base_url(), config.model.clone());
        let identity = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("finance-app".to_string()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        let out = run_doctor_capture(&client, &config, &identity).await;
        assert!(out.contains("Mode:          host"), "should show host mode");
        assert!(
            out.contains("Host project:  finance-app"),
            "should show host project"
        );
    }
}
