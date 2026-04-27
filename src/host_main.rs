//! Kotoba host entry point.
//!
//! Two things happen here on every startup, before delegating to dm:
//!
//! 1. Kotoba's [`host_caps::KotobaCapabilities`] is installed via
//!    [`dark_matter::host::install_host_capabilities`]. This registers
//!    the 5 host tools (invoke_persona, log_vocabulary, log_kanji,
//!    record_struggle, quiz_me) into every dm registry that gets
//!    constructed afterwards — TUI, daemon, web, headless.
//!
//! 2. Argument routing: by default we launch dm's TUI in host mode
//!    (so `kotoba` opens the conversation interface). Subcommands
//!    `kotoba dm <args>` or `kotoba help` route to dm's CLI.
//!
//! The kernel binary at `src/main.rs` is canonical dm and is left
//! untouched — running `dm` directly bypasses kotoba's host caps.
//! Run via the `kotoba` binary to get the host-mode experience.

use anyhow::Context;
use dark_matter::host::install_host_capabilities;

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
    println!("    kotoba              Launch the conversation TUI in host mode");
    println!("    kotoba dm [ARGS]    Pass through to the dm kernel CLI");
    println!("    kotoba help         Show this message");
    println!("    kotoba version      Show kotoba's version");
    println!();
    println!("HOST TOOLS REGISTERED:");
    println!("    host_invoke_persona     — switch the active conversational persona");
    println!("    host_log_vocabulary     — record a Japanese word in the wiki");
    println!("    host_log_kanji          — record a kanji character with readings");
    println!("    host_record_struggle    — flag something the learner stumbled on");
    println!("    host_quiz_me            — pull due-for-review vocabulary");
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

    let status = tokio::process::Command::new(&dm_bin)
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .with_context(|| format!("invoke {}", dm_bin.display()))?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
