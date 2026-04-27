use dark_matter::config::Config;
use dark_matter::wiki::{IngestOutcome, Wiki};
use std::env;
use std::path::Path;

mod domain;
mod host_caps;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Host Project domain logic.
    // The host project parses its own arguments. If the user invokes it with
    // a `dm` subcommand or flag, it yields control to the dark-matter kernel.
    let args: Vec<String> = env::args().collect();
    host_caps::install()?;

    if args.len() > 1 && args[1] == "dm" {
        println!("Delegating to dark-matter kernel...");

        // 2. Kernel initialization.
        let config = Config::load()?;
        println!("Using Ollama model: {}", config.model);

        // 3. Launch the TUI.
        // In a real host project, you would pass `args[2..]` into the CLI parser
        // or a chain runner. For this skeleton, we just start the TUI.
        // Uncomment the line below to actually run the TUI:
        // dark_matter::tui::run::run_tui(&config, None).await?;
        println!("TUI launch disabled in skeleton to prevent taking over terminal.");
        return Ok(());
    }

    let transactions = domain::sample_transactions();
    let balance = domain::projected_balance_cents(&transactions);
    let wiki_result = ingest_domain_module(Path::new(env!("CARGO_MANIFEST_DIR")))?;
    let config = Config::load()?;
    let host_tool_result =
        host_caps::call_installed_host_echo(&config, "host capability online").await?;

    println!("Host skeleton domain logic running.");
    println!("Projected balance: ${:.2}", balance as f64 / 100.0);
    println!("Host tool result: {}", host_tool_result);
    println!("Wiki tracked host file: {}", wiki_result);
    println!("Run with `cargo run -- dm` to see the kernel delegation pattern.");

    Ok(())
}

fn ingest_domain_module(project_root: &Path) -> anyhow::Result<String> {
    let project_root = project_root.canonicalize()?;
    let source_path = project_root.join("src/domain.rs").canonicalize()?;
    let content = std::fs::read_to_string(&source_path)?;
    let wiki = Wiki::open(&project_root)?;

    let message = match wiki.ingest_file(&project_root, &source_path, &content)? {
        IngestOutcome::Ingested { page_rel } => page_rel,
        IngestOutcome::Skipped(reason) => format!("skipped: {:?}", reason),
    };

    Ok(message)
}
