//! `/tool`/`/permissions`/`/mcp`-family command support.
//!
//! Phase 1.2 module 5 (seeded C55). Houses tool-management and
//! permissions dispatch handlers extracted from
//! `commands::execute()`. Migrated: `/tool list` (C55), `/tool disable` (C56),
//! `/tool enable` (C57), `/permissions` (C58), `/mcp` (C59).
//! Family closed: 5/5.

use super::SlashResult;
use crate::tui::app::{App, EntryKind};

/// `/tool list` — forward the literal `/tool list` request to the daemon via `user_tx`.
///
/// Sends `"/tool list"` on `user_tx`, dropping any send error (channel-closed
/// is a no-op at dispatch time). Returns `SlashResult::Done` unconditionally.
pub(super) async fn handle_tool_list(user_tx: &tokio::sync::mpsc::Sender<String>) -> SlashResult {
    user_tx.send("/tool list".to_string()).await.ok();
    SlashResult::Done
}

/// `/tool disable <name>` — forward a tool-disable request to the daemon via `user_tx`.
///
/// Returns `Error("Usage: /tool disable <tool-name>")` if `name` is empty.
/// Otherwise sends `"/tool disable {name}"` on `user_tx`, dropping any send
/// error (channel-closed is a no-op at dispatch time), and returns
/// `SlashResult::Done`.
pub(super) async fn handle_tool_disable(
    name: String,
    user_tx: &tokio::sync::mpsc::Sender<String>,
) -> SlashResult {
    if name.is_empty() {
        return SlashResult::Error("Usage: /tool disable <tool-name>".into());
    }
    user_tx.send(format!("/tool disable {}", name)).await.ok();
    SlashResult::Done
}

/// `/tool enable <name>` — forward a tool-enable request to the daemon via `user_tx`.
///
/// Returns `Error("Usage: /tool enable <tool-name>")` if `name` is empty.
/// Otherwise sends `"/tool enable {name}"` on `user_tx`, dropping any send
/// error (channel-closed is a no-op at dispatch time), and returns
/// `SlashResult::Done`.
pub(super) async fn handle_tool_enable(
    name: String,
    user_tx: &tokio::sync::mpsc::Sender<String>,
) -> SlashResult {
    if name.is_empty() {
        return SlashResult::Error("Usage: /tool enable <tool-name>".into());
    }
    user_tx.send(format!("/tool enable {}", name)).await.ok();
    SlashResult::Done
}

/// `/permissions` — forward a permissions inspection request to the daemon via `user_tx`.
///
/// Sends `"/permissions"` on `user_tx`, dropping any send error
/// (channel-closed is a no-op at dispatch time). Returns `SlashResult::Done`
/// unconditionally.
pub(super) async fn handle_permissions(user_tx: &tokio::sync::mpsc::Sender<String>) -> SlashResult {
    user_tx.send("/permissions".to_string()).await.ok();
    SlashResult::Done
}

/// `/mcp` — render the connected MCP server list as a `SystemInfo` entry.
///
/// Reads `app.mcp_servers` (a `Vec<(String, usize)>` of name+tool-count pairs).
/// Empty list pushes a connect-hint message pointing at
/// `~/.dm/mcp_servers.json`. Otherwise formats one row per server as
/// `"  {name} ({n} tool[s])"` with singular/plural agreement, joined under a
/// `"MCP servers:"` heading. Always returns `SlashResult::Done`.
pub(super) fn handle_mcp(app: &mut App) -> SlashResult {
    if app.mcp_servers.is_empty() {
        app.push_entry(
            EntryKind::SystemInfo,
            "No MCP servers connected. Add servers to ~/.dm/mcp_servers.json".to_string(),
        );
    } else {
        let lines: Vec<String> = app
            .mcp_servers
            .iter()
            .map(|(name, n)| format!("  {} ({} tool{})", name, n, if *n == 1 { "" } else { "s" }))
            .collect();
        app.push_entry(
            EntryKind::SystemInfo,
            format!("MCP servers:\n{}", lines.join("\n")),
        );
    }
    SlashResult::Done
}
