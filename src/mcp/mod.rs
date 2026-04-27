//! Model Context Protocol — clients, server, and configuration.
//!
//! `client` spawns external MCP servers and proxies tool calls to them;
//! `server` exposes dm itself over MCP; `config` parses operator-supplied
//! server entries from global `~/.dm/mcp_servers.json` plus optional
//! project-local `.dm/mcp_servers.json`. Lets host projects extend dm's
//! tool surface without recompiling the kernel.

pub mod client;
pub mod config;
pub mod manage;
pub mod server;
pub mod types;
