//! Background daemon and client-server protocol.
//!
//! Orchestrates headless tasks, session persistence, and API endpoints
//! for agentic and editor integrations.

pub mod client;
pub mod commands;
pub mod persistence;
pub mod protocol;
pub mod scheduler;
pub mod server;
pub mod session_manager;
pub mod watchdog;

pub use client::{daemon_pid_path, daemon_socket_exists, daemon_socket_path, DaemonClient};
pub use server::run_daemon;
