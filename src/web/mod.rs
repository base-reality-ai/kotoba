//! Axum web runtime for browser and API access to dm sessions.
//!
//! `server` builds routes and websocket handling, `handlers` translates backend
//! events into JSON for clients, and `state` carries shared channels, config,
//! auth, and session metadata across requests.

pub mod handlers;
pub mod server;
pub mod state;
