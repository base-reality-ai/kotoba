//! HTTP client and supporting utilities for the Ollama LLM backend.
//!
//! `client` runs the streaming POST loop; `health` probes for installed
//! models; `pull` triggers downloads; `retry` wraps transient failures.
//! All other dm subsystems talk to Ollama through these primitives.

pub mod client;
pub mod health;
pub mod hints;
pub mod model_hints;
pub mod pull;
pub mod retry;
pub mod types;
