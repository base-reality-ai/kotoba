//! The Dark Matter (dm) core library.
//!
//! Provides the foundational AI coding agent capabilities, including multi-agent
//! orchestration, the comprehension wiki, and the robust tools surface. When
//! `dm` is spawned into a host project, this library serves as its embedded kernel.

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
// dm controls the filenames it creates (lowercase by convention) and
// matches against Rust ecosystem standards (`.rs`, `.md`, `.toml`,
// etc., always lowercase). Case-sensitive `.ends_with(".rs")` is
// intentional — see `wiki/ingest.rs`, `wiki/lint.rs`, `logging.rs`,
// `orchestrate/mod.rs`, `panic_hook.rs` for representative call sites.
#![allow(clippy::case_sensitive_file_extension_comparisons)]
// Cast truncation is intentional in this codebase: timing arithmetic
// (`Duration::as_millis() as u64`, where u64 holds 584 million years
// of milliseconds), budget percentages (`(usize as f64 * PCT) as usize`,
// bounded by token-count inputs), and bench/ratio calculations on
// internal-bounded data. dm does not process untrusted numerical
// input that could overflow these casts. See `compaction.rs:142–145`,
// `bench/runner.rs:136–189`, `bench/compare.rs:64–85` for representative
// call sites.
#![allow(clippy::cast_possible_truncation)]

// Make `dark_matter` resolve to *this* crate from within the library, so the
// same source files (e.g. doctor.rs, tools/registry.rs) can reference the
// host module via `dark_matter::host::*` — and the path resolves identically
// whether the file is being compiled as part of the library OR pulled into a
// binary via `mod foo;`. Without this, `dark_matter::host` only works from
// binary context (where the library is an external crate); within the library,
// only `crate::host` worked, which was the dual-type-identity footgun the
// kotoba paradigm-gap concept page documents.
extern crate self as dark_matter;

// Library entry point for integration tests.
// Only the modules needed by tests are re-exported here.
pub mod agents;
pub mod api;
pub mod bench;
pub mod changeset;
pub mod compaction;
pub mod config;
pub mod conversation;
pub mod daemon;
pub mod doctor;
pub mod document;
pub mod error_hints;
pub mod eval;
pub mod exit_codes;
pub mod format;
pub mod git;
pub mod gpu;
pub mod host;
pub mod identity;
pub mod index;
pub mod init;
pub mod logging;
pub mod mcp;
pub mod memory;
pub mod models;
pub mod notify;
pub mod ollama;
pub mod orchestrate;
pub mod panic_hook;
pub mod permissions;
pub mod plugins;
pub mod routing;
pub mod run;
pub mod security;
pub mod session;
pub mod share;
pub mod spawn;
pub mod summarize;
pub mod sync;
pub mod system_prompt;
pub mod telemetry;
pub mod templates;
pub mod testfix;
pub mod todo;
pub mod tokens;
pub mod tools;
pub mod translate;
pub mod tui;
pub mod util;
pub mod warnings;
pub mod web;
pub mod wiki;
