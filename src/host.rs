//! Host-project capability extension contract.
//!
//! Dark Matter is a *spine*: a host project grows around it and supplies its
//! own domain capabilities. The [`HostCapabilities`] trait is the stable
//! library entry point host crates implement to register custom tools at
//! startup, alongside dm's kernel tools, without recompiling dm.
//!
//! See `VISION.md` ("Capability extension") for the paradigm.
//!
//! # Naming convention
//!
//! Host-supplied tools MUST carry a `host_` prefix in their
//! [`Tool::name`] — e.g. `host_categorize_transaction`.
//! The prefix is enforced at registration time by
//! [`ToolRegistry::register_host`](crate::tools::registry::ToolRegistry::register_host).
//! Kernel tools carry no prefix. The split prevents name collisions and lets
//! identity-aware code paths (wiki snippet, system prompt) reason about which
//! tools belong to which layer.
//!
//! `host_` (underscore) was chosen over `host:` because Ollama / OpenAI-style
//! tool-call schemas restrict function names to `[A-Za-z0-9_-]`. The prefix
//! mirrors the existing `wiki_` convention used by [`crate::tools::wiki_search`]
//! and friends.
//!
//! # Example
//!
//! ```no_run
//! use async_trait::async_trait;
//! use dark_matter::host::HostCapabilities;
//! use dark_matter::tools::{Tool, ToolResult};
//! use serde_json::{json, Value};
//!
//! struct HostEchoTool;
//!
//! #[async_trait]
//! impl Tool for HostEchoTool {
//!     fn name(&self) -> &'static str { "host_echo" }
//!     fn description(&self) -> &'static str { "Echoes the host's input." }
//!     fn parameters(&self) -> Value {
//!         json!({
//!             "type": "object",
//!             "properties": { "message": { "type": "string" } },
//!             "required": ["message"]
//!         })
//!     }
//!     async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
//!         let message = args.get("message").and_then(Value::as_str).unwrap_or("");
//!         Ok(ToolResult { content: message.to_string(), is_error: false })
//!     }
//! }
//!
//! struct MyHost;
//!
//! impl HostCapabilities for MyHost {
//!     fn tools(&self) -> Vec<Box<dyn Tool>> {
//!         vec![Box::new(HostEchoTool)]
//!     }
//! }
//! ```
//!
//! The host installs its capabilities once at startup. Every subsequent
//! [`default_registry`](crate::tools::registry::default_registry) call
//! (TUI, daemon, web, chain, sub-agents) automatically merges the
//! installed host tools alongside kernel tools:
//!
//! ```no_run
//! # use dark_matter::host::{HostCapabilities, install_host_capabilities};
//! # use dark_matter::tools::Tool;
//! # struct MyHost;
//! # impl HostCapabilities for MyHost {
//! #     fn tools(&self) -> Vec<Box<dyn Tool>> { vec![] }
//! # }
//! // Call once, before launching dm's TUI / daemon / etc.
//! install_host_capabilities(Box::new(MyHost)).expect("install host caps");
//! ```
//!
//! For ad-hoc registries (tests, demo paths) a host can also merge
//! directly without going through the global hook:
//!
//! ```no_run
//! # use dark_matter::host::HostCapabilities;
//! # use dark_matter::tools::{Tool, ToolResult, registry::ToolRegistry};
//! # struct MyHost;
//! # impl HostCapabilities for MyHost {
//! #     fn tools(&self) -> Vec<Box<dyn Tool>> { vec![] }
//! # }
//! let mut registry = ToolRegistry::new();
//! registry.extend_with_host(&MyHost).expect("host registration");
//! ```

use crate::tools::{Tool, ToolResult};
use std::sync::OnceLock;

/// Contract a host project's crate implements to extend dm's tool registry.
///
/// `dm spawn`-produced projects depend on `dark-matter` as a library and
/// implement this trait on a domain-owned type. The dm runtime calls
/// [`HostCapabilities::tools`] once during startup and merges every returned
/// tool into the global registry under the `host_` namespace.
///
/// Implementers should return owned trait objects (`Box<dyn Tool>`) so the
/// registry can take ownership and erase the concrete type.
///
/// This trait is the *narrow* surface: today it only exposes tool registration.
/// Future capabilities (slash commands, agent personas, wiki ingestors) will
/// land as additional methods with default no-op implementations so existing
/// host crates do not break.
///
/// `Send + Sync` are required because installed capabilities are stored in a
/// process-global slot consulted from every default-registry construction —
/// including async tasks on the tokio runtime.
pub trait HostCapabilities: Send + Sync {
    /// Tools to merge into dm's registry alongside its kernel tools.
    ///
    /// Each tool's [`Tool::name`] MUST start with `host_`; otherwise
    /// registration fails with a descriptive error. Default `vec![]` lets a
    /// host opt in to other capability methods without supplying tools.
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        Vec::new()
    }
}

/// Required prefix for every host-supplied tool name. See module docs.
pub const HOST_TOOL_PREFIX: &str = "host_";

/// Process-global slot for host capabilities. Set once at startup via
/// [`install_host_capabilities`]; consulted by
/// [`crate::tools::registry::default_registry`] and friends so every kernel
/// surface picks up host tools without per-call-site plumbing.
static HOST_CAPS: OnceLock<Box<dyn HostCapabilities>> = OnceLock::new();

/// Install the host project's capabilities for this process.
///
/// Call once at startup, before launching the TUI / daemon / chain. Every
/// subsequent default-registry construction merges the installed host tools
/// alongside kernel tools. In kernel mode (canonical dm), no host installs
/// capabilities and the registry behaves exactly as before.
///
/// Returns an error on the second call so a host crate cannot accidentally
/// shadow its own capabilities mid-run with a different set. The error
/// includes a `Try:` hint pointing at [`ToolRegistry::extend_with_host`]
/// for the ad-hoc-registry path.
///
/// [`ToolRegistry::extend_with_host`]: crate::tools::registry::ToolRegistry::extend_with_host
#[allow(dead_code)]
pub fn install_host_capabilities(
    caps: Box<dyn HostCapabilities>,
) -> Result<(), HostCapabilitiesAlreadyInstalled> {
    HOST_CAPS
        .set(caps)
        .map_err(|_| HostCapabilitiesAlreadyInstalled)
}

/// Accessor for the currently-installed host capabilities. Returns `None`
/// in kernel mode and in any spawned host that hasn't called
/// [`install_host_capabilities`] yet.
///
/// `pub` (not `pub(crate)`) so the dm binary can read it via
/// `dark_matter::host::installed_host_capabilities()` after the
/// kotoba paradigm-gap fix removed the binary's local `mod host;`
/// duplication.
pub fn installed_host_capabilities() -> Option<&'static dyn HostCapabilities> {
    HOST_CAPS.get().map(|b| b.as_ref())
}

/// Returned by [`install_host_capabilities`] when capabilities have already
/// been installed in this process. Holds no payload — the caller already
/// knows which install they attempted.
#[derive(Debug)]
pub struct HostCapabilitiesAlreadyInstalled;

impl std::fmt::Display for HostCapabilitiesAlreadyInstalled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "host capabilities already installed for this process. Try: install once at startup, or use ToolRegistry::extend_with_host for ad-hoc registries."
        )
    }
}

impl std::error::Error for HostCapabilitiesAlreadyInstalled {}

/// Errors surfaced by [`invoke_tool`] before the tool itself runs.
/// Tool-side errors (returned by [`Tool::call`]) are propagated as
/// [`InvokeError::Tool`] so the caller can distinguish dispatch
/// failures from in-tool failures.
#[derive(Debug)]
pub enum InvokeError {
    /// `name` did not start with [`HOST_TOOL_PREFIX`]. Carries the
    /// offending name so the caller can echo it in error messages.
    MissingPrefix(String),
    /// No host capabilities have been installed in this process via
    /// [`install_host_capabilities`].
    NoHostInstalled,
    /// The installed host capabilities did not register a tool with
    /// `name`.
    UnknownTool(String),
    /// The tool ran but returned an error from its `call(...)` future.
    Tool(anyhow::Error),
}

impl std::fmt::Display for InvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvokeError::MissingPrefix(name) => write!(
                f,
                "host tool `{}` does not carry the `{}` prefix. Try: only host-namespaced tools are reachable.",
                name, HOST_TOOL_PREFIX
            ),
            InvokeError::NoHostInstalled => write!(
                f,
                "no host capabilities installed in this process. Try: install a HostCapabilities impl at startup before invoking host tools."
            ),
            InvokeError::UnknownTool(name) => write!(
                f,
                "host tool `{}` not registered by the installed host capabilities. Try: confirm the host's `tools()` returns a tool whose `name()` matches.",
                name
            ),
            InvokeError::Tool(e) => write!(f, "host tool failed: {}", e),
        }
    }
}

impl std::error::Error for InvokeError {}

/// Resolve a host-namespaced tool against the installed
/// [`HostCapabilities`] registry and dispatch it.
///
/// Enforces the [`HOST_TOOL_PREFIX`] guard, builds a one-shot view of
/// the host's tools, finds the named tool, and awaits `Tool::call`.
/// Used by the daemon's `host.invoke` RPC and by any other surface
/// that needs deterministic host-tool dispatch without going through
/// the model loop.
///
/// Returns [`InvokeError`] on every failure shape (missing prefix,
/// no host installed, unknown tool, tool-side error). Successful
/// invocations return the tool's [`ToolResult`] verbatim — `is_error`
/// stays meaningful for tool-internal failures.
pub async fn invoke_tool(name: &str, args: serde_json::Value) -> Result<ToolResult, InvokeError> {
    if !name.starts_with(HOST_TOOL_PREFIX) {
        return Err(InvokeError::MissingPrefix(name.to_string()));
    }
    let caps = installed_host_capabilities().ok_or(InvokeError::NoHostInstalled)?;
    let tool = caps
        .tools()
        .into_iter()
        .find(|t| t.name() == name)
        .ok_or_else(|| InvokeError::UnknownTool(name.to_string()))?;
    tool.call(args).await.map_err(InvokeError::Tool)
}
