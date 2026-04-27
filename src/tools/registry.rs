use super::{Tool, ToolResult};
use crate::mcp::types::McpTool;
use crate::ollama::types::{FunctionDefinition, ToolDefinition};
use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// An MCP-backed tool registered in the registry.
pub struct McpToolEntry {
    pub server_name: String,
    pub definition: ToolDefinition,
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    pub mcp_tools: Vec<McpToolEntry>,
    disabled: std::collections::HashSet<String>,
    /// Telemetry: how many times a tool whose name starts with `wiki_` was
    /// dispatched through this registry.
    pub wiki_tool_calls: AtomicUsize,
    /// Telemetry: how many tool results contained the drift warning marker.
    ///
    /// Contract: drift warnings emitted by wiki-aware tools (e.g.
    /// `post_write_wiki_sync`, `check_source_drift`) carry the structured
    /// prefix `[wiki-drift]`. The counter increments on substring match
    /// against that marker — *not* on the human-readable phrase
    /// "may be stale", which collides with unrelated UI hints in
    /// `wiki/summary.rs`.
    pub wiki_drift_warnings: AtomicUsize,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
            mcp_tools: Vec::new(),
            disabled: std::collections::HashSet::new(),
            wiki_tool_calls: AtomicUsize::new(0),
            wiki_drift_warnings: AtomicUsize::new(0),
        }
    }

    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Register a tool supplied by the host project.
    ///
    /// Enforces the `host_` namespace defined in [`crate::host`]: any tool
    /// whose [`Tool::name`] does not start with [`crate::host::HOST_TOOL_PREFIX`]
    /// is rejected with a descriptive error. Also rejects collisions with
    /// already-registered tools (kernel-side or otherwise) so a host crate
    /// cannot silently shadow a kernel tool by reusing its wire name.
    ///
    /// Library-only entry point: the canonical `dm` binary doesn't host
    /// capabilities, so the bin target sees this method as dead. Spawned
    /// host crates link the lib and call this from their startup wiring;
    /// the inline test module exercises the production path.
    #[allow(dead_code)]
    pub fn register_host(&mut self, tool: Box<dyn Tool>) -> Result<()> {
        let name = tool.name();
        if !name.starts_with(crate::host::HOST_TOOL_PREFIX) {
            bail!(
                "host tool '{}' must start with '{}' prefix. Try: rename `fn name(&self)` to return \"{}{}\".",
                name,
                crate::host::HOST_TOOL_PREFIX,
                crate::host::HOST_TOOL_PREFIX,
                name
            );
        }
        if self.tools.contains_key(name) {
            bail!(
                "host tool '{}' collides with a tool already registered. Try: choose a different `host_*` name.",
                name
            );
        }
        if self
            .mcp_tools
            .iter()
            .any(|e| e.definition.function.name == name)
        {
            bail!(
                "host tool '{}' collides with an MCP tool of the same name. Try: choose a different `host_*` name or rename the MCP tool.",
                name
            );
        }
        self.tools.insert(name.to_string(), Arc::from(tool));
        Ok(())
    }

    /// Merge every tool produced by `host` into this registry under the
    /// `host_` namespace. Stops at the first registration failure (for
    /// example, a missing prefix or a name collision) and surfaces the
    /// underlying error so the host can fail fast at startup.
    ///
    /// Library-only entry point: see [`Self::register_host`] for the
    /// kernel/host split rationale.
    #[allow(dead_code)]
    pub fn extend_with_host(&mut self, host: &dyn crate::host::HostCapabilities) -> Result<()> {
        for tool in host.tools() {
            self.register_host(tool)?;
        }
        Ok(())
    }

    /// Register an MCP-provided tool. Its calls are routed through the agent.
    pub fn register_mcp(&mut self, server_name: String, mcp_tool: McpTool) {
        if mcp_tool.name.trim().is_empty() {
            crate::warnings::push_warning(format!(
                "MCP server '{}': skipping tool with empty name. Try: fix the server's tools/list response to include a non-empty tool name.",
                server_name
            ));
            return;
        }
        if let Some(existing) = self
            .mcp_tools
            .iter()
            .find(|e| e.definition.function.name == mcp_tool.name)
        {
            crate::warnings::push_warning(format!(
                "MCP server '{}': tool '{}' already registered by '{}', skipping duplicate. Try: rename one MCP tool or disable one of the duplicate servers.",
                server_name, mcp_tool.name, existing.server_name
            ));
            return;
        }
        if mcp_tool.input_schema.is_none() {
            crate::warnings::push_warning(format!(
                "MCP server '{}': tool '{}' has no input_schema, using default. Try: update the MCP server to return an object input_schema.",
                server_name, mcp_tool.name
            ));
        }
        let def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: mcp_tool.name.clone(),
                description: mcp_tool.description.unwrap_or_default(),
                parameters: mcp_tool
                    .input_schema
                    .unwrap_or(serde_json::json!({"type":"object","properties":{}})),
            },
        };
        self.mcp_tools.push(McpToolEntry {
            server_name,
            definition: def,
        });
    }

    pub fn disable(&mut self, name: &str) -> bool {
        if self.tools.contains_key(name) {
            self.disabled.insert(name.to_string());
            true
        } else {
            false
        }
    }

    pub fn enable(&mut self, name: &str) -> bool {
        self.disabled.remove(name)
    }

    pub fn disabled_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.disabled.iter().cloned().collect();
        names.sort();
        names
    }

    #[allow(dead_code)]
    pub fn is_disabled(&self, name: &str) -> bool {
        self.disabled.contains(name)
    }

    /// All tool definitions — both built-in and MCP — to send to Ollama.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self
            .tools
            .values()
            .filter(|t| !self.disabled.contains(t.name()))
            .map(|t| t.definition())
            .collect();
        defs.extend(self.mcp_tools.iter().map(|e| e.definition.clone()));
        defs
    }

    /// Which MCP server owns this tool name (None if it's a built-in tool).
    pub fn mcp_server_for(&self, tool_name: &str) -> Option<&str> {
        self.mcp_tools
            .iter()
            .find(|e| e.definition.function.name == tool_name)
            .map(|e| e.server_name.as_str())
    }

    /// Remove all registered tools (built-in and MCP). Used by --no-tools flag.
    pub fn clear(&mut self) {
        self.tools.clear();
        self.mcp_tools.clear();
    }

    /// Whether the named tool is read-only (safe to run concurrently).
    /// Returns `false` for unknown tools and MCP tools (conservative default).
    pub fn is_read_only(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|t| t.is_read_only())
    }

    /// Collect system-prompt usage hints from all registered built-in tools.
    /// Returns a formatted block ready for injection into the system prompt.
    ///
    /// Identity-blind: emits a flat list. Prefer
    /// [`Self::system_prompt_hints_for`] in identity-aware contexts so host
    /// capabilities are framed distinctly from the kernel substrate.
    ///
    /// Production callers all migrated to the identity-aware sibling; this
    /// flat method is kept as a stable lib-public entry point for ad-hoc
    /// uses (tests, scripts) that don't have an `Identity` handy.
    #[allow(dead_code)]
    pub fn system_prompt_hints(&self) -> String {
        self.system_prompt_hints_for(&crate::identity::Identity::default_kernel())
    }

    /// Identity-aware sibling of [`Self::system_prompt_hints`].
    ///
    /// In **kernel mode** the output is identical to the flat list (preserves
    /// canonical dm behavior — no surprise framing for the chain that runs in
    /// kernel mode).
    ///
    /// In **host mode** tools are partitioned by the `host_` prefix and
    /// emitted under two headers: host-supplied capabilities first
    /// ("Host capabilities"), then the kernel substrate ("Substrate"). The
    /// agent can then reason about the host's domain layer distinctly from
    /// the always-present kernel tools.
    #[allow(dead_code)]
    pub fn system_prompt_hints_for(&self, identity: &crate::identity::Identity) -> String {
        let mut entries: Vec<_> = self
            .tools
            .iter()
            .filter(|(_, tool)| tool.system_prompt_hint().is_some())
            .collect();
        entries.sort_by_key(|(name, _)| (*name).clone());

        if !identity.is_host() {
            let mut hints = String::new();
            for (name, tool) in &entries {
                if let Some(hint) = tool.system_prompt_hint() {
                    write!(hints, "### {}\n{}\n\n", name, hint)
                        .expect("write to String never fails");
                }
            }
            return hints;
        }

        let (host_entries, kernel_entries): (Vec<_>, Vec<_>) = entries
            .iter()
            .partition(|(name, _)| name.starts_with(crate::host::HOST_TOOL_PREFIX));

        let mut hints = String::new();
        if !host_entries.is_empty() {
            hints.push_str("## Host capabilities\n\n");
            for (name, tool) in &host_entries {
                if let Some(hint) = tool.system_prompt_hint() {
                    write!(hints, "### {}\n{}\n\n", name, hint)
                        .expect("write to String never fails");
                }
            }
        }
        if !kernel_entries.is_empty() {
            hints.push_str("## Substrate (kernel tools)\n\n");
            for (name, tool) in &kernel_entries {
                if let Some(hint) = tool.system_prompt_hint() {
                    write!(hints, "### {}\n{}\n\n", name, hint)
                        .expect("write to String never fails");
                }
            }
        }
        hints
    }

    pub fn tool_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn suggest_tool(&self, name: &str) -> Option<&str> {
        let names = self.tool_names();

        // Substring match: input contains a registered name, or vice versa
        for &registered in &names {
            if registered.contains(name) || name.contains(registered) {
                return Some(registered);
            }
        }

        // Edit distance match
        let mut best: Option<(&str, usize)> = None;
        for &registered in &names {
            let dist = crate::util::levenshtein(name, registered);
            if dist <= 3 && best.is_none_or(|(_, d)| dist < d) {
                best = Some((registered, dist));
            }
        }
        best.map(|(name, _)| name)
    }

    pub async fn call(&self, name: &str, args: Value) -> Result<ToolResult> {
        if self.disabled.contains(name) {
            return Ok(ToolResult {
                content: format!(
                    "Tool '{}' is currently disabled. Use /tool enable {} to re-enable.",
                    name, name
                ),
                is_error: true,
            });
        }
        if name.starts_with("wiki_") {
            self.wiki_tool_calls.fetch_add(1, Ordering::Relaxed);
            crate::telemetry::record_wiki_tool_call();
        }
        let result = match self.tools.get(name) {
            Some(tool) => tool.call(args).await,
            None => {
                if let Some(suggestion) = self.suggest_tool(name) {
                    bail!("Unknown tool: '{}'. Try: '{}'?", name, suggestion);
                }
                bail!(
                    "Unknown tool: '{}'. Try: use one of {}.",
                    name,
                    self.tool_names().join(", ")
                );
            }
        };
        if let Ok(ref r) = result {
            if r.content.contains("[wiki-drift]") {
                self.wiki_drift_warnings.fetch_add(1, Ordering::Relaxed);
                crate::telemetry::record_wiki_drift_warning();
            }
        }
        result
    }
}

/// Build the default tool registry for a given session.
/// `session_id` and `config_dir` are needed by the todo tools.
/// `base_url` and `model` are needed by the `AgentTool` to spawn sub-agents.
/// `embed_model` is used to register `SemanticSearchTool` if an index exists.
pub fn default_registry(
    session_id: &str,
    config_dir: &Path,
    base_url: &str,
    model: &str,
    embed_model: &str,
) -> ToolRegistry {
    default_registry_inner(
        session_id,
        config_dir,
        base_url,
        model,
        embed_model,
        true,
        None,
    )
}

/// Build a registry for sub-agents spawned by `AgentTool`.
/// Identical to `default_registry` but excludes `AgentTool` to prevent
/// unbounded nesting regardless of the depth cap.
pub fn sub_agent_registry(
    session_id: &str,
    config_dir: &Path,
    base_url: &str,
    model: &str,
    embed_model: &str,
) -> ToolRegistry {
    default_registry_inner(
        session_id,
        config_dir,
        base_url,
        model,
        embed_model,
        false,
        None,
    )
}

/// Build a default registry and wire an event channel into `AgentTool` so it
/// can emit `AgentSpawned` / `AgentFinished` events to the TUI.
pub fn default_registry_with_events(
    session_id: &str,
    config_dir: &Path,
    base_url: &str,
    model: &str,
    embed_model: &str,
    event_tx: tokio::sync::mpsc::Sender<crate::tui::BackendEvent>,
) -> ToolRegistry {
    default_registry_inner(
        session_id,
        config_dir,
        base_url,
        model,
        embed_model,
        true,
        Some(event_tx),
    )
}

fn default_registry_inner(
    session_id: &str,
    config_dir: &Path,
    base_url: &str,
    model: &str,
    embed_model: &str,
    include_agent: bool,
    event_tx: Option<tokio::sync::mpsc::Sender<crate::tui::BackendEvent>>,
) -> ToolRegistry {
    use super::{
        agent::AgentTool,
        apply_diff::ApplyDiffTool,
        ask_user_question, bash,
        chain::ChainControlTool,
        file_edit, file_read, file_write, glob, grep,
        ls::LsTool,
        multi_edit::MultiEditTool,
        notebook::{NotebookEditTool, NotebookReadTool},
        semantic_search::SemanticSearchTool,
        todo::{TodoReadTool, TodoWriteTool},
        web_fetch, web_search,
        wiki_lookup::WikiLookupTool,
        wiki_search::WikiSearchTool,
    };
    let mut registry = ToolRegistry::new();

    if include_agent {
        let mut agent = AgentTool::new(
            base_url.to_string(),
            model.to_string(),
            config_dir.to_path_buf(),
        )
        .with_embed_model(embed_model.to_string());
        if let Some(tx) = event_tx {
            agent = agent.with_event_tx(tx);
        }
        registry.register(agent);
    }

    registry.register(ApplyDiffTool);
    registry.register(ask_user_question::AskUserQuestionTool);
    registry.register(bash::BashTool);
    registry.register(file_read::FileReadTool);
    registry.register(file_write::FileWriteTool);
    registry.register(file_edit::FileEditTool);
    registry.register(glob::GlobTool);
    registry.register(grep::GrepTool);
    registry.register(web_fetch::WebFetchTool);
    registry.register(web_search::WebSearchTool);
    registry.register(WikiLookupTool);
    registry.register(WikiSearchTool);
    registry.register(LsTool);
    registry.register(MultiEditTool);
    registry.register(NotebookReadTool);
    registry.register(NotebookEditTool);
    registry.register(ChainControlTool);
    registry.register(TodoWriteTool::new(
        session_id.to_string(),
        config_dir.to_path_buf(),
    ));
    registry.register(TodoReadTool::new(
        session_id.to_string(),
        config_dir.to_path_buf(),
    ));

    let cwd = std::env::current_dir().unwrap_or_default();
    let project_id = crate::index::project_hash(&cwd);
    let index_dir = config_dir.join("index").join(&project_id);
    if index_dir.exists() {
        let embed_client =
            crate::ollama::client::OllamaClient::new(base_url.to_string(), embed_model.to_string());
        registry.register(SemanticSearchTool::new(
            embed_client,
            config_dir.to_path_buf(),
            cwd,
        ));
    }

    // Merge host-supplied capabilities if a host has installed any. In kernel
    // mode (canonical dm) and in any host that hasn't called
    // `install_host_capabilities`, this is a no-op. Failures here are surfaced
    // as warnings instead of panics so a misconfigured host (e.g. tools with
    // colliding names) doesn't take down the runtime — the operator sees the
    // warning via `dm doctor` / startup banner and can fix the host crate.
    if let Some(caps) = crate::host::installed_host_capabilities() {
        if let Err(e) = registry.extend_with_host(caps) {
            crate::warnings::push_warning(format!(
                "host capabilities: {}. Try: review your `HostCapabilities::tools()` impl for prefix and collision rules.",
                e
            ));
        }
    }

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{Tool, ToolResult};
    use async_trait::async_trait;
    use serde_json::{json, Value};

    /// Minimal tool stub for testing registry dispatch.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "echoes args"
        }
        fn parameters(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                content: args.to_string(),
                is_error: false,
            })
        }
    }

    struct AnotherTool;

    #[async_trait]
    impl Tool for AnotherTool {
        fn name(&self) -> &'static str {
            "another"
        }
        fn description(&self) -> &'static str {
            "another tool"
        }
        fn parameters(&self) -> Value {
            json!({"type":"object","properties":{}})
        }
        async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                content: "another".to_string(),
                is_error: false,
            })
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let r = ToolRegistry::new();
        assert!(r.definitions().is_empty());
    }

    #[test]
    fn register_adds_tool_to_definitions() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        let defs = r.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "echo");
    }

    #[tokio::test]
    async fn call_dispatches_to_registered_tool() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        let res = r.call("echo", json!({"x": 1})).await.unwrap();
        assert!(res.content.contains("\"x\""));
    }

    #[tokio::test]
    async fn call_unknown_tool_returns_error() {
        let r = ToolRegistry::new();
        let res = r.call("not_registered", json!({})).await;
        assert!(res.is_err());
        assert!(res.err().unwrap().to_string().contains("Unknown tool"));
    }

    #[test]
    fn clear_removes_all_tools() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        assert_eq!(r.definitions().len(), 2);
        r.clear();
        assert!(r.definitions().is_empty());
    }

    #[test]
    fn definitions_includes_multiple_registered_tools() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        let defs = r.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"another"));
    }

    #[test]
    fn mcp_server_for_returns_none_for_builtin() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        assert!(r.mcp_server_for("echo").is_none());
    }

    #[test]
    fn mcp_server_for_returns_server_name_for_mcp_tool() {
        use crate::mcp::types::McpTool;
        let mut r = ToolRegistry::new();
        r.register_mcp(
            "my_server".to_string(),
            McpTool {
                name: "remote_tool".to_string(),
                description: Some("does something".to_string()),
                input_schema: None,
            },
        );
        assert_eq!(r.mcp_server_for("remote_tool"), Some("my_server"));
        assert!(r.mcp_server_for("nonexistent").is_none());
    }

    #[test]
    fn register_mcp_appears_in_definitions() {
        use crate::mcp::types::McpTool;
        let mut r = ToolRegistry::new();
        r.register_mcp(
            "srv".to_string(),
            McpTool {
                name: "mcp_fn".to_string(),
                description: Some("mcp description".to_string()),
                input_schema: None,
            },
        );
        let defs = r.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "mcp_fn");
        assert_eq!(defs[0].function.description, "mcp description");
    }

    #[test]
    fn is_read_only_returns_false_for_unknown() {
        let r = ToolRegistry::new();
        assert!(!r.is_read_only("nonexistent"));
    }

    #[test]
    fn is_read_only_returns_false_for_mutating_tool() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool); // default is_read_only() → false
        assert!(!r.is_read_only("echo"));
    }

    #[test]
    fn is_read_only_returns_true_for_read_only_tool() {
        // Use FileReadTool which overrides is_read_only → true
        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        assert!(r.is_read_only("read_file"));
    }

    #[test]
    fn system_prompt_hints_collects_tool_hints() {
        let mut r = ToolRegistry::new();
        // EchoTool has no system_prompt_hint (default None)
        r.register(EchoTool);
        // FileReadTool has a hint
        r.register(crate::tools::file_read::FileReadTool);
        let hints = r.system_prompt_hints();
        assert!(
            hints.contains("read_file"),
            "should include read_file tool name"
        );
        assert!(
            hints.contains("read a file before modifying"),
            "should include read_file hint content"
        );
        assert!(
            !hints.contains("### echo"),
            "should not include tools without hints"
        );
    }

    /// Identity-aware grouping pin: in host mode, `system_prompt_hints_for`
    /// emits a "## Host capabilities" header before host_* tools and a
    /// "## Substrate (kernel tools)" header before kernel tools, ordered
    /// stably. Kernel tools never appear under the host header even though
    /// both buckets share the same registry.
    #[test]
    fn system_prompt_hints_for_groups_in_host_mode() {
        struct HostHinted;
        #[async_trait]
        impl Tool for HostHinted {
            fn name(&self) -> &'static str {
                "host_hinted"
            }
            fn description(&self) -> &'static str {
                "host hinted"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            fn system_prompt_hint(&self) -> Option<&'static str> {
                Some("Use the host's domain echo.")
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "host".into(),
                    is_error: false,
                })
            }
        }
        let mut r = ToolRegistry::new();
        // Kernel-side tool that ships a hint: file_read overrides
        // `system_prompt_hint`.
        r.register(crate::tools::file_read::FileReadTool);
        r.register_host(Box::new(HostHinted))
            .expect("register_host");

        let host_identity = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        let out = r.system_prompt_hints_for(&host_identity);

        // Both group headers present.
        assert!(out.contains("## Host capabilities"), "out: {out}");
        assert!(out.contains("## Substrate (kernel tools)"), "out: {out}");
        // Host tools rendered under host header.
        let host_idx = out.find("## Host capabilities").expect("host header");
        let substrate_idx = out
            .find("## Substrate (kernel tools)")
            .expect("substrate header");
        assert!(
            host_idx < substrate_idx,
            "host group must precede substrate: {out}"
        );
        let host_body = &out[host_idx..substrate_idx];
        assert!(
            host_body.contains("### host_hinted"),
            "host_hinted must be under Host header: {host_body}"
        );
        assert!(
            !host_body.contains("### read_file"),
            "kernel tool leaked into host group: {host_body}"
        );
        let substrate_body = &out[substrate_idx..];
        assert!(
            substrate_body.contains("### read_file"),
            "kernel tool missing from substrate: {substrate_body}"
        );
        assert!(
            !substrate_body.contains("### host_hinted"),
            "host tool leaked into substrate: {substrate_body}"
        );
    }

    /// Kernel mode preserves the existing flat output verbatim — no group
    /// headers, no surprise framing for the canonical chain that runs in
    /// kernel mode.
    #[test]
    fn system_prompt_hints_for_kernel_mode_is_flat() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        let kernel = crate::identity::Identity::default_kernel();
        let out = r.system_prompt_hints_for(&kernel);
        assert!(
            !out.contains("## Host capabilities"),
            "no host header in kernel mode: {out}"
        );
        assert!(
            !out.contains("## Substrate"),
            "no substrate header in kernel mode: {out}"
        );
        assert!(out.contains("### read_file"), "out: {out}");
        // Equivalent to the legacy method.
        assert_eq!(out, r.system_prompt_hints());
    }

    /// Host mode suppresses the host bucket when no host capability contributes
    /// a prompt hint. The kernel substrate remains visible so host agents still
    /// see the inherited dm tools they need to operate.
    #[test]
    fn system_prompt_hints_for_host_mode_suppresses_empty_host_bucket() {
        struct HostNoHint;
        #[async_trait]
        impl Tool for HostNoHint {
            fn name(&self) -> &'static str {
                "host_no_hint"
            }
            fn description(&self) -> &'static str {
                "host tool without a prompt hint"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "host".into(),
                    is_error: false,
                })
            }
        }

        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        r.register_host(Box::new(HostNoHint))
            .expect("register_host");

        let host_identity = crate::identity::Identity {
            mode: crate::identity::Mode::Host,
            host_project: Some("finance-app".into()),
            canonical_dm_revision: None,
            canonical_dm_repo: None,
            source: None,
        };
        let out = r.system_prompt_hints_for(&host_identity);

        assert!(
            !out.contains("## Host capabilities"),
            "host bucket should be suppressed when empty: {out}"
        );
        assert!(
            out.contains("## Substrate (kernel tools)"),
            "kernel bucket should remain visible: {out}"
        );
        assert!(out.contains("### read_file"), "out: {out}");
        assert!(
            !out.contains("### host_no_hint"),
            "host tool with no hint should stay out of prompt hints: {out}"
        );
    }

    #[test]
    fn system_prompt_hints_empty_when_no_hints() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        let hints = r.system_prompt_hints();
        assert!(hints.is_empty(), "should be empty when no tools have hints");
    }

    #[test]
    fn suggest_tool_exact_substring() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        assert_eq!(r.suggest_tool("read"), Some("read_file"));
    }

    #[test]
    fn suggest_tool_edit_distance() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::bash::BashTool);
        assert_eq!(r.suggest_tool("bassh"), Some("bash"));
    }

    #[test]
    fn suggest_tool_no_match() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        assert!(r.suggest_tool("xyzzy").is_none());
    }

    #[test]
    fn suggest_tool_contains_match() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_edit::FileEditTool);
        // "edit_file" is a substring of "edit_file_v2"
        assert_eq!(r.suggest_tool("edit_file_v2"), Some("edit_file"));
    }

    #[test]
    fn suggest_tool_empty_registry_returns_none() {
        // Pins the A1 rewrite: the tie-break path starts with `best: None`
        // and must return None cleanly when the registry has nothing to
        // match against.
        let r = ToolRegistry::new();
        assert_eq!(r.suggest_tool("foo"), None);
    }

    #[test]
    fn suggest_tool_picks_minimum_distance() {
        // Pins the tie-break: "red_file" is Levenshtein 1 from "read_file"
        // and ≥3 from the alternatives, so it must pick "read_file".
        let mut r = ToolRegistry::new();
        r.register(crate::tools::file_read::FileReadTool);
        r.register(crate::tools::file_write::FileWriteTool);
        r.register(crate::tools::glob::GlobTool);
        assert_eq!(r.suggest_tool("red_file"), Some("read_file"));
    }

    #[tokio::test]
    async fn call_unknown_tool_shows_suggestion() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::bash::BashTool);
        let err = r.call("bassh", json!({})).await.unwrap_err();
        let msg = err.to_string();
        // Canonical form: "Unknown tool: '<typo>'. Try: '<suggestion>'?"
        assert!(
            msg.contains("Unknown tool"),
            "missing 'Unknown tool' prefix: {}",
            msg
        );
        assert!(msg.contains("'bassh'"), "typo not echoed: {}", msg);
        assert!(
            msg.contains("Try: 'bash'"),
            "missing canonical suggestion form: {}",
            msg
        );
    }

    /// Pin the no-match branch of the unknown-tool path: when fuzzy matching
    /// finds nothing, the error must list the available tools rather than
    /// claim a suggestion.
    #[tokio::test]
    async fn call_unknown_tool_with_no_close_match_lists_available() {
        let mut r = ToolRegistry::new();
        r.register(crate::tools::bash::BashTool);
        let err = r.call("xqzv4r", json!({})).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Unknown tool"),
            "missing 'Unknown tool' prefix: {}",
            msg
        );
        assert!(
            msg.contains("Try: use one of"),
            "missing canonical fallback form: {}",
            msg
        );
        assert!(
            msg.contains("bash"),
            "available tools list should include 'bash': {}",
            msg
        );
    }

    #[test]
    fn tool_names_returns_sorted() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        let names = r.tool_names();
        assert_eq!(names, vec!["another", "echo"]);
    }

    /// Directive metric guard: the production tool registry must register
    /// at least two `wiki_*` tools (`wiki_lookup`, `wiki_search`) so the
    /// model can query the wiki mid-conversation. Pins the "wiki tools
    /// registered ≥ 2" metric the chain tester reports each cycle. A
    /// future cycle that renames or removes either tool trips this test.
    #[test]
    fn default_registry_meets_wiki_tools_directive_metric() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let r = default_registry(
            "test-session",
            tmp.path(),
            "http://localhost:11434",
            "gemma:1b",
            "mxbai-embed-large",
        );
        let names = r.tool_names();
        assert!(
            names.contains(&"wiki_lookup"),
            "wiki_lookup missing from default registry; have: {:?}",
            names
        );
        assert!(
            names.contains(&"wiki_search"),
            "wiki_search missing from default registry; have: {:?}",
            names
        );
        let wiki_count = names.iter().filter(|n| n.starts_with("wiki_")).count();
        assert!(
            wiki_count >= 2,
            "directive metric: ≥ 2 wiki_* tools required, got {}: {:?}",
            wiki_count,
            names
        );
    }

    /// Wire-name registration guard for the edit-side tool family. The wire
    /// name (returned by `Tool::name()`) doesn't always match the file stem:
    /// `file_edit.rs::name() == "edit_file"`, `file_write.rs::name() ==
    /// "write_file"` (verb-object), while `multi_edit.rs::name() ==
    /// "multi_edit"` and `apply_diff.rs::name() == "apply_diff"` (file
    /// stem). See `feedback_tool_wire_name_vs_file_name`. A future cycle
    /// that renames any `fn name()` without re-checking
    /// `default_registry_inner` registration would leave the tool
    /// unreachable through the production dispatch path — every per-tool
    /// drift/usage test calls the tool struct directly and would not
    /// catch the registration drop. This test pins the four canonical
    /// wire names directly so the regression fails at test time.
    #[test]
    fn default_registry_includes_all_edit_tools_by_wire_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let r = default_registry(
            "test-session",
            tmp.path(),
            "http://localhost:11434",
            "gemma:1b",
            "mxbai-embed-large",
        );
        let names = r.tool_names();
        for required in &["edit_file", "multi_edit", "write_file", "apply_diff"] {
            assert!(
                names.contains(required),
                "edit-side tool '{}' missing from default registry; have: {:?}",
                required,
                names
            );
        }
    }

    /// Directive metric guard: `[wiki-drift]` marker must reach the tool
    /// result through the production registry's dispatch path. Pins the
    /// "wiki drift warnings emitted ≥ 1" tracking metric end-to-end —
    /// per-tool drift tests already exist (`file_edit_appends_drift_*`),
    /// but they call the tool struct directly and bypass `r.call(...)`.
    /// A future change that disables `post_write_wiki_sync` in the
    /// registry's dispatch, unregisters `edit_file`, or strips the
    /// marker between the tool and the registry will fail this test.
    /// (Tool name is `edit_file`, not `file_edit`; the source module
    /// is named `file_edit.rs` for symmetry with `file_read`/`file_write`.)
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn default_registry_dispatches_edit_file_with_drift_marker() {
        // Serialize with other tests that touch the global drift counter
        // (`live_file_edit_dispatch_increments_global_drift_counter` asserts
        // `local_atomic == global` and would race if either ran concurrently).
        let _telem = crate::telemetry::telemetry_test_guard();
        let _guard = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().expect("tempdir");
        let proj = dir.path().canonicalize().expect("canonicalize");
        let orig = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(&proj).expect("set cwd");

        // Pre-create + pre-ingest a source file so an entity page exists.
        let file = proj.join("drift_probe.rs");
        std::fs::write(&file, "fn probe() {}\n").expect("write src");
        {
            let wiki = crate::wiki::Wiki::open(&proj).expect("open wiki");
            wiki.ingest_file(&proj, &file, "fn probe() {}\n")
                .expect("seed ingest");
        }

        // Force drift on next edit: disable auto-ingest so the wiki page
        // stays at its pre-edit timestamp while the source advances.
        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        let cfg_dir = dir.path().join("cfg");
        std::fs::create_dir_all(&cfg_dir).expect("cfg dir");
        let registry = default_registry(
            "test-session",
            &cfg_dir,
            "http://localhost:11434",
            "gemma:1b",
            "mxbai-embed-large",
        );

        let res = registry
            .call(
                "edit_file",
                json!({
                    "path": "drift_probe.rs",
                    "old_string": "fn probe() {}",
                    "new_string": "fn probe() { updated }"
                }),
            )
            .await
            .expect("registry dispatch");

        // Restore process-global state BEFORE assertions so a panic
        // here doesn't leak cwd / env into other tests.
        std::env::set_current_dir(&orig).expect("restore cwd");
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!res.is_error, "edit should succeed: {}", res.content);
        assert!(
            res.content.contains("[wiki-drift]"),
            "registry-dispatched edit_file must preserve [wiki-drift] marker: {}",
            res.content
        );
    }

    /// `register_host` rejects a tool whose name lacks the `host_` prefix —
    /// the namespace boundary the spawn paradigm depends on. The error must
    /// be actionable: it names the offending tool, the required prefix, and
    /// gives a `Try:` hint with the corrected name.
    #[test]
    fn register_host_rejects_unprefixed_name() {
        let mut r = ToolRegistry::new();
        let err = r.register_host(Box::new(EchoTool)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("'echo'"), "should name the tool: {msg}");
        assert!(msg.contains("host_"), "should mention the prefix: {msg}");
        assert!(msg.contains("Try:"), "missing actionable hint: {msg}");
        assert!(
            r.tool_names().is_empty(),
            "rejected tool must not be registered: {:?}",
            r.tool_names()
        );
    }

    /// `register_host` accepts a properly prefixed tool and dispatches it
    /// through the normal `call(...)` path — confirming that the host
    /// namespace lives in the same registry as kernel tools.
    #[tokio::test]
    async fn register_host_accepts_prefixed_name_and_is_callable() {
        struct HostEcho;
        #[async_trait]
        impl Tool for HostEcho {
            fn name(&self) -> &'static str {
                "host_echo"
            }
            fn description(&self) -> &'static str {
                "host echo"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: args.to_string(),
                    is_error: false,
                })
            }
        }
        let mut r = ToolRegistry::new();
        r.register_host(Box::new(HostEcho)).expect("register_host");
        assert!(r.tool_names().contains(&"host_echo"));
        let res = r.call("host_echo", json!({"x": 1})).await.unwrap();
        assert!(!res.is_error);
        assert!(res.content.contains("\"x\""), "content: {}", res.content);
    }

    /// A host tool whose name collides with an existing kernel tool is
    /// rejected so the host cannot silently shadow kernel behavior.
    #[test]
    fn register_host_rejects_collision_with_kernel_tool() {
        struct BogusKernelShadow;
        #[async_trait]
        impl Tool for BogusKernelShadow {
            fn name(&self) -> &'static str {
                "host_kernel_shadow"
            }
            fn description(&self) -> &'static str {
                "shadow"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "first".into(),
                    is_error: false,
                })
            }
        }
        struct CollidingHost;
        #[async_trait]
        impl Tool for CollidingHost {
            fn name(&self) -> &'static str {
                "host_kernel_shadow"
            }
            fn description(&self) -> &'static str {
                "collide"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "second".into(),
                    is_error: false,
                })
            }
        }
        let mut r = ToolRegistry::new();
        r.register_host(Box::new(BogusKernelShadow))
            .expect("first register");
        let err = r.register_host(Box::new(CollidingHost)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("collides"), "msg: {msg}");
        assert!(msg.contains("host_kernel_shadow"), "msg: {msg}");
    }

    /// MCP tools can carry any wire name an external server advertises —
    /// including a `host_`-prefixed name. The host-vs-MCP namespace boundary
    /// is enforced by `register_host`'s explicit MCP collision check, not by
    /// the prefix rule alone. A future change that drops that branch would
    /// silently let a host tool shadow an MCP tool of the same name (or vice
    /// versa via dispatch order). This test pins the branch with the
    /// canonical error shape: identifies the offender, names "MCP tool",
    /// and emits a `Try:` hint.
    #[test]
    fn register_host_rejects_collision_with_mcp_tool() {
        use crate::mcp::types::McpTool;
        struct CollidingHost;
        #[async_trait]
        impl Tool for CollidingHost {
            fn name(&self) -> &'static str {
                "host_mcp_collide"
            }
            fn description(&self) -> &'static str {
                "host side of the collision"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "host".into(),
                    is_error: false,
                })
            }
        }
        let mut r = ToolRegistry::new();
        r.register_mcp(
            "external-server".to_string(),
            McpTool {
                name: "host_mcp_collide".to_string(),
                description: Some("mcp side of the collision".into()),
                input_schema: None,
            },
        );
        // MCP tool registered first — host_register must refuse to add a
        // duplicate-named host tool.
        let err = r
            .register_host(Box::new(CollidingHost))
            .expect_err("MCP collision must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("'host_mcp_collide'"),
            "should name the offender: {msg}"
        );
        assert!(
            msg.contains("MCP tool"),
            "should identify the collision class as MCP: {msg}"
        );
        assert!(msg.contains("Try:"), "missing actionable hint: {msg}");

        // The MCP entry remains untouched; the host registration was rejected.
        assert_eq!(r.mcp_tools.len(), 1);
        assert!(
            !r.tool_names().contains(&"host_mcp_collide"),
            "host tool must not be registered after collision rejection: {:?}",
            r.tool_names()
        );
        // MCP tool itself is still surfaced via `mcp_server_for`.
        assert_eq!(
            r.mcp_server_for("host_mcp_collide"),
            Some("external-server")
        );
    }

    /// `extend_with_host` merges every tool returned by a `HostCapabilities`
    /// implementation. Stops at the first failure and surfaces the
    /// registration error.
    #[test]
    fn extend_with_host_merges_tools_and_propagates_errors() {
        use crate::host::HostCapabilities;
        struct GoodTool;
        #[async_trait]
        impl Tool for GoodTool {
            fn name(&self) -> &'static str {
                "host_good"
            }
            fn description(&self) -> &'static str {
                "good"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "ok".into(),
                    is_error: false,
                })
            }
        }
        struct GoodHost;
        impl HostCapabilities for GoodHost {
            fn tools(&self) -> Vec<Box<dyn Tool>> {
                vec![Box::new(GoodTool)]
            }
        }
        let mut r = ToolRegistry::new();
        r.extend_with_host(&GoodHost).expect("extend ok");
        assert!(r.tool_names().contains(&"host_good"));

        // Surface failure path: a host that returns an unprefixed tool fails
        // fast with the prefix error so a misconfigured host crate doesn't
        // boot up with partial registration.
        struct BadHost;
        impl HostCapabilities for BadHost {
            fn tools(&self) -> Vec<Box<dyn Tool>> {
                vec![Box::new(EchoTool)]
            }
        }
        let mut r2 = ToolRegistry::new();
        let err = r2.extend_with_host(&BadHost).unwrap_err();
        assert!(err.to_string().contains("host_"), "msg: {}", err);
        assert!(r2.tool_names().is_empty());
    }

    /// Pins `extend_with_host` partial-failure semantics: when a host's
    /// `tools()` returns `[good_1, bad, good_2]`, the registration loop
    /// short-circuits on `bad`. `good_1` is registered before the error;
    /// `good_2` is silently skipped. This contract is observable in
    /// production via `default_registry_inner` (which pushes a warning on
    /// error but does not roll back), so a future change to atomic-merge
    /// semantics needs to update both the registration loop and the warning
    /// surface — this test fails first when either drifts.
    #[test]
    fn extend_with_host_partial_failure_keeps_tools_registered_before_error() {
        use crate::host::HostCapabilities;
        struct GoodFirst;
        #[async_trait]
        impl Tool for GoodFirst {
            fn name(&self) -> &'static str {
                "host_good_first"
            }
            fn description(&self) -> &'static str {
                "first good"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "first".into(),
                    is_error: false,
                })
            }
        }
        struct GoodSecond;
        #[async_trait]
        impl Tool for GoodSecond {
            fn name(&self) -> &'static str {
                "host_good_second"
            }
            fn description(&self) -> &'static str {
                "second good"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "second".into(),
                    is_error: false,
                })
            }
        }
        struct PartialHost;
        impl HostCapabilities for PartialHost {
            fn tools(&self) -> Vec<Box<dyn Tool>> {
                // Order matters: registration walks left-to-right and bails
                // at the first prefix violation.
                vec![
                    Box::new(GoodFirst),
                    Box::new(EchoTool), // unprefixed — fails the prefix rule
                    Box::new(GoodSecond),
                ]
            }
        }
        let mut r = ToolRegistry::new();
        let err = r.extend_with_host(&PartialHost).unwrap_err();
        assert!(
            err.to_string().contains("host_"),
            "error must surface prefix violation: {err}"
        );

        let names = r.tool_names();
        assert!(
            names.contains(&"host_good_first"),
            "first valid host tool must be registered before the error: {names:?}"
        );
        assert!(
            !names.contains(&"echo"),
            "the offending tool must not be registered: {names:?}"
        );
        assert!(
            !names.contains(&"host_good_second"),
            "subsequent tools after the error are silently skipped: {names:?}"
        );
    }

    /// Default `HostCapabilities::tools()` returns an empty vec, letting a
    /// host opt into other (future) capability methods without supplying
    /// tools. `extend_with_host` against such a host is a no-op.
    #[test]
    fn extend_with_host_default_is_noop() {
        use crate::host::HostCapabilities;
        struct EmptyHost;
        impl HostCapabilities for EmptyHost {}
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.extend_with_host(&EmptyHost).expect("noop");
        assert_eq!(r.tool_names(), vec!["echo"]);
    }

    /// Sub-agents need wiki access too — they reason about the same project
    /// the parent agent does. `sub_agent_registry` omits only `AgentTool`
    /// (to prevent unbounded nesting); the wiki tool surface must remain.
    #[test]
    fn sub_agent_registry_includes_wiki_tools() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let r = sub_agent_registry(
            "test-session",
            tmp.path(),
            "http://localhost:11434",
            "gemma:1b",
            "mxbai-embed-large",
        );
        let names = r.tool_names();
        assert!(names.contains(&"wiki_lookup"), "have: {:?}", names);
        assert!(names.contains(&"wiki_search"), "have: {:?}", names);
        assert!(
            !names.contains(&"agent"),
            "sub-agent registry must exclude AgentTool to prevent nesting; have: {:?}",
            names
        );
    }

    #[test]
    fn disable_hides_from_definitions() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        assert_eq!(r.definitions().len(), 2);
        assert!(r.disable("echo"));
        let defs = r.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].function.name, "another");
    }

    #[tokio::test]
    async fn disable_blocks_call() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        assert!(r.disable("echo"));
        let res = r.call("echo", json!({})).await.unwrap();
        assert!(res.is_error);
        assert!(
            res.content.contains("disabled"),
            "should mention disabled: {}",
            res.content
        );
    }

    #[tokio::test]
    async fn enable_restores_tool() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.disable("echo");
        r.enable("echo");
        let res = r.call("echo", json!({"x": 1})).await.unwrap();
        assert!(!res.is_error);
        assert!(res.content.contains("\"x\""));
    }

    #[test]
    fn disable_unknown_returns_false() {
        let mut r = ToolRegistry::new();
        assert!(!r.disable("nonexistent"));
    }

    #[test]
    fn is_disabled_reflects_state() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        assert!(!r.is_disabled("echo"));
        r.disable("echo");
        assert!(r.is_disabled("echo"));
        r.enable("echo");
        assert!(!r.is_disabled("echo"));
    }

    #[test]
    fn disabled_names_sorted() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register(AnotherTool);
        r.disable("echo");
        r.disable("another");
        assert_eq!(r.disabled_names(), vec!["another", "echo"]);
    }

    #[test]
    fn all_default_tool_definitions_have_valid_schemas() {
        let registry = default_registry(
            "test-session",
            &std::path::PathBuf::from("/tmp"),
            "http://localhost:11434",
            "test-model",
            "",
        );
        let defs = registry.definitions();
        assert!(!defs.is_empty(), "registry should have tools");
        for def in &defs {
            assert!(!def.function.name.is_empty(), "tool name must not be empty");
            assert!(
                !def.function.description.is_empty(),
                "tool {} has empty description",
                def.function.name
            );
            assert!(
                def.function.parameters.is_object(),
                "tool {} parameters must be an object",
                def.function.name
            );
            let props = &def.function.parameters["properties"];
            assert!(
                props.is_object(),
                "tool {} must have properties object",
                def.function.name
            );
        }
    }

    #[test]
    fn clear_also_removes_mcp_tools() {
        use crate::mcp::types::McpTool;
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        r.register_mcp(
            "srv".to_string(),
            McpTool {
                name: "m".to_string(),
                description: None,
                input_schema: Some(serde_json::json!({"type":"object","properties":{}})),
            },
        );
        assert_eq!(r.definitions().len(), 2);
        r.clear();
        assert!(r.definitions().is_empty());
    }

    #[test]
    fn register_mcp_empty_name_warns() {
        use crate::mcp::types::McpTool;
        let marker = "srv_empty_7f3a";
        let mut r = ToolRegistry::new();
        r.register_mcp(
            marker.to_string(),
            McpTool {
                name: String::new(),
                description: None,
                input_schema: None,
            },
        );
        assert!(
            r.mcp_tools.is_empty(),
            "empty name should not be registered"
        );
        let warnings = crate::warnings::peek_warnings();
        let relevant: Vec<_> = warnings.iter().filter(|w| w.contains(marker)).collect();
        assert!(
            relevant.iter().any(|w| w.contains("empty name")),
            "warnings: {:?}",
            relevant
        );
        assert!(
            relevant.iter().any(|w| w.contains("Try:")),
            "warning must include next step: {:?}",
            relevant
        );
    }

    #[test]
    fn register_mcp_duplicate_name_warns() {
        use crate::mcp::types::McpTool;
        let marker = "dup_tool_8b2c";
        let mut r = ToolRegistry::new();
        r.register_mcp(
            "server-a-8b2c".to_string(),
            McpTool {
                name: marker.to_string(),
                description: Some("first".into()),
                input_schema: None,
            },
        );
        r.register_mcp(
            "server-b-8b2c".to_string(),
            McpTool {
                name: marker.to_string(),
                description: Some("second".into()),
                input_schema: None,
            },
        );
        assert_eq!(r.mcp_tools.len(), 1, "duplicate should be rejected");
        assert_eq!(r.mcp_tools[0].server_name, "server-a-8b2c");
        let warnings = crate::warnings::peek_warnings();
        let relevant: Vec<_> = warnings.iter().filter(|w| w.contains(marker)).collect();
        assert!(
            relevant
                .iter()
                .any(|w| w.contains("duplicate") && w.contains("server-b-8b2c")),
            "warnings: {:?}",
            relevant
        );
        assert!(
            relevant.iter().any(|w| w.contains("Try:")),
            "warning must include next step: {:?}",
            relevant
        );
    }

    #[test]
    fn register_mcp_missing_schema_uses_default() {
        use crate::mcp::types::McpTool;
        let marker = "srv_schema_4c9d";
        let mut r = ToolRegistry::new();
        r.register_mcp(
            marker.to_string(),
            McpTool {
                name: "no_schema_tool".to_string(),
                description: None,
                input_schema: None,
            },
        );
        assert_eq!(r.mcp_tools.len(), 1, "tool should still be registered");
        let params = &r.mcp_tools[0].definition.function.parameters;
        assert_eq!(params["type"], "object", "should use default object schema");
        let warnings = crate::warnings::peek_warnings();
        let relevant: Vec<_> = warnings.iter().filter(|w| w.contains(marker)).collect();
        assert!(
            relevant
                .iter()
                .any(|w| w.contains("no input_schema") && w.contains("Try:")),
            "warning must include missing schema and next step: {:?}",
            relevant
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_tool_calls_increments_on_wiki_tool() {
        let _telem = crate::telemetry::telemetry_test_guard();
        let _cwd = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut r = ToolRegistry::new();
        r.register(crate::tools::wiki_search::WikiSearchTool);
        assert_eq!(r.wiki_tool_calls.load(Ordering::Relaxed), 0);
        let _ = r.call("wiki_search", json!({"query": "test"})).await;
        assert_eq!(r.wiki_tool_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn wiki_tool_calls_unchanged_for_non_wiki_tool() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        let _ = r.call("echo", json!({})).await;
        assert_eq!(r.wiki_tool_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn wiki_drift_warnings_increments_on_drift_marker() {
        let _telem = crate::telemetry::telemetry_test_guard();
        struct DriftTool;
        #[async_trait]
        impl Tool for DriftTool {
            fn name(&self) -> &'static str {
                "drift_emitter"
            }
            fn description(&self) -> &'static str {
                "emits drift"
            }
            fn parameters(&self) -> Value {
                json!({"type":"object","properties":{}})
            }
            async fn call(&self, _: Value) -> anyhow::Result<ToolResult> {
                Ok(ToolResult {
                    content: "[wiki-drift] Wiki page 'x' may be stale. Run /wiki refresh."
                        .to_string(),
                    is_error: false,
                })
            }
        }
        let mut r = ToolRegistry::new();
        r.register(DriftTool);
        assert_eq!(r.wiki_drift_warnings.load(Ordering::Relaxed), 0);
        let _ = r.call("drift_emitter", json!({})).await;
        assert_eq!(r.wiki_drift_warnings.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn wiki_drift_warnings_unchanged_without_marker() {
        let mut r = ToolRegistry::new();
        r.register(EchoTool);
        let _ = r.call("echo", json!({})).await;
        assert_eq!(r.wiki_drift_warnings.load(Ordering::Relaxed), 0);
    }

    /// End-to-end: a real `wiki_search` dispatched through the registry
    /// must increment BOTH the local atomic AND the global
    /// `crate::telemetry::WIKI_TOOL_CALLS` counter. The pre-existing
    /// `wiki_tool_calls_increments_on_wiki_tool` only asserts on the local
    /// atomic; this test closes the symmetric gap to the C8 drift-counter
    /// test.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn live_wiki_search_dispatch_increments_global_tool_call_counter() {
        let _telem = crate::telemetry::telemetry_test_guard();
        let _cwd = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let mut registry = ToolRegistry::new();
        registry.register(crate::tools::wiki_search::WikiSearchTool);

        let pre = crate::telemetry::snapshot();
        assert_eq!(pre, (0, 0), "globals must be zeroed by reset()");
        assert_eq!(registry.wiki_tool_calls.load(Ordering::Relaxed), 0);

        // Dispatch through the registry — the path the agent actually takes.
        // Query content is irrelevant to the counter; the increment fires
        // before tool.call() runs (registry gate on `name.starts_with("wiki_")`).
        let _ = registry.call("wiki_search", json!({"query": "any"})).await;

        let post = crate::telemetry::snapshot();
        assert_eq!(
            post.0, 1,
            "global wiki_tool_calls must increment via live dispatch (got {})",
            post.0
        );
        assert_eq!(
            post.1, 0,
            "drift counter must stay 0 — tool-call path is orthogonal to drift marker (got {})",
            post.1
        );
        assert_eq!(
            registry.wiki_tool_calls.load(Ordering::Relaxed),
            post.0,
            "local atomic and global counter must agree for a single dispatch"
        );
        assert_eq!(
            registry.wiki_drift_warnings.load(Ordering::Relaxed),
            0,
            "local drift counter unaffected by non-drifting tool result"
        );
    }

    /// End-to-end: a real `file_edit` dispatched through the registry on
    /// a stale wiki must increment the *global* `crate::telemetry` drift
    /// counter (not just the synthetic local atomic). Closes the gap
    /// between the C7 wiring and the live dispatch path.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn live_file_edit_dispatch_increments_global_drift_counter() {
        let _telem = crate::telemetry::telemetry_test_guard();
        let _cwd = crate::tools::CWD_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().canonicalize().unwrap();
        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&proj).unwrap();

        // Pre-ingest an entity page so the source file has a wiki record.
        let file = proj.join("drift_e2e.rs");
        std::fs::write(&file, "fn probe() {}\n").unwrap();
        {
            let wiki = crate::wiki::Wiki::open(&proj).unwrap();
            wiki.ingest_file(&proj, &file, "fn probe() {}\n").unwrap();
        }

        // Disable auto-ingest so the edit leaves the wiki page stale.
        std::env::set_var("DM_WIKI_AUTO_INGEST", "0");

        // Build a fresh registry and dispatch through it — the path the
        // agent actually takes.
        let mut registry = ToolRegistry::new();
        registry.register(crate::tools::file_edit::FileEditTool);
        let res = registry
            .call(
                "edit_file",
                json!({
                    "path": "drift_e2e.rs",
                    "old_string": "fn probe() {}",
                    "new_string": "fn probe() { updated }"
                }),
            )
            .await
            .unwrap();

        // Restore process-global state BEFORE assertions so a panic
        // doesn't leak the cwd/env into the next test.
        std::env::set_current_dir(&orig).unwrap();
        std::env::remove_var("DM_WIKI_AUTO_INGEST");

        assert!(!res.is_error, "edit should succeed: {}", res.content);
        assert!(
            res.content.contains("[wiki-drift]"),
            "result must carry the structured marker: {}",
            res.content
        );

        let (tool_calls, drift) = crate::telemetry::snapshot();
        assert_eq!(
            tool_calls, 0,
            "file_edit is not a wiki_* tool — tool_calls must not increment"
        );
        assert!(
            drift >= 1,
            "drift counter must move through live dispatch (got {})",
            drift
        );

        // Local atomic on this fresh registry should also reflect the bump.
        assert_eq!(
            registry.wiki_drift_warnings.load(Ordering::Relaxed),
            drift,
            "local atomic and global counter must agree for a single dispatch"
        );
    }
}
