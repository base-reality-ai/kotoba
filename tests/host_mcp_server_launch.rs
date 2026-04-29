//! Acceptance test for Tier 5 — MCP server **actually running** in host mode.
//!
//! `tests/host_mcp_project_scope.rs` pinned config-layer routing (project-local
//! discovery + global-override semantics). This test takes the next step the
//! directive's Tier 5 #2-3 calls for: a real subprocess MCP server spawned
//! from kotoba's project-local `<project>/.dm/mcp_servers.json`, with its
//! tools enumerated via `tools/list` and one tool actually invoked via
//! `tools/call`.
//!
//! Mirrors canonical `mcp/client.rs` test patterns (lines 287-302) which
//! embed a tiny Python stdio MCP server that handles the standard MCP
//! handshake (initialize → notifications/initialized → tools/list →
//! tools/call). Python3 is the same runtime the canonical suite assumes;
//! the test is skipped with a warning if it isn't available so kotoba CI
//! degrades gracefully on minimal images.
//!
//! Surfaces the gap-shape the directive predicted for Tier 5: actual server
//! spawn from a project-local config (not just config parsing) — does the
//! handshake complete, are tools listed, are tool calls round-trippable?

use dark_matter::mcp::client::McpClient;
use dark_matter::mcp::config::load_configs_with_project;
use serde_json::json;
use tempfile::TempDir;

/// Embedded stdio MCP server. Mirrors `mcp::client::tests::full_server_script`
/// but exposes a kotoba-themed tool name so a kotoba-side test can pin a
/// concrete contract: a project-local server's tool reaches kotoba.
///
/// Flat Python — no indented blocks — because Rust string line-continuation
/// strips leading whitespace. Handles exactly one each of: initialize,
/// initialized notification, tools/list, tools/call. After the call, the
/// process exits cleanly. That matches the McpClient test pattern.
const KOTOBA_MCP_PYTHON: &str = "import sys, json\n\
req = json.loads(sys.stdin.readline())\n\
sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'protocolVersion':'2024-11-05','capabilities':{}}}) + '\\n')\n\
sys.stdout.flush()\n\
sys.stdin.readline()\n\
req = json.loads(sys.stdin.readline())\n\
sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'tools':[{'name':'kotoba_dictionary_lookup','description':'Look up a Japanese word in a stub dictionary.','input_schema':{'type':'object','properties':{'word':{'type':'string'}}}}]}}) + '\\n')\n\
sys.stdout.flush()\n\
req = json.loads(sys.stdin.readline())\n\
word = req.get('params', {}).get('arguments', {}).get('word', '')\n\
sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'content':[{'type':'text','text':'mcp-stub:' + word}]}}) + '\\n')\n\
sys.stdout.flush()\n";

fn python3_available() -> bool {
    which::which("python3").is_ok()
}

fn write_servers_json(dir: &std::path::Path, contents: &str) {
    std::fs::create_dir_all(dir).expect("create dir");
    std::fs::write(dir.join("mcp_servers.json"), contents).expect("write json");
}

#[tokio::test]
async fn project_local_mcp_server_actually_launches_and_lists_tools() {
    if !python3_available() {
        eprintln!(
            "host_mcp_server_launch: skipping — python3 not on PATH. Install \
             python3 to exercise the real-spawn dimension of Tier 5."
        );
        return;
    }

    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    // Project-local config points at the embedded Python MCP server.
    let server_json = format!(
        r#"[
            {{
                "name": "kotoba-dict-stub",
                "command": "python3",
                "args": ["-c", {script}]
            }}
        ]"#,
        script = serde_json::Value::String(KOTOBA_MCP_PYTHON.to_string())
    );
    write_servers_json(&project.path().join(".dm"), &server_json);

    // Step 1: kotoba's host project loads its MCP configs the same way dm
    // session-init does (canonical/main.rs:3397).
    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(
        configs.len(),
        1,
        "expected single project-local server config; got: {:?}",
        configs.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
    let cfg = &configs[0];
    assert_eq!(cfg.name, "kotoba-dict-stub");

    // Step 2: spawn the server as a real subprocess and complete the
    // initialize handshake.
    let arg_refs: Vec<&str> = cfg.args.iter().map(String::as_str).collect();
    let mut client = McpClient::spawn(&cfg.command, &arg_refs)
        .await
        .expect("spawn project-local MCP server");

    // Step 3: tools/list returns the kotoba-themed tool.
    let tools = client
        .list_tools()
        .await
        .expect("list_tools from project-local MCP server");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(
        names.contains(&"kotoba_dictionary_lookup"),
        "expected kotoba_dictionary_lookup in MCP server tools; got: {:?}",
        names
    );

    // Step 4: tools/call round-trips through the spawned process. The stub
    // server echoes "mcp-stub:<word>" so we can assert the path actually
    // reached the server (vs. canonical mocking the response).
    let result = client
        .call_tool("kotoba_dictionary_lookup", json!({ "word": "学校" }))
        .await
        .expect("call_tool round-trips");
    assert!(
        result.contains("mcp-stub:学校"),
        "expected echoed word from stub server; got: {result}"
    );
}

#[tokio::test]
async fn missing_command_in_project_local_config_surfaces_spawn_error() {
    // Defensive: a project-local config with an unreachable command must
    // produce an actionable error from `McpClient::spawn`, not panic. This
    // matches what dm session-init at canonical/main.rs:3420 already
    // tolerates (logs to stderr and continues).
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    write_servers_json(
        &project.path().join(".dm"),
        r#"[
            {
                "name": "kotoba-broken",
                "command": "/nonexistent/binary/that/cannot/be/found-12345",
                "args": []
            }
        ]"#,
    );

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(configs.len(), 1);
    let cfg = &configs[0];
    let arg_refs: Vec<&str> = cfg.args.iter().map(String::as_str).collect();

    let err = McpClient::spawn(&cfg.command, &arg_refs)
        .await
        .err()
        .expect("spawn should fail for nonexistent command");
    let msg = err.to_string();
    assert!(
        !msg.is_empty(),
        "spawn error must carry a message for the operator"
    );
}
