//! Acceptance test for Tier 5 — MCP server scope routing in host mode.
//!
//! Run-27 (canonical `b1417e5`) shipped `--mcp-scope {project|global}` plus
//! `mcp::config::load_configs_with_project(global_dir, project_root)`. The
//! contract: a host project may add runtime capabilities by dropping
//! `<project>/.dm/mcp_servers.json` next to its identity, and same-name
//! entries override the operator-wide `~/.dm/mcp_servers.json` for that
//! project only.
//!
//! Kotoba's `run_session` invokes dm with cwd at the kotoba project root, so
//! dm's session init (`main.rs:3396`) calls `load_configs_with_project(...,
//! cwd)` and finds `<project>/.dm/mcp_servers.json`. This test pins that
//! structural guarantee end-to-end at the config layer:
//!
//! 1. A project-local entry is discovered when the global config is empty.
//! 2. A project-local entry with the same name as a global entry **replaces**
//!    the global one (kotoba's MCP override semantics).
//! 3. Global-only entries pass through unchanged when the project also
//!    contributes additional servers — kotoba doesn't lose access to the
//!    operator's global toolkit.
//!
//! These assertions cover the config plumbing without spawning processes.
//! Actual server launch is exercised by the Tier 8 end-to-end smoke (when
//! kotoba ships a stdio MCP example binary).

use dark_matter::mcp::config::load_configs_with_project;
use tempfile::TempDir;

fn write_servers_json(dir: &std::path::Path, contents: &str) {
    std::fs::create_dir_all(dir).expect("create dir");
    std::fs::write(dir.join("mcp_servers.json"), contents).expect("write json");
}

#[test]
fn project_local_mcp_server_is_discovered_in_host_mode() {
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    // Project-local server only — global is empty.
    write_servers_json(
        &project.path().join(".dm"),
        r#"[
            {
                "name": "kotoba-vocab-service",
                "command": "/usr/bin/false",
                "args": ["--stub"]
            }
        ]"#,
    );

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(
        configs.len(),
        1,
        "expected single project-local server entry, got: {:?}",
        configs.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
    assert_eq!(configs[0].name, "kotoba-vocab-service");
    assert_eq!(configs[0].command, "/usr/bin/false");
    assert_eq!(configs[0].args, vec!["--stub".to_string()]);
}

#[test]
fn project_local_entry_overrides_same_name_global_entry() {
    // The kotoba-on-host paradigm: an operator may have a `dictionary` MCP
    // server in `~/.dm/mcp_servers.json` for general use, but the kotoba
    // project ships a Japanese-specialized variant under the same name in
    // its project-local config. The project's variant must win FOR THAT
    // project, without leaking back to other projects on the same machine.
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    write_servers_json(
        global.path(),
        r#"[
            {"name": "dictionary", "command": "/opt/generic-dict", "args": []}
        ]"#,
    );
    write_servers_json(
        &project.path().join(".dm"),
        r#"[
            {"name": "dictionary", "command": "/opt/jp-dict", "args": ["--lang", "ja"]}
        ]"#,
    );

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(
        configs.len(),
        1,
        "same-name override should not duplicate; got: {:?}",
        configs.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
    let dict = &configs[0];
    assert_eq!(dict.name, "dictionary");
    assert_eq!(
        dict.command, "/opt/jp-dict",
        "project-local command should win"
    );
    assert_eq!(dict.args, vec!["--lang".to_string(), "ja".to_string()]);
}

#[test]
fn project_additions_coexist_with_global_only_entries() {
    // Mixed scope: operator has `linter` globally; kotoba adds a
    // domain-specific `vocab-recall` locally. The session should see BOTH.
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    write_servers_json(
        global.path(),
        r#"[
            {"name": "linter", "command": "/usr/local/bin/linter-mcp", "args": []}
        ]"#,
    );
    write_servers_json(
        &project.path().join(".dm"),
        r#"[
            {"name": "vocab-recall", "command": "/opt/kotoba-vocab", "args": ["--mode", "recall"]}
        ]"#,
    );

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(configs.len(), 2);

    let names: Vec<&str> = configs.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.contains(&"linter"),
        "global linter missing: {:?}",
        names
    );
    assert!(
        names.contains(&"vocab-recall"),
        "project vocab-recall missing: {:?}",
        names
    );
}

#[test]
fn missing_project_local_config_falls_back_to_global_only() {
    // Common case for non-kotoba host projects (or kotoba before any MCP
    // wiring): no project-local config file. The loader must not fail; it
    // should just return the global entries unchanged.
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");
    // No `<project>/.dm/mcp_servers.json` at all.

    write_servers_json(
        global.path(),
        r#"[
            {"name": "fmt", "command": "/usr/local/bin/fmt-mcp", "args": []}
        ]"#,
    );

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0].name, "fmt");
}

#[test]
fn malformed_project_local_config_falls_back_silently() {
    // Defensive: a hand-edited `<project>/.dm/mcp_servers.json` with broken
    // JSON should not crash the kotoba session — canonical's loader returns
    // an empty list for malformed files, so the session continues with
    // global-only servers. Mirrors `load_configs_malformed_json_returns_empty`
    // in canonical's mcp::config tests.
    let global = TempDir::new().expect("global tempdir");
    let project = TempDir::new().expect("project tempdir");

    write_servers_json(
        global.path(),
        r#"[{"name": "a", "command": "x", "args": []}]"#,
    );
    write_servers_json(&project.path().join(".dm"), "{not valid json");

    let configs = load_configs_with_project(global.path(), project.path());
    assert_eq!(
        configs.len(),
        1,
        "malformed project-local should fall back to global-only"
    );
    assert_eq!(configs[0].name, "a");
}
