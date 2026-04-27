use std::process::Command;
use tempfile::TempDir;

fn isolated_project() -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let tmp = TempDir::new().expect("tempdir");
    let home = tmp.path().join("home");
    let project = tmp.path().join("project");
    std::fs::create_dir_all(&home).expect("create home");
    std::fs::create_dir_all(project.join(".dm")).expect("create project .dm");
    (tmp, home, project)
}

#[test]
fn mcp_list_project_scope_reads_project_local_config() {
    let (_tmp, home, project) = isolated_project();
    std::fs::write(
        project.join(".dm/mcp_servers.json"),
        r#"[
  {
    "name": "project-server",
    "command": "node",
    "args": ["server.js"],
    "env": {},
    "enabled": true
  }
]"#,
    )
    .expect("write project mcp config");

    let output = Command::new(env!("CARGO_BIN_EXE_dm"))
        .args(["--mcp-list", "--mcp-scope", "project"])
        .current_dir(&project)
        .env("HOME", &home)
        .output()
        .expect("run dm --mcp-list");

    assert!(
        output.status.success(),
        "dm --mcp-list failed. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("MCP scope: project (") && stdout.contains(".dm"),
        "stdout missing project scope header. stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("project-server")
            && stdout.contains("yes")
            && stdout.contains("node server.js"),
        "stdout missing project-local MCP entry. stdout:\n{stdout}"
    );
}

#[test]
fn mcp_not_found_hint_points_at_opposite_scope() {
    let (_tmp, home, project) = isolated_project();

    let output = Command::new(env!("CARGO_BIN_EXE_dm"))
        .args(["--mcp-remove", "missing-server", "--mcp-scope", "project"])
        .current_dir(&project)
        .env("HOME", &home)
        .output()
        .expect("run dm --mcp-remove");

    assert!(
        output.status.success(),
        "dm --mcp-remove exited non-zero. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No MCP server with name 'missing-server' in project scope")
            && stderr.contains("Try: dm --mcp-list --mcp-scope global"),
        "stderr missing opposite-scope hint. stderr:\n{stderr}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Removed 'missing-server'"),
        "missing remove should not print success. stdout:\n{stdout}"
    );
}
