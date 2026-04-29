//! Tier-2 regression test for the kotoba v0.3 chain-orchestration host
//! routing gap (canonical-side fix).
//!
//! Mirrors `kotoba/tests/host_chain_orchestration_routing.rs` with the
//! asserts flipped — after the canonical fix lands, the chain pointer
//! MUST land under `<project>/.dm/last_chain.json` (project-local) when
//! the calling process's cwd has a host-mode `identity.toml`. The
//! kernel-mode counterpart lives in `orchestrate_kernel_routing.rs`.
//!
//! Single-test binary so the cwd / HOME overrides don't leak into other
//! integration tests' parallel scheduling.

use std::path::Path;
use tempfile::TempDir;

fn write_host_identity(root: &Path, host_project: &str) {
    let dm = root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        format!("mode = \"host\"\nhost_project = \"{}\"\n", host_project),
    )
    .expect("write identity.toml");
}

#[test]
fn save_last_chain_pointer_in_host_mode_lands_project_local() {
    let fake_home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir");
    write_host_identity(project.path(), "kotoba");

    // The kotoba reproduction test had to pre-create ~/.dm/ because the
    // pre-fix orchestrate code didn't create_dir_all its parent. The fix
    // now does — so we deliberately do NOT pre-create either dir, and
    // rely on the canonical implementation to land the pointer correctly.
    let prior_home = std::env::var_os("HOME");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_var("HOME", fake_home.path());
    std::env::set_current_dir(project.path()).expect("chdir project");

    let workspace = project.path().join("chain-workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    dark_matter::orchestrate::save_last_chain_pointer(&workspace);

    // Restore process state before any asserts so a panic doesn't leak.
    if let Some(prev) = prior_home {
        std::env::set_var("HOME", prev);
    } else {
        std::env::remove_var("HOME");
    }
    if let Some(prev) = prior_cwd {
        let _ = std::env::set_current_dir(prev);
    }

    let project_local = project.path().join(".dm").join("last_chain.json");
    let leaked_home = fake_home.path().join(".dm").join("last_chain.json");
    assert!(
        project_local.is_file(),
        "host-mode pointer must land at {} after canonical fix",
        project_local.display()
    );
    assert!(
        !leaked_home.exists(),
        "host-mode pointer must NOT leak to {}",
        leaked_home.display()
    );

    // Pointer JSON contains the workspace path — sanity check.
    let body = std::fs::read_to_string(&project_local).expect("read pointer");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("parse pointer JSON");
    assert_eq!(
        parsed["workspace"].as_str(),
        Some(workspace.to_string_lossy().as_ref())
    );
}
