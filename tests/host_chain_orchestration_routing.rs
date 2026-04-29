//! Tier-1 paradigm-gap regression test (kotoba v0.3).
//!
//! Originally pinned the BROKEN pre-fix behavior: chain orchestration
//! reached for `dirs::home_dir().join(".dm")` directly, leaking host-
//! project chain pointers and alias resolutions to `~/.dm/`.
//!
//! Canonical-dm `9259acb` (Run-32) closed the gap — orchestrate routes
//! through `Config::config_dir` (project-local in host mode) and
//! `Config::global_config_dir` (operator-global) per Run-31's pattern.
//! After kotoba synced the canonical fix, this test was flipped to
//! confirm the FIX: chain pointer now lands in `<project>/.dm/`, not
//! `~/.dm/`.
//!
//! See `.dm/wiki/concepts/paradigm-gap-chain-orchestration-host-routing.md`
//! (status: RESOLVED in 9259acb).
//!
//! This test is intentionally a **single-test binary** so its `$HOME`
//! override never leaks into other test binaries' parallel scheduling.

#[test]
fn save_last_chain_pointer_writes_project_local_in_host_mode() {
    let fake_home = tempfile::tempdir().expect("home tempdir");
    let project = tempfile::tempdir().expect("project tempdir");

    // Stand up a kotoba-shaped host project: identity.toml says
    // mode=host. Post-fix, the chain orchestrator routes its pointer
    // through Config::config_dir, which Run-31 made identity-aware.
    let project_dm = project.path().join(".dm");
    std::fs::create_dir_all(&project_dm).expect("project .dm dir");
    std::fs::write(
        project_dm.join("identity.toml"),
        "mode = \"host\"\n\
host_project = \"kotoba\"\n\
canonical_dm_revision = \"9259acb\"\n",
    )
    .expect("write identity");

    // Override HOME — pre-fix this is where the pointer leaked. Post-
    // fix, the project-local routing means HOME's `.dm/` should stay
    // untouched.
    let prior_home = std::env::var_os("HOME");
    std::env::set_var("HOME", fake_home.path());

    // Pre-create the fake `~/.dm/` so we can assert it stays empty
    // (vs. the directory simply not existing — distinguishes "no leak"
    // from "directory was never created").
    std::fs::create_dir_all(fake_home.path().join(".dm")).expect("fake ~/.dm");

    // The orchestrator's `save_last_chain_pointer` reads the workspace
    // path's project root via Config (post-fix). Workspace under the
    // host project; pointer should land at `<project>/.dm/`.
    let workspace = project.path().join("chain-workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_current_dir(project.path()).expect("chdir project");
    dark_matter::orchestrate::save_last_chain_pointer(&workspace);
    if let Some(prev) = prior_cwd {
        let _ = std::env::set_current_dir(prev);
    }

    // Restore HOME before any asserts so a panic mid-test still leaves
    // the env clean for any sibling test binaries that get this PID
    // recycled by cargo.
    if let Some(prev) = prior_home {
        std::env::set_var("HOME", prev);
    } else {
        std::env::remove_var("HOME");
    }

    let leaked = fake_home.path().join(".dm").join("last_chain.json");
    let project_local = project_dm.join("last_chain.json");

    assert!(
        project_local.exists(),
        "post-9259acb: orchestrate::save_last_chain_pointer should land \
         at <project>/.dm/last_chain.json (Config::config_dir routing). \
         Did not find {}. If the file ended up under HOME instead, the \
         canonical fix has regressed.",
        project_local.display()
    );
    assert!(
        !leaked.exists(),
        "post-9259acb: orchestrate must NOT leak to HOME's .dm/ in host \
         mode. Found leaked pointer at {}, which would mean the fix \
         regressed.",
        leaked.display()
    );
}
