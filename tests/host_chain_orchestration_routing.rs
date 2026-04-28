//! Tier-1 paradigm-gap regression test (kotoba v0.3).
//!
//! Run-31 (canonical-dm `111791e`) routed `Config::config_dir` to
//! `<project>/.dm` in host mode after auditing 13 Config consumers. The
//! `dark_matter::orchestrate::*` module was missed by that sweep — it
//! bypasses `Config` entirely and reaches for `dirs::home_dir().join(".dm")`
//! directly in seven places (chain-pointer save/load/clear, two alias-
//! resolution sites, and `load_chain_config`'s allow-list).
//!
//! Concrete consequence for kotoba: if a host project ever spawns
//! `dark_matter::orchestrate::run_orchestration` from its own process —
//! which Tier 1 of v0.3 explicitly considers — every chain pointer and
//! alias lookup silently leaks to `~/.dm/`, escaping the project
//! boundary that Run-31 was supposed to establish.
//!
//! This test pins the broken behavior. When the canonical fix lands
//! (orchestrate routes through `Config::config_dir` like the rest of the
//! kernel), the assert flips from "pointer in HOME" to "pointer in
//! project root" and the gap doc moves to RESOLVED.
//!
//! See `.dm/wiki/concepts/paradigm-gap-chain-orchestration-host-routing.md`
//! for the full diagnosis + recommended fix.
//!
//! This test is intentionally a **single-test binary** so its `$HOME`
//! override never leaks into other test binaries' parallel scheduling.

#[test]
fn save_last_chain_pointer_writes_to_home_even_when_project_is_host_mode() {
    let fake_home = tempfile::tempdir().expect("home tempdir");
    let project = tempfile::tempdir().expect("project tempdir");

    // Stand up a kotoba-shaped host project: identity.toml says
    // mode=host, host_project=kotoba. Run-31 routes Config to
    // `<project>/.dm` for processes opened in this project. The chain
    // orchestrator should respect that — but does not.
    let project_dm = project.path().join(".dm");
    std::fs::create_dir_all(&project_dm).expect("project .dm dir");
    std::fs::write(
        project_dm.join("identity.toml"),
        "mode = \"host\"\n\
host_project = \"kotoba\"\n\
canonical_dm_revision = \"111791e\"\n",
    )
    .expect("write identity");

    // Override HOME so we can observe the leak without polluting the
    // developer's real `~/.dm/`. `dirs::home_dir()` consults `$HOME`
    // first on Linux/Mac, which is what the orchestrate module uses.
    let prior_home = std::env::var_os("HOME");
    std::env::set_var("HOME", fake_home.path());

    // `save_last_chain_pointer` doesn't `create_dir_all` its parent —
    // it best-efforts the write and swallows the error. For the leak
    // observation to land on disk we must pre-create the kernel-mode
    // `~/.dm/` directory under the fake HOME. (Quietly noted: the
    // best-effort write itself is a secondary gap — IO errors should
    // surface to a warnings sink rather than vanish.)
    std::fs::create_dir_all(fake_home.path().join(".dm")).expect("fake ~/.dm");

    // Workspace is project-local — the natural choice for a host-mode
    // chain run. A correctly-routed orchestrator would write its
    // pointer next to the workspace, under `<project>/.dm/`.
    let workspace = project.path().join("chain-workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    dark_matter::orchestrate::save_last_chain_pointer(&workspace);

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
        leaked.exists(),
        "GAP: orchestrate::save_last_chain_pointer wrote to a path other than \
         the fake HOME's .dm/last_chain.json. The code is supposed to land \
         the pointer at ~/.dm/last_chain.json regardless of host identity, \
         which is the gap. If this assert fails because the file ended up \
         under <project>/.dm/, the canonical fix has landed — flip this test \
         to assert `project_local.exists()` and update the gap doc."
    );
    assert!(
        !project_local.exists(),
        "GAP: orchestrate is identity-blind — it writes to HOME-relative \
         .dm/ rather than the host project's .dm/. The pointer landed at \
         {} (project-local), which would mean the fix already shipped.",
        project_local.display()
    );
}
