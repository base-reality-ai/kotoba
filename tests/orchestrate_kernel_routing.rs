//! Tier-2 backwards-compat regression test: in kernel mode (no
//! `identity.toml`), `save_last_chain_pointer` MUST still write to
//! `~/.dm/last_chain.json` — Run-31's hard rule that existing operators
//! see no movement when `mode = kernel`.
//!
//! Single-test binary so the cwd / HOME overrides don't leak into other
//! integration tests' parallel scheduling.

use tempfile::TempDir;

#[test]
fn save_last_chain_pointer_in_kernel_mode_stays_in_home_dm() {
    let fake_home = TempDir::new().expect("home tempdir");
    let project = TempDir::new().expect("project tempdir (kernel)");
    // No identity.toml → kernel mode by default.

    let prior_home = std::env::var_os("HOME");
    let prior_cwd = std::env::current_dir().ok();
    std::env::set_var("HOME", fake_home.path());
    std::env::set_current_dir(project.path()).expect("chdir project");

    let workspace = project.path().join("chain-workspace");
    std::fs::create_dir_all(&workspace).expect("workspace dir");
    dark_matter::orchestrate::save_last_chain_pointer(&workspace);

    if let Some(prev) = prior_home {
        std::env::set_var("HOME", prev);
    } else {
        std::env::remove_var("HOME");
    }
    if let Some(prev) = prior_cwd {
        let _ = std::env::set_current_dir(prev);
    }

    let kernel_target = fake_home.path().join(".dm").join("last_chain.json");
    let project_local = project.path().join(".dm").join("last_chain.json");
    assert!(
        kernel_target.is_file(),
        "kernel-mode pointer must remain at {} (Run-31 backwards-compat)",
        kernel_target.display()
    );
    assert!(
        !project_local.exists(),
        "kernel mode must not write a project-local pointer at {}",
        project_local.display()
    );
}
