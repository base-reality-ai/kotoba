//! End-to-end smoke for `dm sync` (Tier 2 cycle 14).
//!
//! Exercises the full apply flow against a real `dm` binary subprocess:
//! build a minimal local canonical git repo, set up a host project
//! pinned at the initial canonical HEAD, advance the canonical with a
//! new kernel file, then run `dm sync --status` + `dm sync` and verify
//! host bytes update, `canonical_dm_revision` bumps, the audit log
//! lands in `.dm/wiki/log.md`, and the staging directory is removed.

use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn run_git(dir: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("failed to run git {:?}: {}", args, e));
    assert!(
        output.status.success(),
        "git {:?} failed:\nstdout={}\nstderr={}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn git_commit(repo: &Path, message: &str) {
    run_git(
        repo,
        &[
            "-c",
            "user.name=dm-test",
            "-c",
            "user.email=dm-test@example.invalid",
            "commit",
            "-m",
            message,
        ],
    );
}

fn current_head(repo: &Path) -> String {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn write_file(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

/// Build a tiny compilable Rust crate that resembles what a host project
/// looks like after `dm spawn`. Both the canonical seed and the host
/// project start from this shape so the cargo-check gate has a real
/// crate to compile.
fn write_minimal_crate(root: &Path) {
    write_file(
        root,
        "Cargo.toml",
        "[package]\nname = \"finance-app\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
    );
    write_file(root, "src/lib.rs", "pub fn host_lib() {}\n");
}

#[test]
fn dm_sync_apply_flow_against_local_canonical() {
    // 1. Build a local canonical git repo with a minimal compilable
    //    crate. The host project will pin to this revision and later
    //    sync against an advanced HEAD.
    let canonical_tmp = TempDir::new().unwrap();
    run_git(canonical_tmp.path(), &["init"]);
    write_minimal_crate(canonical_tmp.path());
    run_git(canonical_tmp.path(), &["add", "."]);
    git_commit(canonical_tmp.path(), "canonical: initial");
    let pinned_rev = current_head(canonical_tmp.path());

    // 2. Set up the host project as if `dm spawn` had run: a copy of
    //    the canonical at the pinned revision, plus `.dm/identity.toml`
    //    pointing back at the local canonical.
    let host_tmp = TempDir::new().unwrap();
    write_minimal_crate(host_tmp.path());
    let dm_dir = host_tmp.path().join(".dm");
    std::fs::create_dir_all(&dm_dir).unwrap();
    let canonical_url = canonical_tmp.path().to_string_lossy();
    std::fs::write(
        dm_dir.join("identity.toml"),
        format!(
            "mode = \"host\"\nhost_project = \"finance-app\"\ncanonical_dm_revision = \"{pinned_rev}\"\ncanonical_dm_repo = \"{canonical_url}\"\n"
        ),
    )
    .unwrap();

    // 3. Advance the canonical with a new kernel file. After this, host
    //    is "behind by 1 commit" — exactly what dm sync exists to pull.
    write_file(
        canonical_tmp.path(),
        "src/synced.rs",
        "pub fn synced() -> u8 { 42 }\n",
    );
    run_git(canonical_tmp.path(), &["add", "src/synced.rs"]);
    git_commit(canonical_tmp.path(), "canonical: add synced.rs");
    let head_rev = current_head(canonical_tmp.path());
    assert_ne!(pinned_rev, head_rev, "canonical HEAD must have advanced");

    let bin = env!("CARGO_BIN_EXE_dm");

    // 4. `dm sync --status` reports the revision drift without changing
    //    anything — read-only operation, no apply yet.
    let status = Command::new(bin)
        .args(["sync", "--status"])
        .current_dir(host_tmp.path())
        .output()
        .expect("dm sync --status");
    assert!(
        status.status.success(),
        "dm sync --status exited non-zero. stderr:\n{}",
        String::from_utf8_lossy(&status.stderr),
    );
    let status_out = String::from_utf8_lossy(&status.stdout);
    assert!(
        status_out.contains(&format!("Pinned revision: {pinned_rev}")),
        "status missing pinned revision. stdout:\n{status_out}",
    );
    assert!(
        status_out.contains(&format!("Canonical HEAD: {head_rev}")),
        "status missing canonical HEAD. stdout:\n{status_out}",
    );
    assert!(
        status_out.contains("pinned and canonical HEAD differ"),
        "status must report drift. stdout:\n{status_out}",
    );

    // 5. `dm sync` runs the full apply flow: clone, mirror host into
    //    .dm/sync-staging/staged, overlay canonical kernel changes,
    //    cargo-check gate, swap into host tree, bump revision, append
    //    audit log, remove staging.
    let apply = Command::new(bin)
        .arg("sync")
        .current_dir(host_tmp.path())
        .output()
        .expect("dm sync apply");
    assert!(
        apply.status.success(),
        "dm sync (apply) exited non-zero. stderr:\n{}\nstdout:\n{}",
        String::from_utf8_lossy(&apply.stderr),
        String::from_utf8_lossy(&apply.stdout),
    );
    let apply_out = String::from_utf8_lossy(&apply.stdout);
    assert!(
        apply_out.contains("Applied 1 kernel file(s)."),
        "apply must report 1 file applied. stdout:\n{apply_out}",
    );
    assert!(
        apply_out.contains(&format!("Updated canonical_dm_revision to {head_rev}")),
        "apply must report revision update. stdout:\n{apply_out}",
    );
    assert!(
        apply_out.contains("cargo check: passed."),
        "apply must run the real cargo-check gate (not the skip path). stdout:\n{apply_out}",
    );
    assert!(
        apply_out.contains("Removed sync staging."),
        "apply must clean up the staging tree on success. stdout:\n{apply_out}",
    );

    // 6. Verify the host tree actually carries the canonical bytes.
    let synced_path = host_tmp.path().join("src/synced.rs");
    assert!(synced_path.is_file(), "host must have new kernel file");
    assert_eq!(
        std::fs::read_to_string(&synced_path).unwrap(),
        "pub fn synced() -> u8 { 42 }\n",
        "host bytes must match canonical@HEAD"
    );

    // 7. Verify identity revision was bumped.
    let identity = std::fs::read_to_string(dm_dir.join("identity.toml")).unwrap();
    assert!(
        identity.contains(&format!("canonical_dm_revision = \"{head_rev}\"")),
        "identity must reflect new revision. contents:\n{identity}",
    );

    // 8. Verify the wiki audit log captured the apply.
    let log = std::fs::read_to_string(host_tmp.path().join(".dm/wiki/log.md")).unwrap();
    assert!(
        log.contains("dm sync: applied 1 kernel file(s)"),
        "audit log missing apply entry:\n{log}",
    );
    assert!(
        log.contains(&format!("({pinned_rev} -> {head_rev})")),
        "audit log missing revision transition:\n{log}",
    );

    // 9. Verify the staging tree was cleaned up after success.
    assert!(
        !host_tmp.path().join(".dm/sync-staging").exists(),
        "staging directory must be removed after successful apply"
    );

    // 10. Re-running `dm sync --status` after the apply should now
    //     report "up to date" — the host is pinned at the canonical
    //     HEAD it just synced to.
    let status_after = Command::new(bin)
        .args(["sync", "--status"])
        .current_dir(host_tmp.path())
        .output()
        .expect("dm sync --status after apply");
    assert!(status_after.status.success());
    let status_after_out = String::from_utf8_lossy(&status_after.stdout);
    assert!(
        status_after_out.contains("Status: up to date"),
        "post-apply status must report up to date. stdout:\n{status_after_out}",
    );
}
