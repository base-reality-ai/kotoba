use std::path::PathBuf;
use std::process::Command;

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// Ignored in kotoba: this canonical-paradigm-validation test builds the
// `examples/host-skeleton` example. The example's Cargo.toml depends on
// `dark-matter` by package name, but kotoba's package is named `kotoba`
// (with `dark_matter` exposed only as the lib name). The kotoba-side
// workaround in examples/host-skeleton/Cargo.toml sidesteps the build,
// but this canonical test still expects the canonical `dark-matter`
// package shape. See
// `.dm/wiki/concepts/paradigm-gap-spawn-example-cargo-rewrite.md`. When
// the canonical fix lands and `dm spawn` rewrites example Cargo.toml
// files at spawn time, flip the `#[ignore]` here.
#[test]
#[ignore = "host-skeleton example expects canonical package name; kotoba renamed package"]
fn host_skeleton_ingests_domain_file_into_host_wiki() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let example = repo.join("examples/host-skeleton");
    let wiki_dir = example.join(".dm");
    let _ = std::fs::remove_dir_all(&wiki_dir);
    let _cleanup = RemoveOnDrop(wiki_dir.clone());

    let output = Command::new("cargo")
        .args(["run", "--manifest-path"])
        .arg(example.join("Cargo.toml"))
        .arg("--quiet")
        .current_dir(&repo)
        .output()
        .expect("run host-skeleton");

    assert!(
        output.status.success(),
        "host-skeleton failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Host skeleton domain logic running."),
        "missing domain output: {stdout}"
    );
    assert!(
        stdout.contains("Wiki tracked host file: entities/src_domain_rs.md"),
        "missing wiki ingest output: {stdout}"
    );
    assert!(
        stdout.contains("Host tool result: host capability online"),
        "missing host capability output: {stdout}"
    );

    let page = example.join(".dm/wiki/entities/src_domain_rs.md");
    assert!(page.exists(), "domain entity page missing at {:?}", page);
    let page_text = std::fs::read_to_string(&page).expect("read entity page");

    assert!(
        page_text.contains("sources:\n  - src/domain.rs"),
        "entity page should point at host-domain source:\n{page_text}"
    );
}
