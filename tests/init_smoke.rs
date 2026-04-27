use std::process::Command;
use tempfile::TempDir;

#[test]
fn test_init_appends_to_gitignore() {
    let tmp = TempDir::new().unwrap();
    let bin = env!("CARGO_BIN_EXE_dm");

    // Create a dummy .gitignore to simulate an existing project
    let gitignore_path = tmp.path().join(".gitignore");
    std::fs::write(&gitignore_path, "target/\n").unwrap();

    // Initialize an empty git repo so dm recognizes it as a git project
    let git_status = Command::new("git")
        .args(["init"])
        .current_dir(tmp.path())
        .status()
        .expect("failed to execute git init");
    assert!(git_status.success(), "git init failed");

    // Run `dm init`
    let status = Command::new(bin)
        .arg("init")
        .current_dir(tmp.path())
        .status()
        .expect("failed to execute dm init");

    assert!(status.success(), "dm init failed");

    // Read the .gitignore and assert that both .dm/ and .dm-workspace/ are present
    let content = std::fs::read_to_string(&gitignore_path).unwrap();
    assert!(
        content.contains(".dm/"),
        ".gitignore should contain .dm/ after dm init"
    );
    assert!(
        content.contains(".dm-workspace/"),
        ".gitignore should contain .dm-workspace/ after dm init"
    );
    assert!(
        content.contains("target/"),
        ".gitignore should still contain original lines"
    );
}
