/// Auto-detect the appropriate lint command for the current directory.
pub fn detect_lint_cmd(cwd: &std::path::Path) -> String {
    if cwd.join("Cargo.toml").exists() {
        "cargo clippy 2>&1".to_string()
    } else if cwd.join("package.json").exists() {
        "npx eslint .".to_string()
    } else if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
        "ruff check .".to_string()
    } else if cwd.join("go.mod").exists() {
        "go vet ./... 2>&1".to_string()
    } else {
        "cargo clippy 2>&1".to_string()
    }
}

/// Auto-detect the appropriate test command for the current directory.
pub fn detect_test_cmd(cwd: &std::path::Path) -> String {
    if cwd.join("Cargo.toml").exists() {
        "cargo test 2>&1".to_string()
    } else if cwd.join("package.json").exists() {
        "npm test 2>&1".to_string()
    } else if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
        "pytest 2>&1".to_string()
    } else if cwd.join("go.mod").exists() {
        "go test ./... 2>&1".to_string()
    } else {
        "cargo test 2>&1".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_lint_cargo_for_rust_project() {
        // Use the dm project itself (has Cargo.toml)
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cmd = detect_lint_cmd(&cwd);
        assert!(
            cmd.contains("cargo clippy"),
            "expected cargo clippy, got: {}",
            cmd
        );
    }

    #[test]
    fn detect_lint_node_for_js_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let cmd = detect_lint_cmd(dir.path());
        assert!(
            cmd.contains("eslint"),
            "expected eslint for package.json: {}",
            cmd
        );
    }

    #[test]
    fn detect_lint_python_for_pyproject() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        let cmd = detect_lint_cmd(dir.path());
        assert!(
            cmd.contains("ruff"),
            "expected ruff for pyproject.toml: {}",
            cmd
        );
    }

    #[test]
    fn detect_lint_python_for_setup_py() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("setup.py"), "").unwrap();
        let cmd = detect_lint_cmd(dir.path());
        assert!(cmd.contains("ruff"), "expected ruff for setup.py: {}", cmd);
    }

    #[test]
    fn detect_lint_fallback_for_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = detect_lint_cmd(dir.path());
        assert!(
            cmd.contains("cargo clippy"),
            "expected fallback cargo clippy: {}",
            cmd
        );
    }

    #[test]
    fn detect_lint_go_for_go_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module example.com/m").unwrap();
        let cmd = detect_lint_cmd(dir.path());
        assert!(
            cmd.contains("go vet"),
            "expected go vet for go.mod: {}",
            cmd
        );
    }

    // ── detect_test_cmd ─────────────────────────────────────────────────────

    #[test]
    fn detect_test_cargo_for_rust() {
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cmd = detect_test_cmd(&cwd);
        assert!(
            cmd.contains("cargo test"),
            "expected cargo test, got: {}",
            cmd
        );
    }

    #[test]
    fn detect_test_npm_for_node() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(cmd.contains("npm test"), "expected npm test: {}", cmd);
    }

    #[test]
    fn detect_test_pytest_for_python() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(cmd.contains("pytest"), "expected pytest: {}", cmd);
    }

    #[test]
    fn detect_test_pytest_for_setup_py() {
        // Mirrors `detect_lint_python_for_setup_py` on the test side. The
        // production code accepts either pyproject.toml *or* setup.py as a
        // Python project signal; without this test, a refactor that drops
        // the setup.py branch from `detect_test_cmd` (but keeps it in
        // `detect_lint_cmd`) would silently fall through to `cargo test`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("setup.py"), "").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("pytest"),
            "expected pytest for setup.py: {}",
            cmd
        );
    }

    #[test]
    fn detect_test_go_for_go_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module example.com/m").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(cmd.contains("go test"), "expected go test: {}", cmd);
    }

    /// Pin the if-let precedence in `detect_test_cmd` for projects with
    /// multiple language signals. Order is: Cargo.toml > package.json >
    /// pyproject.toml/setup.py > go.mod > fallback. Without this test, a
    /// future re-ordering of the if-let chain (e.g. moving Python ahead of
    /// Cargo for Maturin-style dual projects) would silently change which
    /// command runs in mixed-stack repos.
    #[test]
    fn detect_test_cmd_precedence_pins_cargo_then_npm_then_python_then_go() {
        // Cargo wins over package.json.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("cargo test"),
            "Cargo.toml + package.json must yield cargo test (Cargo wins): {}",
            cmd
        );

        // Cargo wins over pyproject.toml.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("cargo test"),
            "Cargo.toml + pyproject.toml must yield cargo test: {}",
            cmd
        );

        // Cargo wins over go.mod.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("cargo test"),
            "Cargo.toml + go.mod must yield cargo test: {}",
            cmd
        );

        // npm wins over python.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("npm test"),
            "package.json + pyproject.toml must yield npm test (npm wins): {}",
            cmd
        );

        // pytest wins over go.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("pytest"),
            "pyproject.toml + go.mod must yield pytest (Python wins): {}",
            cmd
        );

        // Full stack: Cargo still wins.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[project]").unwrap();
        std::fs::write(dir.path().join("go.mod"), "module x").unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("cargo test"),
            "all-signals project must yield cargo test (Cargo at top of chain): {}",
            cmd
        );
    }

    #[test]
    fn detect_test_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = detect_test_cmd(dir.path());
        assert!(
            cmd.contains("cargo test"),
            "expected fallback cargo test: {}",
            cmd
        );
    }
}
