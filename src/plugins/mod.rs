//! Filesystem plugin discovery for external dm tools.
//!
//! Scans `<config_dir>/plugins/` for executable `dm-tool-<name>` binaries,
//! validates names, and returns deterministic plugin metadata for registration
//! with the tool surface.

use std::path::{Path, PathBuf};

pub struct PluginConfig {
    pub name: String,
    pub path: PathBuf,
    pub args: Vec<String>,
}

/// Discover plugins in `<config_dir>/plugins/`. A plugin is any executable
/// named `dm-tool-<name>` in that directory.
pub fn discover_plugins(config_dir: &Path) -> Vec<PluginConfig> {
    let plugins_dir = config_dir.join("plugins");
    let Ok(entries) = std::fs::read_dir(&plugins_dir) else {
        return Vec::new();
    };

    let mut plugins: Vec<PluginConfig> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            if !is_executable(&e.path()) {
                return None;
            }
            let filename = e.file_name().to_string_lossy().to_string();
            let name = filename.strip_prefix("dm-tool-")?.to_string();
            if name.is_empty() {
                return None;
            }
            if !name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                return None;
            }
            Some(PluginConfig {
                name,
                path: e.path(),
                args: Vec::new(),
            })
        })
        .collect();

    // Sort by name for deterministic ordering
    plugins.sort_by(|a, b| a.name.cmp(&b.name));
    plugins
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.extension().map(|e| e == "exe").unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    fn make_executable(dir: &TempDir, name: &str) -> PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, b"#!/bin/sh\necho hi\n").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();
        p
    }

    fn make_executable_in(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"#!/bin/sh\necho hi\n").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();
        p
    }

    fn make_non_executable(dir: &TempDir, name: &str) -> PathBuf {
        let p = dir.path().join(name);
        std::fs::write(&p, b"not executable").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&p, perms).unwrap();
        p
    }

    #[test]
    fn is_executable_true_for_executable_file() {
        let dir = TempDir::new().unwrap();
        let p = make_executable(&dir, "my-script");
        assert!(
            is_executable(&p),
            "file with mode 0o755 should be executable"
        );
    }

    #[test]
    fn is_executable_false_for_non_executable_file() {
        let dir = TempDir::new().unwrap();
        let p = make_non_executable(&dir, "my-data");
        assert!(
            !is_executable(&p),
            "file with mode 0o644 should not be executable"
        );
    }

    #[test]
    fn discover_empty_dir_returns_no_plugins() {
        let config_dir = TempDir::new().unwrap();
        std::fs::create_dir(config_dir.path().join("plugins")).unwrap();
        let plugins = discover_plugins(config_dir.path());
        assert!(plugins.is_empty());
    }

    #[test]
    fn discover_missing_plugins_dir_returns_empty() {
        let config_dir = TempDir::new().unwrap();
        // No "plugins" directory created
        let plugins = discover_plugins(config_dir.path());
        assert!(plugins.is_empty());
    }

    #[test]
    fn discover_finds_executable_dm_tool_plugin() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();
        let plugins_dir_as_tempdir = TempDir::new().unwrap();
        // Create inside the actual plugins dir
        let p = plugins_dir.join("dm-tool-myplugin");
        std::fs::write(&p, b"#!/bin/sh\necho hi\n").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();
        drop(plugins_dir_as_tempdir);

        let plugins = discover_plugins(config_dir.path());
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "myplugin");
    }

    #[test]
    fn discover_ignores_non_dm_tool_files() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        // Create a non-dm-tool executable
        let p = plugins_dir.join("other-tool");
        std::fs::write(&p, b"#!/bin/sh").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();

        let plugins = discover_plugins(config_dir.path());
        assert!(plugins.is_empty(), "non dm-tool-* files should be ignored");
    }

    #[test]
    fn discover_ignores_non_executable_plugins() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        // Create dm-tool-* file but NOT executable
        let p = plugins_dir.join("dm-tool-noexec");
        std::fs::write(&p, b"#!/bin/sh").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&p, perms).unwrap();

        let plugins = discover_plugins(config_dir.path());
        assert!(
            plugins.is_empty(),
            "non-executable plugins should be ignored"
        );
    }

    #[test]
    fn discover_sorts_plugins_by_name() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        for name in &["dm-tool-zzz", "dm-tool-aaa", "dm-tool-mmm"] {
            let p = plugins_dir.join(name);
            std::fs::write(&p, b"#!/bin/sh").unwrap();
            let mut perms = std::fs::metadata(&p).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&p, perms).unwrap();
        }

        let plugins = discover_plugins(config_dir.path());
        assert_eq!(plugins.len(), 3);
        assert_eq!(plugins[0].name, "aaa");
        assert_eq!(plugins[1].name, "mmm");
        assert_eq!(plugins[2].name, "zzz");
    }

    #[test]
    fn discover_plugin_has_empty_args_by_default() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        let p = plugins_dir.join("dm-tool-mytool");
        std::fs::write(&p, b"#!/bin/sh").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();

        let plugins = discover_plugins(config_dir.path());
        assert_eq!(plugins.len(), 1);
        assert!(
            plugins[0].args.is_empty(),
            "plugins should have no default args"
        );
    }

    #[test]
    fn discover_skips_plugin_with_dots_in_name() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        make_executable_in(&plugins_dir, "dm-tool-foo.bar");
        let plugins = discover_plugins(config_dir.path());
        assert!(
            plugins.is_empty(),
            "plugin with dots in name should be skipped"
        );
    }

    #[test]
    fn discover_skips_plugin_with_spaces_in_name() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        make_executable_in(&plugins_dir, "dm-tool-bad name");
        let plugins = discover_plugins(config_dir.path());
        assert!(
            plugins.is_empty(),
            "plugin with spaces in name should be skipped"
        );
    }

    #[test]
    fn discover_allows_hyphen_underscore_names() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        make_executable_in(&plugins_dir, "dm-tool-my-plugin_v2");
        let plugins = discover_plugins(config_dir.path());
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "my-plugin_v2");
    }

    #[test]
    fn discover_skips_empty_name() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        make_executable_in(&plugins_dir, "dm-tool-");
        let plugins = discover_plugins(config_dir.path());
        assert!(
            plugins.is_empty(),
            "plugin with empty name should be skipped"
        );
    }

    #[test]
    fn discover_plugin_path_contains_plugin_file() {
        let config_dir = TempDir::new().unwrap();
        let plugins_dir = config_dir.path().join("plugins");
        std::fs::create_dir(&plugins_dir).unwrap();

        let p = plugins_dir.join("dm-tool-myplugin");
        std::fs::write(&p, b"#!/bin/sh").unwrap();
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&p, perms).unwrap();

        let plugins = discover_plugins(config_dir.path());
        assert_eq!(plugins.len(), 1);
        assert!(
            plugins[0].path.ends_with("dm-tool-myplugin"),
            "plugin path should end with plugin filename: {:?}",
            plugins[0].path
        );
    }
}
