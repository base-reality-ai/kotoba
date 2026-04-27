use crate::orchestrate::types::ChainConfig;

const CONTINUOUS_DEV_YAML: &str = include_str!("continuous-dev.yaml");
const SELF_IMPROVE_YAML: &str = include_str!("self-improve.yaml");
const PROJECT_AUDIT_YAML: &str = include_str!("project-audit.yaml");

const PRESETS: &[(&str, &str)] = &[
    ("continuous-dev", CONTINUOUS_DEV_YAML),
    ("self-improve", SELF_IMPROVE_YAML),
    ("project-audit", PROJECT_AUDIT_YAML),
];

/// Resolve a built-in chain preset by name.
///
/// Returns `Some(ChainConfig)` when `name` matches a registered preset's
/// bareword key (e.g. `"continuous-dev"`).
///
/// Precedence: preset name takes precedence over a same-named file in the
/// current directory. To disambiguate toward a file, prefix it with `./`
/// (e.g. `./continuous-dev`) — the dispatch site treats anything containing
/// a path separator as a file path and skips preset resolution.
pub fn resolve_chain_preset(name: &str) -> Option<ChainConfig> {
    let yaml = PRESETS.iter().find(|(n, _)| *n == name).map(|(_, y)| *y)?;
    serde_yaml::from_str::<ChainConfig>(yaml).ok()
}

/// List all registered built-in preset names.
pub fn list_chain_presets() -> Vec<&'static str> {
    PRESETS.iter().map(|(n, _)| *n).collect()
}

#[cfg(test)]
macro_rules! preset_shape_tests {
    (
        preset: $preset:expr,
        module: $mod_ident:ident,
        node_names: $names:expr,
        node_roles: $roles:expr,
        loop_forever: $loop_forever:expr,
        max_cycles: $max_cycles:expr,
    ) => {
        mod $mod_ident {
            use super::*;

            #[test]
            fn resolves() {
                let cfg = resolve_chain_preset($preset).expect(concat!($preset, " should resolve"));
                assert_eq!(cfg.name, $preset);
            }

            #[test]
            fn has_expected_node_names_and_roles() {
                let cfg = resolve_chain_preset($preset).unwrap();
                let names: Vec<&str> = cfg.nodes.iter().map(|n| n.name.as_str()).collect();
                let roles: Vec<&str> = cfg.nodes.iter().map(|n| n.role.as_str()).collect();
                let expected_names: Vec<&str> = $names.iter().copied().collect();
                let expected_roles: Vec<&str> = $roles.iter().copied().collect();
                assert_eq!(names, expected_names);
                assert_eq!(roles, expected_roles);
            }

            #[test]
            fn loop_flag_matches_expected() {
                let cfg = resolve_chain_preset($preset).unwrap();
                assert_eq!(cfg.loop_forever, $loop_forever);
            }

            #[test]
            fn max_cycles_matches_expected() {
                let cfg = resolve_chain_preset($preset).unwrap();
                assert_eq!(cfg.max_cycles, $max_cycles);
            }

            #[test]
            fn input_wiring_is_linear_dag() {
                let cfg = resolve_chain_preset($preset).unwrap();
                assert!(cfg.nodes[0].input_from.is_none(), "entry node has no input");
                for i in 1..cfg.nodes.len() {
                    let prev = cfg.nodes[i - 1].name.as_str();
                    assert_eq!(
                        cfg.nodes[i].input_from.as_deref(),
                        Some(prev),
                        "node {} should read from {}",
                        cfg.nodes[i].name,
                        prev
                    );
                }
            }

            #[test]
            fn workspace_is_dm_workspace() {
                let cfg = resolve_chain_preset($preset).unwrap();
                assert!(cfg.workspace.is_relative());
                assert_eq!(cfg.workspace, std::path::PathBuf::from(".dm-workspace"));
            }

            #[test]
            fn passes_validate_chain_config() {
                let cfg = resolve_chain_preset($preset).unwrap();
                let _ = crate::orchestrate::validate_chain_config(&cfg)
                    .expect("preset should validate");
            }

            #[test]
            fn listed_in_preset_registry() {
                assert!(list_chain_presets().contains(&$preset));
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    preset_shape_tests! {
        preset: "continuous-dev",
        module: continuous_dev,
        node_names: ["planner", "builder", "tester"],
        node_roles: ["planner", "builder", "tester"],
        loop_forever: true,
        max_cycles: 100,
    }

    preset_shape_tests! {
        preset: "self-improve",
        module: self_improve,
        node_names: ["planner", "builder", "tester"],
        node_roles: ["planner", "builder", "tester"],
        loop_forever: true,
        max_cycles: 1000,
    }

    preset_shape_tests! {
        preset: "project-audit",
        module: project_audit,
        node_names: ["reader", "linter", "reporter"],
        node_roles: ["planner", "builder", "tester"],
        loop_forever: false,
        max_cycles: 1,
    }

    // ── Retained free tests ────────────────────────────────────────────────

    #[test]
    fn unknown_preset_returns_none() {
        assert!(resolve_chain_preset("does-not-exist").is_none());
        assert!(resolve_chain_preset("").is_none());
        assert!(resolve_chain_preset("./continuous-dev").is_none());
    }

    #[test]
    fn project_audit_preset_is_single_pass() {
        let cfg = resolve_chain_preset("project-audit").unwrap();
        assert!(
            !cfg.loop_forever && cfg.max_cycles == 1,
            "project-audit must be single-pass: !loop_forever && max_cycles == 1"
        );
    }

    // ── Cross-preset invariants ────────────────────────────────────────────

    #[test]
    fn project_audit_roles_decoupled_from_names() {
        let cfg = resolve_chain_preset("project-audit").unwrap();
        for node in &cfg.nodes {
            assert_ne!(
                node.name, node.role,
                "project-audit node '{}' has name == role; roles must stay \
                 planner/builder/tester so role-gated injection still fires",
                node.name
            );
        }
    }

    #[test]
    fn all_preset_names_are_unique() {
        let names = list_chain_presets();
        let set: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            set.len(),
            names.len(),
            "duplicate preset names in registry: {:?}",
            names
        );
    }

    #[test]
    fn all_preset_descriptions_are_unique() {
        let descs: Vec<String> = list_chain_presets()
            .iter()
            .map(|n| {
                resolve_chain_preset(n)
                    .and_then(|c| c.description)
                    .unwrap_or_default()
            })
            .collect();
        let set: HashSet<&str> = descs.iter().map(String::as_str).collect();
        assert_eq!(
            set.len(),
            descs.len(),
            "duplicate preset descriptions — copy-paste canary"
        );
    }

    #[test]
    fn preset_table_has_three_entries() {
        assert_eq!(
            list_chain_presets().len(),
            3,
            "preset registry count changed — update this test and add tests for the new preset"
        );
    }

    #[test]
    fn every_preset_description_is_non_empty() {
        for name in list_chain_presets() {
            let cfg = resolve_chain_preset(name).unwrap();
            let desc = cfg.description.unwrap_or_default();
            assert!(
                !desc.trim().is_empty(),
                "preset '{}' has empty description",
                name
            );
        }
    }

    #[test]
    fn every_preset_has_three_nodes() {
        for name in list_chain_presets() {
            let cfg = resolve_chain_preset(name).unwrap();
            assert_eq!(
                cfg.nodes.len(),
                3,
                "preset '{}' has {} nodes, expected 3",
                name,
                cfg.nodes.len()
            );
        }
    }
}
