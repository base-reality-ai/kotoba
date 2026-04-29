//! Acceptance test for Tier 6 — sub-agent spawn from a host process.
//!
//! Canonical `tools::registry::sub_agent_registry` (registry.rs:378) builds
//! the registry for sub-agents that `dark_matter::tools::agent::AgentTool`
//! invokes. The identical capability-merge pipeline (registry.rs:495-508)
//! consults `dark_matter::host::installed_host_capabilities()` for both
//! `default_registry` and `sub_agent_registry`. So when kotoba's
//! `KotobaCapabilities` is installed at startup, sub-agents spawned by the
//! kotoba process inherit ALL seven host_* tools.
//!
//! `tests/host_capabilities.rs` already pins this with a `host_probe` stub.
//! This test pins the kotoba-specific contract:
//!
//! 1. All seven actual kotoba host capabilities appear in the sub-agent
//!    registry by name (host_invoke_persona, host_log_vocabulary,
//!    host_log_kanji, host_record_struggle, host_quiz_me, host_plan_session,
//!    host_record_session). If a future refactor adds, removes, or renames
//!    a capability without updating tests, this fails loudly.
//! 2. Wiki tools (`wiki_search`, `wiki_lookup`) are present too — the
//!    second-brain recall path Tier 3 wired must remain reachable from a
//!    sub-agent.
//! 3. `agent` tool is NOT in the sub-agent registry — canonical's
//!    `include_agent: false` guards against unbounded sub-agent nesting.
//! 4. `system_prompt_hints_for(host_identity)` produces the host-aware
//!    grouping with `## Host capabilities` followed by `## Substrate
//!    (kernel tools)`. This proves a sub-agent's system prompt will steer
//!    the model toward kotoba's host caps first.
//!
//! Lives in its own integration-test binary because `install_host_capabilities`
//! writes to a process-global `OnceLock` — same isolation pattern as
//! `tests/host_capabilities.rs`.

use dark_matter::host::install_host_capabilities;
use dark_matter::identity;
use dark_matter::tools::registry::sub_agent_registry;

#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

const KOTOBA_HOST_CAP_NAMES: &[&str] = &[
    "host_invoke_persona",
    "host_log_vocabulary",
    "host_log_kanji",
    "host_record_struggle",
    "host_quiz_me",
    "host_plan_session",
    "host_record_session",
];

fn write_host_identity(project_root: &std::path::Path, name: &str) {
    let dm = project_root.join(".dm");
    std::fs::create_dir_all(&dm).expect("create .dm");
    std::fs::write(
        dm.join("identity.toml"),
        format!("mode = \"host\"\nhost_project = \"{}\"\n", name),
    )
    .expect("write identity.toml");
}

#[tokio::test]
async fn sub_agent_registry_in_kotoba_process_sees_all_host_caps_and_routes_hints_host_first() {
    let project = tempfile::TempDir::new().expect("project tempdir");
    write_host_identity(project.path(), "kotoba");

    // Install kotoba's full host capability set. This is what `kotoba_main`
    // does at process start. The OnceLock is process-global; this test runs
    // in its own integration-test binary so the install does not race other
    // suites.
    install_host_capabilities(Box::new(host_caps::KotobaCapabilities))
        .expect("install kotoba caps");

    let sub = sub_agent_registry(
        "kotoba-sub-agent",
        project.path(),
        "http://localhost:11434",
        "claude-opus",
        "mxbai-embed-large",
    );
    let names = sub.tool_names();

    // 1. All seven kotoba host caps are visible to sub-agents.
    for cap in KOTOBA_HOST_CAP_NAMES {
        assert!(
            names.contains(cap),
            "sub_agent_registry must include kotoba host cap '{}'; got: {:?}",
            cap,
            names
        );
    }

    // 2. Wiki tools (the recall path from Tier 3) are also reachable.
    assert!(
        names.contains(&"wiki_search"),
        "sub_agent_registry must include wiki_search; got: {:?}",
        names
    );
    assert!(
        names.contains(&"wiki_lookup"),
        "sub_agent_registry must include wiki_lookup; got: {:?}",
        names
    );

    // 3. The `agent` tool is excluded from sub-agent registries by canonical
    // design (registry.rs:391 `include_agent: false`) so a sub-agent cannot
    // recursively spawn its own sub-agents.
    assert!(
        !names.contains(&"agent"),
        "sub_agent_registry must NOT include `agent` (no nested spawning); got: {:?}",
        names
    );

    // 4. Host-aware system_prompt_hints_for routes the kotoba caps to a
    // dedicated `## Host capabilities` section ahead of `## Substrate
    // (kernel tools)`. This is what the sub-agent's model sees as the
    // domain steer — without host-mode grouping, kotoba's domain tools
    // would render as just-another kernel tool and the persona would be
    // less likely to call them.
    let host_identity = identity::load_at(project.path()).expect("load host identity");
    assert!(
        host_identity.is_host(),
        "test setup error: identity.toml not picked up; got: {:?}",
        host_identity
    );

    let hints = sub.system_prompt_hints_for(&host_identity);
    let host_section = hints
        .find("## Host capabilities")
        .expect("hints should include `## Host capabilities` in host mode");
    let kernel_section = hints
        .find("## Substrate (kernel tools)")
        .expect("hints should include `## Substrate (kernel tools)` in host mode");
    assert!(
        host_section < kernel_section,
        "host capabilities should appear before kernel substrate in host-mode \
         hints; got hints:\n{}",
        hints
    );

    // Each kotoba host cap renders its `### <name>` heading in the host
    // section. The model uses these to know which host_* tool to reach for.
    for cap in KOTOBA_HOST_CAP_NAMES {
        let heading = format!("### {}", cap);
        assert!(
            hints.contains(&heading),
            "system prompt hints should contain heading `{}`; got hints:\n{}",
            heading,
            hints
        );
    }
}
