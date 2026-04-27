//! Integration test for kotoba's host capabilities.
//!
//! Verifies that:
//! - The 5 tools install cleanly via `install_host_capabilities`
//! - All carry the `host_` prefix (registry would reject otherwise)
//! - Each tool has a non-empty description and a parameters schema
//!
//! End-to-end wiki round-trips (calling a tool, reading the resulting
//! page) live in tool-level unit tests inside `src/host_caps.rs`.
//! This test is the cross-binary contract check.

use dark_matter::host::HostCapabilities;

// `host_caps` is a binary-only module (`src/host_caps.rs`); for the test we
// reach into it through the kotoba bin's module path, which cargo exposes via
// the `path` attribute below. The binary still owns the canonical copy.
#[path = "../src/host_caps.rs"]
mod host_caps;

#[path = "../src/domain.rs"]
#[allow(dead_code)]
mod domain;

#[test]
fn capabilities_register_five_tools_with_host_prefix() {
    let caps = host_caps::KotobaCapabilities;
    let tools = caps.tools();
    assert_eq!(
        tools.len(),
        5,
        "expected exactly 5 host tools, got {}",
        tools.len()
    );

    let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    let expected = [
        "host_invoke_persona",
        "host_log_vocabulary",
        "host_log_kanji",
        "host_record_struggle",
        "host_quiz_me",
    ];
    for name in expected {
        assert!(
            names.contains(&name),
            "missing expected host tool: {} (got: {:?})",
            name,
            names
        );
    }
}

#[test]
fn every_tool_has_description_and_parameters() {
    let caps = host_caps::KotobaCapabilities;
    for tool in caps.tools() {
        assert!(
            !tool.description().is_empty(),
            "tool {} has empty description",
            tool.name()
        );
        let params = tool.parameters();
        assert!(
            params.is_object(),
            "tool {} parameters must be a JSON object",
            tool.name()
        );
        assert_eq!(
            params.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "tool {} parameters.type must be 'object'",
            tool.name()
        );
    }
}

#[test]
fn slugify_handles_japanese_and_ascii() {
    assert_eq!(host_caps::slugify("学校"), "学校");
    assert_eq!(host_caps::slugify("ありがとう"), "ありがとう");
    assert_eq!(host_caps::slugify("hello world"), "hello-world");
    assert_eq!(host_caps::slugify(""), "untitled");
}
