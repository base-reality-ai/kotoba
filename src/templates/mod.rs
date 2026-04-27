//! Custom user prompt templates.
//!
//! Scans and manages markdown template files stored in `~/.dm/templates/`
//! for reuse in standard chain or agent runs.

use std::path::Path;

pub struct TemplateInfo {
    pub name: String,
    pub description: String,
}

/// Scan `config_dir/templates`/*.md and return info for each template.
pub fn list_templates(config_dir: &Path) -> Vec<TemplateInfo> {
    let templates_dir = config_dir.join("templates");
    let pattern = templates_dir.join("*.md");
    let pattern_str = pattern.to_string_lossy().to_string();

    let mut results = Vec::new();
    for entry in glob::glob(&pattern_str).into_iter().flatten().flatten() {
        let name = entry
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let description = std::fs::read_to_string(&entry)
            .ok()
            .and_then(|content| parse_frontmatter_description(&content))
            .unwrap_or_default();
        results.push(TemplateInfo { name, description });
    }
    results
}

/// Load a template by name, strip frontmatter, substitute args.
/// Returns Err if the file doesn't exist.
pub fn load_template(config_dir: &Path, name: &str, args: &[String]) -> anyhow::Result<String> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        anyhow::bail!(
            "Invalid template name '{}': must contain only alphanumeric, dash, or underscore characters",
            name
        );
    }

    let path = config_dir.join("templates").join(format!("{}.md", name));
    if !path.exists() {
        anyhow::bail!("Template '{}' not found at {}", name, path.display());
    }
    let content = std::fs::read_to_string(&path)?;

    // Parse named args from frontmatter before stripping it
    let named_args = parse_frontmatter_named_args(&content);

    // Strip frontmatter
    let body = strip_frontmatter(&content);

    // Substitute positional {{0}}, {{1}}, ... and {{ARG_N}}
    let mut result = body;
    for (i, arg_val) in args.iter().enumerate() {
        result = result.replace(&format!("{{{{{}}}}}", i), arg_val);
    }

    // Substitute named args: {{NAME}} -> args[index]
    for (i, arg_name) in named_args.iter().enumerate() {
        if let Some(val) = args.get(i) {
            result = result.replace(&format!("{{{{{}}}}}", arg_name), val);
        }
    }

    // Warn on unresolved placeholders
    let mut unresolved = Vec::new();
    let mut search_start = 0;
    while let Some(open) = result[search_start..].find("{{") {
        let abs_open = search_start + open;
        if let Some(close) = result[abs_open..].find("}}") {
            let placeholder = &result[abs_open..abs_open + close + 2];
            unresolved.push(placeholder.to_string());
            search_start = abs_open + close + 2;
        } else {
            break;
        }
    }
    if !unresolved.is_empty() {
        crate::warnings::push_warning(format!(
            "template '{}' has unresolved placeholders: {}",
            name,
            unresolved.join(", ")
        ));
    }

    Ok(result)
}

/// Strip YAML frontmatter (between first --- and next ---) from content.
fn strip_frontmatter(content: &str) -> String {
    if !content.starts_with("---") {
        return content.to_string();
    }
    // Find the closing ---
    let after_first = &content[3..];
    if let Some(end_pos) = after_first.find("\n---") {
        // end_pos is relative to after_first; +3 for "---", +4 for "\n---"
        let body_start = 3 + end_pos + 4; // skip past "\n---"
                                          // skip the newline after closing ---
        let body = if content.len() > body_start && content.as_bytes()[body_start] == b'\n' {
            &content[body_start + 1..]
        } else {
            &content[body_start..]
        };
        body.to_string()
    } else {
        content.to_string()
    }
}

/// Parse description from YAML frontmatter.
fn parse_frontmatter_description(content: &str) -> Option<String> {
    let fm = extract_frontmatter(content)?;
    for line in fm.lines() {
        if let Some(rest) = line.strip_prefix("description:") {
            return Some(rest.trim().trim_matches('"').trim_matches('\'').to_string());
        }
    }
    None
}

/// Parse named args list from YAML frontmatter.
fn parse_frontmatter_named_args(content: &str) -> Vec<String> {
    let Some(fm) = extract_frontmatter(content) else {
        return Vec::new();
    };

    // Look for "args: [NAME1, NAME2]" style
    for line in fm.lines() {
        if let Some(rest) = line.strip_prefix("args:") {
            let rest = rest.trim();
            if rest.starts_with('[') && rest.ends_with(']') {
                let inner = &rest[1..rest.len() - 1];
                return inner
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    Vec::new()
}

/// Extract the raw YAML frontmatter string (between first and second ---).
fn extract_frontmatter(content: &str) -> Option<String> {
    if !content.starts_with("---") {
        return None;
    }
    let after_first = &content[3..];
    let end_pos = after_first.find("\n---")?;
    let fm = after_first[..end_pos].trim().to_string();
    Some(fm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_tempdir() -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "dm_templates_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }

    #[test]
    fn load_template_substitutes_args() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(
            tmpl_dir.join("greet.md"),
            "---\ndescription: Greet\nargs: [NAME]\n---\nHello, {{NAME}}!\n",
        )
        .unwrap();

        let result = load_template(&dir, "greet", &["World".to_string()]).unwrap();
        assert!(result.contains("Hello, World!"), "got: {}", result);
        assert!(!result.contains("---"), "frontmatter should be stripped");
    }

    #[test]
    fn load_template_strips_frontmatter() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(
            tmpl_dir.join("simple.md"),
            "---\ndescription: Simple template\n---\nBody content here.\n",
        )
        .unwrap();

        let result = load_template(&dir, "simple", &[]).unwrap();
        assert!(result.contains("Body content here."));
        assert!(!result.contains("description:"));
    }

    #[test]
    fn list_templates_finds_md_files() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(
            tmpl_dir.join("alpha.md"),
            "---\ndescription: Alpha\n---\nContent A\n",
        )
        .unwrap();
        fs::write(
            tmpl_dir.join("beta.md"),
            "---\ndescription: Beta\n---\nContent B\n",
        )
        .unwrap();

        let templates = list_templates(&dir);
        let names: Vec<&str> = templates.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"alpha"),
            "should find alpha, got: {:?}",
            names
        );
        assert!(
            names.contains(&"beta"),
            "should find beta, got: {:?}",
            names
        );
    }

    #[test]
    fn load_template_returns_err_if_missing() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        let result = load_template(&dir, "nonexistent", &[]);
        assert!(result.is_err());
    }

    // ── strip_frontmatter ─────────────────────────────────────────────────────

    #[test]
    fn strip_frontmatter_removes_yaml_block() {
        let content = "---\ntitle: Test\n---\nBody content here.\n";
        let result = strip_frontmatter(content);
        assert_eq!(result, "Body content here.\n");
        assert!(
            !result.contains("title:"),
            "frontmatter not stripped: {result}"
        );
    }

    #[test]
    fn strip_frontmatter_passthrough_when_no_frontmatter() {
        let content = "Just a plain file\nwith no frontmatter.\n";
        assert_eq!(strip_frontmatter(content), content);
    }

    #[test]
    fn strip_frontmatter_passthrough_when_unclosed() {
        // Opening --- but no closing --- → treat as no frontmatter, return as-is
        let content = "---\ntitle: Unclosed\nNo closing delimiter\n";
        let result = strip_frontmatter(content);
        assert_eq!(
            result, content,
            "unclosed frontmatter should be left intact"
        );
    }

    #[test]
    fn strip_frontmatter_preserves_trailing_newline() {
        let content = "---\ndesc: x\n---\nLine A\nLine B\n";
        let result = strip_frontmatter(content);
        assert!(
            result.ends_with('\n'),
            "trailing newline should be preserved"
        );
        assert_eq!(result, "Line A\nLine B\n");
    }

    // ── parse_frontmatter_description ─────────────────────────────────────────

    #[test]
    fn parse_desc_extracts_unquoted_value() {
        let content = "---\ndescription: My template\n---\nbody";
        let desc = parse_frontmatter_description(content);
        assert_eq!(desc.as_deref(), Some("My template"));
    }

    #[test]
    fn parse_desc_strips_double_quotes() {
        let content = "---\ndescription: \"Quoted value\"\n---\nbody";
        let desc = parse_frontmatter_description(content);
        assert_eq!(desc.as_deref(), Some("Quoted value"));
    }

    #[test]
    fn parse_desc_returns_none_when_absent() {
        let content = "---\ntitle: something\n---\nbody";
        assert!(parse_frontmatter_description(content).is_none());
    }

    #[test]
    fn parse_desc_returns_none_with_no_frontmatter() {
        let content = "No frontmatter here.";
        assert!(parse_frontmatter_description(content).is_none());
    }

    // ── parse_frontmatter_named_args ──────────────────────────────────────────

    #[test]
    fn parse_named_args_single() {
        let content = "---\nargs: [NAME]\n---\nHello {{NAME}}";
        let args = parse_frontmatter_named_args(content);
        assert_eq!(args, vec!["NAME"]);
    }

    #[test]
    fn parse_named_args_multiple() {
        let content = "---\nargs: [FIRST, LAST, ROLE]\n---\nbody";
        let args = parse_frontmatter_named_args(content);
        assert_eq!(args, vec!["FIRST", "LAST", "ROLE"]);
    }

    #[test]
    fn parse_named_args_empty_when_no_args_field() {
        let content = "---\ndescription: No args here\n---\nbody";
        let args = parse_frontmatter_named_args(content);
        assert!(args.is_empty());
    }

    #[test]
    fn parse_named_args_empty_when_no_frontmatter() {
        let content = "No frontmatter.";
        let args = parse_frontmatter_named_args(content);
        assert!(args.is_empty());
    }

    // ── extract_frontmatter ───────────────────────────────────────────────────

    #[test]
    fn extract_frontmatter_returns_raw_yaml() {
        let content = "---\ntitle: T\ndesc: D\n---\nbody";
        let fm = extract_frontmatter(content).unwrap();
        assert!(fm.contains("title: T"), "got: {fm}");
        assert!(fm.contains("desc: D"), "got: {fm}");
    }

    #[test]
    fn extract_frontmatter_returns_none_without_dashes() {
        let content = "Just content\nno delimiters";
        assert!(extract_frontmatter(content).is_none());
    }

    // ── unresolved placeholder warning ───────────────────────────────────────

    #[test]
    fn load_template_preserves_unresolved_placeholders() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(
            tmpl_dir.join("two_args.md"),
            "---\nargs: [NAME, ROLE]\n---\nHello {{NAME}}, you are {{ROLE}}.\n",
        )
        .unwrap();

        let result = load_template(&dir, "two_args", &["Alice".to_string()]).unwrap();
        assert!(
            result.contains("Hello Alice"),
            "NAME should be substituted: {result}"
        );
        assert!(
            result.contains("{{ROLE}}"),
            "ROLE should remain unresolved: {result}"
        );
    }

    #[test]
    fn load_template_rejects_path_traversal() {
        let dir = make_tempdir();
        fs::create_dir_all(dir.join("templates")).unwrap();
        let result = load_template(&dir, "../../../etc/passwd", &[]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid template name"), "err: {err}");
    }

    #[test]
    fn load_template_rejects_slash_in_name() {
        let dir = make_tempdir();
        fs::create_dir_all(dir.join("templates")).unwrap();
        let result = load_template(&dir, "sub/template", &[]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid template name"));
    }

    #[test]
    fn load_template_rejects_dot_dot() {
        let dir = make_tempdir();
        fs::create_dir_all(dir.join("templates")).unwrap();
        let result = load_template(&dir, "..", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn load_template_accepts_valid_name_with_dashes() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(tmpl_dir.join("code-review.md"), "Review this code.\n").unwrap();
        let result = load_template(&dir, "code-review", &[]);
        assert!(
            result.is_ok(),
            "dashed name should be valid: {:?}",
            result.err()
        );
    }

    #[test]
    fn load_template_no_unresolved_when_all_args_provided() {
        let dir = make_tempdir();
        let tmpl_dir = dir.join("templates");
        fs::create_dir_all(&tmpl_dir).unwrap();
        fs::write(
            tmpl_dir.join("full.md"),
            "---\nargs: [NAME]\n---\nHello {{NAME}}!\n",
        )
        .unwrap();

        let result = load_template(&dir, "full", &["Bob".to_string()]).unwrap();
        assert!(
            result.contains("Hello Bob!"),
            "NAME should be substituted: {result}"
        );
        assert!(
            !result.contains("{{"),
            "no unresolved placeholders should remain: {result}"
        );
    }
}
