//! Automated code documentation generation.
//!
//! Uses LLMs to generate language-appropriate docstrings (Rust, Python, JSDoc)
//! for source files lacking documentation.

use crate::ollama::client::OllamaClient;
use anyhow::Result;

/// Detect documentation style from file extension.
/// Returns "rust", "python", "jsdoc", or "generic".
pub fn detect_style(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("js") | Some("ts") | Some("jsx") | Some("tsx") => "jsdoc",
        _ => "generic",
    }
}

/// Generate/update documentation comments for all public items in a file.
pub async fn run_document(
    path: &std::path::Path,
    style: &str,
    client: &OllamaClient,
) -> Result<()> {
    let file_contents = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path.display(), e))?;

    let resolved_style = if style == "auto" {
        detect_style(path)
    } else {
        style
    };

    let prompt = format!(
        "Add or update documentation comments for every public function, struct, enum, and trait in this file.\n\
         Use {resolved_style} doc comment style. Do not change any logic. Return the complete file with docs added.\n\n\
         {file_contents}"
    );

    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let response = client.chat(&messages, &[]).await?;
    let response_text = response.message.content.trim().to_string();

    if response_text.is_empty() || response_text == file_contents.trim() {
        println!("Documented {} (no changes)", path.display());
        return Ok(());
    }

    // Atomic write: write to tmp file then rename
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, &response_text)
        .map_err(|e| anyhow::anyhow!("Cannot write tmp file '{}': {}", tmp_path.display(), e))?;
    std::fs::rename(&tmp_path, path)
        .map_err(|e| anyhow::anyhow!("Cannot rename tmp to '{}': {}", path.display(), e))?;

    let n_lines = response_text.lines().count();
    println!("Documented {} ({} lines)", path.display(), n_lines);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_style_from_extension_rs() {
        assert_eq!(detect_style(std::path::Path::new("foo.rs")), "rust");
    }

    #[test]
    fn detect_style_from_extension_py() {
        assert_eq!(detect_style(std::path::Path::new("foo.py")), "python");
    }

    #[test]
    fn detect_style_unknown_falls_back_to_generic() {
        assert_eq!(detect_style(std::path::Path::new("foo.xyz")), "generic");
    }

    #[test]
    fn detect_style_js_returns_jsdoc() {
        assert_eq!(detect_style(std::path::Path::new("foo.js")), "jsdoc");
    }

    #[test]
    fn detect_style_ts_returns_jsdoc() {
        assert_eq!(detect_style(std::path::Path::new("foo.ts")), "jsdoc");
    }
}
