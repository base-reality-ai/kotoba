//! Automated code translation.
//!
//! Implements the `dm translate` command to port source files from
//! one programming language to another using an LLM.

use crate::ollama::client::OllamaClient;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Return the file extension (with dot) for a given target language name.
/// Unknown languages return ".txt".
pub fn target_extension(lang: &str) -> &'static str {
    match lang.to_lowercase().as_str() {
        "python" => ".py",
        "go" => ".go",
        "typescript" => ".ts",
        "java" => ".java",
        "c" => ".c",
        "cpp" | "c++" => ".cpp",
        "rust" => ".rs",
        _ => ".txt",
    }
}

/// Compute the output path for a translation.
/// If `override_path` is Some, return it directly.
/// Otherwise, replace the source extension with the target language extension.
pub fn output_path(src: &Path, lang: &str, override_path: Option<&Path>) -> PathBuf {
    if let Some(p) = override_path {
        return p.to_path_buf();
    }
    let ext = target_extension(lang);
    // Strip leading dot from ext for with_extension
    let ext_no_dot = ext.trim_start_matches('.');
    src.with_extension(ext_no_dot)
}

/// Detect source language name from file extension.
fn source_lang_from_extension(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "Rust",
        Some("py") => "Python",
        Some("js") => "JavaScript",
        Some("ts") => "TypeScript",
        Some("go") => "Go",
        Some("java") => "Java",
        Some("c") => "C",
        Some("cpp") | Some("cc") | Some("cxx") => "C++",
        _ => "unknown",
    }
}

/// Translate a source file to another programming language.
pub async fn run_translate(
    src_path: &Path,
    target_lang: &str,
    out_path: &Path,
    client: &OllamaClient,
) -> Result<()> {
    let source_contents = std::fs::read_to_string(src_path)
        .map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", src_path.display(), e))?;

    let source_lang = source_lang_from_extension(src_path);

    let prompt = format!(
        "Translate the following {source_lang} code to {target_lang}.\n\
         Preserve all logic, functionality, and structure. Use idiomatic {target_lang} style.\n\
         Return only the translated code with no explanation.\n\n\
         {source_contents}"
    );

    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let response = client.chat(&messages, &[]).await?;
    let response_text = response.message.content.trim().to_string();

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow::anyhow!("Cannot create directory '{}': {}", parent.display(), e)
            })?;
        }
    }

    std::fs::write(out_path, &response_text)
        .map_err(|e| anyhow::anyhow!("Cannot write '{}': {}", out_path.display(), e))?;

    let n_lines = response_text.lines().count();
    println!(
        "Translated {} → {} ({} lines)",
        src_path.display(),
        out_path.display(),
        n_lines
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_path_replaces_extension() {
        let p = output_path(Path::new("src/foo.rs"), "python", None);
        assert_eq!(p, Path::new("src/foo.py"));
    }

    #[test]
    fn output_path_respects_override() {
        let p = output_path(Path::new("src/foo.rs"), "python", Some(Path::new("bar.py")));
        assert_eq!(p, Path::new("bar.py"));
    }

    #[test]
    fn target_extension_python() {
        assert_eq!(target_extension("python"), ".py");
    }

    #[test]
    fn target_extension_go() {
        assert_eq!(target_extension("go"), ".go");
    }

    #[test]
    fn target_extension_unknown_falls_back() {
        assert_eq!(target_extension("brainfuck"), ".txt");
    }

    #[test]
    fn target_extension_rust() {
        assert_eq!(target_extension("rust"), ".rs");
    }

    #[test]
    fn target_extension_cpp_both_spellings() {
        assert_eq!(target_extension("cpp"), ".cpp");
        assert_eq!(target_extension("c++"), ".cpp");
    }

    #[test]
    fn target_extension_case_insensitive() {
        assert_eq!(target_extension("PYTHON"), ".py");
        assert_eq!(target_extension("Go"), ".go");
        assert_eq!(target_extension("TypeScript"), ".ts");
    }

    #[test]
    fn output_path_unknown_target_lang_uses_txt() {
        let p = output_path(Path::new("src/main.rs"), "lolcode", None);
        assert_eq!(p, Path::new("src/main.txt"));
    }
}
