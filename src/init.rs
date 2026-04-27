//! Implements `dm --init` for preparing user and project state.
//!
//! Creates missing global config files, starter templates, plugin/eval
//! directories, project `DM.md`, `.dmignore`, and the local `.dm/wiki/` while
//! preserving existing operator-owned files.

use crate::config::Config;
use anyhow::Context;
use std::fmt::Write as _;
use std::path::Path;

const PROJECT_GITIGNORE_ENTRIES: &[&str] = &[".dm/", ".dm-workspace/"];

pub async fn run_init(config: &Config) -> anyhow::Result<()> {
    let cwd = std::env::current_dir()?;
    println!("Initializing dm in: {}", cwd.display());
    println!();

    // 1. Ensure ~/.dm/ exists
    std::fs::create_dir_all(&config.config_dir).with_context(|| {
        format!(
            "Failed to create config directory: {}. Try: check write permissions on the parent directory or set DM_HOME to a writable path.",
            config.config_dir.display(),
        )
    })?;
    println!("✓ Config directory: {}", config.config_dir.display());

    // 2. Create settings.json if missing
    let settings_path = config.config_dir.join("settings.json");
    if !settings_path.exists() {
        let default_settings = serde_json::json!({
            "model": config.model,
            "host": config.host
        });
        std::fs::write(
            &settings_path,
            serde_json::to_string_pretty(&default_settings)?,
        )
        .with_context(|| {
            format!(
                "Failed to write settings file: {}. Try: check write permissions in the config directory or set DM_HOME to a writable path.",
                settings_path.display(),
            )
        })?;
        println!("✓ Created: {}", settings_path.display());
    } else {
        println!("  (exists) {}", settings_path.display());
    }

    // 2.5 Live Ollama probe — shows the user whether their daemon is up
    // and whether the configured model is installed, before we move on.
    let probe_client =
        crate::ollama::client::OllamaClient::new(config.ollama_base_url(), config.model.clone());
    let status = crate::ollama::health::probe(
        &probe_client,
        crate::ollama::health::HEALTH_PROBE_TIMEOUT,
        &config.model,
    )
    .await;
    println!(
        "{}",
        crate::ollama::health::render_ollama_health(&status, &config.host, &config.model),
    );

    // 3. Create config.toml if missing
    let config_toml_path = config.config_dir.join("config.toml");
    if !config_toml_path.exists() {
        let template = "\
# dm configuration
#
# Model aliases — short names for Ollama models
# Usage: dm --model fast
# [aliases]
# fast = \"llama3.2:3b\"
# smart = \"gemma4:26b-128k\"
# code = \"qwen2.5-coder:32b\"

# Prompt-based routing — auto-select models by task type
# [routing]
# quick = \"llama3.2:3b\"       # short questions
# code = \"qwen2.5-coder:32b\"  # code review, diffs, commits
# explain = \"gemma4:26b\"      # explanations
# default = \"gemma4:26b\"      # fallback
";
        std::fs::write(&config_toml_path, template).with_context(|| {
            format!(
                "Failed to write config file: {}. Try: check write permissions in the config directory or set DM_HOME to a writable path.",
                config_toml_path.display(),
            )
        })?;
        println!("✓ Created: {}", config_toml_path.display());
    } else {
        println!("  (exists) {}", config_toml_path.display());
    }

    // 4. Create templates directory with example templates
    let templates_dir = config.config_dir.join("templates");
    if !templates_dir.exists() {
        std::fs::create_dir_all(&templates_dir).with_context(|| {
            format!(
                "Failed to create templates directory: {}. Try: check write permissions in the config directory or set DM_HOME to a writable path.",
                templates_dir.display(),
            )
        })?;
        println!("✓ Created: {}", templates_dir.display());
    } else {
        println!("  (exists) {}", templates_dir.display());
    }

    let code_review_path = templates_dir.join("code-review.md");
    if !code_review_path.exists() {
        std::fs::write(
            &code_review_path,
            "---\ndescription: Review code for quality and correctness\nargs: [FILE]\n---\n\
            Review the following file for code quality, bugs, and style issues: {{FILE}}\n\n\
            Use the FileRead tool to read the file, then provide specific, actionable feedback.\n",
        )
        .with_context(|| {
            format!(
                "Failed to write template file: {}. Try: check write permissions in the templates directory.",
                code_review_path.display(),
            )
        })?;
        println!("✓ Created: {}", code_review_path.display());
    }

    let explain_error_path = templates_dir.join("explain-error.md");
    if !explain_error_path.exists() {
        std::fs::write(
            &explain_error_path,
            "---\ndescription: Explain a compiler or runtime error\nargs: [ERROR]\n---\n\
            Explain this error in plain English and suggest how to fix it:\n\n{{ERROR}}\n",
        )
        .with_context(|| {
            format!(
                "Failed to write template file: {}. Try: check write permissions in the templates directory.",
                explain_error_path.display(),
            )
        })?;
        println!("✓ Created: {}", explain_error_path.display());
    }

    // 5. Create plugins directory
    let plugins_dir = config.config_dir.join("plugins");
    if !plugins_dir.exists() {
        std::fs::create_dir_all(&plugins_dir).with_context(|| {
            format!(
                "Failed to create plugins directory: {}. Try: check write permissions in the config directory or set DM_HOME to a writable path.",
                plugins_dir.display(),
            )
        })?;
        println!("✓ Created: {}", plugins_dir.display());
        println!("  Drop dm-tool-<name> executables here to add custom tools.");
    } else {
        println!("  (exists) {}", plugins_dir.display());
    }

    // 6. Create DM.md in cwd if missing
    let dm_md_path = cwd.join("DM.md");
    if !dm_md_path.exists() {
        let template = generate_dm_md_template(&cwd);
        std::fs::write(&dm_md_path, template).with_context(|| {
            format!(
                "Failed to write project instructions file: {}. Try: check write permissions in the current project directory.",
                dm_md_path.display(),
            )
        })?;
        println!("✓ Created: {}", dm_md_path.display());
    } else {
        println!("  (exists) {}", dm_md_path.display());
    }

    // 7. Create .dmignore if missing
    let dmignore_path = cwd.join(".dmignore");
    if !dmignore_path.exists() {
        let template = "# .dmignore — files to exclude from `dm index`\n\
            # One glob pattern per line; # for comments\n\
            \n\
            .dm-workspace/**\n\
            .dm/**\n\
            target/**\n\
            node_modules/**\n\
            *.min.js\n\
            dist/**\n";
        std::fs::write(&dmignore_path, template).with_context(|| {
            format!(
                "Failed to write ignore file: {}. Try: check write permissions in the current project directory.",
                dmignore_path.display(),
            )
        })?;
        println!("✓ Created: {}", dmignore_path.display());
    } else {
        println!("  (exists) {}", dmignore_path.display());
    }

    // 8. Create ~/.dm/evals/ and starter.yaml
    let evals_dir = config.config_dir.join("evals");
    if !evals_dir.exists() {
        std::fs::create_dir_all(&evals_dir).with_context(|| {
            format!(
                "Failed to create evals directory: {}. Try: check write permissions in the config directory or set DM_HOME to a writable path.",
                evals_dir.display(),
            )
        })?;
        println!("✓ Created: {}", evals_dir.display());
    } else {
        println!("  (exists) {}", evals_dir.display());
    }

    let starter_path = evals_dir.join("starter.yaml");
    if !starter_path.exists() {
        let starter = "name: starter\n\
model: gemma4:26b\n\
tools: []\n\
system: \"\"\n\
cases:\n\
  - id: code-gen\n\
    prompt: \"Write a Rust function that returns the factorial of a u64.\"\n\
    checks:\n\
      - contains: \"fn \"\n\
      - not_contains: \"TODO\"\n\
      - matches: \"fn \\\\w+\\\\(.*\\\\)\"\n\
      - min_length: 40\n\
      - max_length: 2000\n\
  - id: explanation\n\
    prompt: \"Explain what Rust's borrow checker does in 2-3 sentences.\"\n\
    checks:\n\
      - contains: \"borrow\"\n\
      - min_length: 30\n\
      - max_length: 1000\n\
  - id: bug-fix\n\
    prompt: \"Fix this Rust code so it compiles: fn main() { let x = 1; println!({}, x); }\"\n\
    checks:\n\
      - contains: \"{}\"\n\
      - not_contains: \"TODO\"\n\
      - min_length: 20\n";
        std::fs::write(&starter_path, starter).with_context(|| {
            format!(
                "Failed to write starter eval file: {}. Try: check write permissions in the evals directory.",
                starter_path.display(),
            )
        })?;
        println!("✓ Created: {}", starter_path.display());
    } else {
        println!("  (exists) {}", starter_path.display());
    }

    // 9. Scaffold .dm/wiki/ unconditionally. The wiki is how dm remembers
    //    your project across sessions — it's useful even outside a git
    //    repo. When .git exists, add dm state paths to .gitignore so the
    //    per-project state doesn't get committed.
    let dm_dir = cwd.join(".dm");
    let dm_existed = dm_dir.exists();
    crate::wiki::Wiki::open(&cwd).with_context(|| {
        format!(
            "Failed to initialize project wiki at {}. Try: check write permissions in the current project directory.",
            dm_dir.join("wiki").display(),
        )
    })?;
    if dm_existed {
        println!("  (exists) {}", dm_dir.display());
    } else {
        println!("✓ Created: {}", dm_dir.display());
    }

    if cwd.join(".git").exists() {
        let gitignore = cwd.join(".gitignore");
        let existing = if gitignore.exists() {
            std::fs::read_to_string(&gitignore).unwrap_or_default()
        } else {
            String::new()
        };
        if let Some(content) = append_missing_gitignore_entries(existing) {
            std::fs::write(&gitignore, content).with_context(|| {
                format!(
                    "Failed to update git ignore file: {}. Try: check write permissions in the current project directory.",
                    gitignore.display(),
                )
            })?;
            println!("✓ Added dm state paths to .gitignore");
        }
    }

    println!();
    println!("Ready! Start with:");
    println!("  dm                         # interactive TUI");
    println!("  dm \"explain this codebase\"  # one-shot question");

    Ok(())
}

fn append_missing_gitignore_entries(mut existing: String) -> Option<String> {
    let mut changed = false;

    for entry in PROJECT_GITIGNORE_ENTRIES {
        if existing.lines().any(|line| line.trim() == *entry) {
            continue;
        }
        if !existing.ends_with('\n') && !existing.is_empty() {
            existing.push('\n');
        }
        existing.push_str(entry);
        existing.push('\n');
        changed = true;
    }

    changed.then_some(existing)
}

pub fn generate_dm_md_template(cwd: &Path) -> String {
    let dir_name = cwd
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");

    if cwd.join("Cargo.toml").exists() {
        return generate_rust_template(cwd, dir_name);
    }
    if cwd.join("package.json").exists() {
        return generate_node_template(cwd, dir_name);
    }
    if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
        return generate_python_template(dir_name);
    }
    if cwd.join("go.mod").exists() {
        return generate_go_template(dir_name);
    }
    if cwd.join("pom.xml").exists() {
        return generate_java_template(dir_name, "maven");
    }
    if cwd.join("build.gradle").exists() || cwd.join("build.gradle.kts").exists() {
        return generate_java_template(dir_name, "gradle");
    }
    if cwd.join("CMakeLists.txt").exists() || cwd.join("Makefile").exists() {
        return generate_c_template(dir_name);
    }

    // Fallback: unknown project type
    format!(
        "# {}\n\n\
        Describe what this project does, its main purpose, and key technologies.\n\n\
        ## Coding conventions\n\n\
        - Describe style rules, naming conventions, formatting tools\n\
        - Example: \"Use 4-space indentation. Run prettier before committing.\"\n\n\
        ## Commands\n\n\
        - Build: `<build command>`\n\
        - Test: `<test command>`\n\
        - Run: `<run command>`\n\n\
        ## Important files\n\n\
        - List key files the assistant should know about\n",
        dir_name
    )
}

/// Read name and description from Cargo.toml.
fn read_cargo_metadata(cwd: &Path) -> (String, String) {
    let Ok(content) = std::fs::read_to_string(cwd.join("Cargo.toml")) else {
        return (String::new(), String::new());
    };
    let Ok(table) = content.parse::<toml::Table>() else {
        return (String::new(), String::new());
    };
    let Some(package) = table.get("package").and_then(|v| v.as_table()) else {
        return (String::new(), String::new());
    };
    let name = package
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let desc = package
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    (name, desc)
}

/// Read name, description, and script names from package.json.
fn read_package_json(cwd: &Path) -> (String, String, Vec<String>) {
    let Ok(content) = std::fs::read_to_string(cwd.join("package.json")) else {
        return (String::new(), String::new(), Vec::new());
    };
    let Ok(pkg) = serde_json::from_str::<serde_json::Value>(&content) else {
        return (String::new(), String::new(), Vec::new());
    };
    let name = pkg["name"].as_str().unwrap_or("").to_string();
    let desc = pkg["description"].as_str().unwrap_or("").to_string();
    let mut scripts = Vec::new();
    if let Some(s) = pkg["scripts"].as_object() {
        for key in s.keys() {
            scripts.push(key.clone());
        }
        scripts.sort();
    }
    (name, desc, scripts)
}

fn generate_rust_template(cwd: &Path, dir_name: &str) -> String {
    let (pkg_name, desc) = read_cargo_metadata(cwd);
    let title = if pkg_name.is_empty() {
        dir_name.to_string()
    } else {
        pkg_name
    };
    let desc_line = if desc.is_empty() {
        "A Rust project.\n".to_string()
    } else {
        format!("{}\n", desc)
    };

    let has_main = cwd.join("src/main.rs").exists();
    let has_lib = cwd.join("src/lib.rs").exists();
    let mut important = String::from("## Important files\n\n");
    if has_main {
        important.push_str("- `src/main.rs` — entry point\n");
    }
    if has_lib {
        important.push_str("- `src/lib.rs` — library root\n");
    }
    important.push_str("- `Cargo.toml` — dependencies and project config\n");

    format!(
        "# {}\n\n\
        {}\n\
        ## Coding conventions\n\n\
        - Follow standard Rust conventions (rustfmt, clippy clean)\n\
        - Use `anyhow::Result` for error handling in application code\n\
        - Prefer `thiserror` for library error types\n\n\
        ## Commands\n\n\
        - Build: `cargo build`\n\
        - Test: `cargo test`\n\
        - Lint: `cargo clippy`\n\
        - Format: `cargo fmt`\n\n\
        {}\n",
        title, desc_line, important
    )
}

fn generate_node_template(cwd: &Path, dir_name: &str) -> String {
    let (pkg_name, desc, scripts) = read_package_json(cwd);
    let title = if pkg_name.is_empty() {
        dir_name.to_string()
    } else {
        pkg_name
    };
    let desc_line = if desc.is_empty() {
        "A Node.js/JavaScript project.\n".to_string()
    } else {
        format!("{}\n", desc)
    };

    let mut commands = String::from("## Commands\n\n- Install: `npm install`\n");
    for script in &scripts {
        writeln!(commands, "- {}: `npm run {}`", script, script)
            .expect("write to String never fails");
    }

    let mut important = String::from("## Important files\n\n");
    important.push_str("- `package.json` — dependencies and scripts\n");
    if cwd.join("src").is_dir() {
        important.push_str("- `src/` — source code\n");
    }

    format!(
        "# {}\n\n\
        {}\n\
        ## Coding conventions\n\n\
        - Describe style rules, linter config, and formatting tools used\n\n\
        {}\n\
        {}\n",
        title, desc_line, commands, important
    )
}

fn generate_python_template(dir_name: &str) -> String {
    format!(
        "# {}\n\n\
        A Python project.\n\n\
        ## Coding conventions\n\n\
        - Follow PEP 8 style guidelines\n\
        - Use type hints where practical\n\n\
        ## Commands\n\n\
        - Install: `pip install -e .`\n\
        - Test: `pytest`\n\
        - Lint: `ruff check .` or `flake8`\n\
        - Format: `black .` or `ruff format .`\n\n\
        ## Important files\n\n\
        - List key modules and their purpose\n",
        dir_name
    )
}

fn generate_go_template(dir_name: &str) -> String {
    format!(
        "# {}\n\n\
        A Go project.\n\n\
        ## Coding conventions\n\n\
        - Follow standard Go conventions (gofmt, go vet)\n\
        - Use `errors.New` / `fmt.Errorf` for error creation\n\n\
        ## Commands\n\n\
        - Build: `go build ./...`\n\
        - Test: `go test ./...`\n\
        - Lint: `golangci-lint run`\n\
        - Format: `gofmt -w .`\n\n\
        ## Important files\n\n\
        - `go.mod` — module definition and dependencies\n",
        dir_name
    )
}

fn generate_java_template(dir_name: &str, build_tool: &str) -> String {
    let commands = match build_tool {
        "maven" => {
            "\
        - Build: `mvn compile`\n\
        - Test: `mvn test`\n\
        - Package: `mvn package`\n"
        }
        _ => {
            "\
        - Build: `./gradlew build`\n\
        - Test: `./gradlew test`\n\
        - Run: `./gradlew run`\n"
        }
    };
    let build_file = match build_tool {
        "maven" => "- `pom.xml` — dependencies and build config\n",
        _ => "- `build.gradle` — dependencies and build config\n",
    };
    format!(
        "# {}\n\n\
        A Java project ({}).\n\n\
        ## Coding conventions\n\n\
        - Describe style rules and formatting tools used\n\n\
        ## Commands\n\n\
        {}\n\
        ## Important files\n\n\
        {}\
        - `src/main/java/` — source code\n\
        - `src/test/java/` — tests\n",
        dir_name, build_tool, commands, build_file
    )
}

fn generate_c_template(dir_name: &str) -> String {
    format!(
        "# {}\n\n\
        A C/C++ project.\n\n\
        ## Coding conventions\n\n\
        - Describe style rules and formatting tools used\n\n\
        ## Commands\n\n\
        - Build: `make`\n\
        - Test: `make test`\n\
        - Clean: `make clean`\n\n\
        ## Important files\n\n\
        - List key source files and headers\n",
        dir_name
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Rust template ─────────────────────────────────────────────────────

    #[test]
    fn test_template_rust_includes_commands() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"myapp\"\ndescription = \"A cool app\"\n",
        )
        .unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("cargo build"),
            "should include cargo build: {out}"
        );
        assert!(
            out.contains("cargo test"),
            "should include cargo test: {out}"
        );
        assert!(
            out.contains("cargo clippy"),
            "should include cargo clippy: {out}"
        );
        assert!(
            out.contains("A cool app"),
            "should include Cargo.toml description: {out}"
        );
        assert!(
            out.contains("# myapp"),
            "should use package name as title: {out}"
        );
    }

    #[test]
    fn test_template_rust_no_description() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"foo\"\n").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(out.contains("# foo"), "should use package name: {out}");
        assert!(
            out.contains("A Rust project"),
            "should fall back to generic desc: {out}"
        );
    }

    #[test]
    fn test_template_rust_detects_main_and_lib() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"both\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "pub fn lib() {}").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(out.contains("src/main.rs"), "should list main.rs: {out}");
        assert!(out.contains("src/lib.rs"), "should list lib.rs: {out}");
    }

    #[test]
    fn test_template_rust_empty_cargo_toml() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "").unwrap();
        let out = generate_dm_md_template(dir.path());
        // Should still generate a Rust template, using dir name as fallback
        assert!(
            out.contains("cargo build"),
            "should still include Rust commands: {out}"
        );
        assert!(out.starts_with("# "), "should start with h1: {out}");
    }

    // ── Node.js template ──────────────────────────────────────────────────

    #[test]
    fn test_template_node_includes_scripts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("package.json"),
            r#"{"name": "myapp", "description": "A web app", "scripts": {"test": "jest", "build": "tsc"}}"#,
        )
        .unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("npm install"),
            "should include npm install: {out}"
        );
        assert!(
            out.contains("npm run test"),
            "should include test script: {out}"
        );
        assert!(
            out.contains("npm run build"),
            "should include build script: {out}"
        );
        assert!(
            out.contains("A web app"),
            "should include description: {out}"
        );
        assert!(out.contains("# myapp"), "should use package name: {out}");
    }

    #[test]
    fn test_template_node_empty_package_json() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Node.js/JavaScript"),
            "should include Node.js hint: {out}"
        );
        assert!(
            out.contains("npm install"),
            "should include npm install: {out}"
        );
    }

    // ── Python template ───────────────────────────────────────────────────

    #[test]
    fn test_template_python_pyproject() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "[tool.poetry]").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Python project"),
            "expected Python hint: {out}"
        );
        assert!(out.contains("pytest"), "should include pytest: {out}");
    }

    #[test]
    fn test_template_python_setup_py() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("setup.py"), "from setuptools import setup").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Python project"),
            "expected Python hint: {out}"
        );
    }

    // ── Go template ───────────────────────────────────────────────────────

    #[test]
    fn test_template_go_project() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module example.com/foo").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Go project"),
            "should detect Go project: {out}"
        );
        assert!(out.contains("go build"), "should include go build: {out}");
        assert!(out.contains("go test"), "should include go test: {out}");
    }

    // ── Java templates ────────────────────────────────────────────────────

    #[test]
    fn test_template_java_maven() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("pom.xml"), "<project></project>").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Java project (maven)"),
            "should detect Maven project: {out}"
        );
        assert!(
            out.contains("mvn compile"),
            "should include mvn compile: {out}"
        );
        assert!(out.contains("mvn test"), "should include mvn test: {out}");
    }

    #[test]
    fn test_template_java_gradle() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("build.gradle"), "plugins {}").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Java project (gradle)"),
            "should detect Gradle project: {out}"
        );
        assert!(
            out.contains("gradlew build"),
            "should include gradlew build: {out}"
        );
    }

    #[test]
    fn test_template_java_gradle_kts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("build.gradle.kts"), "plugins {}").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("Java project (gradle)"),
            "should detect Gradle KTS project: {out}"
        );
    }

    // ── C/C++ template ────────────────────────────────────────────────────

    #[test]
    fn test_template_c_cmake() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("CMakeLists.txt"),
            "cmake_minimum_required()",
        )
        .unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("C/C++ project"),
            "should detect C/C++ project: {out}"
        );
        assert!(out.contains("make"), "should include make command: {out}");
    }

    #[test]
    fn test_template_c_makefile() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Makefile"), "all:\n\tgcc main.c").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("C/C++ project"),
            "should detect C/C++ via Makefile: {out}"
        );
    }

    // ── Unknown / fallback ────────────────────────────────────────────────

    #[test]
    fn test_template_unknown_project() {
        let dir = TempDir::new().unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("<build command>"),
            "should include placeholder commands: {out}"
        );
    }

    // ── General properties ────────────────────────────────────────────────

    #[test]
    fn test_template_includes_sections() {
        let dir = TempDir::new().unwrap();
        let out = generate_dm_md_template(dir.path());
        for section in &["## Coding conventions", "## Commands", "## Important files"] {
            assert!(
                out.contains(section),
                "missing section '{section}' in: {out}"
            );
        }
    }

    #[test]
    fn test_template_starts_with_h1_heading() {
        let dir = TempDir::new().unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.starts_with("# "),
            "template should start with an h1 heading: {}",
            &out[..50.min(out.len())]
        );
    }

    #[test]
    fn test_template_rust_takes_priority_when_both_cargo_and_package_json_exist() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"foo\"\n").unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let out = generate_dm_md_template(dir.path());
        assert!(
            out.contains("cargo build"),
            "Cargo.toml should take priority: {out}"
        );
    }

    // ── Metadata readers ──────────────────────────────────────────────────

    #[test]
    fn read_cargo_metadata_extracts_name_and_desc() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"mylib\"\ndescription = \"A library\"\n",
        )
        .unwrap();
        let (name, desc) = read_cargo_metadata(dir.path());
        assert_eq!(name, "mylib");
        assert_eq!(desc, "A library");
    }

    #[test]
    fn read_cargo_metadata_missing_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let (name, desc) = read_cargo_metadata(dir.path());
        assert!(name.is_empty());
        assert!(desc.is_empty());
    }

    #[test]
    fn read_package_json_extracts_scripts() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("package.json"),
            r#"{"name": "app", "description": "desc", "scripts": {"test": "jest", "build": "tsc"}}"#,
        )
        .unwrap();
        let (name, desc, scripts) = read_package_json(dir.path());
        assert_eq!(name, "app");
        assert_eq!(desc, "desc");
        assert!(scripts.contains(&"test".to_string()));
        assert!(scripts.contains(&"build".to_string()));
    }

    #[test]
    fn read_package_json_missing_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let (name, desc, scripts) = read_package_json(dir.path());
        assert!(name.is_empty());
        assert!(desc.is_empty());
        assert!(scripts.is_empty());
    }

    // ── Gitignore logic ───────────────────────────────────────────────────

    #[test]
    fn gitignore_dm_check_detects_existing_entries() {
        let content = ".env\n.dm/\n.dm-workspace/\ndist/\n".to_string();
        assert!(
            super::append_missing_gitignore_entries(content).is_none(),
            "should detect existing dm state entries"
        );
    }

    #[test]
    fn gitignore_dm_check_detects_missing_entries() {
        let content = ".env\ndist/\n".to_string();
        let updated = super::append_missing_gitignore_entries(content);
        assert!(updated.is_some(), "should detect missing dm state entries");
    }

    #[test]
    fn gitignore_newline_appended_when_missing() {
        let content = super::append_missing_gitignore_entries(".env".to_string()).unwrap();
        assert!(content.ends_with(".dm-workspace/\n"));
        assert!(
            content.contains(".env\n"),
            "existing content should be on its own line: {content}"
        );
    }

    #[test]
    fn gitignore_newline_not_doubled_when_already_present() {
        let content = super::append_missing_gitignore_entries(".env\n".to_string()).unwrap();
        assert!(
            !content.contains("\n\n"),
            "should not double newline: {content}"
        );
    }

    #[test]
    fn gitignore_dedup_read_and_append() {
        let existing = ".env\ndist/\n".to_string();
        let content = super::append_missing_gitignore_entries(existing)
            .expect("should detect dm state entries are missing");
        assert!(content.contains(".dm/\n"), "should append .dm/");
        assert!(
            content.contains(".dm-workspace/\n"),
            "should append .dm-workspace/"
        );
        assert!(
            content.contains(".env\n"),
            "should preserve existing entries"
        );
    }

    #[test]
    fn gitignore_already_has_dm_entry_skips_write() {
        let existing = ".env\n.dm/\n.dm-workspace/\ndist/\n".to_string();
        let updated = super::append_missing_gitignore_entries(existing);
        assert!(updated.is_none(), "should detect dm state entries exist");
    }

    #[test]
    fn gitignore_adds_workspace_when_only_dm_present() {
        let existing = ".env\n.dm/\ndist/\n".to_string();
        let content = super::append_missing_gitignore_entries(existing)
            .expect("should detect .dm-workspace/ is missing");
        assert_eq!(content.matches(".dm/\n").count(), 1);
        assert!(content.contains(".dm-workspace/\n"));
    }
}
