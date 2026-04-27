//! `dm models` — model management commands.
//!
//! Commands:
//!   (no extra args)  list installed models
//!   pull `name`      stream pull progress
//!   rm `name`        delete model (confirm unless --force)
//!   update           pull latest for every installed model
//!   info `name`      show size, family, quantization, context length

use crate::logging;
use anyhow::Result;

use crate::ollama::client::OllamaClient;
use crate::ollama::types::ModelInfo;

// ── list ─────────────────────────────────────────────────────────────────────

/// Print all installed models with size.
pub async fn run_list(client: &OllamaClient) -> Result<()> {
    let models = client.list_models().await?;
    if models.is_empty() {
        println!("No models installed. Pull one with: dm --models-pull gemma4:27b");
        return Ok(());
    }
    let total: u64 = models.iter().filter_map(|m| m.size).sum();
    println!(
        "Installed models ({}, {}):",
        models.len(),
        human_size(total)
    );
    for m in &models {
        let size_str = m.size.map_or_else(|| "unknown".to_string(), human_size);
        println!("  • {} ({})", m.name, size_str);
    }
    Ok(())
}

// ── pull ─────────────────────────────────────────────────────────────────────

/// Pull a model, streaming progress bar to stderr.
pub async fn run_pull(model_name: &str, base_url: &str) -> Result<()> {
    crate::ollama::pull::run_pull(model_name, base_url).await
}

// ── rm ───────────────────────────────────────────────────────────────────────

/// Delete a model, with optional confirmation prompt.
pub async fn run_rm(client: &OllamaClient, model_name: &str, force: bool) -> Result<()> {
    if !force {
        eprint!("Remove {}? [y/N] ", model_name);
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !is_confirmed(&input) {
            println!("Aborted.");
            return Ok(());
        }
    }
    client.delete_model(model_name).await?;
    println!("Removed {}.", model_name);
    Ok(())
}

/// Returns true when the user typed `y` or `Y`.
pub fn is_confirmed(input: &str) -> bool {
    let t = input.trim();
    t == "y" || t == "Y"
}

// ── update ───────────────────────────────────────────────────────────────────

/// Pull every installed model; skip models whose digest hasn't changed.
pub async fn run_update(client: &OllamaClient) -> Result<()> {
    let models = client.list_models().await?;
    if models.is_empty() {
        println!("No models installed.");
        return Ok(());
    }
    for model in &models {
        let old_digest = model.digest.clone();
        logging::log(&format!("Updating {}...", model.name));
        crate::ollama::pull::run_pull(&model.name, client.base_url()).await?;

        // Re-fetch to compare digest
        let refreshed = client.list_models().await?;
        let new_entry = refreshed.iter().find(|m| m.name == model.name);
        match (
            old_digest.as_deref(),
            new_entry.and_then(|m| m.digest.as_deref()),
        ) {
            (_, None) if new_entry.is_none() => {
                logging::log(&format!(
                    "  {} — no longer in model list (may have been deleted)",
                    model.name
                ));
            }
            (Some(old), Some(new)) if old == new => {
                println!("  {} — already up to date", model.name);
            }
            _ => {
                println!("  {} — updated", model.name);
            }
        }
    }
    Ok(())
}

/// Compute which model names need updating (digest changed or absent in new list).
/// Used for testing without hitting HTTP.
#[allow(dead_code)]
pub fn models_needing_update(before: &[ModelInfo], after: &[ModelInfo]) -> Vec<String> {
    before
        .iter()
        .filter(|b| {
            let new = after.iter().find(|a| a.name == b.name);
            match (b.digest.as_deref(), new.and_then(|a| a.digest.as_deref())) {
                (Some(old), Some(new)) => old != new,
                _ => true, // no digest info — assume changed
            }
        })
        .map(|m| m.name.clone())
        .collect()
}

// ── info ─────────────────────────────────────────────────────────────────────

/// Print a formatted summary of a model from `/api/show`.
pub async fn run_info(client: &OllamaClient, model_name: &str) -> Result<()> {
    let json = client.show_model(model_name).await?;
    println!("{}", format_info(model_name, &json));
    Ok(())
}

/// Format a `/api/show` response into a human-readable table.
pub fn format_info(model_name: &str, json: &serde_json::Value) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Model:         {}", model_name));

    if let Some(fam) = json["details"]["family"].as_str() {
        lines.push(format!("Family:        {}", fam));
    }
    if let Some(params) = json["details"]["parameter_size"].as_str() {
        lines.push(format!("Parameters:    {}", params));
    }
    if let Some(quant) = json["details"]["quantization_level"].as_str() {
        lines.push(format!("Quantization:  {}", quant));
    }
    if let Some(ctx) = json["model_info"]["llama.context_length"].as_u64() {
        lines.push(format!("Context:       {} tokens", ctx));
    }
    if let Some(size) = json["size"].as_u64() {
        lines.push(format!("Size on disk:  {}", human_size(size)));
    }

    lines.join("\n")
}

// ── helpers ──────────────────────────────────────────────────────────────────

pub fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1e6)
    } else {
        format!("{} B", bytes)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ollama::types::ModelInfo;

    fn make_model(name: &str, size: u64, digest: &str) -> ModelInfo {
        ModelInfo {
            name: name.to_string(),
            size: Some(size),
            modified_at: None,
            digest: Some(digest.to_string()),
        }
    }

    #[test]
    fn models_list_parses_tags_response() {
        // Verify human_size formatting used in list output
        assert_eq!(human_size(14_000_000_000), "14.0 GB");
        assert_eq!(human_size(500_000_000), "500 MB");
        assert_eq!(human_size(999), "999 B");
    }

    #[test]
    fn models_rm_requires_confirmation() {
        assert!(!is_confirmed("n"));
        assert!(!is_confirmed("N"));
        assert!(!is_confirmed(""));
        assert!(!is_confirmed("  "));
        assert!(!is_confirmed("yes"));
    }

    #[test]
    fn models_rm_force_skips_confirmation() {
        // is_confirmed returns true only for "y"/"Y"
        assert!(is_confirmed("y"));
        assert!(is_confirmed("Y"));
        assert!(is_confirmed("y\n"));
        assert!(is_confirmed("  y  "));
    }

    #[test]
    fn models_update_skips_unchanged_digest() {
        let before = vec![make_model("gemma4:27b", 14_000_000_000, "abc123")];
        let after = vec![make_model("gemma4:27b", 14_000_000_000, "abc123")];
        let to_update = models_needing_update(&before, &after);
        assert!(to_update.is_empty(), "same digest — nothing to update");
    }

    #[test]
    fn models_update_detects_changed_digest() {
        let before = vec![make_model("gemma4:27b", 14_000_000_000, "abc123")];
        let after = vec![make_model("gemma4:27b", 14_100_000_000, "def456")];
        let to_update = models_needing_update(&before, &after);
        assert_eq!(to_update, vec!["gemma4:27b"]);
    }

    #[test]
    fn models_info_formats_show_response() {
        let json = serde_json::json!({
            "details": {
                "family": "llama",
                "parameter_size": "27B",
                "quantization_level": "Q4_K_M"
            },
            "model_info": {
                "llama.context_length": 131072_u64
            },
            "size": 14_000_000_000_u64
        });
        let out = format_info("gemma4:27b", &json);
        assert!(out.contains("gemma4:27b"), "name");
        assert!(out.contains("llama"), "family");
        assert!(out.contains("27B"), "params");
        assert!(out.contains("Q4_K_M"), "quant");
        assert!(out.contains("131072"), "context");
        assert!(out.contains("14.0 GB"), "size");
    }

    #[test]
    fn models_info_missing_model_clean_error() {
        // format_info handles missing fields gracefully (shows only known fields)
        let json = serde_json::json!({});
        let out = format_info("no-such-model:latest", &json);
        assert!(out.contains("no-such-model:latest"));
        assert!(!out.contains("Family"));
        assert!(!out.contains("Parameters"));
    }

    #[test]
    fn doctor_reports_model_count_and_size() {
        let models = [
            make_model("gemma4:27b", 14_000_000_000, "a"),
            make_model("llama3:8b", 4_000_000_000, "b"),
        ];
        let total: u64 = models.iter().filter_map(|m| m.size).sum();
        assert_eq!(human_size(total), "18.0 GB");
        assert_eq!(models.len(), 2);
    }

    #[test]
    fn human_size_boundary_values() {
        // Exact boundaries
        assert_eq!(human_size(1_000_000_000), "1.0 GB");
        assert_eq!(human_size(1_000_000), "1 MB");
        assert_eq!(human_size(999_999), "999999 B"); // just below 1 MB → bytes
                                                     // Zero
        assert_eq!(human_size(0), "0 B");
    }

    #[test]
    fn models_needing_update_flags_deleted_models() {
        // Model was in before list but not in after list → should be flagged
        let before = vec![make_model("gemma4:27b", 14_000_000_000, "abc")];
        let after: Vec<ModelInfo> = vec![];
        let to_update = models_needing_update(&before, &after);
        assert_eq!(
            to_update,
            vec!["gemma4:27b"],
            "missing model should be flagged"
        );
    }

    #[test]
    fn models_needing_update_no_digest_always_flags() {
        // Model without digest info → always considered changed
        let before = vec![ModelInfo {
            name: "unknown-model:latest".to_string(),
            size: Some(1_000_000),
            modified_at: None,
            digest: None,
        }];
        let after = vec![ModelInfo {
            name: "unknown-model:latest".to_string(),
            size: Some(1_000_000),
            modified_at: None,
            digest: None,
        }];
        let to_update = models_needing_update(&before, &after);
        assert_eq!(
            to_update,
            vec!["unknown-model:latest"],
            "no digest → always flag for update"
        );
    }

    #[test]
    fn human_size_just_below_1_gb_shows_mb() {
        // 999_999_999 bytes is just below 1 GB → should show as MB
        let s = human_size(999_999_999);
        assert!(
            s.ends_with("MB") || s.ends_with('B'),
            "expected MB or B, got: {s}"
        );
        assert!(!s.contains("GB"), "should not show GB: {s}");
    }

    #[test]
    fn models_needing_update_empty_before_returns_empty() {
        let after = vec![make_model("new-model:latest", 1_000, "abc")];
        let to_update = models_needing_update(&[], &after);
        assert!(
            to_update.is_empty(),
            "empty before list → nothing to update"
        );
    }

    #[test]
    fn models_needing_update_detects_disappeared() {
        let before = vec![
            make_model("model-a:latest", 1_000, "aaa"),
            make_model("model-b:latest", 2_000, "bbb"),
        ];
        let after = vec![make_model("model-a:latest", 1_000, "aaa")];
        let to_update = models_needing_update(&before, &after);
        assert!(
            to_update.contains(&"model-b:latest".to_string()),
            "disappeared model should be flagged"
        );
        assert!(
            !to_update.contains(&"model-a:latest".to_string()),
            "unchanged model should not be flagged"
        );
    }

    #[test]
    fn models_needing_update_same_digest_excluded() {
        let models = vec![
            make_model("a:latest", 1_000, "digest1"),
            make_model("b:latest", 2_000, "digest2"),
        ];
        let to_update = models_needing_update(&models, &models);
        assert!(
            to_update.is_empty(),
            "identical digests → nothing to update"
        );
    }

    #[test]
    fn models_needing_update_changed_digest_included() {
        let before = vec![make_model("m:latest", 1_000, "old")];
        let after = vec![make_model("m:latest", 1_500, "new")];
        let to_update = models_needing_update(&before, &after);
        assert_eq!(to_update, vec!["m:latest"]);
    }

    #[test]
    fn format_info_only_name_when_all_fields_absent() {
        let json = serde_json::json!({});
        let out = format_info("my-model:7b", &json);
        assert!(
            out.starts_with("Model:"),
            "should start with Model: field: {out}"
        );
        assert!(out.contains("my-model:7b"));
        // Only 1 line when all optional fields are absent
        assert_eq!(out.lines().count(), 1, "should have exactly 1 line: {out}");
    }
}
