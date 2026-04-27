//! Shared Ollama health probe + renderer used by `dm init` and `dm doctor`.
//!
//! Both commands ask the same question — is Ollama reachable, and is the
//! configured model installed — and should answer it with the same words.
//! Any divergence between init and doctor is a bug.

use crate::ollama::client::OllamaClient;
use std::time::Duration;

/// Default timeout for the `list_models` call used by `probe`. Shared
/// between init and doctor so the two surfaces can't drift.
pub const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, PartialEq, Eq)]
pub enum OllamaHealth {
    ReachableWithModels {
        count: usize,
        configured_installed: bool,
    },
    ReachableEmpty,
    Unreachable,
}

/// Run a bounded probe against Ollama. Never returns an error — all
/// failure modes collapse into `OllamaHealth::Unreachable` so callers
/// can render a single consistent verdict.
pub async fn probe(
    client: &OllamaClient,
    timeout: Duration,
    configured_model: &str,
) -> OllamaHealth {
    match tokio::time::timeout(timeout, client.list_models()).await {
        Ok(Ok(models)) => {
            if models.is_empty() {
                OllamaHealth::ReachableEmpty
            } else {
                let configured_installed = models.iter().any(|m| m.name == configured_model);
                OllamaHealth::ReachableWithModels {
                    count: models.len(),
                    configured_installed,
                }
            }
        }
        _ => OllamaHealth::Unreachable,
    }
}

pub fn render_ollama_health(status: &OllamaHealth, host: &str, model: &str) -> String {
    match status {
        OllamaHealth::ReachableWithModels {
            count,
            configured_installed: true,
        } => format!(
            "✓ Ollama reachable at {} ({} model{} installed; {} ready)",
            host,
            count,
            if *count == 1 { "" } else { "s" },
            model,
        ),
        OllamaHealth::ReachableWithModels {
            count,
            configured_installed: false,
        } => format!(
            "✓ Ollama reachable at {} ({} model{} installed)\n  ⚠ configured model {} is not installed — run: ollama pull {}",
            host,
            count,
            if *count == 1 { "" } else { "s" },
            model,
            model,
        ),
        OllamaHealth::ReachableEmpty => format!(
            "✓ Ollama reachable at {} (no models installed)\n  → run: ollama pull {}",
            host, model,
        ),
        OllamaHealth::Unreachable => format!(
            "⚠ Ollama not reachable at {} — start with: ollama serve",
            host,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_ollama_status_reachable_with_configured_installed() {
        let status = OllamaHealth::ReachableWithModels {
            count: 3,
            configured_installed: true,
        };
        let out = render_ollama_health(&status, "localhost:11434", "gemma4:26b");
        assert!(out.contains("✓"), "should mark success: {out}");
        assert!(
            out.contains("localhost:11434"),
            "should include host: {out}"
        );
        assert!(out.contains("3 models"), "should pluralize count: {out}");
        assert!(
            out.contains("gemma4:26b ready"),
            "should say model is ready: {out}"
        );
        assert!(
            !out.contains("⚠"),
            "should not warn when model installed: {out}"
        );
    }

    #[test]
    fn render_ollama_status_reachable_but_model_missing() {
        let status = OllamaHealth::ReachableWithModels {
            count: 1,
            configured_installed: false,
        };
        let out = render_ollama_health(&status, "localhost:11434", "gemma4:26b-128k");
        assert!(out.contains("✓"), "daemon is up, keep the checkmark: {out}");
        assert!(out.contains("1 model installed"), "singular count: {out}");
        assert!(out.contains("⚠"), "should warn about missing model: {out}");
        assert!(
            out.contains("ollama pull gemma4:26b-128k"),
            "should suggest pull of configured model: {out}",
        );
    }

    #[test]
    fn render_ollama_status_reachable_empty_suggests_pull() {
        let status = OllamaHealth::ReachableEmpty;
        let out = render_ollama_health(&status, "localhost:11434", "gemma4:26b");
        assert!(
            out.contains("no models installed"),
            "should state the condition: {out}"
        );
        assert!(
            out.contains("ollama pull gemma4:26b"),
            "should suggest pulling configured model: {out}",
        );
    }

    #[test]
    fn render_ollama_status_unreachable_suggests_serve() {
        let status = OllamaHealth::Unreachable;
        let out = render_ollama_health(&status, "localhost:11434", "gemma4:26b");
        assert!(out.contains("⚠"), "should warn: {out}");
        assert!(
            out.contains("not reachable"),
            "should state the condition: {out}"
        );
        assert!(
            out.contains("localhost:11434"),
            "should include host: {out}"
        );
        assert!(
            out.contains("ollama serve"),
            "should suggest starting daemon: {out}"
        );
    }

    #[tokio::test]
    async fn probe_against_unreachable_host_collapses_to_unreachable_within_timeout() {
        // 127.0.0.1:1 — port 1 is reserved and nothing listens. ConnectionRefused
        // should surface immediately; if the runtime or reqwest blocks, the
        // 500ms timeout forces `Unreachable`. Either way: non-panic path.
        let client = OllamaClient::new(
            "http://127.0.0.1:1/api".to_string(),
            "test-model".to_string(),
        );
        let start = std::time::Instant::now();
        let status = probe(&client, Duration::from_millis(500), "test-model").await;
        let elapsed = start.elapsed();
        assert_eq!(status, OllamaHealth::Unreachable);
        assert!(
            elapsed < Duration::from_secs(2),
            "probe must honor timeout; took {:?}",
            elapsed,
        );
    }
}
