/// Static hints reused across `src/ollama/*` bail sites. Keeping the
/// wording here means the two error dialects (`format_dm_error` at the
/// headless top level and the `\n  → Try:` append in ollama) stay
/// semantically aligned: `HINT_CONNECT_FAILED` reads the same whether
/// it follows `dm:` on stderr or trails an anyhow chain.
pub const HINT_CONNECT_FAILED: &str = "Is Ollama running? Try: ollama serve";

/// The embed endpoint returned 200 but the body shape was not what we
/// expected (missing `embeddings` array, or non-float values inside).
/// Usually means the selected model isn't an embed model.
pub const HINT_EMBED_MALFORMED: &str =
    "Model may not support embeddings. Try: ollama pull nomic-embed-text";

/// Ollama returned 404/"not found" for `/api/show` or `/api/delete`.
pub const HINT_MODEL_NOT_FOUND: &str = "Model not installed. Try: ollama pull <model>";

/// Classify a `POST /api/pull` non-success status into an actionable hint.
/// Returns `None` for statuses that don't map to a useful next step — the
/// caller should still surface the body text, just without a tail.
pub fn hint_for_pull_status(status: u16) -> Option<&'static str> {
    match status {
        404 => Some("Model name not found on the registry. Check spelling: ollama pull <model>"),
        401 | 403 => Some("Registry auth failed. Check: ollama login"),
        429 => Some("Rate limited by the registry. Wait a moment and retry"),
        500..=599 => Some("Ollama registry is failing. Try again later or check status"),
        _ => None,
    }
}

/// Classify a `POST /api/show` non-success status (404 is special-cased
/// at the call site because it carries the model name).
pub fn hint_for_show_status(status: u16) -> Option<&'static str> {
    match status {
        404 => Some(HINT_MODEL_NOT_FOUND),
        500..=599 => Some("Ollama server error. Try: ollama serve (restart)"),
        _ => None,
    }
}

/// Classify a `DELETE /api/delete` non-success status.
pub fn hint_for_delete_status(status: u16) -> Option<&'static str> {
    match status {
        404 => Some(HINT_MODEL_NOT_FOUND),
        500..=599 => Some("Ollama server error. Try: ollama serve (restart)"),
        _ => None,
    }
}

pub fn hint_for_error(err: &anyhow::Error) -> &'static str {
    let msg = format!("{:#}", err).to_lowercase();

    if msg.contains("connection refused")
        || msg.contains("connection reset")
        || msg.contains("connection closed")
    {
        return "Is Ollama running? Try: ollama serve";
    }
    if msg.contains("timed out") || msg.contains("timeout") {
        return "Ollama is not responding. Check: ollama ps";
    }
    if msg.contains("not found") || msg.contains("no such model") {
        return "Model not available. Try: ollama pull <model>";
    }
    if msg.contains("context length") || msg.contains("too long") || msg.contains("maximum context")
    {
        return "Context too large. Try: /compact or use a model with larger context window";
    }
    if msg.contains("out of memory") {
        return "Out of VRAM. Try: use a smaller quantization (e.g. q4_0) or smaller model";
    }
    if msg.contains("permission denied") || msg.contains("access denied") {
        return "Permission denied. Check file/directory ownership and permissions";
    }
    if msg.contains("no space left")
        || msg.contains("disk full")
        || msg.contains("insufficient storage")
    {
        return "Disk full. Free space in ~/.ollama/ or the working directory";
    }
    if msg.contains("rate limit") || msg.contains("too many requests") || msg.contains("429") {
        return "Rate limited by Ollama. Wait a moment or reduce concurrent requests";
    }
    if msg.contains("model loading") || msg.contains("service unavailable") || msg.contains("503") {
        return "Model is loading. Wait 10-30s and retry";
    }
    if msg.contains("500") && msg.contains("internal server error") {
        return "Ollama internal error. Try: ollama serve (restart) or check logs";
    }

    "Check Ollama status: ollama ps"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_connection_refused() {
        let err = anyhow::anyhow!("connection refused");
        assert!(hint_for_error(&err).contains("ollama serve"));
    }

    #[test]
    fn hint_timeout() {
        let err = anyhow::anyhow!("request timed out");
        assert!(hint_for_error(&err).contains("ollama ps"));
    }

    #[test]
    fn hint_model_not_found() {
        let err = anyhow::anyhow!("model 'llama99' not found");
        assert!(hint_for_error(&err).contains("ollama pull"));
    }

    #[test]
    fn hint_context_overflow() {
        let err = anyhow::anyhow!("context length exceeded");
        assert!(hint_for_error(&err).contains("compact"));
    }

    #[test]
    fn hint_oom() {
        let err = anyhow::anyhow!("CUDA out of memory");
        assert!(hint_for_error(&err).contains("quantization"));
    }

    #[test]
    fn hint_generic_fallback() {
        let err = anyhow::anyhow!("unknown error xyz");
        assert!(hint_for_error(&err).contains("ollama ps"));
    }

    #[test]
    fn hint_connection_closed() {
        let err = anyhow::anyhow!("connection closed by remote");
        assert!(hint_for_error(&err).contains("ollama serve"));
    }

    #[test]
    fn hint_permission_denied() {
        let err = anyhow::anyhow!("permission denied: /var/lib/ollama/models");
        assert!(hint_for_error(&err).contains("Permission denied"));
    }

    #[test]
    fn hint_disk_full() {
        let err = anyhow::anyhow!("no space left on device");
        assert!(hint_for_error(&err).contains("Disk full"));
    }

    #[test]
    fn hint_rate_limited() {
        let err = anyhow::anyhow!("rate limit exceeded");
        assert!(hint_for_error(&err).contains("Rate limited"));
    }

    #[test]
    fn hint_429_status() {
        let err = anyhow::anyhow!("Ollama 429: too many requests");
        assert!(hint_for_error(&err).contains("Rate limited"));
    }

    #[test]
    fn hint_model_loading() {
        let err = anyhow::anyhow!("model loading, please wait");
        assert!(hint_for_error(&err).contains("loading"));
    }

    #[test]
    fn hint_503_status() {
        let err = anyhow::anyhow!("Ollama 503 service unavailable");
        assert!(hint_for_error(&err).contains("loading"));
    }

    #[test]
    fn hint_500_internal_server_error() {
        let err = anyhow::anyhow!("Ollama 500 internal server error");
        assert!(hint_for_error(&err).contains("restart"));
    }

    #[test]
    fn pull_status_404_suggests_spelling() {
        let hint = hint_for_pull_status(404).expect("404 has a hint");
        assert!(
            hint.contains("spelling") || hint.contains("not found"),
            "404 hint should point at the name: {hint}"
        );
    }

    #[test]
    fn pull_status_429_suggests_retry() {
        let hint = hint_for_pull_status(429).expect("429 has a hint");
        assert!(
            hint.to_lowercase().contains("rate"),
            "429 hint should name the condition: {hint}"
        );
    }

    #[test]
    fn pull_status_5xx_suggests_retry_later() {
        for status in [500u16, 502, 503, 599] {
            let hint =
                hint_for_pull_status(status).unwrap_or_else(|| panic!("{status} should have hint"));
            assert!(
                hint.contains("registry") || hint.contains("later"),
                "5xx hint should explain transience: {hint}"
            );
        }
    }

    #[test]
    fn pull_status_200_no_hint() {
        // Success codes shouldn't pretend to have actionable advice.
        assert!(hint_for_pull_status(200).is_none());
    }

    #[test]
    fn show_status_404_uses_model_not_found() {
        let hint = hint_for_show_status(404).expect("404 has a hint");
        assert_eq!(hint, HINT_MODEL_NOT_FOUND);
    }

    #[test]
    fn delete_status_404_uses_model_not_found() {
        let hint = hint_for_delete_status(404).expect("404 has a hint");
        assert_eq!(hint, HINT_MODEL_NOT_FOUND);
    }

    #[test]
    fn static_hints_mention_next_step() {
        // Pillar 1: every hint should name a concrete next action.
        // We check for either "Try:" (the main.rs dialect) or an
        // imperative verb — either pattern passes the "actionable" bar.
        for hint in [
            HINT_CONNECT_FAILED,
            HINT_EMBED_MALFORMED,
            HINT_MODEL_NOT_FOUND,
        ] {
            assert!(
                hint.contains("Try:") || hint.contains("Check") || hint.contains("ollama "),
                "static hint not actionable: {hint}"
            );
        }
    }

    #[test]
    fn status_classifiers_are_consistent_on_404() {
        // All three *_status classifiers should route 404 somewhere
        // actionable — regression guard against a future edit that
        // accidentally drops 404 from one map.
        assert!(hint_for_pull_status(404).is_some());
        assert!(hint_for_show_status(404).is_some());
        assert!(hint_for_delete_status(404).is_some());
    }
}
