use crate::logging;
use anyhow::Result;

/// Pull a model from Ollama, showing a progress bar on stderr.
pub async fn run_pull(model: &str, base_url: &str) -> Result<()> {
    let url = format!("{}/api/pull", base_url);
    let body = serde_json::json!({
        "name": model,
        "stream": true,
    });

    let client = reqwest::Client::new();
    const RETRY_DELAYS: [u64; 3] = [1, 2, 4];

    let mut last_err = None;
    for (attempt, delay) in std::iter::once(0)
        .chain(RETRY_DELAYS.iter().copied())
        .enumerate()
    {
        if attempt > 0 {
            logging::log(&format!(
                "Retrying pull in {}s (attempt {}/4)...",
                delay,
                attempt + 1
            ));
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
        }
        match client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    match crate::ollama::hints::hint_for_pull_status(status.as_u16()) {
                        Some(hint) => anyhow::bail!(
                            "Ollama pull returned {}: {}\n  → {}",
                            status,
                            body_text,
                            hint
                        ),
                        None => {
                            anyhow::bail!("Ollama pull returned {}: {}", status, body_text)
                        }
                    }
                }
                return stream_pull_response(model, resp).await;
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    match last_err {
        Some(e) => Err(anyhow::anyhow!(
            "Failed to connect to Ollama for pull after 4 attempts: {}\n  → {}",
            e,
            crate::ollama::hints::HINT_CONNECT_FAILED
        )),
        None => unreachable!("last_err is always set in the retry loop"),
    }
}

async fn stream_pull_response(model: &str, resp: reqwest::Response) -> Result<()> {
    use futures_util::StreamExt;
    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        let bytes = chunk_result.map_err(|e| anyhow::anyhow!("Stream read error: {}", e))?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf = buf[nl + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Check for error
            if let Some(err_msg) = v["error"].as_str() {
                logging::log("");
                let lower = err_msg.to_lowercase();
                let hint = if lower.contains("not found") || lower.contains("pull model manifest") {
                    Some(crate::ollama::hints::HINT_MODEL_NOT_FOUND)
                } else if lower.contains("connection") || lower.contains("timed out") {
                    Some(crate::ollama::hints::HINT_CONNECT_FAILED)
                } else {
                    None
                };
                return match hint {
                    Some(h) => Err(anyhow::anyhow!("{}\n  → {}", err_msg, h)),
                    None => Err(anyhow::anyhow!("{}", err_msg)),
                };
            }

            let status_str = v["status"].as_str().unwrap_or("");

            // Progress bar if we have total/completed
            if let (Some(total), Some(completed)) = (v["total"].as_u64(), v["completed"].as_u64()) {
                if total > 0 {
                    let pct = (completed * 100 / total) as usize;
                    let filled = pct * 20 / 100;
                    let empty = 20usize.saturating_sub(filled);
                    let bar: String = "█".repeat(filled) + &"░".repeat(empty);

                    let (size_completed, unit_c) = human_size(completed);
                    let (size_total, unit_t) = human_size(total);

                    logging::log(&format!(
                        "Pulling {model}  [{bar}]  {pct:3}%  {size_completed:.1}{unit_c}/{size_total:.1}{unit_t}"
                    ));
                    continue;
                }
            }

            // Success
            if status_str == "success" {
                logging::log("Done.");
                return Ok(());
            }

            // Other status lines
            logging::log(&format!("Pulling {model}: {status_str}"));
        }
    }

    // Stream ended — assume done
    logging::log("Done.");
    Ok(())
}

/// Pull a model, sending progress messages to a channel (for TUI integration).
pub async fn pull_with_progress(
    model: &str,
    base_url: &str,
    progress_tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    let url = format!("{}/api/pull", base_url);
    let body = serde_json::json!({ "name": model, "stream": true });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&body).send().await.map_err(|e| {
        anyhow::anyhow!(
            "Failed to connect to Ollama for pull: {}\n  → {}",
            e,
            crate::ollama::hints::HINT_CONNECT_FAILED
        )
    })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        match crate::ollama::hints::hint_for_pull_status(status.as_u16()) {
            Some(hint) => {
                anyhow::bail!(
                    "Ollama pull returned {}: {}\n  → {}",
                    status,
                    body_text,
                    hint
                )
            }
            None => anyhow::bail!("Ollama pull returned {}: {}", status, body_text),
        }
    }

    stream_pull_with_progress(model, resp, progress_tx).await
}

async fn stream_pull_with_progress(
    model: &str,
    resp: reqwest::Response,
    progress_tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    use futures_util::StreamExt;
    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk_result) = byte_stream.next().await {
        let bytes = chunk_result.map_err(|e| anyhow::anyhow!("Stream read error: {}", e))?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf = buf[nl + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            let v: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(err_msg) = v["error"].as_str() {
                return Err(anyhow::anyhow!("{}", err_msg));
            }

            let status_str = v["status"].as_str().unwrap_or("");

            if let (Some(total), Some(completed)) = (v["total"].as_u64(), v["completed"].as_u64()) {
                if total > 0 {
                    let pct = (completed * 100 / total) as usize;
                    let filled = pct * 20 / 100;
                    let empty = 20usize.saturating_sub(filled);
                    let bar: String = "█".repeat(filled) + &"░".repeat(empty);
                    let (sc, uc) = human_size(completed);
                    let (st, ut) = human_size(total);
                    progress_tx
                        .send(format!(
                            "Pulling {model}  [{bar}]  {pct:3}%  {sc:.1}{uc}/{st:.1}{ut}"
                        ))
                        .await
                        .ok();
                    continue;
                }
            }

            if status_str == "success" {
                progress_tx
                    .send(format!("Model {} pulled successfully.", model))
                    .await
                    .ok();
                return Ok(());
            }

            if !status_str.is_empty() {
                progress_tx
                    .send(format!("Pulling {}: {}", model, status_str))
                    .await
                    .ok();
            }
        }
    }

    progress_tx
        .send(format!("Model {} pulled successfully.", model))
        .await
        .ok();
    Ok(())
}

fn human_size(bytes: u64) -> (f64, &'static str) {
    if bytes >= 1_000_000_000 {
        (bytes as f64 / 1_000_000_000.0, "GB")
    } else {
        (bytes as f64 / 1_000_000.0, "MB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_size_bytes_below_gb_is_mb() {
        let (val, unit) = human_size(500_000_000); // 500 MB
        assert_eq!(unit, "MB");
        assert!((val - 500.0).abs() < 1.0);
    }

    #[test]
    fn human_size_exactly_one_gb_is_gb() {
        let (val, unit) = human_size(1_000_000_000);
        assert_eq!(unit, "GB");
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn human_size_large_value_is_gb() {
        let (val, unit) = human_size(4_500_000_000); // 4.5 GB
        assert_eq!(unit, "GB");
        assert!((val - 4.5).abs() < 0.01);
    }

    #[test]
    fn human_size_small_value_is_mb() {
        let (val, unit) = human_size(1_000_000); // 1 MB
        assert_eq!(unit, "MB");
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn human_size_zero_is_mb() {
        let (val, unit) = human_size(0);
        assert_eq!(unit, "MB");
        assert_eq!(val, 0.0);
    }

    #[test]
    fn human_size_just_below_gb_is_mb() {
        let (_, unit) = human_size(999_999_999);
        assert_eq!(unit, "MB", "999,999,999 bytes should be formatted as MB");
    }

    #[test]
    fn human_size_just_above_gb_is_gb() {
        let (_, unit) = human_size(1_000_000_001);
        assert_eq!(unit, "GB", "1,000,000,001 bytes should be formatted as GB");
    }

    #[test]
    fn pull_url_format() {
        let base = "http://localhost:11434";
        let url = format!("{}/api/pull", base);
        assert_eq!(url, "http://localhost:11434/api/pull");
    }

    #[test]
    fn pull_retry_delays_are_exponential() {
        const DELAYS: [u64; 3] = [1, 2, 4];
        assert_eq!(DELAYS.len(), 3);
        for i in 1..DELAYS.len() {
            assert_eq!(DELAYS[i], DELAYS[i - 1] * 2, "delay should double");
        }
    }
}
