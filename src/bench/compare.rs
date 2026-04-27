use crate::logging;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::StreamEvent;
use anyhow::Result;
use futures_util::StreamExt;
use std::time::Instant;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CompareResult {
    pub model: String,
    pub response: String,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CompareReport {
    pub timestamp: String,
    pub prompt: String,
    pub results: Vec<CompareResult>,
}

pub async fn run_compare(
    models: &[String],
    prompt: &str,
    side_by_side: bool,
    base_url: &str,
) -> Result<()> {
    let mut results: Vec<CompareResult> = Vec::new();

    for model in models {
        let model_client = OllamaClient::new(base_url.to_string(), model.clone());
        let messages = vec![serde_json::json!({"role": "user", "content": prompt})];

        let t_start = Instant::now();
        let stream_result = model_client.chat_stream_with_tools(&messages, &[]).await;

        match stream_result {
            Ok(mut stream) => {
                let mut content = String::new();
                let mut token_count: u64 = 0;
                let mut completion_tokens_from_done: u64 = 0;

                while let Some(event) = stream.next().await {
                    match event {
                        StreamEvent::Token(tok) => {
                            token_count += 1;
                            content.push_str(&tok);
                        }
                        StreamEvent::Done {
                            completion_tokens, ..
                        } => {
                            completion_tokens_from_done = completion_tokens;
                            break;
                        }
                        StreamEvent::Error(e) => {
                            logging::log_err(&format!("Error from model {}: {}", model, e));
                            break;
                        }
                        StreamEvent::Thinking(_) | StreamEvent::ToolCalls(_) => {}
                    }
                }

                let total_ms = t_start.elapsed().as_millis() as u64;
                let effective_tokens = if completion_tokens_from_done > 0 {
                    completion_tokens_from_done
                } else {
                    token_count
                };
                let tokens_per_sec = if total_ms > 0 {
                    (effective_tokens as f64) / (total_ms as f64 / 1000.0)
                } else {
                    0.0
                };

                results.push(CompareResult {
                    model: model.clone(),
                    response: content,
                    total_ms,
                    tokens_per_sec,
                });
            }
            Err(e) => {
                logging::log_err(&format!("Failed to connect to model {}: {}", model, e));
                let total_ms = t_start.elapsed().as_millis() as u64;
                results.push(CompareResult {
                    model: model.clone(),
                    response: format!("[Error: {}]", e),
                    total_ms,
                    tokens_per_sec: 0.0,
                });
            }
        }
    }

    // Print results
    if side_by_side && results.len() >= 2 {
        print_side_by_side(&results);
    } else {
        print_sequential(&results);
    }

    // Save report
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let report = CompareReport {
        timestamp: timestamp.clone(),
        prompt: prompt.to_string(),
        results,
    };

    let bench_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".dm")
        .join("bench");
    std::fs::create_dir_all(&bench_dir)?;
    let save_path = bench_dir.join(format!("compare_{}.json", timestamp));
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&save_path, &json)?;
    println!("\nReport saved to: {}", save_path.display());

    Ok(())
}

fn print_sequential(results: &[CompareResult]) {
    for r in results {
        let sep = format!(
            "━━━ {} ({}ms, {:.1} tok/s) ━━━",
            r.model, r.total_ms, r.tokens_per_sec
        );
        println!("{}", sep);
        println!("{}", r.response);
        println!();
    }
}

fn print_side_by_side(results: &[CompareResult]) {
    // Use crossterm to get terminal width, fall back to 80
    let term_width = crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80);
    let col_width = term_width / 2;

    // Print headers
    let left = &results[0];
    let right = &results[1];
    let left_header = format!(
        "━━━ {} ({}ms, {:.1} tok/s)",
        left.model, left.total_ms, left.tokens_per_sec
    );
    let right_header = format!(
        "━━━ {} ({}ms, {:.1} tok/s)",
        right.model, right.total_ms, right.tokens_per_sec
    );
    println!(
        "{:<col_width$} {}",
        left_header,
        right_header,
        col_width = col_width
    );

    // Split each response into lines
    let left_lines: Vec<&str> = left.response.lines().collect();
    let right_lines: Vec<&str> = right.response.lines().collect();
    let max_lines = left_lines.len().max(right_lines.len());

    for i in 0..max_lines {
        let l = left_lines.get(i).copied().unwrap_or("");
        let r = right_lines.get(i).copied().unwrap_or("");
        // Truncate left side to column width (back off to UTF-8 char boundary)
        let l_display = if l.len() > col_width {
            let mut end = col_width;
            while end > 0 && !l.is_char_boundary(end) {
                end -= 1;
            }
            &l[..end]
        } else {
            l
        };
        println!("{:<col_width$} {}", l_display, r, col_width = col_width);
    }
    println!();

    // Print remaining models (if any) sequentially
    if results.len() > 2 {
        for r in &results[2..] {
            let sep = format!(
                "━━━ {} ({}ms, {:.1} tok/s) ━━━",
                r.model, r.total_ms, r.tokens_per_sec
            );
            println!("{}", sep);
            println!("{}", r.response);
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_report_serializes() {
        let report = CompareReport {
            timestamp: "2026-04-08T000000Z".to_string(),
            prompt: "Hello, world!".to_string(),
            results: vec![
                CompareResult {
                    model: "gemma4:26b".to_string(),
                    response: "Hi there!".to_string(),
                    total_ms: 312,
                    tokens_per_sec: 42.1,
                },
                CompareResult {
                    model: "llama3:8b".to_string(),
                    response: "Hello!".to_string(),
                    total_ms: 98,
                    tokens_per_sec: 87.3,
                },
            ],
        };

        let json = serde_json::to_string(&report).expect("serialize failed");
        let back: CompareReport = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.results.len(), 2);
        assert_eq!(back.results[0].model, "gemma4:26b");
        assert_eq!(back.results[1].model, "llama3:8b");
    }

    #[test]
    fn compare_result_tps_preserved() {
        let r = CompareResult {
            model: "m".to_string(),
            response: "ok".to_string(),
            total_ms: 1000,
            tokens_per_sec: 15.75,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: CompareResult = serde_json::from_str(&json).unwrap();
        assert!((back.tokens_per_sec - 15.75).abs() < 1e-9);
    }

    #[test]
    fn compare_report_empty_results() {
        let report = CompareReport {
            timestamp: "2026-04-11T000000Z".to_string(),
            prompt: "test".to_string(),
            results: Vec::new(),
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: CompareReport = serde_json::from_str(&json).unwrap();
        assert!(back.results.is_empty());
        assert_eq!(back.prompt, "test");
    }

    #[test]
    fn print_sequential_does_not_panic() {
        let results = vec![
            CompareResult {
                model: "m1".into(),
                response: "resp1".into(),
                total_ms: 100,
                tokens_per_sec: 10.0,
            },
            CompareResult {
                model: "m2".into(),
                response: "resp2\nline2".into(),
                total_ms: 200,
                tokens_per_sec: 5.5,
            },
        ];
        // Should not panic; output goes to stdout
        print_sequential(&results);
    }

    #[test]
    fn print_side_by_side_does_not_panic_with_multibyte() {
        // Emoji is 4 bytes; if col_width falls inside it, the old code would panic
        let long_left = "😀".repeat(30); // each emoji is 4 bytes
        let results = vec![
            CompareResult {
                model: "m1".into(),
                response: long_left,
                total_ms: 100,
                tokens_per_sec: 1.0,
            },
            CompareResult {
                model: "m2".into(),
                response: "short".into(),
                total_ms: 50,
                tokens_per_sec: 2.0,
            },
        ];
        // Should not panic even when col_width slices into a multi-byte char
        print_side_by_side(&results);
    }

    #[test]
    fn print_side_by_side_extra_models_printed_sequentially() {
        let results = vec![
            CompareResult {
                model: "m1".into(),
                response: "a".into(),
                total_ms: 10,
                tokens_per_sec: 1.0,
            },
            CompareResult {
                model: "m2".into(),
                response: "b".into(),
                total_ms: 20,
                tokens_per_sec: 2.0,
            },
            CompareResult {
                model: "m3".into(),
                response: "c".into(),
                total_ms: 30,
                tokens_per_sec: 3.0,
            },
        ];
        // Three models: first two side-by-side, third printed sequentially — should not panic
        print_side_by_side(&results);
    }

    #[test]
    fn compare_result_zero_tps_serializes() {
        let r = CompareResult {
            model: "slow".to_string(),
            response: String::new(),
            total_ms: 0,
            tokens_per_sec: 0.0,
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: CompareResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tokens_per_sec, 0.0);
    }

    #[test]
    fn compare_report_prompt_survives_roundtrip() {
        let prompt = "What is 2+2?".to_string();
        let report = CompareReport {
            timestamp: "2026-01-01T000000Z".to_string(),
            prompt: prompt.clone(),
            results: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: CompareReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.prompt, prompt);
    }

    #[test]
    fn utf8_boundary_truncation_is_safe() {
        // Directly test the truncation logic used in print_side_by_side
        let s = "hello 😀 world"; // emoji starts at byte 6
        let col_width = 8usize; // would fall mid-emoji without the fix
        let truncated = if s.len() > col_width {
            let mut end = col_width;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            &s[..end]
        } else {
            s
        };
        // Result must be valid UTF-8 (it's a &str, so it is) and must be ≤ col_width bytes
        assert!(truncated.len() <= col_width);
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
    }
}
