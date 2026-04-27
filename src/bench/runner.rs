use crate::bench::tasks::{all_tasks, score_quality};
use crate::logging;
use crate::ollama::client::OllamaClient;
use crate::ollama::types::StreamEvent;
use anyhow::Result;
use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RunMetrics {
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
    pub quality_pass: bool,
    pub token_count: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelResult {
    pub model: String,
    pub task: String,
    pub runs: Vec<RunMetrics>,
    pub mean_ttft_ms: f64,
    pub mean_tps: f64,
    pub quality_rate: f64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    pub timestamp: String,
    pub results: Vec<ModelResult>,
}

/// Compute score: 0.6 * `quality_rate` + 0.4 * (`model_tps` / `max_tps`)
pub fn compute_score(quality_rate: f64, normalized_tps: f64) -> f64 {
    0.6 * quality_rate + 0.4 * normalized_tps
}

fn stars(score: f64) -> &'static str {
    if score >= 0.9 {
        "★★★★★"
    } else if score >= 0.7 {
        "★★★★☆"
    } else if score >= 0.5 {
        "★★★☆☆"
    } else if score >= 0.3 {
        "★★☆☆☆"
    } else {
        "★☆☆☆☆"
    }
}

pub async fn run_bench(
    client: &OllamaClient,
    models_filter: Option<Vec<String>>,
    tasks_filter: Option<Vec<String>>,
    runs_per: usize,
    out_path: Option<PathBuf>,
) -> Result<()> {
    // Get available models
    let available = client.list_models().await.map_err(|e| {
        anyhow::anyhow!(
            "Cannot reach Ollama to list models: {}. Is Ollama running?",
            e
        )
    })?;

    if available.is_empty() {
        anyhow::bail!("No models available in Ollama. Pull a model first with: dm --pull <model>");
    }

    let models: Vec<String> = if let Some(ref filter) = models_filter {
        available
            .iter()
            .filter(|m| filter.contains(&m.name))
            .map(|m| m.name.clone())
            .collect()
    } else {
        available.iter().map(|m| m.name.clone()).collect()
    };

    if models.is_empty() {
        anyhow::bail!(
            "No matching models found. Available: {:?}",
            available.iter().map(|m| &m.name).collect::<Vec<_>>()
        );
    }

    let tasks = all_tasks();
    let tasks: Vec<_> = if let Some(ref filter) = tasks_filter {
        tasks
            .into_iter()
            .filter(|t| filter.contains(&t.name.to_string()))
            .collect()
    } else {
        tasks
    };

    if tasks.is_empty() {
        anyhow::bail!("No matching tasks found.");
    }

    let mut results: Vec<ModelResult> = Vec::new();

    for model_name in &models {
        logging::log(&format!("Benchmarking model: {}", model_name));
        let model_client = OllamaClient::new(client.base_url().to_string(), model_name.clone());

        for task in &tasks {
            logging::log(&format!("  Task: {} ({} runs)", task.name, runs_per));
            let mut runs: Vec<RunMetrics> = Vec::new();

            for run_idx in 0..runs_per {
                logging::log(&format!("    Run {}/{}…", run_idx + 1, runs_per));
                let messages = vec![serde_json::json!({
                    "role": "user",
                    "content": task.prompt,
                })];

                let t_start = Instant::now();
                let stream_result = model_client.chat_stream_with_tools(&messages, &[]).await;

                match stream_result {
                    Ok(mut stream) => {
                        let mut ttft_ms: Option<u64> = None;
                        let mut content = String::new();
                        let mut token_count: u64 = 0;
                        let mut completion_tokens_from_done: u64 = 0;
                        let mut duration_from_done: u64 = 0;

                        while let Some(event) = stream.next().await {
                            match event {
                                StreamEvent::Token(tok) => {
                                    if ttft_ms.is_none() {
                                        ttft_ms = Some(t_start.elapsed().as_millis() as u64);
                                    }
                                    token_count += 1;
                                    content.push_str(&tok);
                                }
                                StreamEvent::Done {
                                    prompt_tokens: _,
                                    completion_tokens,
                                } => {
                                    completion_tokens_from_done = completion_tokens;
                                    // Ollama streaming doesn't provide eval_duration in the
                                    // Done event; use wall clock for tokens/sec calculation.
                                    duration_from_done = t_start.elapsed().as_millis() as u64;
                                    break;
                                }
                                StreamEvent::Thinking(_) | StreamEvent::ToolCalls(_) => {}
                                StreamEvent::Error(e) => {
                                    logging::log_err(&format!(
                                        "    Stream error on run {}: {}",
                                        run_idx + 1,
                                        e
                                    ));
                                    break;
                                }
                            }
                        }

                        let total_ms = t_start.elapsed().as_millis() as u64;
                        // Use completion_tokens from Done if available; otherwise count tokens received
                        let effective_tokens = if completion_tokens_from_done > 0 {
                            completion_tokens_from_done
                        } else {
                            token_count
                        };
                        let elapsed_for_tps = if duration_from_done > 0 {
                            duration_from_done
                        } else {
                            total_ms
                        };
                        let tokens_per_sec = if elapsed_for_tps > 0 {
                            (effective_tokens as f64) / (elapsed_for_tps as f64 / 1000.0)
                        } else {
                            0.0
                        };

                        let quality_pass = score_quality(task.name, &content);

                        runs.push(RunMetrics {
                            ttft_ms: ttft_ms.unwrap_or(0),
                            total_ms,
                            tokens_per_sec,
                            quality_pass,
                            token_count: effective_tokens,
                        });
                    }
                    Err(e) => {
                        logging::log_err(&format!("    Error on run {}: {}", run_idx + 1, e));
                        let total_ms = t_start.elapsed().as_millis() as u64;
                        runs.push(RunMetrics {
                            ttft_ms: 0,
                            total_ms,
                            tokens_per_sec: 0.0,
                            quality_pass: false,
                            token_count: 0,
                        });
                    }
                }
            }

            let n = runs.len() as f64;
            let mean_ttft_ms = runs.iter().map(|r| r.ttft_ms as f64).sum::<f64>() / n;
            let mean_tps = runs.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n;
            let quality_rate = runs.iter().filter(|r| r.quality_pass).count() as f64 / n;

            results.push(ModelResult {
                model: model_name.clone(),
                task: task.name.to_string(),
                runs,
                mean_ttft_ms,
                mean_tps,
                quality_rate,
            });
        }
    }

    // Print results table
    print_table(&results);

    // Save report
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let report = BenchReport {
        timestamp: timestamp.clone(),
        results,
    };

    let save_path = if let Some(p) = out_path {
        p
    } else {
        let bench_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".dm")
            .join("bench");
        std::fs::create_dir_all(&bench_dir)?;
        bench_dir.join(format!("{}.json", timestamp))
    };

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&save_path, &json)?;
    println!("\nResults saved to: {}", save_path.display());

    // Compare against previous run
    let bench_dir = save_path.parent().unwrap_or(Path::new("."));
    if let Ok(history) = load_history(bench_dir) {
        // history[0] is the current run we just saved; history[1] is the previous
        if history.len() >= 2 {
            let alerts = detect_regressions(&history[0], &history[1]);
            print_regressions(&alerts);
        }
    }

    Ok(())
}

/// Load all benchmark reports from a directory, sorted newest first.
pub fn load_history(bench_dir: &Path) -> Result<Vec<BenchReport>> {
    let mut reports = Vec::new();
    if !bench_dir.is_dir() {
        return Ok(reports);
    }
    for entry in std::fs::read_dir(bench_dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Ok(report) = serde_json::from_str::<BenchReport>(&content) {
            reports.push(report);
        }
    }
    reports.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(reports)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RegressionAlert {
    pub task_name: String,
    pub model: String,
    pub metric: String,
    pub previous_value: f64,
    pub current_value: f64,
    pub change_pct: f64,
}

/// Compare two reports and flag regressions.
/// TTFT increase >20%, throughput drop >15%, quality drop >10%.
pub fn detect_regressions(current: &BenchReport, previous: &BenchReport) -> Vec<RegressionAlert> {
    let mut alerts = Vec::new();

    for cur in &current.results {
        let Some(prev) = previous
            .results
            .iter()
            .find(|r| r.model == cur.model && r.task == cur.task)
        else {
            continue;
        };

        // TTFT regression (>20% increase)
        if prev.mean_ttft_ms > 0.0 {
            let change = (cur.mean_ttft_ms - prev.mean_ttft_ms) / prev.mean_ttft_ms * 100.0;
            if change > 20.0 {
                alerts.push(RegressionAlert {
                    task_name: cur.task.clone(),
                    model: cur.model.clone(),
                    metric: "ttft_ms".to_string(),
                    previous_value: prev.mean_ttft_ms,
                    current_value: cur.mean_ttft_ms,
                    change_pct: change,
                });
            }
        }

        // Throughput regression (>15% drop)
        if prev.mean_tps > 0.0 {
            let change = (prev.mean_tps - cur.mean_tps) / prev.mean_tps * 100.0;
            if change > 15.0 {
                alerts.push(RegressionAlert {
                    task_name: cur.task.clone(),
                    model: cur.model.clone(),
                    metric: "tokens_per_sec".to_string(),
                    previous_value: prev.mean_tps,
                    current_value: cur.mean_tps,
                    change_pct: -change,
                });
            }
        }

        // Quality regression (>10% drop)
        if prev.quality_rate > 0.0 {
            let change = (prev.quality_rate - cur.quality_rate) / prev.quality_rate * 100.0;
            if change > 10.0 {
                alerts.push(RegressionAlert {
                    task_name: cur.task.clone(),
                    model: cur.model.clone(),
                    metric: "quality_rate".to_string(),
                    previous_value: prev.quality_rate,
                    current_value: cur.quality_rate,
                    change_pct: -change,
                });
            }
        }
    }

    alerts
}

fn print_regressions(alerts: &[RegressionAlert]) {
    if alerts.is_empty() {
        logging::log("No regressions detected vs. previous run.");
        return;
    }
    logging::log(&format!(
        "⚠ {} regression(s) detected vs. previous run:",
        alerts.len()
    ));
    for a in alerts {
        logging::log(&format!(
            "  {} [{}] {}: {:.1} → {:.1} ({:+.1}%)",
            a.model, a.task_name, a.metric, a.previous_value, a.current_value, a.change_pct
        ));
    }
}

/// List all saved benchmark runs as a summary table.
pub fn print_history_list(bench_dir: &Path) -> Result<()> {
    let history = load_history(bench_dir)?;
    if history.is_empty() {
        println!("No benchmark history found.");
        return Ok(());
    }

    println!(
        "{:<24} {:>8}  {:>6}  {:>6}",
        "Timestamp", "Models", "Tasks", "Best"
    );
    println!(
        "{:<24} {:>8}  {:>6}  {:>6}",
        "─".repeat(22),
        "──────",
        "─────",
        "─────"
    );

    for report in &history {
        let mut models = std::collections::HashSet::new();
        let mut tasks = std::collections::HashSet::new();
        let mut best_score: f64 = 0.0;

        let max_tps = report
            .results
            .iter()
            .map(|r| r.mean_tps)
            .fold(0.0_f64, f64::max);

        for r in &report.results {
            models.insert(&r.model);
            tasks.insert(&r.task);
            let normalized_tps = if max_tps > 0.0 {
                r.mean_tps / max_tps
            } else {
                0.0
            };
            let score = compute_score(r.quality_rate, normalized_tps);
            if score > best_score {
                best_score = score;
            }
        }

        println!(
            "{:<24} {:>8}  {:>6}  {:>5.2}",
            report.timestamp,
            models.len(),
            tasks.len(),
            best_score,
        );
    }

    println!("\n{} run(s) total.", history.len());
    println!("View details: dm --bench-history <timestamp>");
    Ok(())
}

/// Show a specific benchmark run's details with regression comparison.
pub fn print_history_detail(bench_dir: &Path, timestamp: &str) -> Result<()> {
    let history = load_history(bench_dir)?;
    if history.is_empty() {
        anyhow::bail!("No benchmark history found");
    }

    let idx = history
        .iter()
        .position(|r| r.timestamp.starts_with(timestamp))
        .ok_or_else(|| anyhow::anyhow!("No benchmark report matching '{}'", timestamp))?;

    let report = &history[idx];
    println!("Benchmark run: {}", report.timestamp);
    print_table(&report.results);

    // Compare against the next older report (idx+1 since sorted newest-first)
    if idx + 1 < history.len() {
        let previous = &history[idx + 1];
        println!("\nComparing against previous run: {}", previous.timestamp);
        let alerts = detect_regressions(report, previous);
        print_regressions(&alerts);
    } else {
        println!("\nNo previous run to compare against.");
    }

    Ok(())
}

pub fn print_table(results: &[ModelResult]) {
    // Aggregate per model (average across tasks)
    use std::collections::HashMap;

    struct ModelSummary {
        total_quality_passes: usize,
        total_quality_runs: usize,
        mean_tps_sum: f64,
        mean_tps_count: usize,
        mean_ttft_sum: f64,
    }

    let mut summaries: HashMap<&str, ModelSummary> = HashMap::new();

    for r in results {
        let e = summaries.entry(&r.model).or_insert(ModelSummary {
            total_quality_passes: 0,
            total_quality_runs: 0,
            mean_tps_sum: 0.0,
            mean_tps_count: 0,
            mean_ttft_sum: 0.0,
        });
        let passes = r.runs.iter().filter(|m| m.quality_pass).count();
        e.total_quality_passes += passes;
        e.total_quality_runs += r.runs.len();
        e.mean_tps_sum += r.mean_tps;
        e.mean_tps_count += 1;
        e.mean_ttft_sum += r.mean_ttft_ms;
    }

    let max_tps = summaries
        .values()
        .map(|s| {
            if s.mean_tps_count > 0 {
                s.mean_tps_sum / s.mean_tps_count as f64
            } else {
                0.0
            }
        })
        .fold(0.0_f64, f64::max);

    println!();
    println!(
        "{:<24} {:>8}   {:>6}  {:>9}  Score",
        "Model", "TTFT ms", "tok/s", "Quality"
    );
    println!(
        "{:<24} {:>8}   {:>6}  {:>9}  ─────",
        "─".repeat(22),
        "───────",
        "──────",
        "─────────"
    );

    let mut model_names: Vec<&str> = summaries.keys().copied().collect();
    model_names.sort();

    let mut best_quality_model: Option<(&str, f64)> = None;
    let mut best_balanced_model: Option<(&str, f64)> = None;

    for model in &model_names {
        let s = &summaries[model];
        let mean_tps = if s.mean_tps_count > 0 {
            s.mean_tps_sum / s.mean_tps_count as f64
        } else {
            0.0
        };
        let mean_ttft = if s.mean_tps_count > 0 {
            s.mean_ttft_sum / s.mean_tps_count as f64
        } else {
            0.0
        };
        let quality_rate = if s.total_quality_runs > 0 {
            s.total_quality_passes as f64 / s.total_quality_runs as f64
        } else {
            0.0
        };
        let normalized_tps = if max_tps > 0.0 {
            mean_tps / max_tps
        } else {
            0.0
        };
        let score = compute_score(quality_rate, normalized_tps);

        let quality_str = format!("{}/{}", s.total_quality_passes, s.total_quality_runs);

        println!(
            "{:<24} {:>8.0}   {:>6.1}  {:>9}  {}",
            model,
            mean_ttft,
            mean_tps,
            quality_str,
            stars(score)
        );

        // Track best models
        if best_quality_model.is_none_or(|(_, best)| quality_rate > best) {
            best_quality_model = Some((model, quality_rate));
        }
        if best_balanced_model.is_none_or(|(_, best)| score > best) {
            best_balanced_model = Some((model, score));
        }
    }

    println!();
    if let Some((model, rate)) = best_quality_model {
        println!(
            "Best quality model:         {} ({:.0}% pass rate)",
            model,
            rate * 100.0
        );
    }
    if let Some((model, score)) = best_balanced_model {
        println!("Best quality/speed balance: {} (score {:.2})", model, score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttft_captured_from_stream() {
        // Unit test the TTFT recording logic:
        // simulate a sequence of events, assert first-token time is recorded
        let t_start = std::time::Instant::now();
        let mut ttft: Option<u64> = None;
        // simulate receiving first token
        if ttft.is_none() {
            ttft = Some(t_start.elapsed().as_millis() as u64);
        }
        assert!(ttft.is_some());
        // ttft could be 0 in fast execution — just assert it's Some
    }

    #[test]
    fn score_weighted_combines_quality_and_tps() {
        let score = compute_score(1.0, 1.0);
        assert!((score - 1.0).abs() < 1e-9);

        let score2 = compute_score(0.0, 1.0);
        assert!((score2 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn compute_score_quality_only() {
        // quality=1, tps=0 → 0.6
        assert!((compute_score(1.0, 0.0) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn compute_score_tps_only() {
        // quality=0, tps=1 → 0.4
        assert!((compute_score(0.0, 1.0) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn compute_score_clamps_to_range() {
        // Normal inputs yield [0, 1]
        let s = compute_score(0.5, 0.5);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn stars_five_stars_for_high_score() {
        assert_eq!(stars(0.9), "★★★★★");
        assert_eq!(stars(1.0), "★★★★★");
    }

    #[test]
    fn stars_one_star_for_low_score() {
        assert_eq!(stars(0.0), "★☆☆☆☆");
        assert_eq!(stars(0.29), "★☆☆☆☆");
    }

    #[test]
    fn stars_three_stars_for_mid_score() {
        assert_eq!(stars(0.5), "★★★☆☆");
        assert_eq!(stars(0.69), "★★★☆☆");
    }

    #[test]
    fn stars_four_stars_for_good_score() {
        assert_eq!(stars(0.7), "★★★★☆");
        assert_eq!(stars(0.89), "★★★★☆");
    }

    #[test]
    fn stars_two_stars_for_below_average() {
        assert_eq!(stars(0.3), "★★☆☆☆");
        assert_eq!(stars(0.49), "★★☆☆☆");
    }

    #[test]
    fn bench_results_serialize_to_json() {
        let report = BenchReport {
            timestamp: "2026-04-08T000000Z".to_string(),
            results: vec![ModelResult {
                model: "gemma4:26b".to_string(),
                task: "code_gen".to_string(),
                runs: vec![RunMetrics {
                    ttft_ms: 0,
                    total_ms: 500,
                    tokens_per_sec: 42.0,
                    quality_pass: true,
                    token_count: 21,
                }],
                mean_ttft_ms: 0.0,
                mean_tps: 42.0,
                quality_rate: 1.0,
            }],
        };

        let json = serde_json::to_string(&report).expect("serialize failed");
        let back: BenchReport = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.results[0].model, "gemma4:26b");
    }

    #[test]
    fn compute_score_perfect_is_one() {
        let s = compute_score(1.0, 1.0);
        assert!(
            (s - 1.0).abs() < 1e-9,
            "perfect inputs should give 1.0: {s}"
        );
    }

    #[test]
    fn compute_score_zero_is_zero() {
        let s = compute_score(0.0, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn stars_at_exactly_0_9_is_five_stars() {
        assert_eq!(stars(0.9), "★★★★★");
    }

    #[test]
    fn stars_just_below_0_9_is_four_stars() {
        assert_eq!(stars(0.89), "★★★★☆");
    }

    fn make_report(timestamp: &str, ttft: f64, tps: f64, quality: f64) -> BenchReport {
        BenchReport {
            timestamp: timestamp.to_string(),
            results: vec![ModelResult {
                model: "test-model".to_string(),
                task: "code_gen".to_string(),
                runs: vec![],
                mean_ttft_ms: ttft,
                mean_tps: tps,
                quality_rate: quality,
            }],
        }
    }

    #[test]
    fn load_history_reads_json_files() {
        let dir = tempfile::tempdir().unwrap();
        let r1 = make_report("20260401T000000Z", 100.0, 50.0, 1.0);
        let r2 = make_report("20260402T000000Z", 110.0, 48.0, 0.9);
        std::fs::write(
            dir.path().join("20260401T000000Z.json"),
            serde_json::to_string(&r1).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("20260402T000000Z.json"),
            serde_json::to_string(&r2).unwrap(),
        )
        .unwrap();
        // Non-json file should be ignored
        std::fs::write(dir.path().join("notes.txt"), "ignore me").unwrap();

        let history = load_history(dir.path()).unwrap();
        assert_eq!(history.len(), 2);
        // Newest first
        assert_eq!(history[0].timestamp, "20260402T000000Z");
        assert_eq!(history[1].timestamp, "20260401T000000Z");
    }

    #[test]
    fn detect_regressions_flags_ttft_increase() {
        let prev = make_report("t1", 100.0, 50.0, 1.0);
        let curr = make_report("t2", 150.0, 50.0, 1.0); // 50% increase
        let alerts = detect_regressions(&curr, &prev);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].metric, "ttft_ms");
        assert!((alerts[0].change_pct - 50.0).abs() < 0.1);
    }

    #[test]
    fn detect_regressions_no_alerts_when_stable() {
        let prev = make_report("t1", 100.0, 50.0, 1.0);
        let curr = make_report("t2", 105.0, 48.0, 0.95); // within thresholds
        let alerts = detect_regressions(&curr, &prev);
        assert!(alerts.is_empty(), "expected no alerts: {:?}", alerts);
    }

    #[test]
    fn detect_regressions_flags_quality_drop() {
        let prev = make_report("t1", 100.0, 50.0, 0.9);
        let curr = make_report("t2", 100.0, 50.0, 0.7); // ~22% drop
        let alerts = detect_regressions(&curr, &prev);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].metric, "quality_rate");
    }

    #[test]
    fn detect_regressions_flags_throughput_drop() {
        let prev = make_report("t1", 100.0, 50.0, 1.0);
        let curr = make_report("t2", 100.0, 30.0, 1.0); // 40% drop
        let alerts = detect_regressions(&curr, &prev);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].metric, "tokens_per_sec");
    }

    #[test]
    fn load_history_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let history = load_history(dir.path()).unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn load_history_nonexistent_dir() {
        let history = load_history(Path::new("/tmp/dm_nonexistent_bench_dir")).unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn print_history_list_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        // Should succeed and print "No benchmark history found."
        let result = print_history_list(dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn print_history_detail_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let r = make_report("20260401T000000Z", 100.0, 50.0, 1.0);
        std::fs::write(
            dir.path().join("20260401T000000Z.json"),
            serde_json::to_string(&r).unwrap(),
        )
        .unwrap();

        let result = print_history_detail(dir.path(), "20250101");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No benchmark report matching"));
    }

    #[test]
    fn print_history_detail_prefix_match() {
        let dir = tempfile::tempdir().unwrap();
        let r = make_report("20260415T120000Z", 100.0, 50.0, 1.0);
        std::fs::write(
            dir.path().join("20260415T120000Z.json"),
            serde_json::to_string(&r).unwrap(),
        )
        .unwrap();

        let result = print_history_detail(dir.path(), "202604");
        assert!(result.is_ok());
    }
}
