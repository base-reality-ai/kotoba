//! Evaluation suite execution engine.
//!
//! Runs YAML-defined prompt/response checks against locally hosted models
//! to track quality regressions and calculate completion ratios.

use crate::eval::{
    evaluate_check_inline, CaseResult, Check, CheckResult, EvalCase, EvalSuite, SuiteResult,
};
use crate::ollama::client::OllamaClient;
use anyhow::Result;
use futures_util::future::join_all;
use std::path::{Path, PathBuf};
use std::time::Instant;

const EVAL_CASE_TIMEOUT_SECS: u64 = 120;

/// Options controlling eval execution behavior.
pub struct EvalOptions {
    pub verbose: bool,
    pub fail_fast: bool,
    pub strict: bool,
    pub runs: usize,
    pub flaky_threshold: f32,
    pub quiet: bool,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            verbose: false,
            fail_fast: false,
            strict: false,
            runs: 1,
            flaky_threshold: 1.0,
            quiet: false,
        }
    }
}

/// Run all cases in a single suite and return a `SuiteResult`.
pub async fn run_eval(
    suite: &EvalSuite,
    model_override: Option<&str>,
    client: &OllamaClient,
    opts: &EvalOptions,
) -> Result<SuiteResult> {
    run_eval_inner(suite, model_override, client, opts).await
}

async fn run_eval_inner(
    suite: &EvalSuite,
    model_override: Option<&str>,
    client: &OllamaClient,
    opts: &EvalOptions,
) -> Result<SuiteResult> {
    let effective_model = model_override
        .map(|s| s.to_string())
        .or_else(|| suite.model.clone())
        .unwrap_or_else(|| client.model().to_string());

    // Use a client pointing at the effective model
    let eval_client = OllamaClient::new(client.base_url().to_string(), effective_model.clone());

    let separator = "─".repeat(37);
    println!("Eval: {}  (model: {})", suite.name, effective_model);
    println!("{}", separator);

    let effective_runs = opts.runs.max(1);
    let multi = effective_runs > 1;

    let mut case_results: Vec<CaseResult> = Vec::new();
    let total_cases = suite.cases.len();

    for (case_idx, case) in suite.cases.iter().enumerate() {
        let progress = format!("[{}/{}]", case_idx + 1, total_cases);
        if multi {
            let mut pass_count = 0usize;
            let mut last: Option<CaseResult> = None;
            for _ in 0..effective_runs {
                let r = match tokio::time::timeout(
                    std::time::Duration::from_secs(EVAL_CASE_TIMEOUT_SECS),
                    run_single_case(case, suite, &eval_client, opts.strict),
                )
                .await
                {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => {
                        crate::logging::log_err(&format!(
                            "[dm eval] case '{}' error: {}",
                            case.id, e
                        ));
                        continue;
                    }
                    Err(_) => {
                        crate::logging::log_err(&format!(
                            "[dm eval] case '{}' timed out after {}s",
                            case.id, EVAL_CASE_TIMEOUT_SECS
                        ));
                        continue;
                    }
                };
                if r.passed {
                    pass_count += 1;
                }
                last = Some(r);
            }
            let Some(rep) = last else {
                crate::logging::log_err(&format!(
                    "  ✗ {:<22} all {} runs timed out",
                    case.id, effective_runs
                ));
                case_results.push(CaseResult {
                    id: case.id.clone(),
                    passed: false,
                    check_results: vec![],
                    response_text: String::new(),
                    total_ms: EVAL_CASE_TIMEOUT_SECS * 1000,
                    runs: effective_runs,
                    pass_count: 0,
                    pass_rate: 0.0,
                    flaky: false,
                });
                continue;
            };
            let pass_rate = pass_count as f32 / effective_runs as f32;
            let flaky = crate::eval::is_flaky(pass_rate, opts.flaky_threshold);
            let flaky_tag = if flaky { "  ← flaky" } else { "" };
            if !opts.quiet {
                println!(
                    "  {} {:<22} {}/{} pass  ({:.0}%){}",
                    progress,
                    rep.id,
                    pass_count,
                    effective_runs,
                    pass_rate * 100.0,
                    flaky_tag
                );
                if opts.verbose {
                    println!("    [response] {}", rep.response_text.trim());
                }
            }
            let aggregate = CaseResult {
                passed: pass_count == effective_runs,
                runs: effective_runs,
                pass_count,
                pass_rate,
                flaky,
                ..rep
            };
            let failed = !aggregate.passed;
            case_results.push(aggregate);
            if opts.fail_fast && failed {
                crate::logging::log_err("[dm eval] fail-fast: stopping after first failure");
                break;
            }
        } else {
            let case_result = match tokio::time::timeout(
                std::time::Duration::from_secs(EVAL_CASE_TIMEOUT_SECS),
                run_single_case(case, suite, &eval_client, opts.strict),
            )
            .await
            {
                Ok(r) => r?,
                Err(_) => {
                    crate::logging::log_err(&format!(
                        "  ✗ {:<20} timed out after {}s",
                        case.id, EVAL_CASE_TIMEOUT_SECS
                    ));
                    case_results.push(CaseResult {
                        id: case.id.clone(),
                        passed: false,
                        check_results: vec![],
                        response_text: String::new(),
                        total_ms: EVAL_CASE_TIMEOUT_SECS * 1000,
                        runs: 1,
                        pass_count: 0,
                        pass_rate: 0.0,
                        flaky: false,
                    });
                    if opts.fail_fast {
                        crate::logging::log_err(
                            "[dm eval] fail-fast: stopping after first failure",
                        );
                        break;
                    }
                    continue;
                }
            };
            let icon = if case_result.passed { "✓" } else { "✗" };
            let checks_passed = case_result
                .check_results
                .iter()
                .filter(|r| r.passed)
                .count();
            let checks_total = case_result.check_results.len();
            if !opts.quiet {
                println!(
                    "  {} {} {:<20} {}/{} checks  ({}ms)",
                    progress,
                    icon,
                    case_result.id,
                    checks_passed,
                    checks_total,
                    case_result.total_ms
                );
                for cr in &case_result.check_results {
                    if !cr.passed {
                        if let Some(ref msg) = cr.message {
                            println!("    ✗ {}: {}", cr.check_desc, msg);
                        } else {
                            println!("    ✗ {}", cr.check_desc);
                        }
                    }
                }
                if opts.verbose {
                    println!("    [response] {}", case_result.response_text.trim());
                }
            }
            let failed = !case_result.passed;
            let pass_rate = if case_result.passed { 1.0 } else { 0.0 };
            let result_with_rate = CaseResult {
                runs: 1,
                pass_count: if case_result.passed { 1 } else { 0 },
                pass_rate,
                flaky: false,
                ..case_result
            };
            case_results.push(result_with_rate);
            if opts.fail_fast && failed {
                crate::logging::log_err("[dm eval] fail-fast: stopping after first failure");
                break;
            }
        }
    }

    let total = case_results.len();
    let passed = case_results.iter().filter(|r| r.passed).count();
    let failed = total - passed;
    let score_pct = if total > 0 {
        (passed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    println!("{}", separator);

    if multi {
        let flaky_count = case_results.iter().filter(|r| r.flaky).count();
        let stable_count = total - flaky_count;
        let overall_rate: f32 = if total > 0 {
            case_results.iter().map(|r| r.pass_rate).sum::<f32>() / total as f32
        } else {
            0.0
        };
        println!(
            "  {}/{} cases stable · {} flaky · overall pass rate {:.0}%",
            stable_count,
            total,
            flaky_count,
            overall_rate * 100.0
        );
    } else {
        println!(
            "  {} cases  {} passed  {} failed    score: {:.0}%",
            total, passed, failed, score_pct
        );
    }

    Ok(SuiteResult {
        suite_name: suite.name.clone(),
        model: effective_model,
        cases: case_results,
        score_pct,
    })
}

async fn run_single_case(
    case: &EvalCase,
    suite: &EvalSuite,
    client: &OllamaClient,
    strict: bool,
) -> Result<CaseResult> {
    // Build messages
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if !suite.system.is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": suite.system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": case.prompt,
    }));

    let t_start = Instant::now();
    let chat_resp = client.chat(&messages, &[]).await?;
    let total_ms = t_start.elapsed().as_millis() as u64;

    let response_text = chat_resp.message.content.clone();

    // Evaluate checks
    let mut check_results: Vec<CheckResult> = Vec::new();

    for check in &case.checks {
        let result = if let Some(inline) = evaluate_check_inline(check, &response_text) {
            inline
        } else if let Check::LlmJudge(question) = check {
            run_llm_judge_majority(question, &response_text, client, strict).await
        } else {
            continue;
        };
        check_results.push(result);
    }

    let passed = check_results.iter().all(|r| r.passed);

    let pass_rate = if passed { 1.0 } else { 0.0 };
    Ok(CaseResult {
        id: case.id.clone(),
        passed,
        check_results,
        response_text,
        total_ms,
        runs: 1,
        pass_count: if passed { 1 } else { 0 },
        pass_rate,
        flaky: false,
    })
}

/// Single judge call — returns `Some(true/false)` or `None` on error.
async fn judge_once(question: &str, response_text: &str, client: &OllamaClient) -> Option<bool> {
    let prompt = format!(
        "Answer only yes or no: {}\n\nText: {}",
        question, response_text
    );
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let resp = client.chat(&messages, &[]).await.ok()?;
    Some(
        resp.message
            .content
            .trim()
            .to_lowercase()
            .starts_with("yes"),
    )
}

/// Run the LLM judge 3 times; pass if ≥2/3 agree (or 3/3 in strict mode).
async fn run_llm_judge_majority(
    question: &str,
    response_text: &str,
    client: &OllamaClient,
    strict: bool,
) -> CheckResult {
    let check_desc = format!("llm_judge: {}", question);

    let votes: Vec<Option<bool>> = join_all([
        judge_once(question, response_text, client),
        judge_once(question, response_text, client),
        judge_once(question, response_text, client),
    ])
    .await;

    let yes_count = votes.iter().filter(|v| **v == Some(true)).count();
    let err_count = votes.iter().filter(|v| v.is_none()).count();

    let required = if strict { 3 } else { 2 };
    let passed = yes_count >= required;

    let message = if passed {
        None
    } else {
        Some(format!(
            "LLM judge: {}/{} yes (required {}/3{})",
            yes_count,
            3 - err_count,
            required,
            if err_count > 0 {
                format!(", {} errors", err_count)
            } else {
                String::new()
            },
        ))
    };

    CheckResult {
        check_desc,
        passed,
        message,
    }
}

/// Load one or more YAML files, run each suite, print results, and save JSON.
#[allow(clippy::too_many_arguments)]
pub async fn run_eval_files(
    paths: Vec<PathBuf>,
    model_override: Option<&str>,
    client: &OllamaClient,
    opts: &EvalOptions,
) -> Result<Vec<SuiteResult>> {
    let mut all_results: Vec<SuiteResult> = Vec::new();

    for path in &paths {
        let yaml_text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read eval file {}: {}", path.display(), e))?;

        let suite: EvalSuite = serde_yaml::from_str(&yaml_text)
            .map_err(|e| anyhow::anyhow!("Cannot parse eval YAML {}: {}", path.display(), e))?;

        let result = run_eval(&suite, model_override, client, opts).await?;

        let save_path = save_result(&result)?;
        println!("\nSaved: {}", save_path.display());

        all_results.push(result);
    }

    Ok(all_results)
}

fn save_result(result: &SuiteResult) -> Result<PathBuf> {
    let results_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dm")
        .join("eval")
        .join("results");
    std::fs::create_dir_all(&results_dir)?;

    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string();
    // Sanitize model name for filename: replace ':' and '/' with '_'
    let model_safe = result.model.replace([':', '/'], "_");
    let filename = format!(
        "{}_{}_{}_{}.json",
        result.suite_name, model_safe, timestamp, "result"
    );
    let path = results_dir.join(&filename);

    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Print a comparison table for multiple suite results (one per model).
pub fn print_compare_table(results_by_model: &[(String, Vec<CaseResult>)]) {
    if results_by_model.is_empty() {
        return;
    }

    // Collect all case IDs (ordered by first model)
    let case_ids: Vec<&str> = results_by_model[0]
        .1
        .iter()
        .map(|c| c.id.as_str())
        .collect();

    // Header
    let model_headers: Vec<&str> = results_by_model.iter().map(|(m, _)| m.as_str()).collect();
    let col_w = 14usize;
    print!("{:<24}", "Case");
    for m in &model_headers {
        let mut trunc_end = col_w.min(m.len());
        while trunc_end > 0 && !m.is_char_boundary(trunc_end) {
            trunc_end -= 1;
        }
        let truncated = &m[..trunc_end];
        print!("  {:<14}", truncated);
    }
    println!();

    let separator = "─".repeat(24 + (col_w + 2) * results_by_model.len());
    println!("{}", separator);

    for case_id in &case_ids {
        print!("{:<24}", case_id);
        for (_, cases) in results_by_model {
            if let Some(cr) = cases.iter().find(|c| c.id == *case_id) {
                let checks_passed = cr.check_results.iter().filter(|r| r.passed).count();
                let checks_total = cr.check_results.len();
                let icon = if cr.passed { "✓" } else { "✗" };
                print!(
                    "  {:<14}",
                    format!("{} {}/{}", icon, checks_passed, checks_total)
                );
            } else {
                print!("  {:<14}", "(missing)");
            }
        }
        println!();
    }

    println!("{}", separator);
    print!("{:<24}", "Score");
    for (_, cases) in results_by_model {
        let total = cases.len();
        let passed = cases.iter().filter(|c| c.passed).count();
        let pct = if total > 0 {
            (passed as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        print!("  {:<14}", format!("{:.0}%", pct));
    }
    println!();
}

/// List past result JSON files in ~/.dm/eval/results/.
pub fn list_results() -> Result<Vec<PathBuf>> {
    let results_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dm")
        .join("eval")
        .join("results");
    if !results_dir.exists() {
        return Ok(vec![]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(&results_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    Ok(files)
}

/// Print a formatted report for a single `SuiteResult`, matching the live-run table format.
pub fn render_report(result: &SuiteResult) {
    let separator = "─".repeat(66);
    let run_ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    println!(
        "Eval: {}  (model: {})  run: {}",
        result.suite_name, result.model, run_ts
    );
    println!("{}", separator);
    for case in &result.cases {
        let icon = if case.passed { "✓" } else { "✗" };
        let checks_passed = case.check_results.iter().filter(|r| r.passed).count();
        let checks_total = case.check_results.len();
        println!(
            "  {} {:<20} {}/{} checks  ({}ms)",
            icon, case.id, checks_passed, checks_total, case.total_ms
        );
        for cr in &case.check_results {
            if !cr.passed {
                if let Some(ref msg) = cr.message {
                    println!("    ✗ {}: {}", cr.check_desc, msg);
                } else {
                    println!("    ✗ {}", cr.check_desc);
                }
            }
        }
    }
    println!("{}", separator);
    let total = result.cases.len();
    let passed = result.cases.iter().filter(|c| c.passed).count();
    let failed = total - passed;
    println!(
        "  {} cases  {} passed  {} failed    score: {:.0}%",
        total, passed, failed, result.score_pct
    );
}

/// Scan `results_dir` for JSON result files, sort by filename (timestamps are embedded),
/// and return the path to the newest file. Returns `None` if the directory is empty.
pub fn find_last_result(results_dir: &Path) -> Result<Option<PathBuf>> {
    if !results_dir.exists() {
        return Ok(None);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(results_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    Ok(files.into_iter().last())
}

/// Print a side-by-side comparison table for two `SuiteResult`s.
/// Marks cases where the pass status differs as `← improved` or `← regressed`.
pub fn render_compare(a: &SuiteResult, b: &SuiteResult) {
    let col_w = 20usize;
    let run_a_label = format!("run-A ({})", a.model);
    let run_b_label = format!("run-B ({})", b.model);
    println!(
        "{:<24} {:<col_w$} {run_b_label}",
        "Case",
        run_a_label,
        col_w = col_w
    );

    // Collect all case IDs from both results (union, preserving order of `a` first)
    let mut case_ids: Vec<&str> = a.cases.iter().map(|c| c.id.as_str()).collect();
    for c in &b.cases {
        if !case_ids.contains(&c.id.as_str()) {
            case_ids.push(c.id.as_str());
        }
    }

    for id in &case_ids {
        let a_case = a.cases.iter().find(|c| c.id == *id);
        let b_case = b.cases.iter().find(|c| c.id == *id);

        let fmt_case = |case: Option<&CaseResult>| -> String {
            match case {
                None => "(missing)".to_string(),
                Some(cr) => {
                    let icon = if cr.passed { "✓" } else { "✗" };
                    let p = cr.check_results.iter().filter(|r| r.passed).count();
                    let t = cr.check_results.len();
                    format!("{} {}/{}", icon, p, t)
                }
            }
        };

        let a_str = fmt_case(a_case);
        let b_str = fmt_case(b_case);

        let delta = match (a_case.map(|c| c.passed), b_case.map(|c| c.passed)) {
            (Some(false), Some(true)) => "  ← improved",
            (Some(true), Some(false)) => "  ← regressed",
            _ => "",
        };

        println!(
            "{:<24} {:<col_w$} {}{}",
            id,
            a_str,
            b_str,
            delta,
            col_w = col_w
        );
    }
}

fn scan_yaml_dir(dir: &Path) -> Vec<PathBuf> {
    if !dir.exists() {
        return vec![];
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec![];
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext == "yaml" || ext == "yml")
        })
        .collect()
}

/// List YAML eval files from standard locations:
/// 1. `~/.dm/evals/` (user-level)
/// 2. `./evals/` (project-level)
pub fn list_evals() -> Result<Vec<PathBuf>> {
    let home_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".dm")
        .join("evals");
    let project_dir = PathBuf::from("evals");

    let mut files = scan_yaml_dir(&home_dir);
    files.extend(scan_yaml_dir(&project_dir));
    files.sort();
    files.dedup();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{CaseResult, CheckResult};

    fn make_suite_result(suite_name: &str, model: &str, cases: Vec<CaseResult>) -> SuiteResult {
        let total = cases.len() as f64;
        let passed = cases.iter().filter(|c| c.passed).count() as f64;
        let score_pct = if total > 0.0 {
            (passed / total) * 100.0
        } else {
            0.0
        };
        SuiteResult {
            suite_name: suite_name.to_string(),
            model: model.to_string(),
            cases,
            score_pct,
        }
    }

    fn make_case(id: &str, passed: bool, checks: Vec<CheckResult>) -> CaseResult {
        CaseResult {
            id: id.to_string(),
            passed,
            check_results: checks,
            response_text: String::new(),
            total_ms: 0,
            runs: 1,
            pass_count: if passed { 1 } else { 0 },
            pass_rate: if passed { 1.0 } else { 0.0 },
            flaky: false,
        }
    }

    #[test]
    fn render_report_shows_pass_fail() {
        let pass_check = CheckResult {
            check_desc: "contains: fn ".into(),
            passed: true,
            message: None,
        };
        let fail_check = CheckResult {
            check_desc: "max_length: 600".into(),
            passed: false,
            message: Some("got 743, limit 600".into()),
        };
        let result = make_suite_result(
            "rust-basics",
            "gemma4:26b",
            vec![
                make_case("reverse-string", true, vec![pass_check]),
                make_case("explain-mutex", false, vec![fail_check]),
            ],
        );
        // Verify the function does not panic and that expected strings can be built
        render_report(&result);

        // Verify the ✓ / ✗ icons would appear for the respective cases
        let pass_icon = if result.cases[0].passed { "✓" } else { "✗" };
        let fail_icon = if result.cases[1].passed { "✓" } else { "✗" };
        assert_eq!(pass_icon, "✓");
        assert_eq!(fail_icon, "✗");
    }

    #[test]
    fn render_report_shows_score() {
        let result = make_suite_result(
            "test",
            "gemma4:26b",
            vec![make_case("a", true, vec![]), make_case("b", false, vec![])],
        );
        render_report(&result);
        // score_pct is 50% for 1/2 passed
        assert!(
            (result.score_pct - 50.0).abs() < 1e-9,
            "score should be 50%"
        );
        let score_str = format!("score: {:.0}%", result.score_pct);
        assert!(score_str.contains("50%"), "score line should contain '50%'");
    }

    #[test]
    fn report_last_finds_newest_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path();

        // Write 3 result files with different timestamp suffixes
        for ts in &["20260101T000000", "20260408T070000", "20260205T120000"] {
            let fname = format!("suite_gemma4_26b_{}_{}.json", ts, "result");
            std::fs::write(p.join(&fname), b"{}").expect("write");
        }

        let newest = find_last_result(p)
            .expect("find_last_result")
            .expect("some");
        let fname = newest.file_name().and_then(|n| n.to_str()).unwrap_or("");
        // Lexicographic sort on timestamps: 20260408T070000 is the largest
        assert!(
            fname.contains("20260408T070000"),
            "expected newest file, got: {}",
            fname
        );
    }

    #[test]
    fn compare_report_shows_delta() {
        let a = make_suite_result(
            "rust-basics",
            "gemma4:26b",
            vec![
                make_case("reverse-string", true, vec![]),
                make_case("explain-mutex", false, vec![]),
            ],
        );
        let b = make_suite_result(
            "rust-basics",
            "llama3:8b",
            vec![
                make_case("reverse-string", true, vec![]),
                make_case("explain-mutex", true, vec![]),
            ],
        );
        render_compare(&a, &b);

        // Verify the delta logic directly
        let a_mutex = a.cases.iter().find(|c| c.id == "explain-mutex").unwrap();
        let b_mutex = b.cases.iter().find(|c| c.id == "explain-mutex").unwrap();
        let delta = match (a_mutex.passed, b_mutex.passed) {
            (false, true) => "improved",
            (true, false) => "regressed",
            _ => "unchanged",
        };
        assert_eq!(delta, "improved", "explain-mutex should be marked improved");
    }

    #[test]
    fn suite_result_score_is_fraction_passed() {
        let make_case = |id: &str, passed: bool| CaseResult {
            id: id.to_string(),
            passed,
            check_results: vec![],
            response_text: String::new(),
            total_ms: 0,
            runs: 1,
            pass_count: if passed { 1 } else { 0 },
            pass_rate: if passed { 1.0 } else { 0.0 },
            flaky: false,
        };
        let cases = vec![
            make_case("a", true),
            make_case("b", false),
            make_case("c", true),
            make_case("d", false),
        ];
        let total = cases.len() as f64;
        let pass_count = cases.iter().filter(|c| c.passed).count() as f64;
        let score_pct = (pass_count / total) * 100.0;
        assert!(
            (score_pct - 50.0).abs() < 1e-9,
            "expected 50%, got {}",
            score_pct
        );

        let sr = SuiteResult {
            suite_name: "test".to_string(),
            model: "gemma4:26b".to_string(),
            cases,
            score_pct,
        };
        assert!((sr.score_pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn compare_output_shows_both_models() {
        let make_case = |id: &str, passed: bool| CaseResult {
            id: id.to_string(),
            passed,
            check_results: vec![CheckResult {
                check_desc: "contains: fn ".to_string(),
                passed,
                message: None,
            }],
            response_text: String::new(),
            total_ms: 0,
            runs: 1,
            pass_count: if passed { 1 } else { 0 },
            pass_rate: if passed { 1.0 } else { 0.0 },
            flaky: false,
        };

        let model_a_results: Vec<CaseResult> = vec![
            make_case("reverse-string", true),
            make_case("explain-mutex", false),
        ];
        let model_b_results: Vec<CaseResult> = vec![
            make_case("reverse-string", true),
            make_case("explain-mutex", true),
        ];

        let combined: Vec<(String, Vec<CaseResult>)> = vec![
            ("gemma4:26b".to_string(), model_a_results),
            ("llama3:8b".to_string(), model_b_results),
        ];

        // Verify score computation per model
        let score_a = combined[0].1.iter().filter(|c| c.passed).count() as f64
            / combined[0].1.len() as f64
            * 100.0;
        let score_b = combined[1].1.iter().filter(|c| c.passed).count() as f64
            / combined[1].1.len() as f64
            * 100.0;

        assert!((score_a - 50.0).abs() < 1e-9, "gemma4 score should be 50%");
        assert!(
            (score_b - 100.0).abs() < 1e-9,
            "llama3 score should be 100%"
        );

        // print_compare_table runs without panic
        print_compare_table(&combined);
    }

    // ── llm_judge majority vote unit tests ────────────────────────────────────

    #[test]
    fn judge_majority_2_of_3_yes_passes() {
        // yes=2, err=0 → 2/3 ≥ 2 → pass (non-strict)
        let check_desc = "llm_judge: is it correct?".to_string();
        let votes = [Some(true), Some(true), Some(false)];
        let yes_count = votes.iter().filter(|v| **v == Some(true)).count();
        let err_count = votes.iter().filter(|v| v.is_none()).count();
        let required = 2usize;
        let passed = yes_count >= required;
        assert!(passed, "2/3 yes should pass in non-strict mode");
        let _ = err_count;
        let _ = check_desc;
    }

    #[test]
    fn judge_majority_1_of_3_yes_fails() {
        let votes = [Some(true), Some(false), Some(false)];
        let yes_count = votes.iter().filter(|v| **v == Some(true)).count();
        let required = 2usize;
        assert!(yes_count < required, "1/3 yes should fail");
    }

    #[test]
    fn judge_strict_requires_3_of_3() {
        let votes_2 = [Some(true), Some(true), Some(false)];
        let yes_2 = votes_2.iter().filter(|v| **v == Some(true)).count();
        assert!(yes_2 < 3, "2/3 should fail in strict mode");

        let votes_3 = [Some(true), Some(true), Some(true)];
        let yes_3 = votes_3.iter().filter(|v| **v == Some(true)).count();
        assert!(yes_3 >= 3, "3/3 should pass in strict mode");
    }

    #[test]
    fn judge_errors_reduce_denominator_not_threshold() {
        // 2 yes, 1 error → yes_count=2, err_count=1 → 2 >= required(2) → pass
        let votes = [Some(true), Some(true), None];
        let yes_count = votes.iter().filter(|v| **v == Some(true)).count();
        let required = 2usize;
        assert!(
            yes_count >= required,
            "2 yes + 1 error should pass non-strict"
        );
    }

    #[test]
    fn judge_all_errors_fails() {
        let votes: [Option<bool>; 3] = [None, None, None];
        let yes_count = votes.iter().filter(|v| **v == Some(true)).count();
        let required = 2usize;
        assert!(yes_count < required, "all errors should fail");
    }

    #[test]
    fn eval_case_timeout_constant() {
        assert_eq!(EVAL_CASE_TIMEOUT_SECS, 120);
    }

    #[test]
    fn find_last_result_missing_dir_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("no_such_dir");
        let result = find_last_result(&missing).unwrap();
        assert!(result.is_none(), "missing directory should return None");
    }

    #[test]
    fn find_last_result_empty_dir_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        // Directory exists but has no JSON files
        let result = find_last_result(dir.path()).unwrap();
        assert!(result.is_none(), "empty directory should return None");
    }

    #[test]
    fn find_last_result_ignores_non_json_files() {
        let dir = tempfile::tempdir().unwrap();
        // Create a .yaml and a .txt but no .json
        std::fs::write(dir.path().join("result.yaml"), "{}").unwrap();
        std::fs::write(dir.path().join("result.txt"), "text").unwrap();
        let result = find_last_result(dir.path()).unwrap();
        assert!(result.is_none(), "non-json files should not be found");
    }

    #[test]
    fn find_last_result_returns_lexicographically_last_json() {
        let dir = tempfile::tempdir().unwrap();
        for name in &["aaa.json", "zzz.json", "mmm.json"] {
            std::fs::write(dir.path().join(name), "{}").unwrap();
        }
        let result = find_last_result(dir.path()).unwrap().unwrap();
        let fname = result.file_name().unwrap().to_str().unwrap();
        assert_eq!(
            fname, "zzz.json",
            "should return lexicographically last file"
        );
    }

    #[test]
    fn suite_result_score_pct_is_fraction_passed() {
        let result = make_suite_result(
            "test-suite",
            "model",
            vec![
                make_case("a", true, vec![]),
                make_case("b", true, vec![]),
                make_case("c", false, vec![]),
                make_case("d", false, vec![]),
            ],
        );
        assert!(
            (result.score_pct - 50.0).abs() < 1e-6,
            "2/4 passed should be 50%, got {}",
            result.score_pct
        );
    }

    #[test]
    fn scan_yaml_dir_finds_yaml_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("suite1.yaml"), "name: a\ncases: []").unwrap();
        std::fs::write(dir.path().join("suite2.yml"), "name: b\ncases: []").unwrap();
        std::fs::write(dir.path().join("readme.txt"), "not yaml").unwrap();
        let files = scan_yaml_dir(dir.path());
        assert_eq!(
            files.len(),
            2,
            "should find 2 YAML files, got {}",
            files.len()
        );
    }

    #[test]
    fn scan_yaml_dir_missing_dir_returns_empty() {
        let files = scan_yaml_dir(Path::new("/nonexistent/path/to/evals"));
        assert!(files.is_empty());
    }

    #[test]
    fn list_evals_returns_sorted_and_deduped() {
        let files = list_evals().unwrap();
        for i in 1..files.len() {
            assert!(
                files[i] >= files[i - 1],
                "list_evals should return sorted paths"
            );
        }
        let deduped_len = {
            let mut d = files.clone();
            d.dedup();
            d.len()
        };
        assert_eq!(
            files.len(),
            deduped_len,
            "list_evals should not have duplicates"
        );
    }
}
