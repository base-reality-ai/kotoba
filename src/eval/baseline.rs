//! Baseline persistence and regression diffing for `dm eval --freeze` / `--ci`.

use crate::eval::SuiteResult;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Slim snapshot of one case's pass/fail — enough to detect regressions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaselineCase {
    pub id: String,
    pub passed: bool,
    /// Pass rate when recorded with --eval-runs > 1. None for legacy baselines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_rate: Option<f32>,
}

impl BaselineCase {
    /// Effective pass rate: use stored rate if present, otherwise infer from bool.
    pub fn effective_rate(&self) -> f32 {
        self.pass_rate
            .unwrap_or(if self.passed { 1.0 } else { 0.0 })
    }
}

/// A saved baseline for one suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub suite_name: String,
    pub model: String,
    pub cases: Vec<BaselineCase>,
}

/// A regression: a case that was passing in the baseline but is now failing.
#[derive(Debug, Clone, PartialEq)]
pub struct Regression {
    pub case_id: String,
    /// Check-level failure messages from the current run.
    pub failures: Vec<String>,
}

/// An improvement: a case that was failing in the baseline but is now passing.
#[derive(Debug, Clone, PartialEq)]
pub struct Improvement {
    pub case_id: String,
}

/// Full diff between a baseline and a fresh run.
#[derive(Debug, Clone)]
pub struct BaselineDiff {
    pub regressions: Vec<Regression>,
    pub improvements: Vec<Improvement>,
}

impl BaselineDiff {
    #[allow(dead_code)]
    pub fn is_clean(&self) -> bool {
        self.regressions.is_empty()
    }
}

// ── Path helpers ──────────────────────────────────────────────────────────────

pub fn baseline_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("eval").join("baselines")
}

pub fn baseline_path(config_dir: &Path, suite_name: &str) -> PathBuf {
    baseline_dir(config_dir).join(format!("{}.json", slugify(suite_name)))
}

fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── Core pure functions ───────────────────────────────────────────────────────

/// Save a `SuiteResult` as the canonical baseline for its suite.
pub fn save_baseline(config_dir: &Path, result: &SuiteResult) -> Result<PathBuf> {
    let dir = baseline_dir(config_dir);
    std::fs::create_dir_all(&dir).context("Cannot create eval/baselines directory")?;

    let baseline = Baseline {
        suite_name: result.suite_name.clone(),
        model: result.model.clone(),
        cases: result
            .cases
            .iter()
            .map(|c| BaselineCase {
                id: c.id.clone(),
                passed: c.passed,
                pass_rate: if c.runs > 1 { Some(c.pass_rate) } else { None },
            })
            .collect(),
    };

    let path = baseline_path(config_dir, &result.suite_name);
    let json = serde_json::to_string_pretty(&baseline).context("Serialize baseline")?;
    let tmp_path = path.with_extension("json.tmp");
    std::fs::write(&tmp_path, json)
        .with_context(|| format!("Write baseline tmp to {}", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("Rename baseline to {}", path.display()))?;
    Ok(path)
}

/// Load a baseline for `suite_name`. Returns `None` if no baseline file exists.
pub fn load_baseline(config_dir: &Path, suite_name: &str) -> Result<Option<Baseline>> {
    let path = baseline_path(config_dir, suite_name);
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path)
        .with_context(|| format!("Read baseline {}", path.display()))?;
    let baseline: Baseline = serde_json::from_str(&json)
        .with_context(|| format!("Parse baseline {}", path.display()))?;
    Ok(Some(baseline))
}

/// Diff a fresh `SuiteResult` against its baseline.
/// Returns `None` when no baseline exists for this suite.
pub fn diff_results(baseline: &Baseline, current: &SuiteResult) -> BaselineDiff {
    use std::collections::HashMap;

    let baseline_map: HashMap<&str, bool> = baseline
        .cases
        .iter()
        .map(|c| (c.id.as_str(), c.passed))
        .collect();

    let mut regressions = Vec::new();
    let mut improvements = Vec::new();

    for case in &current.cases {
        let was_passing = baseline_map.get(case.id.as_str()).copied();

        match (was_passing, case.passed) {
            (Some(true), false) => {
                // Was passing, now failing → regression
                let failures: Vec<String> = case
                    .check_results
                    .iter()
                    .filter(|r| !r.passed)
                    .map(|r| {
                        if let Some(ref msg) = r.message {
                            format!("{}: {}", r.check_desc, msg)
                        } else {
                            r.check_desc.clone()
                        }
                    })
                    .collect();
                regressions.push(Regression {
                    case_id: case.id.clone(),
                    failures,
                });
            }
            (Some(false), true) => {
                improvements.push(Improvement {
                    case_id: case.id.clone(),
                });
            }
            _ => {} // new case (no baseline entry) or unchanged → no action
        }
    }

    BaselineDiff {
        regressions,
        improvements,
    }
}

/// Rate-aware diff: regression when pass rate drops more than `regression_threshold`.
/// Falls back to bool comparison for legacy baselines with no rate info.
pub fn diff_results_rate(
    baseline: &Baseline,
    current: &SuiteResult,
    regression_threshold: f32,
) -> BaselineDiff {
    use std::collections::HashMap;

    let baseline_map: HashMap<&str, &BaselineCase> =
        baseline.cases.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut regressions = Vec::new();
    let mut improvements = Vec::new();

    for case in &current.cases {
        let Some(bl) = baseline_map.get(case.id.as_str()) else {
            continue;
        };
        let old_rate = bl.effective_rate();
        let new_rate = case.pass_rate;

        let drop = old_rate - new_rate;
        let rise = new_rate - old_rate;

        if drop >= regression_threshold {
            let failures: Vec<String> = case
                .check_results
                .iter()
                .filter(|r| !r.passed)
                .map(|r| {
                    r.message.as_ref().map_or_else(
                        || r.check_desc.clone(),
                        |m| format!("{}: {}", r.check_desc, m),
                    )
                })
                .collect();
            regressions.push(Regression {
                case_id: case.id.clone(),
                failures,
            });
        } else if rise >= regression_threshold {
            improvements.push(Improvement {
                case_id: case.id.clone(),
            });
        }
    }

    BaselineDiff {
        regressions,
        improvements,
    }
}

/// Pretty-print a `BaselineDiff` to stdout. Returns the number of regressions.
pub fn print_diff(diff: &BaselineDiff, suite_name: &str) -> usize {
    for reg in &diff.regressions {
        println!("REGRESSION  {:<30} was: PASS  now: FAIL", reg.case_id);
        for f in &reg.failures {
            println!("  ✗ {}", f);
        }
    }
    for imp in &diff.improvements {
        println!("improvement {:<30} was: FAIL  now: PASS", imp.case_id);
    }
    let n = diff.regressions.len();
    if n > 0 {
        println!(
            "Exit code 1 — {} regression{} vs baseline (suite: {})",
            n,
            if n == 1 { "" } else { "s" },
            suite_name
        );
    } else if !diff.improvements.is_empty() {
        println!(
            "No regressions vs baseline (suite: {}). {} improvement{}.",
            suite_name,
            diff.improvements.len(),
            if diff.improvements.len() == 1 {
                ""
            } else {
                "s"
            },
        );
    } else {
        println!("No regressions vs baseline (suite: {}).", suite_name);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{CaseResult, CheckResult, SuiteResult};

    fn make_suite(name: &str, cases: &[(&str, bool)]) -> SuiteResult {
        SuiteResult {
            suite_name: name.to_string(),
            model: "test-model".to_string(),
            score_pct: 100.0,
            cases: cases
                .iter()
                .map(|(id, passed)| CaseResult {
                    id: id.to_string(),
                    passed: *passed,
                    check_results: if *passed {
                        vec![]
                    } else {
                        vec![CheckResult {
                            check_desc: "contains: foo".to_string(),
                            passed: false,
                            message: Some("response does not contain \"foo\"".to_string()),
                        }]
                    },
                    response_text: "test response".to_string(),
                    total_ms: 0,
                    runs: 1,
                    pass_count: if *passed { 1 } else { 0 },
                    pass_rate: if *passed { 1.0 } else { 0.0 },
                    flaky: false,
                })
                .collect(),
        }
    }

    fn make_baseline(name: &str, cases: &[(&str, bool)]) -> Baseline {
        Baseline {
            suite_name: name.to_string(),
            model: "test-model".to_string(),
            cases: cases
                .iter()
                .map(|(id, passed)| BaselineCase {
                    id: id.to_string(),
                    passed: *passed,
                    pass_rate: None,
                })
                .collect(),
        }
    }

    #[test]
    fn diff_detects_regression() {
        let baseline = make_baseline("s", &[("case-a", true), ("case-b", true)]);
        let current = make_suite("s", &[("case-a", true), ("case-b", false)]);
        let diff = diff_results(&baseline, &current);
        assert_eq!(diff.regressions.len(), 1);
        assert_eq!(diff.regressions[0].case_id, "case-b");
        assert!(!diff.regressions[0].failures.is_empty());
    }

    #[test]
    fn diff_detects_improvement() {
        let baseline = make_baseline("s", &[("case-a", false)]);
        let current = make_suite("s", &[("case-a", true)]);
        let diff = diff_results(&baseline, &current);
        assert!(diff.regressions.is_empty());
        assert_eq!(diff.improvements.len(), 1);
        assert_eq!(diff.improvements[0].case_id, "case-a");
    }

    #[test]
    fn diff_clean_when_all_same() {
        let baseline = make_baseline("s", &[("a", true), ("b", false)]);
        let current = make_suite("s", &[("a", true), ("b", false)]);
        let diff = diff_results(&baseline, &current);
        assert!(diff.is_clean());
        assert!(diff.improvements.is_empty());
    }

    #[test]
    fn diff_new_case_not_a_regression() {
        let baseline = make_baseline("s", &[("a", true)]);
        // "b" is new — not in baseline
        let current = make_suite("s", &[("a", true), ("b", false)]);
        let diff = diff_results(&baseline, &current);
        assert!(
            diff.is_clean(),
            "new failing case should not be a regression"
        );
    }

    #[test]
    fn save_and_load_baseline_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let suite = make_suite("my-suite", &[("t1", true), ("t2", false)]);
        let path = save_baseline(dir.path(), &suite).unwrap();
        assert!(path.exists());

        let loaded = load_baseline(dir.path(), "my-suite").unwrap().unwrap();
        assert_eq!(loaded.suite_name, "my-suite");
        assert_eq!(loaded.cases.len(), 2);
        assert_eq!(
            loaded.cases[0],
            BaselineCase {
                id: "t1".into(),
                passed: true,
                pass_rate: None
            }
        );
        assert_eq!(
            loaded.cases[1],
            BaselineCase {
                id: "t2".into(),
                passed: false,
                pass_rate: None
            }
        );
    }

    #[test]
    fn load_baseline_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_baseline(dir.path(), "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn slugify_replaces_spaces_and_slashes() {
        assert_eq!(slugify("my suite/name"), "my_suite_name");
        assert_eq!(slugify("hello-world"), "hello-world");
        assert_eq!(slugify("abc 123"), "abc_123");
    }

    #[test]
    fn diff_results_rate_detects_rate_regression() {
        // Old pass_rate=1.0, new=0.0, threshold=0.1 → regression
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: true,
                pass_rate: Some(1.0),
            }],
        };
        let current = make_suite("s", &[("c1", false)]);
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert_eq!(
            diff.regressions.len(),
            1,
            "drop from 1.0 to 0.0 should be regression"
        );
        assert_eq!(diff.regressions[0].case_id, "c1");
    }

    #[test]
    fn diff_results_rate_detects_rate_improvement() {
        // Old pass_rate=0.0, new=1.0, threshold=0.1 → improvement
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: false,
                pass_rate: Some(0.0),
            }],
        };
        let current = make_suite("s", &[("c1", true)]);
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert!(diff.regressions.is_empty());
        assert_eq!(diff.improvements.len(), 1);
        assert_eq!(diff.improvements[0].case_id, "c1");
    }

    #[test]
    fn diff_results_rate_small_change_within_threshold_is_clean() {
        // Old pass_rate=0.8, new=0.75, threshold=0.1 → clean (drop < threshold)
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: true,
                pass_rate: Some(0.8),
            }],
        };
        let mut current = make_suite("s", &[("c1", true)]);
        // Manually set pass_rate on the case
        current.cases[0].pass_rate = 0.75;
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert!(
            diff.is_clean(),
            "small drop within threshold should be clean"
        );
    }

    #[test]
    fn print_diff_no_regressions_returns_zero() {
        let diff = BaselineDiff {
            regressions: Vec::new(),
            improvements: Vec::new(),
        };
        assert_eq!(print_diff(&diff, "my-suite"), 0);
    }

    #[test]
    fn print_diff_with_regression_returns_count() {
        let diff = BaselineDiff {
            regressions: vec![Regression {
                case_id: "case-a".into(),
                failures: vec![],
            }],
            improvements: Vec::new(),
        };
        assert_eq!(print_diff(&diff, "my-suite"), 1);
    }

    #[test]
    fn effective_rate_uses_pass_rate_when_present() {
        let case = BaselineCase {
            id: "c".into(),
            passed: false,
            pass_rate: Some(0.75),
        };
        assert!(
            (case.effective_rate() - 0.75).abs() < 1e-6,
            "expected 0.75, got {}",
            case.effective_rate()
        );
    }

    #[test]
    fn effective_rate_infers_from_passed_when_no_pass_rate() {
        let passing = BaselineCase {
            id: "c".into(),
            passed: true,
            pass_rate: None,
        };
        assert!(
            (passing.effective_rate() - 1.0).abs() < 1e-6,
            "passed=true should give 1.0"
        );
        let failing = BaselineCase {
            id: "c".into(),
            passed: false,
            pass_rate: None,
        };
        assert!(
            (failing.effective_rate() - 0.0).abs() < 1e-6,
            "passed=false should give 0.0"
        );
    }

    #[test]
    fn effective_rate_pass_rate_overrides_passed_bool() {
        // pass_rate=0.5 but passed=true — stored rate wins
        let case = BaselineCase {
            id: "c".into(),
            passed: true,
            pass_rate: Some(0.5),
        };
        assert!(
            (case.effective_rate() - 0.5).abs() < 1e-6,
            "pass_rate should override passed bool"
        );
    }

    #[test]
    fn slugify_dots_become_underscores() {
        // dots are not alphanumeric and not '-', so they become '_'
        assert_eq!(slugify("v1.2.3"), "v1_2_3");
        assert_eq!(slugify("test!case"), "test_case");
    }

    #[test]
    fn is_clean_returns_true_for_empty_regressions() {
        let diff = BaselineDiff {
            regressions: Vec::new(),
            improvements: vec![],
        };
        assert!(diff.is_clean());
    }

    #[test]
    fn is_clean_returns_false_when_regressions_present() {
        let diff = BaselineDiff {
            regressions: vec![Regression {
                case_id: "x".into(),
                failures: vec![],
            }],
            improvements: vec![],
        };
        assert!(!diff.is_clean());
    }

    #[test]
    fn baseline_dir_is_under_config_dir() {
        let dir = tempfile::tempdir().unwrap();
        let bdir = baseline_dir(dir.path());
        assert!(
            bdir.starts_with(dir.path()),
            "baseline_dir should be under config_dir"
        );
        assert!(bdir.to_string_lossy().contains("baselines"));
    }

    #[test]
    fn baseline_path_uses_slugified_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = baseline_path(dir.path(), "my suite");
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert_eq!(filename, "my_suite.json", "spaces should be slugified");
    }

    #[test]
    fn save_baseline_atomic_no_tmp_left() {
        let tmp = tempfile::tempdir().unwrap();
        let suite = make_suite("atomic-test", &[("c1", true)]);
        let path = save_baseline(tmp.path(), &suite).unwrap();
        assert!(path.exists(), "baseline file should exist");
        let tmp_path = path.with_extension("json.tmp");
        assert!(!tmp_path.exists(), "tmp file should be cleaned up");
        let loaded = load_baseline(tmp.path(), "atomic-test").unwrap();
        assert!(loaded.is_some(), "baseline should load after atomic save");
        assert_eq!(loaded.unwrap().cases.len(), 1);
    }

    #[test]
    fn diff_results_rate_boundary_regression() {
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: true,
                pass_rate: Some(1.0),
            }],
        };
        let mut current = make_suite("s", &[("c1", true)]);
        current.cases[0].pass_rate = 0.9;
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert_eq!(
            diff.regressions.len(),
            1,
            "exact threshold drop should be regression"
        );
    }

    #[test]
    fn diff_results_rate_boundary_improvement() {
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: false,
                pass_rate: Some(0.5),
            }],
        };
        let mut current = make_suite("s", &[("c1", true)]);
        current.cases[0].pass_rate = 0.6;
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert_eq!(
            diff.improvements.len(),
            1,
            "exact threshold rise should be improvement"
        );
    }

    #[test]
    fn diff_results_rate_below_threshold_no_regression() {
        let baseline = Baseline {
            suite_name: "s".into(),
            model: "m".into(),
            cases: vec![BaselineCase {
                id: "c1".into(),
                passed: true,
                pass_rate: Some(1.0),
            }],
        };
        let mut current = make_suite("s", &[("c1", true)]);
        current.cases[0].pass_rate = 0.95;
        let diff = diff_results_rate(&baseline, &current, 0.1);
        assert!(
            diff.is_clean(),
            "0.05 drop below 0.1 threshold should be clean"
        );
    }

    #[test]
    fn diff_case_removed_from_current_is_not_regression() {
        // A case present in baseline but absent from current is simply ignored
        let baseline = make_baseline("s", &[("case-a", true), ("case-b", true)]);
        let current = make_suite("s", &[("case-a", true)]); // case-b gone
        let diff = diff_results(&baseline, &current);
        assert!(
            diff.regressions.is_empty(),
            "removed case should not appear as regression"
        );
    }
}
