//! YAML-driven model evaluation suites.
//!
//! Defines suite/case/check result types and inline deterministic checks.
//! `runner` executes prompts against configured models; `baseline` stores
//! comparison data for regressions, flake detection, and quality tracking.

pub mod baseline;
pub mod runner;

use serde::{Deserialize, Serialize};

/// A full evaluation suite loaded from a YAML file.
#[derive(Debug, Deserialize)]
pub struct EvalSuite {
    pub name: String,
    /// Optional model override — CLI `--model` takes precedence.
    pub model: Option<String>,
    /// Tool names to expose to the model (reserved for future use).
    #[serde(default)]
    #[allow(dead_code)]
    pub tools: Vec<String>,
    #[serde(default)]
    pub system: String,
    pub cases: Vec<EvalCase>,
}

/// A single test case within a suite.
#[derive(Debug, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub checks: Vec<Check>,
}

/// Intermediate map used for deserialization of Check from YAML map form.
#[derive(Debug, Deserialize)]
struct CheckMap {
    contains: Option<String>,
    not_contains: Option<String>,
    matches: Option<String>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    llm_judge: Option<String>,
}

/// A single correctness check applied to the model's response.
/// Deserialized from YAML maps like `{ contains: "foo" }`.
#[derive(Debug)]
pub enum Check {
    Contains(String),
    NotContains(String),
    Matches(String),
    MinLength(usize),
    MaxLength(usize),
    LlmJudge(String),
}

impl<'de> Deserialize<'de> for Check {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let m = CheckMap::deserialize(deserializer)?;
        if let Some(v) = m.contains {
            Ok(Check::Contains(v))
        } else if let Some(v) = m.not_contains {
            Ok(Check::NotContains(v))
        } else if let Some(v) = m.matches {
            Ok(Check::Matches(v))
        } else if let Some(v) = m.min_length {
            Ok(Check::MinLength(v))
        } else if let Some(v) = m.max_length {
            Ok(Check::MaxLength(v))
        } else if let Some(v) = m.llm_judge {
            Ok(Check::LlmJudge(v))
        } else {
            Err(serde::de::Error::custom(
                "check must have one of: contains, not_contains, matches, min_length, max_length, llm_judge",
            ))
        }
    }
}

/// Result of evaluating one Check.
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckResult {
    pub check_desc: String,
    pub passed: bool,
    pub message: Option<String>,
}

/// Result of running one `EvalCase` (single run or aggregate of N runs).
#[derive(Debug, Serialize, Deserialize)]
pub struct CaseResult {
    pub id: String,
    /// True when every run passed (`pass_count` == runs).
    pub passed: bool,
    /// Representative check breakdown (from the last run when runs > 1).
    pub check_results: Vec<CheckResult>,
    pub response_text: String,
    pub total_ms: u64,
    /// Number of runs executed (1 for legacy/single-run results).
    #[serde(default = "default_one_usize")]
    pub runs: usize,
    /// Number of runs that passed.
    #[serde(default = "default_one_if_passed")]
    pub pass_count: usize,
    /// Pass rate across all runs (0.0–1.0).
    #[serde(default)]
    pub pass_rate: f32,
    /// True when neither always passing nor always failing (depends on threshold).
    #[serde(default)]
    pub flaky: bool,
}

fn default_one_usize() -> usize {
    1
}
fn default_one_if_passed() -> usize {
    1
} // fine for deserializing legacy — pass_rate is authoritative

/// Returns true when `pass_rate` is between the stable extremes defined by `threshold`.
/// - threshold 1.0: flaky = rate > 0.0 && rate < 1.0
/// - threshold 0.8: flaky = rate > 0.2 && rate < 0.8
pub fn is_flaky(pass_rate: f32, threshold: f32) -> bool {
    pass_rate > (1.0 - threshold) && pass_rate < threshold
}

/// Aggregate result for an entire suite run.
#[derive(Debug, Serialize, Deserialize)]
pub struct SuiteResult {
    pub suite_name: String,
    pub model: String,
    pub cases: Vec<CaseResult>,
    pub score_pct: f64,
}

/// Evaluate a single non-LLM check against `response`.
/// Returns `None` for `LlmJudge` (handled in runner via a second call).
pub fn evaluate_check_inline(check: &Check, response: &str) -> Option<CheckResult> {
    match check {
        Check::Contains(needle) => {
            let passed = response.contains(needle.as_str());
            Some(CheckResult {
                check_desc: format!("contains: {:?}", needle),
                passed,
                message: if passed {
                    None
                } else {
                    Some(format!("response does not contain {:?}", needle))
                },
            })
        }
        Check::NotContains(needle) => {
            let passed = !response.contains(needle.as_str());
            Some(CheckResult {
                check_desc: format!("not_contains: {:?}", needle),
                passed,
                message: if passed {
                    None
                } else {
                    Some(format!("response contains {:?} (should not)", needle))
                },
            })
        }
        Check::Matches(pattern) => {
            let result = regex::Regex::new(pattern);
            match result {
                Ok(re) => {
                    let passed = re.is_match(response);
                    Some(CheckResult {
                        check_desc: format!("matches: {:?}", pattern),
                        passed,
                        message: if passed {
                            None
                        } else {
                            Some(format!("response does not match regex {:?}", pattern))
                        },
                    })
                }
                Err(e) => Some(CheckResult {
                    check_desc: format!("matches: {:?}", pattern),
                    passed: false,
                    message: Some(format!("invalid regex {:?}: {}", pattern, e)),
                }),
            }
        }
        Check::MinLength(min) => {
            let len = response.len();
            let passed = len >= *min;
            Some(CheckResult {
                check_desc: format!("min_length: {}", min),
                passed,
                message: if passed {
                    None
                } else {
                    Some(format!("got length {}, need at least {}", len, min))
                },
            })
        }
        Check::MaxLength(max) => {
            let len = response.len();
            let passed = len <= *max;
            Some(CheckResult {
                check_desc: format!("max_length: {}", max),
                passed,
                message: if passed {
                    None
                } else {
                    Some(format!("got {}, limit {}", len, max))
                },
            })
        }
        Check::LlmJudge(_) => None, // handled by runner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_contains_passes() {
        let check = Check::Contains("fn ".to_string());
        let result = evaluate_check_inline(&check, "pub fn hello() {}").unwrap();
        assert!(result.passed);
        assert!(result.message.is_none());
    }

    #[test]
    fn check_contains_fails() {
        let check = Check::Contains("fn ".to_string());
        let result = evaluate_check_inline(&check, "let x = 1;").unwrap();
        assert!(!result.passed);
        assert!(result.message.is_some());
    }

    #[test]
    fn check_matches_regex_passes() {
        let check = Check::Matches(r"fn \w+\(.*\)".to_string());
        let result = evaluate_check_inline(&check, "fn reverse(s: &str) -> String {").unwrap();
        assert!(result.passed);
    }

    #[test]
    fn check_matches_regex_fails() {
        let check = Check::Matches(r"fn \w+\(.*\)".to_string());
        let result = evaluate_check_inline(&check, "let x = 42;").unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn check_min_length_boundary() {
        let check_pass = Check::MinLength(5);
        let result_pass = evaluate_check_inline(&check_pass, "hello").unwrap();
        assert!(result_pass.passed, "exact boundary should pass");

        let check_fail = Check::MinLength(6);
        let result_fail = evaluate_check_inline(&check_fail, "hello").unwrap();
        assert!(!result_fail.passed, "one-under boundary should fail");
    }

    #[test]
    fn eval_suite_roundtrips_yaml() {
        let yaml = r#"
name: test-suite
model: gemma4:26b
tools: []
system: ""
cases:
  - id: case-1
    prompt: "Write a hello world in Rust."
    checks:
      - contains: "fn main"
      - not_contains: "TODO"
      - min_length: 20
      - max_length: 500
"#;
        let suite: EvalSuite = serde_yaml::from_str(yaml).expect("parse failed");
        assert_eq!(suite.name, "test-suite");
        assert_eq!(suite.cases.len(), 1);
        assert_eq!(suite.cases[0].id, "case-1");
        assert_eq!(suite.cases[0].checks.len(), 4);

        // Re-serialize the result types (EvalSuite itself is Deserialize only)
        let sr = SuiteResult {
            suite_name: suite.name.clone(),
            model: "gemma4:26b".to_string(),
            cases: vec![],
            score_pct: 100.0,
        };
        let json = serde_json::to_string(&sr).expect("serialize failed");
        let back: SuiteResult = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.suite_name, "test-suite");
        assert!((back.score_pct - 100.0).abs() < 1e-9);
    }

    // ── variance / flaky detection tests ──────────────────────────────────────

    fn make_case(pass_count: usize, runs: usize) -> CaseResult {
        let pass_rate = pass_count as f32 / runs as f32;
        CaseResult {
            id: "x".to_string(),
            passed: pass_count == runs,
            check_results: vec![],
            response_text: String::new(),
            total_ms: 0,
            runs,
            pass_count,
            pass_rate,
            flaky: false,
        }
    }

    #[test]
    fn multi_run_pass_rate_calculation() {
        let c = make_case(7, 10);
        assert!((c.pass_rate - 0.7).abs() < 1e-5);
        assert!(!c.passed); // not all 10 passed
        assert_eq!(c.pass_count, 7);
        assert_eq!(c.runs, 10);
    }

    #[test]
    fn flaky_detection_default_threshold() {
        // default threshold 1.0: flaky = rate not 0% and not 100%
        assert!(is_flaky(0.8, 1.0), "80% should be flaky");
        assert!(is_flaky(0.5, 1.0), "50% should be flaky");
        assert!(is_flaky(0.01, 1.0), "1% should be flaky");
        assert!(!is_flaky(0.0, 1.0), "0% is stable-fail");
        assert!(!is_flaky(1.0, 1.0), "100% is stable-pass");
    }

    #[test]
    fn flaky_detection_custom_threshold() {
        // threshold 0.8: flaky = rate > 0.2 && rate < 0.8
        assert!(is_flaky(0.5, 0.8), "50% is flaky at 0.8 threshold");
        assert!(
            !is_flaky(0.15, 0.8),
            "15% is not flaky — below (1-0.8)=0.2 band"
        );
        assert!(!is_flaky(0.85, 0.8), "85% is not flaky — above 0.8 band");
        assert!(!is_flaky(0.0, 0.8), "0% stable-fail");
        assert!(!is_flaky(1.0, 0.8), "100% stable-pass");
    }

    #[test]
    fn stable_all_pass_not_flagged() {
        // 10/10 runs — should never be flaky regardless of threshold
        assert!(!is_flaky(1.0, 1.0));
        assert!(!is_flaky(1.0, 0.8));
        assert!(!is_flaky(1.0, 0.5));
    }

    #[test]
    fn stable_all_fail_not_flagged() {
        // 0/10 runs — should never be flaky regardless of threshold
        assert!(!is_flaky(0.0, 1.0));
        assert!(!is_flaky(0.0, 0.8));
        assert!(!is_flaky(0.0, 0.5));
    }

    #[test]
    fn json_output_includes_pass_rate() {
        let c = make_case(8, 10);
        let json = serde_json::to_string(&c).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["runs"].as_u64(), Some(10));
        assert_eq!(v["pass_count"].as_u64(), Some(8));
        let rate = v["pass_rate"].as_f64().expect("pass_rate present");
        assert!((rate - 0.8).abs() < 1e-4);
        assert_eq!(v["flaky"].as_bool(), Some(false));
    }
}
