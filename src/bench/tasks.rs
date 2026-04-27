pub struct BenchTask {
    pub name: &'static str,
    pub prompt: &'static str,
}

pub fn all_tasks() -> Vec<BenchTask> {
    vec![
        BenchTask {
            name: "code_gen",
            prompt: "Write a Rust function that reverses a string in-place.",
        },
        BenchTask {
            name: "explain",
            prompt: "Explain what a mutex is in one paragraph.",
        },
        BenchTask {
            name: "bug_fix",
            prompt: "Fix this Python: `def add(a,b) return a+b`",
        },
        BenchTask {
            name: "refactor",
            prompt:
                "Refactor this into idiomatic Rust: `let mut v = Vec::new(); v.push(1); v.push(2);`",
        },
    ]
}

pub fn score_quality(task_name: &str, output: &str) -> bool {
    match task_name {
        "code_gen" => output.contains("fn") && output.contains('{') && output.contains('}'),
        "explain" => output.len() >= 100 && output.len() <= 600,
        "bug_fix" => output.contains("def") && output.contains(':'),
        "refactor" => output.contains("vec!"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_heuristic_code_gen_pass() {
        assert!(score_quality(
            "code_gen",
            "fn reverse(s: &mut String) { s.chars().rev() }"
        ));
    }

    #[test]
    fn quality_heuristic_code_gen_fail() {
        assert!(!score_quality("code_gen", "just some text without braces"));
    }

    #[test]
    fn quality_heuristic_explain_pass() {
        // Must be 100-600 chars
        let text = "A mutex is a synchronization primitive that provides mutual exclusion. \
                    It ensures that only one thread can access a shared resource at a time. \
                    When a thread acquires a mutex, others block until it is released.";
        assert!(text.len() >= 100 && text.len() <= 600);
        assert!(score_quality("explain", text));
    }

    #[test]
    fn quality_heuristic_explain_too_short() {
        assert!(!score_quality("explain", "Too short."));
    }

    #[test]
    fn quality_heuristic_explain_too_long() {
        let text = "x".repeat(601);
        assert!(!score_quality("explain", &text));
    }

    #[test]
    fn quality_heuristic_bug_fix_pass() {
        assert!(score_quality("bug_fix", "def add(a, b): return a + b"));
    }

    #[test]
    fn quality_heuristic_bug_fix_fail() {
        assert!(!score_quality("bug_fix", "just prose without def or colon"));
    }

    #[test]
    fn quality_heuristic_refactor_pass() {
        assert!(score_quality("refactor", "let v = vec![1, 2];"));
    }

    #[test]
    fn quality_heuristic_refactor_fail() {
        assert!(!score_quality(
            "refactor",
            "let mut v = Vec::new(); v.push(1);"
        ));
    }

    #[test]
    fn quality_heuristic_unknown_task_always_false() {
        assert!(!score_quality("unknown_task", "anything"));
        assert!(!score_quality("", ""));
    }

    #[test]
    fn all_tasks_returns_four_tasks() {
        let tasks = all_tasks();
        assert_eq!(tasks.len(), 4);
        let names: Vec<&str> = tasks.iter().map(|t| t.name).collect();
        assert!(names.contains(&"code_gen"));
        assert!(names.contains(&"explain"));
        assert!(names.contains(&"bug_fix"));
        assert!(names.contains(&"refactor"));
    }

    #[test]
    fn all_tasks_prompts_are_non_empty() {
        for task in all_tasks() {
            assert!(
                !task.prompt.is_empty(),
                "task '{}' has empty prompt",
                task.name
            );
        }
    }
}
