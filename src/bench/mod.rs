//! Model benchmarking and capability evaluation.
//!
//! Defines standardized coding tasks to evaluate LLM performance, track history,
//! and compare throughput/quality across different models.

pub mod compare;
pub mod runner;
pub mod tasks;
pub use runner::{print_history_detail, print_history_list, run_bench};
