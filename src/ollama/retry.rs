//! Exponential backoff helper for Ollama transient-error retries.
//!
//! Shared pure function so the TUI mid-session retry loop and any future
//! retry sites compute the same delay schedule. `2^attempt * base_ms`,
//! capped at `cap_ms`, with saturating math so large `attempt` values
//! can't panic or wrap.

/// Compute the backoff delay (in ms) for the Nth retry attempt (0-indexed:
/// `attempt=0` is the first retry after the initial failure).
///
/// Shifts are clamped to 20 so `1u64 << attempt` can't overflow; the
/// result is then capped at `cap_ms` so the caller gets a bounded wait
/// even if `base_ms * 2^attempt` would exceed it.
pub fn next_backoff_ms(attempt: u32, base_ms: u64, cap_ms: u64) -> u64 {
    base_ms.saturating_mul(1u64 << attempt.min(20)).min(cap_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_backoff_attempt_zero_returns_base() {
        assert_eq!(next_backoff_ms(0, 500, 30_000), 500);
    }

    #[test]
    fn next_backoff_attempt_one_doubles_base() {
        assert_eq!(next_backoff_ms(1, 500, 30_000), 1_000);
    }

    #[test]
    fn next_backoff_attempt_three_is_eight_times_base() {
        assert_eq!(next_backoff_ms(3, 500, 30_000), 4_000);
    }

    #[test]
    fn next_backoff_saturates_at_cap_for_large_attempts() {
        // attempt=100 would overflow a naive 1<<attempt; the .min(20) clamp
        // keeps the shift in range and the .min(cap_ms) pins the result.
        assert_eq!(next_backoff_ms(100, 500, 30_000), 30_000);
    }
}
