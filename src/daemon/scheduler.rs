use std::path::Path;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScheduledTask {
    pub id: String,
    pub cron: String,
    pub prompt: String,
    pub model: Option<String>,
    pub last_run: Option<String>, // RFC3339 or None
    pub next_run: Option<String>, // RFC3339 or None
}

fn schedules_path(config_dir: &Path) -> std::path::PathBuf {
    config_dir.join("schedules.json")
}

pub fn load_schedules(config_dir: &Path) -> anyhow::Result<Vec<ScheduledTask>> {
    let path = schedules_path(config_dir);
    if !path.exists() {
        return Ok(vec![]);
    }
    let raw = std::fs::read_to_string(&path)?;
    let tasks: Vec<ScheduledTask> = serde_json::from_str(&raw)?;
    Ok(tasks)
}

pub fn save_schedules(config_dir: &Path, tasks: &[ScheduledTask]) -> anyhow::Result<()> {
    let path = schedules_path(config_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(tasks)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Generate a timestamp-based unique task ID.
pub fn generate_task_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{:x}{:04x}", t.as_millis(), count % 0xffff)
}

/// Returns tasks where `next_run` is `Some` and <= now (RFC3339).
/// Tasks with `next_run: None` are never returned.
pub fn next_due(tasks: &[ScheduledTask]) -> Vec<&ScheduledTask> {
    let now = chrono::Utc::now();
    tasks
        .iter()
        .filter(|t| {
            if let Some(ref nr) = t.next_run {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(nr) {
                    return dt <= now;
                }
            }
            false
        })
        .collect()
}

/// Advance `task.next_run` to the next time matching the cron expression.
/// Also sets `task.last_run` to now. Falls back to +1 day on parse failure.
pub fn advance_next_run(task: &mut ScheduledTask) {
    let now = chrono::Utc::now();
    task.last_run = Some(now.to_rfc3339());
    let next = next_cron_time(&task.cron, now).unwrap_or_else(|| now + chrono::Duration::days(1));
    task.next_run = Some(next.to_rfc3339());
}

/// Parse a 5-field cron expression and compute the next matching time after `after`.
/// Fields: minute hour day-of-month month day-of-week (0=Sun or 7=Sun).
/// Supports: `*`, single values, comma-separated lists, ranges (`1-5`), step (`*/15`).
/// Returns None on parse failure.
fn next_cron_time(
    cron: &str,
    after: chrono::DateTime<chrono::Utc>,
) -> Option<chrono::DateTime<chrono::Utc>> {
    use chrono::{Datelike, Timelike};
    let fields: Vec<&str> = cron.split_whitespace().collect();
    if fields.len() != 5 {
        return None;
    }
    let minutes = parse_cron_field(fields[0], 0, 59)?;
    let hours = parse_cron_field(fields[1], 0, 23)?;
    let doms = parse_cron_field(fields[2], 1, 31)?;
    let months = parse_cron_field(fields[3], 1, 12)?;
    let dows = parse_cron_field_dow(fields[4])?;

    // Start one minute after `after`
    let mut candidate = after + chrono::Duration::minutes(1);
    // Zero out seconds
    candidate = candidate.with_second(0).unwrap_or(candidate);

    // Search up to 366 days ahead to avoid infinite loops
    let limit = after + chrono::Duration::days(366);
    while candidate < limit {
        if !months.contains(&(candidate.month() as u8)) {
            // Skip to first day of next month
            candidate = advance_to_next_month(candidate);
            continue;
        }
        if !doms.contains(&(candidate.day() as u8)) {
            candidate += chrono::Duration::days(1);
            candidate = candidate.with_hour(0)?.with_minute(0)?;
            continue;
        }
        // day-of-week: chrono uses Mon=0..Sun=6 for weekday().num_days_from_monday(),
        // but cron uses 0=Sun,1=Mon,...,6=Sat
        let cron_dow = candidate.weekday().num_days_from_sunday() as u8;
        if !dows.contains(&cron_dow) {
            candidate += chrono::Duration::days(1);
            candidate = candidate.with_hour(0)?.with_minute(0)?;
            continue;
        }
        if !hours.contains(&(candidate.hour() as u8)) {
            candidate += chrono::Duration::hours(1);
            candidate = candidate.with_minute(0)?;
            continue;
        }
        if !minutes.contains(&(candidate.minute() as u8)) {
            candidate += chrono::Duration::minutes(1);
            continue;
        }
        return Some(candidate);
    }
    None // No match within 366 days
}

fn advance_to_next_month(dt: chrono::DateTime<chrono::Utc>) -> chrono::DateTime<chrono::Utc> {
    use chrono::Datelike;
    use chrono::Timelike;
    let (y, m) = if dt.month() == 12 {
        (dt.year() + 1, 1)
    } else {
        (dt.year(), dt.month() + 1)
    };
    dt.with_year(y)
        .and_then(|d| d.with_month(m))
        .and_then(|d| d.with_day(1))
        .and_then(|d| d.with_hour(0))
        .and_then(|d| d.with_minute(0))
        .and_then(|d| d.with_second(0))
        .unwrap_or(dt + chrono::Duration::days(31))
}

/// Parse a single cron field into a set of allowed values.
/// Supports: `*`, `*/step`, `N`, `N-M`, `N-M/step`, comma-separated combos.
fn parse_cron_field(field: &str, min: u8, max: u8) -> Option<Vec<u8>> {
    let mut result = Vec::new();
    for part in field.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        if let Some(step_part) = part.strip_prefix("*/") {
            let step: u8 = step_part.parse().ok()?;
            if step == 0 {
                return None;
            }
            if step > max - min {
                return None;
            }
            let mut v = min;
            while v <= max {
                result.push(v);
                v = v.checked_add(step)?;
            }
        } else if part == "*" {
            for v in min..=max {
                result.push(v);
            }
        } else if part.contains('-') {
            let (range_str, step) = if part.contains('/') {
                let mut sp = part.splitn(2, '/');
                let r = sp.next()?;
                let s: u8 = sp.next()?.parse().ok()?;
                (r, s)
            } else {
                (part, 1)
            };
            let mut bounds = range_str.splitn(2, '-');
            let lo: u8 = bounds.next()?.parse().ok()?;
            let hi: u8 = bounds.next()?.parse().ok()?;
            if lo < min || hi > max || lo > hi {
                return None;
            }
            if step == 0 {
                return None;
            }
            if step > hi - lo {
                return None;
            }
            let mut v = lo;
            while v <= hi {
                result.push(v);
                v = v.checked_add(step)?;
            }
        } else {
            let v: u8 = part.parse().ok()?;
            if v < min || v > max {
                return None;
            }
            result.push(v);
        }
    }
    result.sort_unstable();
    result.dedup();
    Some(result)
}

/// Parse day-of-week field. Like `parse_cron_field(0,7)` but normalizes 7 → 0 (both mean Sunday).
fn parse_cron_field_dow(field: &str) -> Option<Vec<u8>> {
    let mut vals = parse_cron_field(field, 0, 7)?;
    // Normalize: 7 means Sunday (same as 0)
    for v in &mut vals {
        if *v == 7 {
            *v = 0;
        }
    }
    vals.sort_unstable();
    vals.dedup();
    Some(vals)
}

/// Parse `"<cron5> <prompt>"` where `<cron5>` is the first 5 whitespace-separated tokens.
/// Returns `(cron_expr, prompt)`.
pub fn parse_schedule_add(input: &str) -> anyhow::Result<(String, String)> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    if tokens.len() < 6 {
        anyhow::bail!(
            "schedule-add requires at least 6 tokens: <min> <hour> <dom> <month> <dow> <prompt...>"
        );
    }
    let cron = tokens[..5].join(" ");
    let prompt = tokens[5..].join(" ");
    Ok((cron, prompt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(next_run: Option<&str>) -> ScheduledTask {
        ScheduledTask {
            id: generate_task_id(),
            cron: "0 9 * * 1-5".to_string(),
            prompt: "summarize git log".to_string(),
            model: None,
            last_run: None,
            next_run: next_run.map(|s| s.to_string()),
        }
    }

    #[test]
    fn next_due_returns_overdue_tasks() {
        // past time: 2000-01-01T00:00:00Z is definitely in the past
        let t = make_task(Some("2000-01-01T00:00:00Z"));
        let tasks = vec![t];
        let due = next_due(&tasks);
        assert_eq!(due.len(), 1, "overdue task should be returned");
    }

    #[test]
    fn next_due_skips_future_tasks() {
        // year 3000 is definitely in the future
        let t = make_task(Some("3000-01-01T00:00:00Z"));
        let tasks = vec![t];
        let due = next_due(&tasks);
        assert_eq!(due.len(), 0, "future task must not be returned");
    }

    #[test]
    fn next_due_skips_none_next_run() {
        let t = make_task(None);
        let tasks = vec![t];
        let due = next_due(&tasks);
        assert_eq!(due.len(), 0, "task with no next_run must never be due");
    }

    #[test]
    fn schedule_roundtrips_json() {
        let t = ScheduledTask {
            id: "abc123".to_string(),
            cron: "0 9 * * *".to_string(),
            prompt: "hello world".to_string(),
            model: Some("gemma4:26b".to_string()),
            last_run: None,
            next_run: Some("2025-01-01T09:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&t).expect("serialize");
        let t2: ScheduledTask = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(t.id, t2.id);
        assert_eq!(t.cron, t2.cron);
        assert_eq!(t.prompt, t2.prompt);
        assert_eq!(t.model, t2.model);
        assert_eq!(t.next_run, t2.next_run);
    }

    #[test]
    fn schedule_id_is_unique() {
        let a = generate_task_id();
        let b = generate_task_id();
        let c = generate_task_id();
        assert_ne!(a, b, "IDs must be distinct");
        assert_ne!(b, c, "IDs must be distinct");
        assert_ne!(a, c, "IDs must be distinct");
    }

    #[test]
    fn advance_weekday_task_schedules_future_weekday_at_9am() {
        use chrono::{Datelike, Timelike};
        let mut t = make_task(Some("2000-01-01T00:00:00Z"));
        let before = chrono::Utc::now();
        advance_next_run(&mut t);
        let next_str = t
            .next_run
            .as_ref()
            .expect("next_run must be set after advance");
        let next_dt =
            chrono::DateTime::parse_from_rfc3339(next_str).expect("next_run must be valid RFC3339");
        assert!(
            next_dt > before,
            "next_run after advance must be in the future"
        );
        // Cron is "0 9 * * 1-5" — should be 9:00 on a weekday
        assert_eq!(next_dt.hour(), 9);
        assert_eq!(next_dt.minute(), 0);
        let dow = next_dt.weekday().num_days_from_monday();
        assert!(dow < 5, "should be a weekday (Mon-Fri), got {}", dow);
    }

    #[test]
    fn parse_schedule_add_splits_correctly() {
        let input = "0 9 * * 1-5 summarize git log for today";
        let (cron, prompt) = parse_schedule_add(input).expect("parse");
        assert_eq!(cron, "0 9 * * 1-5");
        assert_eq!(prompt, "summarize git log for today");
    }

    #[test]
    fn parse_schedule_add_single_word_prompt() {
        let input = "0 0 * * * cleanup";
        let (cron, prompt) = parse_schedule_add(input).expect("parse");
        assert_eq!(cron, "0 0 * * *");
        assert_eq!(prompt, "cleanup");
    }

    #[test]
    fn parse_schedule_add_too_few_tokens() {
        let result = parse_schedule_add("0 9 * *");
        assert!(result.is_err(), "should fail with fewer than 6 tokens");
    }

    #[test]
    fn parse_schedule_add_exactly_six_tokens() {
        // Minimum valid input: 5 cron fields + 1 prompt word
        let (cron, prompt) = parse_schedule_add("* * * * * go").expect("should parse");
        assert_eq!(cron, "* * * * *");
        assert_eq!(prompt, "go");
    }

    #[test]
    fn advance_sets_last_run() {
        let mut t = make_task(None);
        assert!(t.last_run.is_none(), "last_run should start as None");
        advance_next_run(&mut t);
        assert!(t.last_run.is_some(), "last_run should be set after advance");
        // Verify last_run is valid RFC3339
        let lr = t.last_run.as_ref().unwrap();
        chrono::DateTime::parse_from_rfc3339(lr).expect("last_run must be valid RFC3339");
    }

    #[test]
    fn next_due_invalid_rfc3339_skipped() {
        let mut t = make_task(None);
        t.next_run = Some("not-a-date".to_string());
        let tasks = vec![t];
        let due = next_due(&tasks);
        assert_eq!(due.len(), 0, "invalid RFC3339 should be treated as not-due");
    }

    #[test]
    fn load_schedules_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let tasks = load_schedules(dir.path()).expect("should not error on missing file");
        assert!(tasks.is_empty(), "no schedules.json → empty vec");
    }

    #[test]
    fn save_and_load_schedules_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let tasks = vec![ScheduledTask {
            id: "id1".to_string(),
            cron: "0 9 * * *".to_string(),
            prompt: "hello".to_string(),
            model: None,
            last_run: None,
            next_run: Some("2025-06-01T09:00:00Z".to_string()),
        }];
        save_schedules(dir.path(), &tasks).expect("save should succeed");
        let loaded = load_schedules(dir.path()).expect("load should succeed");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "id1");
        assert_eq!(loaded[0].prompt, "hello");
    }

    // ── Cron expression parsing tests ────────────────────────────────────────

    #[test]
    fn parse_cron_field_star() {
        let vals = parse_cron_field("*", 0, 5).unwrap();
        assert_eq!(vals, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn parse_cron_field_star_step() {
        let vals = parse_cron_field("*/15", 0, 59).unwrap();
        assert_eq!(vals, vec![0, 15, 30, 45]);
    }

    #[test]
    fn parse_cron_field_single() {
        let vals = parse_cron_field("5", 0, 59).unwrap();
        assert_eq!(vals, vec![5]);
    }

    #[test]
    fn parse_cron_field_range() {
        let vals = parse_cron_field("1-5", 0, 7).unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn parse_cron_field_range_with_step() {
        let vals = parse_cron_field("0-23/6", 0, 23).unwrap();
        assert_eq!(vals, vec![0, 6, 12, 18]);
    }

    #[test]
    fn parse_cron_field_comma_list() {
        let vals = parse_cron_field("1,15,30", 0, 59).unwrap();
        assert_eq!(vals, vec![1, 15, 30]);
    }

    #[test]
    fn parse_cron_field_dow_normalizes_7_to_0() {
        let vals = parse_cron_field_dow("7").unwrap();
        assert_eq!(vals, vec![0], "7 should normalize to 0 (Sunday)");
    }

    #[test]
    fn parse_cron_field_dow_range_1_5() {
        let vals = parse_cron_field_dow("1-5").unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn parse_cron_field_invalid_returns_none() {
        assert!(parse_cron_field("abc", 0, 59).is_none());
        assert!(parse_cron_field("*/0", 0, 59).is_none());
        assert!(parse_cron_field("", 0, 59).is_none());
    }

    /// `*/N` with N larger than the field's range collapses to `[min]` —
    /// the user almost certainly meant something else (e.g. `*/120` for
    /// minutes is meaningless on a 0-59 field). Reject rather than silently
    /// scheduling on minute 0 only. Same idea for `lo-hi/step` when step
    /// exceeds the range size.
    #[test]
    fn parse_cron_field_rejects_step_larger_than_range() {
        // `*/N` for minutes (range = 59): 60 and above should reject.
        assert!(
            parse_cron_field("*/60", 0, 59).is_none(),
            "*/60 minutes is meaningless (range = 59); should reject"
        );
        assert!(parse_cron_field("*/120", 0, 59).is_none());
        assert!(parse_cron_field("*/255", 0, 59).is_none());
        // Hours (range = 23): step ≥ 24 should reject.
        assert!(
            parse_cron_field("*/24", 0, 23).is_none(),
            "*/24 hours is meaningless (range = 23); should reject"
        );
        // Range form with oversized step: also reject.
        assert!(
            parse_cron_field("0-23/24", 0, 23).is_none(),
            "0-23/24 is meaningless (range = 23); should reject"
        );
        assert!(parse_cron_field("0-59/60", 0, 59).is_none());

        // Boundary: step == range is the largest valid step that still
        // produces two values (min and the next-after); accept it.
        // `*/59` minutes → [0, 59].
        let vals = parse_cron_field("*/59", 0, 59).expect("step == range stays valid");
        assert_eq!(vals, vec![0, 59]);
    }

    #[test]
    fn parse_cron_field_comma_list_with_invalid_part_returns_none() {
        // A valid first part must not mask a later invalid part — the whole
        // expression has to be rejected so `next_cron_time` doesn't silently
        // schedule on a half-parsed field.
        assert!(parse_cron_field("5,abc", 0, 59).is_none());
        assert!(parse_cron_field("5,", 0, 59).is_none());
        assert!(parse_cron_field(",5", 0, 59).is_none());
        // Range followed by a garbage part.
        assert!(parse_cron_field("0-5,xyz", 0, 59).is_none());
        // Step followed by a garbage part.
        assert!(parse_cron_field("*/15,nope", 0, 59).is_none());
    }

    #[test]
    fn parse_cron_field_rejects_out_of_range_values() {
        // Single value out of range
        assert!(
            parse_cron_field("60", 0, 59).is_none(),
            "60 exceeds minute max 59"
        );
        assert!(
            parse_cron_field("25", 0, 23).is_none(),
            "25 exceeds hour max 23"
        );
        assert!(
            parse_cron_field("0", 1, 12).is_none(),
            "0 below month min 1"
        );
        assert!(
            parse_cron_field("13", 1, 12).is_none(),
            "13 exceeds month max 12"
        );
        assert!(
            parse_cron_field("32", 1, 31).is_none(),
            "32 exceeds dom max 31"
        );
        // Range with out-of-range bounds
        assert!(
            parse_cron_field("0-60", 0, 59).is_none(),
            "range hi exceeds max"
        );
        assert!(
            parse_cron_field("0-12", 1, 12).is_none(),
            "range lo below min"
        );
        // Inverted range
        assert!(
            parse_cron_field("10-5", 0, 59).is_none(),
            "inverted range lo > hi"
        );
        // Comma list with one out-of-range value
        assert!(
            parse_cron_field("1,99", 0, 59).is_none(),
            "99 in comma list exceeds max"
        );
    }

    #[test]
    fn next_cron_time_every_minute() {
        use chrono::Timelike;
        let base = chrono::DateTime::parse_from_rfc3339("2024-06-15T10:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next = next_cron_time("* * * * *", base).unwrap();
        // Should be 10:31:00
        assert_eq!(next.minute(), 31);
        assert_eq!(next.hour(), 10);
    }

    #[test]
    fn next_cron_time_daily_at_9am() {
        use chrono::Timelike;
        let base = chrono::DateTime::parse_from_rfc3339("2024-06-15T10:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next = next_cron_time("0 9 * * *", base).unwrap();
        // Already past 9am today, so next is tomorrow 9:00
        assert_eq!(next.hour(), 9);
        assert_eq!(next.minute(), 0);
        use chrono::Datelike;
        assert_eq!(next.day(), 16);
    }

    #[test]
    fn next_cron_time_weekdays_only() {
        use chrono::Datelike;
        // 2024-06-15 is Saturday
        let base = chrono::DateTime::parse_from_rfc3339("2024-06-15T08:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next = next_cron_time("0 9 * * 1-5", base).unwrap();
        // Next weekday is Monday June 17
        assert_eq!(next.weekday(), chrono::Weekday::Mon);
        assert_eq!(next.day(), 17);
    }

    #[test]
    fn next_cron_time_specific_months() {
        use chrono::Datelike;
        // June, looking for January
        let base = chrono::DateTime::parse_from_rfc3339("2024-06-15T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next = next_cron_time("0 0 1 1 *", base).unwrap();
        assert_eq!(next.month(), 1);
        assert_eq!(next.year(), 2025);
    }

    /// `next_cron_time` operates exclusively in `DateTime<Utc>` — UTC has no
    /// DST, so the spring-forward (non-existent local minute) and fall-back
    /// (twice-existent local minute) edge cases that plague local-time cron
    /// implementations don't apply here. This test pins that contract:
    /// passing UTC times equivalent to local-DST transitions returns a sane
    /// forward-monotonic result with no panic, no skipped day, no double-fire.
    ///
    /// If a future refactor moves any of `next_cron_time` /
    /// `advance_to_next_month` to a DST-bearing timezone (e.g. local time
    /// for "2 AM in user's TZ" semantics), the new code path needs explicit
    /// handling for the spring-forward gap and fall-back duplicate.
    #[test]
    fn next_cron_time_uses_utc_and_is_dst_immune() {
        // US 2026 spring-forward: 2026-03-08 02:00 EST → 03:00 EDT.
        // The corresponding UTC range is contiguous (06:00 → 07:00 UTC,
        // continuous tick). Pick a UTC moment that, if interpreted as
        // US/Eastern, would land in the non-existent 02:30 local slot.
        let spring_base = chrono::DateTime::parse_from_rfc3339("2026-03-08T06:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // Daily-at-06:30 schedule. Because base is exactly 06:30 UTC and
        // the search starts at base+1min, the next match is the SAME UTC
        // time on the next day — a clean 24-hour forward jump.
        let next = next_cron_time("30 6 * * *", spring_base)
            .expect("next match must exist within 366 days");
        let delta = next - spring_base;
        assert_eq!(
            delta,
            chrono::Duration::hours(24),
            "spring-forward UTC should produce a clean 24h forward jump (got {})",
            delta
        );

        // US 2026 fall-back: 2026-11-01 02:00 EDT → 01:00 EST.
        // The corresponding UTC range is also contiguous (05:00 → 06:00 UTC).
        // Pick a UTC moment that, if interpreted as US/Eastern, would land
        // in the doubly-existent 01:30 local slot (05:30 UTC = 01:30 EDT
        // and 06:30 UTC = 01:30 EST). UTC fires each instance exactly once.
        let fallback_base = chrono::DateTime::parse_from_rfc3339("2026-11-01T05:30:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next_fb = next_cron_time("30 5 * * *", fallback_base)
            .expect("next match must exist within 366 days");
        let delta_fb = next_fb - fallback_base;
        assert_eq!(
            delta_fb,
            chrono::Duration::hours(24),
            "fall-back UTC should produce a clean 24h forward jump (got {})",
            delta_fb
        );

        // Sanity: monotonicity — both returns are strictly after their input.
        assert!(next > spring_base);
        assert!(next_fb > fallback_base);
    }

    #[test]
    fn next_cron_time_invalid_expr_returns_none() {
        let base = chrono::Utc::now();
        assert!(next_cron_time("bad", base).is_none());
        assert!(next_cron_time("* * *", base).is_none());
    }

    #[test]
    fn next_cron_time_returns_none_for_impossible_schedule() {
        // Feb 31 never exists. Schedule is grammatically valid (parses
        // cleanly) but no calendar day satisfies it, so the 366-day
        // search must exhaust and return None rather than loop forever
        // or fall through silently. Closes the line-130 "No match
        // within 366 days" branch.
        let base = chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(next_cron_time("0 0 31 2 *", base).is_none());
        // Apr 31 also never exists.
        assert!(next_cron_time("0 0 31 4 *", base).is_none());
    }

    #[test]
    fn advance_next_run_uses_cron() {
        use chrono::Timelike;
        let mut t = ScheduledTask {
            id: "test".to_string(),
            cron: "30 14 * * *".to_string(), // daily at 14:30
            prompt: "test".to_string(),
            model: None,
            last_run: None,
            next_run: None,
        };
        advance_next_run(&mut t);
        let next = t.next_run.as_ref().expect("next_run should be set");
        let dt = chrono::DateTime::parse_from_rfc3339(next).expect("valid RFC3339");
        assert_eq!(dt.hour(), 14);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn next_cron_time_every_15_minutes() {
        use chrono::Timelike;
        let base = chrono::DateTime::parse_from_rfc3339("2024-06-15T10:07:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let next = next_cron_time("*/15 * * * *", base).unwrap();
        assert_eq!(next.minute(), 15);
        assert_eq!(next.hour(), 10);
    }
}
