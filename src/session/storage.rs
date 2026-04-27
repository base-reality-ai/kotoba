use super::Session;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// Serde contract: fields populated from session JSON on disk. Callers read
// only `id` + `updated_at` today; other fields are retained for schema
// stability and future consumers.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SessionMeta {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub cwd: String,
    #[serde(default)]
    pub host_project: Option<String>,
    pub model: String,
    #[serde(default)]
    pub turn_count: u32,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

/// Count the number of user messages in a conversation — used as a
/// display-time fallback when a session has no persisted `turn_count`
/// (older sessions written before that field existed).
pub fn count_user_messages(messages: &[serde_json::Value]) -> u32 {
    messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .count() as u32
}

/// Render a token count for the `/sessions` listing.
///
/// - Under 1 000: exact number ("0", "500")
/// - 1 000 – 999 999: one-decimal "k" ("12.3k", "1.0k")
/// - ≥ 1 000 000: one-decimal "M" ("1.2M")
pub fn human_tokens(n: u64) -> String {
    if n < 1_000 {
        format!("{}", n)
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

/// A session that matched a search query, with relevance score and a text snippet.
pub struct SessionMatch {
    pub session: Session,
    pub score: usize,
    pub snippet: String,
}

/// Search all saved sessions for `query`. Returns up to 10 matches sorted by score descending.
/// Score = total case-insensitive occurrences of each query term across title + all message content.
pub fn search_sessions(config_dir: &Path, query: &str) -> Result<Vec<SessionMatch>> {
    let terms: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).collect();

    if terms.is_empty() {
        return Ok(Vec::new());
    }

    // list() only loads metadata; we need full sessions to search message bodies.
    let stubs = list(config_dir)?;
    let mut matches: Vec<SessionMatch> = Vec::new();

    for stub in stubs {
        // load() reads the full session including messages
        let Ok(session) = load(config_dir, &stub.id) else {
            continue;
        };

        // Build a searchable corpus of (text, source_index) pairs.
        // source_index 0 = title, 1+ = messages
        let title_text = session.title.as_deref().unwrap_or("").to_lowercase();

        // Count term hits in title
        let mut score: usize = 0;
        for term in &terms {
            score += count_occurrences(&title_text, term);
        }

        // Count term hits in each message; track best snippet
        let mut best_msg_score: usize = 0;
        let mut best_snippet = String::new();

        for msg in &session.messages {
            let Some(content) = msg["content"].as_str() else {
                continue;
            };
            let content_lower = content.to_lowercase();
            let msg_score: usize = terms
                .iter()
                .map(|t| count_occurrences(&content_lower, t))
                .sum();
            if msg_score > best_msg_score {
                best_msg_score = msg_score;
                // Take first 120 chars as snippet
                best_snippet = content.chars().take(120).collect();
            }
            score += msg_score;
        }

        if score > 0 {
            let snippet = if best_snippet.is_empty() {
                title_text.chars().take(120).collect()
            } else {
                best_snippet
            };
            matches.push(SessionMatch {
                session,
                score,
                snippet,
            });
        }
    }

    // Sort by score descending, take top 10
    matches.sort_by(|a, b| b.score.cmp(&a.score));
    matches.truncate(10);
    Ok(matches)
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        count += 1;
        start += pos + needle.len();
    }
    count
}

fn sessions_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("sessions")
}

/// Remove session files older than `max_age_days`. Returns the number of files removed.
pub fn prune(config_dir: &Path, max_age_days: u64) -> Result<usize> {
    let dir = sessions_dir(config_dir);
    if !dir.exists() {
        return Ok(0);
    }
    let cutoff =
        std::time::SystemTime::now() - std::time::Duration::from_secs(max_age_days * 86400);
    let mut pruned = 0;
    for entry in std::fs::read_dir(&dir)?.filter_map(|e| e.ok()) {
        if entry.path().extension().is_some_and(|x| x == "json") {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if mtime < cutoff && std::fs::remove_file(entry.path()).is_ok() {
                        pruned += 1;
                    }
                }
            }
        }
    }
    Ok(pruned)
}

fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty() {
        anyhow::bail!("Session ID cannot be empty");
    }
    if !id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        anyhow::bail!(
            "Invalid session ID '{}': only alphanumeric characters, dashes, and underscores are allowed",
            id
        );
    }
    Ok(())
}

fn session_path(config_dir: &Path, id: &str) -> Result<PathBuf> {
    validate_session_id(id)?;
    Ok(sessions_dir(config_dir).join(format!("{}.json", id)))
}

pub fn save(config_dir: &Path, session: &Session) -> Result<()> {
    let dir = sessions_dir(config_dir);
    std::fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    let path = session_path(config_dir, &session.id)?;
    let tmp_path = path.with_extension("json.tmp");
    let content = serde_json::to_string(session).context("Failed to serialize session")?;
    std::fs::write(&tmp_path, &content)
        .with_context(|| format!("Failed to write tmp session {:?}", tmp_path))?;
    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("Failed to rename session file {:?}", path))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

pub fn load(config_dir: &Path, id: &str) -> Result<Session> {
    let path = session_path(config_dir, id)?;
    let content =
        std::fs::read_to_string(&path).with_context(|| format!("Session '{}' not found", id))?;
    let session: Session =
        serde_json::from_str(&content).context("Failed to parse session file")?;
    Ok(session)
}

/// Load the most recently modified session (for --continue)
pub fn load_latest(config_dir: &Path) -> Result<Option<Session>> {
    let dir = sessions_dir(config_dir);
    if !dir.exists() {
        return Ok(None);
    }

    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| {
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, e.path()))
        })
        .collect();

    entries.sort_by(|a, b| b.0.cmp(&a.0));

    match entries.first() {
        None => Ok(None),
        Some((_, path)) => {
            let content = std::fs::read_to_string(path)?;
            let session: Session = serde_json::from_str(&content)?;
            Ok(Some(session))
        }
    }
}

pub fn update_title(config_dir: &Path, id: &str, title: &str) -> Result<()> {
    let mut session = load(config_dir, id)?;
    session.title = Some(title.to_string());
    save(config_dir, &session)
}

pub fn delete(config_dir: &Path, session_id: &str) -> Result<()> {
    let path = session_path(config_dir, session_id)?;
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("Failed to delete session '{}'", session_id))?;
    }
    Ok(())
}

pub fn list(config_dir: &Path) -> Result<Vec<Session>> {
    let dir = sessions_dir(config_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|x| x == "json") {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                if let Ok(s) = serde_json::from_str::<Session>(&content) {
                    sessions.push(s);
                }
            }
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(sessions)
}

pub fn list_meta(config_dir: &Path) -> Result<Vec<SessionMeta>> {
    let dir = sessions_dir(config_dir);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut metas = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|x| x == "json") {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                if let Ok(m) = serde_json::from_str::<SessionMeta>(&content) {
                    metas.push(m);
                }
            }
        }
    }
    metas.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(metas)
}

/// Render saved sessions as an ASCII fork tree, walking `parent_id` chains.
///
/// Takes a formatter closure so this function stays independent of the
/// per-row formatter that lives in `tui/commands.rs`. Roots (sessions with
/// `parent_id: None`) and sessions whose `parent_id` points at a session
/// not in `sessions` (dangling — parent was deleted) are both rendered as
/// top-level. Siblings and roots are sorted by `updated_at` descending so
/// the most recent branch appears first.
///
/// Independent roots are separated by a single blank line. The return
/// value ends with a trailing newline when non-empty, and is empty for
/// `&[]`.
///
/// A visited-set guard prevents infinite recursion in the unlikely
/// event of a parent-cycle (shouldn't happen — `fork()` always creates
/// a fresh UUID — but defense in depth since the field is deserialized
/// from arbitrary on-disk JSON). After rendering all roots, an orphan
/// pass emits any session not yet visited as an additional top-level
/// entry, guaranteeing every session in the input appears in the output
/// regardless of cycles or dangling references.
pub fn format_session_tree<F>(sessions: &[Session], format_row: F) -> String
where
    F: Fn(&Session) -> String,
{
    use std::collections::{HashMap, HashSet};

    if sessions.is_empty() {
        return String::new();
    }

    let ids: HashSet<&str> = sessions.iter().map(|s| s.id.as_str()).collect();
    let mut children_of: HashMap<&str, Vec<&Session>> = HashMap::new();
    let mut roots: Vec<&Session> = Vec::new();

    for s in sessions {
        match s.parent_id.as_deref() {
            Some(pid) if ids.contains(pid) => {
                children_of.entry(pid).or_default().push(s);
            }
            // Either None (true root) or Some(pid) where pid is absent from
            // this list (dangling — parent was deleted). Both render as
            // top-level entries so the session isn't lost from view.
            _ => roots.push(s),
        }
    }

    roots.sort_by(|a, b| {
        b.updated_at
            .cmp(&a.updated_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    for v in children_of.values_mut() {
        v.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| a.id.cmp(&b.id))
        });
    }

    let mut out = String::new();
    let mut visited: HashSet<String> = HashSet::new();
    for (i, root) in roots.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        render_node(
            root,
            "",
            true,
            true,
            &children_of,
            &format_row,
            &mut out,
            &mut visited,
        );
    }

    // Orphan pass — renders any session not reached by the root-rooted walk.
    // Fallback for pathological inputs (e.g. A.parent_id = B and
    // B.parent_id = A, where neither is a root and both sit in children_of):
    // without this pass the tree would return empty and silently lose data.
    // Only reachable from corrupted on-disk JSON, not from fork().
    let mut orphans: Vec<&Session> = sessions
        .iter()
        .filter(|s| !visited.contains(&s.id))
        .collect();
    if !orphans.is_empty() {
        orphans.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        for s in orphans {
            if !out.is_empty() {
                out.push('\n');
            }
            render_node(
                s,
                "",
                true,
                true,
                &children_of,
                &format_row,
                &mut out,
                &mut visited,
            );
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn render_node<F>(
    node: &Session,
    prefix: &str,
    is_last: bool,
    is_root: bool,
    children_of: &std::collections::HashMap<&str, Vec<&Session>>,
    format_row: &F,
    out: &mut String,
    visited: &mut std::collections::HashSet<String>,
) where
    F: Fn(&Session) -> String,
{
    if !visited.insert(node.id.clone()) {
        return;
    }

    if is_root {
        out.push_str(&format_row(node));
    } else {
        out.push_str(prefix);
        out.push_str(if is_last { "└── " } else { "├── " });
        out.push_str(&format_row(node));
    }
    out.push('\n');

    let next_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    let empty = Vec::new();
    let children = children_of.get(node.id.as_str()).unwrap_or(&empty);
    for (i, child) in children.iter().enumerate() {
        let last = i == children.len() - 1;
        render_node(
            child,
            &next_prefix,
            last,
            false,
            children_of,
            format_row,
            out,
            visited,
        );
    }
}

/// Persist a session stub for the outgoing session on `/new`.
///
/// **Read-then-update**: if `id` exists on disk, load the full session,
/// mutate `title` + `model` + `updated_at`, and save. This preserves
/// every on-disk field the caller doesn't explicitly track — critically
/// `parent_id` (forked-session lineage), but also `turn_count`,
/// `prompt_tokens`, `completion_tokens`, `compact_failures`, and the
/// message history.
///
/// If `id` is not on disk (e.g. `/new` fires before the session ever
/// persisted), fall back to creating a minimal stub so `/sessions`
/// still lists the session.
pub fn update_or_create_stub(
    config_dir: &Path,
    id: &str,
    title: Option<String>,
    model: &str,
    cwd: &str,
) -> Result<()> {
    let now = Utc::now();
    match load(config_dir, id) {
        Ok(mut existing) => {
            existing.title = title;
            existing.model = model.to_string();
            existing.updated_at = now;
            save(config_dir, &existing)
        }
        Err(_) => save(
            config_dir,
            &Session {
                id: id.to_string(),
                title,
                model: model.to_string(),
                messages: vec![],
                cwd: cwd.to_string(),
                host_project: crate::identity::load_at(Path::new(cwd))
                    .ok()
                    .and_then(|identity| identity.host_project),
                created_at: now,
                updated_at: now,
                compact_failures: 0,
                turn_count: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                parent_id: None,
            },
        ),
    }
}

/// Build the tail of a "session save failed" warning — the part after the
/// caller-specific prefix. Returns exactly:
///
/// ```text
/// session save failed: {err}{separator}Check: available disk space in {config_dir}
/// ```
///
/// Callers supply `separator`: `". "` for single-line TUI entries (e.g. `/new`)
/// or `"\n    "` for multi-line stderr output (e.g. SIGTERM/SIGINT handlers).
/// Keeps the "what went wrong + what to check" wording locked across every
/// save-failure site so users see one consistent idiom regardless of trigger.
pub fn format_save_failure_tail(err: &anyhow::Error, config_dir: &Path, separator: &str) -> String {
    format!(
        "session save failed: {}{}Check: available disk space in {}",
        err,
        separator,
        config_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn format_save_failure_tail_single_line_separator_for_tui() {
        let err = anyhow::anyhow!("disk full");
        let dir = std::path::Path::new("/home/user/.config/dm");
        let tail = format_save_failure_tail(&err, dir, ". ");
        assert_eq!(
            tail,
            "session save failed: disk full. Check: available disk space in /home/user/.config/dm"
        );
        assert!(!tail.contains('\n'), "got: {:?}", tail);
    }

    #[test]
    fn format_save_failure_tail_multiline_separator_indents_check_hint() {
        let err = anyhow::anyhow!("no space left on device");
        let dir = std::path::Path::new("/tmp/cfg");
        let tail = format_save_failure_tail(&err, dir, "\n    ");
        assert_eq!(
            tail,
            "session save failed: no space left on device\n    Check: available disk space in /tmp/cfg"
        );
        assert!(tail.contains("\n    Check:"), "got: {:?}", tail);
    }

    #[test]
    fn round_trip_save_load() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        save(dir.path(), &sess).unwrap();
        let loaded = load(dir.path(), &sess.id).unwrap();
        assert_eq!(loaded.id, sess.id);
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.model, "gemma4:26b");
    }

    #[test]
    fn load_missing_session_errors() {
        let dir = TempDir::new().unwrap();
        let result = load(dir.path(), "nonexistent-id");
        assert!(result.is_err());
    }

    #[test]
    fn list_returns_empty_for_new_dir() {
        let dir = TempDir::new().unwrap();
        let sessions = list(dir.path()).unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn list_returns_saved_sessions() {
        let dir = TempDir::new().unwrap();
        let sess1 = Session::new("/a".to_string(), "model1".to_string());
        let sess2 = Session::new("/b".to_string(), "model2".to_string());
        save(dir.path(), &sess1).unwrap();
        save(dir.path(), &sess2).unwrap();
        let sessions = list(dir.path()).unwrap();
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn session_rename_updates_title() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        save(dir.path(), &sess).unwrap();
        update_title(dir.path(), &sess.id, "new name").unwrap();
        let loaded = load(dir.path(), &sess.id).unwrap();
        assert_eq!(loaded.title, Some("new name".to_string()));
    }

    #[test]
    fn session_rename_unknown_id_errors() {
        let dir = TempDir::new().unwrap();
        let result = update_title(dir.path(), "nonexistent-id", "some name");
        assert!(result.is_err());
    }

    #[test]
    fn search_sessions_finds_match() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.title = Some("Rust refactor session".to_string());
        sess.push_message(
            serde_json::json!({"role": "user", "content": "How do I refactor this Rust code?"}),
        );
        save(dir.path(), &sess).unwrap();

        let results = search_sessions(dir.path(), "refactor").unwrap();
        assert!(
            !results.is_empty(),
            "should find session matching 'refactor'"
        );
        assert_eq!(results[0].session.id, sess.id);
    }

    #[test]
    fn search_sessions_returns_empty_on_no_match() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "Hello world"}));
        save(dir.path(), &sess).unwrap();

        let results = search_sessions(dir.path(), "xyzzy_nonexistent").unwrap();
        assert!(results.is_empty(), "should return empty for no match");
    }

    #[test]
    fn search_sessions_ranks_by_frequency() {
        let dir = TempDir::new().unwrap();

        // Session with low frequency
        let mut sess_low = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess_low.push_message(serde_json::json!({"role": "user", "content": "rust is cool"}));
        save(dir.path(), &sess_low).unwrap();

        // Session with high frequency
        let mut sess_high = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess_high.push_message(
            serde_json::json!({"role": "user", "content": "rust rust rust rust rust"}),
        );
        save(dir.path(), &sess_high).unwrap();

        let results = search_sessions(dir.path(), "rust").unwrap();
        assert!(!results.is_empty(), "should find matching sessions");
        assert_eq!(
            results[0].session.id, sess_high.id,
            "high frequency session should rank first"
        );
        assert!(
            results[0].score >= results[1].score,
            "results should be sorted by score"
        );
    }

    #[test]
    fn load_latest_returns_most_recent() {
        let dir = TempDir::new().unwrap();
        let sess1 = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        save(dir.path(), &sess1).unwrap();
        // Small sleep ensures different mtime
        std::thread::sleep(std::time::Duration::from_millis(10));
        let sess2 = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        save(dir.path(), &sess2).unwrap();
        let latest = load_latest(dir.path()).unwrap().unwrap();
        assert_eq!(latest.id, sess2.id);
    }

    #[test]
    fn list_meta_returns_empty_for_new_dir() {
        let dir = TempDir::new().unwrap();
        let metas = list_meta(dir.path()).unwrap();
        assert!(metas.is_empty());
    }

    #[test]
    fn list_meta_returns_saved_sessions() {
        let dir = TempDir::new().unwrap();
        let sess1 = Session::new("/a".to_string(), "model1".to_string());
        let sess2 = Session::new("/b".to_string(), "model2".to_string());
        save(dir.path(), &sess1).unwrap();
        save(dir.path(), &sess2).unwrap();
        let metas = list_meta(dir.path()).unwrap();
        assert_eq!(metas.len(), 2);
    }

    #[test]
    fn list_meta_has_correct_fields() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.title = Some("Test Title".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        save(dir.path(), &sess).unwrap();
        let metas = list_meta(dir.path()).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].id, sess.id);
        assert_eq!(metas[0].title.as_deref(), Some("Test Title"));
        assert_eq!(metas[0].model, "gemma4:26b");
        assert_eq!(metas[0].cwd, "/tmp");
    }

    #[test]
    fn list_meta_sorted_by_updated_at() {
        let dir = TempDir::new().unwrap();
        let sess1 = Session::new("/a".to_string(), "m1".to_string());
        save(dir.path(), &sess1).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let sess2 = Session::new("/b".to_string(), "m2".to_string());
        save(dir.path(), &sess2).unwrap();
        let metas = list_meta(dir.path()).unwrap();
        assert_eq!(metas[0].id, sess2.id, "most recent should be first");
    }

    #[test]
    fn delete_removes_saved_session() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        assert!(load(dir.path(), &sess.id).is_ok());
        delete(dir.path(), &sess.id).unwrap();
        assert!(load(dir.path(), &sess.id).is_err());
    }

    #[test]
    fn delete_nonexistent_session_succeeds() {
        let dir = TempDir::new().unwrap();
        delete(dir.path(), "nonexistent-id").unwrap();
    }

    #[test]
    fn load_corrupt_json_errors() {
        let dir = TempDir::new().unwrap();
        let sdir = dir.path().join("sessions");
        std::fs::create_dir_all(&sdir).unwrap();
        std::fs::write(sdir.join("bad.json"), "{ not valid json !!!").unwrap();
        let result = load(dir.path(), "bad");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("parse") || err_msg.contains("Failed"),
            "should mention parse failure: {}",
            err_msg
        );
    }

    #[test]
    fn load_latest_returns_none_when_no_sessions_dir() {
        let dir = TempDir::new().unwrap();
        let result = load_latest(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn search_sessions_empty_query_returns_empty() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "m".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        save(dir.path(), &sess).unwrap();
        let results = search_sessions(dir.path(), "").unwrap();
        assert!(results.is_empty(), "empty query should return no results");
    }

    #[test]
    fn search_sessions_whitespace_query_returns_empty() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "m".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        save(dir.path(), &sess).unwrap();
        let results = search_sessions(dir.path(), "   ").unwrap();
        assert!(
            results.is_empty(),
            "whitespace-only query should return no results"
        );
    }

    #[test]
    fn search_sessions_title_only_match_uses_title_as_snippet() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "m".to_string());
        sess.title = Some("debugging rust lifetimes".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "unrelated content here"}));
        save(dir.path(), &sess).unwrap();
        let results = search_sessions(dir.path(), "lifetimes").unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            results[0].snippet.contains("lifetimes"),
            "snippet should fall back to title: {}",
            results[0].snippet
        );
    }

    #[test]
    fn save_is_atomic_no_tmp_left_behind() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        let sdir = dir.path().join("sessions");
        let tmp_files: Vec<_> = std::fs::read_dir(&sdir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .collect();
        assert!(
            tmp_files.is_empty(),
            "no .tmp files should remain after save"
        );
        let loaded = load(dir.path(), &sess.id).unwrap();
        assert_eq!(loaded.id, sess.id);
    }

    #[test]
    fn save_overwrites_existing_session_atomically() {
        let dir = TempDir::new().unwrap();
        let mut sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        save(dir.path(), &sess).unwrap();
        let loaded = load(dir.path(), &sess.id).unwrap();
        assert_eq!(loaded.messages.len(), 1);
    }

    #[test]
    fn count_occurrences_basic() {
        assert_eq!(count_occurrences("aaa", "a"), 3);
        assert_eq!(count_occurrences("hello world hello", "hello"), 2);
        assert_eq!(count_occurrences("no match here", "xyz"), 0);
        assert_eq!(count_occurrences("", "a"), 0);
        assert_eq!(count_occurrences("anything", ""), 0);
    }

    #[test]
    fn prune_removes_old_sessions() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        let path = dir
            .path()
            .join("sessions")
            .join(format!("{}.json", sess.id));
        // Backdate the file mtime to 100 days ago using touch
        let old_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 100 * 86400;
        let dt = chrono::DateTime::from_timestamp(old_ts as i64, 0).unwrap();
        let ts = dt.format("%Y%m%d%H%M.%S").to_string();
        std::process::Command::new("touch")
            .args(["-t", &ts, path.to_str().unwrap()])
            .status()
            .unwrap();
        let pruned = prune(dir.path(), 90).unwrap();
        assert_eq!(pruned, 1, "should prune 1 old session");
        assert!(!path.exists(), "old session file should be removed");
    }

    #[test]
    fn validate_session_id_accepts_normal() {
        assert!(validate_session_id("abc-123").is_ok());
        assert!(validate_session_id("session_1").is_ok());
        assert!(validate_session_id("A1b2C3").is_ok());
    }

    #[test]
    fn validate_session_id_rejects_path_traversal() {
        assert!(validate_session_id("../../etc/passwd").is_err());
        assert!(validate_session_id("..").is_err());
    }

    #[test]
    fn validate_session_id_rejects_slashes() {
        assert!(validate_session_id("foo/bar").is_err());
        assert!(validate_session_id("foo\\bar").is_err());
    }

    #[test]
    fn validate_session_id_rejects_empty() {
        assert!(validate_session_id("").is_err());
    }

    #[test]
    fn session_path_rejects_malicious_id() {
        let dir = TempDir::new().unwrap();
        assert!(session_path(dir.path(), "../../../etc/passwd").is_err());
    }

    #[test]
    fn prune_preserves_recent_sessions() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        let pruned = prune(dir.path(), 90).unwrap();
        assert_eq!(pruned, 0, "recent session should not be pruned");
        let sessions = list(dir.path()).unwrap();
        assert_eq!(sessions.len(), 1, "session should still exist");
    }

    #[test]
    fn save_produces_compact_json() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "test-model".to_string());
        save(dir.path(), &sess).unwrap();
        let path = session_path(dir.path(), &sess.id).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            !content.contains('\n'),
            "compact JSON should not contain newlines"
        );
        let loaded: Session = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.id, sess.id);
    }

    #[test]
    fn update_title_produces_compact_json() {
        let dir = TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        update_title(dir.path(), &sess.id, "New Title").unwrap();
        let path = session_path(dir.path(), &sess.id).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            !content.contains('\n'),
            "compact JSON should not contain newlines"
        );
        let loaded: Session = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.title.as_deref(), Some("New Title"));
    }

    #[cfg(unix)]
    #[test]
    fn session_file_has_restricted_permissions() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        let path = session_path(dir.path(), &sess.id).unwrap();
        let mode = std::fs::metadata(&path).unwrap().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "session file should be owner-only rw: {:o}",
            mode
        );
    }

    #[test]
    fn count_user_messages_empty_is_zero() {
        assert_eq!(count_user_messages(&[]), 0);
    }

    #[test]
    fn count_user_messages_counts_only_user_role() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "hello"}),
            serde_json::json!({"role": "user", "content": "bye"}),
            serde_json::json!({"role": "tool", "content": "t"}),
        ];
        assert_eq!(count_user_messages(&msgs), 2);
    }

    #[test]
    fn count_user_messages_all_user() {
        let msgs: Vec<_> = (0..5)
            .map(|i| serde_json::json!({"role": "user", "content": format!("msg {}", i)}))
            .collect();
        assert_eq!(count_user_messages(&msgs), 5);
    }

    #[test]
    fn human_tokens_zero() {
        assert_eq!(human_tokens(0), "0");
    }

    #[test]
    fn human_tokens_small_exact() {
        assert_eq!(human_tokens(500), "500");
        assert_eq!(human_tokens(999), "999");
    }

    #[test]
    fn human_tokens_thousands() {
        assert_eq!(human_tokens(12_345), "12.3k");
        assert_eq!(human_tokens(1_000), "1.0k");
    }

    #[test]
    fn human_tokens_millions() {
        assert_eq!(human_tokens(1_234_567), "1.2M");
    }

    #[test]
    fn list_meta_defaults_turn_and_token_fields_for_old_sessions() {
        // A SessionMeta deserialized from a Session written before the
        // turn_count / prompt_tokens / completion_tokens fields existed must
        // still load cleanly with all three at 0 — this is what `/sessions`
        // uses to decide whether to show a real count or the estimated
        // fallback.
        let dir = TempDir::new().unwrap();
        let sdir = dir.path().join("sessions");
        std::fs::create_dir_all(&sdir).unwrap();
        let legacy = r#"{
            "id": "legacy-id",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "cwd": "/tmp",
            "model": "gemma4:26b",
            "messages": []
        }"#;
        std::fs::write(sdir.join("legacy-id.json"), legacy).unwrap();
        let metas = list_meta(dir.path()).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].turn_count, 0);
        assert_eq!(metas[0].prompt_tokens, 0);
        assert_eq!(metas[0].completion_tokens, 0);
    }

    #[cfg(unix)]
    #[test]
    fn sessions_dir_has_restricted_permissions() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::TempDir::new().unwrap();
        let sess = Session::new("/tmp".to_string(), "m".to_string());
        save(dir.path(), &sess).unwrap();
        let sessions = sessions_dir(dir.path());
        let mode = std::fs::metadata(&sessions).unwrap().mode() & 0o777;
        assert_eq!(mode, 0o700, "sessions dir should be owner-only: {:o}", mode);
    }

    #[test]
    fn update_or_create_stub_preserves_parent_id_when_session_exists() {
        // Regression: /new used to hardcode parent_id: None when saving
        // the outgoing session, silently orphaning forked sessions.
        let dir = TempDir::new().unwrap();
        let mut parent = Session::new("/tmp".into(), "m1".into());
        parent.parent_id = Some("root-abc".into());
        save(dir.path(), &parent).unwrap();

        update_or_create_stub(
            dir.path(),
            &parent.id,
            Some("renamed".into()),
            "m2",
            "/newcwd",
        )
        .unwrap();

        let reloaded = load(dir.path(), &parent.id).unwrap();
        assert_eq!(
            reloaded.parent_id.as_deref(),
            Some("root-abc"),
            "parent_id must survive the update"
        );
    }

    #[test]
    fn update_or_create_stub_preserves_turn_count_and_tokens() {
        // App state doesn't mirror turn_count / tokens / compact_failures;
        // the read-then-update path must leave them untouched.
        let dir = TempDir::new().unwrap();
        let mut s = Session::new("/tmp".into(), "m1".into());
        s.turn_count = 42;
        s.prompt_tokens = 1234;
        s.completion_tokens = 567;
        s.compact_failures = 2;
        save(dir.path(), &s).unwrap();

        update_or_create_stub(dir.path(), &s.id, None, "m2", "/tmp").unwrap();

        let reloaded = load(dir.path(), &s.id).unwrap();
        assert_eq!(reloaded.turn_count, 42);
        assert_eq!(reloaded.prompt_tokens, 1234);
        assert_eq!(reloaded.completion_tokens, 567);
        assert_eq!(reloaded.compact_failures, 2);
    }

    #[test]
    fn update_or_create_stub_updates_title_model_and_timestamp() {
        let dir = TempDir::new().unwrap();
        let s = Session::new("/tmp".into(), "old-model".into());
        let original_updated_at = s.updated_at;
        save(dir.path(), &s).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));

        update_or_create_stub(
            dir.path(),
            &s.id,
            Some("new-title".into()),
            "new-model",
            "/tmp",
        )
        .unwrap();

        let reloaded = load(dir.path(), &s.id).unwrap();
        assert_eq!(reloaded.title.as_deref(), Some("new-title"));
        assert_eq!(reloaded.model, "new-model");
        assert!(
            reloaded.updated_at > original_updated_at,
            "updated_at must advance"
        );
    }

    #[test]
    fn update_or_create_stub_creates_stub_when_missing() {
        // First-turn /new fires before the session has ever persisted.
        // No on-disk file → must create a minimal stub so /sessions lists it.
        let dir = TempDir::new().unwrap();
        let id = uuid::Uuid::new_v4().to_string();

        update_or_create_stub(dir.path(), &id, Some("first".into()), "m", "/tmp").unwrap();

        let reloaded = load(dir.path(), &id).unwrap();
        assert_eq!(reloaded.id, id);
        assert_eq!(reloaded.title.as_deref(), Some("first"));
        assert_eq!(reloaded.model, "m");
        assert!(reloaded.messages.is_empty());
        assert!(reloaded.parent_id.is_none(), "fresh stub has no lineage");
    }

    #[test]
    fn update_or_create_stub_records_host_project_from_cwd_identity() {
        // The fallback stub-creation path (cycle 13 added `host_project`)
        // derives identity from the session's cwd. A host-mode project
        // must surface its `host_project` so `/sessions list` and the
        // dashboard can disambiguate sessions across spawned projects.
        let config_dir = TempDir::new().unwrap();
        let project_dir = TempDir::new().unwrap();
        std::fs::create_dir_all(project_dir.path().join(".dm")).unwrap();
        std::fs::write(
            project_dir.path().join(".dm").join("identity.toml"),
            "mode = \"host\"\nhost_project = \"finance-app\"\n",
        )
        .unwrap();
        let id = uuid::Uuid::new_v4().to_string();

        update_or_create_stub(
            config_dir.path(),
            &id,
            Some("first".into()),
            "m",
            project_dir.path().to_str().unwrap(),
        )
        .unwrap();

        let reloaded = load(config_dir.path(), &id).unwrap();
        assert_eq!(
            reloaded.host_project.as_deref(),
            Some("finance-app"),
            "stub must capture host_project from cwd identity",
        );
    }

    #[test]
    fn update_or_create_stub_records_no_host_project_in_kernel_mode() {
        // Mirror of the host-mode test for the default kernel path.
        // No `.dm/identity.toml` → identity defaults to kernel → host_project is None.
        let config_dir = TempDir::new().unwrap();
        let project_dir = TempDir::new().unwrap();
        let id = uuid::Uuid::new_v4().to_string();

        update_or_create_stub(
            config_dir.path(),
            &id,
            Some("first".into()),
            "m",
            project_dir.path().to_str().unwrap(),
        )
        .unwrap();

        let reloaded = load(config_dir.path(), &id).unwrap();
        assert!(
            reloaded.host_project.is_none(),
            "kernel-mode stub must not invent a host_project, got {:?}",
            reloaded.host_project,
        );
    }

    #[test]
    fn update_or_create_stub_does_not_touch_messages() {
        // App state does not track `messages` (runner-managed). The update
        // path must preserve whatever is on disk, NOT zero it.
        let dir = TempDir::new().unwrap();
        let mut s = Session::new("/tmp".into(), "m".into());
        s.messages = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "content": "hello"}),
        ];
        save(dir.path(), &s).unwrap();

        update_or_create_stub(dir.path(), &s.id, Some("t".into()), "m", "/tmp").unwrap();

        let reloaded = load(dir.path(), &s.id).unwrap();
        assert_eq!(reloaded.messages.len(), 2, "existing messages must survive");
    }

    #[test]
    fn update_or_create_stub_is_idempotent_across_repeat_calls() {
        // Shutdown paths (SIGTERM handler, panic hook) may fire more than
        // once under adversarial conditions (supervisor re-signal, nested
        // panics). Repeated calls must not corrupt state — each call should
        // leave the session in the same observable shape.
        let dir = TempDir::new().unwrap();
        let mut s = Session::new("/tmp".into(), "m1".into());
        s.parent_id = Some("root-xyz".into());
        s.turn_count = 7;
        s.prompt_tokens = 500;
        save(dir.path(), &s).unwrap();

        for _ in 0..3 {
            update_or_create_stub(dir.path(), &s.id, Some("renamed".into()), "m2", "/tmp").unwrap();
        }

        let reloaded = load(dir.path(), &s.id).unwrap();
        assert_eq!(reloaded.title.as_deref(), Some("renamed"));
        assert_eq!(reloaded.model, "m2");
        assert_eq!(
            reloaded.parent_id.as_deref(),
            Some("root-xyz"),
            "repeat calls must preserve parent_id"
        );
        assert_eq!(reloaded.turn_count, 7, "turn_count survives");
        assert_eq!(reloaded.prompt_tokens, 500, "tokens survive");
    }

    fn mk(id: &str, parent: Option<&str>, offset_secs: i64) -> Session {
        let mut s = Session::new("/tmp".into(), "m".into());
        s.id = id.into();
        s.parent_id = parent.map(String::from);
        s.updated_at = chrono::Utc::now() + chrono::Duration::seconds(offset_secs);
        s
    }

    #[test]
    fn format_session_tree_empty_returns_empty() {
        let out = format_session_tree(&[], |s| s.id.clone());
        assert!(out.is_empty());
    }

    #[test]
    fn format_session_tree_single_root_has_no_tree_chars() {
        let sessions = vec![mk("R", None, 0)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R\n");
        assert!(!out.contains('├'));
        assert!(!out.contains('└'));
        assert!(!out.contains('│'));
    }

    #[test]
    fn format_session_tree_root_with_only_child_uses_last_connector() {
        let sessions = vec![mk("R", None, 0), mk("C", Some("R"), 10)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R\n└── C\n");
    }

    #[test]
    fn format_session_tree_two_children_use_branch_then_last() {
        // C1 newer than C2, so C1 sorts first (branch), C2 last.
        let sessions = vec![
            mk("R", None, 0),
            mk("C1", Some("R"), 20),
            mk("C2", Some("R"), 10),
        ];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R\n├── C1\n└── C2\n");
    }

    #[test]
    fn format_session_tree_depth_three_uses_vertical_continuation() {
        // C1 is not last, so its descendants get "│   " continuation.
        let sessions = vec![
            mk("R", None, 0),
            mk("C1", Some("R"), 20),
            mk("C2", Some("R"), 10),
            mk("G", Some("C1"), 30),
        ];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R\n├── C1\n│   └── G\n└── C2\n");
    }

    #[test]
    fn format_session_tree_depth_three_uses_space_continuation_when_parent_is_last() {
        // Only one child C, so C is last → its descendant uses "    " prefix.
        let sessions = vec![
            mk("R", None, 0),
            mk("C", Some("R"), 10),
            mk("G", Some("C"), 20),
        ];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R\n└── C\n    └── G\n");
    }

    #[test]
    fn format_session_tree_dangling_parent_promotes_to_root() {
        // D's parent points at a session not in the list → D renders as root,
        // so deleting a parent doesn't silently hide descendants from the tree.
        let sessions = vec![mk("D", Some("missing"), 0)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "D\n");
    }

    #[test]
    fn format_session_tree_roots_separated_by_blank_line() {
        // R1 newer than R2 → R1 first, blank line, then R2.
        let sessions = vec![mk("R1", None, 20), mk("R2", None, 10)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out, "R1\n\nR2\n");
    }

    #[test]
    fn format_session_tree_roots_sorted_by_updated_at_desc() {
        // R_old is older than R_new — most recent should appear first.
        let sessions = vec![mk("R_old", None, 0), mk("R_new", None, 100)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        let pos_new = out.find("R_new").expect("R_new present");
        let pos_old = out.find("R_old").expect("R_old present");
        assert!(pos_new < pos_old, "newer root must come first");
    }

    #[test]
    fn format_session_tree_cycle_guard_terminates() {
        // A duplicate-id entry would otherwise be rendered twice under the same
        // parent. The visited-set makes each id appear at most once — and, more
        // importantly, guarantees termination on any pathological input.
        let sessions = vec![
            mk("R", None, 0),
            mk("C", Some("R"), 10),
            mk("C", Some("R"), 20),
        ];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(out.matches("── C").count(), 1, "C rendered exactly once");
    }

    #[test]
    fn format_session_tree_true_cycle_renders_both_nodes() {
        // Pathological: A ↔ B. Neither is a root. Before the orphan pass,
        // this rendered empty (silent data loss). With the orphan pass, both
        // nodes surface as top-level entries.
        let sessions = vec![mk("A", Some("B"), 10), mk("B", Some("A"), 5)];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert!(out.contains('A'), "cycle node A must render: {out:?}");
        assert!(out.contains('B'), "cycle node B must render: {out:?}");
    }

    #[test]
    fn format_session_tree_tied_updated_at_sorted_by_id() {
        // Determinism: two roots with identical updated_at must order by id
        // ascending to give reproducible output across runs.
        let t = chrono::Utc::now();
        let mut a = Session::new("/tmp".into(), "m".into());
        a.id = "z".into();
        a.updated_at = t;
        let mut b = Session::new("/tmp".into(), "m".into());
        b.id = "a".into();
        b.updated_at = t;
        let sessions = vec![a, b];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert!(
            out.starts_with("a\n"),
            "tied timestamps → tiebreak by id ascending: {out:?}"
        );
    }

    #[test]
    fn format_session_tree_sibling_tied_updated_at_sorted_by_id() {
        // Same determinism, but applied to siblings under a common parent.
        let t = chrono::Utc::now();
        let mut root = Session::new("/tmp".into(), "m".into());
        root.id = "root".into();
        root.updated_at = t + chrono::Duration::seconds(100);
        let mut c1 = Session::new("/tmp".into(), "m".into());
        c1.id = "z".into();
        c1.parent_id = Some("root".into());
        c1.updated_at = t;
        let mut c2 = Session::new("/tmp".into(), "m".into());
        c2.id = "a".into();
        c2.parent_id = Some("root".into());
        c2.updated_at = t;
        let sessions = vec![root, c1, c2];
        let out = format_session_tree(&sessions, |s| s.id.clone());
        assert_eq!(
            out, "root\n├── a\n└── z\n",
            "tied siblings → id-ascending: {out:?}"
        );
    }
}
