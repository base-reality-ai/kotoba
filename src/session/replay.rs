use crate::ollama::client::OllamaClient;
use crate::session::Session;
use anyhow::Context;
use std::path::Path;

/// Returns content of all user messages in order.
pub fn extract_user_turns(session: &Session) -> Vec<String> {
    session
        .messages
        .iter()
        .filter_map(|msg| {
            if msg["role"].as_str() == Some("user") {
                msg["content"].as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Replay a session's user turns against the model.
///
/// - Loads the session from `config_dir/sessions/<session_id>.json`
/// - Extracts all user turns
/// - Skips the first `from_turn` turns (0-indexed)
/// - If `dry_run`, prints what would be sent without calling Ollama
/// - Otherwise calls `client.chat()` for each user turn and prints the response
pub async fn run_replay(
    session_id: &str,
    model_override: Option<String>,
    from_turn: usize,
    dry_run: bool,
    client: &OllamaClient,
    config_dir: &Path,
) -> anyhow::Result<()> {
    let session = crate::session::storage::load(config_dir, session_id)
        .with_context(|| format!("Cannot load session '{}'", session_id))?;

    let turns = extract_user_turns(&session);

    // Determine the effective model
    let model = model_override.unwrap_or_else(|| session.model.clone());

    // Slice off the first `from_turn` turns
    let turns_to_replay = if from_turn >= turns.len() {
        &[][..]
    } else {
        &turns[from_turn..]
    };

    let n = turns_to_replay.len();

    if dry_run {
        for (i, text) in turns_to_replay.iter().enumerate() {
            println!("[turn {}] {}", from_turn + i, text);
        }
        println!(
            "Dry run: {} turn{} would replay",
            n,
            if n == 1 { "" } else { "s" }
        );
        return Ok(());
    }

    // Build a client with the correct model
    let replay_client = OllamaClient::new(client.base_url().to_string(), model);

    let mut messages: Vec<serde_json::Value> = Vec::new();
    let mut completed = 0usize;

    for (i, text) in turns_to_replay.iter().enumerate() {
        let turn_idx = from_turn + i;
        println!("--- Turn {} ---", turn_idx);
        println!("User: {}", text);

        messages.push(serde_json::json!({
            "role": "user",
            "content": text,
        }));

        match replay_client.chat(&messages, &[]).await {
            Ok(response) => {
                println!("Assistant: {}", response.message.content);
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": response.message.content,
                }));
                completed += 1;
            }
            Err(e) => {
                crate::warnings::push_warning(format!(
                    "replay turn {}/{} failed: {}. Stopping replay.",
                    turn_idx, n, e
                ));
                messages.pop();
                break;
            }
        }
    }

    println!(
        "Replay complete — {}/{} turn{}",
        completed,
        n,
        if n == 1 { "" } else { "s" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;

    fn make_session_with_messages() -> Session {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "hello"}));
        sess.push_message(serde_json::json!({"role": "assistant", "content": "world"}));
        sess.push_message(serde_json::json!({"role": "tool", "content": "tool output"}));
        sess.push_message(serde_json::json!({"role": "user", "content": "second user message"}));
        sess
    }

    #[test]
    fn replay_extracts_user_turns_only() {
        let sess = make_session_with_messages();
        let turns = extract_user_turns(&sess);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0], "hello");
        assert_eq!(turns[1], "second user message");
    }

    #[test]
    fn replay_from_turn_3_skips_first_2() {
        // Build a session with 5 user turns
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        for i in 0..5 {
            sess.push_message(
                serde_json::json!({"role": "user", "content": format!("turn {}", i)}),
            );
            sess.push_message(serde_json::json!({"role": "assistant", "content": "ok"}));
        }
        let turns = extract_user_turns(&sess);
        assert_eq!(turns.len(), 5);

        // from_turn=2 (0-indexed) means skip turns 0 and 1, replay turns 2,3,4
        let from_turn = 2;
        let replayed = if from_turn >= turns.len() {
            &[][..]
        } else {
            &turns[from_turn..]
        };
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0], "turn 2");
    }

    #[test]
    fn replay_extracts_empty_for_no_user_messages() {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "assistant", "content": "hello"}));
        let turns = extract_user_turns(&sess);
        assert!(turns.is_empty());
    }

    #[test]
    fn replay_from_turn_beyond_end_yields_empty_slice() {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": "only one"}));
        let turns = extract_user_turns(&sess);
        // from_turn=5 with 1 turn available → empty
        let from_turn = 5;
        let replayed = if from_turn >= turns.len() {
            &[][..]
        } else {
            &turns[from_turn..]
        };
        assert!(
            replayed.is_empty(),
            "from_turn beyond end should yield empty slice"
        );
    }

    #[test]
    fn replay_extracts_content_order_preserved() {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        for msg in ["alpha", "beta", "gamma"] {
            sess.push_message(serde_json::json!({"role": "user", "content": msg}));
        }
        let turns = extract_user_turns(&sess);
        assert_eq!(turns, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn replay_from_turn_zero_replays_all() {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        for i in 0..3 {
            sess.push_message(serde_json::json!({"role": "user", "content": format!("msg{}", i)}));
        }
        let turns = extract_user_turns(&sess);
        let from_turn = 0;
        let replayed = if from_turn >= turns.len() {
            &[][..]
        } else {
            &turns[from_turn..]
        };
        assert_eq!(replayed.len(), 3, "from_turn=0 should replay all turns");
        assert_eq!(replayed[0], "msg0");
    }

    #[test]
    fn replay_skips_user_messages_with_non_string_content() {
        // Non-string content (e.g., array) should be silently skipped
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "user", "content": ["array", "value"]}));
        sess.push_message(serde_json::json!({"role": "user", "content": "valid"}));
        let turns = extract_user_turns(&sess);
        assert_eq!(turns, vec!["valid"], "non-string content should be skipped");
    }

    #[test]
    fn replay_context_accumulates() {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let turns = vec!["hello", "how are you", "goodbye"];
        for text in &turns {
            messages.push(serde_json::json!({"role": "user", "content": text}));
            messages.push(serde_json::json!({"role": "assistant", "content": "response"}));
        }
        assert_eq!(
            messages.len(),
            6,
            "3 turns should produce 6 messages (3 user + 3 assistant)"
        );
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(messages[4]["content"], "goodbye");
    }

    #[test]
    fn replay_error_breaks_loop_leaves_clean_context() {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let mut completed = 0usize;

        let turns = ["first", "second", "third"];
        for (i, text) in turns.iter().enumerate() {
            messages.push(serde_json::json!({"role": "user", "content": text}));

            if i == 1 {
                // Simulate failure on second turn
                messages.pop();
                break;
            }
            messages.push(serde_json::json!({"role": "assistant", "content": "ok"}));
            completed += 1;
        }

        assert_eq!(completed, 1, "only first turn should succeed");
        assert_eq!(messages.len(), 2, "should have 1 user + 1 assistant");
        assert_eq!(messages[0]["content"], "first");
        assert_eq!(messages[1]["role"], "assistant");
    }

    #[test]
    fn replay_completion_count_format() {
        let completed = 2usize;
        let n = 5usize;
        let msg = format!(
            "Replay complete — {}/{} turn{}",
            completed,
            n,
            if n == 1 { "" } else { "s" }
        );
        assert!(msg.contains("2/5"), "should show completed/total: {msg}");
        assert!(msg.contains("turns"), "should pluralize: {msg}");
    }

    #[test]
    fn replay_context_error_pops_user_message() {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        messages.push(serde_json::json!({"role": "user", "content": "first"}));
        messages.push(serde_json::json!({"role": "assistant", "content": "ok"}));
        // Simulate failed turn: push user msg then pop it
        messages.push(serde_json::json!({"role": "user", "content": "failed"}));
        messages.pop(); // simulates error path
        assert_eq!(messages.len(), 2, "failed turn should be popped");
        assert_eq!(messages[1]["content"], "ok");
    }

    #[test]
    fn replay_empty_session_yields_empty_turns() {
        let sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        let turns = extract_user_turns(&sess);
        assert!(turns.is_empty(), "empty session should have no user turns");
    }

    #[test]
    fn replay_system_messages_not_extracted() {
        let mut sess = Session::new("/tmp".to_string(), "gemma4:26b".to_string());
        sess.push_message(serde_json::json!({"role": "system", "content": "system prompt"}));
        sess.push_message(serde_json::json!({"role": "user", "content": "user msg"}));
        let turns = extract_user_turns(&sess);
        assert_eq!(turns.len(), 1, "only user messages should be extracted");
        assert_eq!(turns[0], "user msg");
    }
}
