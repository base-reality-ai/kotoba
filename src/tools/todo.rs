use super::{Tool, ToolResult};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

// ── Data model ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub id: String,
    pub content: String,
    pub status: TodoStatus,
    pub priority: TodoPriority,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn todos_path(config_dir: &Path, session_id: &str) -> PathBuf {
    config_dir
        .join("todos")
        .join(format!("{}.json", session_id))
}

pub fn read_todos(config_dir: &Path, session_id: &str) -> Vec<Todo> {
    let path = todos_path(config_dir, session_id);
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub fn write_todos(config_dir: &Path, session_id: &str, todos: &[Todo]) -> Result<()> {
    let path = todos_path(config_dir, session_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(todos)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Priority ordering: high < medium < low (lower = higher priority)
fn priority_order(p: &TodoPriority) -> u8 {
    match p {
        TodoPriority::High => 0,
        TodoPriority::Medium => 1,
        TodoPriority::Low => 2,
    }
}

pub fn format_todos(todos: &[Todo]) -> String {
    if todos.is_empty() {
        return "No todos.".to_string();
    }

    // Sort: in_progress first, then pending by priority, then completed
    let mut sorted = todos.to_vec();
    sorted.sort_by_key(|t| match t.status {
        TodoStatus::InProgress => (0u8, 0u8),
        TodoStatus::Pending => (1, priority_order(&t.priority)),
        TodoStatus::Completed => (2, priority_order(&t.priority)),
    });

    let lines: Vec<String> = sorted
        .iter()
        .map(|t| {
            let marker = match t.status {
                TodoStatus::Completed => "[x]",
                TodoStatus::InProgress => "[~]",
                TodoStatus::Pending => "[ ]",
            };
            let prio = match t.priority {
                TodoPriority::High => "HIGH",
                TodoPriority::Medium => "MED ",
                TodoPriority::Low => "LOW ",
            };
            format!("{} {}  {} (id: {})", marker, prio, t.content, t.id)
        })
        .collect();

    lines.join("\n")
}

// ── TodoWriteTool ─────────────────────────────────────────────────────────────

pub struct TodoWriteTool {
    session_id: String,
    config_dir: PathBuf,
}

impl TodoWriteTool {
    pub fn new(session_id: String, config_dir: PathBuf) -> Self {
        TodoWriteTool {
            session_id,
            config_dir,
        }
    }
}

#[async_trait]
impl Tool for TodoWriteTool {
    fn name(&self) -> &'static str {
        "todo_write"
    }

    fn description(&self) -> &'static str {
        "Create or replace the full todo list for the current task. Pass the complete \
         updated list each time — this replaces the previous list entirely. Use this \
         to track your progress on multi-step tasks and keep the user informed about \
         what you're doing and what's left."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete todo list, replacing any previous list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id":       { "type": "string", "description": "Unique identifier for this todo item." },
                            "content":  { "type": "string", "description": "Description of the task." },
                            "status":   { "type": "string", "enum": ["pending", "in_progress", "completed"] },
                            "priority": { "type": "string", "enum": ["high", "medium", "low"] }
                        },
                        "required": ["id", "content", "status", "priority"]
                    }
                }
            },
            "required": ["todos"]
        })
    }

    async fn call(&self, args: Value) -> Result<ToolResult> {
        let todos_val = args["todos"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: todos"))?;

        let todos: Vec<Todo> = serde_json::from_value(Value::Array(todos_val.clone()))
            .map_err(|e| anyhow::anyhow!("Invalid todos format: {}", e))?;

        write_todos(&self.config_dir, &self.session_id, &todos)
            .map_err(|e| anyhow::anyhow!("Failed to save todos: {}", e))?;

        let formatted = format_todos(&todos);
        let content = format!(
            "✓ Todo list updated ({} items)\n\n{}",
            todos.len(),
            formatted
        );

        Ok(ToolResult {
            content,
            is_error: false,
        })
    }
}

// ── TodoReadTool ──────────────────────────────────────────────────────────────

pub struct TodoReadTool {
    session_id: String,
    config_dir: PathBuf,
}

impl TodoReadTool {
    pub fn new(session_id: String, config_dir: PathBuf) -> Self {
        TodoReadTool {
            session_id,
            config_dir,
        }
    }
}

#[async_trait]
impl Tool for TodoReadTool {
    fn name(&self) -> &'static str {
        "todo_read"
    }

    fn description(&self) -> &'static str {
        "Read the current todo list for this session. Returns all todos with their \
         status and priority. Use this to check your progress and what remains."
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use todo to manage a task list for tracking multi-step work. \
              Add items before starting, check them off as you go.",
        )
    }

    async fn call(&self, _args: Value) -> Result<ToolResult> {
        let todos = read_todos(&self.config_dir, &self.session_id);
        if todos.is_empty() {
            return Ok(ToolResult {
                content: "No todos yet.".to_string(),
                is_error: false,
            });
        }
        Ok(ToolResult {
            content: format_todos(&todos),
            is_error: false,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_and_read_todos() {
        let dir = TempDir::new().unwrap();
        let writer = TodoWriteTool::new("test-session".into(), dir.path().to_path_buf());
        let reader = TodoReadTool::new("test-session".into(), dir.path().to_path_buf());

        let result = writer
            .call(json!({
                "todos": [{
                    "id": "1",
                    "content": "Implement feature",
                    "status": "pending",
                    "priority": "high"
                }]
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Implement feature"));
        assert!(result.content.contains("Todo list updated (1 items)"));

        let read_result = reader.call(json!({})).await.unwrap();
        assert!(!read_result.is_error);
        assert!(read_result.content.contains("Implement feature"));
    }

    #[tokio::test]
    async fn read_empty_returns_no_todos() {
        let dir = TempDir::new().unwrap();
        let reader = TodoReadTool::new("test-session".into(), dir.path().to_path_buf());
        let result = reader.call(json!({})).await.unwrap();
        assert_eq!(result.content, "No todos yet.");
    }

    #[tokio::test]
    async fn sorting_order() {
        let dir = TempDir::new().unwrap();
        let writer = TodoWriteTool::new("test-session".into(), dir.path().to_path_buf());

        writer
            .call(json!({
                "todos": [
                    {"id":"c","content":"Completed","status":"completed","priority":"high"},
                    {"id":"p","content":"Pending low","status":"pending","priority":"low"},
                    {"id":"i","content":"In progress","status":"in_progress","priority":"medium"},
                    {"id":"h","content":"Pending high","status":"pending","priority":"high"}
                ]
            }))
            .await
            .unwrap();

        let reader = TodoReadTool::new("test-session".into(), dir.path().to_path_buf());
        let result = reader.call(json!({})).await.unwrap();
        let lines: Vec<&str> = result.content.lines().collect();

        // in_progress comes first
        assert!(lines[0].contains("In progress"));
        // pending high before pending low
        assert!(lines[1].contains("Pending high"));
        assert!(lines[2].contains("Pending low"));
        // completed last
        assert!(lines[3].contains("Completed"));
    }

    #[test]
    fn format_todos_empty_returns_no_todos() {
        assert_eq!(format_todos(&[]), "No todos.");
    }

    #[test]
    fn format_todos_markers_match_status() {
        let todos = vec![
            Todo {
                id: "a".into(),
                content: "done".into(),
                status: TodoStatus::Completed,
                priority: TodoPriority::Low,
            },
            Todo {
                id: "b".into(),
                content: "wip".into(),
                status: TodoStatus::InProgress,
                priority: TodoPriority::High,
            },
            Todo {
                id: "c".into(),
                content: "todo".into(),
                status: TodoStatus::Pending,
                priority: TodoPriority::Medium,
            },
        ];
        let out = format_todos(&todos);
        assert!(out.contains("[x]"), "completed should have [x]: {out}");
        assert!(out.contains("[~]"), "in-progress should have [~]: {out}");
        assert!(out.contains("[ ]"), "pending should have [ ]: {out}");
    }

    #[test]
    fn priority_order_high_less_than_low() {
        assert!(priority_order(&TodoPriority::High) < priority_order(&TodoPriority::Low));
        assert!(priority_order(&TodoPriority::High) < priority_order(&TodoPriority::Medium));
        assert!(priority_order(&TodoPriority::Medium) < priority_order(&TodoPriority::Low));
    }

    #[test]
    fn todo_serde_round_trips() {
        let t = Todo {
            id: "abc".into(),
            content: "do something".into(),
            status: TodoStatus::InProgress,
            priority: TodoPriority::High,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Todo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "abc");
        assert_eq!(back.status, TodoStatus::InProgress);
        assert_eq!(back.priority, TodoPriority::High);
    }

    #[tokio::test]
    async fn write_tool_missing_todos_parameter_errors() {
        let dir = TempDir::new().unwrap();
        let writer = TodoWriteTool::new("s".into(), dir.path().to_path_buf());
        let err = writer.call(json!({})).await.unwrap_err();
        assert!(
            err.to_string().contains("todos"),
            "error should mention 'todos': {err}"
        );
    }

    #[tokio::test]
    async fn write_tool_invalid_todo_format_errors() {
        let dir = TempDir::new().unwrap();
        let writer = TodoWriteTool::new("s".into(), dir.path().to_path_buf());
        // Pass todos with an invalid status value
        let err = writer
            .call(
                json!({"todos": [{"id":"1","content":"x","status":"NOT_VALID","priority":"high"}]}),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("invalid") || err.to_string().contains("todos"),
            "error should indicate invalid format: {err}"
        );
    }

    #[test]
    fn read_todos_returns_empty_on_corrupt_json() {
        let dir = TempDir::new().unwrap();
        // Write garbage JSON
        let path = dir.path().join("todos").join("sess.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "not valid json {{{{").unwrap();
        let result = read_todos(dir.path(), "sess");
        assert!(result.is_empty(), "corrupt JSON should return empty vec");
    }

    #[test]
    fn write_and_read_todos_roundtrip() {
        let dir = TempDir::new().unwrap();
        let todos = vec![
            Todo {
                id: "x1".into(),
                content: "Task A".into(),
                status: TodoStatus::Pending,
                priority: TodoPriority::High,
            },
            Todo {
                id: "x2".into(),
                content: "Task B".into(),
                status: TodoStatus::Completed,
                priority: TodoPriority::Low,
            },
        ];
        write_todos(dir.path(), "mysession", &todos).unwrap();
        let loaded = read_todos(dir.path(), "mysession");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "x1");
        assert_eq!(loaded[1].status, TodoStatus::Completed);
    }

    #[test]
    fn format_todos_shows_all_priority_labels() {
        let todos = vec![
            Todo {
                id: "1".into(),
                content: "h".into(),
                status: TodoStatus::Pending,
                priority: TodoPriority::High,
            },
            Todo {
                id: "2".into(),
                content: "m".into(),
                status: TodoStatus::Pending,
                priority: TodoPriority::Medium,
            },
            Todo {
                id: "3".into(),
                content: "l".into(),
                status: TodoStatus::Pending,
                priority: TodoPriority::Low,
            },
        ];
        let out = format_todos(&todos);
        assert!(out.contains("HIGH"), "should show HIGH: {out}");
        assert!(out.contains("MED"), "should show MED: {out}");
        assert!(out.contains("LOW"), "should show LOW: {out}");
    }

    #[test]
    fn format_todos_includes_id_in_output() {
        let todos = vec![Todo {
            id: "unique-id-42".into(),
            content: "some task".into(),
            status: TodoStatus::Pending,
            priority: TodoPriority::Medium,
        }];
        let out = format_todos(&todos);
        assert!(
            out.contains("unique-id-42"),
            "output should include the todo id: {out}"
        );
    }
}
