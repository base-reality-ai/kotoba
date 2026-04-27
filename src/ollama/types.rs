use serde::{Deserialize, Serialize};

/// A streaming chunk from POST /api/chat (one JSON object per line)
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    pub message: Option<ChunkMessage>,
    pub done: bool,
    pub done_reason: Option<String>,
    /// Tokens used for the prompt (present in the final done=true chunk)
    #[serde(default)]
    pub prompt_eval_count: u64,
    /// Tokens generated in this response (present in the final done=true chunk)
    #[serde(default)]
    pub eval_count: u64,
    /// Time spent generating tokens in nanoseconds (present in the final done=true chunk)
    #[serde(default)]
    pub eval_duration: u64,
}

/// Message content in a chunk — used for both streaming and non-streaming responses
#[derive(Debug, Clone, Deserialize)]
pub struct ChunkMessage {
    pub role: Option<String>,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

/// Response from GET /api/tags
#[derive(Debug, Deserialize)]
pub struct TagsResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub size: Option<u64>,
    pub modified_at: Option<String>,
    pub digest: Option<String>,
}

// ── Tool calling types ────────────────────────────────────────────────────────

/// A tool definition sent to Ollama in the chat request
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String, // always "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A tool call requested by the model in its response
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "deserialize_args")]
    pub arguments: serde_json::Value,
}

/// Deserialize tool call arguments tolerantly.
///
/// Ollama's smaller models sometimes return arguments as a JSON-encoded string
/// (`"{"key":"val"}"`) rather than an object (`{"key":"val"}`). This handles
/// both forms, and falls back to an empty object for anything else.
fn deserialize_args<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = serde_json::Value::deserialize(deserializer)?;
    match v {
        serde_json::Value::Object(_) => Ok(v),
        serde_json::Value::String(s) => match serde_json::from_str(&s) {
            Ok(v) => Ok(v),
            Err(e) => {
                crate::warnings::push_warning(format!(
                    "Malformed tool call arguments ({}), using empty args. Raw: {:.200}. Try: ask the model to retry with valid JSON tool arguments, or switch to a tool-capable model.",
                    e, s
                ));
                Ok(serde_json::json!({}))
            }
        },
        _ => {
            crate::warnings::push_warning(format!(
                "Unexpected tool call arguments type: {}, using empty args. Try: ask the model to retry with JSON object tool arguments, or switch to a tool-capable model.",
                v
            ));
            Ok(serde_json::json!({}))
        }
    }
}

/// Events produced by the streaming chat API (used by the TUI agent task)
#[derive(Debug)]
pub enum StreamEvent {
    /// A text token from the model
    Token(String),
    /// Reasoning/thinking content from reasoning models (DeepSeek-R1, `QwQ`, etc.)
    Thinking(String),
    /// Tool calls the model wants to make (accumulated from the stream)
    ToolCalls(Vec<ToolCall>),
    /// Stream is complete, with token usage counts
    Done {
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    /// A parse or transport error
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── deserialize_args ──────────────────────────────────────────────────────

    fn parse_tool_call(json: &str) -> serde_json::Value {
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        tc.function.arguments
    }

    #[test]
    fn deserialize_args_object_passthrough() {
        let json = r#"{"function":{"name":"bash","arguments":{"command":"echo hi"}}}"#;
        let args = parse_tool_call(json);
        assert_eq!(args["command"], "echo hi");
    }

    #[test]
    fn deserialize_args_json_encoded_string() {
        // Ollama sometimes wraps args as a JSON-encoded string
        let json = r#"{"function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}"#;
        let args = parse_tool_call(json);
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn deserialize_args_unknown_type_falls_back_to_empty_object() {
        let _g = crate::warnings::WARNINGS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::warnings::drain_warnings();
        // A bare integer is neither object nor string → falls back to {}
        let json = r#"{"function":{"name":"bash","arguments":42}}"#;
        let args = parse_tool_call(json);
        assert!(args.is_object());
        assert!(args.as_object().unwrap().is_empty());
        let warnings = crate::warnings::drain_warnings();
        assert!(
            warnings.iter().any(|w| w.contains("Try:")),
            "warning missing next-step hint: {warnings:?}"
        );
    }

    #[test]
    fn deserialize_args_malformed_string_falls_back_to_empty_object() {
        let _g = crate::warnings::WARNINGS_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::warnings::drain_warnings();
        let json = r#"{"function":{"name":"bash","arguments":"not valid json"}}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert!(tc.function.arguments.is_object());
        assert!(tc.function.arguments.as_object().unwrap().is_empty());
        let warnings = crate::warnings::drain_warnings();
        assert!(
            warnings.iter().any(|w| w.contains("Try:")),
            "warning missing next-step hint: {warnings:?}"
        );
    }

    #[test]
    fn deserialize_args_null_falls_back_to_empty_object() {
        let json = r#"{"function":{"name":"bash","arguments":null}}"#;
        let args = parse_tool_call(json);
        assert!(args.is_object());
        assert!(args.as_object().unwrap().is_empty());
    }

    // ── ChatResponse::empty ───────────────────────────────────────────────────

    #[test]
    fn chat_response_empty_has_zero_tokens() {
        let r = ChatResponse::empty();
        assert_eq!(r.prompt_tokens, 0);
        assert_eq!(r.completion_tokens, 0);
        assert_eq!(r.duration_ms, 0);
        assert!(r.message.content.is_empty());
    }

    #[test]
    fn chat_response_empty_role_is_assistant() {
        let r = ChatResponse::empty();
        assert_eq!(r.message.role.as_deref(), Some("assistant"));
    }

    // ── ChunkMessage defaults ──────────────────────────────────────────────────

    #[test]
    fn chunk_message_default_tool_calls_is_empty() {
        // Verify serde default: deserializing a minimal message gives no tool_calls
        let msg: ChunkMessage = serde_json::from_str(r#"{"content":"hi"}"#).unwrap();
        assert!(msg.tool_calls.is_empty());
        assert_eq!(msg.content, "hi");
    }

    #[test]
    fn deserialize_args_bool_falls_back_to_empty_object() {
        // true is neither object nor string → falls back to {}
        let json = r#"{"function":{"name":"bash","arguments":true}}"#;
        let tc: ToolCall = serde_json::from_str(json).unwrap();
        assert!(tc.function.arguments.is_object());
        assert!(tc.function.arguments.as_object().unwrap().is_empty());
    }

    #[test]
    fn chat_chunk_done_chunk_deserializes_token_counts() {
        let json = r#"{"done":true,"prompt_eval_count":10,"eval_count":20}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, 10);
        assert_eq!(chunk.eval_count, 20);
    }

    #[test]
    fn chat_chunk_non_done_has_zero_token_defaults() {
        let json = r#"{"done":false,"message":{"content":"token"}}"#;
        let chunk: ChatChunk = serde_json::from_str(json).unwrap();
        assert!(!chunk.done);
        assert_eq!(chunk.prompt_eval_count, 0);
        assert_eq!(chunk.eval_count, 0);
    }
}

/// Result of a non-streaming `chat()` call, including token usage
#[derive(Debug)]
pub struct ChatResponse {
    pub message: ChunkMessage,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub duration_ms: u64,
}

impl ChatResponse {
    /// Create an empty response (used as fallback when a nudge call fails).
    pub fn empty() -> Self {
        Self {
            message: ChunkMessage {
                role: Some("assistant".to_string()),
                content: String::new(),
                thinking: None,
                tool_calls: Vec::new(),
            },
            prompt_tokens: 0,
            completion_tokens: 0,
            duration_ms: 0,
        }
    }
}
