use super::{Tool, ToolResult};
use crate::ollama::types::{FunctionDefinition, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

pub const TOOL_NAME: &str = "ask_user_question";

pub struct AskUserQuestionTool;

#[async_trait]
impl Tool for AskUserQuestionTool {
    fn name(&self) -> &'static str {
        TOOL_NAME
    }

    fn description(&self) -> &'static str {
        "Ask the user a question and wait for their response. Use this when you need \
         clarification, additional information, or a decision from the user before \
         proceeding. The user will see the question and can type a free-form answer, \
         or choose from provided options if given."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user."
                },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of suggested answers for the user to choose from."
                }
            },
            "required": ["question"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn system_prompt_hint(&self) -> Option<&'static str> {
        Some(
            "Use ask_user_question when you need clarification or a decision from the user \
              before proceeding. Prefer this over guessing.",
        )
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters(),
            },
        }
    }

    /// Never called — the agent intercepts `ask_user_question` before reaching the registry.
    async fn call(&self, _args: Value) -> Result<ToolResult> {
        anyhow::bail!("ask_user_question must be handled by the agent, not the registry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> AskUserQuestionTool {
        AskUserQuestionTool
    }

    #[test]
    fn name_is_ask_user_question() {
        assert_eq!(tool().name(), TOOL_NAME);
        assert_eq!(tool().name(), "ask_user_question");
    }

    #[test]
    fn description_is_non_empty() {
        assert!(!tool().description().is_empty());
    }

    #[test]
    fn parameters_has_required_question_field() {
        let params = tool().parameters();
        let required = params["required"].as_array().unwrap();
        assert!(
            required.iter().any(|v| v.as_str() == Some("question")),
            "parameters must require 'question'"
        );
    }

    #[test]
    fn parameters_question_is_string_type() {
        let params = tool().parameters();
        assert_eq!(
            params["properties"]["question"]["type"].as_str(),
            Some("string")
        );
    }

    #[test]
    fn parameters_options_is_array_type() {
        let params = tool().parameters();
        assert_eq!(
            params["properties"]["options"]["type"].as_str(),
            Some("array")
        );
    }

    #[test]
    fn definition_name_matches_tool_name() {
        let def = tool().definition();
        assert_eq!(def.function.name, TOOL_NAME);
    }

    #[test]
    fn definition_type_is_function() {
        let def = tool().definition();
        assert_eq!(def.tool_type, "function");
    }

    #[tokio::test]
    async fn call_returns_error_result() {
        let result = tool().call(json!({"question": "foo?"})).await;
        assert!(
            result.is_err(),
            "call() must bail — handled by agent, not registry"
        );
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("ask_user_question"),
            "error should mention tool name"
        );
    }
}
