use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: &'static str, // "2.0"
    pub id: u64,
    pub method: String,
    pub params: Value,
}

#[derive(Deserialize)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// A tool as returned by the MCP `tools/list` method
#[derive(Debug, Clone, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_rpc_request_serializes() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let s = serde_json::to_string(&req).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["method"], "tools/list");
    }

    #[test]
    fn json_rpc_response_with_result_deserializes() {
        let raw = r#"{"id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.id, Some(1));
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn json_rpc_response_with_error_deserializes() {
        let raw = r#"{"id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    #[test]
    fn mcp_tool_with_description_deserializes() {
        let raw = r#"{"name":"bash","description":"Run a shell command","input_schema":{"type":"object"}}"#;
        let tool: McpTool = serde_json::from_str(raw).unwrap();
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description.as_deref(), Some("Run a shell command"));
        assert!(tool.input_schema.is_some());
    }

    #[test]
    fn mcp_tool_without_description_deserializes() {
        let raw = r#"{"name":"noop"}"#;
        let tool: McpTool = serde_json::from_str(raw).unwrap();
        assert_eq!(tool.name, "noop");
        assert!(tool.description.is_none());
        assert!(tool.input_schema.is_none());
    }

    #[test]
    fn json_rpc_response_null_id_deserializes() {
        let raw = r#"{"id":null,"result":{"ok":true}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(raw).unwrap();
        assert!(resp.id.is_none(), "null id should deserialize to None");
    }

    #[test]
    fn json_rpc_request_params_preserved() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 42,
            method: "initialize".to_string(),
            params: json!({"protocolVersion": "2024-11-05"}),
        };
        let s = serde_json::to_string(&req).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["params"]["protocolVersion"], "2024-11-05");
    }
}
