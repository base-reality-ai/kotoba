use async_trait::async_trait;
use dark_matter::config::Config;
use dark_matter::host::{install_host_capabilities, HostCapabilities};
use dark_matter::tools::registry::default_registry;
use dark_matter::tools::{Tool, ToolResult};
use serde_json::{json, Value};

pub struct HostSkeletonCapabilities;

impl HostCapabilities for HostSkeletonCapabilities {
    fn tools(&self) -> Vec<Box<dyn Tool>> {
        vec![Box::new(HostEchoTool)]
    }
}

struct HostEchoTool;

#[async_trait]
impl Tool for HostEchoTool {
    fn name(&self) -> &'static str {
        "host_echo"
    }

    fn description(&self) -> &'static str {
        "Echoes a message through the host project capability layer."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo."
                }
            },
            "required": ["message"]
        })
    }

    async fn call(&self, args: Value) -> anyhow::Result<ToolResult> {
        let message = args.get("message").and_then(Value::as_str).unwrap_or("");
        Ok(ToolResult {
            content: message.to_string(),
            is_error: false,
        })
    }
}

pub fn install() -> anyhow::Result<()> {
    install_host_capabilities(Box::new(HostSkeletonCapabilities))?;
    Ok(())
}

pub async fn call_installed_host_echo(config: &Config, message: &str) -> anyhow::Result<String> {
    let registry = default_registry(
        "host-skeleton-demo",
        &config.config_dir,
        &config.ollama_base_url(),
        &config.model,
        &config.embed_model,
    );
    let result = registry
        .call("host_echo", json!({ "message": message }))
        .await?;

    Ok(result.content)
}
