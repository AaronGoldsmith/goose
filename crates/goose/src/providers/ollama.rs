use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::task::spawn_blocking;

use super::base::{Provider, Usage};
use super::configs::OllamaProviderConfig;
use super::types::message::Message;
use super::utils::{messages_to_openai_spec, openai_response_to_message, tools_to_openai_spec};
use crate::tool::Tool;

pub const OLLAMA_HOST: &str = "http://localhost:11434";
const OLLAMA_MODEL: &str = "qwen2.5";

pub struct OllamaProvider {
    client: Client,
    config: OllamaProviderConfig,
}

impl OllamaProvider {
    pub async fn new(config: OllamaProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600)) // 10 minutes timeout
            .build()?;

        // Verify Ollama is running by checking health endpoint
        let health_url = format!("{}", config.host.trim_end_matches('/'));
        let health_url_clone = health_url.clone();
        
        // Use spawn_blocking for the health check since reqwest::blocking::get is blocking
        let health_check = spawn_blocking(move || {
            reqwest::blocking::get(&health_url_clone)
        }).await??;

        if !health_check.status().is_success() {
            return Err(anyhow!("Ollama service is not available at {}", health_url));
        }

        Ok(Self { client, config })
    }

    fn get_usage(data: &Value) -> Result<Usage> {
        let usage = data
            .get("usage")
            .ok_or_else(|| anyhow!("No usage data in response"))?;

        let input_tokens = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);

        let output_tokens = usage
            .get("completion_tokens")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);

        let total_tokens = usage
            .get("total_tokens")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .or_else(|| match (input_tokens, output_tokens) {
                (Some(input), Some(output)) => Some(input + output),
                _ => None,
            });

        Ok(Usage::new(input_tokens, output_tokens, total_tokens))
    }

    async fn post(&self, payload: Value) -> Result<Value> {
        let url = format!(
            "{}/v1/chat/completions",
            self.config.host.trim_end_matches('/')
        );

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        match response.status() {
            StatusCode::OK => Ok(response.json().await?),
            status if status == StatusCode::TOO_MANY_REQUESTS || status.as_u16() >= 500 => {
                Err(anyhow!("Server error: {}", status))
            }
            _ => Err(anyhow!("Request failed: {}\nPayload: {}", response.status(), payload)),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn complete(
        &self,
        model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<(Message, Usage)> {
        let system_message = json!({
            "role": "system",
            "content": system
        });

        let messages_spec = messages_to_openai_spec(messages);
        let tools_spec = if !tools.is_empty() {
            tools_to_openai_spec(tools)?
        } else {
            vec![]
        };

        let mut messages_array = vec![system_message];
        messages_array.extend(messages_spec);

        let mut payload = json!({
            "model": model,
            "messages": messages_array
        });

        if !tools_spec.is_empty() {
            payload
                .as_object_mut()
                .unwrap()
                .insert("tools".to_string(), json!(tools_spec));
        }
        if let Some(temp) = temperature {
            payload
                .as_object_mut()
                .unwrap()
                .insert("temperature".to_string(), json!(temp));
        }
        if let Some(tokens) = max_tokens {
            payload
                .as_object_mut()
                .unwrap()
                .insert("max_tokens".to_string(), json!(tokens));
        }

        let response = self.post(payload).await?;

        // Parse response
        let message = openai_response_to_message(response.clone())?;
        let usage = Self::get_usage(&response)?;

        Ok((message, usage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Helper function to create a provider with mock server
    async fn setup_mock_provider(response_body: Value) -> Result<OllamaProvider> {
        let mock_server = MockServer::start().await;
        
        // Mock the health check endpoint
        Mock::given(method("GET"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        // Mock the completions endpoint
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
            .mount(&mock_server)
            .await;

        let config = OllamaProviderConfig {
            host: mock_server.uri(),
        };

        OllamaProvider::new(config).await
    }

    // Helper function to create a provider for integration tests
    async fn setup_integration_provider() -> Result<OllamaProvider> {
        let config = OllamaProviderConfig {
            host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| String::from("http://localhost:11434")),
        };
        OllamaProvider::new(config).await
    }

    // Helper function for completion tests
    async fn test_completion(provider: &OllamaProvider) -> Result<(Message, Usage)> {
        let messages = vec![Message::user("Hello?")?];
        provider
            .complete(
                OLLAMA_MODEL,
                "You are a helpful assistant.",
                &messages,
                &[],
                None,
                None,
            )
            .await
    }

    // Helper function for tool tests
    async fn test_tools(provider: &OllamaProvider) -> Result<(Message, Usage)> {
        let messages = vec![Message::user("Can you read the test.txt file?")?];
        let tool = Tool::new(
            "read_file",
            "Read the content of a file",
            json!({
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read"
                    }
                },
                "required": ["filename"]
            }),
        );

        provider
            .complete(
                OLLAMA_MODEL,
                "You are a helpful assistant.",
                &messages,
                &[tool],
                None,
                None,
            )
            .await
    }

    #[tokio::test]
    async fn test_complete_basic() -> Result<()> {
        let response_body = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm here to help. How can I assist you today? 😊",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 17,
                "total_tokens": 29
            }
        });

        let provider = setup_mock_provider(response_body).await?;
        let (message, usage) = test_completion(&provider).await?;

        assert_eq!(message.text(), "Hello! I'm here to help. How can I assist you today? 😊");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(17));
        assert_eq!(usage.total_tokens, Some(29));

        Ok(())
    }

    #[tokio::test]
    async fn test_complete_tool_request() -> Result<()> {
        let response_body = json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_h5d3s25w",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"filename\":\"test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 63,
                "completion_tokens": 70,
                "total_tokens": 133
            }
        });

        let provider = setup_mock_provider(response_body).await?;
        let (message, usage) = test_tools(&provider).await?;

        let tool_requests = message.tool_request();
        assert_eq!(tool_requests.len(), 1);
        let Ok(tool_call) = &tool_requests[0].call else { panic!("should be tool call") };

        assert_eq!(tool_call.name, "read_file");
        assert_eq!(tool_call.parameters, json!({"filename": "test.txt"}));

        assert_eq!(usage.input_tokens, Some(63));
        assert_eq!(usage.output_tokens, Some(70));
        assert_eq!(usage.total_tokens, Some(133));

        Ok(())
    }

    // Integration tests that run against a real Ollama server
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_complete_integration() -> Result<()> {
        let provider = setup_integration_provider().await?;
        let (message, _) = test_completion(&provider).await?;
        
        // Less strict assertions for integration test
        assert!(!message.text().is_empty());
        println!("Integration test completion response: {}", message.text());
        
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_tools_integration() -> Result<()> {
        let provider = setup_integration_provider().await?;
        let (message, _) = test_tools(&provider).await?;

        let tool_requests = message.tool_request();
        assert!(!tool_requests.is_empty());
        
        let Ok(tool_call) = &tool_requests[0].call else { panic!("should be tool call") };
        assert_eq!(tool_call.name, "read_file");
        assert!(tool_call.parameters.get("filename").is_some());

        Ok(())
    }

    // Error case tests
    #[tokio::test]
    async fn test_server_error() -> Result<()> {
        let mock_server = MockServer::start().await;
        
        // Mock health check success but completions failure
        Mock::given(method("GET"))
            .and(path("/"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&mock_server)
            .await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock_server)
            .await;

        let config = OllamaProviderConfig {
            host: mock_server.uri(),
        };

        let provider = OllamaProvider::new(config).await?;
        let result = test_completion(&provider).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Server error: 500"));

        Ok(())
    }
}
