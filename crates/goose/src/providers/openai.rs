use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client; // we are using blocking API here to make sync calls
use reqwest::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;

use super::base::{Provider, Usage};
use super::configs::OpenAiProviderConfig;
use super::types::message::Message;
use super::utils::{
    check_openai_context_length_error, messages_to_openai_spec, openai_response_to_message,
    tools_to_openai_spec,
};
use crate::tool::Tool;

pub struct OpenAiProvider {
    client: Client,
    config: OpenAiProviderConfig,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600)) // 10 minutes timeout
            .build()?;

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
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&payload)
            .send()
            .await?;

        match response.status() {
            StatusCode::OK => Ok(response.json().await?),
            status if status == StatusCode::TOO_MANY_REQUESTS || status.as_u16() >= 500 => {
                // Implement retry logic here if needed
                Err(anyhow!("Server error: {}", status))
            }
            _ => Err(anyhow!(
                "Request failed: {}\nPayload: {}",
                response.status(),
                payload
            )),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn complete(
        &self,
        model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<(Message, Usage)> {
        // Not checking for o1 model here since system message is not supported by o1
        let system_message = json!({
            "role": "system",
            "content": system
        });

        // Convert messages and tools to OpenAI format
        let messages_spec = messages_to_openai_spec(messages);
        let tools_spec = if !tools.is_empty() {
            tools_to_openai_spec(tools)?
        } else {
            vec![]
        };

        // Build payload
        // create messages array with system message first
        let mut messages_array = vec![system_message];
        messages_array.extend(messages_spec);

        let mut payload = json!({
            "model": model,
            "messages": messages_array
        });

        // Add optional parameters
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

        //println!("{}", serde_json::to_string_pretty(&payload).unwrap());

        // Make request
        let response = self.post(payload).await?;

        // Check for context length error if single message
        if let Some(error) = response.get("error") {
            if messages.len() == 1 {
                if let Some(err) = check_openai_context_length_error(error) {
                    return Err(err.into());
                }
            }
            return Err(anyhow!("OpenAI API error: {}", error));
        }

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

    async fn _setup_mock_server(response_body: Value) -> (MockServer, OpenAiProvider) {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
            .mount(&mock_server)
            .await;

        // Create the OpenAiProvider with the mock server's URL as the host
        let config = OpenAiProviderConfig {
            host: mock_server.uri(),
            api_key: "test_api_key".to_string(),
        };

        let provider = OpenAiProvider::new(config).unwrap();
        (mock_server, provider)
    }

    #[tokio::test]
    async fn test_complete_basic() -> Result<()> {
        // Mock response for normal completion
        let response_body = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 15,
                "total_tokens": 27
            }
        });

        let (_, provider) = _setup_mock_server(response_body).await;

        // Prepare input messages
        let messages = vec![Message::user("Hello?")?];

        // Call the complete method
        let (message, usage) = provider
            .complete(
                "gpt-3.5-turbo",
                "You are a helpful assistant.",
                &messages,
                &[],
                None,
                None,
            )
            .await?;

        // Assert the response
        assert_eq!(message.text(), "Hello! How can I assist you today?");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(15));
        assert_eq!(usage.total_tokens, Some(27));

        Ok(())
    }

    #[tokio::test]
    async fn test_complete_tool_request() -> Result<()> {
        // Mock response for tool calling
        let response_body = json!({
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"San Francisco, CA\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        });

        let (_, provider) = _setup_mock_server(response_body).await;

        // Input messages
        let messages = vec![Message::user("What's the weather in San Francisco?")?];

        // Define the tool using builder pattern
        let tool = Tool::new(
            "get_weather",
            "Gets the current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. New York, NY"
                    }
                },
                "required": ["location"]
            }),
        );

        // Call the complete method
        let (message, usage) = provider
            .complete(
                "gpt-3.5-turbo",
                "You are a helpful assistant.",
                &messages,
                &[tool],
                Some(0.3),
                None,
            )
            .await?;

        // Assert the response
        let tool_requests = message.tool_request();
        assert_eq!(tool_requests.len(), 1);
        let Ok(tool_call) = &tool_requests[0].call else {
            panic!("should be tool call")
        };

        assert_eq!(tool_call.name, "get_weather");
        assert_eq!(
            tool_call.parameters,
            json!({"location": "San Francisco, CA"})
        );

        assert_eq!(usage.input_tokens, Some(20));
        assert_eq!(usage.output_tokens, Some(15));
        assert_eq!(usage.total_tokens, Some(35));

        Ok(())
    }
}
