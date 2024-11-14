use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;

use super::base::{Provider, Usage};
use super::configs::DatabricksProviderConfig;
use super::types::message::Message;
use super::utils::{
    check_openai_context_length_error, messages_to_openai_spec, openai_response_to_message,
    tools_to_openai_spec,
};
use crate::tool::Tool;

pub struct DatabricksProvider {
    client: Client,
    config: DatabricksProviderConfig,
}

impl DatabricksProvider {
    pub fn new(config: DatabricksProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600)) // 10 minutes timeout
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert("Authorization", format!("Bearer {}", config.token).parse()?);
                headers
            })
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

    async fn post(&self, model: &str, payload: Value) -> Result<Value> {
        let url = format!(
            "{}/serving-endpoints/{}/invocations",
            self.config.host.trim_end_matches('/'),
            model
        );

        let response = self.client.post(&url).json(&payload).send().await?;

        match response.status() {
            StatusCode::OK => Ok(response.json().await?),
            status if status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error() => {
                // Implement retry logic here if needed
                Err(anyhow!("Server error: {}", status))
            }
            _ => {
                let status = response.status();
                let err_text = response.text().await.unwrap_or_default();
                Err(anyhow!("Request failed: {}: {}", status, err_text))
            }
        }
    }
}

#[async_trait]
impl Provider for DatabricksProvider {
    async fn complete(
        &self,
        model: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<(Message, Usage)> {
        // Prepare messages and tools
        let messages_spec = messages_to_openai_spec(messages);
        let tools_spec = if !tools.is_empty() {
            tools_to_openai_spec(tools)?
        } else {
            vec![]
        };

        // Build payload with system message
        let mut messages_array = vec![json!({ "role": "system", "content": system })];
        messages_array.extend(messages_spec);

        let mut payload = json!({ "messages": messages_array });

        // Add optional parameters
        if !tools_spec.is_empty() {
            payload["tools"] = json!(tools_spec);
        }
        if let Some(temp) = temperature {
            payload["temperature"] = json!(temp);
        }
        if let Some(tokens) = max_tokens {
            payload["max_tokens"] = json!(tokens);
        }

        // Remove null values
        let payload = serde_json::Value::Object(
            payload
                .as_object()
                .unwrap()
                .iter()
                .filter(|&(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );

        // Make request
        let response = self.post(model, payload).await?;

        // Handle errors
        if let Some(error) = response.get("error") {
            if messages.len() == 1 {
                if let Some(err) = check_openai_context_length_error(error) {
                    return Err(err.into());
                }
            }
            return Err(anyhow!("Databricks API error: {}", error));
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
    use anyhow::Result;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_databricks_completion() -> Result<()> {
        // Start a mock server
        let mock_server = MockServer::start().await;

        // Mock response for completion
        let mock_response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35
            }
        });

        // Expected request body
        let system = "You are a helpful assistant.";
        let expected_request_body = json!({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "Hello"}
            ]
        });

        // Set up the mock to intercept the request and respond with the mocked response
        Mock::given(method("POST"))
            .and(path("/serving-endpoints/my-databricks-model/invocations"))
            .and(header("Authorization", "Bearer test_token"))
            .and(body_json(expected_request_body.clone()))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_response))
            .expect(1) // Expect exactly one matching request
            .mount(&mock_server)
            .await;

        // Create the DatabricksProvider with the mock server's URL as the host
        let config = DatabricksProviderConfig {
            host: mock_server.uri(),
            token: "test_token".to_string(),
        };

        let provider = DatabricksProvider::new(config)?;

        // Prepare input
        let model = "my-databricks-model";
        let messages = vec![Message::user("Hello")?];
        let tools = vec![]; // Empty tools list

        // Call the complete method
        let (reply_message, reply_usage) = provider
            .complete(
                model, system, &messages, &tools, None, // temperature
                None, // max_tokens
            )
            .await?;

        // Assert the response
        assert_eq!(reply_message.text(), "Hello!");
        assert_eq!(reply_usage.total_tokens, Some(35));

        Ok(())
    }
}
