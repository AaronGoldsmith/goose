use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::{json, Value};

use super::types::content::{Content, Text};
use super::types::message::{Message, Role};
use crate::errors::AgentError;
use crate::tool::Tool;

/// Convert internal Message format to OpenAI's API message specification
pub fn messages_to_openai_spec(messages: &[Message]) -> Vec<Value> {
    let mut messages_spec = Vec::new();

    for message in messages {
        let mut converted = json!({
            "role": message.role
        });

        let mut output = Vec::new();

        for content in &message.content {
            match content {
                Content::Text(Text { text }) => {
                    converted["content"] = json!(text);
                }
                Content::ToolRequest(tool_request) => match &tool_request.call {
                    Ok(tool_call) => {
                        let sanitized_name = sanitize_function_name(&tool_call.name);
                        let tool_calls = converted
                            .as_object_mut()
                            .unwrap()
                            .entry("tool_calls")
                            .or_insert(json!([]));

                        tool_calls.as_array_mut().unwrap().push(json!({
                            "id": tool_request.id,
                            "type": "function",
                            "function": {
                                "name": sanitized_name,
                                "arguments": tool_call.parameters.to_string(),
                            }
                        }));
                    }
                    Err(e) => {
                        output.push(json!({
                            "role": "tool",
                            "content": format!("Error: {}", e),
                            "tool_call_id": tool_request.id
                        }));
                    }
                },
                Content::ToolResponse(tool_response) => {
                    match &tool_response.output {
                        Ok(value) => {
                            // Handle image results
                            if let Some(s) = value.as_str() {
                                if s.starts_with("image:") {
                                    output.push(json!({
                                        "role": "tool",
                                        "content": [{
                                            "type": "text",
                                            "text": "This tool result included an image that is uploaded in the next message."
                                        }],
                                        "tool_call_id": tool_response.request_id
                                    }));
                                    continue;
                                }
                            }
                            output.push(json!({
                                "role": "tool",
                                "content": value.to_string(),
                                "tool_call_id": tool_response.request_id
                            }));
                        }
                        Err(e) => {
                            output.push(json!({
                                "role": "tool",
                                "content": format!("Error: {}", e),
                                "tool_call_id": tool_response.request_id
                            }));
                        }
                    }
                }
            }
        }

        if converted.get("content").is_some() || converted.get("tool_calls").is_some() {
            output.insert(0, converted);
        }
        messages_spec.extend(output);
    }

    messages_spec
}

/// Convert internal Tool format to OpenAI's API tool specification
pub fn tools_to_openai_spec(tools: &[Tool]) -> Result<Vec<Value>> {
    let mut tool_names = std::collections::HashSet::new();
    let mut result = Vec::new();

    for tool in tools {
        if !tool_names.insert(&tool.name) {
            return Err(anyhow!("Duplicate tool name: {}", tool.name));
        }

        result.push(json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }));
    }

    Ok(result)
}

/// Convert OpenAI's API response to internal Message format
pub fn openai_response_to_message(response: Value) -> Result<Message> {
    let original = response["choices"][0]["message"].clone();
    let mut content = Vec::new();

    if let Some(text) = original.get("content") {
        if let Some(text_str) = text.as_str() {
            content.push(Content::text(text_str));
        }
    }

    if let Some(tool_calls) = original.get("tool_calls") {
        if let Some(tool_calls_array) = tool_calls.as_array() {
            for tool_call in tool_calls_array {
                let id = tool_call["id"].as_str().unwrap_or_default().to_string();
                let function_name = tool_call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let arguments = tool_call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();

                if !is_valid_function_name(&function_name) {
                    let error = AgentError::ToolNotFound(format!(
                        "The provided function name '{}' had invalid characters, it must match this regex [a-zA-Z0-9_-]+",
                        function_name
                    ));
                    content.push(Content::tool_request_error(id, error));
                } else {
                    match serde_json::from_str::<Value>(&arguments) {
                        Ok(params) => {
                            content.push(Content::tool_request_success(id, function_name, params));
                        }
                        Err(e) => {
                            let error = AgentError::InvalidParameters(format!(
                                "Could not interpret tool use parameters for id {}: {}",
                                id, e
                            ));
                            content.push(Content::tool_request_error(id, error));
                        }
                    }
                }
            }
        }
    }

    Message::new(Role::Assistant, content)
}

fn sanitize_function_name(name: &str) -> String {
    let re = Regex::new(r"[^a-zA-Z0-9_-]").unwrap();
    re.replace_all(name, "_").to_string()
}

fn is_valid_function_name(name: &str) -> bool {
    let re = Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap();
    re.is_match(name)
}

#[derive(Debug, thiserror::Error)]
#[error("Input message too long. Message: {0}")]
pub struct InitialMessageTooLargeError(String);

pub fn check_openai_context_length_error(error: &Value) -> Option<InitialMessageTooLargeError> {
    let code = error.get("code")?.as_str()?;
    if code == "context_length_exceeded" || code == "string_above_max_length" {
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error")
            .to_string();
        Some(InitialMessageTooLargeError(message))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const OPENAI_TOOL_USE_RESPONSE: &str = r#"{
        "choices": [{
            "role": "assistant",
            "message": {
                "tool_calls": [{
                    "id": "1",
                    "function": {
                        "name": "example_fn",
                        "arguments": "{\"param\": \"value\"}"
                    }
                }]
            }
        }],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 25,
            "total_tokens": 35
        }
    }"#;

    #[test]
    fn test_messages_to_openai_spec() -> Result<()> {
        let message = Message::user("Hello")?;
        let spec = messages_to_openai_spec(&[message]);

        assert_eq!(spec.len(), 1);
        assert_eq!(spec[0]["role"], "user");
        assert_eq!(spec[0]["content"], "Hello");
        Ok(())
    }

    #[test]
    fn test_tools_to_openai_spec() -> Result<()> {
        let parameters = json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Test parameter"
                }
            },
            "required": ["input"]
        });

        let tool = Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters,
        };

        let spec = tools_to_openai_spec(&[tool])?;

        assert_eq!(spec.len(), 1);
        assert_eq!(spec[0]["type"], "function");
        assert_eq!(spec[0]["function"]["name"], "test_tool");
        Ok(())
    }

    #[test]
    fn test_sanitize_function_name() {
        assert_eq!(sanitize_function_name("hello-world"), "hello-world");
        assert_eq!(sanitize_function_name("hello world"), "hello_world");
        assert_eq!(sanitize_function_name("hello@world"), "hello_world");
    }

    #[test]
    fn test_is_valid_function_name() {
        assert!(is_valid_function_name("hello-world"));
        assert!(is_valid_function_name("hello_world"));
        assert!(!is_valid_function_name("hello world"));
        assert!(!is_valid_function_name("hello@world"));
    }

    #[test]
    fn test_messages_to_openai_spec_complex() -> Result<()> {
        let mut messages = vec![
            Message::assistant("Hello!")?,
            Message::user("How are you?")?,
            Message::new(
                Role::Assistant,
                vec![Content::tool_request("tool1", json!({"param1": "value1"}))],
            )?,
        ];
        let Content::ToolRequest(request) = &messages[2].content[0] else {
            panic!("should be request")
        };
        messages.push(Message::new(
            Role::User,
            vec![Content::tool_response(&request.id, "Result".into())],
        )?);

        let spec = messages_to_openai_spec(&messages);

        assert_eq!(spec.len(), 4);
        assert_eq!(spec[0]["role"], "assistant");
        assert_eq!(spec[0]["content"], "Hello!");
        assert_eq!(spec[1]["role"], "user");
        assert_eq!(spec[1]["content"], "How are you?");
        assert_eq!(spec[2]["role"], "assistant");
        assert!(spec[2]["tool_calls"].is_array());
        assert_eq!(spec[3]["role"], "tool");
        assert_eq!(spec[3]["content"], "\"Result\"");
        assert_eq!(spec[3]["tool_call_id"], spec[2]["tool_calls"][0]["id"]);

        Ok(())
    }

    #[test]
    fn test_tools_to_openai_spec_duplicate() -> Result<()> {
        let parameters = json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Test parameter"
                }
            },
            "required": ["input"]
        });

        let tool1 = Tool {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: parameters.clone(),
        };

        let tool2 = Tool {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
            parameters: parameters.clone(),
        };

        let result = tools_to_openai_spec(&[tool1, tool2]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Duplicate tool name"));

        Ok(())
    }

    #[test]
    fn test_tools_to_openai_spec_empty() -> Result<()> {
        let spec = tools_to_openai_spec(&[])?;
        assert!(spec.is_empty());
        Ok(())
    }

    #[test]
    fn test_openai_response_to_message_text() -> Result<()> {
        let response = json!({
            "choices": [{
                "role": "assistant",
                "message": {
                    "content": "Hello from John Cena!"
                }
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25,
                "total_tokens": 35
            }
        });

        let message = openai_response_to_message(response)?;
        assert_eq!(message.text(), "Hello from John Cena!");
        assert!(matches!(message.role, Role::Assistant));

        Ok(())
    }

    #[test]
    fn test_openai_response_to_message_valid_toolrequest() -> Result<()> {
        let response: Value = serde_json::from_str(OPENAI_TOOL_USE_RESPONSE)?;
        let message = openai_response_to_message(response)?;

        let tool_requests = message.tool_request();
        assert_eq!(tool_requests.len(), 1);
        let tool_call = tool_requests[0].call.as_ref().unwrap();
        assert_eq!(tool_call.name, "example_fn");
        assert_eq!(tool_call.parameters, json!({"param": "value"}));

        Ok(())
    }

    #[test]
    fn test_openai_response_to_message_invalid_func_name() -> Result<()> {
        let mut response: Value = serde_json::from_str(OPENAI_TOOL_USE_RESPONSE)?;
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] =
            json!("invalid fn");

        let message = openai_response_to_message(response)?;
        let tool_requests = message.tool_request();

        match &tool_requests[0].call {
            Err(AgentError::ToolNotFound(msg)) => {
                assert!(msg.starts_with("The provided function name"));
            }
            _ => panic!("Expected ToolNotFound error"),
        }

        Ok(())
    }

    #[test]
    fn test_openai_response_to_message_json_decode_error() -> Result<()> {
        let mut response: Value = serde_json::from_str(OPENAI_TOOL_USE_RESPONSE)?;
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] =
            json!("invalid json {");

        let message = openai_response_to_message(response)?;
        let tool_requests = message.tool_request();

        match &tool_requests[0].call {
            Err(AgentError::InvalidParameters(msg)) => {
                assert!(msg.starts_with("Could not interpret tool use parameters"));
            }
            _ => panic!("Expected InvalidParameters error"),
        }

        Ok(())
    }

    #[test]
    fn test_check_openai_context_length_error() {
        let error = json!({
            "code": "context_length_exceeded",
            "message": "This message is too long"
        });

        let result = check_openai_context_length_error(&error);
        assert!(result.is_some());
        assert_eq!(
            result.unwrap().to_string(),
            "Input message too long. Message: This message is too long"
        );

        let error = json!({
            "code": "other_error",
            "message": "Some other error"
        });

        let result = check_openai_context_length_error(&error);
        assert!(result.is_none());
    }
}
