use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::ToolCall;
use std::time::{SystemTime, UNIX_EPOCH};

use super::content::{Content, Text, ToolRequest, ToolResponse};
use super::objectid::create_object_id;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub id: String,
    pub created: i64,
    pub content: Vec<Content>,
}

impl Message {
    pub fn new(role: Role, content: Vec<Content>) -> Result<Self> {
        let msg = Self {
            role,
            id: create_object_id("msg"),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64,
            content,
        };
        msg.validate()?;
        Ok(msg)
    }

    fn validate(&self) -> Result<()> {
        match self.role {
            Role::User => {
                if !self.has_text() && !self.has_tool_response() {
                    return Err(anyhow!("User message must include a Text or ToolResponse"));
                }
                if self.has_tool_request() {
                    return Err(anyhow!("User message does not support ToolRequest"));
                }
            }
            Role::Assistant => {
                if !self.has_text() && !self.has_tool_request() {
                    return Err(anyhow!(
                        "Assistant message must include a Text or ToolRequest"
                    ));
                }
                if self.has_tool_response() {
                    return Err(anyhow!("Assistant message does not support ToolResponse"));
                }
            }
        }
        Ok(())
    }

    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|content| match content {
                Content::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn tool_request(&self) -> Vec<ToolRequest> {
        self.content
            .iter()
            .filter_map(|content| match content {
                Content::ToolRequest(tool_request) => Some(tool_request.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_response(&self) -> Vec<ToolResponse> {
        self.content
            .iter()
            .filter_map(|content| match content {
                Content::ToolResponse(tool_response) => Some(tool_response.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn has_text(&self) -> bool {
        self.content.iter().any(|c| matches!(c, Content::Text(_)))
    }

    pub fn has_tool_request(&self) -> bool {
        self.content
            .iter()
            .any(|c| matches!(c, Content::ToolRequest(_)))
    }

    pub fn has_tool_response(&self) -> bool {
        self.content
            .iter()
            .any(|c| matches!(c, Content::ToolResponse(_)))
    }

    pub fn user<S: Into<String>>(text: S) -> Result<Self> {
        Self::new(Role::User, vec![Content::Text(Text::new(text))])
    }

    pub fn assistant<S: Into<String>>(text: S) -> Result<Self> {
        Self::new(Role::Assistant, vec![Content::Text(Text::new(text))])
    }

    pub fn assistant_with_tool_calls(
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Result<Self> {
        let mut content = Vec::new();
        if let Some(text) = text {
            content.push(Content::Text(Text::new(text)));
        }
        for tool_call in tool_calls {
            content.push(Content::tool_request(&tool_call.name, tool_call.parameters));
        }
        Self::new(Role::Assistant, content)
    }

    pub fn tool_result(id: String, output: Value) -> Result<Self> {
        Self::new(Role::User, vec![Content::tool_response(&id, output)])
    }

    pub fn summary(&self) -> String {
        let content_summaries: Vec<String> = self.content.iter().map(|c| c.summary()).collect();
        format!("message:{:?}\n{}", self.role, content_summaries.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn test_user_message() -> Result<()> {
        let user_message = Message::user("abcd")?;
        assert!(matches!(user_message.role, Role::User));
        assert_eq!(user_message.text(), "abcd");
        Ok(())
    }

    #[test]
    fn test_assistant_message() -> Result<()> {
        let assistant_message = Message::assistant("abcd")?;
        assert!(matches!(assistant_message.role, Role::Assistant));
        assert_eq!(assistant_message.text(), "abcd");
        Ok(())
    }

    #[test]
    fn test_message_tool_request() -> Result<()> {
        let message = Message::new(
            Role::Assistant,
            vec![
                Content::tool_request("tool", json!({})),
                Content::tool_request("tool", json!({})),
            ],
        )?;

        let tool_requests = message.tool_request();
        assert_eq!(tool_requests.len(), 2);
        assert_eq!(tool_requests[0].clone().call.unwrap().name, "tool");
        Ok(())
    }

    #[test]
    fn test_message_tool_response() -> Result<()> {
        let message = Message::new(
            Role::User,
            vec![
                Content::tool_response("1", json!("result")),
                Content::tool_response("2", json!("result")),
            ],
        )?;

        let tool_responses = message.tool_response();
        assert_eq!(tool_responses.len(), 2);
        assert_eq!(tool_responses[0].clone().output.unwrap(), json!("result"));
        Ok(())
    }

    #[test]
    fn test_message_validation() -> Result<()> {
        // Valid user message
        let message = Message::user("Hello")?;
        assert_eq!(message.text(), "Hello");

        // Valid assistant message
        let message = Message::assistant("Hello")?;
        assert_eq!(message.text(), "Hello");

        // Invalid message: user with tool_request
        let response = Message::new(
            Role::User,
            vec![Content::text(""), Content::tool_request("tool", json!({}))],
        );
        assert!(response.is_err());

        // Invalid message: assistant with tool_response
        let response = Message::new(
            Role::Assistant,
            vec![
                Content::text(""),
                Content::tool_response("1", json!("result")),
            ],
        );
        assert!(response.is_err());
        Ok(())
    }

    #[test]
    fn test_serialization() -> Result<()> {
        // Test basic message serialization
        let message = Message::user("Hello, world!")?;
        let serialized = serde_json::to_string(&message)?;
        let deserialized: Message = serde_json::from_str(&serialized)?;
        assert_eq!(message.text(), deserialized.text());
        assert!(matches!(deserialized.role, Role::User));

        // Test complex message with tool use
        let message = Message::new(
            Role::Assistant,
            vec![
                Content::text("Using tool"),
                Content::tool_request("test_tool", json!({"key": "value"})),
            ],
        )?;

        let serialized = serde_json::to_string(&message)?;
        let deserialized: Message = serde_json::from_str(&serialized)?;

        assert_eq!(message.text(), deserialized.text());
        assert_eq!(
            message.tool_request().len(),
            deserialized.tool_request().len()
        );
        assert_eq!(
            message.tool_request()[0].clone().call.unwrap().name,
            deserialized.tool_request()[0].clone().call.unwrap().name
        );

        // Verify JSON structure
        let json_value: Value = serde_json::from_str(&serialized)?;
        assert!(json_value.get("id").is_some());
        assert!(json_value.get("created").is_some());
        assert!(json_value.get("role").is_some());
        assert!(json_value.get("content").is_some());

        Ok(())
    }
}
