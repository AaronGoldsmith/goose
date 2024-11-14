use crate::errors::{AgentError, AgentResult};
use crate::tool::ToolCall;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// Text content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Text {
    pub text: String,
}

impl Text {
    /// Create a new Text content with the given text
    pub fn new<S: Into<String>>(text: S) -> Self {
        Self { text: text.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// A request for a tool call from the agent
pub struct ToolRequest {
    /// A unique identifier for this tool request
    pub id: String,
    /// The requested tool call, which is a Result because it may be malformed
    pub call: AgentResult<ToolCall>,
}

impl ToolRequest {
    /// Create a new well-formed ToolRequest
    pub fn new<S: Into<String>>(name: S, parameters: Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            call: Ok(ToolCall::new(name, parameters)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResponse {
    /// The unique identifier for the original tool request
    pub request_id: String,
    /// The output of the tool call, which is a Result because the call may have errored
    // TODO should this be a Value or a string? the llms only accept strings today, but should
    // the decision to convert it be pushed into the provider?
    pub output: AgentResult<Value>,
}

impl ToolResponse {
    /// Create a new well-formed ToolRequest
    pub fn new<S: Into<String>>(request_id: S, output: Value) -> Self {
        Self {
            request_id: request_id.into(),
            output: Ok(output.into()),
        }
    }
}

// Enum to handle all content types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum Content {
    Text(Text),
    ToolRequest(ToolRequest),
    ToolResponse(ToolResponse),
}

impl Content {
    /// Create a new Text content
    pub fn text<S: Into<String>>(text: S) -> Self {
        Content::Text(Text::new(text))
    }

    /// Create a new ToolRequest with generated id
    pub fn tool_request<S: Into<String>>(name: S, parameters: Value) -> Self {
        Content::ToolRequest(ToolRequest::new(name, parameters))
    }

    /// Create a new ToolRequest content with a successful tool call
    pub fn tool_request_success<T: Into<String>, S: Into<String>>(
        id: T,
        name: S,
        parameters: Value,
    ) -> Self {
        Content::ToolRequest(ToolRequest {
            id: id.into(),
            call: Ok(ToolCall::new(name, parameters)),
        })
    }

    /// Create a new ToolRequest content with an error
    pub fn tool_request_error<S: Into<String>>(id: S, error: AgentError) -> Self {
        Content::ToolRequest(ToolRequest {
            id: id.into(),
            call: Err(error),
        })
    }

    /// Create a new ToolResponse content
    pub fn tool_response<S: Into<String>>(request_id: S, output: Value) -> Self {
        Content::ToolResponse(ToolResponse::new(request_id, output))
    }

    /// Create a new ToolResponse content with a successful result
    pub fn tool_response_success<S: Into<String>>(request_id: S, output: Value) -> Self {
        Content::ToolResponse(ToolResponse::new(request_id, output))
    }

    /// Create a new ToolResponse content with an error
    pub fn tool_response_error<S: Into<String>>(request_id: S, error: AgentError) -> Self {
        Content::ToolResponse(ToolResponse {
            request_id: request_id.into(),
            output: Err(error),
        })
    }

    pub fn summary(&self) -> String {
        match self {
            Content::Text(t) => t.text.clone(),
            Content::ToolRequest(t) => match &t.call {
                Ok(call) => format!(
                    "content:tool_use:{}\nparameters:{}",
                    call.name,
                    serde_json::to_string(&call.parameters).unwrap_or_default()
                ),
                Err(err) => format!("content:tool_use:error\nerror:{}", err.to_string()),
            },
            Content::ToolResponse(t) => match &t.output {
                Ok(value) => format!(
                    "content:tool_result\noutput:{}",
                    serde_json::to_string(&value).unwrap_or("".to_string())
                ),
                Err(err) => format!("content:tool_result:error\nerror:{}", err.to_string()),
            },
        }
    }
}
