use anyhow::Result;
use serde_json::{Value, json};
use futures::stream::BoxStream;
use async_stream;

use crate::providers::base::Provider;
use crate::providers::types::message::{Message, Role};
use crate::providers::types::content::Content;
use crate::systems::System;
use crate::tool::{Tool, ToolCall};
use crate::errors::{AgentError, AgentResult};

/// Agent integrates a foundational LLM with the systems it needs to pilot
pub struct Agent {
    systems: Vec<Box<dyn System>>,
    provider: Box<dyn Provider>,
    model: String,
}

impl Agent {
    /// Create a new Agent with the specified provider and model
    pub fn new(provider: Box<dyn Provider>, model: String) -> Self {
        Self {
            systems: Vec::new(),
            provider,
            model,
        }
    }

    /// Add a system to the agent
    pub fn add_system(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
    }

    /// Get all tools from all systems with proper system prefixing
    fn get_prefixed_tools(&self) -> Vec<Tool> {
        let mut tools = Vec::new();
        for system in &self.systems {
            for tool in system.tools() {
                tools.push(Tool::new(
                    &format!("{}__{}", system.name(), tool.name),
                    &tool.description,
                    tool.parameters.clone(),
                ));
            }
        }
        tools
    }

    /// Find the appropriate system for a tool call based on the prefixed name
    fn get_system_for_tool(&self, prefixed_name: &str) -> Option<&Box<dyn System>> {
        let parts: Vec<&str> = prefixed_name.split("__").collect();
        if parts.len() != 2 {
            return None;
        }
        let system_name = parts[0];
        self.systems.iter().find(|sys| sys.name() == system_name)
    }

    /// Dispatch a single tool call to the appropriate system
    async fn dispatch_tool_call(&self, tool_call: ToolCall) -> AgentResult<Value> {
        let system = self.get_system_for_tool(&tool_call.name)
            .ok_or_else(|| AgentError::ToolNotFound(tool_call.name.clone()))?;

        let tool_name = tool_call.name.split("__").nth(1)
            .ok_or_else(|| AgentError::InvalidToolName(tool_call.name.clone()))?;
        let system_tool_call = ToolCall::new(tool_name, tool_call.parameters);
        
        system.call(system_tool_call).await
    }

    /// Create a stream that yields each message as it's generated by the agent.
    /// This includes both the assistant's responses and any tool responses.
    pub fn reply(&self, messages: &[Message]) -> BoxStream<'_, Result<Message>> {
        let mut messages = messages.to_vec();
        let tools = self.get_prefixed_tools();
        
        Box::pin(async_stream::try_stream! {
            loop {
                // Get completion from provider
                let (response, _) = self.provider.complete(
                    &self.model,
                    "You are an AI agent called Goose. You use tools of connected systems to solve problems.",
                    &messages,
                    &tools,
                    None,
                    None,
                ).await?;

                // Add the assistant's response to the conversation history
                messages.push(response.clone());
                
                // Yield the assistant's response
                yield response.clone();

                // If there are tool calls, handle them and continue the conversation
                if response.has_tool_request() {
                    // Handle all tool calls in the response
                    let mut content = Vec::new();
                    for tool_request in response.tool_request() {
                        let output = match &tool_request.call {
                            Ok(call) => match self.dispatch_tool_call(call.clone()).await {
                                Ok(value) => value,
                                Err(e) => json!({ "error": e.to_string() }),
                            },
                            Err(e) => json!({ "error": format!("Invalid parameters: {}", e) }),
                        };

                        content.push(Content::tool_response(&tool_request.id, output));
                    }

                    // Create and add the tool response message
                    let tool_response = Message::new(Role::User, content)
                        .expect("Failed to create tool response message");
                    messages.push(tool_response.clone());
                    yield tool_response;
                } else {
                    // No more tool calls, end the conversation
                    break;
                }
            }
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::mock::MockProvider;
    use crate::providers::types::content::Content;
    use async_trait::async_trait;
    use serde_json::json;
    use std::collections::HashMap;
    use futures::StreamExt;

    // Mock system for testing
    struct MockSystem {
        name: String,
        tools: Vec<Tool>,
    }

    impl MockSystem {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                tools: vec![Tool::new(
                    "echo",
                    "Echoes back the input",
                    json!({"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}),
                )],
            }
        }
    }

    #[async_trait]
    impl System for MockSystem {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock system for testing"
        }

        fn instructions(&self) -> &str {
            "Mock system instructions"
        }

        fn tools(&self) -> &[Tool] {
            &self.tools
        }

        async fn status(&self) -> anyhow::Result<HashMap<String, Value>> {
            Ok(HashMap::new())
        }

        async fn call(&self, tool_call: ToolCall) -> AgentResult<Value> {
            match tool_call.name.as_str() {
                "echo" => Ok(tool_call.parameters),
                _ => Err(AgentError::ToolNotFound(tool_call.name)),
            }
        }
    }

    #[tokio::test]
    async fn test_simple_response() -> Result<()> {
        let response = Message::new(
            Role::Assistant,
            vec![Content::text("Hello!")],
        )?;
        let provider = MockProvider::new(vec![response.clone()]);
        let agent = Agent::new(Box::new(provider), "test-model".to_string());

        let initial_message = Message::new(
            Role::User,
            vec![Content::text("Hi")],
        )?;
        let initial_messages = vec![initial_message];
        let messages: Vec<Message> = agent.reply(&initial_messages)
            .collect::<Vec<Result<Message>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<Message>>>()?;

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], response);
        Ok(())
    }

    #[tokio::test]
    async fn test_tool_call() -> Result<()> {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                Message::new(
                    Role::Assistant,
                    vec![Content::tool_request(
                        "test__echo",
                        json!({"message": "test"}),
                    )],
                )?,
                Message::new(
                    Role::Assistant,
                    vec![Content::text("Done!")],
                )?,
            ])),
            "test-model".to_string(),
        );

        agent.add_system(Box::new(MockSystem::new("test")));

        let initial_message = Message::new(
            Role::User,
            vec![Content::text("Echo test")],
        )?;
        let initial_messages = vec![initial_message];
        let messages: Vec<Message> = agent.reply(&initial_messages)
            .collect::<Vec<Result<Message>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<Message>>>()?;

        // Should have three messages: tool request, response, and model text
        assert_eq!(messages.len(), 3);
        assert!(messages[0].has_tool_request());
        assert_eq!(messages[2].content[0], Content::text("Done!"));
        Ok(())
    }

    #[tokio::test]
    async fn test_invalid_tool() -> Result<()> {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                Message::new(
                    Role::Assistant,
                    vec![Content::tool_request(
                        "invalid.tool",
                        json!({}),
                    )],
                )?,
                Message::new(
                    Role::Assistant,
                    vec![Content::text("Error occurred")],
                )?,
            ])),
            "test-model".to_string(),
        );

        agent.add_system(Box::new(MockSystem::new("test")));

        let initial_message = Message::new(
            Role::User,
            vec![Content::text("Invalid tool")],
        )?;
        let initial_messages = vec![initial_message];
        let messages: Vec<Message> = agent.reply(&initial_messages)
            .collect::<Vec<Result<Message>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<Message>>>()?;

        // Should have three messages: failed tool request, fail response, and model text
        assert_eq!(messages.len(), 3);
        assert!(messages[0].has_tool_request());
        assert_eq!(messages[2].content[0], Content::text("Error occurred"));
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_tool_calls() -> Result<()> {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                Message::new(
                    Role::Assistant,
                    vec![
                        Content::tool_request(
                            "test__echo",
                            json!({"message": "first"}),
                        ),
                        Content::tool_request(
                            "test__echo",
                            json!({"message": "second"}),
                        ),
                    ],
                )?,
                Message::new(
                    Role::Assistant,
                    vec![Content::text("All done!")],
                )?,
            ])),
            "test-model".to_string(),
        );

        agent.add_system(Box::new(MockSystem::new("test")));

        let initial_message = Message::new(
            Role::User,
            vec![Content::text("Multiple calls")],
        )?;
        let initial_messages = vec![initial_message];
        let messages: Vec<Message> = agent.reply(&initial_messages)
            .collect::<Vec<Result<Message>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<Message>>>()?;

        // Should have three messages: tool requests, response, and model text
        assert_eq!(messages.len(), 3);
        assert!(messages[0].has_tool_request());
        assert_eq!(messages[2].content[0], Content::text("All done!"));
        Ok(())
    }

    #[tokio::test]
    async fn test_conversation_flow() -> Result<()> {
        let mut agent = Agent::new(
            Box::new(MockProvider::new(vec![
                // First interaction: tool request
                Message::new(
                    Role::Assistant,
                    vec![Content::tool_request(
                        "test__echo",
                        json!({"message": "first"}),
                    )],
                )?,
                // Second interaction: final response
                Message::new(
                    Role::Assistant,
                    vec![Content::text("Done!")],
                )?,
            ])),
            "test-model".to_string(),
        );

        agent.add_system(Box::new(MockSystem::new("test")));

        let initial_message = Message::new(
            Role::User,
            vec![Content::text("Test conversation")],
        )?;
        let initial_messages = vec![initial_message];
        let messages: Vec<Message> = agent.reply(&initial_messages)
            .collect::<Vec<Result<Message>>>()
            .await
            .into_iter()
            .collect::<Result<Vec<Message>>>()?;

        assert_eq!(messages.len(), 3);
        
        // First message should be the tool request
        assert!(messages[0].has_tool_request());
        let tool_requests: Vec<_> = messages[0].tool_request().into_iter().collect();
        assert_eq!(tool_requests[0].call.as_ref().unwrap().name, "test__echo");
        
        // Second message should be the tool response
        assert_eq!(messages[1].content[0], Content::tool_response_success(tool_requests[0].id.clone(), json!({"message": "first"})));

        // Third message should be the final response
        assert_eq!(messages[2].content[0], Content::text("Done!"));

        Ok(())
    }
}