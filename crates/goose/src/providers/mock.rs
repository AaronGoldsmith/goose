use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::Mutex;

use crate::providers::base::{Provider, Usage};
use crate::providers::types::content::Content;
use crate::providers::types::message::{Message, Role};
use crate::tool::Tool;

/// A mock provider that returns pre-configured responses for testing
pub struct MockProvider {
    responses: Arc<Mutex<Vec<Message>>>,
}

impl MockProvider {
    /// Create a new mock provider with a sequence of responses
    pub fn new(responses: Vec<Message>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn complete(
        &self,
        _model: &str,
        _system_prompt: &str,
        _messages: &[Message],
        _tools: &[Tool],
        _temperature: Option<f32>,
        _max_tokens: Option<i32>,
    ) -> Result<(Message, Usage)> {
        let mut responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            // Return empty response if no more pre-configured responses
            Ok((
                Message::new(Role::Assistant, vec![Content::text("")])
                    .expect("Failed to create message"),
                Usage::default(),
            ))
        } else {
            Ok((responses.remove(0), Usage::default()))
        }
    }
}
