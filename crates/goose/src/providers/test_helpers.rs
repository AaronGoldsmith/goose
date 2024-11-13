#[cfg(test)]
use super::*;
use anyhow::Result;
use serde_json::json;

// Helper function to create a provider for integration tests
pub(crate) async fn setup_integration_provider() -> Result<OllamaProvider> {
    let config = OllamaProviderConfig {
        host: std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| String::from(ollama::OLLAMA_HOST)),
    };
    OllamaProvider::new(config).await
}

// Helper function for completion tests
pub(crate) async fn test_completion(provider: &OllamaProvider) -> Result<(Message, Usage)> {
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
pub(crate) async fn test_tools(provider: &OllamaProvider) -> Result<(Message, Usage)> {
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
