use crate::providers::ollama;

// Unified enum to wrap different provider configurations
pub enum ProviderConfig {
    OpenAi(OpenAiProviderConfig),
    Databricks(DatabricksProviderConfig),
    Ollama(OllamaProviderConfig),
}

// Define specific config structs for each provider
pub struct OpenAiProviderConfig {
    pub host: String,
    pub api_key: String,
}

pub struct DatabricksProviderConfig {
    pub host: String,
    pub token: String,
}

pub struct OllamaProviderConfig {
    pub host: String,
}

impl From<&std::env::Vars> for OllamaProviderConfig {
    fn from(_env: &std::env::Vars) -> Self {
        // Note: std::env::Vars doesn't provide a way to read values without consuming them
        // So we use std::env::var directly since we know the key we want
        let host = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| String::from(ollama::OLLAMA_HOST));

        Self { host }
    }
}
