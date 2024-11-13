use super::{
    base::Provider, configs::ProviderConfig, databricks::DatabricksProvider, ollama::OllamaProvider,
    openai::OpenAiProvider,
};
use anyhow::Error;

pub enum ProviderType {
    OpenAi,
    Databricks,
    Ollama,
}

pub async fn get_provider(
    provider_type: ProviderType,
    config: ProviderConfig,
) -> Result<Box<dyn Provider + Send + Sync>, Error> {
    match (provider_type, config) {
        (ProviderType::OpenAi, ProviderConfig::OpenAi(openai_config)) => {
            Ok(Box::new(OpenAiProvider::new(openai_config)?))
        }
        (ProviderType::Databricks, ProviderConfig::Databricks(databricks_config)) => {
            Ok(Box::new(DatabricksProvider::new(databricks_config)?))
        }
        (ProviderType::Ollama, ProviderConfig::Ollama(ollama_config)) => {
            Ok(Box::new(OllamaProvider::new(ollama_config).await?))
        }
        _ => Err(Error::msg("Provider type and config mismatch")),
    }
}
