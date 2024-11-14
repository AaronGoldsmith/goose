use serde::{Deserialize, Serialize};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AgentError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Tool execution failed: {0}")]
    ExecutionError(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid tool name format: {0}")]
    InvalidToolName(String),

}

pub type AgentResult<T> = Result<T, AgentError>;
