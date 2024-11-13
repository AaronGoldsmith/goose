use axum::{
    extract::State,
    http::{self, HeaderMap, StatusCode},
    routing::post,
    Json, Router, response::IntoResponse,
};
use futures::{stream::StreamExt, Stream};
use goose::{
    agent::Agent,
    developer::DeveloperSystem,
    providers::{
        configs::{OpenAiProviderConfig, ProviderConfig},
        factory::{self, ProviderType},
        types::{
            content::Content,
            message::{Message, Role},
        },
    },
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{convert::Infallible, pin::Pin, task::{Context, Poll}};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use bytes::Bytes;

// Request type matching the Python implementation
#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
}

// Shared application state
struct AppState {
    model: String,
    provider_config: OpenAiProviderConfig,
}

// Manual Clone implementation since we know OpenAiProviderConfig can be cloned
impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            provider_config: OpenAiProviderConfig {
                host: self.provider_config.host.clone(),
                api_key: self.provider_config.api_key.clone(),
            },
        }
    }
}

// Custom SSE response type that implements the Vercel AI SDK protocol
struct SseResponse {
    rx: ReceiverStream<String>,
}

impl SseResponse {
    fn new(rx: ReceiverStream<String>) -> Self {
        Self { rx }
    }
}

impl Stream for SseResponse {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_next(cx).map(|opt| opt.map(|s| Ok(Bytes::from(s))))
    }
}

impl IntoResponse for SseResponse {
    fn into_response(self) -> axum::response::Response {
        let stream = self;
        let body = axum::body::Body::from_stream(stream);

        http::Response::builder()
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .header("x-vercel-ai-data-stream", "v1")
            .body(body)
            .unwrap()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Get configuration from environment
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4".to_string());

    // Create app state
    let state = AppState {
        model: model.clone(),
        provider_config: OpenAiProviderConfig {
            host: "https://api.openai.com".to_string(),
            api_key,
        },
    };

    // Create router with CORS support
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/reply", post(chat_handler))
        .layer(cors)
        .with_state(state);

    // Run server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    info!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}

// Convert JSON message to our Message type
fn convert_message(value: &Value) -> Option<Message> {
    let role = value.get("role")?.as_str()?;
    let role = match role {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => return None,
    };

    let content = if let Some(content) = value.get("content") {
        if let Some(text) = content.as_str() {
            vec![Content::text(text)]
        } else if let Some(array) = content.as_array() {
            array
                .iter()
                .filter_map(|item| {
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        Some(Content::text(text))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            return None;
        }
    } else {
        Vec::new()
    };

    Message::new(role, content).ok()
}

async fn chat_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatRequest>,
) -> Result<SseResponse, StatusCode> {
    // Check protocol header (optional in our case)
    if let Some(protocol) = headers.get("x-protocol") {
        if protocol.to_str().map(|p| p != "data").unwrap_or(true) {
            return Err(StatusCode::BAD_REQUEST);
        }
    }

    // Create channel for streaming
    let (tx, rx) = mpsc::channel(100);
    let stream = ReceiverStream::new(rx);

    // Setup agent with developer system
    let system = Box::new(DeveloperSystem::new());
    let provider = factory::get_provider(
        ProviderType::OpenAi,
        ProviderConfig::OpenAi(state.provider_config),
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut agent = Agent::new(provider, state.model);
    agent.add_system(system);

    // Convert incoming messages
    let messages: Vec<Message> = request
        .messages
        .iter()
        .filter_map(convert_message)
        .collect();

    // Spawn task to handle streaming
    tokio::spawn(async move {
        let mut stream = agent.reply(&messages);

        while let Some(response) = stream.next().await {
            match response {
                Ok(message) => {
                    match message.role {
                        Role::User => {
                            // Handle tool results if present
                            if let Some(Content::ToolResponse(tool_data)) = message.content.first() {
                                let result = json!({
                                    "toolCallId": tool_data.request_id,
                                    "result": tool_data.output
                                });
                                let _ = tx.send(format!("a:{}\n", result.to_string())).await;
                            }
                        }
                        Role::Assistant => {
                            for content in message.content {
                                match content {
                                    Content::ToolRequest(request) => {
                                        if let Ok(call) = request.call {
                                            let tool_call = json!({
                                                "toolCallId": request.id,
                                                "toolName": call.name,
                                                "args": call.parameters
                                            });
                                            let _ = tx.send(format!("9:{}\n", tool_call.to_string())).await;
                                        }
                                    }
                                    _ => {
                                        let text = content.summary();
                                        // Split text by newlines and send each line separately
                                        for line in text.lines() {
                                            if let Err(e) = tx.send(format!("0:\"{}\"\n", line)).await {
                                                tracing::error!("Error sending line through channel: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Error processing message: {}", e);
                    break;
                }
            }
        }

        // Send finish message
        let finish = json!({
            "finishReason": "stop",
            "usage": {
                "promptTokens": 0,
                "completionTokens": 0
            }
        });
        let _ = tx.send(format!("d:{}\n", finish.to_string())).await;
    });

    Ok(SseResponse::new(stream))
}
