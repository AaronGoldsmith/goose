use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

use crate::errors::{AgentError, AgentResult};
use crate::systems::System;
use crate::tool::{Tool, ToolCall};

pub struct DeveloperSystem {
    tools: Vec<Tool>,
    cwd: Mutex<PathBuf>,
    active_files: Mutex<HashSet<PathBuf>>,
    env: Mutex<HashMap<String, String>>,
    file_history: Mutex<HashMap<PathBuf, Vec<String>>>, // Moved file_history here
}

impl DeveloperSystem {
    pub fn new() -> Self {
        // TODO i suggest we make working_dir required, we're seeing abs paths are more clear
        let bash_tool = Tool::new(
            "bash",
            "Run commands in a bash shell. Perform bash-related operations in a specific order: \
            1. Change the working directory (if provided) \
            2. Source a file (if provided) \
            3. Run a shell command (if provided) \
            At least one of the parameters must be provided.",
            json!({
                "type": "object",
                "required": [],
                "properties": {
                    "command": {
                        "type": "string",
                        "default": null,
                        "description": "The bash shell command to run."
                    },
                    "source_path": {
                        "type": "string",
                        "default": null,
                        "description": "The file to source before running the command."
                    },
                    "working_dir": {
                        "type": "string",
                        "default": null,
                        "description": "The directory to change to. The path must be absolute or relative to the current working directory. Defaults to the current working directory."
                    }
                }
            }),
        );

        let text_editor_tool = Tool::new(
            "text_editor",
            "Perform text editing operations on files. The `command` parameter specifies the operation to perform.",
            json!({
                "type": "object",
                "required": ["command", "path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to file or directory."
                    },
                    "command": {
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "The commands to run."
                    },
                    "new_str": {
                        "type": "string",
                        "default": null,
                        "description": "Required for `str_replace` and `insert` commands."
                    },
                    "old_str": {
                        "type": "string",
                        "default": null,
                        "description": "Required for `str_replace` command."
                    },
                    "file_text": {
                        "type": "string",
                        "default": null,
                        "description": "Required for `create` command."
                    },
                    "insert_line": {
                        "type": "integer",
                        "default": null,
                        "description": "Required for `insert` command."
                    }
                }
            }),
        );

        Self {
            tools: vec![bash_tool, text_editor_tool],
            cwd: Mutex::new(std::env::current_dir().unwrap()),
            active_files: Mutex::new(HashSet::new()),
            env: Mutex::new(std::env::vars().collect()),
            file_history: Mutex::new(HashMap::new()), // Initialize file_history
        }
    }

    // TODO i suggest we switch to all abs paths
    // Helper method to resolve a path relative to cwd
    fn resolve_path(&self, path_str: &str) -> AgentResult<PathBuf> {
        let cwd = self.cwd.lock().unwrap();
        let path = Path::new(path_str);
        let resolved_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            cwd.join(path)
        };

        Ok(resolved_path)
    }

    // Implement bash tool functionality
    async fn bash(&self, params: Value) -> AgentResult<Value> {
        let working_dir = params
            .get("working_dir")
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let command = params.get("command").and_then(|v| v.as_str());
        let source_path = params.get("source_path").and_then(|v| v.as_str());

        if command.is_none() && source_path.is_none() {
            return Err(AgentError::InvalidParameters(
                "At least one of 'command' or 'source_path' must be provided".into(),
            ));
        }

        let mut outputs = Vec::new();

        // Change working directory
        let new_cwd = self.resolve_path(working_dir)?;
        if !new_cwd.is_dir() {
            return Err(AgentError::InvalidParameters(format!(
                "The directory '{}' does not exist",
                new_cwd.display()
            )));
        }
        {
            let mut cwd = self.cwd.lock().unwrap();
            *cwd = new_cwd.clone();
        }
        outputs.push(format!("Changed directory to: {}", new_cwd.display()));

        // Source a file
        if let Some(source) = source_path {
            let source_file = self.resolve_path(source)?;
            if !source_file.is_file() {
                return Err(AgentError::InvalidParameters(format!(
                    "The file '{}' does not exist",
                    source
                )));
            }
            // Scope the mutex lock for cwd and env
            {
                let cwd = self.cwd.lock().unwrap().clone();
                let env = self.env.lock().unwrap().clone();

                let source_command = format!("source \"{}\" && env", source_file.display());

                let output = Command::new("bash")
                    .arg("-c")
                    .arg(&source_command)
                    .current_dir(cwd)
                    .envs(env)
                    .output()
                    .map_err(|e| {
                        AgentError::ExecutionError(format!(
                            "Failed to execute source command: {}",
                            e
                        ))
                    })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(AgentError::ExecutionError(stderr.to_string()));
                }

                let stdout = String::from_utf8_lossy(&output.stdout);

                // Parse env variables from stdout
                let mut new_env = HashMap::new();
                for line in stdout.lines() {
                    if let Some((key, value)) = line.split_once('=') {
                        new_env.insert(key.to_string(), value.to_string());
                    }
                }

                // Update self.env
                {
                    let mut env = self.env.lock().unwrap();
                    *env = new_env;
                }
            }

            outputs.push(format!("Sourced {}", source));
        }

        // Execute command
        if let Some(cmd_str) = command {
            // TODO these messages should be more clear, these are alternatives solved with other tools
            // TODO we also need guardrails
            // Disallow certain commands for safety
            let disallowed_commands = ["cat", "cd", "source", "rm", "kill"];
            for disallowed in &disallowed_commands {
                if cmd_str.trim_start().starts_with(disallowed) {
                    return Err(AgentError::InvalidParameters(format!(
                        "The command '{}' is not allowed",
                        disallowed
                    )));
                }
            }

            let cwd = self.cwd.lock().unwrap();
            let env = self.env.lock().unwrap();

            // Redirect stderr to stdout to interleave outputs
            let cmd_with_redirect = format!("{} 2>&1", cmd_str);

            // Execute the command
            let output = Command::new("bash")
                .arg("-c")
                .arg(cmd_with_redirect)
                .current_dir(&*cwd)
                .envs(&*env)
                .output()
                .map_err(|e| AgentError::ExecutionError(e.to_string()))?;

            let output_str = String::from_utf8_lossy(&output.stdout).to_string();

            if !output.status.success() {
                return Err(AgentError::ExecutionError(output_str));
            }
            outputs.push(output_str);
        }

        Ok(json!({ "result": outputs.join("\n") }))
    }

    // Implement text_editor tool functionality
    async fn text_editor(&self, params: Value) -> AgentResult<Value> {
        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::InvalidParameters("Missing 'command' parameter".into()))?;

        let path_str = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::InvalidParameters("Missing 'path' parameter".into()))?;

        let path = self.resolve_path(path_str)?;

        match command {
            "view" => self.text_editor_view(&path).await,
            "create" => {
                let file_text = params
                    .get("file_text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        AgentError::InvalidParameters("Missing 'file_text' parameter".into())
                    })?;

                self.text_editor_create(&path, file_text).await
            }
            "str_replace" => {
                let old_str = params
                    .get("old_str")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        AgentError::InvalidParameters("Missing 'old_str' parameter".into())
                    })?;
                let new_str = params
                    .get("new_str")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        AgentError::InvalidParameters("Missing 'new_str' parameter".into())
                    })?;

                self.text_editor_str_replace(&path, old_str, new_str).await
            }
            "insert" => {
                let insert_line = params
                    .get("insert_line")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        AgentError::InvalidParameters("Missing 'insert_line' parameter".into())
                    })?;
                let new_str = params
                    .get("new_str")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        AgentError::InvalidParameters("Missing 'new_str' parameter".into())
                    })?;

                self.text_editor_insert(&path, insert_line as usize, new_str)
                    .await
            }
            "undo_edit" => self.text_editor_undo_edit(&path).await,
            _ => Err(AgentError::InvalidParameters(format!(
                "Unknown command '{}'",
                command
            ))),
        }
    }

    async fn text_editor_view(&self, path: &PathBuf) -> AgentResult<Value> {
        if path.is_file() {
            // Read the file content
            let content = std::fs::read_to_string(path)
                .map_err(|e| AgentError::ExecutionError(format!("Failed to read file: {}", e)))?;

            // Add to active files
            self.active_files.lock().unwrap().insert(path.clone());

            Ok(json!({ "content": content }))
        } else if path.is_dir() {
            // List directory contents
            let entries = std::fs::read_dir(path).map_err(|e| {
                AgentError::ExecutionError(format!("Failed to read directory: {}", e))
            })?;

            let mut files = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| {
                    AgentError::ExecutionError(format!("Failed to read directory entry: {}", e))
                })?;
                files.push(entry.file_name().to_string_lossy().into_owned());
            }

            Ok(json!({ "contents": files }))
        } else {
            Err(AgentError::InvalidParameters(format!(
                "The path '{}' does not exist",
                path.display()
            )))
        }
    }

    async fn text_editor_create(&self, path: &PathBuf, file_text: &str) -> AgentResult<Value> {
        // Check if file already exists and is active
        if path.exists() && !self.active_files.lock().unwrap().contains(path) {
            return Err(AgentError::InvalidParameters(format!(
                "File '{}' exists but is not active. View it first before overwriting.",
                path.display()
            )));
        }

        // Save history for undo
        self.save_file_history(path)?;

        // Write to the file
        std::fs::write(path, file_text)
            .map_err(|e| AgentError::ExecutionError(format!("Failed to write file: {}", e)))?;

        // Add to active files
        self.active_files.lock().unwrap().insert(path.clone());

        Ok(json!({ "result": format!("Successfully wrote to {}", path.display()) }))
    }

    async fn text_editor_str_replace(
        &self,
        path: &PathBuf,
        old_str: &str,
        new_str: &str,
    ) -> AgentResult<Value> {
        // Check if file exists and is active
        if !path.exists() {
            return Err(AgentError::InvalidParameters(format!(
                "File '{}' does not exist",
                path.display()
            )));
        }
        if !self.active_files.lock().unwrap().contains(path) {
            return Err(AgentError::InvalidParameters(format!(
                "You must view '{}' before editing it",
                path.display()
            )));
        }

        // Read content
        let content = std::fs::read_to_string(path)
            .map_err(|e| AgentError::ExecutionError(format!("Failed to read file: {}", e)))?;

        // Ensure 'old_str' appears exactly once
        if content.matches(old_str).count() != 1 {
            return Err(AgentError::InvalidParameters(
                "'old_str' must appear exactly once in the file".into(),
            ));
        }

        // Save history for undo
        self.save_file_history(path)?;

        // Replace and write back
        let new_content = content.replace(old_str, new_str);
        std::fs::write(path, new_content)
            .map_err(|e| AgentError::ExecutionError(format!("Failed to write file: {}", e)))?;

        Ok(json!({ "result": "Successfully replaced text" }))
    }

    async fn text_editor_insert(
        &self,
        path: &PathBuf,
        insert_line: usize,
        new_str: &str,
    ) -> AgentResult<Value> {
        // Check if file exists and is active
        if !path.exists() {
            return Err(AgentError::InvalidParameters(format!(
                "File '{}' does not exist",
                path.display()
            )));
        }
        if !self.active_files.lock().unwrap().contains(path) {
            return Err(AgentError::InvalidParameters(format!(
                "You must view '{}' before editing it",
                path.display()
            )));
        }

        // Read lines
        let content = std::fs::read_to_string(path)
            .map_err(|e| AgentError::ExecutionError(format!("Failed to read file: {}", e)))?;
        let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();

        if insert_line > lines.len() {
            return Err(AgentError::InvalidParameters(
                "Insert line number is out of range".into(),
            ));
        }

        // Save history for undo
        self.save_file_history(path)?;

        // Insert new string after the specified line
        lines.insert(insert_line, new_str.to_string());

        // Write back to file
        std::fs::write(path, lines.join("\n"))
            .map_err(|e| AgentError::ExecutionError(format!("Failed to write file: {}", e)))?;

        Ok(json!({ "result": "Successfully inserted text" }))
    }

    async fn text_editor_undo_edit(&self, path: &PathBuf) -> AgentResult<Value> {
        let mut history = self.file_history.lock().unwrap();
        if let Some(contents) = history.get_mut(path) {
            if let Some(previous_content) = contents.pop() {
                // Write previous content back to file
                std::fs::write(path, previous_content).map_err(|e| {
                    AgentError::ExecutionError(format!("Failed to write file: {}", e))
                })?;
                Ok(json!({ "result": "Successfully undid the last edit" }))
            } else {
                Err(AgentError::InvalidParameters(
                    "No edit history available to undo".into(),
                ))
            }
        } else {
            Err(AgentError::InvalidParameters(
                "No edit history available to undo".into(),
            ))
        }
    }

    fn save_file_history(&self, path: &PathBuf) -> AgentResult<()> {
        let mut history = self.file_history.lock().unwrap();
        let content = if path.exists() {
            std::fs::read_to_string(path)
                .map_err(|e| AgentError::ExecutionError(format!("Failed to read file: {}", e)))?
        } else {
            String::new()
        };
        history
            .entry(path.clone())
            .or_insert_with(Vec::new)
            .push(content);
        Ok(())
    }
}

#[async_trait]
impl System for DeveloperSystem {
    fn name(&self) -> &str {
        "DeveloperSystem"
    }

    fn description(&self) -> &str {
        "A system that provides bash and text editing capabilities"
    }

    fn instructions(&self) -> &str {
        "Use the provided tools to perform bash operations and text editing tasks."
    }

    fn tools(&self) -> &[Tool] {
        &self.tools
    }

    async fn status(&self) -> AnyhowResult<HashMap<String, Value>> {
        let cwd = self.cwd.lock().unwrap().display().to_string();
        let active_files: Vec<String> = self
            .active_files
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.display().to_string())
            .collect();

        Ok(HashMap::from([
            ("cwd".to_string(), json!(cwd)),
            ("active_files".to_string(), json!(active_files)),
        ]))
    }

    async fn call(&self, tool_call: ToolCall) -> AgentResult<Value> {
        match tool_call.name.as_str() {
            "bash" => self.bash(tool_call.parameters).await,
            "text_editor" => self.text_editor(tool_call.parameters).await,
            _ => Err(AgentError::ToolNotFound(tool_call.name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::OnceCell;

    // Use OnceCell to initialize the system once for all tests
    static DEV_SYSTEM: OnceCell<DeveloperSystem> = OnceCell::const_new();

    async fn get_system() -> &'static DeveloperSystem {
        DEV_SYSTEM
            .get_or_init(|| async { DeveloperSystem::new() })
            .await
    }

    #[tokio::test]
    async fn test_bash_missing_parameters() {
        let system = get_system().await;

        let tool_call = ToolCall::new("bash", json!({"working_dir": "."}));
        let error = system.call(tool_call).await.unwrap_err();
        assert!(matches!(error, AgentError::InvalidParameters(_)));
    }

    #[tokio::test]
    async fn test_bash_change_directory() {
        let system = get_system().await;

        let tool_call = ToolCall::new("bash", json!({ "working_dir": ".", "command": "pwd" }));
        let result = system.call(tool_call).await.unwrap();
        assert!(result["result"]
            .as_str()
            .unwrap()
            .contains("Changed directory to"));
    }

    #[tokio::test]
    async fn test_bash_invalid_directory() {
        let system = get_system().await;

        let tool_call = ToolCall::new("bash", json!({ "working_dir": "non_existent_dir" }));
        let error = system.call(tool_call).await.unwrap_err();
        assert!(matches!(error, AgentError::InvalidParameters(_)));
    }

    #[tokio::test]
    async fn test_text_editor_create_and_view_file() {
        let system = get_system().await;

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let file_path_str = file_path.to_str().unwrap();

        // Create a new file
        let create_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "create",
                "path": file_path_str,
                "file_text": "Hello, world!"
            }),
        );
        let create_result = system.call(create_call).await.unwrap();
        assert!(create_result["result"]
            .as_str()
            .unwrap()
            .contains("Successfully wrote to"));

        // View the file
        let view_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "view",
                "path": file_path_str
            }),
        );
        let view_result = system.call(view_call).await.unwrap();
        assert_eq!(view_result["content"].as_str().unwrap(), "Hello, world!");

        temp_dir.close().unwrap();
    }

    #[tokio::test]
    async fn test_text_editor_str_replace() {
        let system = get_system().await;

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let file_path_str = file_path.to_str().unwrap();

        // Create a new file
        let create_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "create",
                "path": file_path_str,
                "file_text": "Hello, world!"
            }),
        );
        system.call(create_call).await.unwrap();

        // View the file to make it active
        let view_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "view",
                "path": file_path_str
            }),
        );
        system.call(view_call).await.unwrap();

        // Replace string
        let replace_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "str_replace",
                "path": file_path_str,
                "old_str": "world",
                "new_str": "Rust"
            }),
        );
        let replace_result = system.call(replace_call).await.unwrap();
        assert!(replace_result["result"]
            .as_str()
            .unwrap()
            .contains("Successfully replaced text"));

        // View the file again
        let view_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "view",
                "path": file_path_str
            }),
        );
        let view_result = system.call(view_call).await.unwrap();
        assert_eq!(view_result["content"].as_str().unwrap(), "Hello, Rust!");

        temp_dir.close().unwrap();
    }

    #[tokio::test]
    async fn test_text_editor_undo_edit() {
        let system = get_system().await;

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        let file_path_str = file_path.to_str().unwrap();

        // Create a new file
        let create_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "create",
                "path": file_path_str,
                "file_text": "First line"
            }),
        );
        system.call(create_call).await.unwrap();

        // View the file to make it active
        let view_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "view",
                "path": file_path_str
            }),
        );
        system.call(view_call).await.unwrap();

        // Insert a new line
        let insert_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "insert",
                "path": file_path_str,
                "insert_line": 1,
                "new_str": "Second line"
            }),
        );
        system.call(insert_call).await.unwrap();

        // Undo the edit
        let undo_call = ToolCall::new(
            "text_editor",
            json!({
                "command": "undo_edit",
                "path": file_path_str
            }),
        );
        let undo_result = system.call(undo_call).await.unwrap();
        assert!(undo_result["result"]
            .as_str()
            .unwrap()
            .contains("Successfully undid the last edit"));

        // View the file again
        let view_result = system
            .call(ToolCall::new(
                "text_editor",
                json!({
                    "command": "view",
                    "path": file_path_str
                }),
            ))
            .await
            .unwrap();
        assert_eq!(view_result["content"].as_str().unwrap(), "First line");

        temp_dir.close().unwrap();
    }
}
