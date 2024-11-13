use cliclack::input;
use console::style;
use ctrlc;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct ConfigOptions {
    pub provider: Option<String>,
    pub host: Option<String>,
    pub token: Option<String>,
    pub processor: Option<String>,
    pub accelerator: Option<String>,
}

pub fn handle_configure(options: ConfigOptions) -> Result<(), Box<dyn Error>> {
    ctrlc::set_handler(move || {}).expect("setting Ctrl-C handler");
    
    cliclack::clear_screen()?;

    cliclack::intro(style(" create-app ").on_cyan().black())?;
    let provider = prompt(options.provider, "Enter provider name:");
    let host = prompt(options.host, "Enter host URL:");
    let token = prompt(options.token, "Enter token:");
    let processor = prompt(options.processor, "Enter processor:");
    let accelerator = prompt(options.accelerator, "Enter accelerator:");

    let final_config = ConfigOptions {
        provider: Some(provider),
        host: Some(host),
        token: Some(token),
        processor: Some(processor),
        accelerator: Some(accelerator),
    };
    match save_to_yaml(&final_config) {
        Ok(path) => println!("\nConfiguration saved to: {:?}", path),
        Err(e) => eprintln!("Failed to save configuration: {}", e),
    }
    Ok(())
}

// Helper function to prompt the user
fn prompt(value: Option<String>, message: &str) -> String {
    value.unwrap_or_else(|| {
        input(message)
            .interact()
            .expect("Failed to get input")
    })
}

fn save_to_yaml(config: &ConfigOptions) -> Result<PathBuf, Box<dyn Error>> {
    // Locate the config directory
    let mut path = dirs::home_dir().ok_or("Failed to find home directory")?;
    path.push(".config");
    path.push("goose");

    // TODO: set to profile1.yaml temporarily to avoid overriting the existing config
    path.push("profile1.yaml");

    // Ensure the ~/.config/goose directory exists
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;  // Create the directory if it doesn't exist
    }

    // Serialize the configuration to YAML and save it to the file
    let yaml_string = serde_yaml::to_string(config)?;
    let mut file = File::create(&path)?;
    file.write_all(yaml_string.as_bytes())?;

    Ok(path)
}