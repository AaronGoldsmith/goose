use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenCounter {
    tokenizers: HashMap<String, Tokenizer>,
}

const GPT_4O_TOKENIZER_KEY: &str = "Xenova--gpt-4o";
const CLAUDE_TOKENIZER_KEY: &str = "Xenova--claude-tokenizer";
const QWEN_TOKENIZER_KEY: &str = "Qwen--Qwen2.5-Coder-32B-Instruct";

impl TokenCounter {
    // static method to get the tokenizer files directory
    fn tokenizer_files_dir() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tokenizer_files");
        path
    }

    fn load_tokenizer(&mut self, tokenizer_key: &str, path: Option<PathBuf>) {
        // if path is not provided, use the default path
        let tokenizer_path = path.unwrap_or_else(|| {
            Self::tokenizer_files_dir()
                .join(tokenizer_key)
                .join("tokenizer.json")
        });

        match Tokenizer::from_file(tokenizer_path) {
            Ok(tokenizer) => {
                self.tokenizers.insert(tokenizer_key.to_string(), tokenizer);
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer {}: {}", tokenizer_key, e);
            }
        }
    }

    pub fn new() -> Self {
        let mut counter = TokenCounter {
            tokenizers: HashMap::new(),
        };
        // Add default tokenizers
        for tokenizer_key in [GPT_4O_TOKENIZER_KEY, CLAUDE_TOKENIZER_KEY] {
            counter.load_tokenizer(tokenizer_key, None);
        }
        counter
    }

    pub fn add_tokenizer(&mut self, tokenizer_key: &str, path: Option<PathBuf>) {
        self.load_tokenizer(tokenizer_key, path);
    }

    fn model_to_tokenizer_key(model_name: Option<&str>) -> &str {
        let model_name = model_name.unwrap_or("gpt-4o").to_lowercase();
        if model_name.contains("claude") {
            CLAUDE_TOKENIZER_KEY
        } else if model_name.contains("qwen") {
            QWEN_TOKENIZER_KEY
        } else {
            // default
            GPT_4O_TOKENIZER_KEY
        }
    }

    pub fn count_tokens(&self, text: &str, model_name: Option<&str>) -> usize {
        let tokenizer_key = Self::model_to_tokenizer_key(model_name);
        dbg!(&model_name, &tokenizer_key);
        let tokenizer = self
            .tokenizers
            .get(tokenizer_key)
            .expect("Tokenizer not found");
        let encoding = tokenizer.encode(text, false).unwrap();
        encoding.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_tokenizer_and_count_tokens() {
        let mut counter = TokenCounter::new();
        counter.add_tokenizer(QWEN_TOKENIZER_KEY, None);
        let text = "Hey there!";
        let count = counter.count_tokens(text, Some("qwen2.5-ollama"));
        println!("Token count for '{}': {:?}", text, count);

        assert_eq!(count, 3);
    }

    // Update the default tokenizer test similarly
    #[test]
    fn test_specific_claude_tokenizer() {
        let counter = TokenCounter::new();
        let text = "Hello, how are you?";
        let count = counter.count_tokens(text, Some("claude-3-5-sonnet-2"));

        println!("Token count for '{}': {:?}", text, count);
        assert_eq!(count, 6);
    }

    #[test]
    fn test_default_gpt_4o_tokenizer() {
        let counter = TokenCounter::new();
        let count = counter.count_tokens("Hey there!", None);
        assert_eq!(count, 3);
    }
}
