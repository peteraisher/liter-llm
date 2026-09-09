---
id: legacy_rust_guides_error_handling
language: rust
target: rust
level: typecheck
requires: []
side_effect: network
---

```rust
use liter_llm::{
    ChatCompletionRequest, ClientConfigBuilder, DefaultClient, LiterLlmError, LlmClient, Message,
    UserContent, UserMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ClientConfigBuilder::new(std::env::var("OPENAI_API_KEY")?).build();
    let client = DefaultClient::new(config, None)?;

    let request = ChatCompletionRequest {
        model: "openai/gpt-4o".to_owned(),
        messages: vec![Message::User(UserMessage {
            content: UserContent::Text("Hello".into()),
            name: None,
        })],
        ..Default::default()
    };

    match client.chat(request).await {
        Ok(response) => {
            if let Some(text) = response.choices.first()
                .and_then(|choice| choice.message.content.as_ref())
                .and_then(|content| content.as_text())
            {
                println!("{text}");
            }
        }
        // Transient errors — worth retrying or falling back to another model.
        Err(e) if e.is_transient() => eprintln!("transient failure: {e}"),
        // Terminal errors — branch on specific variants where the response differs.
        Err(LiterLlmError::Authentication { message, .. }) => eprintln!("auth failed: {message}"),
        Err(LiterLlmError::ContextWindowExceeded { message }) => {
            eprintln!("prompt too long: {message}")
        }
        Err(LiterLlmError::BudgetExceeded { message, .. }) => {
            eprintln!("budget exceeded: {message}")
        }
        Err(e) => eprintln!("llm error ({}): {e}", e.error_type()),
    }

    Ok(())
}
```
