# ProSA Ollama

ProSA processor to handle Ollama API.

It leverages the [Ollama-rs](https://crates.io/crates/ollama-rs) crate to handle Ollama calls.

With this processor, you can:
- Download Ollama models
- List available Ollama models
- Get detailed information about a specific model
- Make AI requests
- Request AI embeddings

## Configuration

To configure your processor, you need to set the following parameters:
```yaml
ollama:
  url: "http://localhost:11434"
  models:
    - "mistral"
    - "devstral"
  service: "PROC_SERVICE_NAME"
```
