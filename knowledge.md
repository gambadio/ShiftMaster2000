# LLM Provider API Integration Knowledge Base

**Last Updated:** 2025-10-17

This document contains comprehensive research on integrating multiple LLM provider APIs, including OpenAI, OpenRouter, Azure OpenAI, and custom OpenAI-compatible endpoints.

---

## Table of Contents

1. [OpenAI API](#openai-api)
2. [OpenRouter API](#openrouter-api)
3. [Azure OpenAI Service](#azure-openai-service)
4. [Custom OpenAI-Compatible Endpoints](#custom-openai-compatible-endpoints)
5. [Common Patterns and Best Practices](#common-patterns-and-best-practices)
6. [Code Examples](#code-examples)

---

## OpenAI API

### Authentication

**Method:** API Key authentication via HTTP headers

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-..."  # Or use os.getenv("OPENAI_API_KEY")
)
```

**Header Format:**
```
Authorization: Bearer YOUR_API_KEY
```

### Base URL and Endpoints

- **Base URL:** `https://api.openai.com/v1`
- **Chat Completions Endpoint:** `POST /v1/chat/completions`
- **Models List Endpoint:** `GET /v1/models`

### Chat Completions API

#### Required Parameters
- `model` (string): Model ID (e.g., "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
- `messages` (array): List of message objects with `role` and `content`

#### Message Roles
- `system`: Sets the behavior/instructions for the assistant
- `user`: User input messages
- `assistant`: Assistant responses (for conversation history)
- `tool`: Tool/function call results
- `developer`: New role for system-level instructions (GPT-5 models)

#### Optional Parameters
- `temperature` (number, 0-2): Sampling temperature (default: 1.0)
- `max_tokens` (integer): **DEPRECATED** - Use `max_completion_tokens` instead
- `max_completion_tokens` (integer): Max tokens for completion including reasoning tokens
- `top_p` (number, 0-1): Nucleus sampling parameter
- `frequency_penalty` (number, -2 to 2): Penalize token frequency
- `presence_penalty` (number, -2 to 2): Penalize token presence
- `stream` (boolean): Enable streaming responses
- `response_format` (object): Control output format (e.g., `{"type": "json_object"}`)
- `n` (integer): Number of completions to generate
- `stop` (string or array): Stop sequences
- `logprobs` (boolean): Return log probabilities
- `top_logprobs` (integer): Number of top logprobs to return
- `seed` (integer): For reproducible sampling
- `tools` (array): Available tools/functions
- `tool_choice` (string/object): Control tool usage ("none", "auto", or specific tool)
- `parallel_tool_calls` (boolean): Enable parallel function calling

#### Reasoning Models (o1, o3 series)
- `reasoning_effort` (string): Controls reasoning depth
  - Options: "minimal", "low", "medium" (default), "high"
  - Higher effort = more reasoning tokens, slower response
- `max_completion_tokens`: Includes both visible output AND reasoning tokens
- **Note:** O-series models don't support `temperature`, `top_p`, or streaming in some versions

#### Advanced Parameters
- `logit_bias` (object): Modify token likelihoods (token_id: bias value from -100 to 100)
- `prediction` (object): Predicted output for latency optimization
- `metadata` (object): 16 key-value pairs for tracking (max key: 64 chars, max value: 512 chars)
- `modalities` (array): Output types ["text"] or ["text", "audio"]
- `audio` (object): Audio output parameters (for gpt-4o-audio models)

### Response Format

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response text here",
                "refusal": null,  # Non-null if model refuses
                "tool_calls": null  # Array if tools were called
            },
            "finish_reason": "stop",  # "stop", "length", "tool_calls", "content_filter"
            "logprobs": null
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "completion_tokens_details": {
            "reasoning_tokens": 0  # For reasoning models
        }
    },
    "system_fingerprint": "fp_..."
}
```

### Streaming

Enable with `stream=True`. Returns Server-Sent Events (SSE):

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

**Stream Response Format:**
- Each chunk has `object: "chat.completion.chunk"`
- `choices[0].delta` contains incremental content
- Final chunk has `finish_reason` set

### Model Fetching

```python
# List all available models
models = client.models.list()

# Get specific model details
model = client.models.retrieve("gpt-4o")
```

### Error Handling

Common HTTP status codes:
- `401`: Invalid API key
- `429`: Rate limit exceeded
- `500`: Server error
- `503`: Service unavailable

---

## OpenRouter API

### Authentication

**Method:** API Key via Bearer token

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-..."  # OpenRouter API key
)
```

**Header Format:**
```
Authorization: Bearer YOUR_OPENROUTER_API_KEY
```

### Base URL and Endpoints

- **Base URL:** `https://openrouter.ai/api/v1`
- **Chat Completions:** `POST /api/v1/chat/completions`

### OpenRouter-Specific Features

#### Optional Headers
```python
headers = {
    "HTTP-Referer": "https://your-site.com",  # For rankings
    "X-Title": "Your App Name"  # For app title on openrouter.ai
}
```

#### Model Selection
- Use full model identifier with provider prefix: `"openai/gpt-4o"`, `"anthropic/claude-3-opus"`
- If model is omitted, uses user's default
- Access 200+ models from multiple providers

#### OpenRouter-Specific Parameters
- `models` (array): List of fallback models for routing
- `route` (string): Routing strategy (e.g., "fallback")
- `provider` (object): Provider preferences for routing
- `transforms` (array): Prompt transformation pipelines

#### Model Routing Example
```python
response = client.chat.completions.create(
    model="openai/gpt-4o",
    models=["openai/gpt-4o", "anthropic/claude-3-opus"],  # Fallbacks
    route="fallback",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Supported Parameters

OpenRouter supports most OpenAI parameters:
- `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`
- `stream`, `stop`, `n`
- `tools`, `tool_choice`
- `response_format` (for supported models)

**Additional Parameters:**
- `top_k` (integer): Top-K sampling (not available for OpenAI models)
- `repetition_penalty` (number, 0-2): Alternative to frequency penalty
- `min_p` (number, 0-1): Min-P sampling
- `top_a` (number, 0-1): Top-A sampling

### Response Format

OpenRouter normalizes responses to OpenAI format with additions:

```python
{
    "id": "gen-...",
    "choices": [{
        "finish_reason": "stop",  # Normalized
        "native_finish_reason": "stop",  # Raw from provider
        "message": {
            "role": "assistant",
            "content": "Response text"
        }
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    },
    "model": "openai/gpt-4o"
}
```

### Streaming

Fully compatible with OpenAI streaming:

```python
stream = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")
```

### Error Handling

```python
{
    "error": {
        "code": 400,
        "message": "Error description",
        "metadata": {
            "provider": "openai",
            "raw": "..."
        }
    }
}
```

### Provider-Specific Quirks

1. **response_format:** Only supported by OpenAI models and select others
   - Check model page on openrouter.ai/models
   - Use `require_parameters: true` in provider preferences

2. **Unsupported parameters** are silently ignored rather than causing errors

3. **Cost tracking:** OpenRouter provides detailed cost information in responses

---

## Azure OpenAI Service

### Authentication Methods

#### Method 1: API Key (Legacy)
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://YOUR-RESOURCE-NAME.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21"
)
```

**Header Format:**
```
api-key: YOUR_AZURE_OPENAI_KEY
```

#### Method 2: Microsoft Entra ID (Recommended)
```python
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint="https://YOUR-RESOURCE-NAME.openai.azure.com/",
    azure_ad_token_provider=token_provider,
    api_version="2024-10-21"
)
```

**Header Format:**
```
Authorization: Bearer YOUR_AZURE_AD_TOKEN
```

**Generate token via Azure CLI:**
```bash
az account get-access-token --resource https://cognitiveservices.azure.com
```

### Endpoint Configuration

**URL Structure:**
```
https://{RESOURCE-NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT-ID}/chat/completions?api-version={API-VERSION}
```

**Components:**
- `RESOURCE-NAME`: Your Azure OpenAI resource name
- `DEPLOYMENT-ID`: Name of your deployed model (NOT the model name itself)
- `API-VERSION`: E.g., "2024-10-21", "2025-04-01-preview"

### Chat Completions API

**Key Differences from OpenAI:**
1. Uses deployment name instead of model name
2. Requires `api-version` query parameter
3. URL includes deployment ID in path
4. Different authentication methods

```python
response = client.chat.completions.create(
    model="gpt-4o",  # This is your DEPLOYMENT name, not model
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=800,
    temperature=0.7
)
```

### Azure-Specific Features

#### Responses API (New)
Azure has a new stateful Responses API:

**Base Endpoint:** `/openai/v1/responses`

**Key Features:**
- Stateful conversation management
- Response chaining via `previous_response_id`
- 30-day data retention
- Supports computer-use models

```python
# Create response
response = client.responses.create(
    model="gpt-4.1-nano",  # Deployment name
    input="This is a test"
)

# Chain responses
second_response = client.responses.create(
    model="gpt-4.1-nano",
    previous_response_id=response.id,
    input=[{"role": "user", "content": "Follow-up question"}]
)

# Retrieve response
retrieved = client.responses.retrieve("resp_...")

# Delete response
client.responses.delete("resp_...")
```

#### Supported API Versions
- `2024-10-21` (stable)
- `2025-04-01-preview` (latest features)
- `preview` (bleeding edge)

#### Region Availability
Azure OpenAI Responses API is available in:
- Australia East
- East US, East US 2, West US, West US 3
- France Central
- Japan East
- Norway East
- Poland Central
- South India
- Sweden Central
- Switzerland North
- UAE North
- UK South

### Model Deployment

**Important:** In Azure, you must:
1. Deploy a model in Azure Portal/AI Foundry
2. Give it a deployment name (can differ from model name)
3. Use the deployment name in API calls, NOT the model name

```bash
# Environment variables
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o"  # Your deployment name
```

### Streaming

Identical to OpenAI streaming:

```python
stream = client.chat.completions.create(
    model="gpt-4o",  # Deployment name
    messages=[{"role": "user", "content": "Hello"}],
    stream=True
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")
```

### Error Handling

Azure returns similar error codes to OpenAI:
- `401`: Authentication failed
- `404`: Deployment not found
- `429`: Rate limit or quota exceeded
- `500`: Internal server error

### Azure-Specific Considerations

1. **Content Filtering:** Azure adds content filtering at multiple levels
   - Response includes `content_filter_results`
   - Categories: Hate, SelfHarm, Sexual, Violence
   - Each has severity level and filtered flag

2. **Data Residency:** Data stays in selected Azure region

3. **Quota Management:** Per-deployment token-per-minute (TPM) limits

4. **Cost Structure:** Different from OpenAI - billed through Azure subscription

---

## Custom OpenAI-Compatible Endpoints

Many providers offer OpenAI-compatible APIs (LocalAI, Ollama, vLLM, etc.).

### General Pattern

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Custom endpoint
    api_key="optional-key-or-dummy"  # Some require it, some don't
)

response = client.chat.completions.create(
    model="model-name",  # Provider-specific model name
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Common Custom Endpoints

#### LocalAI
```python
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)
```

#### Ollama
```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```

#### vLLM
```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
```

### Model Listing

Most support the `/v1/models` endpoint:

```python
models = client.models.list()
for model in models.data:
    print(model.id)
```

### Parameter Support

**Widely Supported:**
- `messages`, `temperature`, `max_tokens`, `stream`

**Variable Support:**
- `top_p`, `frequency_penalty`, `presence_penalty`
- `stop` sequences
- `response_format` (JSON mode)

**Rarely Supported:**
- `tools` / function calling
- `logprobs`
- `seed`

**Best Practice:** Check provider documentation for supported parameters. Unsupported parameters are typically ignored.

---

## Common Patterns and Best Practices

### 1. Unified Client Pattern

Create a factory function for consistent client creation:

```python
def create_llm_client(provider: str, **kwargs):
    if provider == "openai":
        return OpenAI(api_key=kwargs.get("api_key"))

    elif provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=kwargs.get("api_key")
        )

    elif provider == "azure":
        return AzureOpenAI(
            azure_endpoint=kwargs.get("endpoint"),
            api_key=kwargs.get("api_key"),
            api_version=kwargs.get("api_version", "2024-10-21")
        )

    elif provider == "custom":
        return OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key", "not-needed")
        )
```

### 2. Error Handling

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

def safe_completion(client, **params):
    try:
        return client.chat.completions.create(**params)

    except RateLimitError as e:
        # Handle rate limits - maybe retry with backoff
        print(f"Rate limited: {e}")
        raise

    except APIConnectionError as e:
        # Network error - retry or fail gracefully
        print(f"Connection error: {e}")
        raise

    except APIError as e:
        # Other API errors
        print(f"API error: {e.status_code} - {e.message}")
        raise
```

### 3. Streaming Handler

```python
def stream_completion(client, **params):
    params["stream"] = True
    stream = client.chat.completions.create(**params)

    full_response = ""
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield delta

    return full_response
```

### 4. Model Detection

```python
def fetch_available_models(client, provider: str):
    """Fetch models with provider-specific handling"""
    try:
        models = client.models.list()

        if provider == "azure":
            # Azure returns deployments, not models
            return [model.id for model in models.data]

        elif provider == "openrouter":
            # OpenRouter uses provider/model format
            return [model.id for model in models.data]

        else:
            # Standard OpenAI format
            return [model.id for model in models.data]

    except Exception as e:
        print(f"Could not fetch models: {e}")
        return []
```

### 5. Parameter Normalization

```python
def normalize_params(params: dict, provider: str) -> dict:
    """Normalize parameters across providers"""
    normalized = params.copy()

    # Azure uses deployment names
    if provider == "azure" and "model" in normalized:
        # model is actually deployment_name in Azure
        pass

    # Remove unsupported params for specific providers
    if provider == "custom":
        # Many custom endpoints don't support these
        normalized.pop("frequency_penalty", None)
        normalized.pop("presence_penalty", None)
        normalized.pop("logprobs", None)

    return normalized
```

### 6. Response Format Handling

```python
def request_json_response(client, provider: str, **params):
    """Request JSON response with provider compatibility check"""

    # Check if provider supports response_format
    supports_json_mode = provider in ["openai", "azure", "openrouter"]

    if supports_json_mode:
        params["response_format"] = {"type": "json_object"}
    else:
        # Fallback: add JSON instruction to system message
        messages = params.get("messages", [])
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\nRespond in valid JSON format."
        else:
            messages.insert(0, {
                "role": "system",
                "content": "Respond in valid JSON format."
            })

    return client.chat.completions.create(**params)
```

### 7. Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for OpenAI models"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))
```

---

## Code Examples

### Multi-Provider Configuration

```python
import os
from openai import OpenAI, AzureOpenAI
from typing import Literal

ProviderType = Literal["openai", "openrouter", "azure", "custom"]

class LLMConfig:
    def __init__(
        self,
        provider: ProviderType,
        api_key: str = None,
        base_url: str = None,
        azure_endpoint: str = None,
        api_version: str = "2024-10-21"
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version

    def create_client(self):
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)

        elif self.provider == "openrouter":
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )

        elif self.provider == "azure":
            return AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )

        elif self.provider == "custom":
            return OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed"
            )

# Usage
config = LLMConfig(provider="openai", api_key="sk-...")
client = config.create_client()
```

### Universal Completion Function

```python
def universal_completion(
    provider: str,
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    stream: bool = False,
    **kwargs
):
    """Universal completion function supporting multiple providers"""

    # Create client based on provider
    if provider == "openai":
        client = OpenAI(api_key=kwargs.get("api_key"))
    elif provider == "openrouter":
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=kwargs.get("api_key")
        )
    elif provider == "azure":
        client = AzureOpenAI(
            azure_endpoint=kwargs.get("endpoint"),
            api_key=kwargs.get("api_key"),
            api_version=kwargs.get("api_version", "2024-10-21")
        )
    else:
        client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key", "not-needed")
        )

    # Make request
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }

    # Add OpenRouter-specific headers if needed
    if provider == "openrouter":
        kwargs.setdefault("extra_headers", {})
        if "http_referer" in kwargs:
            kwargs["extra_headers"]["HTTP-Referer"] = kwargs["http_referer"]
        if "x_title" in kwargs:
            kwargs["extra_headers"]["X-Title"] = kwargs["x_title"]

    return client.chat.completions.create(**params)
```

### Streaming with Progress

```python
from typing import Generator

def stream_with_callback(
    client,
    messages: list,
    model: str,
    on_token: callable = None,
    on_complete: callable = None
) -> str:
    """Stream completion with callbacks"""

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    full_response = ""

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                if on_token:
                    on_token(delta)

    if on_complete:
        on_complete(full_response)

    return full_response

# Usage
def print_token(token):
    print(token, end="", flush=True)

def on_done(text):
    print(f"\n\nComplete! ({len(text)} chars)")

response = stream_with_callback(
    client,
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",
    on_token=print_token,
    on_complete=on_done
)
```

### Retry Logic with Exponential Backoff

```python
import time
from openai import RateLimitError, APIError

def completion_with_retry(
    client,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **params
):
    """Completion with exponential backoff retry"""

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**params)

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            delay = base_delay * (2 ** attempt)
            print(f"Rate limited. Retrying in {delay}s...")
            time.sleep(delay)

        except APIError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                # Server error - retry
                delay = base_delay * (2 ** attempt)
                print(f"Server error. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
```

---

## Summary

### Quick Reference Table

| Feature | OpenAI | OpenRouter | Azure OpenAI | Custom |
|---------|--------|------------|--------------|--------|
| Authentication | API Key | API Key | API Key / Entra ID | Varies |
| Base URL | api.openai.com/v1 | openrouter.ai/api/v1 | {resource}.openai.azure.com | Custom |
| Model Format | model-name | provider/model | deployment-name | varies |
| Streaming | ✅ | ✅ | ✅ | Usually |
| JSON Mode | ✅ | ✅ (select models) | ✅ | Varies |
| Function Calling | ✅ | ✅ (select models) | ✅ | Rarely |
| Model Listing | ✅ | ✅ | ✅ | Usually |
| Reasoning Models | ✅ (o1, o3) | ✅ | ✅ | No |

### Key Takeaways

1. **Use OpenAI SDK for all providers** - it's the most compatible
2. **Always handle authentication errors gracefully**
3. **Check provider docs for parameter support** - don't assume everything works
4. **Implement retry logic** for production systems
5. **Azure requires deployment names, not model names**
6. **OpenRouter provides unified access** to 200+ models
7. **Streaming is universally supported** but implementation may vary
8. **Token counting differs** - use tiktoken for OpenAI, check provider docs for others

### Security Best Practices

1. **Never hardcode API keys** - use environment variables or secure vaults
2. **Use Entra ID for Azure** instead of API keys when possible
3. **Implement rate limiting** in your application
4. **Validate and sanitize user inputs** before sending to LLMs
5. **Monitor usage and costs** across all providers
6. **Rotate API keys regularly**
7. **Use HTTPS** for all API communications
8. **Store API keys in Azure Key Vault or similar** for production

---

## Additional Resources

- **OpenAI API Reference:** https://platform.openai.com/docs/api-reference
- **OpenRouter Documentation:** https://openrouter.ai/docs
- **Azure OpenAI Documentation:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **OpenAI Python SDK:** https://github.com/openai/openai-python
- **Tiktoken (token counting):** https://github.com/openai/tiktoken
