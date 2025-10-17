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
7. [Multi-Sheet Excel File Handling](#multi-sheet-excel-file-handling)

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

---

## Multi-Sheet Excel File Handling

**Last Updated:** 2025-10-17

This section provides comprehensive guidance for reading and writing multi-sheet Excel files in Python/Streamlit applications, with specific focus on Microsoft Teams Shifts integration.

---

### Table of Contents

1. [Reading Multi-Sheet Excel Files](#reading-multi-sheet-excel-files)
2. [Writing Multi-Sheet Excel Files](#writing-multi-sheet-excel-files)
3. [Microsoft Teams Shifts Excel Format](#microsoft-teams-shifts-excel-format)
4. [Streamlit File Upload Integration](#streamlit-file-upload-integration)
5. [Complete Implementation Examples](#complete-implementation-examples)

---

### Reading Multi-Sheet Excel Files

#### Using Pandas to Read Excel Files

**Required Library:**
```bash
pip install pandas openpyxl
```

**Note:** `openpyxl` is required for reading `.xlsx` files. Use `xlrd` for legacy `.xls` files.

#### 1. List All Sheet Names

```python
import pandas as pd

# Method 1: Using ExcelFile
excel_file = pd.ExcelFile('file.xlsx')
sheet_names = excel_file.sheet_names
print(f"Available sheets: {sheet_names}")

# Method 2: Using pd.read_excel with sheet_name=None
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)
print(f"Sheet names: {list(all_sheets.keys())}")
```

#### 2. Read Specific Sheet by Name

```python
# Read single sheet by name
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# Read single sheet by index (0-based)
df = pd.read_excel('file.xlsx', sheet_name=0)
```

#### 3. Read Multiple Specific Sheets

```python
# Method 1: Pass list of sheet names or indices
sheets_dict = pd.read_excel(
    'file.xlsx',
    sheet_name=['Sheet1', 'Sheet2', 3]  # Mix of names and indices
)

# Access individual sheets
df1 = sheets_dict['Sheet1']
df2 = sheets_dict['Sheet2']
df3 = sheets_dict[3]  # Third sheet by index

# Method 2: Using ExcelFile context manager (recommended for multiple reads)
with pd.ExcelFile('file.xlsx') as xls:
    df1 = pd.read_excel(xls, 'Sheet1', index_col=None, na_values=['NA'])
    df2 = pd.read_excel(xls, 'Sheet2', index_col=None, na_values=['NA'])
```

#### 4. Read All Sheets at Once

```python
# Read all sheets into a dictionary
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)

# Iterate through all sheets
for sheet_name, df in all_sheets.items():
    print(f"\nSheet: {sheet_name}")
    print(f"Shape: {df.shape}")
    print(df.head())
```

#### 5. Advanced Reading with Different Parameters per Sheet

```python
import pandas as pd

with pd.ExcelFile('file.xlsx') as xls:
    # Read first sheet with custom parameters
    shifts_df = pd.read_excel(
        xls,
        'Schichten',
        header=0,
        parse_dates=['Startdatum', 'Enddatum'],
        na_values=['', 'N/A']
    )

    # Read second sheet with different parameters
    timeoff_df = pd.read_excel(
        xls,
        'Arbeitsfreie Zeit',
        header=0,
        skiprows=0,
        usecols=['Mitglied', 'Startdatum', 'Enddatum', 'Beschreibung']
    )
```

#### 6. Handling Variable Sheet Names

```python
def find_sheet_by_pattern(excel_file, patterns):
    """Find sheet name matching any of the provided patterns."""
    sheet_names = excel_file.sheet_names

    for pattern in patterns:
        # Exact match
        if pattern in sheet_names:
            return pattern

        # Case-insensitive match
        for name in sheet_names:
            if name.lower() == pattern.lower():
                return name

    return None

# Usage
excel_file = pd.ExcelFile('teams_export.xlsx')

# Try multiple possible names
shifts_sheet = find_sheet_by_pattern(
    excel_file,
    ['Schichten', 'Shifts', 'schichten', 'shifts']
)

if shifts_sheet:
    df = pd.read_excel(excel_file, shifts_sheet)
else:
    print("Shifts sheet not found")
```

#### 7. Best Practices for Reading Excel Files

```python
import pandas as pd
from typing import Dict, List, Optional

def read_excel_sheets(
    file_path: str,
    sheet_names: Optional[List[str]] = None,
    **read_kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Robust Excel reading with error handling.

    Args:
        file_path: Path to Excel file
        sheet_names: List of sheet names to read (None = all sheets)
        **read_kwargs: Additional arguments passed to pd.read_excel

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    try:
        with pd.ExcelFile(file_path) as xls:
            available_sheets = xls.sheet_names

            # Determine which sheets to read
            if sheet_names is None:
                sheets_to_read = available_sheets
            else:
                # Validate requested sheets exist
                sheets_to_read = []
                for name in sheet_names:
                    if name in available_sheets:
                        sheets_to_read.append(name)
                    else:
                        print(f"Warning: Sheet '{name}' not found. Available: {available_sheets}")

            # Read sheets
            result = {}
            for sheet in sheets_to_read:
                try:
                    result[sheet] = pd.read_excel(xls, sheet, **read_kwargs)
                    print(f"✓ Read sheet '{sheet}': {result[sheet].shape}")
                except Exception as e:
                    print(f"✗ Error reading sheet '{sheet}': {e}")

            return result

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return {}
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return {}

# Usage
sheets = read_excel_sheets(
    'teams_export.xlsx',
    sheet_names=['Schichten', 'Arbeitsfreie Zeit'],
    na_values=['', 'N/A'],
    parse_dates=True
)
```

---

### Writing Multi-Sheet Excel Files

#### Using Pandas ExcelWriter

**Required Libraries:**
```bash
pip install pandas openpyxl  # For .xlsx with basic formatting
# OR
pip install pandas xlsxwriter  # For .xlsx with advanced formatting
```

#### 1. Basic Multi-Sheet Writing with openpyxl

```python
import pandas as pd

# Create sample data
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})

# Write to Excel with multiple sheets
with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

print("Excel file created with 2 sheets")
```

#### 2. Advanced Formatting with XlsxWriter

```python
import pandas as pd

# Create data
df1 = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Orange'],
    'Quantity': [10, 15, 8],
    'Price': [1.50, 0.80, 1.20]
})

df2 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
})

# Write with formatting
with pd.ExcelWriter('formatted_output.xlsx', engine='xlsxwriter') as writer:
    # Write first sheet
    df1.to_excel(writer, sheet_name='Products', index=False, startrow=1)

    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet1 = writer.sheets['Products']

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D3D3D3',
        'border': 1
    })

    currency_format = workbook.add_format({'num_format': '$#,##0.00'})

    # Write custom header
    for col_num, value in enumerate(df1.columns.values):
        worksheet1.write(1, col_num, value, header_format)

    # Format price column
    worksheet1.set_column('C:C', 12, currency_format)

    # Set column widths
    worksheet1.set_column('A:A', 15)  # Product column
    worksheet1.set_column('B:B', 10)  # Quantity column

    # Add title
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'align': 'center'
    })
    worksheet1.merge_range('A1:C1', 'Product Inventory', title_format)

    # Write second sheet
    df2.to_excel(writer, sheet_name='Contacts', index=False)
    worksheet2 = writer.sheets['Contacts']

    # Autofit columns for second sheet
    for i, col in enumerate(df2.columns):
        column_width = max(df2[col].astype(str).map(len).max(), len(col)) + 2
        worksheet2.set_column(i, i, column_width)

print("Formatted Excel file created")
```

#### 3. Column Width Auto-Adjustment

```python
import pandas as pd

def autofit_columns(writer, sheet_name, df):
    """Auto-adjust column widths based on content."""
    worksheet = writer.sheets[sheet_name]

    for i, col in enumerate(df.columns):
        # Find max length in column (including header)
        max_length = max(
            df[col].astype(str).map(len).max(),  # Max data length
            len(str(col))  # Header length
        )
        # Add padding
        column_width = min(max_length + 2, 50)  # Cap at 50
        worksheet.set_column(i, i, column_width)

# Usage
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    autofit_columns(writer, 'Data', df)
```

#### 4. Appending to Existing Excel File (openpyxl only)

```python
import pandas as pd
from openpyxl import load_workbook

# Create initial file
df1 = pd.DataFrame({'A': [1, 2, 3]})
with pd.ExcelWriter('existing.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)

# Append new sheet to existing file
df2 = pd.DataFrame({'B': [4, 5, 6]})
with pd.ExcelWriter('existing.xlsx', engine='openpyxl', mode='a') as writer:
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

print("Sheet2 appended to existing file")
```

#### 5. Writing with Conditional Formatting

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Score': [85, 92, 78, 95]
})

with pd.ExcelWriter('conditional.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Scores', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Scores']

    # Add conditional formatting (highlight scores >= 90)
    green_format = workbook.add_format({'bg_color': '#C6EFCE'})

    worksheet.conditional_format('B2:B5', {
        'type': 'cell',
        'criteria': '>=',
        'value': 90,
        'format': green_format
    })

    # Set column widths
    worksheet.set_column('A:A', 15)
    worksheet.set_column('B:B', 10)
```

#### 6. Best Practices for Writing Excel Files

```python
import pandas as pd
from typing import Dict
from pathlib import Path

def write_excel_sheets(
    file_path: str,
    sheets_data: Dict[str, pd.DataFrame],
    engine: str = 'xlsxwriter',
    apply_formatting: bool = True
) -> bool:
    """
    Write multiple DataFrames to Excel with optional formatting.

    Args:
        file_path: Output Excel file path
        sheets_data: Dict mapping sheet names to DataFrames
        engine: 'xlsxwriter' or 'openpyxl'
        apply_formatting: Whether to apply basic formatting

    Returns:
        True if successful, False otherwise
    """
    try:
        with pd.ExcelWriter(file_path, engine=engine) as writer:
            for sheet_name, df in sheets_data.items():
                # Write sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                if apply_formatting and engine == 'xlsxwriter':
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]

                    # Header formatting
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#D3D3D3',
                        'border': 1,
                        'align': 'center'
                    })

                    # Write headers with format
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)

                    # Auto-adjust column widths
                    for i, col in enumerate(df.columns):
                        max_length = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        )
                        worksheet.set_column(i, i, min(max_length + 2, 50))

                print(f"✓ Written sheet '{sheet_name}': {df.shape}")

        print(f"✓ Excel file created: {file_path}")
        return True

    except Exception as e:
        print(f"✗ Error writing Excel file: {e}")
        return False

# Usage
sheets = {
    'Shifts': shifts_df,
    'TimeOff': timeoff_df,
    'Summary': summary_df
}

success = write_excel_sheets(
    'teams_export.xlsx',
    sheets,
    engine='xlsxwriter',
    apply_formatting=True
)
```

---

### Microsoft Teams Shifts Excel Format

#### Typical Teams Shifts Export Structure

Microsoft Teams Shifts typically exports data with **multiple sheets** containing different types of information:

1. **Shifts Sheet** (German: "Schichten")
2. **Time-Off Sheet** (German: "Arbeitsfreie Zeit")
3. **Members Sheet** (German: "Mitglieder")

#### 1. Shifts Sheet Column Structure

**German Column Names (Typical Export):**
- `Mitglied` - Member/Employee name
- `Startdatum` - Start date
- `Startzeit` - Start time
- `Enddatum` - End date
- `Endzeit` - End time
- `Schichtnotizen` - Shift notes
- `Themenfarbe` - Theme color (color code 1-13)
- `Gruppe` - Group name
- `E-Mail-Adresse` - Email address

**English Equivalents:**
- `Member` or `Employee`
- `Start date`
- `Start time`
- `End date`
- `End time`
- `Shift notes` or `Notes`
- `Theme color` or `Color`
- `Group`
- `Email` or `Email address`

**Example Data:**
```python
shifts_df = pd.DataFrame({
    'Mitglied': ['Alice Smith', 'Bob Jones'],
    'Startdatum': ['1/15/2025', '1/15/2025'],
    'Startzeit': ['08:00', '10:00'],
    'Enddatum': ['1/15/2025', '1/15/2025'],
    'Endzeit': ['16:00', '18:00'],
    'Schichtnotizen': ['Morning shift', 'Afternoon shift'],
    'Themenfarbe': ['2. Blau', '4. Lila'],
    'Gruppe': ['Support Team', 'Support Team'],
    'E-Mail-Adresse': ['alice@company.com', 'bob@company.com']
})
```

#### 2. Time-Off Sheet Column Structure

**German Column Names:**
- `Mitglied` - Member name
- `Startdatum` - Start date
- `Enddatum` - End date
- `Beschreibung` - Description
- `Themenfarbe` - Theme color (usually "13. Grau" for time-off)
- `E-Mail-Adresse` - Email address

**Example Data:**
```python
timeoff_df = pd.DataFrame({
    'Mitglied': ['Alice Smith', 'Bob Jones'],
    'Startdatum': ['1/20/2025', '1/22/2025'],
    'Enddatum': ['1/22/2025', '1/24/2025'],
    'Beschreibung': ['Vacation', 'Sick leave'],
    'Themenfarbe': ['13. Grau', '13. Grau'],
    'E-Mail-Adresse': ['alice@company.com', 'bob@company.com']
})
```

#### 3. Members Sheet Column Structure

**German Column Names:**
- `Name` - Member name
- `E-Mail-Adresse` - Email address
- `Gruppe` - Group membership
- `Tags` - Additional tags

#### 4. Teams Color Coding System

Teams uses 13 predefined colors for shift categorization:

| Code | German Name | English | Typical Use |
|------|-------------|---------|-------------|
| 1 | Weiß | White | Operation Lead |
| 2 | Blau | Blue | Contact Team, Dispatcher (07:00-16:00) |
| 3 | Grün | Green | Contact Team, SOB roles |
| 4 | Lila | Purple | Late shifts (10:00-19:00) |
| 5 | Rosa | Pink | Special assignments |
| 6 | Gelb | Yellow | Late shifts (09:00-18:00) |
| 8 | Dunkelblau | Dark Blue | Project work |
| 9 | Dunkelgrün | Dark Green | WoVe, PCV roles |
| 10 | Dunkelviolett | Dark Purple | Pikett (on-call) |
| 11 | Dunkelrosa | Dark Pink | People Developer |
| 12 | Dunkelgelb | Dark Yellow | Livechat shifts |
| 13 | Grau | Gray | Time-off, holidays, sick |

**Note:** Color 7 is not used in the Teams system.

#### 5. Date and Time Formatting Requirements

**For Teams Import:**
- **Date Format:** `M/D/YYYY` (e.g., "1/15/2025", "12/5/2024")
- **Time Format:** `HH:MM` in 24-hour format (e.g., "08:00", "14:30")
- **Timezone:** Times are typically local to the Teams organization

```python
import pandas as pd
from datetime import datetime

def format_for_teams_import(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame for Teams Shifts import."""
    df_copy = df.copy()

    # Convert dates to Teams format (M/D/YYYY)
    date_columns = ['Startdatum', 'Enddatum']
    for col in date_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col]).dt.strftime('%-m/%-d/%Y')

    # Ensure time format is HH:MM
    time_columns = ['Startzeit', 'Endzeit']
    for col in time_columns:
        if col in df_copy.columns:
            # Normalize to HH:MM format
            df_copy[col] = pd.to_datetime(
                df_copy[col], format='%H:%M'
            ).dt.strftime('%H:%M')

    return df_copy
```

#### 6. Complete Teams Import/Export Example

```python
import pandas as pd
from typing import Dict, Tuple

def parse_teams_export(
    file_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse Microsoft Teams Shifts export file.

    Returns:
        Tuple of (shifts_df, timeoff_df, members_df)
    """
    # Read all sheets
    with pd.ExcelFile(file_path) as xls:
        sheet_names = xls.sheet_names

        # Find sheets (support German and English names)
        shifts_sheet = None
        timeoff_sheet = None
        members_sheet = None

        for name in sheet_names:
            name_lower = name.lower()
            if 'schicht' in name_lower or 'shift' in name_lower:
                shifts_sheet = name
            elif 'arbeitsfreie' in name_lower or 'time' in name_lower or 'off' in name_lower:
                timeoff_sheet = name
            elif 'mitglied' in name_lower or 'member' in name_lower:
                members_sheet = name

        # Read sheets
        shifts_df = pd.read_excel(xls, shifts_sheet) if shifts_sheet else pd.DataFrame()
        timeoff_df = pd.read_excel(xls, timeoff_sheet) if timeoff_sheet else pd.DataFrame()
        members_df = pd.read_excel(xls, members_sheet) if members_sheet else pd.DataFrame()

    return shifts_df, timeoff_df, members_df


def create_teams_export(
    shifts_data: pd.DataFrame,
    timeoff_data: pd.DataFrame,
    members_data: pd.DataFrame,
    output_path: str
) -> bool:
    """
    Create Teams Shifts-compatible Excel file.

    Args:
        shifts_data: Shifts DataFrame with columns: Mitglied, Startdatum, Startzeit, etc.
        timeoff_data: Time-off DataFrame
        members_data: Members DataFrame
        output_path: Output file path

    Returns:
        True if successful
    """
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Format data for Teams
            shifts_formatted = format_for_teams_import(shifts_data)
            timeoff_formatted = format_for_teams_import(timeoff_data)

            # Write sheets
            shifts_formatted.to_excel(writer, sheet_name='Schichten', index=False)
            timeoff_formatted.to_excel(writer, sheet_name='Arbeitsfreie Zeit', index=False)
            members_data.to_excel(writer, sheet_name='Mitglieder', index=False)

            # Get workbook and sheets for formatting
            workbook = writer.book

            # Format each sheet
            for sheet_name in ['Schichten', 'Arbeitsfreie Zeit', 'Mitglieder']:
                worksheet = writer.sheets[sheet_name]

                # Header format
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#4472C4',
                    'font_color': 'white',
                    'border': 1
                })

                # Get DataFrame for column count
                if sheet_name == 'Schichten':
                    df = shifts_formatted
                elif sheet_name == 'Arbeitsfreie Zeit':
                    df = timeoff_formatted
                else:
                    df = members_data

                # Apply header formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Auto-adjust columns
                for i, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    worksheet.set_column(i, i, min(max_length + 2, 40))

        print(f"✓ Teams export created: {output_path}")
        return True

    except Exception as e:
        print(f"✗ Error creating Teams export: {e}")
        return False
```

---

### Streamlit File Upload Integration

#### 1. Basic Excel File Upload

```python
import streamlit as st
import pandas as pd

# File uploader
uploaded_file = st.file_uploader(
    "Upload Excel file",
    type=["xlsx", "xls"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Read Excel file
    try:
        # Method 1: Direct read (reads first sheet by default)
        df = pd.read_excel(uploaded_file)
        st.success(f"File uploaded: {uploaded_file.name}")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error reading file: {e}")
```

#### 2. Multi-Sheet Preview with Selection

```python
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read all sheets
    with pd.ExcelFile(uploaded_file) as xls:
        sheet_names = xls.sheet_names

        # Let user select sheet
        selected_sheet = st.selectbox("Select sheet to view:", sheet_names)

        # Read selected sheet
        df = pd.read_excel(xls, selected_sheet)

        # Display info
        st.info(f"Sheet '{selected_sheet}' has {len(df)} rows and {len(df.columns)} columns")

        # Preview data
        st.dataframe(df.head(10))

        # Show column info
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(col_info)
```

#### 3. Multi-Sheet Upload with Teams Shifts Parsing

```python
import streamlit as st
import pandas as pd
from typing import Dict

def parse_teams_shifts_file(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Parse uploaded Teams Shifts Excel file."""
    sheets = {}

    try:
        with pd.ExcelFile(uploaded_file) as xls:
            sheet_names = xls.sheet_names

            st.info(f"Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")

            # Auto-detect sheet types
            for name in sheet_names:
                df = pd.read_excel(xls, name)
                sheets[name] = df

                # Categorize by content
                if 'schicht' in name.lower() or 'shift' in name.lower():
                    st.success(f"✓ Found Shifts sheet: {name}")
                elif 'arbeitsfreie' in name.lower() or 'time-off' in name.lower():
                    st.success(f"✓ Found Time-off sheet: {name}")
                elif 'mitglied' in name.lower() or 'member' in name.lower():
                    st.success(f"✓ Found Members sheet: {name}")

        return sheets

    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return {}

# Streamlit app
st.title("Teams Shifts File Parser")

uploaded_file = st.file_uploader(
    "Upload Teams Shifts Export",
    type=["xlsx"],
    help="Upload the Excel file exported from Microsoft Teams Shifts"
)

if uploaded_file is not None:
    sheets = parse_teams_shifts_file(uploaded_file)

    if sheets:
        # Let user select which sheet to view
        tab_names = list(sheets.keys())
        tabs = st.tabs(tab_names)

        for i, (sheet_name, df) in enumerate(sheets.items()):
            with tabs[i]:
                st.subheader(f"Sheet: {sheet_name}")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

                # Show preview
                st.dataframe(df.head(20))

                # Download button for this sheet
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download {sheet_name} as CSV",
                    data=csv,
                    file_name=f"{sheet_name}.csv",
                    mime="text/csv"
                )
```

#### 4. Dual File Upload (Shifts + Time-Off)

```python
import streamlit as st
import pandas as pd

st.title("Teams Shifts Import")

col1, col2 = st.columns(2)

with col1:
    shifts_file = st.file_uploader(
        "Upload Shifts File",
        type=["xlsx", "xls"],
        key="shifts"
    )

with col2:
    timeoff_file = st.file_uploader(
        "Upload Time-Off File",
        type=["xlsx", "xls"],
        key="timeoff"
    )

if shifts_file and timeoff_file:
    # Read both files
    shifts_df = pd.read_excel(shifts_file)
    timeoff_df = pd.read_excel(timeoff_file)

    st.success("Both files uploaded successfully!")

    # Display in tabs
    tab1, tab2 = st.tabs(["Shifts", "Time-Off"])

    with tab1:
        st.write(f"**Shifts:** {len(shifts_df)} entries")
        st.dataframe(shifts_df)

    with tab2:
        st.write(f"**Time-Off:** {len(timeoff_df)} entries")
        st.dataframe(timeoff_df)

    # Combine and process
    if st.button("Process Combined Data"):
        # Add type column
        shifts_df['Type'] = 'shift'
        timeoff_df['Type'] = 'time_off'

        # Combine
        combined_df = pd.concat([shifts_df, timeoff_df], ignore_index=True)

        st.write(f"**Combined:** {len(combined_df)} total entries")
        st.dataframe(combined_df)
```

#### 5. Complete Streamlit Integration Example

```python
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

def process_teams_file(uploaded_file):
    """Process uploaded Teams Shifts file."""
    with pd.ExcelFile(uploaded_file) as xls:
        sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}
    return sheets

def create_download_excel(sheets_dict):
    """Create downloadable Excel file from sheets dictionary."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output.getvalue()

# Main app
st.title("📊 Teams Shifts File Manager")

# Upload section
st.header("1. Upload File")
uploaded_file = st.file_uploader(
    "Choose Teams Shifts Excel file",
    type=["xlsx"],
    help="Upload the exported Excel file from Microsoft Teams Shifts"
)

if uploaded_file:
    # Process file
    with st.spinner("Processing file..."):
        sheets = process_teams_file(uploaded_file)

    st.success(f"✓ Loaded {len(sheets)} sheets")

    # Sheet selection
    st.header("2. Preview Data")
    selected_sheet = st.selectbox("Select sheet:", list(sheets.keys()))

    if selected_sheet:
        df = sheets[selected_sheet]

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        # Data preview
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # Column info
        with st.expander("📋 Column Details"):
            col_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count(),
                'Unique': df.nunique()
            })
            st.dataframe(col_df, use_container_width=True)

    # Export section
    st.header("3. Export Data")

    export_sheets = st.multiselect(
        "Select sheets to export:",
        list(sheets.keys()),
        default=list(sheets.keys())
    )

    if export_sheets:
        export_data = {name: sheets[name] for name in export_sheets}
        excel_data = create_download_excel(export_data)

        st.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name=f"teams_shifts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
```

---

### Complete Implementation Examples

#### Example 1: Teams Shifts Import/Export Handler

```python
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
from datetime import datetime
from io import BytesIO

class TeamsShiftsHandler:
    """Complete handler for Teams Shifts Excel files."""

    # German column mappings
    SHIFTS_COLUMNS = {
        'de': ['Mitglied', 'Startdatum', 'Startzeit', 'Enddatum', 'Endzeit',
               'Schichtnotizen', 'Themenfarbe', 'Gruppe', 'E-Mail-Adresse'],
        'en': ['Member', 'Start date', 'Start time', 'End date', 'End time',
               'Shift notes', 'Theme color', 'Group', 'Email']
    }

    TIMEOFF_COLUMNS = {
        'de': ['Mitglied', 'Startdatum', 'Enddatum', 'Beschreibung', 'Themenfarbe', 'E-Mail-Adresse'],
        'en': ['Member', 'Start date', 'End date', 'Description', 'Theme color', 'Email']
    }

    COLOR_CODES = {
        '1. Weiß': 'Operation Lead',
        '2. Blau': 'Contact Team',
        '3. Grün': 'SOB roles',
        '4. Lila': 'Late shift (10-19)',
        '5. Rosa': 'Special assignment',
        '6. Gelb': 'Late shift (09-18)',
        '8. Dunkelblau': 'Project work',
        '9. Dunkelgrün': 'WoVe/PCV',
        '10. Dunkelviolett': 'Pikett',
        '11. Dunkelrosa': 'People Developer',
        '12. Dunkelgelb': 'Livechat',
        '13. Grau': 'Time-off'
    }

    @staticmethod
    def detect_sheet_type(df: pd.DataFrame) -> str:
        """Detect if sheet is shifts, time-off, or members."""
        columns_lower = [col.lower() for col in df.columns]

        # Check for time columns (shifts have both start/end time)
        has_time_cols = any('zeit' in col or 'time' in col for col in columns_lower)
        has_date_cols = any('datum' in col or 'date' in col for col in columns_lower)

        if has_time_cols and has_date_cols:
            return 'shifts'
        elif has_date_cols and any('beschreibung' in col or 'description' in col for col in columns_lower):
            return 'timeoff'
        else:
            return 'members'

    @staticmethod
    def normalize_columns(df: pd.DataFrame, sheet_type: str) -> pd.DataFrame:
        """Normalize column names to German standard."""
        df_copy = df.copy()

        if sheet_type == 'shifts':
            column_map = {
                # English to German
                'member': 'Mitglied',
                'start date': 'Startdatum',
                'start time': 'Startzeit',
                'end date': 'Enddatum',
                'end time': 'Endzeit',
                'shift notes': 'Schichtnotizen',
                'notes': 'Schichtnotizen',
                'theme color': 'Themenfarbe',
                'color': 'Themenfarbe',
                'group': 'Gruppe',
                'email': 'E-Mail-Adresse',
                'email address': 'E-Mail-Adresse'
            }
        elif sheet_type == 'timeoff':
            column_map = {
                'member': 'Mitglied',
                'start date': 'Startdatum',
                'end date': 'Enddatum',
                'description': 'Beschreibung',
                'theme color': 'Themenfarbe',
                'email': 'E-Mail-Adresse'
            }
        else:
            return df_copy

        # Apply mapping
        df_copy.columns = [
            column_map.get(col.lower(), col) for col in df_copy.columns
        ]

        return df_copy

    @staticmethod
    def format_for_teams(df: pd.DataFrame, sheet_type: str) -> pd.DataFrame:
        """Format DataFrame for Teams import."""
        df_copy = df.copy()

        # Date columns
        date_cols = ['Startdatum', 'Enddatum']
        for col in date_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                df_copy[col] = df_copy[col].dt.strftime('%-m/%-d/%Y')

        # Time columns (shifts only)
        if sheet_type == 'shifts':
            time_cols = ['Startzeit', 'Endzeit']
            for col in time_cols:
                if col in df_copy.columns:
                    # Ensure HH:MM format
                    df_copy[col] = pd.to_datetime(
                        df_copy[col], format='%H:%M', errors='coerce'
                    ).dt.strftime('%H:%M')

        return df_copy

    @classmethod
    def parse_file(cls, file_path_or_buffer) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """
        Parse Teams Shifts Excel file.

        Returns:
            Dict mapping sheet names to (DataFrame, sheet_type) tuples
        """
        result = {}

        with pd.ExcelFile(file_path_or_buffer) as xls:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                sheet_type = cls.detect_sheet_type(df)
                df_normalized = cls.normalize_columns(df, sheet_type)

                result[sheet_name] = (df_normalized, sheet_type)

        return result

    @classmethod
    def create_export(
        cls,
        shifts_df: pd.DataFrame,
        timeoff_df: pd.DataFrame = None,
        members_df: pd.DataFrame = None
    ) -> BytesIO:
        """
        Create Teams-compatible Excel export.

        Returns:
            BytesIO object containing the Excel file
        """
        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Format and write shifts
            shifts_formatted = cls.format_for_teams(shifts_df, 'shifts')
            shifts_formatted.to_excel(writer, sheet_name='Schichten', index=False)

            # Format and write time-off if provided
            if timeoff_df is not None and not timeoff_df.empty:
                timeoff_formatted = cls.format_for_teams(timeoff_df, 'timeoff')
                timeoff_formatted.to_excel(writer, sheet_name='Arbeitsfreie Zeit', index=False)

            # Write members if provided
            if members_df is not None and not members_df.empty:
                members_df.to_excel(writer, sheet_name='Mitglieder', index=False)

            # Apply formatting to all sheets
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })

            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]

                # Get corresponding DataFrame
                if sheet_name == 'Schichten':
                    df = shifts_formatted
                elif sheet_name == 'Arbeitsfreie Zeit':
                    df = timeoff_formatted
                else:
                    df = members_df

                # Apply header formatting
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Auto-adjust columns
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(str(col)))
                    worksheet.set_column(i, i, min(max_len + 2, 40))

        output.seek(0)
        return output

# Streamlit app using the handler
def main():
    st.title("🔄 Teams Shifts File Converter")

    uploaded_file = st.file_uploader(
        "Upload Teams Shifts Excel",
        type=["xlsx"],
        help="Upload exported file from Microsoft Teams Shifts"
    )

    if uploaded_file:
        # Parse file
        with st.spinner("Parsing file..."):
            sheets = TeamsShiftsHandler.parse_file(uploaded_file)

        # Display results
        st.success(f"✓ Parsed {len(sheets)} sheets")

        # Show each sheet
        for sheet_name, (df, sheet_type) in sheets.items():
            with st.expander(f"📄 {sheet_name} ({sheet_type})"):
                st.dataframe(df.head(10))
                st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Export section
        st.header("Export")

        # Get shifts and time-off DataFrames
        shifts_df = None
        timeoff_df = None
        members_df = None

        for sheet_name, (df, sheet_type) in sheets.items():
            if sheet_type == 'shifts':
                shifts_df = df
            elif sheet_type == 'timeoff':
                timeoff_df = df
            elif sheet_type == 'members':
                members_df = df

        if shifts_df is not None:
            if st.button("Generate Teams Export"):
                export_data = TeamsShiftsHandler.create_export(
                    shifts_df,
                    timeoff_df,
                    members_df
                )

                st.download_button(
                    label="📥 Download Teams Export",
                    data=export_data,
                    file_name=f"teams_shifts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
```

#### Example 2: Robust Multi-Sheet Parser with Validation

```python
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class SheetType(Enum):
    SHIFTS = "shifts"
    TIMEOFF = "timeoff"
    MEMBERS = "members"
    UNKNOWN = "unknown"

@dataclass
class SheetValidation:
    """Validation result for a sheet."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    column_count: int

class MultiSheetValidator:
    """Validate multi-sheet Excel files."""

    REQUIRED_COLUMNS = {
        SheetType.SHIFTS: ['Mitglied', 'Startdatum', 'Startzeit', 'Enddatum', 'Endzeit'],
        SheetType.TIMEOFF: ['Mitglied', 'Startdatum', 'Enddatum'],
        SheetType.MEMBERS: ['Name', 'E-Mail-Adresse']
    }

    @classmethod
    def detect_type(cls, df: pd.DataFrame) -> SheetType:
        """Auto-detect sheet type from columns."""
        columns_lower = {col.lower() for col in df.columns}

        # Check shifts (has time columns)
        if {'startzeit', 'endzeit'}.issubset(columns_lower) or \
           {'start time', 'end time'}.issubset(columns_lower):
            return SheetType.SHIFTS

        # Check time-off (has description)
        if 'beschreibung' in columns_lower or 'description' in columns_lower:
            return SheetType.TIMEOFF

        # Check members
        if 'tags' in columns_lower or ('name' in columns_lower and 'gruppe' in columns_lower):
            return SheetType.MEMBERS

        return SheetType.UNKNOWN

    @classmethod
    def validate_sheet(
        cls,
        df: pd.DataFrame,
        sheet_type: SheetType
    ) -> SheetValidation:
        """Validate a single sheet."""
        errors = []
        warnings = []

        # Check if sheet is empty
        if df.empty:
            errors.append("Sheet is empty")
            return SheetValidation(False, errors, warnings, 0, 0)

        # Get required columns for this type
        required_cols = cls.REQUIRED_COLUMNS.get(sheet_type, [])

        # Check for required columns
        missing_cols = []
        for req_col in required_cols:
            # Case-insensitive check
            if not any(req_col.lower() == col.lower() for col in df.columns):
                missing_cols.append(req_col)

        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        # Check for null values in key columns
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(df)) * 100
                if null_pct > 50:
                    warnings.append(f"Column '{col}' has {null_pct:.1f}% null values")

        # Date validation
        date_cols = [col for col in df.columns if 'datum' in col.lower() or 'date' in col.lower()]
        for col in date_cols:
            try:
                pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                warnings.append(f"Column '{col}' may contain invalid dates")

        is_valid = len(errors) == 0

        return SheetValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            column_count=len(df.columns)
        )

    @classmethod
    def validate_file(
        cls,
        file_path_or_buffer
    ) -> Dict[str, Tuple[SheetType, SheetValidation]]:
        """Validate entire Excel file."""
        results = {}

        with pd.ExcelFile(file_path_or_buffer) as xls:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                sheet_type = cls.detect_type(df)
                validation = cls.validate_sheet(df, sheet_type)

                results[sheet_name] = (sheet_type, validation)

        return results

# Streamlit app with validation
import streamlit as st

def show_validation_results(results: Dict[str, Tuple[SheetType, SheetValidation]]):
    """Display validation results in Streamlit."""

    for sheet_name, (sheet_type, validation) in results.items():
        with st.expander(f"📋 {sheet_name} - {sheet_type.value.title()}"):
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", validation.row_count)
            col2.metric("Columns", validation.column_count)

            if validation.is_valid:
                col3.success("✓ Valid")
            else:
                col3.error("✗ Invalid")

            # Errors
            if validation.errors:
                st.error("**Errors:**")
                for error in validation.errors:
                    st.write(f"• {error}")

            # Warnings
            if validation.warnings:
                st.warning("**Warnings:**")
                for warning in validation.warnings:
                    st.write(f"• {warning}")

# Main app
st.title("📊 Excel File Validator")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    with st.spinner("Validating file..."):
        results = MultiSheetValidator.validate_file(uploaded_file)

    show_validation_results(results)

    # Summary
    total_sheets = len(results)
    valid_sheets = sum(1 for _, (_, v) in results.items() if v.is_valid)

    if valid_sheets == total_sheets:
        st.success(f"✓ All {total_sheets} sheets are valid!")
    else:
        st.warning(f"⚠️ {valid_sheets}/{total_sheets} sheets are valid")
```

---

### Summary and Best Practices

#### Key Takeaways

1. **Use `pd.ExcelFile` context manager** for reading multiple sheets efficiently
2. **Always check sheet names** before accessing - they may vary
3. **Normalize column names** to handle German/English variations
4. **Use `xlsxwriter` for advanced formatting**, `openpyxl` for basic needs
5. **Validate data** before processing, especially dates and times
6. **Handle errors gracefully** with try/except blocks
7. **Provide user feedback** in Streamlit with progress indicators

#### Common Pitfalls to Avoid

1. ❌ **Don't assume sheet names** - always list them first
2. ❌ **Don't read files multiple times** - use ExcelFile context manager
3. ❌ **Don't ignore date/time formats** - Teams requires specific formats
4. ❌ **Don't forget to close files** - use context managers
5. ❌ **Don't hardcode column names** - support multiple languages
6. ❌ **Don't skip validation** - invalid data causes import failures

#### Recommended Libraries

```bash
# Core
pip install pandas>=2.0.0

# Excel engines
pip install openpyxl>=3.1.0      # For .xlsx (read/write, append mode)
pip install xlsxwriter>=3.1.0    # For .xlsx (advanced formatting)
pip install xlrd>=2.0.0           # For legacy .xls files

# Streamlit
pip install streamlit>=1.28.0
```

#### Where to Find Information

- **Reading Multi-Sheet Files:** Section "Reading Multi-Sheet Excel Files"
- **Writing with Formatting:** Section "Writing Multi-Sheet Excel Files"
- **Teams Shifts Format:** Section "Microsoft Teams Shifts Excel Format"
- **Streamlit Integration:** Section "Streamlit File Upload Integration"
- **Complete Examples:** Section "Complete Implementation Examples"
