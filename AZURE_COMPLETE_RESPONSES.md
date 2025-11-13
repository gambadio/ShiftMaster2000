# Azure OpenAI: Ensuring Complete Responses (No Truncation)

## Problem
When using Azure GPT-5 for plan generation, the model truncates the response mid-sentence and acts like it can continue in the next turn. The `finish_reason` shows `"length"` instead of `"stop"`.

## Root Cause
The `max_completion_tokens` parameter is too low for the task. When the model hits this limit, it stops generating mid-response.

## Solution: Increase `max_completion_tokens`

### Understanding `finish_reason`
Azure OpenAI returns a `finish_reason` field that indicates why the model stopped:

- ✅ **`stop`**: Complete output - model finished naturally
- ⚠️ **`length`**: **TRUNCATED** - hit `max_completion_tokens` limit
- 🚫 **`content_filter`**: Content was filtered out
- ⏳ **`null`**: Response still streaming/in progress

### Recommended Token Limits

```python
# Task-based recommendations for max_completion_tokens

# Small tasks (Q&A, simple queries)
max_completion_tokens = 1000-2000

# Medium tasks (explanations, summaries)
max_completion_tokens = 4000-8000

# Large tasks (plans, detailed analysis)
max_completion_tokens = 16000-32000

# Very large tasks (comprehensive plans, documentation)
max_completion_tokens = 64000-100000

# Maximum supported by GPT-5
max_completion_tokens = 128000  # DO NOT EXCEED
```

### Checking for Truncation

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://YOUR-RESOURCE.openai.azure.com",
    api_key="YOUR-KEY",
    api_version="2024-10-21"
)

response = client.chat.completions.create(
    model="gpt-5-deployment-name",
    messages=[
        {"role": "system", "content": "You are a helpful planning assistant."},
        {"role": "user", "content": "Create a detailed project plan..."}
    ],
    max_completion_tokens=16000,  # ← Increased from 4096
    stream=False
)

# Check if response was complete
finish_reason = response.choices[0].finish_reason

if finish_reason == "stop":
    print("✅ COMPLETE: Model finished naturally")
    print(response.choices[0].message.content)
    
elif finish_reason == "length":
    print("⚠️ TRUNCATED: Increase max_completion_tokens!")
    print(f"Used tokens: {response.usage.completion_tokens}")
    print(f"Current limit: 16000")
    print(f"Recommendation: Try 32000-64000 tokens")
    
elif finish_reason == "content_filter":
    print("🚫 FILTERED: Content violated policy")
```

### Streaming with Truncation Detection

```python
def stream_with_truncation_check(client, messages, max_tokens=16000):
    """Stream response and check for truncation"""
    
    stream = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        max_completion_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True}  # Get usage stats
    )
    
    full_response = ""
    finish_reason = None
    
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                print(delta, end="", flush=True)
            
            # Check finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
    
    print("\n")
    
    # Warn if truncated
    if finish_reason == "length":
        print(f"\n⚠️ WARNING: Response was truncated!")
        print(f"Consider increasing max_completion_tokens from {max_tokens}")
        print(f"Current response length: ~{len(full_response)} chars")
    elif finish_reason == "stop":
        print(f"\n✅ Complete response generated")
    
    return full_response, finish_reason
```

## Azure GPT-5 Specific Considerations

### Context vs Completion Tokens

**Important**: GPT-5 models have **different limits** for input and output:

```python
# GPT-5 Token Limits
CONTEXT_WINDOW = 400_000  # Max input tokens (prompt + history)
MAX_OUTPUT_TOKENS = 128_000  # Max completion tokens (response)

# Your effective limits:
max_total_tokens = min(
    CONTEXT_WINDOW,
    len(prompt_tokens) + max_completion_tokens
)
```

### Using `max_completion_tokens` (not `max_tokens`)

```python
# ❌ WRONG (deprecated for Azure)
response = client.chat.completions.create(
    model="gpt-5",
    messages=messages,
    max_tokens=4096  # Old parameter, may not work
)

# ✅ CORRECT (Azure GPT-5)
response = client.chat.completions.create(
    model="gpt-5",
    messages=messages,
    max_completion_tokens=16000  # New parameter
)
```

### Your Current Settings

From `models.py` in ShiftMaster2000:
```python
# OLD (causes truncation)
max_tokens: int = 4096

# NEW (allows complete responses)
max_tokens: int = 16000  # or higher for complex tasks
```

## Best Practices

### 1. Start with Conservative Limits
```python
# For plan generation
max_completion_tokens = 16000  # Good starting point
```

### 2. Monitor and Adjust
```python
if finish_reason == "length":
    # Double the limit and retry
    max_completion_tokens *= 2
```

### 3. Log Token Usage
```python
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# For GPT-5 reasoning models
if hasattr(response.usage, 'completion_tokens_details'):
    reasoning = response.usage.completion_tokens_details.reasoning_tokens
    print(f"Reasoning tokens: {reasoning}")
```

### 4. Set UI Warnings
```python
# In your Streamlit app
if finish_reason == "length":
    st.warning(
        "⚠️ The response was truncated. "
        "Increase 'Max Tokens' in settings to get complete output."
    )
```

## Code Changes for ShiftMaster2000

### 1. Update Default in `models.py`
```python
# Change default from 4096 to 16000
max_tokens: int = 16000  # Allows longer complete responses
```

### 2. Add Truncation Check in `llm_client.py`
```python
def complete(self, messages, ...):
    # ... existing code ...
    
    result = {
        "content": content,
        "usage": {...},
        "finish_reason": choice.finish_reason,  # Already there
        "model": response.model
    }
    
    # Warn if truncated
    if choice.finish_reason == "length":
        print("⚠️ WARNING: Response truncated! Increase max_completion_tokens")
    
    return result
```

### 3. Update UI in `app.py`
```python
# In settings sidebar
st.info(
    "💡 Tip: For long plans, use 16,000-32,000 tokens. "
    "Check 'finish_reason' to ensure complete responses."
)

# After generation
if result.get("finish_reason") == "length":
    st.warning(
        "⚠️ Plan was truncated. Increase Max Tokens in settings."
    )
```

## Testing

```python
# Test with a complex planning task
messages = [
    {"role": "system", "content": "You are a detailed project planner."},
    {"role": "user", "content": """
        Create a comprehensive 12-month project plan for building a 
        new e-commerce platform including:
        - Architecture design
        - Team structure
        - Development phases
        - Testing strategy
        - Deployment plan
        - Risk mitigation
        
        Be very detailed with each phase.
    """}
]

# Test with different limits
for max_tokens in [4096, 8192, 16000, 32000]:
    print(f"\n=== Testing with {max_tokens} tokens ===")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages,
        max_completion_tokens=max_tokens
    )
    
    print(f"Used: {response.usage.completion_tokens}")
    print(f"Finish: {response.choices[0].finish_reason}")
    
    if response.choices[0].finish_reason == "stop":
        print(f"✅ {max_tokens} is sufficient!")
        break
```

## Summary

**The key to preventing truncation:**

1. ✅ Use `max_completion_tokens` (not `max_tokens`)
2. ✅ Set it to **16,000+** for planning tasks
3. ✅ Monitor `finish_reason` in responses
4. ✅ Increase limit if you see `"length"`
5. ✅ Maximum supported: **128,000** tokens for GPT-5

**Quick Fix for Your Issue:**
```python
# In your ShiftMaster2000 settings
max_tokens = 32000  # Up from 4096

# Or dynamically adjust based on task
if "plan" in task_description.lower():
    max_tokens = 32000
else:
    max_tokens = 8000
```
