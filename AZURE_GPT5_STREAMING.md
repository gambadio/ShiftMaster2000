# Azure GPT-5 Streaming and Reasoning

## Key Findings from Official Documentation

### ✅ What Works
1. **Streaming is supported** for Azure GPT-5 models
2. **`reasoning_effort` parameter is supported** with values: `minimal`, `low`, `medium`, `high`
3. **`max_completion_tokens` is the correct parameter** (not `max_tokens`)
4. **`stream_options: { include_usage: true }`** makes Azure return usage stats in final chunk

### 🔍 How Azure GPT-5 Reasoning Works

According to [Azure OpenAI Reasoning Models Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning):

**Azure reasoning models do NOT stream the thinking process in real-time**. Instead:
- The model **thinks internally** and generates hidden `reasoning_tokens`
- These reasoning tokens are **not returned in the response**
- They are **only visible in the usage statistics** at the end
- The `completion_tokens_details.reasoning_tokens` field shows how many tokens were used for thinking

### ❌ What Azure Does NOT Support

From the docs under "Not Supported" section:
- `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` (for reasoning models)
- `logprobs`, `top_logprobs`, `logit_bias`, `max_tokens`

### 📊 Usage Statistics Structure

For Azure GPT-5, the usage object looks like:
```json
{
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 1843,
    "total_tokens": 1863,
    "completion_tokens_details": {
      "reasoning_tokens": 448  // Hidden thinking tokens
    }
  }
}
```

### 🆚 Comparison: Azure vs OpenRouter

| Feature | Azure GPT-5 | OpenRouter GPT-5 |
|---------|-------------|------------------|
| Streams content | ✅ Yes | ✅ Yes |
| Streams thinking | ❌ No (only usage stats) | ✅ Yes (via `reasoning` field) |
| `reasoning_effort` | ✅ Yes (minimal/low/medium/high) | ✅ Yes (via `extra_body`) |
| Thinking visibility | Only token count | Full thinking text |

### 🔧 Correct Azure Implementation

```python
# Correct parameters for Azure GPT-5
params = {
    "model": azure_deployment_name,  # NOT the model name
    "messages": messages,
    "max_completion_tokens": 128000,  # NOT max_tokens (max 128k for output)
    "reasoning_effort": "high",  # Optional: minimal/low/medium/high
    "stream": True,
    "stream_options": {"include_usage": True}  # Get usage in final chunk
}

# Temperature, top_p etc. should NOT be included for reasoning models
```

### ⚠️ **CRITICAL: Preventing Truncated Responses**

**Problem**: If `max_completion_tokens` is too low (e.g., 4096), the model will truncate long responses and return `finish_reason: "length"` instead of `finish_reason: "stop"`.

**Solution**: 
- For **planning/generation tasks**: Use at least **16,000-32,000** tokens
- For **complex multi-step tasks**: Use **64,000-100,000** tokens  
- **Maximum supported**: 128,000 output tokens (GPT-5 limit)

**How to check**:
```python
response = client.chat.completions.create(...)
if response.choices[0].finish_reason == "length":
    print("⚠️ Response was truncated! Increase max_completion_tokens")
elif response.choices[0].finish_reason == "stop":
    print("✅ Complete response generated")
```

### 📝 Getting Reasoning Summaries

Azure offers a **Responses API** (not Chat Completions API) to get reasoning summaries:

```python
response = client.responses.create(
    model="gpt-5",
    input="Your question here",
    reasoning={
        "effort": "medium",
        "summary": "auto"  # or "detailed" (gpt-5 doesn't support "concise")
    }
)
```

This returns a summary of the model's thinking, but **not the full step-by-step reasoning**.

### ⚠️ Important Note

From the Azure docs:

> "Attempting to extract raw reasoning through methods other than the reasoning summary parameter are not supported, may violate the Acceptable Use Policy, and may result in throttling or suspension when detected."

**This means Azure deliberately hides the full reasoning process** for reasoning models.

### 🎯 Conclusion

For the ShiftMaster2000 app:
- ✅ Azure GPT-5 **can stream the final answer** in real-time
- ❌ Azure GPT-5 **cannot stream the thinking process** like OpenRouter does
- ✅ We can show "Model is thinking..." with a spinner
- ✅ We can show reasoning token count at the end
- ❌ We cannot show live thinking tokens like OpenRouter

**Recommendation**: If you want to see the actual thinking process in real-time, use OpenRouter. If you're okay with just seeing the final answer streaming (with thinking happening invisibly), Azure works fine.
