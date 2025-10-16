"""
LLM Manager with Extended Thinking and Reasoning Support

Supports:
- Claude extended thinking with budget_tokens
- OpenAI reasoning_effort parameter
- Streaming with callbacks
- MCP integration
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional, Callable, List
import requests
import json

from models import LLMConfig


async def call_llm_with_reasoning(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Call LLM with reasoning support and streaming.

    Args:
        prompt: System prompt
        config: LLM configuration including reasoning parameters
        user_message: User message to send
        on_chunk: Callback for streaming content chunks
        on_thinking: Callback for thinking/reasoning chunks

    Returns:
        Dict with 'content', 'thinking', 'usage', and 'model' keys
    """
    if config.model_family.lower() == "claude":
        return await _call_claude(prompt, config, user_message, on_chunk, on_thinking)
    elif config.model_family.lower() == "openai":
        return await _call_openai(prompt, config, user_message, on_chunk, on_thinking)
    else:
        # Custom/fallback
        return await _call_generic(prompt, config, user_message, on_chunk)


async def _call_claude(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Call Claude with extended thinking support"""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic>=0.40.0")

    client = anthropic.Anthropic()

    # Build messages
    messages = [
        {"role": "user", "content": user_message}
    ]

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.model_name,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "system": prompt,
        "messages": messages,
    }

    # Add extended thinking if configured
    extra_headers = {}
    if config.budget_tokens and config.budget_tokens >= 1024:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.budget_tokens
        }

    # Add interleaved thinking beta header if enabled
    if config.enable_interleaved_thinking:
        extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

    if extra_headers:
        params["extra_headers"] = extra_headers

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_claude(client, params, on_chunk, on_thinking)
    else:
        response = client.messages.create(**params)
        return _parse_claude_response(response)


async def _stream_claude(
    client,
    params: Dict[str, Any],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Stream Claude response"""
    full_content = []
    full_thinking = []
    usage_info = {}

    with client.messages.stream(**params) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if hasattr(event, "content_block") and hasattr(event.content_block, "type"):
                    if event.content_block.type == "thinking":
                        # Thinking block started
                        pass
            elif event.type == "content_block_delta":
                if hasattr(event, "delta"):
                    if event.delta.type == "text_delta":
                        text = event.delta.text
                        full_content.append(text)
                        if on_chunk:
                            on_chunk(text)
                    elif event.delta.type == "thinking_delta":
                        thinking_text = event.delta.thinking
                        full_thinking.append(thinking_text)
                        if on_thinking:
                            on_thinking(thinking_text)
            elif event.type == "message_stop":
                # Stream complete
                pass

    # Get final message
    final_message = stream.get_final_message()

    return {
        "content": "".join(full_content),
        "thinking": "".join(full_thinking) if full_thinking else None,
        "usage": {
            "input_tokens": final_message.usage.input_tokens if hasattr(final_message, "usage") else 0,
            "output_tokens": final_message.usage.output_tokens if hasattr(final_message, "usage") else 0,
        },
        "model": final_message.model,
    }


def _parse_claude_response(response) -> Dict[str, Any]:
    """Parse non-streaming Claude response"""
    content_parts = []
    thinking_parts = []

    for block in response.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)

    return {
        "content": "".join(content_parts),
        "thinking": "".join(thinking_parts) if thinking_parts else None,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "model": response.model,
    }


async def _call_openai(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Call OpenAI with reasoning_effort support"""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.105.0")

    client = openai.OpenAI()

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    # Add reasoning_effort for o1/o3/GPT-5 models
    if config.reasoning_effort:
        params["reasoning_effort"] = config.reasoning_effort

    # Add JSON mode if requested
    if config.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_openai(client, params, on_chunk, on_thinking)
    else:
        response = client.chat.completions.create(**params)
        return _parse_openai_response(response)


async def _stream_openai(
    client,
    params: Dict[str, Any],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Stream OpenAI response"""
    params["stream"] = True
    full_content = []
    full_reasoning = []
    usage_info = {}
    model_name = params["model"]

    stream = client.chat.completions.create(**params)

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta

            # Handle reasoning content (for o1/o3 models)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                full_reasoning.append(delta.reasoning_content)
                if on_thinking:
                    on_thinking(delta.reasoning_content)

            # Handle regular content
            if hasattr(delta, "content") and delta.content:
                full_content.append(delta.content)
                if on_chunk:
                    on_chunk(delta.content)

        # Capture usage if available
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = {
                "input_tokens": chunk.usage.prompt_tokens,
                "output_tokens": chunk.usage.completion_tokens,
            }

    return {
        "content": "".join(full_content),
        "thinking": "".join(full_reasoning) if full_reasoning else None,
        "usage": usage_info,
        "model": model_name,
    }


def _parse_openai_response(response) -> Dict[str, Any]:
    """Parse non-streaming OpenAI response"""
    message = response.choices[0].message

    return {
        "content": message.content or "",
        "thinking": message.reasoning_content if hasattr(message, "reasoning_content") else None,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        },
        "model": response.model,
    }


async def _call_generic(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Generic OpenAI-compatible API call (fallback)"""
    # This is a synchronous fallback using requests
    # In a real async context, you'd use httpx

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    if on_chunk:
        on_chunk(content)

    return {
        "content": content,
        "thinking": None,
        "usage": data.get("usage", {}),
        "model": data.get("model", config.model_name),
    }


def call_llm_sync(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
) -> Dict[str, Any]:
    """Synchronous wrapper for call_llm_with_reasoning"""
    return asyncio.run(call_llm_with_reasoning(prompt, config, user_message))
