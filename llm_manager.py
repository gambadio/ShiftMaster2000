"""
LLM Manager with Extended Thinking and Reasoning Support

Supports:
- OpenAI (including o1/o3 reasoning models)
- OpenRouter (with reasoning parameter support)
- Azure OpenAI
- Claude (via Anthropic SDK with extended thinking)
- Generic OpenAI-compatible endpoints
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional, Callable

from models import LLMConfig, ProviderType


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
    provider = config.provider_config.provider

    # Route to appropriate provider implementation
    if provider == ProviderType.OPENAI:
        return await _call_openai(prompt, config, user_message, on_chunk, on_thinking)
    elif provider == ProviderType.OPENROUTER:
        return await _call_openrouter(prompt, config, user_message, on_chunk, on_thinking)
    elif provider == ProviderType.AZURE:
        return await _call_azure(prompt, config, user_message, on_chunk, on_thinking)
    else:
        # Custom/generic OpenAI-compatible endpoint
        return await _call_generic(prompt, config, user_message, on_chunk, on_thinking)


async def _call_openai(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Call OpenAI API with reasoning_effort support for o1/o3 models"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    client = OpenAI(api_key=config.provider_config.api_key)

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.provider_config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
    }

    # Add reasoning_effort for o1/o3 models
    if config.reasoning_effort:
        params["reasoning_effort"] = config.reasoning_effort

    # Add JSON mode if requested
    if config.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Add seed for reproducibility
    if config.seed is not None:
        params["seed"] = config.seed

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_openai_style(client, params, on_chunk, on_thinking)
    else:
        response = client.chat.completions.create(**params)
        return _parse_openai_response(response)


async def _call_openrouter(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Call OpenRouter API with reasoning parameter support via extra_body"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    # Build headers
    headers = {}
    if config.provider_config.http_referer:
        headers["HTTP-Referer"] = config.provider_config.http_referer
    if config.provider_config.x_title:
        headers["X-Title"] = config.provider_config.x_title

    client = OpenAI(
        base_url=config.provider_config.get_base_url(),
        api_key=config.provider_config.api_key,
        default_headers=headers if headers else None,
    )

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.provider_config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty,
    }

    # Build extra_body for OpenRouter-specific parameters
    extra_body = {}

    # Add OpenRouter reasoning parameter via extra_body
    reasoning_config = {}
    if config.reasoning_effort:
        reasoning_config["effort"] = config.reasoning_effort
    if config.reasoning_max_tokens:
        reasoning_config["max_tokens"] = config.reasoning_max_tokens
    if config.reasoning_exclude:
        reasoning_config["exclude"] = True

    if reasoning_config:
        extra_body["reasoning"] = reasoning_config

    # Add extra_body to params if not empty
    if extra_body:
        params["extra_body"] = extra_body

    # Add JSON mode if requested
    if config.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Add seed for reproducibility
    if config.seed is not None:
        params["seed"] = config.seed

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_openai_style(client, params, on_chunk, on_thinking)
    else:
        response = client.chat.completions.create(**params)
        return _parse_openai_response(response)


async def _call_azure(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Call Azure OpenAI API"""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    client = AzureOpenAI(
        api_key=config.provider_config.api_key,
        api_version=config.provider_config.api_version,
        azure_endpoint=config.provider_config.azure_endpoint,
    )

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    # Check if this is a reasoning model (o1, o3, gpt-5 series)
    model_name = config.provider_config.model.lower()
    is_reasoning_model = any(x in model_name for x in ['o1', 'o3', 'gpt-5'])

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.provider_config.azure_deployment,  # Azure uses deployment name
        "messages": messages,
        "max_completion_tokens": config.max_tokens,  # Azure requires max_completion_tokens
    }

    # Only add these parameters for non-reasoning models
    if not is_reasoning_model:
        params["temperature"] = config.temperature
        params["top_p"] = config.top_p
        params["frequency_penalty"] = config.frequency_penalty
        params["presence_penalty"] = config.presence_penalty

    # Add JSON mode if requested
    if config.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Add seed for reproducibility
    if config.seed is not None:
        params["seed"] = config.seed

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_openai_style(client, params, on_chunk, on_thinking)
    else:
        response = client.chat.completions.create(**params)
        return _parse_openai_response(response)


async def _call_generic(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Generic OpenAI-compatible API call"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    client = OpenAI(
        base_url=config.provider_config.base_url,
        api_key=config.provider_config.api_key,
    )

    # Build messages
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    # Build request parameters
    params: Dict[str, Any] = {
        "model": config.provider_config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    # Add JSON mode if requested
    if config.json_mode:
        params["response_format"] = {"type": "json_object"}

    # Stream or non-stream
    if config.enable_streaming:
        return await _stream_openai_style(client, params, on_chunk, on_thinking)
    else:
        response = client.chat.completions.create(**params)
        return _parse_openai_response(response)


async def _stream_openai_style(
    client,
    params: Dict[str, Any],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Stream OpenAI-style response"""
    params["stream"] = True
    full_content = []
    full_reasoning = []
    usage_info = {}
    model_name = params.get("model", "unknown")

    stream = client.chat.completions.create(**params)

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta

            # Handle reasoning content (for o1/o3 models and OpenRouter)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                full_reasoning.append(delta.reasoning_content)
                if on_thinking:
                    on_thinking(delta.reasoning_content)

            # Handle reasoning field (OpenRouter format)
            if hasattr(delta, "reasoning") and delta.reasoning:
                full_reasoning.append(delta.reasoning)
                if on_thinking:
                    on_thinking(delta.reasoning)

            # Handle regular content
            if hasattr(delta, "content") and delta.content:
                full_content.append(delta.content)
                if on_chunk:
                    on_chunk(delta.content)

        # Capture usage if available
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = {
                "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
            }

    return {
        "content": "".join(full_content),
        "thinking": "".join(full_reasoning) if full_reasoning else None,
        "usage": usage_info,
        "model": model_name,
    }


def _parse_openai_response(response) -> Dict[str, Any]:
    """Parse non-streaming OpenAI-style response"""
    message = response.choices[0].message

    # Extract reasoning/thinking from various possible fields
    thinking = None
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        thinking = message.reasoning_content
    elif hasattr(message, "reasoning") and message.reasoning:
        thinking = message.reasoning
    elif hasattr(message, "reasoning_details") and message.reasoning_details:
        # OpenRouter format with reasoning_details
        thinking = str(message.reasoning_details)

    return {
        "content": message.content or "",
        "thinking": thinking,
        "usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        },
        "model": response.model,
    }


def call_llm_sync(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
) -> Dict[str, Any]:
    """Synchronous wrapper for call_llm_with_reasoning"""
    return asyncio.run(call_llm_with_reasoning(prompt, config, user_message))
