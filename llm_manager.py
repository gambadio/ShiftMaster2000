"""
LLM Manager with Extended Thinking, Reasoning, and Tool Use Support

Supports:
- OpenAI (including o1/o3 reasoning models)
- OpenRouter (with reasoning parameter support)
- Azure OpenAI
- Claude (via Anthropic SDK with extended thinking)
- Generic OpenAI-compatible endpoints
- Tool/Function calling for MiniZinc integration (experimental)
"""

from __future__ import annotations
import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
import httpx

from models import LLMConfig, ProviderType

# Very long timeout for reasoning models that can think for extended periods
# The read timeout is especially important - reasoning models may not send data for minutes
# while "thinking", which would otherwise trigger a read timeout
LLM_TIMEOUT = httpx.Timeout(
    connect=60.0,    # 60 seconds to establish connection
    read=1800.0,     # 30 minutes to wait for data (reasoning models need this!)
    write=60.0,      # 60 seconds to send data
    pool=60.0        # 60 seconds to get connection from pool
)


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

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(api_key=config.provider_config.api_key, timeout=LLM_TIMEOUT)

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

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(
        base_url=config.provider_config.get_base_url(),
        api_key=config.provider_config.api_key,
        default_headers=headers if headers else None,
        timeout=LLM_TIMEOUT,
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

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = AzureOpenAI(
        api_key=config.provider_config.api_key,
        api_version=config.provider_config.api_version,
        azure_endpoint=config.provider_config.azure_endpoint,
        timeout=LLM_TIMEOUT,
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
    else:
        # For reasoning models, add reasoning_effort if specified
        # Azure GPT-5 supports: minimal, low, medium, high (default: medium)
        if config.reasoning_effort:
            print(f"[DEBUG] Adding reasoning_effort for Azure: {config.reasoning_effort}")
            params["reasoning_effort"] = config.reasoning_effort

    # Add stream_options to get usage in final chunk (for streaming)
    if config.enable_streaming:
        params["stream_options"] = {"include_usage": True}

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

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(
        base_url=config.provider_config.base_url,
        api_key=config.provider_config.api_key,
        timeout=LLM_TIMEOUT,
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
    
    print(f"[DEBUG] Starting stream for model: {model_name}")
    print(f"[DEBUG] Stream params: {list(params.keys())}")

    try:
        stream = client.chat.completions.create(**params)
        print("[DEBUG] Stream object created successfully")
        
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            
            # Debug first few chunks and every 50th chunk
            if chunk_count <= 3 or chunk_count % 50 == 0:
                print(f"[DEBUG] Chunk #{chunk_count}")
            
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Handle reasoning content (for o1/o3 models and OpenRouter)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    full_reasoning.append(delta.reasoning_content)
                    if on_thinking:
                        on_thinking(delta.reasoning_content)
                    if chunk_count <= 5:
                        print(f"[REASONING] content: {delta.reasoning_content[:100]}")

                # Handle reasoning field (OpenRouter format)
                if hasattr(delta, "reasoning") and delta.reasoning:
                    full_reasoning.append(delta.reasoning)
                    if on_thinking:
                        on_thinking(delta.reasoning)
                    if chunk_count <= 5:
                        print(f"[REASONING] field: {delta.reasoning[:100]}")

                # Handle regular content
                if hasattr(delta, "content") and delta.content:
                    full_content.append(delta.content)
                    if on_chunk:
                        on_chunk(delta.content)

            # Capture usage if available (Azure sends this in final chunk when stream_options is set)
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0),
                }
                # Capture reasoning tokens if available (Azure GPT-5 format)
                if hasattr(chunk.usage, "completion_tokens_details"):
                    details = chunk.usage.completion_tokens_details
                    if hasattr(details, "reasoning_tokens"):
                        usage_info["reasoning_tokens"] = details.reasoning_tokens
                        print(f"[DEBUG] Reasoning tokens: {details.reasoning_tokens}")

        print(f"[DEBUG] Stream complete: {chunk_count} chunks, {len(full_content)} content pieces, {len(full_reasoning)} reasoning pieces")
        
    except Exception as e:
        print(f"[ERROR] Stream error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
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


# =============================================================================
# Tool Calling Support (Experimental - MiniZinc Integration)
# =============================================================================

async def call_llm_with_tools(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable[[str, str], None]] = None,
    max_tool_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Call LLM with tool/function calling support (agentic loop).

    This enables the LLM to use tools like MiniZinc for constraint optimization
    during schedule generation.

    Args:
        prompt: System prompt
        config: LLM configuration
        user_message: User message to send
        on_chunk: Callback for streaming content chunks
        on_thinking: Callback for thinking/reasoning chunks
        on_tool_call: Callback for tool calls (tool_name, arguments_json)
        max_tool_iterations: Maximum tool call iterations to prevent infinite loops

    Returns:
        Dict with 'content', 'thinking', 'usage', 'model', and 'tool_calls' keys
    """
    # Check if any tools are enabled
    if not config.enable_minizinc_tool and not config.enable_query_tool:
        # No tools enabled, use regular call
        return await call_llm_with_reasoning(prompt, config, user_message, on_chunk, on_thinking)

    # Build tools list dynamically
    tools = []
    tool_handlers = {}

    # Add MiniZinc tool if enabled
    if config.enable_minizinc_tool:
        try:
            from minizinc_tool import MINIZINC_TOOL_SCHEMA, process_tool_call as minizinc_process
            tools.append(MINIZINC_TOOL_SCHEMA)
            # Wrap handler to pass configured solver
            solver_name = config.minizinc_solver
            tool_handlers["run_minizinc"] = lambda tc, s=solver_name: minizinc_process(tc, s)
            print(f"[TOOLS] MiniZinc tool enabled with solver: {solver_name or 'auto'}")
        except ImportError:
            print("[WARN] minizinc_tool module not available")

    # Add Query tool if enabled
    if config.enable_query_tool:
        try:
            from query_tool import QUERY_TOOL_SCHEMA, process_tool_call as query_handler
            tools.append(QUERY_TOOL_SCHEMA)
            tool_handlers["query_schedule_data"] = query_handler
            print("[TOOLS] Query tool enabled")
        except ImportError:
            print("[WARN] query_tool module not available")

    # If no tools could be loaded, fall back to regular call
    if not tools:
        print("[WARN] No tools available, falling back to regular call")
        return await call_llm_with_reasoning(prompt, config, user_message, on_chunk, on_thinking)

    # Create unified tool processor
    def process_any_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        function = tool_call.get("function", {})
        function_name = function.get("name", "")
        handler = tool_handlers.get(function_name)
        if handler:
            return handler(tool_call)
        return {"success": False, "error": f"Unknown tool: {function_name}"}

    print(f"[TOOLS] {len(tools)} tool(s) available: {list(tool_handlers.keys())}")

    # Route to appropriate provider with tools
    provider = config.provider_config.provider

    if provider == ProviderType.OPENAI:
        return await _call_with_tools_openai(
            prompt, config, user_message, tools, process_any_tool_call,
            on_chunk, on_thinking, on_tool_call, max_tool_iterations
        )
    elif provider == ProviderType.AZURE:
        return await _call_with_tools_azure(
            prompt, config, user_message, tools, process_any_tool_call,
            on_chunk, on_thinking, on_tool_call, max_tool_iterations
        )
    elif provider == ProviderType.OPENROUTER:
        return await _call_with_tools_openrouter(
            prompt, config, user_message, tools, process_any_tool_call,
            on_chunk, on_thinking, on_tool_call, max_tool_iterations
        )
    else:
        # Generic/Custom - try OpenAI-style tool calling
        return await _call_with_tools_generic(
            prompt, config, user_message, tools, process_any_tool_call,
            on_chunk, on_thinking, on_tool_call, max_tool_iterations
        )


async def _call_with_tools_openai(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
    on_tool_call: Optional[Callable[[str, str], None]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Call OpenAI API with tool calling support"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(api_key=config.provider_config.api_key, timeout=LLM_TIMEOUT)

    # Initialize conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    all_tool_calls = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    final_content = ""
    final_thinking = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[TOOLS] Iteration {iteration}/{max_iterations}")

        # Force tool use on first iteration, then let LLM decide
        # This ensures MiniZinc is actually called when enabled
        tool_choice = "required" if iteration == 1 else "auto"

        # Build request parameters
        params: Dict[str, Any] = {
            "model": config.provider_config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        # Add reasoning_effort for o1/o3 models
        if config.reasoning_effort:
            params["reasoning_effort"] = config.reasoning_effort

        # Note: JSON mode cannot be used with tool calling
        # We'll get JSON in the final response

        # Make the call (non-streaming for tool calls for simplicity)
        response = client.chat.completions.create(**params)

        # Update usage
        if response.usage:
            total_usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0)
            total_usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0)

        message = response.choices[0].message

        # Check for tool calls
        if message.tool_calls:
            print(f"[TOOLS] LLM requested {len(message.tool_calls)} tool call(s)")

            # Add assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"[TOOLS] Executing: {tool_name}")

                # Notify callback
                if on_tool_call:
                    on_tool_call(tool_name, tool_args)

                # Execute the tool
                tool_result = tool_processor({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                })

                # Record the tool call
                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

                print(f"[TOOLS] {tool_name} result: success={tool_result.get('success', False)}")

        else:
            # No tool calls - this is the final response
            final_content = message.content or ""

            # Extract reasoning if present
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                final_thinking = message.reasoning_content

            # Stream the final content if callback provided
            if on_chunk and final_content:
                on_chunk(final_content)

            break

    return {
        "content": final_content,
        "thinking": final_thinking,
        "usage": total_usage,
        "model": config.provider_config.model,
        "tool_calls": all_tool_calls,
    }


async def _call_with_tools_azure(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
    on_tool_call: Optional[Callable[[str, str], None]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Call Azure OpenAI API with tool calling support"""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = AzureOpenAI(
        api_key=config.provider_config.api_key,
        api_version=config.provider_config.api_version,
        azure_endpoint=config.provider_config.azure_endpoint,
        timeout=LLM_TIMEOUT,
    )

    # Check if this is a reasoning model
    model_name = config.provider_config.model.lower()
    is_reasoning_model = any(x in model_name for x in ['o1', 'o3', 'gpt-5'])

    # Initialize conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    all_tool_calls = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    final_content = ""
    final_thinking = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[TOOLS-AZURE] Iteration {iteration}/{max_iterations}")

        # Force tool use on first iteration, then let LLM decide
        # This ensures MiniZinc is actually called when enabled
        tool_choice = "required" if iteration == 1 else "auto"

        # Build request parameters
        params: Dict[str, Any] = {
            "model": config.provider_config.azure_deployment,
            "messages": messages,
            "max_completion_tokens": config.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        # Only add temperature etc. for non-reasoning models
        if not is_reasoning_model:
            params["temperature"] = config.temperature
            params["top_p"] = config.top_p
        else:
            if config.reasoning_effort:
                params["reasoning_effort"] = config.reasoning_effort

        # Make the call
        response = client.chat.completions.create(**params)

        # Update usage
        if response.usage:
            total_usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0)
            total_usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0)

        message = response.choices[0].message

        # Check for tool calls
        if message.tool_calls:
            print(f"[TOOLS-AZURE] LLM requested {len(message.tool_calls)} tool call(s)")

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"[TOOLS-AZURE] Executing: {tool_name}")

                if on_tool_call:
                    on_tool_call(tool_name, tool_args)

                tool_result = tool_processor({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                })

                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

                print(f"[TOOLS-AZURE] {tool_name} result: success={tool_result.get('success', False)}")

        else:
            final_content = message.content or ""

            if hasattr(message, "reasoning_content") and message.reasoning_content:
                final_thinking = message.reasoning_content

            if on_chunk and final_content:
                on_chunk(final_content)

            break

    return {
        "content": final_content,
        "thinking": final_thinking,
        "usage": total_usage,
        "model": config.provider_config.azure_deployment,
        "tool_calls": all_tool_calls,
    }


async def _call_with_tools_openrouter(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
    on_tool_call: Optional[Callable[[str, str], None]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Call OpenRouter API with tool calling support"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    # Debug: Print API key info (masked for security)
    api_key = config.provider_config.api_key
    base_url = config.provider_config.get_base_url()
    model = config.provider_config.model
    if api_key:
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "***"
        print(f"[DEBUG-OPENROUTER] API Key: {masked_key} (len={len(api_key)})")
    else:
        print(f"[DEBUG-OPENROUTER] API Key: NONE/EMPTY")
    print(f"[DEBUG-OPENROUTER] Base URL: {base_url}")
    print(f"[DEBUG-OPENROUTER] Model: {model}")

    # Quick test: Try a simple API call first to verify auth works
    import requests
    print(f"[DEBUG-OPENROUTER] Testing auth with direct request...")
    test_resp = requests.get(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    print(f"[DEBUG-OPENROUTER] Auth test response: {test_resp.status_code} - {test_resp.text[:200] if test_resp.text else 'empty'}")

    # Test simple chat completion WITHOUT tools
    print(f"[DEBUG-OPENROUTER] Testing simple chat completion (no tools)...")
    simple_test = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10
        }
    )
    print(f"[DEBUG-OPENROUTER] Simple chat test: {simple_test.status_code} - {simple_test.text[:300] if simple_test.text else 'empty'}")

    # Build headers
    headers = {}
    if config.provider_config.http_referer:
        headers["HTTP-Referer"] = config.provider_config.http_referer
    if config.provider_config.x_title:
        headers["X-Title"] = config.provider_config.x_title

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=headers if headers else None,
        timeout=LLM_TIMEOUT,
    )

    # Initialize conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    all_tool_calls = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    final_content = ""
    final_thinking = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[TOOLS-OPENROUTER] Iteration {iteration}/{max_iterations}")

        # Force tool use on first iteration, then let LLM decide
        # This ensures MiniZinc is actually called when enabled
        tool_choice = "required" if iteration == 1 else "auto"

        # Build request parameters
        params: Dict[str, Any] = {
            "model": config.provider_config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        # Add OpenRouter reasoning via extra_body
        extra_body = {}
        reasoning_config = {}
        if config.reasoning_effort:
            reasoning_config["effort"] = config.reasoning_effort
        if config.reasoning_max_tokens:
            reasoning_config["max_tokens"] = config.reasoning_max_tokens
        if reasoning_config:
            extra_body["reasoning"] = reasoning_config
        if extra_body:
            params["extra_body"] = extra_body

        # Make the call
        response = client.chat.completions.create(**params)

        # Update usage
        if response.usage:
            total_usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0)
            total_usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0)

        message = response.choices[0].message

        # Check for tool calls
        if message.tool_calls:
            print(f"[TOOLS-OPENROUTER] LLM requested {len(message.tool_calls)} tool call(s)")

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"[TOOLS-OPENROUTER] Executing: {tool_name}")

                if on_tool_call:
                    on_tool_call(tool_name, tool_args)

                tool_result = tool_processor({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                })

                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

                print(f"[TOOLS-OPENROUTER] {tool_name} result: success={tool_result.get('success', False)}")

        else:
            final_content = message.content or ""

            if hasattr(message, "reasoning") and message.reasoning:
                final_thinking = message.reasoning
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                final_thinking = message.reasoning_content

            if on_chunk and final_content:
                on_chunk(final_content)

            break

    return {
        "content": final_content,
        "thinking": final_thinking,
        "usage": total_usage,
        "model": config.provider_config.model,
        "tool_calls": all_tool_calls,
    }


async def _call_with_tools_generic(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    tools: List[Dict[str, Any]],
    tool_processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
    on_tool_call: Optional[Callable[[str, str], None]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Call generic OpenAI-compatible API with tool calling support"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai>=1.0.0")

    # Use very long timeout (30 min) for reasoning models that think for extended periods
    client = OpenAI(
        base_url=config.provider_config.base_url,
        api_key=config.provider_config.api_key,
        timeout=LLM_TIMEOUT,
    )

    # Initialize conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message}
    ]

    all_tool_calls = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    final_content = ""
    final_thinking = None
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[TOOLS-GENERIC] Iteration {iteration}/{max_iterations}")

        # Force tool use on first iteration, then let LLM decide
        # This ensures MiniZinc is actually called when enabled
        tool_choice = "required" if iteration == 1 else "auto"

        params: Dict[str, Any] = {
            "model": config.provider_config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        try:
            response = client.chat.completions.create(**params)
        except Exception as e:
            # Some endpoints don't support tools - fall back to regular call
            print(f"[TOOLS-GENERIC] Tool calling not supported: {e}")
            print("[TOOLS-GENERIC] Falling back to regular call without tools")
            return await _call_generic(prompt, config, user_message, on_chunk, on_thinking)

        if response.usage:
            total_usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0)
            total_usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0)

        message = response.choices[0].message

        if message.tool_calls:
            print(f"[TOOLS-GENERIC] LLM requested {len(message.tool_calls)} tool call(s)")

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                print(f"[TOOLS-GENERIC] Executing: {tool_name}")

                if on_tool_call:
                    on_tool_call(tool_name, tool_args)

                tool_result = tool_processor({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                })

                all_tool_calls.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })

                print(f"[TOOLS-GENERIC] {tool_name} result: success={tool_result.get('success', False)}")

        else:
            final_content = message.content or ""

            if on_chunk and final_content:
                on_chunk(final_content)

            break

    return {
        "content": final_content,
        "thinking": final_thinking,
        "usage": total_usage,
        "model": config.provider_config.model,
        "tool_calls": all_tool_calls,
    }


def call_llm_with_tools_sync(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    on_tool_call: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper for call_llm_with_tools"""
    return asyncio.run(call_llm_with_tools(
        prompt, config, user_message,
        on_tool_call=on_tool_call
    ))
