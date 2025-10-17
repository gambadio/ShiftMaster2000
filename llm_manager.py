"""
LLM Manager with Extended Thinking and Reasoning Support

Supports:
- Claude extended thinking with budget_tokens
- OpenAI reasoning_effort parameter
- Streaming with callbacks
- MCP integration
- Tool calling (solver integration)
"""

from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional, Callable, List
import requests
import json
import logging

from models import LLMConfig

logger = logging.getLogger(__name__)


async def call_llm_with_reasoning(
    prompt: str,
    config: LLMConfig,
    user_message: str = "Produce the schedule now.",
    on_chunk: Optional[Callable[[str], None]] = None,
    on_thinking: Optional[Callable[[str], None]] = None,
    enable_tools: bool = False,
) -> Dict[str, Any]:
    """
    Call LLM with reasoning support and streaming.

    Args:
        prompt: System prompt
        config: LLM configuration including reasoning parameters
        user_message: User message to send
        on_chunk: Callback for streaming content chunks
        on_thinking: Callback for thinking/reasoning chunks
        enable_tools: Whether to enable tool calling (solver integration)

    Returns:
        Dict with 'content', 'thinking', 'usage', 'model', and optionally 'tool_calls' keys
    """
    if config.model_family.lower() == "claude":
        return await _call_claude(prompt, config, user_message, on_chunk, on_thinking, enable_tools)
    elif config.model_family.lower() == "openai":
        return await _call_openai(prompt, config, user_message, on_chunk, on_thinking, enable_tools)
    else:
        # Custom/fallback
        return await _call_generic(prompt, config, user_message, on_chunk)


async def _call_claude(
    prompt: str,
    config: LLMConfig,
    user_message: str,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
    enable_tools: bool = False,
) -> Dict[str, Any]:
    """Call Claude with extended thinking and tool support"""
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

    # Add tools if enabled
    if enable_tools:
        tools = _get_available_tools()
        if tools:
            params["tools"] = tools

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

        # Handle tool calls
        if response.stop_reason == "tool_use":
            return await _handle_tool_calls_claude(client, params, response, on_chunk, on_thinking)

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
    enable_tools: bool = False,
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
    enable_tools: bool = False,
) -> Dict[str, Any]:
    """Synchronous wrapper for call_llm_with_reasoning"""
    return asyncio.run(call_llm_with_reasoning(prompt, config, user_message, enable_tools=enable_tools))


# === Tool Calling Support ===

def _get_available_tools() -> List[Dict[str, Any]]:
    """
    Returns list of tool definitions for LLM.

    Currently supports the MiniZinc constraint solver tool when enabled.
    """
    tools = []

    # Check if solver mode is enabled
    try:
        import streamlit as st
        if st.session_state.get("enable_solver", False):
            from solver_utils import check_minizinc_available
            from solver_models import SolverRequest

            is_available, _ = check_minizinc_available()
            if is_available:
                # Add solver tool
                tools.append({
                    "name": "solve_with_minizinc",
                    "description": "Generate an optimized shift schedule using constraint programming. "
                                   "This tool uses mathematical optimization to find provably optimal or near-optimal "
                                   "shift assignments that satisfy hard constraints and minimize soft constraint violations. "
                                   "Use this when you need to generate a schedule with complex fairness requirements, "
                                   "strict rotation constraints, or need mathematical guarantees about schedule quality.",
                    "input_schema": SolverRequest.model_json_schema()
                })
    except Exception as e:
        logger.warning(f"Failed to load solver tool: {e}")

    return tools


def _execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool call and return results.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool

    Returns:
        Tool execution result as dict
    """
    if tool_name == "solve_with_minizinc":
        try:
            from solver_service import solve_with_minizinc
            from solver_models import SolverRequest
            import streamlit as st

            # Inject weights from session state if available
            if "solver_weights" in st.session_state:
                if "rules" not in tool_input:
                    tool_input["rules"] = {}
                if "weights" not in tool_input["rules"]:
                    tool_input["rules"]["weights"] = st.session_state.solver_weights

            # Inject constraint rules from session state
            if "no_consecutive_late" in st.session_state:
                if "rules" not in tool_input:
                    tool_input["rules"] = {}
                tool_input["rules"]["no_consecutive_late"] = st.session_state.no_consecutive_late
            if "pikett_gap_days" in st.session_state:
                if "rules" not in tool_input:
                    tool_input["rules"] = {}
                tool_input["rules"]["pikett_gap_days"] = st.session_state.pikett_gap_days
            if "fr_dispatcher_per_week" in st.session_state:
                if "rules" not in tool_input:
                    tool_input["rules"] = {}
                tool_input["rules"]["fr_dispatcher_per_week"] = st.session_state.fr_dispatcher_per_week

            # Inject solver backend and timeout from session state
            if "options" not in tool_input:
                tool_input["options"] = {}
            if "solver_backend" in st.session_state:
                tool_input["options"]["solver"] = st.session_state.solver_backend
            if "solver_timeout" in st.session_state:
                tool_input["options"]["time_limit_ms"] = st.session_state.solver_timeout * 1000

            # Validate and execute
            request = SolverRequest.model_validate(tool_input)
            response = solve_with_minizinc(request)
            return response.model_dump()

        except Exception as e:
            logger.exception("Solver execution failed")
            return {
                "status": "ERROR",
                "message": f"Tool execution error: {str(e)}",
                "stats": {"solver": "none", "time_ms": 0}
            }

    raise ValueError(f"Unknown tool: {tool_name}")


async def _handle_tool_calls_claude(
    client,
    params: Dict[str, Any],
    response,
    on_chunk: Optional[Callable[[str], None]],
    on_thinking: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """
    Handle tool calls from Claude and continue conversation.

    This implements the tool use loop:
    1. LLM requests tool use
    2. Execute tool
    3. Send results back to LLM
    4. LLM continues with final answer
    """
    messages = params["messages"].copy()

    # Parse initial response
    initial_result = _parse_claude_response(response)

    # Extract tool uses from response
    tool_uses = []
    for block in response.content:
        if block.type == "tool_use":
            tool_uses.append({
                "id": block.id,
                "name": block.name,
                "input": block.input
            })

    if not tool_uses:
        return initial_result

    # Add assistant message with tool uses
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    # Execute each tool and collect results
    tool_results = []
    for tool_use in tool_uses:
        logger.info(f"Executing tool: {tool_use['name']}")

        try:
            result = _execute_tool(tool_use["name"], tool_use["input"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use["id"],
                "content": json.dumps(result, indent=2)
            })
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_use['name']}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use["id"],
                "content": f"Error: {str(e)}",
                "is_error": True
            })

    # Add tool results as user message
    messages.append({
        "role": "user",
        "content": tool_results
    })

    # Update params with extended conversation
    params["messages"] = messages

    # Call LLM again with tool results
    final_response = client.messages.create(**params)

    # Parse final response
    final_result = _parse_claude_response(final_response)

    # Merge thinking from both calls
    if initial_result.get("thinking") and final_result.get("thinking"):
        final_result["thinking"] = initial_result["thinking"] + "\n\n" + final_result["thinking"]
    elif initial_result.get("thinking"):
        final_result["thinking"] = initial_result["thinking"]

    # Add tool call metadata
    final_result["tool_calls"] = tool_uses
    final_result["tool_results"] = tool_results

    return final_result
