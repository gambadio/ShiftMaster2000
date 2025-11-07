"""
Unified LLM client supporting multiple providers (OpenAI, OpenRouter, Azure, Custom)

This module provides a consistent interface for interacting with various LLM providers,
handling provider-specific quirks, authentication, and response formats.

Based on research documented in knowledge.md
"""

from __future__ import annotations
import time
from typing import Optional, Dict, Any, Generator, List
from datetime import datetime

try:
    from openai import OpenAI, AzureOpenAI
    from openai import APIError, RateLimitError, APIConnectionError
except ImportError:
    raise ImportError(
        "OpenAI SDK is required. Install with: pip install openai"
    )

from models import (
    LLMConfig,
    LLMProviderConfig,
    ProviderType,
    ChatMessage,
    ChatSession
)


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Features:
    - Multi-provider support (OpenAI, OpenRouter, Azure, Custom)
    - Automatic retry with exponential backoff
    - Streaming support with callbacks
    - Model fetching and caching
    - Token usage tracking
    - Reasoning token display (for o1/o3 models)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with configuration.

        Args:
            config: LLMConfig instance with provider and generation settings
        """
        self.config = config
        self.provider_config = config.provider_config
        self.client = self._create_client()

    def _create_client(self):
        """Create the appropriate client based on provider type"""
        provider = self.provider_config.provider
        api_key = self.provider_config.api_key

        if provider == ProviderType.OPENAI:
            return OpenAI(api_key=api_key)

        elif provider == ProviderType.OPENROUTER:
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={
                    **({"HTTP-Referer": self.provider_config.http_referer}
                       if self.provider_config.http_referer else {}),
                    **({"X-Title": self.provider_config.x_title}
                       if self.provider_config.x_title else {})
                }
            )

        elif provider == ProviderType.AZURE:
            if not self.provider_config.azure_endpoint:
                raise ValueError("Azure endpoint is required for Azure provider")
            return AzureOpenAI(
                azure_endpoint=self.provider_config.azure_endpoint,
                api_key=api_key,
                api_version=self.provider_config.api_version
            )

        elif provider == ProviderType.CUSTOM:
            base_url = self.provider_config.get_base_url()
            if not base_url:
                raise ValueError("Base URL is required for custom provider")
            return OpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed"
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def fetch_models(self) -> List[str]:
        """
        Fetch available models from the provider.

        Returns:
            List of model IDs/names
        """
        try:
            models = self.client.models.list()
            model_ids = [model.id for model in models.data]

            # Cache the models in provider config
            self.provider_config.available_models = model_ids

            return model_ids

        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def _get_model_max_tokens(self, model_name: str) -> int:
        """
        Get maximum supported tokens for a model.
        Returns the model's max if known, otherwise returns a high default.
        """
        # Known model limits (can be expanded)
        model_limits = {
            # OpenAI models
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "o1": 100000,
            "o1-mini": 65536,
            "o3-mini": 200000,
            # Azure/OpenAI GPT-5 reasoning family (400k ctx, 272k input / 128k output per MS docs)
            "gpt-5-pro": 400000,
            "gpt-5-codex": 400000,
            "gpt-5": 400000,
            "gpt-5-mini": 400000,
            "gpt-5-nano": 400000,
            # Anthropic models
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3.5-sonnet": 200000,
            "claude-3.7-sonnet": 200000,
            # Google models
            "gemini-pro": 30720,
            "gemini-1.5-pro": 1000000,
            # Meta models
            "llama-3.1-405b": 128000,
        }

        # Check exact match or partial match
        for known_model, limit in model_limits.items():
            if known_model in model_name.lower():
                return limit

        # Default to high limit for unknown models
        return 128000

    def _build_completion_params(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build parameters for chat completion request"""

        # Determine model name (Azure uses deployment name)
        model_name = self.provider_config.model
        if self.provider_config.provider == ProviderType.AZURE:
            model_name = self.provider_config.azure_deployment or model_name

        # Check if this is a reasoning model (o1, o3, gpt-5 series)
        is_reasoning_model = any(x in model_name.lower() for x in ['o1', 'o3', 'gpt-5'])

        # Handle max_tokens with fallback
        requested_max_tokens = self.config.max_tokens
        model_max_tokens = self._get_model_max_tokens(model_name)
        actual_max_tokens = min(requested_max_tokens, model_max_tokens)

        # Log if we're falling back
        if actual_max_tokens < requested_max_tokens:
            print(f"Note: Requested max_tokens ({requested_max_tokens}) exceeds model limit ({model_max_tokens}). Using {actual_max_tokens}.")

        # Azure requires max_completion_tokens instead of max_tokens
        token_param_name = "max_completion_tokens" if self.provider_config.provider == ProviderType.AZURE else "max_tokens"

        params = {
            "model": model_name,
            "messages": messages,
            token_param_name: actual_max_tokens,
            "stream": stream,
        }

        # Reasoning models don't support temperature, top_p, frequency_penalty, presence_penalty
        if not is_reasoning_model:
            params["temperature"] = self.config.temperature
            params["top_p"] = self.config.top_p
            params["frequency_penalty"] = self.config.frequency_penalty
            params["presence_penalty"] = self.config.presence_penalty

        # Add optional parameters
        if self.config.seed is not None:
            params["seed"] = self.config.seed

        if self.config.stop_sequences:
            params["stop"] = self.config.stop_sequences

        # JSON mode (if supported by provider)
        if self.config.json_mode:
            supports_json = self.provider_config.provider in [
                ProviderType.OPENAI,
                ProviderType.AZURE,
                ProviderType.OPENROUTER
            ]
            if supports_json:
                params["response_format"] = {"type": "json_object"}

        # Reasoning effort for o1/o3 models
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort

        # Apply any parameter overrides
        if override_params:
            params.update(override_params)

        return params

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a completion request with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_retries: Maximum retry attempts for rate limits/server errors
            override_params: Optional dict to override config parameters

        Returns:
            Dict with 'content', 'usage', and optional 'reasoning_tokens'
        """
        params = self._build_completion_params(messages, stream=False, override_params=override_params)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)

                # Extract response data
                choice = response.choices[0]
                content = choice.message.content
                usage = response.usage

                # Extract reasoning tokens if available (o1/o3 models)
                reasoning_tokens = None
                if hasattr(usage, 'completion_tokens_details'):
                    details = usage.completion_tokens_details
                    if hasattr(details, 'reasoning_tokens'):
                        reasoning_tokens = details.reasoning_tokens

                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "reasoning_tokens": reasoning_tokens
                    },
                    "finish_reason": choice.finish_reason,
                    "model": response.model
                }

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                delay = 2 ** attempt
                print(f"Rate limited. Retrying in {delay}s...")
                time.sleep(delay)

            except APIConnectionError as e:
                if attempt == max_retries - 1:
                    raise
                delay = 2 ** attempt
                print(f"Connection error. Retrying in {delay}s...")
                time.sleep(delay)

            except APIError as e:
                if e.status_code >= 500 and attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"Server error {e.status_code}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    def stream_complete(
        self,
        messages: List[Dict[str, str]],
        on_token: Optional[callable] = None,
        on_complete: Optional[callable] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, str]:
        """
        Stream a completion response with optional callbacks.

        Args:
            messages: List of message dictionaries
            on_token: Optional callback for each token (receives token string)
            on_complete: Optional callback when complete (receives full response)
            override_params: Optional dict to override config parameters

        Yields:
            Token strings as they arrive

        Returns:
            Complete response text
        """
        params = self._build_completion_params(messages, stream=True, override_params=override_params)

        stream = self.client.chat.completions.create(**params)
        full_response = ""

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    if on_token:
                        on_token(delta)
                    yield delta

        if on_complete:
            on_complete(full_response)

        return full_response

    def chat(
        self,
        user_message: str,
        session: ChatSession,
        system_prompt: Optional[str] = None
    ) -> ChatMessage:
        """
        Send a chat message and update session state.

        Args:
            user_message: User's message content
            session: ChatSession to track conversation
            system_prompt: Optional system prompt (added if messages is empty)

        Returns:
            ChatMessage with assistant's response
        """
        # Add system prompt if this is the first message
        if not session.messages and system_prompt:
            session.messages.append(ChatMessage(
                role="system",
                content=system_prompt,
                timestamp=datetime.now().isoformat()
            ))

        # Add user message to session
        user_msg = ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat()
        )
        session.messages.append(user_msg)

        # Build messages for API
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages
        ]

        # Get completion
        result = self.complete(api_messages)

        # Create assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=result["content"],
            timestamp=datetime.now().isoformat(),
            reasoning_tokens=result["usage"].get("reasoning_tokens")
        )
        session.messages.append(assistant_msg)

        # Update session stats
        usage = result["usage"]
        session.total_prompt_tokens += usage["prompt_tokens"]
        session.total_completion_tokens += usage["completion_tokens"]
        if usage.get("reasoning_tokens"):
            session.total_reasoning_tokens += usage["reasoning_tokens"]

        return assistant_msg


def create_llm_client(config: LLMConfig) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        config: LLMConfig instance

    Returns:
        Configured LLMClient instance
    """
    return LLMClient(config)


def validate_provider_config(config: LLMProviderConfig) -> tuple[bool, Optional[str]]:
    """
    Validate provider configuration.

    Args:
        config: LLMProviderConfig to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    provider = config.provider

    # Check API key (required for all except some custom endpoints)
    if not config.api_key and provider != ProviderType.CUSTOM:
        return False, "API key is required"

    # Provider-specific validation
    if provider == ProviderType.AZURE:
        if not config.azure_endpoint:
            return False, "Azure endpoint is required"
        if not config.azure_deployment:
            return False, "Azure deployment name is required"

    elif provider == ProviderType.CUSTOM:
        if not config.base_url:
            return False, "Base URL is required for custom provider"

    return True, None
