"""
# @file purpose: Defines ChatLiteLLM for universal LLM provider access

This file implements the ChatLiteLLM class for browser-use, providing access to
virtually any LLM through LiteLLM's unified interface. LiteLLM supports 100+ 
providers including OpenAI, Anthropic, Google, Cohere, and many others through
a single, consistent API.

LiteLLM acts as a universal adapter, allowing seamless switching between different
LLM providers without changing the browser-use integration code.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer
from browser_use.llm.messages import BaseMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatLiteLLM(BaseChatModel):
	"""
	A wrapper around LiteLLM that implements the BaseChatModel protocol.
	
	LiteLLM provides a unified interface to 100+ LLM providers including:
	- OpenAI (GPT-3.5, GPT-4, GPT-4o, etc.)
	- Anthropic (Claude 3, Claude 3.5, etc.)
	- Google (Gemini, PaLM, etc.)
	- Cohere (Command, Command-R, etc.)
	- Meta (Llama 2, Llama 3, etc.)
	- Mistral AI, DeepSeek, Groq, and many more
	
	This allows browser-use to work with virtually any LLM provider
	through a single, consistent interface.
	
	Args:
		model: The model name with provider prefix (e.g., "openai/gpt-4", "anthropic/claude-3-5-sonnet-20241022", "google/gemini-pro")
		temperature: Temperature for response generation (0.0 to 1.0)
		max_tokens: Maximum tokens in the response
		api_key: API key for the provider (optional, can use environment variables)
		api_base: Custom API base URL (optional)
		custom_llm_provider: Explicit provider name (optional, usually auto-detected)
		**kwargs: Additional parameters passed to LiteLLM
	"""

	# Model configuration
	model: str

	# Model parameters
	temperature: float | None = None
	max_tokens: int | None = None
	top_p: float | None = None
	frequency_penalty: float | None = None
	presence_penalty: float | None = None
	stop: list[str] | None = None

	# Provider configuration
	api_key: str | None = None
	api_base: str | None = None
	custom_llm_provider: str | None = None

	# Additional LiteLLM parameters
	litellm_params: Dict[str, Any] | None = None

	@property
	def provider(self) -> str:
		return 'litellm'

	@property
	def name(self) -> str:
		return self.model

	def _get_litellm_client(self):
		"""Get the LiteLLM completion function."""
		try:
			import litellm
			return litellm.acompletion
		except ImportError:
			raise ImportError(
				'LiteLLM is not installed. Please install it using: '
				'pip install litellm'
			)

	def _prepare_completion_params(self) -> Dict[str, Any]:
		"""Prepare parameters for LiteLLM completion call."""
		params: Dict[str, Any] = {
			'model': self.model,
		}

		# Add optional parameters if they are set
		if self.temperature is not None:
			params['temperature'] = self.temperature
		if self.max_tokens is not None:
			params['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			params['top_p'] = self.top_p
		if self.frequency_penalty is not None:
			params['frequency_penalty'] = self.frequency_penalty
		if self.presence_penalty is not None:
			params['presence_penalty'] = self.presence_penalty
		if self.stop is not None:
			params['stop'] = self.stop
		if self.api_key is not None:
			params['api_key'] = self.api_key
		if self.api_base is not None:
			params['api_base'] = self.api_base
		if self.custom_llm_provider is not None:
			params['custom_llm_provider'] = self.custom_llm_provider

		# Add any additional LiteLLM parameters
		if self.litellm_params:
			params.update(self.litellm_params)

		return params

	def _get_usage(self, response: Any) -> ChatInvokeUsage | None:
		"""Extract usage information from LiteLLM response."""
		if hasattr(response, 'usage') and response.usage:
			usage = response.usage
			return ChatInvokeUsage(
				prompt_tokens=getattr(usage, 'prompt_tokens', 0),
				completion_tokens=getattr(usage, 'completion_tokens', 0),
				total_tokens=getattr(usage, 'total_tokens', 0),
				prompt_cached_tokens=getattr(usage, 'prompt_cached_tokens', None),
				prompt_cache_creation_tokens=getattr(usage, 'prompt_cache_creation_tokens', None),
				prompt_image_tokens=getattr(usage, 'prompt_image_tokens', None),
			)
		return None

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the LiteLLM model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""
		try:
			import litellm
		except ImportError:
			raise ImportError(
				'LiteLLM is not installed. Please install it using: '
				'pip install litellm'
			)

		# Serialize messages to LiteLLM format
		litellm_messages = LiteLLMMessageSerializer.serialize_messages(messages)

		# Prepare completion parameters
		params = self._prepare_completion_params()
		params['messages'] = litellm_messages

		try:
			if output_format is None:
				# Regular completion
				response = await litellm.acompletion(**params)
				
				usage = self._get_usage(response)
				content = response.choices[0].message.content or ''  # type: ignore
				
				return ChatInvokeCompletion(
					completion=content,
					usage=usage,
				)
			else:
				# Structured output using function calling
				schema = SchemaOptimizer.create_optimized_json_schema(output_format)
				tool_name = output_format.__name__
				
				# Remove title from schema as it's not needed for function calling
				schema.pop('title', None)
				
				tools = [{
					'type': 'function',
					'function': {
						'name': tool_name,
						'description': f'Return a JSON object of type {tool_name}',
						'parameters': schema,
					},
				}]
				
				params['tools'] = tools
				params['tool_choice'] = {'type': 'function', 'function': {'name': tool_name}}
				
				response = await litellm.acompletion(**params)
				
				# Extract function call result
				message = response.choices[0].message  # type: ignore
				if not message.tool_calls:
					raise ModelProviderError(
						message='Expected tool_calls in response but got none',
						model=self.name,
					)
				
				# Parse the function arguments
				tool_call = message.tool_calls[0]
				raw_args = tool_call.function.arguments
				
				if isinstance(raw_args, str):
					parsed_args = json.loads(raw_args)
				else:
					parsed_args = raw_args
				
				# Validate and create the output format instance
				parsed_response = output_format.model_validate(parsed_args)
				usage = self._get_usage(response)
				
				return ChatInvokeCompletion(
					completion=parsed_response,
					usage=usage,
				)

		except Exception as e:
			# Handle LiteLLM-specific exceptions
			error_str = str(e).lower()
			
			# Check for rate limit errors
			if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
				raise ModelRateLimitError(str(e), model=self.name) from e
			
			# General provider error
			raise ModelProviderError(str(e), model=self.name) from e 