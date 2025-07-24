"""
# @file purpose: Defines message serialization for LiteLLM integration

This file provides message serialization functionality for the LiteLLM provider,
converting browser-use internal message formats to LiteLLM-compatible formats.
LiteLLM expects OpenAI-compatible message formats, so this serializer leverages
the existing OpenAI serializer for consistency and maintainability.
"""

from typing import Any, Dict, List

from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer


class LiteLLMMessageSerializer:
	"""
	Serializer for converting between custom message types and LiteLLM message formats.
	
	LiteLLM uses OpenAI-compatible message formats, so this serializer delegates
	to the existing OpenAI serializer for consistency and to avoid code duplication.
	"""

	@staticmethod
	def serialize_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
		"""
		Serialize a list of custom messages to LiteLLM-compatible format.
		
		Args:
			messages: List of custom message objects
			
		Returns:
			List of dictionaries in LiteLLM/OpenAI-compatible format
		"""
		# LiteLLM uses OpenAI-compatible message format, so we can reuse the OpenAI serializer
		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
		
		# Convert the OpenAI message params to plain dictionaries for LiteLLM
		litellm_messages = []
		for msg in openai_messages:
			# Convert TypedDict to regular dict for LiteLLM compatibility
			litellm_messages.append(dict(msg))
		
		return litellm_messages 