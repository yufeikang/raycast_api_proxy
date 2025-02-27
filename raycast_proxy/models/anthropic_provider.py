import logging
import os

import anthropic

from raycast_proxy.utils import json_dumps

from .base import ApiProviderAbc, _get_default_model_dict, _get_model_extra_info

logger = logging.getLogger(__name__)


class AnthropicProvider(ApiProviderAbc):
    api_type = "anthropic"

    @classmethod
    def is_start_available(cls):
        return os.environ.get("ANTHROPIC_API_KEY")

    def __init__(self, api_key=None, max_token=None, **kwargs) -> None:
        super().__init__()
        logger.info("Init Anthropic API")
        self.max_tokens = kwargs.get("max_tokens", os.environ.get("MAX_TOKENS", 8192))
        self.temperature = kwargs.get("temperature", os.environ.get("TEMPERATURE", 0.5))
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    async def chat_completions(self, raycast_data: dict):
        messages = self.__build_anthropic_messages(raycast_data)
        model = raycast_data["model"]

        try:
            response = await self.anthropic_client.messages.create(
                max_tokens=self.max_tokens,
                model=model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in response:
                if chunk.type == "content_block_delta":
                    yield f'data: {json_dumps({"text": chunk.delta.text})}\n\n'
                elif chunk.type == "message_stop":
                    yield f'data: {json_dumps({"text": "", "finish_reason": "stop"})}\n\n'
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            yield f'data: {json_dumps({"text": str(e), "finish_reason": "error"})}\n\n'

    def __build_anthropic_messages(self, raycast_data: dict):
        anthropic_messages = []
        for msg in raycast_data["messages"]:
            if "system_instructions" in msg["content"]:
                anthropic_messages.append(
                    {"role": "system", "content": msg["content"]["system_instructions"]}
                )
            if "command_instructions" in msg["content"]:
                anthropic_messages.append(
                    {
                        "role": "system",
                        "content": msg["content"]["command_instructions"],
                    }
                )
            if "text" in msg["content"]:
                anthropic_messages.append(
                    {"role": msg["author"], "content": msg["content"]["text"]}
                )
        return anthropic_messages

    async def translate_completions(self, raycast_data: dict):
        messages = [
            {
                "role": "system",
                "content": f"Translate the following text to {raycast_data['target']}:",
            },
            {"role": "user", "content": raycast_data["q"]},
        ]
        model = os.environ.get("ANTHROPIC_TRANSLATE_MODEL", "claude-3-opus-20240229")

        try:
            response = await self.anthropic_client.messages.create(
                max_tokens=self.max_tokens,
                model=model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in response:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic translation error: {e}")
            yield f"Error: {str(e)}"

    async def get_models(self):
        default_models = _get_default_model_dict("claude-3-5-sonnet-20240620")
        models = [
            {
                "id": "claude-3-5-sonnet-20240620",
                "model": "claude-3-5-sonnet-20240620",
                "name": "Claude 3.5 Sonnet",
                "provider": "anthropic",
                "provider_name": "Anthropic",
                "provider_brand": "anthropic",
                "context": 32,
                **_get_model_extra_info("claude-3-5-sonnet-20240620"),
            },
            {
                "id": "claude-3-opus-20240229",
                "model": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "provider_name": "Anthropic",
                "provider_brand": "anthropic",
                "context": 32,
                **_get_model_extra_info("claude-3-opus-20240229"),
            },
            {
                "id": "claude-3-sonnet-20240229",
                "model": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "provider": "anthropic",
                "provider_name": "Anthropic",
                "provider_brand": "anthropic",
                "context": 16,
                **_get_model_extra_info("claude-3-sonnet-20240229"),
            },
        ]
        return {"default_models": default_models, "models": models}
