import logging
import os
import re
from typing import List

import google.genai as genai
import google.genai.types as genai_types

from app.utils import json_dumps
from .base import ApiProviderAbc, _get_default_model_dict, _get_model_extra_info

logger = logging.getLogger(__name__)


class GeminiProvider(ApiProviderAbc):
    api_type = "gemini"

    def __init__(
        self,
        api_key=None,
        allow_model_patterns: List[str] = [],
        skip_models_patterns: List[str] = [],
        temperature: float = 0.5,
        harm_threshold: str = "BLOCK_ONLY_HIGH",
        grounding_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        logger.info("Init Google API")
        self.allow_model_patterns = allow_model_patterns or [
            "^.*-latest$",
            "^.*-exp$",
        ]
        self.skip_models_patterns = skip_models_patterns or [
            "^gemini-1.0-.*$",
        ]
        self.temperature = (
            kwargs.get("temperature") or os.environ.get("TEMPERATURE") or temperature
        )
        self.harm_threshold = (
            kwargs.get("harm_threshold")
            or os.environ.get("GOOGLE_HARM_THRESHOLD")
            or harm_threshold
        )
        self.grounding_threshold = (
            kwargs.get("grounding_threshold")
            or os.environ.get("GOOGLE_GROUNDING_THRESHOLD")
            or grounding_threshold
        )
        self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    @classmethod
    def is_start_available(cls):
        return os.environ.get("GOOGLE_API_KEY")

    async def chat_completions(self, raycast_data: dict):
        model_name = raycast_data["model"]
        system_instruction = "\n".join(
            filter(
                None,
                [
                    raycast_data.get("system_instruction"),
                    raycast_data.get("additional_system_instructions"),
                ],
            )
        )
        temperature = raycast_data.get("temperature") or self.temperature
        web_search = any(
            tool["name"] == "web_search" and tool["type"] == "remote_tool"
            for tool in raycast_data.get("tools", [])
        )
        google_message = []
        for msg in raycast_data["messages"]:
            content = {"role": "user"}
            parts = []
            if "system_instructions" in msg["content"]:
                parts.append({"text": msg["content"]["system_instructions"]})
            if "command_instructions" in msg["content"]:
                parts.append({"text": msg["content"]["command_instructions"]})
            if "text" in msg["content"]:
                parts.append({"text": msg["content"]["text"]})
            if "author" in msg:
                role = "user" if msg["author"] == "user" else "model"
                content["role"] = role
            if "temperature" in msg["content"]:
                temperature = msg["content"]["temperature"]
            content["parts"] = parts
            google_message.append(content)

        logger.debug(f"text: {google_message}")
        result = self.__generate_content(
            model_name, google_message, temperature, system_instruction, web_search
        )
        for chunk in result:
            if chunk.prompt_feedback:
                feedback = chunk.prompt_feedback.block_reason_message
                logger.debug(f"Gemini response finish: {feedback}")
                yield f'data: {json_dumps({"text": "", "finish_reason": feedback})}\n\n'
            else:
                logger.debug(f"Gemini chat_completions response chunk: {chunk.text}")
                yield f'data: {json_dumps({"text": chunk.text})}\n\n'

    async def translate_completions(self, raycast_data: dict):
        model_name = raycast_data.get("model", "gemini-pro")
        target_language = raycast_data["target"]
        google_message = f"translate the following text to {target_language}:\n"
        google_message += raycast_data["q"]
        logger.debug(f"text: {google_message}")
        result = self.__generate_content(model_name, google_message, temperature=0.8)
        for chunk in result:
            if chunk.prompt_feedback:
                feedback = chunk.prompt_feedback.block_reason_message
                logger.debug(f"Gemini response finish: {feedback}")
                yield feedback
            else:
                logger.debug(
                    f"Gemini translate_completions response chunk: {chunk.text}"
                )
                yield chunk.text

    def __generate_content(
        self, model, contents, temperature, system_instruction=None, web_search=False
    ):
        safety_settings = [
            genai_types.SafetySetting(**d)
            for d in [
                {"category": category, "threshold": self.harm_threshold}
                for category in [
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
            ]
        ]
        google_search_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearchRetrieval(
                dynamic_retrieval_config=genai_types.DynamicRetrievalConfig(
                    dynamic_threshold=self.grounding_threshold
                )
            )
        )
        tools = [google_search_tool] if web_search else []
        return self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                candidate_count=1,
                system_instruction=system_instruction,
                temperature=temperature,
                safety_settings=safety_settings or None,
                tools=tools or None,
            ),
        )

    async def get_models(self):
        genai_models = self.client.models.list()
        models = []
        for model in genai_models:
            model_id = model.name.replace("models/", "")
            if not any(re.match(f, model_id) for f in self.allow_model_patterns):
                logger.debug(f"Skipping model: {model_id}, not match any allow filter")
                continue
            if any(re.match(f, model_id) for f in self.skip_models_patterns):
                logger.debug(f"Skipping model: {model_id} match skip filter")
                continue
            models.append(
                {
                    **_get_model_extra_info(model_id),
                    "id": model_id,
                    "model": model_id,
                    "name": model.display_name,
                    "description": model.description,
                    "provider": "google",
                    "provider_name": "Google",
                    "provider_brand": "google",
                    "context": int(model.input_token_limit / 1000),
                }
            )
        return {
            "default_models": _get_default_model_dict(models[0]["id"]),
            "models": models,
        }
