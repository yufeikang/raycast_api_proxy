import json
import logging
import os
import re
from functools import cache
from typing import List

import openai

from app.utils import json_dumps
from .base import ApiProviderAbc, _get_default_model_dict, _get_model_extra_info

logger = logging.getLogger(__name__)


class OpenAIProvider(ApiProviderAbc):

    api_type = "openai"

    @classmethod
    def is_start_available(cls):
        return os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        provider: str = "openai",
        allow_model_patterns: List[str] = [],
        skip_models_patterns: List[str] = [],
        temperature: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.provider = provider or os.environ.get(
            "OPENAI_PROVIDER", "openai"
        )  # for openai api compatible provider
        api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.allow_model_patterns = allow_model_patterns
        if not allow_model_patterns and provider == "openai":
            self.allow_model_patterns = ["gpt-\\d+", "o1"]
            logger.debug(f"Model filter: {self.allow_model_patterns}")
        self.skip_models_patterns = skip_models_patterns
        if not skip_models_patterns and provider == "openai":
            self.skip_models_patterns = [
                # include all realtime models
                ".+realtime.\\+",
                ".+audio.\\+",
                ".+\\d{4}-\\d{2}-\\d{2}$",
            ]
            logger.debug(f"Skip model filter: {self.skip_models_patterns}")
        if kwargs and "is_azure" in kwargs:
            logger.info("Init Azure API")
            del kwargs["is_azure"]
            self.openai_client = openai.AsyncAzureOpenAI(**kwargs)
        else:
            logger.info(f"Init OpenAI API via {self.provider}")
            logger.info(f"OpenAI API base url: {openai.base_url}")
            self.openai_client = openai.AsyncOpenAI(
                api_key=api_key, base_url=base_url, **kwargs
            )

    def __build_openai_messages(self, raycast_data: dict):
        openai_messages = []
        temperature = self.temperature
        for msg in raycast_data["messages"]:
            if "system_instructions" in msg["content"]:
                openai_messages.append(
                    {
                        "role": "system",
                        "content": msg["content"]["system_instructions"],
                    }
                )
            if "command_instructions" in msg["content"]:
                openai_messages.append(
                    {
                        "role": "system",
                        "content": msg["content"]["command_instructions"],
                    }
                )
            if "additional_system_instructions" in raycast_data:
                openai_messages.append(
                    {
                        "role": "system",
                        "content": raycast_data["additional_system_instructions"],
                    }
                )
            if "text" in msg["content"]:
                openai_messages.append(
                    {"role": msg["author"], "content": msg["content"]["text"]}
                )
            if "temperature" in msg["content"]:
                temperature = msg["content"]["temperature"]
        return openai_messages, temperature

    async def chat_completions(self, raycast_data: dict):
        openai_messages, temperature = self.__build_openai_messages(raycast_data)
        model = raycast_data["model"]
        tools = []
        if (
            "image_generation_tool" in raycast_data
            and raycast_data["image_generation_tool"]
        ):
            tools.append(self.__build_openai_function_img_tool(raycast_data))
        async for i in self.__warp_chat(
            openai_messages, model, temperature, tools=tools, stream=True
        ):
            yield i

    async def __warp_chat(self, messages, model, temperature, **kwargs):
        functions = {}
        current_function_id = None
        async for choice, error in self.__chat(messages, model, temperature, **kwargs):
            if error:
                error_message = (
                    error.body.get("message", {}) if error.body else error.message
                )
                yield f'data: {json_dumps({"text":error_message, "finish_reason":"error"})}\n\n'
                return
            if choice.delta and choice.delta.content:
                yield f'data: {json_dumps({"text": choice.delta.content})}\n\n'
            if choice.delta.tool_calls:
                logger.debug(f"Tool calls: {choice.delta}")
                for tool_call in choice.delta.tool_calls:
                    logger.debug(f"Tool call: {tool_call}")
                    if tool_call.id and tool_call.type == "function":
                        current_function_id = tool_call.id
                        if current_function_id not in functions:
                            functions[current_function_id] = {
                                "delta": choice.delta,
                                "name": tool_call.function.name,
                                "args": "",
                            }
                    # add arguments stream string to the current function
                    functions[current_function_id][
                        "args"
                    ] += tool_call.function.arguments
                    continue
            if choice.finish_reason is not None:
                logger.debug(f"Finish reason: {choice.finish_reason}")
                if choice.finish_reason == "tool_calls":
                    continue
                yield f'data: {json_dumps({"text": "", "finish_reason": choice.finish_reason})}\n\n'
        if functions:
            logger.debug(f"Tool functions: {functions}")
            for tool_call_id, tool in functions.items():
                delta, name, args = tool["delta"], tool["name"], tool["args"]
                logger.debug(f"Tool call: {name} with args: {args}")
                args = json.loads(args)
                messages.append(delta)  # add the tool call to messages
                tool_res = None
                if name == "generate_image":
                    yield f'data: {json_dumps({"text": "Generating image..."})}\n\n'
                    tool_res = await self.__generate_image(**args)
                else:
                    logger.error(f"Unknown tool function: {name}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool["name"],
                        "content": tool_res,
                    }
                )
                async for i in self.__warp_chat(messages, model, temperature, **kwargs):
                    yield i

    def __build_openai_function_img_tool(self, raycast_data: dict):
        return {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on dall-e-3",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate the image for dall-e-3 model",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }

    async def __generate_image(self, prompt, model="dall-e-3"):
        try:
            res = await self.openai_client.images.generate(
                model=model,
                prompt=prompt,
                response_format="url",
            )
            return json.dumps({"url": res.data[0].url})
        except openai.OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            return json.dumps({"error": str(e)})

    async def translate_completions(self, raycast_data: dict):
        messages = [
            {"role": "system", "content": "Translate the following text:"},
            {
                "role": "system",
                "content": f"The target language is: {raycast_data['target']}",
            },
            {"role": "user", "content": raycast_data["q"]},
        ]
        model = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-3.5-turbo")
        logger.debug(f"Translating: {raycast_data['q']} with model: {model}")
        async for choice, error in self.__chat(messages, model=model, temperature=0.8):
            if error:
                error_message = (
                    error.body.get("message", {}) if error.body else error.message
                )
                yield f"Error: {error_message}"
                return
            if choice.delta:
                yield choice.delta.content

    def __merge_messages(self, messages):
        """
        merge same role messages to one message
        """
        merged_messages = []
        for msg in messages:
            if not merged_messages:
                merged_messages.append(msg)
                continue
            last_msg = merged_messages[-1]
            if last_msg.get("role") == msg.get("role"):
                last_msg["content"] += "\n" + msg.get("content")
            else:
                merged_messages.append(msg)
        return merged_messages

    async def __chat(self, messages, model, temperature, **kwargs):
        if "tools" in kwargs and not kwargs["tools"]:
            # pop tools from kwargs, empty tools will cause error
            kwargs.pop("tools")
        stream = "stream" in kwargs and kwargs["stream"]
        try:
            not_support_system_role_models = ["o1", "o1-mini", "deepseek-reasoner"]
            for m in messages:
                # check model is o1 replace role system to user
                if (
                    model in not_support_system_role_models
                    and m.get("role") == "system"
                ):
                    m["role"] = "user"
            messages = self.__merge_messages(messages)
            logger.debug(f"openai chat, messages: {messages}")
            res = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=1 if model.startswith("o1") else temperature,
                **kwargs,
            )
        except openai.OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            yield None, e
            return
        if not stream:
            choice = res.choices[0]
            choice.delta = res.choices[0].message
            yield choice, None
            return
        async for chunk in res:
            if not chunk.choices:
                logger.error(f"OpenAI error: {chunk}")
                yield None, None
                return
            yield chunk.choices[0], None

    @cache
    async def get_models(self):
        default_models = _get_default_model_dict("openai-gpt-4o-mini")
        """
        {
            "id": "gpt-4o",
            "created": 1715367049,
            "object": "model",
            "owned_by": "system"
        }
        """
        openai_models = (await self.openai_client.models.list()).data
        models = []
        for model in openai_models:
            if self.allow_model_patterns and all(
                not re.match(f, model.id) for f in self.allow_model_patterns
            ):
                logger.debug(f"Skipping model: {model.id}, not match any allow filter")
                continue
            if any(re.match(f, model.id) for f in self.skip_models_patterns):
                logger.debug(f"Skipping model: {model.id} match skip filter")
                continue

            model_id = f"{self.provider}-{model.id}"
            logger.debug(f"Allowed model: {model.id}")
            models.append(
                {
                    "id": model_id,
                    "model": model.id,
                    "name": f"{self.provider} {model.id}",
                    "provider": "openai",
                    "provider_name": self.provider,
                    "provider_brand": self.provider,
                    "context": 16,
                    **_get_model_extra_info(model.id),
                }
            )
        return {"default_models": default_models, "models": models}
