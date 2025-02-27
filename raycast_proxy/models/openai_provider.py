import json
import logging
import os
import re
from functools import cache
from typing import List

import openai

from raycast_proxy.utils import json_dumps

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
            logger.debug("Model filter: %s", self.allow_model_patterns)
        self.skip_models_patterns = skip_models_patterns
        if not skip_models_patterns and provider == "openai":
            self.skip_models_patterns = [
                # include all realtime models
                ".+realtime.\\+",
                ".+audio.\\+",
                ".+\\d{4}-\\d{2}-\\d{2}$",
            ]
            logger.debug("Skip model filter: %s", self.skip_models_patterns)
        if kwargs and "is_azure" in kwargs:
            logger.info("Init Azure API")
            del kwargs["is_azure"]
            self.openai_client = openai.AsyncAzureOpenAI(**kwargs)
        else:
            logger.info("Init OpenAI API via %s", self.provider)
            logger.info("OpenAI API base url: %s", openai.base_url)
            self.openai_client = openai.AsyncOpenAI(
                api_key=api_key, base_url=base_url, **kwargs
            )

    def __build_openai_messages(self, messages, additional_system_instructions=None):
        openai_messages = []
        temperature = self.temperature
        if additional_system_instructions:
            openai_messages.append(
                {"role": "system", "content": additional_system_instructions}
            )
        for msg in messages:

            # may by deprecated
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
            # First handle tool calls if present
            tool_call_message = None
            if "tool_calls" in msg:
                tool_call = msg["tool_calls"][0]  # Get first tool call
                function_args = tool_call["function"]["arguments"]
                if isinstance(function_args, dict):
                    function_args = json.dumps(function_args)

                tool_call_id = tool_call.get("id", "")
                logger.debug("Processing function call: %s", tool_call_id)

                # Create assistant's function call message
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": function_args,
                            },
                        }
                    ],
                }

            # Then handle any text content
            if "text" in msg["content"]:
                if msg["author"] == "assistant":
                    # If this is an assistant message with tool calls,
                    # update the tool call message with the text content
                    if tool_call_message:
                        tool_call_message["content"] = msg["content"]["text"]
                    else:
                        # Regular assistant message without tool calls
                        openai_messages.append(
                            {"role": "assistant", "content": msg["content"]["text"]}
                        )
                elif msg["author"] != "tool":
                    # tool responses are handled separately
                    # Handle other roles (user, system)
                    openai_messages.append(
                        {"role": msg["author"], "content": msg["content"]["text"]}
                    )

            # Add the tool call message if we created one
            if tool_call_message:
                openai_messages.append(tool_call_message)

            # Handle tool responses
            if msg["author"] == "tool":
                tool_call_id = msg["tool_call_id"]
                function_name = msg["name"]
                logger.debug(
                    "Processing tool response for: %s, function: %s",
                    tool_call_id,
                    function_name,
                )

                # Find the last assistant message with matching tool_calls
                last_assistant_msg = None
                for prev_msg in openai_messages[::-1]:
                    if (
                        prev_msg.get("role") == "assistant"
                        and "tool_calls" in prev_msg
                        and any(
                            call["id"] == tool_call_id
                            for call in prev_msg["tool_calls"]
                        )
                    ):
                        last_assistant_msg = prev_msg
                        break

                if not last_assistant_msg:
                    logger.warning(
                        "Skipping tool response - no matching assistant message with tool calls found: %s",
                        tool_call_id,
                    )
                    continue

                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": msg["content"]["text"],
                    }
                )
                logger.debug("Added tool response matching call: %s", tool_call_id)

            if "temperature" in msg["content"]:
                temperature = msg["content"]["temperature"]

        return openai_messages, temperature

    async def chat_completions(self, raycast_data: dict):
        openai_messages, temperature = self.__build_openai_messages(
            raycast_data["messages"], raycast_data.get("additional_system_instructions")
        )
        model = raycast_data["model"]
        tools = []
        if "tools" in raycast_data:
            for tool in raycast_data["tools"]:
                if tool["type"] == "local_tool" and "function" in tool:
                    tools.append({"type": "function", "function": tool["function"]})
        elif (
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
                logger.debug("Tool calls: %s", choice.delta)
                # Stream individual tool call updates
                for tool_call in choice.delta.tool_calls:
                    tool_call_data = {
                        "text": "",
                        "finish_reason": None,
                        "tool_calls": [{"index": 0}],
                    }

                    if tool_call.id and tool_call.type == "function":
                        current_function_id = tool_call.id
                        if current_function_id not in functions:
                            functions[current_function_id] = {
                                "delta": choice.delta,
                                "name": tool_call.function.name,
                                "args": "",
                            }
                            # Add function info to tool call data
                            tool_call_data["tool_calls"][0].update(
                                {
                                    "id": current_function_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": "",
                                    },
                                }
                            )

                    if tool_call.function.arguments:
                        # Add arguments to both tracking dict and current response
                        functions[current_function_id][
                            "args"
                        ] += tool_call.function.arguments
                        tool_call_data["tool_calls"][0]["function"] = {
                            "arguments": tool_call.function.arguments
                        }

                    yield f"data: {json_dumps(tool_call_data)}\n\n"
                continue

            if choice.finish_reason is not None:
                logger.debug("Finish reason: %s", choice.finish_reason)
                if choice.finish_reason == "tool_calls":
                    # Send final tool calls event with complete arguments
                    complete_tool_calls = []
                    for tool_call_id, tool in functions.items():
                        complete_tool_calls.append(
                            {
                                "name": tool["name"],
                                "arguments": tool["args"],
                                "id": tool_call_id,
                            }
                        )
                    yield f'data: {json_dumps({"finish_reason":"tool_calls", "text":"", "tool_calls":complete_tool_calls})}\n\n'
                    continue
                yield f'data: {json_dumps({"text": "", "finish_reason": choice.finish_reason})}\n\n'
        # Only continue conversation for image generation
        if functions and any(
            tool["name"] == "generate_image" for tool in functions.values()
        ):
            logger.debug("Processing image generation function")
            for tool_call_id, tool in functions.items():
                if tool["name"] == "generate_image":
                    args = json.loads(tool["args"])
                    tool_res = await self.__generate_image(**args)
                    new_messages = messages + [
                        tool["delta"],  # add the tool call
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool["name"],
                            "content": tool_res,
                        },
                    ]
                    # Continue conversation with new messages
                    async for i in self.__warp_chat(
                        new_messages, model, temperature, **kwargs
                    ):
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
            logger.error("OpenAI error: %s", e)
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
        logger.debug("Translating: %s with model: %s", raycast_data["q"], model)
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
        merge same role messages to one message, preserving function call sequences
        """
        merged_messages = []
        tool_sequences = []  # Store tool call/response pairs in order
        current_normal_messages = []  # Store non-tool messages

        for msg in messages:
            # Handle function calls
            if "tool_calls" in msg:
                # Flush any pending normal messages
                merged_messages.extend(current_normal_messages)
                current_normal_messages = []

                # Start a new tool sequence
                tool_sequences.append({"call": msg, "response": None})
                continue

            # Handle tool responses
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                # Find the matching call and add the response
                for seq in tool_sequences:
                    if (
                        seq["call"]["tool_calls"][0]["id"] == msg["tool_call_id"]
                        and seq["response"] is None
                    ):
                        seq["response"] = msg
                        break
                continue

            # Handle normal messages for merging
            if not current_normal_messages:
                current_normal_messages.append(msg)
                continue

            last_msg = current_normal_messages[-1]
            if last_msg.get("role") == msg.get("role"):
                if (
                    last_msg.get("content") is not None
                    and msg.get("content") is not None
                ):
                    last_msg["content"] += "\n" + msg.get("content")
                elif msg.get("content") is not None:
                    last_msg["content"] = msg.get("content")
            else:
                current_normal_messages.append(msg)

        # Add any remaining normal messages
        merged_messages.extend(current_normal_messages)

        # Add tool sequences in order, ensuring each call is followed by its response
        for seq in tool_sequences:
            if seq["call"]:
                merged_messages.append(seq["call"])
            if seq["response"]:
                merged_messages.append(seq["response"])

        logger.debug("Final messages: %s", [msg.get("role") for msg in merged_messages])
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
            logger.debug("openai chat, messages: %s", messages)
            res = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=1 if model.startswith("o1") else temperature,
                **kwargs,
            )
        except openai.OpenAIError as e:
            logger.error("OpenAI error: %s", e)
            yield None, e
            return
        if not stream:
            choice = res.choices[0]
            choice.delta = res.choices[0].message
            yield choice, None
            return
        async for chunk in res:
            if not chunk.choices:
                logger.error("OpenAI error: %s", chunk)
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
                logger.debug("Skipping model: %s, not match any allow filter", model.id)
                continue
            if any(re.match(f, model.id) for f in self.skip_models_patterns):
                logger.debug("Skipping model: %s match skip filter", model.id)
                continue

            model_id = f"{self.provider}-{model.id}"
            logger.debug("Allowed model: %s", model.id)
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
