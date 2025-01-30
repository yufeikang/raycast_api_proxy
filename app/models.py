import abc
import json
import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import List

import anthropic
import google.generativeai as genai
import openai
import yaml
from google.generativeai import GenerativeModel

from app.utils import json_dumps

logger = logging.getLogger(__name__)


class ApiProviderAbc(abc.ABC):

    api_type = None

    @classmethod
    @abc.abstractmethod
    def is_start_available(cls):
        pass

    @abc.abstractmethod
    async def chat_completions(self, raycast_data: dict):
        pass

    @abc.abstractmethod
    async def translate_completions(self, raycast_data: dict):
        pass

    async def get_models(self):
        pass


def _get_default_model_dict(model_name: str):
    return {
        "chat": model_name,
        "quick_ai": model_name,
        "commands": model_name,
        "api": model_name,
        "emoji_search": model_name,
    }


def _get_model_extra_info(name=""):
    """
    "capabilities": {
          "web_search": "full" / "always_on"
          "image_generation": "full"
      },
      "abilities": {
          "web_search": {
              "toggleable": true
          },
         "image_generation": {
                "model": "dall-e-3"
         },
        "system_message": {
            "supported": true
        },
        "temperature": {
            "supported": true
        }
      },

    """
    ext = {
        "description": "model description",
        "requires_better_ai": True,
        "features": ["chat", "quick_ai", "commands", "api", "emoji_search"],
        "suggestions": ["chat", "quick_ai", "commands", "api", "emoji_search"],
        "capabilities": {},
        "abilities": {},
        "availability": "public",
        "status": None,
        "speed": 3,
        "intelligence": 3,
    }
    if "gpt-4" in name:
        ext["capabilities"] = {
            "web_search": "full",
            "image_generation": "full",
        }
        ext["abilities"] = {
            "web_search": {
                "toggleable": True,
            },
            "image_generation": {
                "model": "dall-e-3",
            },
            "system_message": {
                "supported": True,
            },
            "temperature": {
                "supported": True,
            },
        }
    # o1 models don't support system_message and temperature
    if "o1" in name:
        ext["abilities"] = {
            "system_message": {
                "supported": False,
            },
            "temperature": {
                "supported": False,
            },
        }
    return ext


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
                            "description": "The prompt to generate the image for dall-e-3 model, e.g. 'a cat in the forest', please generate the prompt by usr input",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }

    async def __generate_image(self, prompt, model="dall-e-3"):
        # return '{"url": "https://images.ctfassets.net/kftzwdyauwt9/1ZTOGp7opuUflFmI2CsATh/df5da4be74f62c70d35e2f5518bf2660/ChatGPT_Carousel1.png?w=828&q=90&fm=webp"}'  # debug image
        try:
            res = await self.openai_client.images.generate(
                model=model,
                prompt=prompt,
                response_format="url",
                # size="256x256",
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
        # stream = "tools" not in kwargs
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
                # stream=stream,
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


class GeminiProvider(ApiProviderAbc):
    api_type = "gemini"

    def __init__(
        self,
        api_key=None,
        allow_model_patterns: List[str] = [],
        skip_models_patterns: List[str] = [],
        temperature: float = 0.5,
        **kwargs
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
            kwargs.get("temperature") or
            os.environ.get("TEMPERATURE") or
            temperature
        )
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))

    @classmethod
    def is_start_available(cls):
        return os.environ.get("GOOGLE_API_KEY")

    async def chat_completions(self, raycast_data: dict):
        model_name = raycast_data["model"]
        model = genai.GenerativeModel(model_name)
        google_message = []
        temperature = self.temperature
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
            model, google_message, temperature, stream=True
        )
        try:
            for chunk in result:
                logger.debug(f"Gemini chat_completions response chunk: {chunk.text}")
                yield f'data: {json_dumps({"text": chunk.text})}\n\n'
        except genai.types.BlockedPromptException as e:
            logger.debug(f"Gemini response finish: {e}")
            yield f'data: {json_dumps({"text": "", "finish_reason": e})}\n\n'

    async def translate_completions(self, raycast_data: dict):
        model_name = raycast_data.get("model", "gemini-pro")
        model = genai.GenerativeModel(model_name)
        target_language = raycast_data["target"]
        google_message = f"translate the following text to {target_language}:\n"
        google_message += raycast_data["q"]
        logger.debug(f"text: {google_message}")
        result = self.__generate_content(
            model, google_message, temperature=0.8, stream=True
        )
        try:
            for chunk in result:
                logger.debug(
                    f"Gemini translate_completions response chunk: {chunk.text}"
                )
                yield chunk.text
        except genai.types.BlockedPromptException as e:
            logger.debug(f"Gemini response finish: {e}")

    def __generate_content(
        self, model: GenerativeModel, google_message, temperature, stream=False
    ):
        return model.generate_content(
            google_message,
            stream=stream,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                temperature=temperature,
            ),
        )

    async def get_models(self):
        genai_models = genai.list_models()
        models = []
        for model in genai_models:
            model_id = model.name.replace("models/", "")
            if not any(re.match(f, model_id) for f in self.allow_model_patterns):
                logger.debug(f"Skipping model: {model_id}, not match any allow filter")
                continue
            if any(re.match(f, model_id) for f in self.skip_models_patterns):
                logger.debug(f"Skipping model: {model_id} match skip filter")
                continue
            models.append({
                "id": model_id,
                "model": model_id,
                "name": model.display_name,
                "provider": "google",
                "provider_name": "Google",
                "provider_brand": "google",
                "context": 16,
                **_get_model_extra_info(model_id),
            })
        return {
            "default_models": _get_default_model_dict(models[0]["id"]),
            "models": models
        }


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


MODELS_DICT = {}
MODELS_AVAILABLE = []
DEFAULT_MODELS = {}
AVAILABLE_DEFAULT_MODELS = []


async def _add_available_model(api: ApiProviderAbc):
    global MODELS_DICT, MODELS_AVAILABLE, DEFAULT_MODELS, AVAILABLE_DEFAULT_MODELS

    models = await api.get_models()
    MODELS_AVAILABLE.extend(models["models"])
    AVAILABLE_DEFAULT_MODELS.append(models["default_models"])
    MODELS_DICT.update({model["model"]: api for model in models["models"]})


async def init_models():
    config_file = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    if not config_file.exists():
        logger.info(f"Config file not found: {config_file}")
        await init_models_from_env()
        return
    config = yaml.load(config_file.read_text(), Loader=yaml.FullLoader)
    # get all implement for ProviderApiAbc
    impl = {cls.api_type: cls for cls in ApiProviderAbc.__subclasses__()}
    for model_config in config["models"]:
        api_type = model_config.get("api_type")
        try:
            logger.info(f"Init model: {model_config['provider_name']}")
            api = impl[api_type](
                **model_config["params"], provider=model_config["provider_name"]
            )
            await _add_available_model(api)

        except KeyError:
            logger.error(f"Unknown api type: {api_type}")
        except Exception as e:
            logger.error(f"Error init model: {model_config}, {e}")


async def init_models_from_env():
    logger.warning("Use config.yml, this method is deprecated")
    if GeminiProvider.is_start_available():
        logger.info("Google API is available")
        _api = GeminiProvider()
        await _add_available_model(_api)
    if OpenAIProvider.is_start_available():
        logger.info("OpenAI API is available")
        _api = OpenAIProvider()
        await _add_available_model(_api)
    if AnthropicProvider.is_start_available():
        logger.info("Anthropic API is available")
        _api = AnthropicProvider()
        await _add_available_model(_api)


def get_bot(model_id):
    if not model_id:
        return next(iter(MODELS_DICT.values()))
    logger.debug(f"Getting bot for model: {model_id}")
    if model_id not in MODELS_DICT:
        logger.error(f"Model not found: {model_id}")
        return None
    return MODELS_DICT.get(model_id)
