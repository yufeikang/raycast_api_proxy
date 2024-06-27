import abc
import json
import logging
import os

import anthropic
import google.generativeai as genai
import openai
from google.generativeai import GenerativeModel

from app.utils import json_dumps

logger = logging.getLogger(__name__)

MAX_TOKENS = os.environ.get("MAX_TOKENS", 1024)


class ChatBotAbc(abc.ABC):

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

    def get_models(self):
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
        }
    return ext


class OpenAIChatBot(ChatBotAbc):

    @classmethod
    def is_start_available(cls):
        return os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )

    def __init__(self) -> None:
        super().__init__()
        openai.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        is_azure = openai.api_type in ("azure", "azure_ad", "azuread")
        if is_azure:
            logger.info("Using Azure API")
            self.openai_client = openai.AsyncAzureOpenAI(
                azure_endpoint=os.environ.get("OPENAI_AZURE_ENDPOINT"),
                azure_deployment=os.environ.get("AZURE_DEPLOYMENT_ID", None),
            )
        else:
            logger.info("Using OpenAI API")
            self.openai_client = openai.AsyncOpenAI()

    def __build_openai_messages(self, raycast_data: dict):
        openai_messages = []
        temperature = os.environ.get("TEMPERATURE", 0.5)
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
            openai_messages, model, temperature, tools=tools
        ):
            yield i

    async def __warp_chat(self, messages, model, temperature, **kwargs):
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
                has_valid_tool = False
                messages.append(choice.delta)  # add the tool call to messages
                for tool_call in choice.delta.tool_calls:
                    tool_call_id = tool_call.id
                    tool_function_name = tool_call.function.name
                    logger.debug(f"Tool call: {tool_function_name}")
                    if tool_function_name == "generate_image":
                        if not tool_call.function.arguments:
                            continue
                        function_args = json.loads(tool_call.function.arguments)
                        yield f"data: {json_dumps({'text': 'Generating image...'})}\n\n"
                        fun_res = await self.__generate_image(**function_args)
                        # add to messages
                        has_valid_tool = True
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_function_name,
                                "content": fun_res,
                            }
                        )
                if has_valid_tool:
                    async for i in self.__warp_chat(messages, model, temperature):
                        yield i
                    continue
            if choice.finish_reason is not None:
                yield f'data: {json_dumps({"text": "", "finish_reason": choice.finish_reason})}\n\n'

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

    async def __chat(self, messages, model, temperature, **kwargs):
        if "tools" in kwargs and not kwargs["tools"]:
            # pop tools from kwargs, empty tools will cause error
            kwargs.pop("tools")
        stream = "tools" not in kwargs
        try:
            logger.debug(f"openai chat stream: {stream}")
            res = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                n=1,
                temperature=temperature,
                stream=stream,
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
            yield chunk.choices[0], None

    def get_models(self):
        default_models = _get_default_model_dict("openai-gpt-4o")
        models = [
            {
                "id": "openai-gpt-3.5-turbo",
                "model": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 16,
                **_get_model_extra_info("gpt-3.5-turbo"),
            },
            {
                "id": "openai-gpt-4o",
                "model": "gpt-4o",
                "name": "GPT-4o",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 8,
                **_get_model_extra_info("gpt-4o"),
            },
            {
                "id": "openai-gpt-4-turbo",
                "model": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 8,
                **_get_model_extra_info("gpt-4-turbo"),
            },
        ]
        return {"default_models": default_models, "models": models}


class GeminiChatBot(ChatBotAbc):
    def __init__(self) -> None:
        super().__init__()
        logger.info("Using Google API")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=google_api_key)

    @classmethod
    def is_start_available(cls):
        return os.environ.get("GOOGLE_API_KEY")

    async def chat_completions(self, raycast_data: dict):
        model_name = raycast_data["model"]
        model = genai.GenerativeModel(model_name)
        google_message = []
        temperature = os.environ.get("TEMPERATURE", 0.5)
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
                max_output_tokens=MAX_TOKENS,
                temperature=temperature,
            ),
        )

    def get_models(self):
        default_models = _get_default_model_dict("gemini-pro")
        models = [
            {
                "id": "gemini-pro",
                "model": "gemini-pro",
                "name": "Gemini Pro",
                "provider": "google",
                "provider_name": "Google",
                "provider_brand": "google",
                "context": 16,
                **_get_model_extra_info(),
            },
            {
                "id": "gemini-1.5-pro",
                "model": "gemini-1.5-pro-latest",
                "name": "Gemini 1.5 Pro",
                "provider": "google",
                "provider_name": "Google",
                "provider_brand": "google",
                "context": 16,
                **_get_model_extra_info(),
            },
        ]
        return {"default_models": default_models, "models": models}


class AnthropicChatBot(ChatBotAbc):
    @classmethod
    def is_start_available(cls):
        return os.environ.get("ANTHROPIC_API_KEY")

    def __init__(self) -> None:
        super().__init__()
        logger.info("Using Anthropic API")
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    async def chat_completions(self, raycast_data: dict):
        messages = self.__build_anthropic_messages(raycast_data)
        model = raycast_data["model"]
        temperature = os.environ.get("TEMPERATURE", 0.5)

        try:
            response = await self.anthropic_client.messages.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
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
            content = {}
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
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.8,
                stream=True,
            )
            async for chunk in response:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
        except Exception as e:
            logger.error(f"Anthropic translation error: {e}")
            yield f"Error: {str(e)}"

    def get_models(self):
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
if GeminiChatBot.is_start_available():
    logger.info("Google API is available")
    _bot = GeminiChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    DEFAULT_MODELS = _models["default_models"]
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})
if OpenAIChatBot.is_start_available():
    logger.info("OpenAI API is available")
    _bot = OpenAIChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    DEFAULT_MODELS.update(_models["default_models"])
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})
if AnthropicChatBot.is_start_available():
    logger.info("Anthropic API is available")
    _bot = AnthropicChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    DEFAULT_MODELS.update(_models["default_models"])
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})


def get_bot(model_id):
    if not model_id:
        return next(iter(MODELS_DICT.values()))
    return MODELS_DICT.get(model_id)
