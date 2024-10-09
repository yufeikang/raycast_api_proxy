import abc
import json
import logging
import os

import anthropic
import google.generativeai as genai
import openai
from google.generativeai import GenerativeModel

from app.utils import (
    json_dumps,
    get_file_info, 
    generate_file_url,
    logger,
)

logger = logging.getLogger(__name__)

# 尝试将环境变量转换为整数，如果失败则使用默认值 1024
try:
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
except ValueError:
    MAX_TOKENS = 1024
    logger.warning("环境变量 MAX_TOKENS 不是有效的整数，使用默认值 1024")
    
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL")


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
    # 定义所有模型的默认属性
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

    # 定义各个模型的特定属性
    model_info = {
        "gpt-3.5-turbo": {
            "description": (
                "GPT-3.5 Turbo is OpenAI’s fastest model, making it ideal for tasks that require quick "
                "response times with basic language processing capabilities.\n"
            ),
            "speed": 2,
            "intelligence": 2,
        },
        "gpt-4-turbo": {
            "description": (
                "The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and "
                "function calling.\n"
            ),
            "speed": 0,
            "intelligence": 4,
            "capabilities": {
                "web_search": "full",
                "image_generation": "full",
            },
            "abilities": {
                "web_search": {"toggleable": True},
                "image_generation": {"model": "dall-e-3"},
                "vision": {
                    "formats": [
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                    ],
                },
            },
        },
        "gpt-4o": {
            "description": (
                "GPT-4o is the most advanced and fastest model from OpenAI, making it a great choice for "
                "complex everyday problems and deeper conversations.\n"
            ),
            "speed": 2,
            "intelligence": 5,
            "capabilities": {
                "web_search": "full",
                "image_generation": "full",
            },
            "abilities": {
                "web_search": {"toggleable": True},
                "image_generation": {"model": "dall-e-3"},
                "vision": {
                    "formats": [
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                    ],
                },
            },
        },
        "gpt-4o-mini": {
            "description": (
                "GPT-4o mini is a highly intelligent and fast model that is ideal for a variety of everyday tasks.\n"
            ),
            "requires_better_ai": False,
            "speed": 2,
            "intelligence": 4,
            "capabilities": {
                "web_search": "full",
                "image_generation": "full",
            },
            "abilities": {
                "web_search": {"toggleable": True},
                "image_generation": {"model": "dall-e-3"},
                "vision": {
                    "formats": [
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                    ],
                },
            },
        },
        "o1-preview": {
            "description": (
                "o1-preview is a reasoning model designed to solve hard problems across domains. "
                "These models think before they answer, producing a long internal chain of thought before responding to the user.\n"
            ),
            "speed": 1,
            "intelligence": 5,
        },
        "o1-mini": {
            "description": (
                "o1-mini is a faster and cheaper reasoning model particularly good at coding, math, and science. "
                "These models think before they answer, producing a long internal chain of thought before responding to the user.\n"
            ),
            "speed": 2,
            "intelligence": 4,
        },
    }

    # 更新默认属性，如果模型有特定属性
    if name in model_info:
        ext.update(model_info[name])

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
            # Initialize message content for the current message
            message_content = []

            # Handle text content
            if "text" in msg["content"]:
                message_content.append(
                    {"type": "text", "text": msg["content"]["text"]}
                )

            # Handle attachments
            if "attachments" in msg["content"]:
                for attachment in msg["content"]["attachments"]:
                    attachment_id = attachment.get("id")
                    attachment_type = attachment.get("type")

                    if attachment_type == "file" and attachment_id:
                        # Get the file information using the attachment ID
                        file_info = get_file_info(attachment_id)
                        if file_info:
                            # Generate the file URL
                            file_url = generate_file_url(file_info['key'])
                            # Append the image URL to the message content
                            message_content.append({
                                "type": "image_url",
                                "image_url": {"url": file_url}
                            })
                        else:
                            logger.error(f"File with id {attachment_id} not found.")

            # If there's any content to send, add it to openai_messages
            if message_content:
                openai_messages.append({
                    "role": msg["author"],
                    "content": message_content
                })
            
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

    async def __chat(self, messages, model, temperature, **kwargs):
        if "tools" in kwargs and not kwargs["tools"]:
            # pop tools from kwargs, empty tools will cause error
            kwargs.pop("tools")
        # stream = "tools" not in kwargs
        stream = "stream" in kwargs and kwargs["stream"]
        try:
            logger.debug(f"openai chat stream: {stream}")
            res = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                n=1,
                temperature=temperature,
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

    def get_models(self):
        default_models = _get_default_model_dict("openai-gpt-4o-mini")
        models = [
            {
                "id": "openai-gpt-3.5-turbo",
                "model": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 16,  # 16,000 tokens
                **_get_model_extra_info("gpt-3.5-turbo"),
            },
            {
                "id": "openai-gpt-4o-mini",
                "model": "gpt-4o-mini",
                "name": "GPT-4o Mini",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 127,  # 127,000 tokens
                **_get_model_extra_info("gpt-4o-mini"),
            },
            {
                "id": "openai-gpt-4o",
                "model": "gpt-4o",
                "name": "GPT-4o",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 127,  # 127,000 tokens
                **_get_model_extra_info("gpt-4o"),
            },
            {
                "id": "openai-gpt-4-turbo",
                "model": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 127,  # 127,000 tokens
                **_get_model_extra_info("gpt-4-turbo"),
            },
            # 添加 o1 系列模型
            {
                "id": "openai-o1-preview",
                "model": "o1-preview",
                "name": "o1-preview",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 127,  # 127,000 tokens
                **_get_model_extra_info("o1-preview"),
            },
            {
                "id": "openai-o1-mini",
                "model": "o1-mini",
                "name": "o1-mini",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 127,  # 127,000 tokens
                **_get_model_extra_info("o1-mini"),
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
AVAILABLE_DEFAULT_MODELS = []
if GeminiChatBot.is_start_available():
    logger.info("Google API is available")
    _bot = GeminiChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    AVAILABLE_DEFAULT_MODELS.append(_models["default_models"])
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})
if OpenAIChatBot.is_start_available():
    logger.info("OpenAI API is available")
    _bot = OpenAIChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    AVAILABLE_DEFAULT_MODELS.append(_models["default_models"])
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})
if AnthropicChatBot.is_start_available():
    logger.info("Anthropic API is available")
    _bot = AnthropicChatBot()
    _models = _bot.get_models()
    MODELS_AVAILABLE.extend(_models["models"])
    AVAILABLE_DEFAULT_MODELS.append(_models["default_models"])
    MODELS_DICT.update({model["model"]: _bot for model in _models["models"]})


DEFAULT_MODELS = next(iter(AVAILABLE_DEFAULT_MODELS))
if DEFAULT_MODEL and DEFAULT_MODEL in MODELS_DICT:
    DEFAULT_MODELS = MODELS_DICT[DEFAULT_MODEL]
    logger.info(f"Using default model: {DEFAULT_MODEL}")


def get_bot(model_id):
    if not model_id:
        return next(iter(MODELS_DICT.values()))
    logger.debug(f"Getting bot for model: {model_id}")
    if model_id not in MODELS_DICT:
        logger.error(f"Model not found: {model_id}")
        return None
    return MODELS_DICT.get(model_id)
