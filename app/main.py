import abc
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import httpx
import openai
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import StreamingResponse
from google.generativeai import GenerativeModel

from app.utils import (
    ProxyRequest,
    json_dumps,
    pass_through_request,
    process_custom_mapping,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


app = FastAPI()

logger = logging.getLogger("proxy")

http_client = httpx.AsyncClient(verify=False)

USER_SESSION = {}  # bearer token -> user email
ALLOWED_USERS = (
    os.environ.get("ALLOWED_USERS").split(",")
    if os.environ.get("ALLOWED_USERS", "")
    else None
)

MAX_TOKENS = os.environ.get("MAX_TOKENS", 1024)

SYNC_DIR = os.environ.get("SYNC_DIR", "./sync")
if not os.path.exists(SYNC_DIR):
    os.makedirs(SYNC_DIR)


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


def _get_bot(model_id):
    if not model_id:
        return next(iter(MODELS_DICT.values()))
    return MODELS_DICT.get(model_id)


def add_user(request: Request, user_email: str):
    bearer_token = request.headers.get("Authorization", "").split(" ")[1]
    if bearer_token not in USER_SESSION:
        logger.info(f"Adding user {user_email} to session")
        USER_SESSION[bearer_token] = user_email


def check_auth(request: Request):
    if not ALLOWED_USERS:
        return True
    bearer_token = request.headers.get("Authorization", "").split(" ")[1]
    if bearer_token not in USER_SESSION:
        logger.warn(f"User not in session: {bearer_token}")
        return False
    user_email = USER_SESSION[bearer_token]
    if user_email not in ALLOWED_USERS:
        logger.debug(f"Allowed users: {ALLOWED_USERS}")
        logger.warn(f"User not allowed: {user_email}")
        return False
    return True


def get_current_user_email(request: Request):
    bearer_token = request.headers.get("Authorization", "").split(" ")[1]
    return USER_SESSION.get(bearer_token)


def get_current_utc_time():
    # 获取当前UTC时间并转换为ISO 8601格式，末尾手动添加'Z'表示UTC时间
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


@app.post("/api/v1/ai/chat_completions")
async def chat_completions(request: Request):
    raycast_data = await request.json()
    if not check_auth(request):
        return Response(status_code=401)
    logger.debug(f"Received chat completion request: {raycast_data}")
    model_name = raycast_data.get("model")
    return StreamingResponse(
        _get_bot(model_name).chat_completions(raycast_data=raycast_data),
        media_type="text/event-stream",
    )


@app.api_route("/api/v1/translations", methods=["POST"])
async def proxy_translations(request: Request):
    raycast_data = await request.json()
    if not check_auth(request):
        return Response(status_code=401)
    result = []
    logger.debug(f"Received translation request: {raycast_data}")
    model_name = raycast_data.get("model")
    async for content in _get_bot(model_name).translate_completions(
        raycast_data=raycast_data
    ):
        result.append(content) if content else None
    translated_text = "".join(result)
    res = {"data": {"translations": [{"translatedText": translated_text}]}}
    return Response(
        status_code=200, content=json_dumps(res), media_type="application/json"
    )


@app.api_route("/api/v1/me", methods=["GET"])
async def proxy(request: Request):
    logger.info("Received request to /api/v1/me")
    headers = {key: value for key, value in request.headers.items()}
    req = ProxyRequest(
        str(request.url),
        request.method,
        headers,
        await request.body(),
        query_params=request.query_params,
    )
    response = await pass_through_request(http_client, req)
    content = response.content
    if response.status_code == 200:
        data = json.loads(content)
        data["eligible_for_pro_features"] = True
        data["has_active_subscription"] = True
        data["eligible_for_ai"] = True
        data["eligible_for_gpt4"] = True
        data["eligible_for_ai_citations"] = True
        data["eligible_for_developer_hub"] = True
        data["eligible_for_application_settings"] = True
        data["eligible_for_cloud_sync"] = True
        data["publishing_bot"] = True
        data["has_pro_features"] = True
        data["has_better_ai"] = True
        data["can_upgrade_to_pro"] = False
        data["admin"] = True
        add_user(request, data["email"])
        content = json_dumps(data, ensure_ascii=False).encode("utf-8")
        content = process_custom_mapping(content, req)
    return Response(
        status_code=response.status_code,
        content=content,
        headers=response.headers,
    )


@app.api_route("/api/v1/ai/models", methods=["GET"])
async def proxy_models(request: Request):
    logger.info("Received request to /api/v1/ai/models")
    headers = {key: value for key, value in request.headers.items()}
    req = ProxyRequest(
        str(request.url),
        request.method,
        headers,
        await request.body(),
        query_params=request.query_params,
    )
    response = await pass_through_request(http_client, req)
    content = response.content
    if response.status_code == 200:
        data = json.loads(content)
        data.update(
            {
                "default_models": DEFAULT_MODELS,
                "models": MODELS_AVAILABLE,
            }
        )
        content = json_dumps(data, ensure_ascii=False).encode("utf-8")
    return Response(
        status_code=response.status_code,
        content=content,
        headers=response.headers,
    )


@app.api_route("/api/v1/me/sync", methods=["GET"])
async def proxy_sync_get(request: Request, after: str = Query(None)):
    if not check_auth(request):
        return Response(status_code=401)
    email = get_current_user_email(request)
    target = f"{SYNC_DIR}/{email}.json"
    if os.path.exists(target):
        with open(f"./sync/{email}.json", "r") as f:
            data = json.loads(f.read())

        # https://backend.raycast.com/api/v1/me/sync?after=2024-02-02T02:27:01.141195Z

        if after:
            after_time = datetime.fromisoformat(after.replace("Z", "+00:00"))
            data["updated"] = [
                item
                for item in data["updated"]
                if datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00"))
                > after_time
            ]

        return Response(json.dumps(data))
    else:
        return Response(json.dumps({"updated": [], "updated_at": None, "deleted": []}))


@app.api_route("/api/v1/me/sync", methods=["PUT"])
async def proxy_sync_put(request: Request):
    if not check_auth(request):
        return Response(status_code=401)
    email = get_current_user_email(request)

    data = await request.body()

    target = f"{SYNC_DIR}/{email}.json"

    if not os.path.exists(target):
        # 移除 request.body 中的 deleted 字段
        data = json.loads(data)
        data["deleted"] = []
        updated_time = get_current_utc_time()
        data["updated_at"] = updated_time
        for item in data["updated"]:
            item["created_at"] = item["client_updated_at"]
            item["updated_at"] = updated_time
        data = json.dumps(data)
        with open(target, "w") as f:
            f.write(data)

    else:
        with open(target, "r") as f:
            old_data = json.loads(f.read())
        new_data = json.loads(data)
        # 查找 old_data["updated"] 字段中是否存在 id 与 new_data["deleted"] 字段的列表中的 id 相同的元素
        # 如果存在则将该元素从 old_data["updated"] 中移除
        cleaned_data_updated = [
            item
            for item in old_data["updated"]
            if item["id"] not in new_data["deleted"]
        ]

        updated_time = get_current_utc_time()

        for data in new_data["updated"]:
            data["created_at"] = data["client_updated_at"]
            data["updated_at"] = updated_time

        # 添加 new_data["updated"] 中的元素到 cleaned_data_updated
        cleaned_data_updated.extend(new_data["updated"])

        new_data = {
            "updated": cleaned_data_updated,
            "updated_at": updated_time,
            "deleted": [],
        }

        with open(target, "w") as f:
            f.write(json.dumps(new_data))

    return Response(json.dumps({"updated_at": updated_time}))


# pass through all other requests
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy_options(request: Request, path: str):
    logger.info(f"Received request: {request.method} {path}")
    headers = {key: value for key, value in request.headers.items()}
    url = str(request.url)
    # add https when running via https gateway
    if "https://" not in url:
        url = url.replace("http://", "https://")
    req = ProxyRequest(
        url,
        request.method,
        headers,
        await request.body(),
        query_params=request.query_params,
    )
    response = await pass_through_request(http_client, req)
    return Response(
        status_code=response.status_code,
        content=response.content,
        headers=response.headers,
    )


if __name__ == "__main__":
    import uvicorn

    current_dir = Path(__file__).parent.parent

    if os.environ.get("CERT_FILE") and os.environ.get("KEY_FILE"):
        ssl_cert_path = Path(os.environ.get("CERT_FILE"))
        ssl_key_path = Path(os.environ.get("KEY_FILE"))
    elif (current_dir / "cert").exists():
        ssl_cert_path = current_dir / "cert" / "backend.raycast.com.cert.pem"
        ssl_key_path = current_dir / "cert" / "backend.raycast.com.key.pem"
    else:
        ssl_cert_path = None
        ssl_key_path = None

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,
        ssl_certfile=ssl_cert_path,
        ssl_keyfile=ssl_key_path,
    )
