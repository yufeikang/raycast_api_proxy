import abc
import json
import logging
import os
from pathlib import Path

import google.generativeai as genai
import httpx
import openai
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from google.generativeai import GenerativeModel

from app.utils import ProxyRequest, pass_through_request

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

app = FastAPI()

logger = logging.getLogger("proxy")

http_client = httpx.AsyncClient()

USER_SESSION = {}  # bearer token -> user email
ALLOWED_USERS = (
    os.environ.get("ALLOWED_USERS").split(",")
    if os.environ.get("ALLOWED_USERS", "")
    else None
)

MAX_TOKENS = os.environ.get("MAX_TOKENS", 1024)


def _get_default_model_dict(model_name: str):
    return {
        "chat": model_name,
        "quick_ai": model_name,
        "commands": model_name,
        "api": model_name,
        "emoji_search": model_name,
    }


def _get_model_extra_info():
    return {
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
        async for choice, error in self.__chat(openai_messages, model, temperature):
            if error:
                error_message = (
                    error.body.get("message", {}) if error.body else error.message
                )
                yield f'data: {json.dumps({"text":error_message, "finish_reason":"error"})}\n\n'
                return
            if choice.finish_reason is not None:
                yield f'data: {json.dumps({"text": "", "finish_reason": choice.finish_reason})}\n\n'
            if choice.delta and choice.delta.content:
                yield f'data: {json.dumps({"text": choice.delta.content})}\n\n'

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

    async def __chat(self, messages, model, temperature):
        try:
            stream = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                n=1,
                temperature=temperature,
                stream=True,
            )
        except openai.OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
            yield None, e
            return
        async for chunk in stream:
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
                **_get_model_extra_info(),
            },
            {
                "id": "openai-gpt-4o",
                "model": "gpt-4o",
                "name": "GPT-4o",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 8,
                **_get_model_extra_info(),
            },
            {
                "id": "openai-gpt-4-turbo",
                "model": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "openai",
                "provider_name": "OpenAI",
                "provider_brand": "openai",
                "context": 8,
                **_get_model_extra_info(),
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
                yield f'data: {json.dumps({"text": chunk.text})}\n\n'
        except genai.types.BlockedPromptException as e:
            logger.debug(f"Gemini response finish: {e}")
            yield f'data: {json.dumps({"text": "", "finish_reason": e})}\n\n'

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
        status_code=200, content=json.dumps(res), media_type="application/json"
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
        data["publishing_bot"] = True
        data["has_pro_features"] = True
        data["has_better_ai"] = True
        data["can_upgrade_to_pro"] = False
        data["admin"] = True
        add_user(request, data["email"])
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
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
        data.update({"default_models": DEFAULT_MODELS, "models": MODELS_AVAILABLE})
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return Response(
        status_code=response.status_code,
        content=content,
        headers=response.headers,
    )


# pass through all other requests
@app.api_route("/{path:path}")
async def proxy_options(request: Request, path: str):
    logger.info(f"Received request: {request.method} {path}")
    headers = {key: value for key, value in request.headers.items()}
    url = str(request.url)
    # add https when running via https gateway
    if "https://" not in url:
        url = url.replace("http://", "https://")
    req = ProxyRequest(
        str(request.url),
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
