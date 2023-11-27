import json
import logging
import os
from pathlib import Path

import httpx
import openai
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

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


openai.api_key = os.environ["OPENAI_API_KEY"]
is_azure = openai.api_type in ("azure", "azure_ad", "azuread")
if is_azure:
    logger.info("Using Azure API")

FORCE_MODEL = os.environ.get("FORCE_MODEL", None)
AZURE_DEPLOYMENT_ID = os.environ.get("AZURE_DEPLOYMENT_ID", None)


@app.post("/api/v1/ai/chat_completions")
async def chat_completions(request: Request):
    raycast_data = await request.json()
    if not check_auth(request):
        return Response(status_code=401)
    logger.info(f"Received chat completion request: {raycast_data}")
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
        if "text" in msg["content"]:
            openai_messages.append({"role": "user", "content": msg["content"]["text"]})
        if "temperature" in msg["content"]:
            temperature = msg["content"]["temperature"]
    model = FORCE_MODEL or raycast_data["model"]

    def openai_stream():
        for response in openai.ChatCompletion.create(
            model=model,
            messages=openai_messages,
            deployment_id=AZURE_DEPLOYMENT_ID if is_azure else None,
            max_tokens=MAX_TOKENS,
            n=1,
            stop=None,
            temperature=temperature,
            stream=True,
        ):
            chunk = response["choices"][0]
            if "finish_reason" in chunk and chunk["finish_reason"] is not None:
                logger.debug(f"OpenAI response finish: {chunk['finish_reason']}")
                yield f'data: {json.dumps({"text": "", "finish_reason": "stop"})}\n\n'
            if "content" in chunk["delta"]:
                logger.debug(f"OpenAI response chunk: {chunk['delta']['content']}")
                yield f'data: {json.dumps({"text": chunk["delta"]["content"]})}\n\n'

    return StreamingResponse(openai_stream(), media_type="text/event-stream")

# Add an api route for /api/v1/status to just say OK
@app.api_route("/api/v1/status", methods=["GET"])
async def proxy_ok(request: Request):
    logger.info("Received request to /api/v1/status")
    data = {}
    data["status"] = "OK"
    content = json.dumps(data, ensure_ascii=False).encode("utf-8")
    return Response(
        status_code=200,
        content=content,
        headers={},
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

    # if no cert, run without ssl
    if not ssl_cert_path or not ssl_key_path:
        uvicorn.run(app, host="0.0.0.0", port=80)
    else:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=443,
            ssl_certfile=ssl_cert_path,
            ssl_keyfile=ssl_key_path,
        )
