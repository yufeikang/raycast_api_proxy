import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import StreamingResponse

from app.middleware import AuthMiddleware
from app.models import DEFAULT_MODELS, MODELS_AVAILABLE, get_bot
from app.sync import router as sync_router
from app.utils import (
    ProxyRequest,
    json_dumps,
    pass_through_request,
    process_custom_mapping,
)

FORMAT = "%(asctime)-15s %(threadName)s %(filename)-15s:%(lineno)d %(levelname)-8s: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger(__package__).setLevel(os.environ.get("LOG_LEVEL", logging.DEBUG))

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(AuthMiddleware)

http_client = httpx.AsyncClient(verify=False)


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


app.include_router(sync_router, prefix="/api/v1/me")


@app.post("/api/v1/ai/chat_completions")
async def chat_completions(request: Request):
    raycast_data = await request.json()
    logger.debug(f"Received chat completion request: {raycast_data}")
    model_name = raycast_data.get("model")
    return StreamingResponse(
        get_bot(model_name).chat_completions(raycast_data=raycast_data),
        media_type="text/event-stream",
    )


@app.api_route("/api/v1/translations", methods=["POST"])
async def proxy_translations(request: Request):
    raycast_data = await request.json()
    result = []
    logger.debug(f"Received translation request: {raycast_data}")
    model_name = raycast_data.get("model")
    async for content in get_bot(model_name).translate_completions(
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


@app.websocket("/cable")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    target_ws = await websockets.connect(
        uri="wss://backend.raycast.com/cable",
    )

    async def forward(client, server):
        # Forward messages from client to server
        if isinstance(client, WebSocket):
            async for message in client.iter_text():
                if isinstance(client, websockets.WebSocketClientProtocol):
                    logger.info(f"Received message from remove server: {message}")
                    await server.send(message)
        elif isinstance(client, websockets.WebSocketClientProtocol):
            async for message in client:
                if isinstance(client, WebSocket):
                    logger.info(f"Received message from Raycast: {message}")
                    await server.send_text(message)

    # Run tasks for forwarding messages in both directions
    await asyncio.gather(forward(websocket, target_ws), forward(target_ws, websocket))


@app.get("/health")
async def health():
    return {"status": "ok"}


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
