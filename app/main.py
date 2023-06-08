import json
import logging
import os
from pathlib import Path

import httpx
import openai
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from app.http import ProxyRequest, pass_through_request

logging.basicConfig(level=logging.INFO)

app = FastAPI()

logger = logging.getLogger("proxy")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

http_client = httpx.AsyncClient()


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()


openai.api_key = os.environ["OPENAI_API_KEY"]
FORCE_MODEL = os.environ.get("FORCE_MODEL", None)


@app.post("/api/v1/ai/chat_completions")
async def chat_completions(request: Request):
    raycast_data = await request.json()
    logger.info(f"Received chat completion request: {raycast_data}")
    openai_messages = []
    temperature = os.environ.get("TEMPERATURE", 0.5)
    for msg in raycast_data["messages"]:
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
            max_tokens=150,
            n=1,
            stop=None,
            temperature=temperature,
            stream=True,
        ):
            chunk = response["choices"][0]
            if "finish_reason" in chunk:
                yield f'data: {json.dumps({"text": "", "finish_reason": "stop"})}\n\n'
            if "content" in chunk["delta"]:
                yield f'data: {json.dumps({"text": chunk["delta"]["content"]})}\n\n'

    return StreamingResponse(openai_stream(), media_type="text/event-stream")


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
        data["publishing_bot"] = True
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
