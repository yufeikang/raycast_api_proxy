import json
import logging
import os
from pathlib import Path

import httpx
import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# get fastAPI logger
logger = logging.getLogger("proxy")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

# 在这里设置您的 OpenAI API 密钥
openai.api_key = os.environ["OPENAI_API_KEY"]

# 用于 HTTPS 的 SSL 证书和密钥文件路径


FORCE_MODEL = os.environ.get("FORCE_MODEL", None)


def modify_me_is_pro(content):
    data = json.loads(content)
    data["eligible_for_pro_features"] = True
    data["has_active_subscription"] = True
    data["publishing_bot"] = True
    return json.dumps(data)


MAP_RESPONSE_MODIFY = {"GET api/v1/me": modify_me_is_pro}


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


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    logger.info(f"Received request: {request.method} {path}")
    async with httpx.AsyncClient() as client:
        raycast_host = request.headers.get("host")
        url = f"https://{raycast_host}/{path}"
        logger.debug(f"Forwarding request to {url}")
        headers = {key: value for key, value in request.headers.items()}
        # disable compression, in docker container, it will cause error, unknown reason
        headers["accept-encoding"] = "identity"
        try:
            response = await client.request(
                request.method,
                url,
                headers=headers,
                data=await request.body(),
                params=request.query_params,
                timeout=60.0,
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500, detail="Error occurred while forwarding request"
            )

        content = response.content

        request_hash = f"{request.method.upper()} {path}"
        if request_hash in MAP_RESPONSE_MODIFY:
            logger.info(f"Modifying response for {request_hash}")
            content = MAP_RESPONSE_MODIFY[request_hash](content)
        logger.info(
            "Response %s, status code: %s, data=%s",
            path,
            response.status_code,
            content,
        )
        return content


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
