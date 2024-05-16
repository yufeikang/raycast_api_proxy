import logging
from dataclasses import dataclass
from typing import Any, Union

import httpx
from fastapi import HTTPException

logger = logging.getLogger("proxy")

RAYCAST_BACKEND = "https://d1w35p7klh9xkc.cloudfront.net"


@dataclass
class ProxyRequest:
    url: str
    method: str
    headers: dict
    body: bytes
    query_params: dict


@dataclass
class ProxyResponse:
    status_code: int
    content: bytes
    headers: dict


async def pass_through_request(client: httpx.AsyncClient, request: ProxyRequest):
    logger.info(f"Received request: {request.method} {request.url}")

    url = request.url

    # if host is not raycast_backend, replace it with raycast_backend
    # when use mitm tool, the host be replaced
    if not url.startswith(RAYCAST_BACKEND):
        url = url.replace(url.split("/")[2], RAYCAST_BACKEND.split("/")[2])

    if not url.startswith("https://"):
        url = url.replace("http://", "https://")

    logger.debug(f"Forwarding request to {url}")
    headers = request.headers
    # disable compression, in docker container, it will cause error, unknown reason
    headers["accept-encoding"] = "identity"
    headers["host"] = "backend.raycast.com"
    try:
        response = await client.request(
            request.method,
            url,
            headers=headers,
            data=request.body,
            params=request.query_params,
            timeout=60.0,
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail="Error occurred while forwarding request"
        )
    content = None
    if response.content is not None:
        content = response.content
        logger.debug(
            "Response %s, status code: %s, data=%s",
            url,
            response.status_code,
            content,
        )
    filtered_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key not in filtered_headers
    }
    return ProxyResponse(
        status_code=response.status_code, content=content, headers=response_headers
    )
