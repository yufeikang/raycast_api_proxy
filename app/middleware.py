import asyncio
import logging
import os

import httpx
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

USER_SESSION = {}  # bearer token -> user email
ALLOWED_USERS = (
    os.environ.get("ALLOWED_USERS").split(",")
    if os.environ.get("ALLOWED_USERS", "")
    else None
)
ALLOWED_USERS_TOKEN = []
FORBIDDEN_USERS_TOKEN = []


lock = asyncio.Lock()


async def get_user_info_by_token(headers: Headers):
    async with lock:
        token = headers.get("Authorization", "").split(" ")[1]
        if token in USER_SESSION:
            return USER_SESSION[token]
        logger.info(f"Getting user info by token: {token}")
        headers = dict(headers)
        headers["accept-encoding"] = "identity"  # disable compression
        httpx_client = httpx.AsyncClient()
        response = await httpx_client.get(
            f"https://backend.raycast.com/api/v1/me",
            headers=headers,
        )
        if response.status_code != 200:
            logger.error(f"Failed to get user info: {response.status_code}")
            return None
        data = response.json()
        return data["email"]


def get_current_user_email(request: Request):
    bearer_token = request.headers.get("Authorization", "").split(" ")[1]
    return USER_SESSION.get(bearer_token)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not ALLOWED_USERS:
            # No need to check auth
            response = await call_next(request)
            return response
        authorization: str = request.headers.get("Authorization")
        if not authorization:
            return Response("Unauthorized", status_code=401)

        token = authorization.split(" ")[1]
        if not token:
            return Response("Unauthorized", status_code=401)

        if token in FORBIDDEN_USERS_TOKEN:
            return Response("Forbidden", status_code=403)

        if token not in ALLOWED_USERS_TOKEN:
            logger.info(f"Checking auth for {token}")
            user_email = await get_user_info_by_token(request.headers)
            logger.info(f"User email: {user_email} for token {token}")
            if user_email not in ALLOWED_USERS:
                logger.warning(f"User {user_email} is not allowed")
                FORBIDDEN_USERS_TOKEN.append(token)
                return Response("Forbidden", status_code=403)
            logger.info(f"User {user_email} is allowed, adding to session")
            USER_SESSION[token] = user_email
            ALLOWED_USERS_TOKEN.append(token)

        response = await call_next(request)
        return response
