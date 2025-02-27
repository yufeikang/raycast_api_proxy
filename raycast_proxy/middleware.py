import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class AuthState:
    """Encapsulates authentication state and operations"""

    allowed_users: Optional[List[str]] = None
    user_sessions: Dict[str, str] = field(default_factory=dict)  # token -> email
    allowed_tokens: List[str] = field(default_factory=list)
    forbidden_tokens: List[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Initialize allowed_users from config.yml or environment variables"""
        config_file = Path(os.environ.get("CONFIG_PATH", "config.yml"))
        if config_file.exists():
            try:
                config = yaml.safe_load(config_file.read_text())
                self.allowed_users = config.get("auth", {}).get("allowed_users")
                logger.info("Loaded allowed users from config file")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")

        # Fall back to environment variable if not set in config
        if not self.allowed_users and os.environ.get("ALLOWED_USERS"):
            self.allowed_users = os.environ.get("ALLOWED_USERS").split(",")
            logger.info("Loaded allowed users from environment variable")

    def is_token_forbidden(self, token: str) -> bool:
        return token in self.forbidden_tokens

    def is_token_allowed(self, token: str) -> bool:
        return token in self.allowed_tokens

    def add_forbidden_token(self, token: str) -> None:
        if token not in self.forbidden_tokens:
            self.forbidden_tokens.append(token)

    def add_allowed_token(self, token: str, email: str) -> None:
        if token not in self.allowed_tokens:
            self.allowed_tokens.append(token)
            self.user_sessions[token] = email

    def get_user_email(self, token: str) -> Optional[str]:
        return self.user_sessions.get(token)

    def requires_auth(self) -> bool:
        return self.allowed_users is not None


# Single instance of AuthState
auth_state = AuthState()


async def get_user_info_by_token(headers: Headers) -> Optional[str]:
    async with auth_state.lock:
        token = headers.get("Authorization", "").split(" ")[1]
        if token in auth_state.user_sessions:
            return auth_state.get_user_email(token)

        logger.info(f"Getting user info by token: {token}")
        headers = dict(headers)
        headers["accept-encoding"] = "identity"  # disable compression
        headers.pop("content-length", None)  # delete content-length

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://backend.raycast.com/api/v1/me",
                headers=headers,
            )
            if response.status_code != 200:
                logger.error(f"Failed to get user info: {response.status_code}")
                return None

            data = response.json()
            return data["email"]


def get_current_user_email(request: Request) -> Optional[str]:
    bearer_token = request.headers.get("Authorization", "").split(" ")[1]
    return auth_state.get_user_email(bearer_token)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not auth_state.requires_auth():
            return await call_next(request)

        authorization: str = request.headers.get("Authorization", "")
        if not authorization:
            return Response("Unauthorized", status_code=401)

        token = authorization.split(" ")[1]
        if not token:
            return Response("Unauthorized", status_code=401)

        if auth_state.is_token_forbidden(token):
            return Response("Forbidden", status_code=403)

        if not auth_state.is_token_allowed(token):
            logger.info(f"Checking auth for {token}")
            user_email = await get_user_info_by_token(request.headers)
            logger.info(f"User email: {user_email} for token {token}")

            if not user_email or user_email not in auth_state.allowed_users:
                logger.warning(f"User {user_email} is not allowed")
                auth_state.add_forbidden_token(token)
                return Response("Forbidden", status_code=403)

            logger.info(f"User {user_email} is allowed, adding to session")
            auth_state.add_allowed_token(token, user_email)

        response = await call_next(request)
        return response
