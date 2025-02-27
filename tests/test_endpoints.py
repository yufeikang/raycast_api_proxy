import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from raycast_proxy.main import chat_completions, init_models, proxy_models
from raycast_proxy.models import DEFAULT_MODELS, MODELS_AVAILABLE, get_bot


@pytest_asyncio.fixture(autouse=True)
async def mock_init_models():
    # Mock model initialization
    with patch("raycast_proxy.models.init_models"):
        yield


@pytest.fixture
def mock_bot():
    class MockBot:
        async def chat_completions(self, raycast_data):
            yield f'data: {json.dumps({"text": "Hello, how can I help you?"})}\n\n'
            yield f'data: {json.dumps({"text": "", "finish_reason": "stop"})}\n\n'

    return MockBot()


# Create test app without authentication middleware
app = FastAPI()
app.add_api_route("/api/v1/ai/models", proxy_models, methods=["GET"])
app.add_api_route("/api/v1/ai/chat_completions", chat_completions, methods=["POST"])

client = TestClient(app)


@pytest.fixture
def mock_openai_response():
    return {
        "data": [
            {
                "id": "gpt-4",
                "created": 1687882410,
                "object": "model",
                "owned_by": "openai",
            },
            {
                "id": "gpt-3.5-turbo",
                "created": 1677610602,
                "object": "model",
                "owned_by": "openai",
            },
        ]
    }


@pytest.fixture
def mock_chat_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello, how can I help you?"},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.mark.asyncio
async def test_models_endpoint(mock_openai_response):
    async def mock_request(*args, **kwargs):
        response = AsyncMock()
        response.status_code = 200
        response.content = json.dumps(mock_openai_response).encode()
        response.headers = {}
        return response

    with patch("httpx.AsyncClient.request", side_effect=mock_request):
        response = client.get("/api/v1/ai/models")

        assert response.status_code == 200
        data = response.json()
        assert "default_models" in data
        assert "models" in data
        assert data["default_models"] == DEFAULT_MODELS
        assert data["models"] == MODELS_AVAILABLE


@pytest.mark.asyncio
async def test_chat_completions_endpoint(mock_chat_response, mock_bot):
    chat_request = {
        "messages": [{"author": "user", "content": {"text": "Hello"}}],
        "model": "gpt-3.5-turbo",
        "provider": "openai",
    }

    with patch("raycast_proxy.main.get_bot", return_value=mock_bot):

        response = client.post("/api/v1/ai/chat_completions", json=chat_request)

        assert response.status_code == 200
        content = b"".join(response.iter_bytes())
        assert b"Hello, how can I help you?" in content
        assert b"finish_reason" in content


@pytest.mark.asyncio
async def test_chat_completions_error_handling():
    chat_request = {
        "messages": [{"author": "user", "content": {"text": "Hello"}}],
        "model": "invalid-model",
        "provider": "openai",
    }

    # Mock get_bot to return None for invalid model
    with patch("raycast_proxy.main.get_bot", return_value=None):
        response = client.post("/api/v1/ai/chat_completions", json=chat_request)

        # Should return 500 with detail message for invalid model
        assert response.status_code == 500
        assert response.json() == {"detail": "Model not found: invalid-model"}


@pytest.mark.asyncio
async def test_models_error_handling():
    async def mock_request(*args, **kwargs):
        response = AsyncMock()
        response.status_code = 500
        response.content = b'{"error": "Internal Server Error"}'
        response.headers = {}
        return response

    with patch("httpx.AsyncClient.request", side_effect=mock_request):

        response = client.get("/api/v1/ai/models")

        assert response.status_code == 500
        assert "error" in response.json()
