import typing
import unittest
from unittest.mock import MagicMock, patch

from raycast_proxy.models.gemini_provider import GeminiProvider


class TestGeminiChatBot(unittest.IsolatedAsyncioTestCase):
    @patch("google.genai.Client")
    async def test_greeting(self, mock_client_class):
        # Mock the model class and generation result
        class MockStreamResult:
            text = "Mock response"
            prompt_feedback = None

        model_instance = MagicMock()
        model_instance.generate_content_stream.return_value = [MockStreamResult()]

        models = MagicMock()
        models.models = model_instance

        mock_client = MagicMock()
        mock_client.models = model_instance
        mock_client_class.return_value = mock_client

        # Create bot instance with mock API key
        bot = GeminiProvider(api_key="mock_key")

        raycast_data = {
            "image_generation_tool": True,
            "locale": "en-JP",
            "messages": [
                {"author": "user", "content": {"text": "hello"}},
                {
                    "author": "assistant",
                    "content": {
                        "text": "Hello! I'm Gemini, a chatbot. How can I help you today?"
                    },
                },
                {"author": "user", "content": {"text": "I need help with something"}},
            ],
            "model": "gemini-1.5-pro-latest",
            "provider": "google",
            "source": "ai_chat",
            "system_instruction": "markdown",
            "temperature": 0.5,
            "web_search_tool": True,
        }

        # Run test
        responses = []
        async for response in bot.chat_completions(raycast_data):
            responses.append(response)

        # Verify the correct methods were called
        mock_client.models.generate_content_stream.assert_called_once()
        assert len(responses) > 0
        assert all("data:" in response for response in responses)
