import typing
import unittest

from app.main import GeminiChatBot


class TestGeminiChatBot(unittest.IsolatedAsyncioTestCase):
    async def test_greeting(self):
        bot = GeminiChatBot()
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
        # assert res is Generator
        async for i in bot.chat_completions(raycast_data):
            print(i)
