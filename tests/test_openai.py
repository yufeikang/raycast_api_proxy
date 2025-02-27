import unittest

from raycast_proxy.models.openai_provider import OpenAIProvider


class TestOpenAIChatBot(unittest.IsolatedAsyncioTestCase):
    async def test_greeting(self):
        bot = OpenAIProvider()
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
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "source": "ai_chat",
            "system_instruction": "markdown",
            "temperature": 0.5,
            "web_search_tool": True,
        }

        async for i in bot.chat_completions(raycast_data):
            print(i)
