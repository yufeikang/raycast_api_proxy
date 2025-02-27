import abc
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def _get_default_model_dict(model_name: str) -> Dict[str, str]:
    return {
        "chat": model_name,
        "quick_ai": model_name,
        "commands": model_name,
        "api": model_name,
        "emoji_search": model_name,
        "tools": model_name,
    }


def _get_model_extra_info(name="") -> Dict:
    """
    Get extra model information based on model name
    """
    ext = {
        "description": "model description",
        "requires_better_ai": True,
        "features": ["chat", "quick_ai", "commands", "api", "emoji_search"],
        "suggestions": ["chat", "quick_ai", "commands", "api", "emoji_search"],
        "capabilities": {},
        "abilities": {},
        "availability": "public",
        "status": None,
        "speed": 3,
        "intelligence": 3,
    }
    if "gpt-4" in name:
        ext["capabilities"] = {
            "web_search": "full",
            "image_generation": "full",
        }
        ext["abilities"] = {
            "web_search": {
                "toggleable": True,
            },
            "image_generation": {
                "model": "dall-e-3",
            },
            "system_message": {
                "supported": True,
            },
            "temperature": {
                "supported": True,
            },
            "tools": {
                "supported": True,
            },
        }
    # o1 models don't support system_message and temperature
    if "o1" in name:
        ext["abilities"] = {
            "system_message": {
                "supported": False,
            },
            "temperature": {
                "supported": False,
            },
        }
    if "gemini" in name:
        ext["abilities"] = {
            "system_message": {
                "supported": True,
            },
            "temperature": {
                "supported": True,
            },
        }
        if "gemini-2" in name:
            ext["abilities"]["web_search"] = {"toggleable": True}
    return ext


class ApiProviderAbc(abc.ABC):
    api_type = None

    @classmethod
    @abc.abstractmethod
    def is_start_available(cls):
        pass

    @abc.abstractmethod
    async def chat_completions(self, raycast_data: dict):
        pass

    @abc.abstractmethod
    async def translate_completions(self, raycast_data: dict):
        pass

    async def get_models(self):
        pass
