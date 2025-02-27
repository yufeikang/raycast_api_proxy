import logging
import os
from pathlib import Path

import yaml

from .base import ApiProviderAbc, _get_default_model_dict
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider
from .anthropic_provider import AnthropicProvider

logger = logging.getLogger(__name__)

MODELS_DICT = {}
MODELS_AVAILABLE = []
DEFAULT_MODELS = {}
AVAILABLE_DEFAULT_MODELS = []


async def _add_available_model(api: ApiProviderAbc, default_model: str = None):
    global MODELS_DICT, MODELS_AVAILABLE, DEFAULT_MODELS, AVAILABLE_DEFAULT_MODELS

    models = await api.get_models()
    MODELS_AVAILABLE.extend(models["models"])
    AVAILABLE_DEFAULT_MODELS.append(models["default_models"])
    MODELS_DICT.update({model["model"]: api for model in models["models"]})

    if default_model and default_model in MODELS_DICT:
        DEFAULT_MODELS.update(_get_default_model_dict(default_model))
    else:
        DEFAULT_MODELS.update(next(iter(AVAILABLE_DEFAULT_MODELS)))


async def init_models():
    config_file = Path(os.environ.get("CONFIG_PATH", "config.yml"))
    if not config_file.exists():
        logger.info(f"Config file not found: {config_file}")
        await init_models_from_env()
        return
    config = yaml.load(config_file.read_text(), Loader=yaml.FullLoader)
    default_model = config.get("default_model")
    # get all implement for ProviderApiAbc
    impl = {cls.api_type: cls for cls in ApiProviderAbc.__subclasses__()}
    for model_config in config["models"]:
        api_type = model_config.get("api_type")
        try:
            logger.info(f"Init model: {model_config['provider_name']}")
            api = impl[api_type](
                **model_config["params"], provider=model_config["provider_name"]
            )
            await _add_available_model(api, default_model)

        except KeyError:
            logger.error(f"Unknown api type: {api_type}")
        except Exception as e:
            logger.error(f"Error init model: {model_config}, {e}")


async def init_models_from_env():
    logger.warning("Use config.yml, this method is deprecated")
    default_model = os.environ.get("DEFAULT_MODEL")
    if GeminiProvider.is_start_available():
        logger.info("Google API is available")
        _api = GeminiProvider()
        await _add_available_model(_api, default_model)
    if OpenAIProvider.is_start_available():
        logger.info("OpenAI API is available")
        _api = OpenAIProvider()
        await _add_available_model(_api, default_model)
    if AnthropicProvider.is_start_available():
        logger.info("Anthropic API is available")
        _api = AnthropicProvider()
        await _add_available_model(_api, default_model)


def get_bot(model_id):
    if not model_id:
        return next(iter(MODELS_DICT.values()))
    logger.debug(f"Getting bot for model: {model_id}")
    if model_id not in MODELS_DICT:
        logger.error(f"Model not found: {model_id}")
        return None
    return MODELS_DICT.get(model_id)
