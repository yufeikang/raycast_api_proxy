import os
import sys
import inspect
import logging

from loguru import logger


# Intercept standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


log_level = os.environ.get("LOG_LEVEL", "DEBUG")

# Configure standard logging to use our interceptor
# This helps with third-party libraries that use standard logging
logging.basicConfig(
    handlers=[InterceptHandler()],
    level=log_level,
    force=True,
)

# Add a new handler with desired format
logger.remove()
logger.add(
    sys.stderr,
    level=log_level,
    colorize=True,
)
