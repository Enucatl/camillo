import logging
import sys

from loguru import logger

from camillo.settings import settings


class InterceptHandler(logging.Handler):
    """Route standard logging records through Loguru.

    Third-party libraries still use the stdlib logging API, so this bridge keeps
    one configured sink and one formatting policy for the whole service.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib log record while preserving caller depth."""
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging() -> None:
    """Configure process logging with Loguru and stdlib interception."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(settings.log_level)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>",
    )
