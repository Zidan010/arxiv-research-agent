"""
Application-wide logging configuration.

Called once at startup. Log level is controlled via the
LOG_LEVEL environment variable so verbosity can be adjusted per-environment without a code change.
"""

import logging
import sys

from config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    level_name = settings.LOG_LEVEL.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        # Fail loudly but not fatally on a typo'd LOG_LEVEL — default to INFO
        # rather than crashing app startup over a logging misconfiguration.
        logging.getLogger(__name__).warning(
            "Invalid LOG_LEVEL '%s'; falling back to INFO", settings.LOG_LEVEL
        )
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )