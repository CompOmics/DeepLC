"""Main command line interface to DeepLC."""

import logging
import sys

LOGGER = logging.getLogger(__name__)

# TODO: Add CLI functionality

def _setup_logging(passed_level):
    log_mapping = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if passed_level.lower() not in log_mapping:
        print(
            "Invalid log level. Should be one of the following: ",
            ", ".join(log_mapping.keys()),
        )
        exit(1)

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_mapping[passed_level.lower()],
    )
