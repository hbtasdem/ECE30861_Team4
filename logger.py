import os
import sys


# read env vars once
LOG_FILE = os.environ.get("LOG_FILE")
LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 0))

# open file in append mode


def log_error(message: str):
    if LOG_LEVEL >= LOG_LEVEL:
        with open(LOG_FILE, "a") as log_file:
            log_file.write(message)