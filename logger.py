# Your software must produce a log file stored in the location named
# in the environment bariable $LOG_FILE and using the verbosity level
# indicated in the environment variable $LOG_LEVEL
# 0 means silent
# 1 means informational messags 
# 2 means debug messages 
# Default log verbosity is 0
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# read env vars once
LOG_FILE = os.environ.get("LOG_FILE")
LOG_LEVEL = int(os.environ.get("LOG_LEVEL", 0)) # default to 0

# Handle invalid LOG_FILE input 
if LOG_FILE is None:
    sys.exit(1)
if not os.path.isfile(LOG_FILE):
    sys.exit(1)

# Start with a blank log file each time 
open(LOG_FILE, "w").close()

# Then open file in append mode and write message
# LOG_LEVEL 1 informational messages
def info(msg: str):
    if LOG_LEVEL >= 1:
        with open(LOG_FILE, "a") as log_file:
            log_file.write("Info:" + msg + "\n")

# LOG_LEVEL 2 debug messages
# This will include info and debug messages
def debug(msg: str):
    if LOG_LEVEL == 2:
        with open(LOG_FILE, "a") as log_file:
            log_file.write("Debug:" + msg + "\n")
