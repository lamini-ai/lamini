from llama import *

import os

api_key = os.environ.get("LAMINI_API_KEY", None)


MISSING_API_KEY_MESSAGE = """LAMINI_API_KEY not found.
Please set it as an environment variable LAMINI_API_KEY, set it as lamini.api_key, or set it in ~/.lamini/configure.yaml
Find your LAMINI_API_KEY at https://app.lamini.ai/account"""


max_workers = os.environ.get("LAMINI_MAX_WORKERS", 12)
batch_size = os.environ.get("LAMINI_BATCH_SIZE", 4)
