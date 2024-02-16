# Turn of isort, because alphabetic order for the following imports causes circular dependency issues

# isort: off
from lamini.error import error

from lamini.runners.llama_v2_runner import LlamaV2Runner
from lamini.runners.basic_model_runner import BasicModelRunner
from lamini.runners.mistral_runner import MistralRunner
from lamini.api.lamini import Lamini
from lamini.classify.llama_classifier import LaminiClassifier, BinaryLaminiClassifier
from lamini.api.classifier import Classifier
from lamini.api.embedding import Embedding

import os

api_key = os.environ.get("LAMINI_API_KEY", None)
api_url = os.environ.get("LAMINI_API_URL", None)

MISSING_API_KEY_MESSAGE = """LAMINI_API_KEY not found.
Please set it as an environment variable LAMINI_API_KEY, set it as lamini.api_key, or set it in ~/.lamini/configure.yaml
Find your LAMINI_API_KEY at https://app.lamini.ai/account"""


max_workers = os.environ.get("LAMINI_MAX_WORKERS", 10)
batch_size = os.environ.get("LAMINI_BATCH_SIZE", 5)
retry = os.environ.get("LAMINI_RETRY", False)
