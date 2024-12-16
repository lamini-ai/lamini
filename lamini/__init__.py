# Turn of isort, because alphabetic order for the following imports causes circular dependency issues

import os

api_key = os.environ.get("LAMINI_API_KEY", None)
api_url = os.environ.get("LAMINI_API_URL", None)

MISSING_API_KEY_MESSAGE = """LAMINI_API_KEY not found.
Please set it as an environment variable LAMINI_API_KEY, set it as lamini.api_key, or set it in ~/.lamini/configure.yaml
Find your LAMINI_API_KEY at https://app.lamini.ai/account"""

# When inference call failed, how much retry should we perform.
retry_limit = int(os.environ.get("LAMINI_RETRY_LIMIT", 3))

max_workers = int(os.environ.get("LAMINI_MAX_WORKERS", 4))
batch_size = int(os.environ.get("LAMINI_BATCH_SIZE", 5))
static_batching = bool(os.environ.get("LAMINI_STATIC_BATCHING", False))
bypass_reservation = bool(os.environ.get("LAMINI_BYPASS_RESERVATION", False))
gate_pipeline_batch_completions = bool(
    os.environ.get("GATE_PIPELINE_BATCH_COMPLETIONS", False)
)

__version__ = "3.2.3"

# isort: off

from lamini.api.lamini import Lamini
from lamini.api.classifier import Classifier
from lamini.api.embedding import Embedding
from lamini.api.model_downloader import ModelDownloader
from lamini.api.model_downloader import ModelType
from lamini.api.model_downloader import DownloadedModel
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.split_response_node import SplitResponseNode
from lamini.api.streaming_completion import StreamingCompletion