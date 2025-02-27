# Turn of isort, because alphabetic order for the following imports causes circular dependency issues

from importlib.metadata import version
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
static_batching = bool(os.environ.get("LAMINI_STATIC_BATCHING", True))
bypass_reservation = bool(os.environ.get("LAMINI_BYPASS_RESERVATION", True))
gate_pipeline_batch_completions = bool(
    os.environ.get("GATE_PIPELINE_BATCH_COMPLETIONS", True)
)

__version__ = version("lamini")

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
from lamini.api.memory_rag import MemoryRAG


from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_generator import BaseGenerator
from lamini.experiment.base_validator import BaseValidator
from lamini.experiment.base_agentic_pipeline import BaseAgenticPipeline
from lamini.experiment.base_memory_experiment import BaseMemoryExperiment
from lamini.experiment.memory_rag_experiment import MemoryRAGExperiment
from lamini.api.openai_client import BaseOpenAIClient

from lamini.experiment.generators import (
    QuestionToConceptGenerator,
    ConceptToSQLInterpretationGenerator,
    QuestionsToConceptsGenerator,
    SchemaToSQLGenerator,
    SQLDebuggerGenerator,
)

from lamini.experiment.validators import (
    FactualityValidator,
    SQLValidator,
    SQLScoreValidator,
)

from lamini.index.lamini_index import LaminiIndex
