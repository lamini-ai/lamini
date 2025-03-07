import logging
from typing import Union, List, Dict
import os
from pydantic import BaseModel

from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_agentic_pipeline import BaseAgenticPipeline
from lamini.api.openai_client import BaseOpenAIClient


class BaseMemoryExperiment:
    """Base class for running experiments with memory-enabled LLM pipelines.

    This class provides a framework for executing experiments that combine agentic pipelines
    with memory capabilities (like Memory Tuning or Memory RAG). It handles pipeline execution,
    result recording, and client management.

    Attributes:
        agentic_pipeline (BaseAgenticPipeline): The pipeline to execute
        record_dir (str): Directory for storing experiment results
        model (str): The memory model identifier
        client (BaseOpenAIClient): Client for LLM interactions
    """

    def __init__(
        self,
        agentic_pipeline: BaseAgenticPipeline,
        model: str = None,  # "memory" model for MemoryTuning or MemoryRAG
        client: BaseOpenAIClient = None,
        record_dir: str = None,
    ):

        self.agentic_pipeline = agentic_pipeline
        if record_dir:
            self.record_dir = record_dir
            self.agentic_pipeline.record_dir = record_dir
        else:
            # If None, just inherit the record_dir from the agentic pipeline
            self.record_dir = self.agentic_pipeline.record_dir

        self.model = model

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        if client is None:
            api_key = os.getenv("LAMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set LAMINI_API_KEY environment variable")

            api_url = "https://app.lamini.ai"
            client = BaseOpenAIClient(api_url=api_url, api_key=api_key)
        self.client = client

    def get_response_schema(self, output_type: Union[BaseModel, Dict, None]):
        """Generate a JSON schema for validating structured outputs.

        Converts Pydantic models or dictionary specifications into JSON schemas
        that can be used to validate LLM outputs.

        Args:
            output_type (Union[BaseModel, Dict, None]): The output structure specification.
                Can be a Pydantic model, dict of field definitions, or None.

        Returns:
            Optional[dict]: A JSON schema for output validation, or None if no type specified.

        Example:
            >>> schema = experiment.get_response_schema(MyPydanticModel)
            >>> schema = experiment.get_response_schema({"field1": "str", "field2": "int"})
        """
        if output_type is None:
            return None

        # Check if output_type is BaseModel or inherits from it
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            output_type_fields = output_type.__fields__
        else:
            output_type_fields = output_type

        # Create response schema from output_type
        response_schema = {
            "type": "object",
            "properties": {
                k: {"type": "string" if v == "str" else v}
                for k, v in output_type_fields.items()
            },
            "required": list(output_type_fields.keys()),
        }
        return response_schema

    def __call__(
        self, prompt_obj: Union[PromptObject, List[PromptObject]], debug: bool = False
    ):
        """Execute the memory experiment on one or more prompt objects.

        Runs the configured agentic pipeline with memory capabilities on the provided
        inputs, with optional debug logging.

        Args:
            prompt_obj (Union[PromptObject, List[PromptObject]]): The input(s) to process.
                Can be either a single PromptObject or a list for batch processing.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.

        Returns:
            List[PromptObject]: The results of the experiment execution.

        Note:
            Results are automatically recorded if the pipeline has recording enabled.
        """
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.debug(f"Running Memory Experiment...")

        results = self.agentic_pipeline(prompt_obj, debug=debug)
        self.logger.debug(f"[Memory Experiment] Agentic Pipeline results: {results}")

        return results
