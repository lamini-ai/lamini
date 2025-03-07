from typing import Union, List, Dict, Callable
import os
import asyncio
from pydantic import BaseModel

from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_agentic_pipeline import BaseAgenticPipeline
from lamini.index.lamini_index import LaminiIndex
from lamini.experiment.base_memory_experiment import BaseMemoryExperiment
from lamini.experiment.base_memory_rag import BaseMemoryRAG


class MemoryRAGExperiment(BaseMemoryExperiment):
    """Experiment class for running RAG (Retrieval-Augmented Generation) with memory capabilities.

    Extends BaseMemoryExperiment to provide RAG-specific functionality, including index
    building, similarity search, and memory-augmented evaluation.

    Args:
        agentic_pipeline (BaseAgenticPipeline): Pipeline to execute
        record_dir (str, optional): Directory for storing experiment results and index
        rag_keys (Dict[str, List[str]], optional): Keys to extract from pipeline steps
            for RAG index building. Format: {step_name: [key1, key2, ...]}
        model (str, optional): Default model to use for evaluation

    Example rag_keys:
        {
            "concept_generator_input": ["question"],
            "sql_generator_output": ["sql_query"]
        }
    """

    def __init__(
        self,
        agentic_pipeline: BaseAgenticPipeline,
        record_dir: str = None,
        rag_keys: Dict[
            str, List[str]
        ] = None,  # {"concept_generator_input": ["question"], "sql_generator_output": ["sql_query"]}
        model: str = None,
    ):

        self.rag_keys = rag_keys
        super().__init__(
            agentic_pipeline=agentic_pipeline,
            record_dir=record_dir,
        )

        self.init_memory_rag()

    def init_memory_rag(self):
        """Initialize the BaseMemoryRAG instance with the experiment's record directory."""
        self.memory_rag = BaseMemoryRAG(index_path=self.record_dir)

    def get_similar(self, query: str, k: int = 3):
        """Query the memory index for similar entries.

        Args:
            query (str): Text to find similar entries for
            k (int, optional): Number of similar entries to retrieve. Defaults to 3.

        Returns:
            List[str]: k most similar entries from the index
        """
        return self.memory_rag.query_memory_index(query, k=k)

    def evaluate(
        self,
        prompt: str,
        output_type: Union[BaseModel, Dict],
        model: str = None,
        retrieved_prompt_key: str = None,
        k: int = 3,
    ):
        """Execute a RAG-augmented evaluation using the memory index.

        Args:
            prompt (str): Input prompt to evaluate
            output_type (Union[BaseModel, Dict]): Expected output structure
            model (str, optional): Model to use. Falls back to instance default
            retrieved_prompt_key (str, optional): Key to replace with retrieved content
            k (int, optional): Number of similar entries to retrieve. Defaults to 3.

        Returns:
            dict: The structured response from the model

        Raises:
            ValueError: If no model is specified or available
        """
        if model is None:
            model = self.model
            if model is None:
                raise ValueError(
                    "Model not specified. Please specify a model or set the model in the MemoryRAGExperiment."
                )

        return self.memory_rag.memory_rag(
            prompt=prompt,
            model=model,
            output_type=output_type,
            retrieved_prompt_key=retrieved_prompt_key,
            k=k,
        )

    def __call__(
        self, prompt_obj: Union[PromptObject, List[PromptObject]], debug: bool = False
    ):
        """Execute the experiment and build the memory index.

        Runs the agentic pipeline on inputs and uses the results to build
        a RAG memory index for future queries.

        Args:
            prompt_obj (Union[PromptObject, List[PromptObject]]): Input(s) to process
            debug (bool, optional): Enable debug logging. Defaults to False.

        Returns:
            List[PromptObject]: Results from pipeline execution

        Note:
            After execution, use get_similar() or evaluate() to leverage the built index
        """
        results = super().__call__(prompt_obj, debug=debug)
        self.memory_rag.build_memory_index(results=results, rag_keys=self.rag_keys)
        self.logger.debug(
            f"Memory RAG index built. Run get_similar(your_string) to get similar embeddings. Run evaluate(your_string, model, output_type) to run your LLM with Memory RAG."
        )

        return results
