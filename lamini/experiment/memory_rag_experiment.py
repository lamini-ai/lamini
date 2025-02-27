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
        self.memory_rag = BaseMemoryRAG(index_path=self.record_dir)

    def get_similar(self, query: str, k: int = 3):
        return self.memory_rag.query_memory_index(query, k=k)

    def evaluate(
        self,
        prompt: str,
        output_type: Union[BaseModel, Dict],
        model: str = None,
        retrieved_prompt_key: str = None,
        k: int = 3,
    ):
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
        results = super().__call__(prompt_obj, debug=debug)
        self.memory_rag.build_memory_index(results=results, rag_keys=self.rag_keys)
        self.logger.debug(
            f"Memory RAG index built. Run get_similar(your_string) to get similar embeddings. Run evaluate(your_string, model, output_type) to run your LLM with Memory RAG."
        )

        return results
