from typing import Union, List, Dict, Callable
import os
import asyncio
from pydantic import BaseModel
import logging

from lamini.generation.base_prompt_object import PromptObject
from lamini.index.lamini_index import LaminiIndex
from lamini.api.openai_client import BaseOpenAIClient


class BaseMemoryRAG:
    """Base class for implementing RAG (Retrieval-Augmented Generation) with memory capabilities.

    This class provides functionality to build and query a memory index for RAG operations,
    process results into text, and execute memory-augmented generation tasks.

    Attributes:
        index_path (str): Path to store/load the memory index
        client (BaseOpenAIClient): Client for LLM interactions
        memory_rag_index (LaminiIndex): The constructed memory index for RAG operations
    """

    def __init__(
        self,
        index_path: str = None,
        client: BaseOpenAIClient = None,
    ):
        self.index_path = index_path

        if index_path is None:
            self.index_path = os.getcwd()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        if client is None:
            api_key = os.getenv("LAMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set LAMINI_API_KEY environment variable")

            api_url = "https://app.lamini.ai"
            client = BaseOpenAIClient(api_url=api_url, api_key=api_key)
        self.client = client

    def _process_step_to_text(
        self, prompt_obj: PromptObject, rag_keys: Dict[str, List[str]] = None
    ):
        """Convert a single prompt object's data into text format for indexing.

        Args:
            prompt_obj (PromptObject): The prompt object to process
            rag_keys (Dict[str, List[str]], optional): Specific keys to extract from data.
                Format: {step_name: [key1, key2, ...]}

        Returns:
            List[str]: Single-element list containing the formatted text
        """
        rag_data_text = ""
        if rag_keys:
            for step_name, keys in rag_keys.items():
                rag_data = prompt_obj.data[step_name]
                for key in keys:
                    if key in rag_data:
                        rag_data_text += f"{key.title()}: {rag_data[key]}\n"
        else:
            for key, values_dict in prompt_obj.data.items():
                if key.endswith("_input") or key.endswith("_output"):
                    for k, v in values_dict.items():
                        rag_data_text += f"{k.title()}: {v}\n"

        print("rag_data_text", rag_data_text)
        return [f"{rag_data_text}"]

    def process_results_to_text(
        self, results: List[PromptObject], rag_keys: Dict[str, List[str]] = None
    ):
        """Convert multiple prompt objects' data into text format for indexing.

        Args:
            results (List[PromptObject]): List of prompt objects to process
            rag_keys (Dict[str, List[str]], optional): Specific keys to extract from data

        Returns:
            List[List[str]]: Processed text data for each prompt object
        """
        return [self._process_step_to_text(result, rag_keys) for result in results]

    def build_memory_index(
        self,
        results: List[PromptObject],
        save_index: bool = True,
        results_to_texts_function: Callable = None,
        rag_keys: Dict[str, List[str]] = None,
    ):
        """Build and optionally save a memory index from prompt objects.

        Args:
            results (List[PromptObject]): Prompt objects to index
            save_index (bool, optional): Whether to save the index to disk. Defaults to True.
            results_to_texts_function (Callable, optional): Custom function to convert results to text.
                Defaults to self.process_results_to_text.
            rag_keys (Dict[str, List[str]], optional): Specific keys to extract from data

        Note:
            The index is saved to {index_path}/memory_index if save_index is True
        """
        self.logger.info(f"Building Memory RAG index...")

        if results_to_texts_function is None:
            results_to_texts_function = self.process_results_to_text

        if rag_keys is None:
            rag_texts = results_to_texts_function(results)
        else:
            rag_texts = results_to_texts_function(results, rag_keys)

        index = LaminiIndex()
        self.memory_rag_index = index.build_index(rag_texts)

        self.logger.info(f"Memory RAG index built.")

        if save_index:
            self.logger.info(f"Saving index to {self.index_path}/memory_index")
            os.makedirs(f"{self.index_path}/memory_index", exist_ok=True)
            self.memory_rag_index.save_index(f"{self.index_path}/memory_index")

    def query_memory_index(self, query_text: str, k: int = 3):
        """Query the memory index for similar entries.

        Args:
            query_text (str): The text to find similar entries for
            k (int, optional): Number of similar entries to retrieve. Defaults to 3.

        Returns:
            Optional[List[str]]: k most similar entries, or None if index not found

        Raises:
            Exception: If index cannot be loaded
        """
        if self.memory_rag_index is None:
            # Try to load from the default path
            self.logger.info(
                f"Memory RAG index not found. Trying to load from default path {self.index_path}/memory_index"
            )
            try:
                index = LaminiIndex()
                self.memory_rag_index = index.load_index(
                    f"{self.index_path}/memory_index"
                )
            except Exception as e:
                self.logger.error(
                    f"Memory RAG index not found. Please build the index first with build_memory_rag. Error: {e}"
                )
            return None

        query_embedding = self.memory_rag_index.get_embeddings(query_text)[0]
        similar = self.memory_rag_index.query_with_embedding(query_embedding, k=k)

        return similar

    def add_similar_to_prompt(
        self,
        prompt: str,
        similar: Union[str, List[str]],
        retrieved_prompt_key: str = None,
    ):
        """Augment a prompt with retrieved similar content.

        Args:
            prompt (str): The original prompt
            similar (Union[str, List[str]]): Similar content to add
            retrieved_prompt_key (str, optional): Key to replace in prompt with similar content.
                If None, appends to end of prompt.

        Returns:
            str: The augmented prompt
        """
        # Turn similar into a string
        if isinstance(similar, list):
            similar_str = "\n".join(similar)
        else:
            similar_str = similar

        if retrieved_prompt_key is None:
            # Append to prompt
            prompt += similar_str
        else:
            # Replace {retrieved_prompt_key} with similar
            prompt = prompt.replace(f"{{{retrieved_prompt_key}}}", similar_str)

    def get_response_schema(self, output_type: Union[BaseModel, Dict, None]):
        """Generate a JSON schema for validating structured outputs.

        Args:
            output_type (Union[BaseModel, Dict, None]): Output structure specification

        Returns:
            Optional[dict]: JSON schema for output validation, or None if no type specified
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

    def memory_rag(
        self,
        prompt: str,
        model: str,
        output_type: Union[BaseModel, Dict],
        retrieved_prompt_key: str = None,
        k: int = 3,
    ):
        """Execute a RAG operation with memory augmentation.

        Args:
            prompt (str): The input prompt
            model (str): The model to use for generation
            output_type (Union[BaseModel, Dict]): Expected output structure
            retrieved_prompt_key (str, optional): Key to replace with retrieved content
            k (int, optional): Number of similar entries to retrieve. Defaults to 3.

        Returns:
            dict: The structured response from the model
        """
        similar = self.query_memory_index(prompt, k)

        self.add_similar_to_prompt(prompt, similar, retrieved_prompt_key)

        # Use similar info in model inference
        response_schema = self.get_response_schema(output_type)
        response = asyncio.run(
            self.client.execute_completion(
                model=model,
                prompt=prompt,
                response_schema=response_schema,
            )
        )

        return response

    def __call__(
        self,
        prompt: str,
        model: str,
        output_type: Union[BaseModel, Dict],
        retrieved_prompt_key: str = None,
        k: int = 3,
    ):
        """Convenience method to execute memory RAG operation.

        Args:
            prompt (str): The input prompt
            model (str): The model to use for generation
            output_type (Union[BaseModel, Dict]): Expected output structure
            retrieved_prompt_key (str, optional): Key to replace with retrieved content
            k (int, optional): Number of similar entries to retrieve. Defaults to 3.

        Returns:
            dict: The structured response from the model
        """
        return self.memory_rag(
            prompt=prompt,
            model=model,
            output_type=output_type,
            retrieved_prompt_key=retrieved_prompt_key,
            k=k,
        )
