from pydantic import BaseModel
from typing import Union, Dict, List, Optional, Iterator, AsyncIterator
import logging
import re
import warnings
import asyncio
import os

from lamini.generation.base_prompt_object import PromptObject
from lamini.api.openai_client import BaseOpenAIClient


class BaseGenerator:
    """A base class for implementing LLM-based generators in a pipeline.

    This class provides the foundation for creating generators that process input through
    language models, with support for structured input/output, templated instructions,
    and customizable pre/post processing.

    The generator handles:
    - Instruction template management with metadata injection
    - LLM client interaction and response processing
    - Structured output validation
    - Result storage and transformation
    - Error handling and logging
    """

    def __init__(
        self,
        name: str,
        instruction: str,
        client: BaseOpenAIClient = None,
        model: Optional[str] = None,
        role: Optional[str] = "",
        output_type: Union[BaseModel, Dict, None] = None,
        instruction_search_pattern: Optional[str] = r"\{(.*?)\}",
    ):
        if client is None:
            api_key = os.getenv("LAMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set LAMINI_API_KEY environment variable")

            api_url = "https://app.lamini.ai"
            client = BaseOpenAIClient(api_url=api_url, api_key=api_key)

        # Assign object properties
        self.name = name
        self.client = client
        self.model = model
        self.role = role
        self.instruction = instruction
        self.output_type = output_type

        self.input = {}
        self.output = {}

        if self.instruction:
            instruction_metadata_pattern = re.compile(
                instruction_search_pattern, re.DOTALL
            )
            self.metadata_keys = instruction_metadata_pattern.findall(self.instruction)

            # Warn users of a potential invalid instruction
            if not self.metadata_keys:
                warnings.warn(
                    f"No metadata keys were detected for {self.name}! Proceed knowing this generator will not be using any data associated with PromptObjects passed in execution!"
                )

            for key in self.metadata_keys:
                self.input[key] = "str"
        if self.output_type:
            self.output.update(self.output_type)

        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    def __call__(self, prompt_obj: PromptObject, debug=False, *args, **kwargs):
        """Execute the generator on a prompt object.

        Orchestrates the complete generation process: transform input, generate response,
        and process results.

        Args:
            prompt_obj (PromptObject): The input prompt to process
            debug (bool, optional): Enable debug logging. Defaults to False.
            *args, **kwargs: Additional arguments passed to generate()

        Returns:
            PromptObject: The processed prompt object with generation results
        """
        if debug:
            self.logger.setLevel(logging.DEBUG)
        prompt_obj = self.transform_prompt(prompt_obj)
        self.logger.debug(f"Prompt: {prompt_obj.prompt}")

        result = self.generate(prompt_obj, debug, *args, **kwargs)
        self.logger.debug(f"Response after generation: {result.response}")

        result = self.process_results(result)
        self.logger.debug(f"Result after post-processing: {result}")

        return result

    def get_response_schema(self, output_type: Union[BaseModel, Dict, None]):
        """Create a JSON schema for validating LLM output.

        Converts Pydantic models or dictionary specifications into JSON schemas
        that the LLM can use to structure its output.

        Args:
            output_type (Union[BaseModel, Dict, None]): Output structure specification

        Returns:
            Optional[dict]: JSON schema for output validation, or None if no type specified
        """
        if output_type is None:
            return None

        # Check if output_type is BaseModel or inherits from it
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            return output_type.model_json_schema()
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

    def generate(self, prompt_obj: PromptObject, debug=False):
        """Generate a response using the configured LLM.

        Executes the core generation logic by sending the prepared prompt to the LLM
        and handling the structured response.

        Args:
            prompt_obj (PromptObject): The prepared prompt object
            debug (bool, optional): Enable debug logging. Defaults to False.

        Returns:
            PromptObject: Prompt object with generated response

        Raises:
            Exception: If generation fails, with detailed error logging
        """
        if debug:
            self.logger.setLevel(logging.DEBUG)

        """Process a single data point through this step"""
        try:
            response_schema = self.get_response_schema(output_type=self.output_type)

            # Execute completion
            prompt_obj.response = asyncio.run(
                self.client.execute_completion(
                    model=self.model,
                    prompt=prompt_obj.prompt,
                    response_schema=response_schema,
                    schema_name=f"{self.name}_response",
                )
            )

            return prompt_obj

        except Exception as e:
            self.logger.error(f"Error in generator {self.name}: {str(e)}")
            raise

    def transform_prompt(
        self,
        prompt_obj: PromptObject,
        debug=False,
    ):
        """Prepare the prompt object for generation.

        Handles:
        - Preprocessing if defined
        - Metadata injection into instruction template
        - Role prefixing if specified
        - Original prompt preservation

        Args:
            prompt_obj (PromptObject): The prompt object to transform
            debug (bool, optional): Enable debug logging. Defaults to False.

        Returns:
            PromptObject: The transformed prompt object

        Raises:
            ValueError: If required metadata keys are missing from input data
        """
        if debug:
            self.logger.setLevel(logging.DEBUG)

        def set_orig_prompt(target_prompt: PromptObject, set_from_prompt: PromptObject):
            if target_prompt.orig_prompt is None:
                target_prompt.orig_prompt = PromptObject(
                    prompt=set_from_prompt.prompt, data=set_from_prompt.data
                )

        try:
            set_orig_prompt(prompt_obj, prompt_obj)
        except Exception as e:
            self.logger.error(f"Error in generator {self.name}. Error: {str(e)}")
            raise

        if hasattr(self, "preprocess"):
            mod_prompt_obj = self.preprocess(prompt_obj)

            if mod_prompt_obj is not None:
                if prompt_obj.orig_prompt is None:
                    set_orig_prompt(mod_prompt_obj, prompt_obj)
                else:
                    mod_prompt_obj.orig_prompt = prompt_obj.orig_prompt
                prompt_obj = mod_prompt_obj

        if self.metadata_keys:
            metadata_values = {}
            try:
                for item in self.metadata_keys:
                    if item not in prompt_obj.data:
                        raise ValueError(
                            f"Key {item} not found in input data to generator {self.name}"
                        )
                    metadata_values[item] = prompt_obj.data[item]

                # Store these as inputs to the generator
                prompt_obj.data[self.name + "_input"] = metadata_values

                # Format the instruction prompt
                prompt_obj.prompt = self.instruction.format(**metadata_values)
            except Exception as e:
                self.logger.error(
                    f"Likely missing or incorrect keys in input data to {self.name}. Expected keys: {self.metadata_keys}. Error: {str(e)}"
                )
                raise

        if self.role:
            prompt_obj.prompt = self.role + "\n\n" + prompt_obj.prompt

        assert isinstance(prompt_obj, PromptObject)
        return prompt_obj

    def process_results(self, prompt_obj: PromptObject):
        """Process and store generation results.

        Handles:
        - Storing results in prompt object data
        - Applying postprocessing if defined
        - Maintaining data consistency
        - Supporting result splitting into multiple objects

        Args:
            prompt_obj (PromptObject): Prompt object with generation results

        Returns:
            Union[PromptObject, List[PromptObject]]: Processed results, possibly split
                into multiple prompt objects by postprocessing

        Note:
            - Stores results under {generator_name}_output in data
            - Updates prompt object data with structured output fields if defined
            - Preserves original prompt reference in any new prompt objects
        """
        assert prompt_obj is not None

        # Store the result of this generator in the data
        prompt_obj.data[self.name + "_output"] = prompt_obj.response
        if self.output_type:
            prompt_obj.data.update(prompt_obj.response)

        if prompt_obj.response is None and len(prompt_obj.error) > 0:
            # Result from the generation call to remote LLM inference API
            # failed, just return the (unupdated) prompt_obj.
            return prompt_obj
        if hasattr(self, "postprocess"):
            mod_prompt_obj = self.postprocess(prompt_obj)
            if mod_prompt_obj is not None:
                if isinstance(mod_prompt_obj, list):
                    for item in mod_prompt_obj:
                        item.orig_prompt = prompt_obj.orig_prompt
                else:
                    mod_prompt_obj.orig_prompt = prompt_obj.orig_prompt
                prompt_obj = mod_prompt_obj
        assert (
            prompt_obj is None
            or isinstance(prompt_obj, PromptObject)
            or (
                isinstance(prompt_obj, list)
                and all(isinstance(item, PromptObject) for item in prompt_obj)
            )
        )
        return prompt_obj
