from pydantic import BaseModel, create_model
from typing import Union, Dict, Optional, Type

from lamini.api.openai_client import BaseOpenAIClient
from lamini.experiment.base_generator import BaseGenerator


class DefaultOutputType(BaseModel):
    """Default output type for validators.

    A simple Pydantic model that includes only an is_valid boolean field.

    Attributes:
        is_valid (bool): Indicates whether validation passed
    """
    is_valid: bool


class BaseValidator(BaseGenerator):
    """Base class for implementing validators in a pipeline.

    Validators are specialized generators that always include a boolean validation field
    in their output. They inherit from BaseGenerator and add validation-specific
    functionality.

    Args:
        name (str): Name of the validator
        instruction (str): Template for the validation prompt
        client (BaseOpenAIClient, optional): Client for LLM interactions
        output_type (Optional[Dict], optional): Output structure specification
        model (Optional[str], optional): Model identifier for LLM
        role (Optional[str], optional): System role prefix for prompts
        instruction_search_pattern (Optional[str], optional): Regex for finding template variables
        is_valid_field (str, optional): Name of the boolean validation field

    Raises:
        ValueError: If output_type is specified but doesn't include a proper boolean validation field
    """

    def __init__(
        self,
        name: str,
        instruction: str,
        client: BaseOpenAIClient = None,
        output_type: Optional[Dict] = None,
        model: Optional[str] = None,
        role: Optional[str] = "",
        instruction_search_pattern: Optional[str] = r"\{(.*?)\}",
        is_valid_field: str = None,
    ):
        super().__init__(
            name=name,
            client=client,
            instruction=instruction,
            output_type=output_type,
            model=model,
            role=role,
            instruction_search_pattern=instruction_search_pattern,
        )

        self.is_valid_field = is_valid_field
        self.output[self.is_valid_field] = "bool"

        if output_type is not None:
            if is_valid_field not in output_type:
                raise ValueError(
                    f"Output format must have a boolean field, set using is_valid_field, which is currently set to '{is_valid_field}'"
                )
            if output_type[is_valid_field] != "bool":
                raise ValueError(
                    f"Output format of the field '{is_valid_field}' must be type 'bool'!"
                )
            self.output_type = self.build_dynamic_model(output_type)
        elif self.is_valid_field is not None:
            # Create an output format with a boolean field called self.is_valid_field
            self.output_type = create_model(
                "DynamicModel",
                **{self.is_valid_field: (bool, ...)},
            )
        else:
            # Create a default output format with a boolean field called "is_valid"
            self.output_type = DefaultOutputType

    def build_dynamic_model(self, data: dict) -> Type[BaseModel]:
        """Build a Pydantic model from a dictionary specification.

        Creates a dynamic Pydantic model with fields based on the input dictionary,
        ensuring the validation field is present.

        Args:
            data (dict): Field specifications where values indicate field types

        Returns:
            Type[BaseModel]: A new Pydantic model class with the specified fields

        Note:
            - "bool" strings are converted to boolean fields
            - Other values use their Python type
            - The validation field is always added if not present
        """
        fields = {}

        # Process each field in the data
        for key, value in data.items():
            # Check if value is the string "bool" for boolean fields
            if value == "bool":
                fields[key] = (bool, ...)
            # For all other types, infer from the value
            else:
                fields[key] = (type(value), ...)

        # Add the is_valid field if not present
        if self.is_valid_field not in fields:
            print("add is valid field", self.is_valid_field)
            fields[self.is_valid_field] = (bool, ...)

        return create_model("DynamicModel", **fields)

    def __call__(self, prompt_obj, debug=False):
        """Execute the validator on a prompt object.

        Extends the base generator's call method to handle validation-specific processing.

        Args:
            prompt_obj (PromptObject): The prompt object to validate
            debug (bool, optional): Enable debug logging. Defaults to False.

        Returns:
            PromptObject: The processed prompt object with validation results
        """
        result = super().__call__(prompt_obj, debug=debug)

        return result
