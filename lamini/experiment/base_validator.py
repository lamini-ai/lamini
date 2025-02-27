from pydantic import BaseModel, create_model
from typing import Union, Dict, Optional, Type

from lamini.api.openai_client import BaseOpenAIClient
from lamini.experiment.base_generator import BaseGenerator


class DefaultOutputType(BaseModel):
    is_valid: bool


class BaseValidator(BaseGenerator):
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
        """
        Build a Pydantic model class with fields inferred from data.
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
        result = super().__call__(prompt_obj, debug=debug)

        return result
