from typing import List, Dict

from lamini.experiment.validators import BaseValidator

class FactualityValidator(BaseValidator):

    instruction = """
    You are a helpful assistant checking if the information is factual, or could've been hallucinated.
    """

    def __init__(
        self,
        model: str,
        instruction_metadata: List[str],
        name: str = "FactualityValidator",
        instruction: str = instruction,
        output_type: Dict = {"thinking_steps": "str", "is_factual": "bool"},
        is_valid_field: str = "is_factual",
    ):

        if instruction_metadata:
            # Append the metadata to the instruction as curly brace placeholders
            for metadata in instruction_metadata:
                instruction += f"\n\n{{{metadata}}}"

        super().__init__(
            name=name,
            model=model,
            instruction=instruction,
            output_type=output_type,
            is_valid_field=is_valid_field,
        )