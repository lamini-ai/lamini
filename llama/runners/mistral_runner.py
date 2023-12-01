from typing import List, Union
import jsonlines
import pandas as pd

from llama.prompts.mistral_prompt import MistralPrompt
from llama.types.type import Type
from llama.engine.typed_lamini import TypedLamini
from llama.runners.llama_v2_runner import LlamaV2Runner

DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."


class MistralRunner(LlamaV2Runner):
    """A class for running and training a Mistral model, using system and user prompts"""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        task_name: str = "mistral_runner_data",
        system_prompt: str = None,
        enable_peft: bool = False,
        config: dict = {},
    ):
        super().__init__(
            model_name=model_name,
            task_name=task_name,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            enable_peft=enable_peft,
            config=config,
            prompt_obj=MistralPrompt(),
        )
