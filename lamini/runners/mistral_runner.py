from typing import List, Union

import pandas as pd
from lamini.runners.base_runner import BaseRunner

DEFAULT_SYSTEM_PROMPT = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."


class MistralRunner(BaseRunner):
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        system_prompt: str = None,
        prompt_template="<s>[INST] {system} {user} [/INST]",
        api_key=None,
        api_url=None,
        config={},
        local_cache_file=None,
    ):
        super().__init__(
            config=config,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            model_name=model_name,
            prompt_template=prompt_template,
            api_key=api_key,
            api_url=api_url,
            local_cache_file=local_cache_file,
        )

    def format_prompt_template(
        self, prompt: Union[str, List[str]], system_prompt: str = None
    ):
        if self.prompt_template is None:
            return prompt

        return self.prompt_template.format(
            system=system_prompt or self.system_prompt, user=prompt
        )

    def load_data(
        self,
        data,
        verbose: bool = False,
        input_key: str = "user",
        output_key: str = "output",
    ):
        super().load_data(
            data,
            verbose,
            input_key,
            output_key,
        )

    def load_data_from_jsonlines(
        self,
        file_path: str,
        verbose: bool = False,
        input_key: str = "user",
        output_key: str = "output",
    ):
        super().load_data_from_jsonlines(
            file_path,
            verbose,
            input_key,
            output_key,
        )

    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        input_key: str = "user",
        output_key: str = "output",
    ):
        super().load_data_from_dataframe(
            df,
            verbose,
            input_key,
            output_key,
        )

    def load_data_from_csv(
        self,
        file_path: str,
        verbose: bool = False,
        input_key: str = "user",
        output_key: str = "output",
    ):
        super().load_data_from_csv(
            file_path,
            verbose,
            input_key,
            output_key,
        )
