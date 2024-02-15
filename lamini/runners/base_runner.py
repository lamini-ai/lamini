from typing import List, Optional, Union

import jsonlines
import pandas as pd
from lamini.api.lamini import Lamini


class BaseRunner:
    def __init__(
        self,
        model_name,
        system_prompt,
        prompt_template,
        api_key,
        api_url,
        config,
        local_cache_file,
    ):
        self.config = config
        self.model_name = model_name
        self.lamini_api = Lamini(
            model_name=model_name,
            api_key=api_key,
            api_url=api_url,
            config=self.config,
            local_cache_file=local_cache_file,
        )
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.data = []

    def __call__(
        self,
        prompt: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ):
        return self.call(prompt, system_prompt, output_type, max_tokens)

    def call(
        self,
        prompt: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ):
        input_objects = self.create_final_prompts(prompt, system_prompt)

        return self.lamini_api.generate(
            prompt=input_objects,
            model_name=self.model_name,
            max_tokens=max_tokens,
            output_type=output_type,
        )

    def create_final_prompts(self, prompt: Union[str, List[str]], system_prompt: str):
        if self.prompt_template is None:
            return prompt

        if isinstance(prompt, str):
            return self.format_prompt_template(prompt, system_prompt=system_prompt)

        final_prompts = [
            self.format_prompt_template(p, system_prompt=system_prompt) for p in prompt
        ]

        return final_prompts

    def format_prompt_template(
        self, prompt: Union[str, List[str]], system_prompt: str = None
    ):
        if self.prompt_template is None:
            return prompt

        return self.prompt_template.format(
            system=system_prompt or self.system_prompt, input=prompt
        )

    def load_data(
        self,
        data,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """
        Load a list of dictionary objects with input-output keys into the LLM
        Each object must have input_key and output_key as keys.
        """

        # Get keys
        if not isinstance(data, list) and not isinstance(data[0], dict):
            raise ValueError(
                f"Data must be a list of dicts with keys input_key={input_key} and optionally output_key={output_key}. Or pass in different input_key and output_key"
            )
        try:
            input_output_objects = [
                {
                    "input": self.format_prompt_template(d[input_key]),
                    "output": d[output_key] if output_key in d else "",
                }
                for d in data
            ]
        except KeyError:
            raise ValueError(
                f"Each object must have input_key={input_key}, and optionally output_key={output_key}, as keys"
            )

        if len(input_output_objects) > 3000:
            self.lamini_api.upload_data(input_output_objects)
        else:
            self.data.extend(input_output_objects)
        if verbose:
            if len(input_output_objects) > 0:
                print("Sample added data: %s" % str(input_output_objects[0]))
            print("Loaded %d data pairs" % len(input_output_objects))
            print("Total data pairs: %d" % len(self.data))

    def load_data_from_jsonlines(
        self,
        file_path: str,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """
        Load a jsonlines file with input output keys into the LLM.
        Each line must be a json object with input_key and output_key as keys.
        """
        data = []
        with open(file_path) as dataset_file:
            data = list(jsonlines.Reader(dataset_file))
        self.load_data(
            data, verbose=verbose, input_key=input_key, output_key=output_key
        )

    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """
        Load a pandas dataframe with input output keys into the LLM.
        Each row must have input_key and output_key as keys.
        """
        if input_key not in df.columns:
            raise ValueError(
                f"Dataframe must have input_key={input_key} as a column, and optionally output_key={output_key}"
            )
        input_output_objects = []
        try:
            for _, row in df.iterrows():
                input_output_objects.append(
                    {
                        "input": self.format_prompt_template(row[input_key]),
                        "output": row[output_key] if output_key in row else "",
                    }
                )
        except KeyError:
            raise ValueError(
                f"Each object must have '{input_key}' and '{output_key}' as keys"
            )
        if len(input_output_objects) > 3000:
            self.lamini_api.upload_data(input_output_objects)
        else:
            self.data.extend(input_output_objects)

        if verbose:
            if len(input_output_objects) > 0:
                print("Sample added data: %s" % str(input_output_objects[0]))
            print("Loaded %d data pairs" % len(input_output_objects))
            print("Total data pairs: %d" % len(self.data))

    def load_data_from_csv(
        self,
        file_path: str,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """
        Load a csv file with input output keys into the LLM.
        Each row must have input_key and output_key as keys.
        The 'system' key is optional and will default to system prompt
        if passed during model initiation else to DEFAULT_SYSTEM_PROMPT.
        """
        df = pd.read_csv(file_path)
        self.load_data_from_dataframe(
            df, verbose=verbose, input_key=input_key, output_key=output_key
        )

    def clear_data(self):
        """Clear the data from the LLM"""
        self.data = []

    def train(
        self,
        limit=500,
        is_public=False,
        **kwargs,
    ):
        """
        Train the LLM on added data. This function blocks until training is complete.
        """
        if len(self.data) < 2 and not self.lamini_api.upload_file_path:
            raise Exception("Submit at least 2 data pairs to train to allow validation")
        if limit is None:
            data = self.data
        elif len(self.data) > limit:
            data = self.data[:limit]
        else:
            data = self.data

        if self.lamini_api.upload_file_path:
            final_status = self.lamini_api.train_and_wait(
                is_public=is_public,
                **kwargs,
            )
        else:
            final_status = self.lamini_api.train_and_wait(
                data,
                is_public=is_public,
                **kwargs,
            )
        try:
            self.model_name = final_status["model_name"]
            self.job_id = final_status["job_id"]
        except KeyError:
            raise Exception("Training failed")

    def evaluate(self) -> List:
        """Get evaluation results"""
        if self.job_id is None:
            raise Exception("Must train before getting results (no job id))")
        self.evaluation = self.lamini_api.evaluate()
        return self.evaluation

    def get_eval_results(self) -> List:
        return self.evaluate()
