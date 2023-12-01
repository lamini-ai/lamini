from typing import List, Union
import jsonlines
import pandas as pd
import os
from llama.prompts.llama_v2_prompt import LlamaV2Prompt, LlamaV2Input, LlamaV2Output
from llama.types.type import Type
from llama.engine.typed_lamini import TypedLamini
from llama.runners.runner import Runner

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class LlamaV2Runner(Runner):
    """A class for running and training a Llama V2 model, using system and user prompts"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        task_name: str = "llama_v2_runner_data",
        system_prompt: str = None,
        enable_peft: bool = False,
        config: dict = {},
        prompt_obj=None,
    ):
        self.model_name = model_name

        if prompt_obj is None:
            self.prompt = LlamaV2Prompt()
        else:
            self.prompt = prompt_obj

        self.llm = TypedLamini(
            id=task_name,
            model_name=model_name,
            config=config,
            prompt_template=self.prompt.prompt_template,
        )
        self.job_id = None
        self.data = []
        self.evaluation = None
        self.enable_peft = enable_peft
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def call(
        self,
        inputs: Union[str, List[str]],
        system_prompt: str = None,
        output_type: Type = None,
        cue: str = "",
    ) -> str:
        """Call the model runner on prompt"""
        system_prompt = system_prompt or self.system_prompt
        output_type = output_type or LlamaV2Output

        if isinstance(inputs, list):
            input_objects = [
                LlamaV2Input(user=i, system=system_prompt, cue=cue) for i in inputs
            ]
        else:
            # Singleton
            input_objects = LlamaV2Input(user=inputs, system=system_prompt, cue=cue)

        output_objects = self.llm(
            input=input_objects,
            output_type=output_type,
            model_name=self.model_name,
            enable_peft=self.enable_peft,
        )
        if isinstance(output_objects, list):
            if hasattr(output_objects[0], "output"):
                outputs = [o["output"] for o in output_objects]
                return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]

            return output_objects
        else:
            if hasattr(output_objects, "output"):
                return output_objects.output
            return output_objects

    def load_data(
        self,
        data,
        verbose: bool = False,
        user_key: str = "user",
        output_key: str = "output",
    ):
        """
        Load a list of dictionary objects with input-output keys into the LLM
        Each object must have 'user' and 'output' as keys.
        """
        # Get keys
        if not isinstance(data, list) and not isinstance(data[0], dict):
            raise ValueError(
                f"Data must be a list of dicts with keys user_key={user_key} and optionally output_key={output_key}. Or pass in different user_key and output_key"
            )
        try:
            input_output_objects = [
                [
                    LlamaV2Input(
                        user=d[user_key],
                        system=self.system_prompt,
                    ),
                    LlamaV2Output(output=d[output_key])
                    if output_key in d
                    else LlamaV2Output(output=""),
                ]
                for d in data
            ]
        except KeyError:
            raise ValueError(
                f"Each object must have user_key={user_key}, and optionally output_key={output_key}, as keys"
            )
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
        user_key: str = "user",
        output_key: str = "output",
    ):
        """
        Load a jsonlines file with input output keys into the LLM.
        Each line must be a json object with 'user' and 'output' as keys.
        """
        data = []
        with open(file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            data = list(reader)
        self.load_data(data, verbose=verbose, user_key=user_key, output_key=output_key)

    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        user_key: str = "user",
        output_key: str = "output",
    ):
        """
        Load a pandas dataframe with input output keys into the LLM.
        Each row must have 'user' and 'output' as keys.
        """
        if user_key not in df.columns:
            raise ValueError(
                f"Dataframe must have user_key={user_key} as a column, and optionally output_key={output_key}"
            )
        input_output_objects = []
        try:
            for _, row in df.iterrows():
                input_output_objects.append(
                    [
                        LlamaV2Input(user=row[user_key], system=self.system_prompt),
                        LlamaV2Output(output=row[output_key])
                        if output_key in df.columns
                        else LlamaV2Output(output=""),
                    ]
                )
        except KeyError:
            raise ValueError("Each object must have 'user' and 'output' as keys")
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
        user_key: str = "user",
        output_key: str = "output",
    ):
        """
        Load a csv file with input output keys into the LLM.
        Each row must have 'user' and 'output' as keys.
        The 'system' key is optional and will default to system prompt
        if passed during model initiation else to DEFAULT_SYSTEM_PROMPT.
        """
        df = pd.read_csv(file_path)
        self.load_data_from_dataframe(
            df, verbose=verbose, user_key=user_key, output_key=output_key
        )

    def upload_file(
        self,
        file_path,
        user_key: str = "user",
        output_key: str = "output",
    ):
        if os.path.getsize(file_path) > 1e7:
            raise Exception("File size is too large, please upload file less than 10MB")

        # Convert file records to appropriate format before uploading file
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:
                reader = jsonlines.Reader(dataset_file)
                data = list(reader)

                for row in data:
                    item = [
                        {
                            "user": row[user_key],
                            "system": self.system_prompt,
                            "cue": "",
                        },
                        {"output": row[output_key] if output_key else ""},
                    ]
                    items.append(item)

        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path).fillna("")
            data_keys = df.columns
            if user_key not in data_keys:
                raise ValueError(
                    f"File must have user_key={user_key} as a column (and optionally output_key={output_key}). You "
                    "can pass in different user_key and output_keys."
                )

            try:
                for _, row in df.iterrows():
                    item = [
                        {
                            "user": row[user_key],
                            "system": self.system_prompt,
                            "cue": "",
                        },
                        {"output": row[output_key] if output_key else ""},
                    ]
                    items.append(item)
            except KeyError:
                raise ValueError("Each object must have 'input' and 'output' as keys")

        else:
            raise Exception(
                "Upload of only csv and jsonlines file supported at the moment."
            )

        self.llm.upload_data(items)

    def clear_data(self):
        """Clear the data from the LLM"""
        self.data = []

    def train(
        self,
        verbose: bool = False,
        limit=500,
        is_public=False,
        **kwargs,
    ):
        """
        Train the LLM on added data. This function blocks until training is complete.
        """
        if len(self.data) < 2 and not self.llm.upload_file_path:
            raise Exception("Submit at least 2 data pairs to train to allow validation")
        if limit is None:
            data = self.data
        elif len(self.data) > limit:
            data = self.data[:limit]
        else:
            data = self.data

        if self.llm.upload_file_path:
            final_status = self.llm.train(
                **kwargs,
            )
        else:
            final_status = self.llm.train(
                data,
                **kwargs,
            )
        try:
            self.model_name = final_status["model_name"]
            self.job_id = final_status["job_id"]
            self.llm.delete_data()
        except KeyError:
            raise Exception("Training failed")

    def evaluate(self) -> List:
        """Get evaluation results"""
        if self.job_id is None:
            raise Exception("Must train before getting results (no job id))")
        self.evaluation = self.llm.evaluate()
        return self.evaluation

    def get_eval_results(self) -> List:
        return self.evaluate()
