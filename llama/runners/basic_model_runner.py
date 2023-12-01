from typing import List, Union
from llama import Type, Context
import jsonlines
import pandas as pd
from llama.prompts.blank_prompt import BlankPrompt
from llama.engine.typed_lamini import TypedLamini
from llama.runners.runner import Runner
import os

class Input(Type):
    input: str = Context(" ")


class Output(Type):
    output: str = Context(" ")


class BasicModelRunner(Runner):
    """A class for running and training a model with a blank prompt (string in, string out)"""

    def __init__(
            self,
            model_name: str = "EleutherAI/pythia-410m-deduped",
            task_name: str = "basic_model_runner_data",
            enable_peft=False,
            config={},
    ):
        self.model_name = model_name
        self.prompt = BlankPrompt()
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

    def call(self, inputs: Union[str, List[str]]) -> str:
        """Call the model runner on prompt"""
        # Alternative way to run it:
        # output = self.llm(self.prompt.input(input=input_string), self.prompt.output)

        if isinstance(inputs, list):
            print("Running batch job on %d number of inputs" % len(inputs))
            input_objects = [Input(input=i) for i in inputs]
        else:
            # Singleton
            input_objects = Input(input=inputs)
        output_objects = self.llm(
            input=input_objects,
            output_type=Output,
            model_name=self.model_name,
            enable_peft=self.enable_peft,
        )
        if isinstance(output_objects, list):
            outputs = [o.output for o in output_objects]
            return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        else:
            return output_objects.output

    def upload_file(self,
                    file_path,
                    input_key: str = "input",
                    output_key: str = "output"):

        if os.path.getsize(file_path) > 1e+7:
            raise Exception("File size is too large, please upload file less than 10MB")

        # Convert file records to appropriate format before uploading file
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:
                reader = jsonlines.Reader(dataset_file)
                data = list(reader)

                for row in data:
                    item = [
                        {"input": row[input_key]},
                        {"output": row[output_key] if output_key else ""}
                    ]
                    items.append(item)

        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path).fillna("")
            data_keys = df.columns
            if input_key not in data_keys:
                raise ValueError(
                    f"File must have input_key={input_key} as a column (and optionally output_key={output_key}). You "
                    "can pass in different input_key and output_keys."
                )

            try:
                for _, row in df.iterrows():
                    item = [
                        {"input": row[input_key]},
                        {"output": row[output_key] if output_key else ""}
                    ]
                    items.append(item)
            except KeyError:
                raise ValueError("Each object must have 'input' and 'output' as keys")

        else:
            raise Exception("Upload of only csv and jsonlines file supported at the moment.")
        print(len(items))
        self.llm.upload_data(items)

    def load_data(
            self,
            data,
            verbose: bool = False,
            input_key: str = "input",
            output_key: str = "output",
    ):
        """
        Load a list of json objects with input-output keys into the LLM
        Each object must have 'input' and 'output' as keys.
        """
        if not isinstance(data, list) or not isinstance(data[0], dict):
            raise ValueError(
                f"Data must be a list of dicts with keys input_key={input_key} and output_key={output_key}. You can pass in different input_key and output_key."
            )

        try:
            input_output_objects = [
                [
                    Input(input=d[input_key]),
                    Output(output=d[output_key]) if output_key else Output(output=""),
                ]
                for d in data
            ]
        except KeyError:
            raise ValueError(
                f"Each object must have input_key={input_key} and output_key={output_key}. You can pass in different input_key and output_keys"
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
            input_key: str = "input",
            output_key: str = "output",
    ):
        """
        Load a jsonlines file with input output keys into the LLM.
        Each line must be a json object with 'input' and 'output' as keys.
        """
        data = []
        with open(file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            data = list(reader)
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
        Each row must have 'input' and 'output' as keys.
        """
        data_keys = df.columns
        if input_key not in data_keys:
            raise ValueError(
                "Dataframe must have input_key={input_key} as a column (and optionally output_key={output_key}). You can pass in different input_key and output_keys."
            )

        input_output_objects = []
        try:
            for _, row in df.iterrows():
                input_output_objects.append(
                    [
                        Input(input=row[input_key]),
                        Output(output=row[output_key])
                        if output_key
                        else Output(output=""),
                    ]
                )
        except KeyError:
            raise ValueError("Each object must have 'input' and 'output' as keys")
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
        Each row must have 'input' and 'output' as keys.
        """
        df = pd.read_csv(file_path)
        self.load_data_from_dataframe(
            df, verbose=verbose, input_key=input_key, output_key=output_key
        )

    def clear_data(self):
        """Clear the data from the LLM"""
        self.llm.delete_data()
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
