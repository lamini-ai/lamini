from typing import List, Union
from llama import Type, Context
import jsonlines
import pandas as pd
import random

from llama.prompts.blank_prompt import BlankPrompt
from llama.engine.typed_lamini import TypedLamini
from llama.runners.runner import Runner


class Input(Type):
    input: str = Context(" ")


class Output(Type):
    output: str = Context(" ")


class AutocompleteRunner(Runner):
    """A class for running and training a model for autocomplete with a blank prompt (string in, string out)"""

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-410m-deduped",
        task_name: str = "autocomplete_runner",  # used as id
        config={},
        enable_peft=False,
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

    def call(self, inputs: Union[str, List[str]]) -> Union[str, List[str]]:
        """Call the model runner on prompt"""
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
            return outputs
        else:
            return output_objects.output

    def evaluate_autocomplete(
        self, data: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        if not (isinstance(data, list) or isinstance(data, str)):
            raise ValueError("Data must be a string or list of strings")

        if isinstance(data, str):
            data = [data]

        input_strings = []
        output_strings = []
        for data_string in data:
            prompt_string, target_string = self.__split_for_autocomplete__(data_string)
            input_strings.append(prompt_string)
            output_strings.append(target_string)

        predictions = self.__call__(input_strings)

        # Return dictionary of paired prompts, targets, and predictions
        return [
            {
                "input": input_strings[i],
                "target_output": output_strings[i],
                "predicted_output": predictions[i]
                if isinstance(predictions, list)
                else predictions,
            }
            for i in range(len(input_strings))
        ]

    def __split_for_autocomplete__(self, data_string: str) -> str:
        # Find random place to split for evaluation
        if len(data_string) == 0:
            return "", ""
        split = random.randint(1, len(data_string))
        # print("slice", split, data_string)
        prompt_string = data_string[:split]
        target_string = data_string[split:]
        return prompt_string, target_string

    def print_verbose(self, data_strings, verbose: bool = False):
        if verbose:
            print(
                "Sample data: %s"
                % str(str(data_strings[0][0].input) + str(data_strings[0][1].output))
            )
            print("Sample split-processed data: %s" % str(data_strings[0]))
            print("Loaded %d datapoints" % len(data_strings))
            print("Total datapoints: %d" % len(self.data))

    def load_data(self, data, verbose: bool = False):
        """
        Load arbitrary singleton types' values into the LLM
        """
        if not isinstance(data, list):
            data = [data]
        data_strings = []
        for data_object in data:
            data_dict = data_object.dict()
            fields = list(data_dict.keys())
            for field in fields:
                data_string = data_dict[field]
                input_string, output_string = self.__split_for_autocomplete__(
                    data_string
                )
                data_strings.append(
                    [Input(input=input_string), Output(output=output_string)]
                )
        self.data.extend(data_strings)
        self.print_verbose(data_strings, verbose)

    def load_data_from_strings(self, data, verbose: bool = False):
        """
        Load a list of strings into the LLM
        """
        if not (isinstance(data, list) or isinstance(data, str)):
            raise ValueError("Data must be a string or list of strings")

        if isinstance(data, str):
            data = [data]

        data_pairs = []
        for data_string in data:
            input_string, output_string = self.__split_for_autocomplete__(data_string)
            data_pairs.append([Input(input=input_string), Output(output=output_string)])

        self.data.extend(data_pairs)
        self.print_verbose(data_pairs, verbose)

    def load_data_from_jsonlines(
        self, file_path: str, keys=None, verbose: bool = False
    ):
        """
        Load a jsonlines file with any keys into the LLM. Assume all independent strings.
        """
        data = []
        with open(file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            data = list(reader)
        extracted_data = []
        for d in data:
            # Append strings in all keys in keys param, if None then use all keys
            d_keys = list(d.keys())
            for k in d_keys:
                if keys is None or k in keys:
                    extracted_data.append(d[k])
        self.load_data_from_strings(extracted_data, verbose=verbose)

    def load_data_from_dataframe(
        self, df: pd.DataFrame, columns=None, verbose: bool = False
    ):
        """
        Load a pandas dataframe into the LLM.
        """
        if columns is None:
            values = df.values
        else:
            values = df[columns].values
        # Flatten values
        flat_values = [item for sublist in values for item in sublist]
        self.load_data_from_strings(flat_values, verbose=verbose)

    def load_data_from_csv(self, file_path: str, columns=None, verbose: bool = False):
        """
        Load a csv file with input output keys into the LLM.
        Each row must have 'input' and 'output' as keys.
        """
        df = pd.read_csv(file_path)
        self.load_data_from_dataframe(df, columns, verbose=verbose)

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
        if len(self.data) < 2:
            raise Exception("Submit at least 2 data pairs to train to allow validation")
        if limit is None:
            data = self.data
        elif len(self.data) > limit:
            data = self.data[:limit]
        else:
            data = self.data

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
