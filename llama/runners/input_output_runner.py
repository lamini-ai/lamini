from typing import List, Union
from llama.engine.typed_lamini import TypedLamini
from llama import Type
import jsonlines
import pandas as pd
from llama.runners.runner import Runner


class InputOutputRunner(Runner):
    """A class for running and training a model with an Input and an Output type"""

    def __init__(
        self,
        input_type: Type,
        output_type: Type,
        model_name: str = "EleutherAI/pythia-410m-deduped",
        task_name: str = "input_output_runner_data",
        enable_peft=False,
        config={},
    ):
        self.input_type = input_type
        self.output_type = output_type

        self.input_type_keys = self.input_type.__fields__.keys()
        self.output_type_keys = self.output_type.__fields__.keys()

        self.model_name = model_name

        self.llm = TypedLamini(
            id=task_name,
            model_name=model_name,
            config=config,
        )
        self.job_id = None
        self.data = []
        self.evaluation = None
        self.enable_peft = enable_peft

    def call(self, inputs: Union[Type, List[Type]]) -> str:
        """Call the model runner on prompt"""

        if isinstance(inputs, list):
            if not isinstance(inputs[0], self.input_type):
                raise ValueError("Input must be of type %s" % self.input_type)
            # Just printing here if batched or not
            print("Running batch job on %d number of inputs" % len(inputs))
        else:
            # Singleton
            if not isinstance(inputs, self.input_type):
                raise ValueError("Input must be of type %s" % self.input_type)

        output_objects = self.llm(
            input=inputs,
            output_type=self.output_type,
            model_name=self.model_name,
            enable_peft=self.enable_peft,
        )

        return output_objects

    def print_verbose(self, input_output_objects: list, verbose: bool):
        if verbose:
            if len(input_output_objects) > 0:
                print("Sample added data: %s" % str(input_output_objects[0]))
            print("Loaded %d data pairs" % len(input_output_objects))
            print("Total data pairs: %d" % len(self.data))

    def load_data(self, data: list, verbose: bool = False):
        if not (
            isinstance(data, list)
            and isinstance(data[0], list)
            and isinstance(data[0][0], self.input_type)
            and isinstance(data[0][1], self.output_type)
        ):
            raise ValueError(
                f"Data must be [[input_object, output_object], [input_object, output_object], ...], that is a list of lists (with inner input and output objects), where inner objects have keys matching input keys: {self.input_type_keys}; and, output keys: {self.output_type_keys} respectively."
            )
        self.data.extend(data)
        self.print_verbose(data, verbose)

    def load_data_from_paired_dicts(
        self,
        data,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        # Expect data to be formatted as [{"input": input_dict, "output": output_dict}, {"input": input_dict, "output": output_dict}, ...]
        if not (
            isinstance(data, list)
            and isinstance(data[0], dict)
            and input_key in data[0]
            and output_key in data[0]
            and isinstance(data[0]["input"], dict)
            and isinstance(data[0]["output"], dict)
        ):
            raise ValueError(
                f"Data must be [{'input': input_dict, 'output': output_dict}, {'input': input_dict, 'output': output_dict}, ...], that is a list of dicts (with inner input and output dicts), where inner dicts have keys matching input keys: {self.input_type_keys}; and, output keys: {self.output_type_keys} respectively. You can pass in different values for input_key={input_key} and output_key={output_key}."
            )

        # Check if data is has the right input and output type keys
        input_datum = data[0][input_key]
        input_keys = input_datum.keys()
        if input_keys != self.input_type_keys:
            raise ValueError(
                f"Keys have to match {self.input_type}. But got {input_keys} instead."
            )

        output_datum = data[0][output_key]
        output_keys = output_datum.keys()
        if output_keys != self.output_type_keys:
            raise ValueError(
                f"Keys have to match {self.output_type}. But got {output_keys} instead."
            )

        # Convert data to Input and Output objects
        input_output_objects = []
        for datum in data:
            try:
                input_object = self.input_type(**datum[input_key])
                output_object = self.output_type(**datum[output_key])
            except:
                print(
                    f"Error converting input-output pair to Input and Output objects; skipping this pair {datum}"
                )
                continue
            input_output_objects.append([input_object, output_object])

        self.data.extend(input_output_objects)
        self.print_verbose(input_output_objects, verbose)

    def load_data_from_paired_lists(self, data, verbose: bool = False):
        # Expect data to be formatted as [[input_dict, output_dict], [input_dict, output_dict], ...]
        if not (
            isinstance(data, list)
            and isinstance(data[0], list)
            and isinstance(data[0][0], dict)
            and isinstance(data[0][1], dict)
        ):
            raise ValueError(
                f"Data must be [[input_dict, output_dict], [input_dict, output_dict], ...], that is a list of lists (inner list is an input-output pair), where input and output are dicts with keys matching input keys: {self.input_type_keys}; and, output keys: {self.output_type_keys} respectively."
            )

        # Check if data is has the right input and output type keys
        input_datum = data[0][0]
        input_keys = input_datum.keys()
        if input_keys != self.input_type_keys:
            raise ValueError(
                f"Keys have to match {self.input_type}. But got {input_keys} instead."
            )

        output_datum = data[0][1]
        output_keys = output_datum.keys()
        if output_keys != self.output_type_keys:
            raise ValueError(
                f"Keys have to match {self.output_type}. But got {output_keys} instead."
            )

        # Convert data to Input and Output objects
        input_output_objects = []
        for pair in data:
            try:
                input_object = self.input_type(**pair[0])
                output_object = self.output_type(**pair[1])
            except:
                print(
                    f"Error converting input-output pair to Input and Output objects; skipping this pair {pair}"
                )
                continue
            input_output_objects.append([input_object, output_object])

        self.data.extend(input_output_objects)
        self.print_verbose(input_output_objects, verbose)

    def has_input_output_prefix_keys(self, dictionary):
        has_input = False
        has_output = False

        for key in dictionary.keys():
            if key.startswith("input-"):
                has_input = True
            elif key.startswith("output-"):
                has_output = True

            # If both prefixes are found, we can stop iterating to save time.
            if has_input and has_output:
                break

        return has_input and has_output

    def load_data_from_jsonlines(
        self,
        file_path: str,
        verbose: bool = False,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """
        Load a jsonlines file with either dict of input dict/output dict, or input- prefix and output- prefix keys.
        """
        input_output_objects = []
        with open(file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            data = list(reader)

        # Check if data is in the right format
        if not (
            isinstance(data[0], dict)
            and (
                (input_key in data[0] and output_key in data[0])
                or (self.has_input_output_prefix_keys(data[0]))
            )
        ):
            raise ValueError(
                f"Data must be in one the following formats: [{'input': input_dict, 'output': output_dict}, {'input': input_dict, 'output': output_dict}, ...], that is a list of dicts (with inner input and output dicts), where inner dicts have keys matching input keys: {self.input_type_keys}; and, output keys: {self.output_type_keys} respectively. Or, it can be flattened dict with input- and output- prefixes on the keys."
            )

        for datum in data:
            # Get input and output keys and dicts
            input_dict = {}
            output_dict = {}
            for k, v in datum.items():
                # Nested input dict and output dict
                if k == input_key:
                    input_dict = v
                elif k == output_key:
                    output_dict = v
                # If it's just one flattened dict with prefix keys, then parse it
                elif k.startswith(f"{input_key}-"):
                    input_dict[k.split(f"{input_key}-")[-1]] = v
                elif k.startswith(f"{output_key}-"):
                    output_dict[k.split(f"{output_key}-")[-1]] = v

            # Parse input and output dicts to Input and Output objects
            print(input_dict, output_dict)
            try:
                input_object = self.input_type(**input_dict)
                output_object = self.output_type(**output_dict)
            except:
                print(
                    f"Error converting input-output pair to Input and Output objects; skipping this pair {datum}"
                )
                continue
            input_output_objects.append([input_object, output_object])

        self.data.extend(input_output_objects)
        self.print_verbose(input_output_objects, verbose)

    def load_data_from_paired_jsonlines(
        self, input_file_path: str, output_file_path: str, verbose: bool = False
    ):
        """
        Load two jsonlines files, one with input keys and the other with output keys.
        """
        data = []
        with open(input_file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            input_data = list(reader)
        with open(output_file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            output_data = list(reader)

        # Make sure the input and output data are the same length
        min_length = min(len(input_data), len(output_data))
        if len(input_data) != len(output_data):
            print(
                f"Input and output data are not the same length, so going with the smaller length: {min_length}"
            )

        data = [[input_data[i], output_data[i]] for i in range(min_length)]
        self.load_data_from_paired_lists(data, verbose=verbose)

    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        input_prefix: str = "input-",
        output_prefix: str = "output-",
    ):
        """
        Load a pandas dataframe with input- prefix and output- prefix keys.
        """
        data_keys = df.columns

        # Get input and output keys
        input_keys = [k for k in data_keys if k.startswith(input_prefix)]
        output_keys = [k for k in data_keys if k.startswith(output_prefix)]

        if not len(input_keys) > 0:
            raise ValueError(
                f"Input keys must start with {input_prefix}. But got {input_keys} instead."
            )
        if not len(output_keys) > 0:
            raise ValueError(
                f"Output keys must start with {output_prefix}. But got {output_keys} instead."
            )

        # Suppress SettingWithCopyWarning for renaming views of a dataframe
        pd.options.mode.chained_assignment = None

        # Grab input subset of dataframe
        input_df = df[input_keys]
        remove_input_prefix = lambda col: col.replace(input_prefix, "")
        input_df.rename(columns=remove_input_prefix, inplace=True)
        # Check if matching input and output keys
        if set(input_df.columns) != set(self.input_type_keys):
            raise ValueError(
                f"Keys have to match "
                + str(self.input_type)
                + ". But got "
                + str(input_df.columns)
                + " instead."
            )

        # Grab output subset of dataframe
        output_df = df[output_keys]
        remove_output_prefix = lambda col: col.replace(output_prefix, "")
        output_df.rename(columns=remove_output_prefix, inplace=True)
        # Check if matching input and output keys
        if set(output_df.columns) != set(self.output_type_keys):
            raise ValueError(
                f"Keys have to match "
                + str(self.output_type)
                + ". But got "
                + str(output_df.columns)
                + " instead."
            )

        # Restore default pandas warning settings
        pd.options.mode.chained_assignment = "warn"

        # Iterate through rows and convert to Input and Output objects
        input_output_objects = []
        for i in range(len(input_df)):
            try:
                input_object = self.input_type(**input_df.iloc[i])
                output_object = self.output_type(**output_df.iloc[i])
            except:
                print(
                    f"Error converting input-output pair to Input and Output objects; skipping this pair {input_df.iloc[i], output_df.iloc[i]}"
                )
                continue
            input_output_objects.append([input_object, output_object])

        self.data.extend(input_output_objects)
        self.print_verbose(input_output_objects, verbose)

    def load_data_from_paired_dataframes(
        self, input_df: pd.DataFrame, output_df: pd.DataFrame, verbose: bool = False
    ):
        """
        Load a pandas dataframes, one with input keys, one with output keys, into the LLM.
        """
        # Make sure the input and output data are the same length
        min_length = min(len(input_df), len(output_df))
        if len(input_df) != len(output_df):
            print(
                f"Input and output data are not the same length, so going with the smaller length: {min_length}"
            )

        # Convert data to Input and Output objects
        input_output_objects = []
        for i in range(min_length):
            try:
                input_object = self.input_type(**input_df.iloc[i])
                output_object = self.output_type(**output_df.iloc[i])
            except:
                print(
                    f"Error converting input-output pair to Input and Output objects; skipping this pair {input_df.iloc[i], output_df.iloc[i]}"
                )
                continue
            input_output_objects.append([input_object, output_object])

        self.data.extend(input_output_objects)
        self.print_verbose(input_output_objects, verbose)

    def load_data_from_csv(
        self,
        file_path: str,
        verbose: bool = False,
        input_prefix: str = "input-",
        output_prefix: str = "output-",
    ):
        """
        Load a csv file with input- prefix and output- prefix keys.
        """
        df = pd.read_csv(file_path)
        self.load_data_from_dataframe(
            df, verbose=verbose, input_prefix=input_prefix, output_prefix=output_prefix
        )

    def load_data_from_paired_csvs(
        self, input_file_path: str, output_file_path: str, verbose: bool = False
    ):
        """
        Load a csv file with input output keys into the LLM.
        Each row must have 'input' and 'output' as keys.
        """
        input_df = pd.read_csv(input_file_path)
        output_df = pd.read_csv(output_file_path)
        self.load_data_from_paired_dataframes(input_df, output_df, verbose=verbose)

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

    def get_eval_results(self) -> List:
        return self.evaluate()
