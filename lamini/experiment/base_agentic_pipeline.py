import logging
import sys
from tqdm import tqdm
from typing import Union, List
import pandas as pd
from datetime import datetime
import os

from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_generator import BaseGenerator
from lamini.experiment.base_validator import BaseValidator


class BaseAgenticPipeline:
    def __init__(
        self,
        generators: dict[str, BaseGenerator],
        validators: dict[str, BaseValidator] = {},
        order: list[
            str
        ] = None,  # order of running the pipeline of generators and validators
        record_dir: str = None,
        record_step: bool = True,
        record_results: bool = True,
    ):
        self.generators = {}
        self.validators = {}

        generator_names = []
        for generator_name, generator in generators.items():
            self.generators[generator_name] = generator
            generator.name = generator_name
            generator_names.append(generator_name)

        validator_names = []
        for validator_name, validator in validators.items():
            self.validators[validator_name] = validator
            validator.name = validator_name
            validator_names.append(validator_name)

        if order:
            self.order = [
                step
                for step in order
                if step in generator_names or step in validator_names
            ]
        else:
            self.order = generator_names + validator_names

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        if record_dir:
            self.record_dir = record_dir
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.record_dir = f"results_agentic_pipeline/{timestamp}"

        os.makedirs(self.record_dir, exist_ok=True)

        self.logger.info(f"Results will be saved to: {self.record_dir}")

        self.record_step = record_step
        self.record_results = record_results

    def _process_and_save_data(
        self,
        prompt_objects: List[PromptObject],
        step_name: str = None,
        is_final_results: bool = False,
    ):
        # Determine file paths
        if is_final_results:
            data_io_path = self.record_dir + "/data_io"
        else:
            prompt_response_path = f"{self.record_dir}/{step_name}_prompt_response"
            data_io_path = f"{self.record_dir}/{step_name}_data_io"
            full_path = f"{self.record_dir}/{step_name}_full"

        # Process full data
        data_io = []
        prompt_response = []
        full_data = []
        for obj in prompt_objects:
            if not is_final_results:
                full_data.append(
                    {
                        "prompt": obj.prompt,
                        "response": obj.response,
                        "data": obj.data,
                    }
                )

                # Record prompt and response pairs
                prompt_response.append(
                    {
                        "prompt": obj.prompt,
                        "response": obj.response,
                    }
                )

            # Record data input and output
            if is_final_results:
                data_io.extend([
                    {
                        k: v
                        for k, v in obj.data.items()
                        if k.endswith("_output") or k.endswith("_input")
                    }
                ])
            else:
                data_io.extend([
                    {
                        k: v
                        for k, v in obj.data.items()
                        if k == f"{step_name}_output" or k == f"{step_name}_input"
                    }
                ])               

        # Save to CSV
        if is_final_results:
            pd.DataFrame(data_io).to_csv(f"{data_io_path}.csv", index=False)
        else:
            # Check if files exist to determine whether to write headers
            write_header_full = not os.path.exists(f"{full_path}.csv")
            write_header_data_io = not os.path.exists(f"{data_io_path}.csv")
            write_header_prompt_response = not os.path.exists(
                f"{prompt_response_path}.csv"
            )

            pd.DataFrame(full_data).to_csv(
                f"{full_path}.csv", mode="a", header=write_header_full, index=False
            )
            pd.DataFrame(data_io).to_csv(
                f"{data_io_path}.csv",
                mode="a",
                header=write_header_data_io,
                index=False,
            )
            pd.DataFrame(prompt_response).to_csv(
                f"{prompt_response_path}.csv",
                mode="a",
                header=write_header_prompt_response,
                index=False,
            )

    def _record_step(
        self, prompt_obj: Union[PromptObject, List[PromptObject]], step_name: str
    ):
        if isinstance(prompt_obj, PromptObject):
            prompt_obj = [prompt_obj]
        self._process_and_save_data(prompt_obj, step_name=step_name)

    def _record_results(self, results: List[PromptObject]):
        self._process_and_save_data(results, is_final_results=True)

    def run_pipeline(
        self, prompt_obj: PromptObject, debug: bool = False, start_from: int = 0
    ):
        with tqdm(self.order[start_from:], leave=False) as pipeline_tracker:
            try:
                for i, step in enumerate(
                    pipeline_tracker
                ):
                    if step in self.generators:
                        prompt_obj = self.generators[step](prompt_obj, debug=debug)
                        if isinstance(prompt_obj, PromptObject):
                            self.logger.debug(f"Generator {step} output: {prompt_obj.response}")
                        else:
                            self.logger.debug(f"Generator {step} output: {prompt_obj}")
                    if step in self.validators:
                        prompt_obj = self.validators[step](prompt_obj, debug=debug)
                        if isinstance(prompt_obj, PromptObject):
                            self.logger.debug(f"Validator {step} output: {prompt_obj.response}")
                        else:
                            self.logger.debug(f"Validator {step} output: {prompt_obj}")
        
                    if self.record_step:
                        self._record_step(prompt_obj, step)
        
                    # If the output is a list, then run the remaining pipeline steps on each item
                    if isinstance(prompt_obj, list):
                        pipeline_tracker.close()
                        current_step_index = start_from + i + 1
                        prompt_obj = [
                            self.run_pipeline(item, debug=debug, start_from=current_step_index)
                            for item in prompt_obj
                        ]
                        break
                pipeline_tracker.close()
                return prompt_obj
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                pipeline_tracker.close()

    def __call__(
        self, prompt_obj: Union[PromptObject, List[PromptObject]], debug: bool = False
    ):
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        if isinstance(prompt_obj, PromptObject):
            self.logger.debug(f"Running Agentic Pipeline on data: {prompt_obj.data}...")
            return self.run_pipeline(prompt_obj, debug=debug)

        assert isinstance(
            prompt_obj[0], PromptObject
        ), f"Passed in wrong type: {type(prompt_obj[0])}"

        self.logger.debug(
            f"Running Batch Agentic Pipeline on {len(prompt_obj)} items..."
        )

        results = []
        for prompt_obj in tqdm(prompt_obj, desc="Batch [Agentic Pipeline]"):
            try:
                result = self.run_pipeline(prompt_obj, debug=debug)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            except ValueError as e:
                print(e)
                print("Pipeline Failed on Prompt Object, failed result appended to dataframe")
                results.append(prompt_obj)

        if self.record_results:
            self._record_results(results)

        return results
