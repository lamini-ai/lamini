import logging
import sys
from tqdm import tqdm
from typing import Union, List, Dict, Any
import pandas as pd
from datetime import datetime
import os
from copy import deepcopy

from lamini.experiment.utils import remove_non_ascii
from lamini.experiment.generators import BaseGenerator, SaveGenerator
from lamini.experiment.validators import BaseValidator
from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_experiment_object import ExperimentObject

class PipelineStep:
    """A class representing a single step in a processing pipeline.

    This class encapsulates a generator or validator worker along with its queue
    of items to process and a reference to the next step in the pipeline.

    Parameters
    ----------
    generator : BaseGenerator
        The worker (generator or validator) that will process items at this step.
    next_step : PipelineStep, optional
        Reference to the next step in the pipeline, by default None.

    Attributes
    ----------
    worker : BaseGenerator
        The worker instance that processes items at this step.
    next : PipelineStep
        Reference to the next pipeline step (None if this is the final step).
    queue : list
        List of items waiting to be processed by this step.

    Notes
    -----
    Each PipelineStep maintains its own queue of items to process, allowing for
    asynchronous processing and branching logic in the pipeline.
    """
    def __init__(self, generator: BaseGenerator, next_step = None):
        self.worker = generator
        self.next = next_step
        self.queue = []

class BaseAgenticPipeline:
    """A pipeline for processing experiment objects through a sequence of generators and validators.

    This class implements a flexible pipeline architecture that can process experiment objects
    through multiple stages of generation and validation, with support for branching logic
    and result recording.

    Parameters
    ----------
    generators : dict[str, BaseGenerator]
        Dictionary mapping generator names to their corresponding BaseGenerator instances.
    validators : dict[str, BaseValidator], optional
        Dictionary mapping validator names to their corresponding BaseValidator instances.
        Default is an empty dict.
    order : list[str], optional
        List specifying the execution order of generators and validators. If None,
        generators are executed first, followed by validators.
    record_dir : str, optional
        Directory path where pipeline results will be saved. If None, a timestamped
        directory will be created.
    record_step : bool, optional
        Whether to record intermediate results from each pipeline step. Default is True.
    record_results : bool, optional
        Whether to record final pipeline results. Default is True.

    Attributes
    ----------
    generators : dict[str, BaseGenerator]
        Registered generator instances.
    validators : dict[str, BaseValidator]
        Registered validator instances.
    name_order : list[str]
        Final execution order of pipeline steps.
    order : list[PipelineStep]
        List of PipelineStep objects representing the complete pipeline.
    record_dir : str
        Directory where results are saved.
    logger : logging.Logger
        Logger instance for pipeline execution.

    Notes
    -----
    The pipeline supports both single-item and batch processing modes. Each step in the
    pipeline can produce either single or multiple outputs, enabling branching workflows.
    Results are automatically saved in JSON Lines format, with separate files for full data,
    data I/O, and prompt-response pairs.

    Examples
    --------
    >>> generators = {"gen1": MyGenerator(), "gen2": AnotherGenerator()}
    >>> validators = {"val1": MyValidator()}
    >>> pipeline = BaseAgenticPipeline(
    ...     generators=generators,
    ...     validators=validators,
    ...     order=["gen1", "val1", "gen2"]
    ... )
    >>> results = pipeline(experiment_object)
    """
    def __init__(
        self,
        generators: dict[str, BaseGenerator],
        validators: dict[str, BaseValidator] = {},
        order: list[str] = None,
        record_dir: str = None,
        record_step: bool = True,
        record_results: bool = True,
        save_keys: list[str] = None,
    ):
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
            self.name_order = [
                step
                for step in order
                if step in generator_names or step in validator_names
            ]
        else:
            self.name_order = generator_names + validator_names

        self.order = []
        for step in self.name_order:
            worker = self.generators[step] if step in self.generators else self.validators[step]
            self.order.append(PipelineStep(worker))
            
        self.order.append(PipelineStep(
            SaveGenerator(
                save_path=self.record_dir + "/pipeline_results.jsonl",
                save_keys=save_keys if save_keys else list(self.order[-1].worker.output.keys())
            )
        ))

        for idx, step in enumerate(self.order[:-1]):
            step.next = self.order[idx+1]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(generators={list(self.generators.keys())}, "
            f"validators={list(self.validators.keys())}, "
            f"order={self.name_order})"
        )

    def __str__(self):
        return self.__repr__()

    def to_json(self):
        """Convert pipeline configuration to a JSON-serializable dictionary.

        Returns:
            dict: Configuration parameters including generators, validators, execution order,
                  and recording settings
        """
        return {
            "generators": {
                name: generator.to_json() for name, generator in self.generators.items()
            },
            "validators": {
                name: validator.to_json() for name, validator in self.validators.items()
            },
            "execution_order": self.name_order,
            "record_directory": self.record_dir,
            "record_step": self.record_step,
            "record_results": self.record_results
        }

    def _process_and_save_data(
        self,
        prompt_objects: List[ExperimentObject],
        step_name: str = None,
        is_final_results: bool = False,
    ):
        """Process and save pipeline data to JSON Lines files.

        Parameters
        ----------
        prompt_objects : List[ExperimentObject]
            List of experiment objects containing data to be processed and saved.
        step_name : str, optional
            Name of the current pipeline step. Required if is_final_results is False.
        is_final_results : bool, optional
            Whether this is processing final pipeline results (True) or intermediate step results (False).
            Default is False.

        Notes
        -----
        The method saves data in different formats depending on is_final_results:

        For intermediate results (is_final_results=False):
        - {step_name}_full.jsonl: Contains prompt, response, and complete data
        - {step_name}_data_io.jsonl: Contains input/output data for the specific step
        - {step_name}_prompt_response.jsonl: Contains prompt-response pairs

        For final results (is_final_results=True):
        - data_io.jsonl: Contains all input/output data across all steps

        All data is processed to remove non-ASCII characters before saving.
        """
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
        for obj in [obj for obj in prompt_objects if obj is not None]:
            if not is_final_results:
                full_data.append(
                    {
                        "prompt": remove_non_ascii(obj.prompt),
                        "response": remove_non_ascii(obj.response),
                        "data": remove_non_ascii(obj.data),
                    }
                )

                # Record prompt and response pairs
                prompt_response.append(
                    {
                        "prompt": remove_non_ascii(obj.prompt),
                        "response": remove_non_ascii(obj.response),
                    }
                )

            # Record data input and output
            if is_final_results:
                data_io.extend([
                    {
                        k: remove_non_ascii(v)
                        for k, v in obj.data.items()
                        if k.endswith("_output") or k.endswith("_input")
                    }
                ])
            else:
                data_io.extend([
                    {
                        k: remove_non_ascii(v)
                        for k, v in obj.data.items()
                        if k == f"{step_name}_output" or k == f"{step_name}_input"
                    }
                ])               

        # Save to CSV
        if is_final_results:
            pd.DataFrame(data_io).to_json(f"{data_io_path}.jsonl", orient="records", lines=True)
        else:
            # Check if files exist to determine whether to write headers
            write_header_full = not os.path.exists(f"{full_path}.csv")
            write_header_data_io = not os.path.exists(f"{data_io_path}.csv")
            write_header_prompt_response = not os.path.exists(
                f"{prompt_response_path}.csv"
            )

            pd.DataFrame(full_data).to_json(
                f"{full_path}.jsonl", mode="a", orient="records", lines=True
            )
            pd.DataFrame(data_io).to_json(
                f"{data_io_path}.jsonl",
                mode="a",
                orient="records",
                lines=True,
            )
            pd.DataFrame(prompt_response).to_json(
                f"{prompt_response_path}.jsonl",
                mode="a",
                orient="records",
                lines=True,
            )

    def _record_step(
        self, prompt_obj: Union[ExperimentObject, List[ExperimentObject]], step_name: str
    ):
        """Record intermediate results from a single pipeline step.

        Parameters
        ----------
        prompt_obj : Union[ExperimentObject, List[ExperimentObject]]
            The output from a pipeline step to be recorded. Can be either a single
            ExperimentObject or a list of ExperimentObjects.
        step_name : str
            The name of the pipeline step that generated these results.

        Notes
        -----
        This internal method converts single ExperimentObjects to a list and delegates
        to _process_and_save_data for the actual recording of results. The results
        are saved in CSV format with separate files for full data, data I/O, and
        prompt-response pairs.
        """
        if isinstance(prompt_obj, ExperimentObject):
            prompt_obj = [prompt_obj]
        self._process_and_save_data(prompt_obj, step_name=step_name)

    def _record_results(self, results: List[ExperimentObject]):
        """Record the final results of the pipeline execution.

        This internal method handles saving the final aggregated results after all pipeline
        steps have completed. It delegates to _process_and_save_data with is_final_results=True.

        Args:
            results (List[ExperimentObject]): The final results from the complete pipeline execution.
        """
        self._process_and_save_data(results, is_final_results=True)

    def run_pipeline(
        self, exp_objs: List[ExperimentObject], debug: bool = False, start_from: int = 0
    ):
        """Execute the pipeline on a single PromptObject.

        Runs the sequence of generators and validators defined in self.order, starting from
        the specified index. Handles branching logic when a step returns multiple objects.

        Args:
            prompt_obj (PromptObject): The input object to process through the pipeline
            debug (bool, optional): Whether to enable debug logging. Defaults to False.
            start_from (int, optional): Index in self.order to start execution from. 
                Defaults to 0.

        Returns:
            Union[PromptObject, List[PromptObject]]: The result(s) of the pipeline execution.
                May be a single PromptObject or a list if any step produced multiple outputs.

        Raises:
            KeyboardInterrupt: Allows clean interruption of pipeline execution
        """
        results = []
        try:
            self.order[start_from].queue.extend(exp_objs)
            for step in self.order[start_from:]:
                with tqdm(total=len(step.queue), desc=f"{step.worker.name} Execution") as pipeline_tracker:
                    while step.queue:
                        #print(f"Length of queue: {len(step.queue)}")
                        exp_obj = step.queue.pop(0)
                        #print(f"Length of queue: {len(step.queue)}")
                        #print(exp_obj)
                        try:
                            output = step.worker(exp_obj, debug=debug)
                            if step.next:
                                if isinstance(output, list):
                                    exp_objs = [
                                        ExperimentObject(experiment_step=exp_obj.step.next, data=deepcopy(item.data))
                                        for item in output if hasattr(item, "response") and item.response
                                    ]
                                    step.next.queue.extend(exp_objs)
                                else:
                                    if hasattr(output, "response") and output.response:
                                        exp_obj.step = exp_obj.step.next
                                        exp_obj.data = deepcopy(output.data)
                                        step.next.queue.append(exp_obj)
                            else:
                                if isinstance(output, list):
                                    results.extend(output)
                                else:
                                    results.append(output)
                        except KeyboardInterrupt:
                            sys.exit(0)
                        except Exception as e:
                            raise e
                        self._record_step(output, step.worker.name)
                        pipeline_tracker.update(1)
        except KeyboardInterrupt:
            sys.exit(0)
        return results

    def pipeline_step_logic(self, exp_obj: ExperimentObject):
        """Validates the logical connections between pipeline steps by checking input/output key compatibility.

        This method ensures that each step in the pipeline has access to all required input keys
        from the previous steps' outputs.

        Parameters
        ----------
        exp_obj : ExperimentObject
            The experiment object to validate against the pipeline steps.

        Raises
        ------
        ValueError
            If any step's required input keys are not present in the accumulated outputs
            from previous steps.

        Notes
        -----
        The validation process:
        1. Starts with the initial experiment object's data
        2. For each step, verifies that all required input keys are available
        3. Accumulates output keys for the next step's validation
        4. Tracks the progression through steps for detailed error reporting
        """
        self.logger.info("Starting pipeline step logic validation")
        prior_step = "(ExperimentObject Input)"
        prior_step_output = deepcopy(exp_obj.data)
        self.logger.debug(f"Initial data keys: {list(prior_step_output.keys())}")

        for step in self.order:
            self.logger.debug(f"Validating step '{step.worker.name}' input requirements")
            if not set(step.worker.input.keys()).issubset(set(prior_step_output.keys())):
                self.logger.error(
                    f"Step validation failed: {step.worker.name} requires keys {step.worker.input.keys()} "
                    f"but only found {prior_step_output.keys()} from {prior_step}"
                )
                raise ValueError(
                    f"Error in {prior_step} connection to {step.worker.name}!",
                    f"Step {step.worker.name} input keys {step.worker.input.keys()} are not all present in {prior_step} output keys {prior_step_output.keys()}"
                )
            prior_step_output.update(step.worker.output)
            prior_step = step.worker.name
        
        self.logger.info("Pipeline step logic validation completed successfully")

    def pipline_spotcheck(self, exp_obj: ExperimentObject, debug: bool = False):
        """Executes a single sample through the pipeline to verify functionality of all steps.

        This method runs a test execution of the complete pipeline using one sample,
        providing detailed logging and error checking at each step.

        Parameters
        ----------
        exp_obj : ExperimentObject
            The experiment object to use for the test execution.
        debug : bool, optional
            Whether to enable debug-level logging, by default False.

        Raises
        ------
        Exception
            If any step returns an empty response or encounters an error during execution.

        Notes
        -----
        The spotcheck process:
        1. Executes each pipeline step sequentially
        2. Handles different output types (list, dict, or direct output)
        3. Records intermediate results
        4. Provides progress tracking via tqdm
        5. Prints step outputs for manual verification
        """
        print("\nRunning pipeline spotcheck - This test executes one sample through the pipeline to verify all steps are working correctly\n")
        self.logger.info("Starting pipeline spotcheck execution")
        with tqdm(total=len(self.order), desc=f"Running Pipeline Spotcheck") as pipeline_tracker:
            for idx, step in enumerate(self.order):
                self.logger.debug(f"Executing step {idx+1}/{len(self.order)}: {step.worker.name}")
                pipeline_tracker.set_description(f"Running Pipeline Spotcheck - {idx+1}/{len(self.order)} - {step.worker.name}")
                try:
                    output = step.worker(exp_obj, debug=debug)
                    if isinstance(output, list):
                        step_output = output[0].data[f"{step.worker.name}_output"]
                        print(step_output)
                        
                        self._record_step(output, step.worker.name)
                        if len(output) == 0:
                            self.logger.warning(f"Step {step.worker.name} returned an empty response, most likely due to an error from step inference (possible causes: prompt too long, invalid input format). Run in debug mode to see more details.")
                            raise Exception(f"Step {step.worker.name} returned an empty response, most likely due to an error from step inference. Check if your prompt length exceeds the model's token limit or if the input format is invalid. Run in debug mode to see more details.")
                        exp_obj = output[0]
                        self.logger.debug(f"Step {step.worker.name} returned list output, using first element")
                    elif isinstance(output, dict):
                        step_output = output.data[f"{step.worker.name}_output"]
                        print(step_output)

                        if not step_output:
                            self.logger.warning(f"Step {step.worker.name} returned an empty response, most likely due to an error from step inference")
                            raise Exception(f"Step {step.worker.name} returned an empty response, most likely due to an error from step inference")
                        self._record_step([output], step.worker.name)
                        exp_obj = output
                    else:
                        step_output = output.data[f"{step.worker.name}_output"]
                        print(step_output)
                        self._record_step([output], step.worker.name)
                        exp_obj = output
                    self.logger.debug(f"Step {step.worker.name} completed successfully")
                except Exception as e:
                    self.logger.error(f"Step {step.worker.name} failed with error: {str(e)}")
                    raise e
                pipeline_tracker.update(1)
        
        self.logger.info("Pipeline spotcheck completed successfully")

    def __call__(
        self, prompt_obj: Union[PromptObject, ExperimentObject, List[PromptObject], List[ExperimentObject]], debug: bool = False
    ):
        """Execute the pipeline on one or more PromptObjects.

        Main entry point for pipeline execution. Handles both single items and batches,
        with proper error handling and result recording. Before processing the full dataset,
        performs validation checks to ensure pipeline configuration is correct.

        Args:
            prompt_obj (Union[PromptObject, ExperimentObject, List[PromptObject], List[ExperimentObject]]): 
                The input(s) to process. Can be either a single object or a list for batch processing.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.

        Returns:
            List[PromptObject]: The results of pipeline execution. Always returns a list,
                even for single inputs.

        Raises:
            AssertionError: If batch input contains non-PromptObject items
            ValueError: If pipeline step logic validation fails (missing required input/output keys)
            Exception: If pipeline spotcheck fails (e.g., empty responses from any step)

        Notes:
            The execution process includes three phases:
            1. Pipeline Logic Validation - Verifies that each step's required input keys
               are available from previous steps' outputs
            2. Pipeline Spotcheck - Runs a single sample through the complete pipeline
               to verify functionality of all steps
            3. Full Dataset Processing - Processes all input data through the validated pipeline
        """
        if not debug:
            logging.disable(logging.CRITICAL)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        exp_objs = []
        if isinstance(prompt_obj, PromptObject):
            self.logger.debug(f"Running Agentic Pipeline on data: {prompt_obj.data}...")
            exp_object = ExperimentObject(
                experiment_step=self.order[0],
                data=prompt_obj.data
            )
            exp_objs.append(exp_object)
        elif isinstance(prompt_obj, ExperimentObject):
            self.logger.debug(f"Running Agentic Pipeline on data: {prompt_obj.data}...")
            exp_objs.append(prompt_obj)
        elif isinstance(prompt_obj, List):
            assert isinstance(
                prompt_obj[0], (ExperimentObject, PromptObject)
            ), f"Passed in wrong type: {type(prompt_obj[0])}"
            
            self.logger.debug(
                f"Running Batch Agentic Pipeline on {len(prompt_obj)} items..."
            )
            for obj in prompt_obj:
                if isinstance(obj, PromptObject):
                    obj = ExperimentObject(
                        experiment_step=self.order[0],
                        data=obj.data
                    )
                exp_objs.append(obj)

        # Pipeline Health Check
        self.pipeline_step_logic(exp_objs[0])
        self.pipline_spotcheck(exp_objs[0], debug=debug)

        #Execute Pipeline
        print("\nPipeline spotcheck Completed! - Executing all data through pipeline\n")

        return self.run_pipeline(exp_objs, debug=debug)

    @classmethod
    def from_json(cls,pipeline_config: Dict[str, Any]):
        """Load a pipeline from a JSON file.    

        Args:
            pipeline_config (Dict[str, Any]): The pipeline configuration.

        Returns:
            BaseAgenticPipeline: The loaded pipeline.
        """
        generators = {}
        validators = {}
        for generator_name, generator_config in pipeline_config['generators'].items():
            generators[generator_name] = BaseGenerator.from_json(generator_config)
        for validator_name, validator_config in pipeline_config['validators'].items():
            validators[validator_name] = BaseValidator.from_json(validator_config)

        return BaseAgenticPipeline(
            generators=generators,
            validators=validators,
            order=pipeline_config['execution_order'],
            record_dir=pipeline_config['record_directory'],
            record_step=pipeline_config['record_step'],
            record_results=pipeline_config['record_results']
        )
