import base64
import enum
import json
import logging
import os
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import jsonlines
import pandas as pd
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.model_downloader import ModelDownloader, ModelType, DownloadedModel
from lamini.api.rest_requests import get_version, make_web_request
from lamini.api.train import Train
from lamini.api.utils.completion import Completion
from lamini.api.utils.sql_completion import SQLCompletion
from lamini.api.utils.sql_token_cache import SQLTokenCache
from lamini.api.utils.upload_client import upload_to_blob
from lamini.error.error import DownloadingModelError

logger = logging.getLogger(__name__)


class Lamini:
    """Main interface for Lamini platform functionality. Key features are:
        1. Generation calls
        2. Data Upload/Downloading
        3. Training orchestration
        4. Evaluation

    Parameters
    ----------
    model_name: str = None
        LLM hugging face ID

    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_type: ModelType = ModelType.transformer,
    ):
        self.config = get_config()
        api_key = api_key or lamini.api_key or get_configured_key(self.config)
        api_url = api_url or lamini.api_url or get_configured_url(self.config)

        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.completion = Completion(api_key, api_url)
        self.sql_completion = SQLCompletion(api_key, api_url)
        self.sql_token_cache = SQLTokenCache(api_key, api_url)
        self.trainer = Train(api_key, api_url)
        self.upload_file_path = None
        self.upload_base_path = None
        self.model_downloader = ModelDownloader(api_key, api_url)
        self.model_type = model_type

    def version(self) -> str:
        """Get the version of the Lamini platform
        Parameters
        ----------
        None
        Returns
        -------
        str
            Returned version fo the platform
        """
        return get_version(self.api_key, self.api_url, self.config)

    def generate_sql(
        self,
        prompt: Union[str, List[str]],
        cache_id: str,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[str, Dict[str, Any]]:
        result = self.sql_completion.generate(
            prompt=prompt,
            cache_id=cache_id,
            model_name=model_name or self.model_name,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        return result

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Generation request to the LLM with the provided prompt.
        Model name will specify which LLM from hugging face to use.
        Output type is used to handle structured output of the response.
        max_tokens and max_new_tokens are related to the total amount of tokens
        the model can use and generate. max_new_tokens is recommended to be used
        over max_tokens to adjust model output.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        output_type: Optional[dict] = None
            Structured output format

        max_tokens: Optional[int] = None
            Max number of tokens for the model's generation

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        Raises
        ------
        DownloadingModelError
            Raised when an issue occurs with the model_name provided has failed to download

        Returns
        -------
        result: Union[str, Dict[str, Any]]
            Generated response from the LLM, strings are returned when output_type is not
            specified, otherwise a dictionary matching the output_type is returned.
        """

        result = self.completion.generate(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        if output_type is None:
            if isinstance(prompt, list):
                result = [single_result["output"] for single_result in result]
            else:
                result = result["output"]
        return result

    async def async_generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """Asynchronous call for a generation request to the LLM with the provided prompt.
        Model name will specify which LLM from hugging face to use.
        Output type is used to handle structured output of the response.
        max_tokens and max_new_tokens are related to the total amount of tokens
        the model can use and generate. max_new_tokens is recommended to be used
        over max_tokens to adjust model output.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        output_type: Optional[dict] = None
            Structured output format

        max_tokens: Optional[int] = None
            Max number of tokens for the model's generation

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        Raises
        ------
        DownloadingModelError
            Raised when an issue occurs with the model_name provided has failed to download

        Returns
        -------
        result: Union[str, Dict[str, Any]]
            Generated response from the LLM, strings are returned when output_type is not
            specified, otherwise a dictionary matching the output_type is returned.
        """

        req_data = self.completion.make_llm_req_map(
            prompt=prompt,
            cache_id=cache_id,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        result = await self.completion.async_generate(req_data)
        if output_type is None:
            if isinstance(prompt, list):
                result = [single_result["output"] for single_result in result]
            else:
                result = result["output"]
        return result

    def upload_data(
        self,
        data: Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]],
        is_public: Optional[bool] = None,
    ) -> str:
        """Upload the provide data to the Lamini Platform

        Parameters
        ----------
        data: Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
            Data to upload

        is_public: Optional[bool] = None
            Flag to indicate if the platform should allow the dataset to be
            publically shared.

        Raises
        ------
        ValueError
            Raised in data is None

        Exception
            Raised if there was a failure during upload

        Returns
        -------
        str
            Dataset designation within the platform
        """

        num_datapoints = 0

        def get_data_str(d):
            nonlocal num_datapoints
            for item in d:
                num_datapoints += 1
                yield json.dumps(item) + "\n"

        if not data:
            raise ValueError("Data pairs cannot be empty.")

        output = self.trainer.get_upload_base_path()
        self.upload_base_path = output["upload_base_path"]

        try:
            if self.upload_base_path == "azure":
                data_str = get_data_str(data)
                response = self.trainer.create_blob_dataset_location(
                    self.upload_base_path, is_public
                )
                self.upload_file_path = response["dataset_location"]
                upload_to_blob(data_str, self.upload_file_path)
                self.trainer.update_blob_dataset_num_datapoints(
                    response["dataset_id"], num_datapoints
                )
                print("Data pairs uploaded to blob.")
            else:
                response = self.trainer.upload_dataset_locally(
                    self.upload_base_path, is_public, data
                )
                self.upload_file_path = response["dataset_location"]
                print("Data pairs uploaded to local.")

            print(
                f"\nYour dataset id is: {response['dataset_id']} . Consider using this in the future to train using the same data. \nEg: "
                f"llm.train(data_or_dataset_id='{response['dataset_id']}')"
            )

        except Exception as e:
            print(f"Error uploading data pairs: {e}")
            raise e

        return response["dataset_id"]

    def upload_file(
        self, file_path: str, input_key: str = "input", output_key: str = "output"
    ) -> None:
        """Upload a provided file to the Lamini Platform

        Parameters
        ----------
        file_path: str
            File path location to upload

        input_key: str = "input"
            Key of the json dictionary to use as the input

        output_key: str = "output"
            Key of the json dictionary to use as the output

        Raises
        ------
        Exception
            Raised if there is an issue with upload

        Returns
        -------
        None
        """

        items = self._upload_file_impl(file_path, input_key, output_key)
        try:
            dataset_id = self.upload_data(items)
            return dataset_id
        except Exception as e:
            print(f"Error reading data file: {e}")
            raise e

    def _upload_file_impl(
        self, file_path: str, input_key: str = "input", output_key: str = "output"
    ) -> Generator[Dict[str, Any], None, None]:
        """Private function to handle file types and loading for upload_file

        Parameters
        ----------
        file_path: str
            File path location to upload

        input_key: str = "input"
            Key of the json dictionary to use as the input

        output_key: str = "output"
            Key of the json dictionary to use as the output

        Raises
        ------
        ValueError
            Raised if input_key is not within the file contents provided

        KeyError
            Raises if input_key or output_key is not within the file contents provided

        Exception
            If a file type outside of csv or jsonlines is provided

        Yields
        -------
        items: Dict[str, Any]
            Contents of the file provided
        """

        if os.path.getsize(file_path) > 1e10:
            raise Exception("File size is too large, please upload file less than 10GB")

        # Convert file records to appropriate format before uploading file
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:

                for row in jsonlines.Reader(dataset_file):
                    yield {"input": row[input_key], "output": row.get(output_key, "")}

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
                    yield {
                        "input": row[input_key],
                        "output": row.get(output_key, ""),
                    }
            except KeyError:
                raise ValueError("Each object must have 'input' and 'output' as keys")

        else:
            raise Exception(
                "Upload of only csv and jsonlines file supported at the moment."
            )
        return items

    def download_model(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        wait: bool = False,
        wait_time_seconds: int = 60,
    ) -> DownloadedModel:
        """Request Lamini Platform to download and cache the specified hugging face model.
        So that the model can be immediately loaded to GPU memory afterwards.
        Right now, only support downloading models from Hugging Face.

        Parameters
        ----------
        hf_model_name: str
            The full name of a hugging face model. Like meta-llama/Llama-3.2-11B-Vision-Instruct
            in https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

        model_type: ModelType
            The type of the requested model.

        Raises
        ------
        Exception
            Raised if there is an issue with upload

        Returns
        -------
        DownloadedModel
        """
        model_name_to_download = self.model_name if model_name is None else model_name
        model_type_to_download = self.model_type if model_type is None else model_type
        if not wait:
            return self.model_downloader.download(
                model_name_to_download, model_type_to_download
            )

        start_time = time.time()

        while True:
            result = self.model_downloader.download(
                model_name_to_download, model_type_to_download
            )

            # Check the status of foo()'s result
            if result.status == "available":
                return result

            # Check if the specified timeout has been exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > wait_time_seconds:
                return result
            INTERVAL_SECONDS = 1
            time.sleep(INTERVAL_SECONDS)

    def add_sql_token_cache(
            self,
            col_val_file: Optional[str] = None,
            wait: bool = False,
            wait_time_seconds: int = 600,
    ):
        col_val_str = None

        if col_val_file:
            with open(col_val_file, 'r') as f:
                col_vals = json.load(f)
                # TODO: in another PR, limit size of col_vals dict
                col_val_str = json.dumps(col_vals)

        start_time = time.time()

        while True:
            res = self.sql_token_cache.add_token_cache(
                base_model_name=self.model_name,
                col_vals=col_val_str,
            )

            if not wait:
                return res
            if res["status"] == "done":
                return res
            elif res["status"] == "failed":
                raise Exception("SQL token cache build failed")

            elapsed_time = time.time() - start_time
            if elapsed_time > wait_time_seconds:
                return res
            INTERVAL_SECONDS = 1
            time.sleep(INTERVAL_SECONDS)

    def delete_sql_token_cache(self, cache_id):
        while True:
            res = self.sql_token_cache.delete_token_cache(cache_id)

            if res["status"] == "done":
                return res
            elif res["status"] == "failed":
                raise Exception("SQL token cache deletion failed")

            INTERVAL_SECONDS = 1
            time.sleep(INTERVAL_SECONDS)

    def list_models(self) -> List[DownloadedModel]:
        return self.model_downloader.list()

    def train(
        self,
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ],
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        is_public: Optional[bool] = None,
        custom_model_name: Optional[str] = None,
    ) -> str:
        """Handler for training jobs through the Trainer object. This submits a training
        job request to the platform using the provided data.

        Parameters
        ----------
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ]
            Data or Id to use for the training job

        finetune_args: Optional[dict] = None
            Arguments that are passed into the Trainer.train function

        gpu_config: Optional[dict] = None
            Configuration for the GPUs on the platform

        is_public: Optional[bool] = None
            Allow public access to the model and dataset

        custom_model_name: Optional[str] = None
            A human-readable name for the model.

        Raises
        ------
        AssertionError
            Raises if dataset_id is None, a dataset_id is generated when data is provided
            to this function instead of an id

        Returns
        -------
        job: str
            Job id for the train job on the platform
        """

        if isinstance(data_or_dataset_id, str):
            dataset_id = data_or_dataset_id
        else:
            dataset_id = self.upload_data(data_or_dataset_id, is_public=is_public)
        assert dataset_id is not None
        base_path = self.trainer.get_upload_base_path()
        self.upload_base_path = base_path["upload_base_path"]
        existing_dataset = self.trainer.get_existing_dataset(
            dataset_id, self.upload_base_path
        )
        self.upload_file_path = existing_dataset["dataset_location"]

        job = self.trainer.train(
            model_name=self.model_name,
            dataset_id=dataset_id,
            upload_file_path=self.upload_file_path,
            finetune_args=finetune_args,
            gpu_config=gpu_config,
            is_public=is_public,
            custom_model_name=custom_model_name,
        )
        job["dataset_id"] = dataset_id
        return job

    # Add alias for tune
    tune = train

    # continuously poll until the job is completed
    def train_and_wait(
        self,
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ],
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        is_public: Optional[bool] = None,
        custom_model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Handler for training jobs through the Trainer object. This submits a training
        job request to the platform using the provided data. This differs from the train
        function in that this function will continuously poll until the job is completed.

        Parameters
        ----------
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ]
            Data or Id to use for the training job

        finetune_args: Optional[dict] = None
            Arguments that are passed into the Trainer.train function

        gpu_config: Optional[dict] = None
            Configuration for the GPUs on the platform

        is_public: Optional[bool] = None
            Allow public access to the model and dataset

        custom_model_name: Optional[str] = None
            A human-readable name for the model.

        kwargs: Dict[str, Any]
            Key word arguments
                verbose
                    output text indicating the job is still runing

        Raises
        ------
        KeyboardInterrupt
            Raised when keyboard interrupt is called

        Returns
        -------
        status: str
            Job status on the platform
        """

        job = self.train(
            data_or_dataset_id,
            finetune_args=finetune_args,
            gpu_config=gpu_config,
            is_public=is_public,
            custom_model_name=custom_model_name,
        )

        try:
            status = self.check_job_status(job["job_id"])
            if status["status"] == "FAILED":
                print(f"Job failed: {status}")
                return status

            while status["status"] not in (
                "COMPLETED",
                "PARTIALLY COMPLETED",
                "FAILED",
                "CANCELLED",
            ):
                if kwargs.get("verbose", False):
                    print(f"job not done. waiting... {status}")
                time.sleep(30)
                status = self.check_job_status(job["job_id"])
                if status["status"] == "FAILED":
                    print(f"Job failed: {status}")
                    return status
                elif status["status"] == "CANCELLED":
                    print(f"Job canceled: {status}")
                    return status
            print(
                f"Finetuning process completed, model name is: {status['model_name']}"
            )
        except KeyboardInterrupt as e:
            print("Cancelling job")
            return self.cancel_job(job["job_id"])

        return status

    # Add alias for tune
    tune_and_wait = train_and_wait

    def cancel_job(self, job_id: str = None) -> str:
        """Cancel to job specified by the id

        Parameters
        ----------
        job_id: str=None
            job id to cancel

        Returns
        -------
        str
            Output from platform of the confirming cancelling of the job
        """

        return self.trainer.cancel_job(job_id)

    def cancel_all_jobs(
        self,
    ) -> str:
        """Cancel all jobs from this user on the platform

        Parameters
        ----------
        None

        Returns
        -------
        str
            Output from platform of the confirming cancelling of the job
        """

        return self.trainer.cancel_all_jobs()

    def resume_job(self, job_id: str = None) -> str:
        """Resume the specific job on the Lamini platform

        Parameters
        ----------
        job_id: str=None
            Job to be resumed

        Returns
        -------
        str:
            Returned status of the platform for the job
        """

        return self.trainer.resume_job(job_id)

    def check_job_status(self, job_id: str = None) -> str:
        """Check the specified job on the Lamini platform

        Parameters
        ----------
        job_id: str=None
            Job to check status

        Returns
        -------
        str
            Returned status of the platform job
        """

        return self.trainer.check_job_status(job_id)

    def get_jobs(self) -> List[str]:
        """Get all jobs for this user on the Lamini Platform

        Parameters
        ----------
        None

        Returns
        -------
        List[str]:
            Returned list of all jobs
        """

        return self.trainer.get_jobs()

    def evaluate(self, job_id: str = None) -> str:
        """Run an evaluation job on the specified training job

        Parameters
        ----------
        job_id: str=None
            Job to evaluate

        Returns
        -------
        str:
            Status of the job on the platform
        """

        return self.trainer.evaluate(job_id)
