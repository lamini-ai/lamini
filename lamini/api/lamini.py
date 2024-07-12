import json
import logging
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Union

import jsonlines
import pandas as pd
from lamini.api.lamini_config import get_config
from lamini.api.rest_requests import get_version
from lamini.api.synchronize import sync
from lamini.api.train import Train
from lamini.api.utils.async_inference_queue import AsyncInferenceQueue
from lamini.api.utils.completion import Completion
from lamini.api.utils.upload_client import get_dataset_name, upload_to_blob
from lamini.generation.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class Lamini:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        local_cache_file: Optional[str] = None,
        config: dict = {},
    ):
        self.config = get_config(config)
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        if sys.version_info >= (3, 10):
            logger.info("Using 3.10 InferenceQueue Interface")
            from lamini.api.utils.async_inference_queue_3_10 import (
                AsyncInferenceQueue as AsyncInferenceQueue310,
            )

            self.async_inference_queue = AsyncInferenceQueue310(
                api_key, api_url, config=config
            )
        else:
            self.async_inference_queue = AsyncInferenceQueue(
                api_key, api_url, config=config
            )

        self.completion = Completion(api_key, api_url, config=config)
        self.trainer = Train(api_key, api_url, config=config)
        self.upload_file_path = None
        self.upload_base_path = None
        self.local_cache_file = local_cache_file
        self.model_config = self.config.get("model_config", None)

    def version(self):
        return get_version(self.api_key, self.api_url, self.config)

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[List] = None,
    ):
        if isinstance(prompt, str) or (isinstance(prompt, list) and len(prompt) == 1):
            result = self.completion.generate(
                prompt=prompt,
                model_name=model_name or self.model_name,
                output_type=output_type,
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens,
            )
            if output_type is None:
                if isinstance(prompt, list) and len(prompt) == 1:
                    result = [single_result["output"] for single_result in result]
                else:
                    result = result["output"]
            return result

        assert isinstance(prompt, list)
        return sync(
            self.async_generate(
                prompt=prompt,
                model_name=model_name,
                output_type=output_type,
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens,
                callback=callback,
                metadata=metadata,
            )
        )

    async def async_generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[List] = None,
    ):
        req_data = self.completion.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(prompt, str) or (isinstance(prompt, list) and len(prompt) == 1):
            result = await self.completion.async_generate(req_data)
            if output_type is None:
                if isinstance(prompt, list) and len(prompt) == 1:
                    result = [single_result["output"] for single_result in result]
                else:
                    result = result["output"]
            return result

        assert isinstance(prompt, list)
        if metadata is not None:
            assert isinstance(metadata, list)
            assert len(metadata) == len(prompt)
        results = await self.async_inference_queue.submit(
            req_data,
            self.local_cache_file,
            callback,
            metadata,
            token_optimizer=TokenOptimizer(model_name or self.model_name),
        )

        if output_type is None:
            results = [single_result["output"] for single_result in results]

        return results

    def upload_data(
        self,
        data: Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]],
        is_public: Optional[bool] = None,
    ):
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

        dataset_id = get_dataset_name()

        try:
            if self.upload_base_path == "azure":
                data_str = get_data_str(data)
                output = self.trainer.create_blob_dataset_location(
                    self.upload_base_path, dataset_id, is_public
                )
                self.upload_file_path = output["dataset_location"]
                upload_to_blob(data_str, self.upload_file_path)
                self.trainer.update_blob_dataset_num_datapoints(
                    dataset_id, num_datapoints
                )
                print("Data pairs uploaded to blob.")
            else:
                output = self.trainer.upload_dataset_locally(
                    self.upload_base_path, dataset_id, is_public, data
                )
                self.upload_file_path = output["dataset_location"]
                print("Data pairs uploaded to local.")

            print(
                f"\nYour dataset id is: {dataset_id} . Consider using this in the future to train using the same data. \nEg: "
                f"llm.train(dataset_id='{dataset_id}')"
            )

        except Exception as e:
            print(f"Error uploading data pairs: {e}")
            raise e

        return dataset_id

    def upload_file(
        self, file_path: str, input_key: str = "input", output_key: str = "output"
    ):
        items = self._upload_file_impl(file_path, input_key, output_key)
        try:
            dataset_id = self.upload_data(items)
            return dataset_id
        except Exception as e:
            print(f"Error reading data file: {e}")
            raise e

    def _upload_file_impl(
        self, file_path: str, input_key: str = "input", output_key: str = "output"
    ):
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

    def train(
        self,
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ],
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        multi_node: Optional[bool] = None,
    ):
        if isinstance(data_or_dataset_id, str):
            dataset_id = data_or_dataset_id
        else:
            dataset_id = self.upload_data(data_or_dataset_id, is_public=is_public)
        assert dataset_id is not None
        base_path = self.trainer.get_upload_base_path()
        self.upload_base_path = base_path["upload_base_path"]
        existing_dataset = self.trainer.get_existing_dataset(
            dataset_id, self.upload_base_path, is_public
        )
        self.upload_file_path = existing_dataset["dataset_location"]

        job = self.trainer.train(
            model_name=self.model_name,
            dataset_id=dataset_id,
            upload_file_path=self.upload_file_path,
            finetune_args=finetune_args,
            gpu_config=gpu_config,
            enable_peft=enable_peft,
            peft_args=peft_args,
            is_public=is_public,
            use_cached_model=use_cached_model,
            multi_node=multi_node,
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
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        multi_node: Optional[bool] = None,
        **kwargs,
    ):
        job = self.train(
            data_or_dataset_id,
            finetune_args=finetune_args,
            gpu_config=gpu_config,
            enable_peft=enable_peft,
            peft_args=peft_args,
            is_public=is_public,
            use_cached_model=use_cached_model,
            multi_node=multi_node,
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

    def cancel_job(self, job_id=None):
        return self.trainer.cancel_job(job_id)

    def cancel_all_jobs(
        self,
    ):
        return self.trainer.cancel_all_jobs()

    def resume_job(self, job_id=None):
        return self.trainer.resume_job(job_id)

    def check_job_status(self, job_id=None):
        return self.trainer.check_job_status(job_id)

    def get_jobs(self):
        return self.trainer.get_jobs()

    def evaluate(self, job_id=None):
        return self.trainer.evaluate(job_id)
