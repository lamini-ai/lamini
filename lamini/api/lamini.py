import json
import logging
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Union

import jsonlines
import pandas as pd
from lamini.api.lamini_config import get_config
from lamini.api.synchronize import sync
from lamini.api.train import Train
from lamini.api.utils.async_inference_queue import AsyncInferenceQueue
from lamini.api.utils.completion import Completion
from lamini.api.utils.upload_client import get_dataset_name, upload_to_blob

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
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 10:
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

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        if isinstance(prompt, str):
            req_data = self.make_llm_req_map(
                prompt=prompt,
                model_name=model_name or self.model_name,
                output_type=output_type,
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens,
            )
            result = self.completion.generate(req_data)
            if output_type is None:
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
    ):
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(prompt, str):
            result = await self.completion.async_generate(req_data)
            if output_type is None:
                result = result["output"]
            return result

        assert isinstance(prompt, list)
        results = await self.async_inference_queue.submit(
            req_data, self.local_cache_file, callback
        )

        if output_type is None:
            results = [single_result["output"] for single_result in results]

        return results

    def upload_data(
        self, data: Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
    ):
        def get_data_str(d):
            for item in d:
                yield json.dumps(item) + "\n"

        if not data:
            raise ValueError("Data pairs cannot be empty.")
        # TODO: check if inside iter empty

        dataset_id = get_dataset_name()
        data_str = get_data_str(data)
        output = self.trainer.create_dataset_location(dataset_id)
        self.upload_base_path, dataset_location = (
            output["upload_base_path"],
            output["dataset_location"],
        )
        self.upload_file_path = dataset_location
        print(
            f"\nYour dataset id is: {dataset_id} . Consider using this in the future to train using the same data. \nEg: "
            f"llm.train(dataset_id='{dataset_id}')"
        )

        try:
            if self.upload_base_path == "azure":
                upload_to_blob(data_str, dataset_location)
                print("Data pairs uploaded to blob.")

        except Exception as e:
            print(f"Error uploading data pairs: {e}")
            raise e

        return dataset_id

    def upload_file(
        self, file_path, input_key: str = "input", output_key: str = "output"
    ):
        if os.path.getsize(file_path) > 2e8:
            raise Exception(
                "File size is too large, please upload file less than 200MB"
            )

        # Convert file records to appropriate format before uploading file
        # TODO: read file in generator
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:
                data = list(jsonlines.Reader(dataset_file))

                for row in data:
                    items.append(
                        {
                            "input": row[input_key],
                            "output": row[output_key] if output_key in row else "",
                        }
                    )

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
                    items.append(
                        {
                            "input": row[input_key],
                            "output": row[output_key] if output_key in row else "",
                        }
                    )
            except KeyError:
                raise ValueError("Each object must have 'input' and 'output' as keys")

        else:
            raise Exception(
                "Upload of only csv and jsonlines file supported at the moment."
            )

        try:
            self.upload_data(items)
        except Exception as e:
            print(f"Error reading data file: {e}")
            raise e

    def train(
        self,
        data: Optional[
            Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ] = None,
        finetune_args: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        dataset_id: Optional[str] = None,
    ):
        if dataset_id:
            output = self.trainer.get_existing_dataset(dataset_id)
            self.upload_base_path, dataset_location = (
                output["upload_base_path"],
                output["dataset_location"],
            )
            self.upload_file_path = dataset_location

        if dataset_id is None and data is not None:
            dataset_id = self.upload_data(data)
            if (
                self.upload_base_path == "azure"
            ):  # if data is uploaded to azure, dont send it with the request
                data = None
        job = self.trainer.train(
            data,
            self.model_name,
            self.upload_file_path,
            finetune_args,
            enable_peft,
            peft_args,
            is_public,
            use_cached_model,
        )
        job["dataset_id"] = dataset_id
        return job

    # continuously poll until the job is completed
    def train_and_wait(
        self,
        data: Optional[List] = None,
        finetune_args: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        **kwargs,
    ):
        job = self.train(
            data,
            finetune_args=finetune_args,
            enable_peft=enable_peft,
            peft_args=peft_args,
            is_public=is_public,
            use_cached_model=use_cached_model,
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

    def cancel_job(self, job_id=None):
        return self.trainer.cancel_job(job_id)

    def cancel_all_jobs(
        self,
    ):
        return self.trainer.cancel_all_jobs()

    def check_job_status(self, job_id=None):
        return self.trainer.check_job_status(job_id)

    def get_jobs(self):
        return self.trainer.get_jobs()

    def evaluate(self, job_id=None):
        return self.trainer.evaluate(job_id)

    def make_llm_req_map(
        self,
        model_name,
        prompt,
        output_type,
        max_tokens,
        max_new_tokens,
    ):
        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["out_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        return req_data
