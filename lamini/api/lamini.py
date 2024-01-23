import os
import time
from typing import List, Optional, Union

import jsonlines
import pandas as pd
from lamini.api.async_inference_queue import AsyncInferenceQueue
from lamini.api.inference_queue import InferenceQueue
from lamini.api.lamini_config import get_config
from lamini.api.train import Train
from lamini.api.utils.upload_client import (
    get_dataset_name,
    upload_to_blob,
    upload_to_local,
)


class Lamini:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        max_retries: Optional[int] = 0,
        base_delay: Optional[int] = 10,
        local_cache_file: Optional[str] = None,
        config: dict = {},
    ):
        self.config = get_config(config)
        self.model_name = model_name
        self.inference_queue = InferenceQueue(api_key, api_url, config=config)
        self.async_inference_queue = AsyncInferenceQueue(
            api_key, api_url, config=config
        )

        self.trainer = Train(api_key, api_url, config=config)
        self.upload_file_path = None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.local_cache_file = local_cache_file
        self.model_config = self.config.get("model_config", None)

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[List[str]] = None,
    ):
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )
        retries = 0

        while retries <= self.max_retries:
            try:
                result = self.inference_queue.submit(req_data, self.local_cache_file)
                break
            except Exception as e:
                print(f"Inference exception: {e}")
                delay = self.base_delay * 2**retries
                retries += 1
                if retries > self.max_retries:
                    if self.max_retries > 0:
                        print(f"Max retries {self.max_retries} reached")
                    raise Exception(e)
                print(f"Retrying #{retries} in {delay} seconds...")
                time.sleep(delay)

        if isinstance(prompt, str) and len(result) == 1:
            if output_type is None:
                return result[0]["output"]
            else:
                return result[0]
        return result

    async def async_generate(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[List[str]] = None,
    ):
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )
        retries = 0

        while retries <= self.max_retries:
            try:
                result = await self.async_inference_queue.submit(
                    req_data, self.local_cache_file
                )
                break
            except Exception as e:
                print(f"Inference exception: {e}")
                delay = self.base_delay * 2**retries
                retries += 1
                if retries > self.max_retries:
                    if self.max_retries > 0:
                        print(f"Max retries {self.max_retries} reached")
                    raise e
                print(f"Retrying #{retries} in {delay} seconds...")
                time.sleep(delay)

        if isinstance(prompt, str) and len(result) == 1:
            if output_type is None:
                return result[0]["output"]
            else:
                return result[0]
        return result

    def upload_data(self, data, blob_dir_name="default"):
        if len(data) == 0:
            raise ValueError("Data pairs cannot be empty.")
        # if len(data) > 1e6:
        #     raise ValueError("Data pairs cannot be more than 1 million.")

        dataset_id = get_dataset_name(data)
        output = self.trainer.upload_data(dataset_id, blob_dir_name)
        upload_base_path, dataset_location = (
            output["upload_base_path"],
            output["dataset_location"],
        )
        self.upload_file_path = dataset_location

        try:
            if upload_base_path == "azure":
                upload_to_blob(data, dataset_location)
                print("Data pairs uploaded to blob.")
            else:
                upload_to_local(data, dataset_location)
                print("Data pairs uploaded.")

        except Exception as e:
            print(f"Error uploading data pairs: {e}")
            raise e

    def upload_file(
        self, file_path, input_key: str = "input", output_key: str = "output"
    ):
        if os.path.getsize(file_path) > 2e8:
            raise Exception(
                "File size is too large, please upload file less than 200MB"
            )

        # Convert file records to appropriate format before uploading file
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:
                reader = jsonlines.Reader(dataset_file)
                data = list(reader)

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
        data: Optional[List] = None,
        finetune_args: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
    ):
        if data and len(data) > 3000:
            self.upload_data(data)
            data = None

        return self.trainer.train(
            data,
            self.model_name,
            self.upload_file_path,
            finetune_args,
            enable_peft,
            peft_args,
            is_public,
            use_cached_model,
        )

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
        stop_tokens,
        max_tokens,
    ):
        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["out_type"] = output_type
        if stop_tokens is not None:
            req_data["stop_tokens"] = stop_tokens
        if max_tokens is not None:
            req_data["max_tokens"] = max_tokens
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        return req_data
