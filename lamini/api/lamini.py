import time

from typing import List, Optional, Union
from lamini.api.inference_queue import InferenceQueue
from lamini.api.train import Train
from lamini.api.lamini_config import get_config
from lamini.api.utils.upload_client import (
    upload_to_blob,
    upload_to_local,
    get_dataset_name,
)
import json
import os


class Lamini:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
    ):
        self.config = get_config(config)
        self.model_name = model_name
        self.inference_queue = InferenceQueue(api_key, api_url, config=config)
        self.trainer = Train(api_key, api_url, config=config)
        self.upload_file_path = None
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
        result = self.inference_queue.submit(req_data)
        if isinstance(prompt, str) and len(result) == 1:
            if output_type is None:
                return result[0]["output"]
            else:
                return result[0]
        return result

    def upload_data(self, data_pairs, blob_dir_name="default"):
        if len(data_pairs) == 0:
            raise ValueError("Data pairs cannot be empty.")
        if len(data_pairs) > 1e+6:
            raise ValueError("Data pairs cannot be more than 1 million.")

        dataset_id = get_dataset_name(data_pairs)
        output = self.trainer.upload_data(dataset_id, blob_dir_name)
        upload_base_path, dataset_location = (
            output["upload_base_path"],
            output["dataset_location"],
        )
        self.upload_file_path = dataset_location

        try:
            if upload_base_path == "azure":
                upload_to_blob(data_pairs, dataset_location)
                print("Data pairs uploaded to blob.")
            else:
                upload_to_local(data_pairs, dataset_location)
                print("Data pairs uploaded.")

        except Exception as e:
            print(f"Error uploading data pairs: {e}")
            raise e

    def upload_file(self, file_path, blob_dir_name="default"):
        if os.path.getsize(file_path) > 1e7:
            raise Exception(
                "File size is too large, please upload a file less than 10MB."
            )
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            self.upload_data(data, blob_dir_name)
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

            while status["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
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
