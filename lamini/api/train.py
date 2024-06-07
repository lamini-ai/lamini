import logging
from typing import Optional

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.api.utils.upload_client import SerializableGenerator

logger = logging.getLogger(__name__)


class Train:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: Optional[dict] = {},
    ):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.ui_url = "https://app.lamini.ai"
        self.model_config = self.config.get("model_config", None)

    def train(
        self,
        model_name: str,
        dataset_id: str,
        upload_file_path: Optional[str] = None,
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        multi_node: Optional[bool] = None,
    ):
        req_data = {"model_name": model_name}
        req_data["dataset_id"] = dataset_id
        if upload_file_path is not None:
            req_data["upload_file_path"] = upload_file_path
        if finetune_args is not None:
            req_data["finetune_args"] = finetune_args
        if gpu_config is not None:
            req_data["gpu_config"] = gpu_config
        if enable_peft is not None:
            req_data["enable_peft"] = enable_peft
        if peft_args is not None:
            req_data["peft_args"] = peft_args
        if is_public is not None:
            req_data["is_public"] = is_public
        if use_cached_model is not None:
            req_data["use_cached_model"] = use_cached_model
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        if multi_node is not None:
            req_data["multi_node"] = multi_node
        url = self.api_prefix + "train"

        job = make_web_request(self.api_key, url, "post", req_data)
        self.job_id = job["job_id"]
        print(
            f"Tuning job submitted! Check status of job {self.job_id} here: {self.ui_url}/train/{self.job_id}"
        )

        return job

    # Add alias for tune
    tune = train

    def precise_train(
        self,
        model_name: str,
        dataset_id: str,
        upload_file_path: Optional[str] = None,
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
    ):
        req_data = {"model_name": model_name}
        req_data["dataset_id"] = dataset_id
        if upload_file_path is not None:
            req_data["upload_file_path"] = upload_file_path
        if finetune_args is not None:
            req_data["finetune_args"] = finetune_args
        if gpu_config is not None:
            req_data["gpu_config"] = gpu_config
        if is_public is not None:
            req_data["is_public"] = is_public
        if use_cached_model is not None:
            req_data["use_cached_model"] = use_cached_model
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        url = self.api_prefix + "precise_train"

        job = make_web_request(self.api_key, url, "post", req_data)
        self.job_id = job["job_id"]
        print(
            f"Tuning job submitted! Check status of job {self.job_id} here: {self.ui_url}/train/{self.job_id}"
        )

        return job

    # Add alias for tune
    precise_tune = precise_train

    def cancel_job(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/cancel"

        return make_web_request(self.api_key, url, "post", {})

    def cancel_all_jobs(
        self,
    ):
        url = self.api_prefix + "train/jobs/cancel"

        return make_web_request(self.api_key, url, "post", {})

    def resume_job(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/resume"

        return make_web_request(self.api_key, url, "post", {})

    def check_job_status(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id)

        return make_web_request(self.api_key, url, "get")

    def get_jobs(self):
        url = self.api_prefix + "train/jobs"

        return make_web_request(self.api_key, url, "get")

    def evaluate(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/eval"

        return make_web_request(self.api_key, url, "get")

    def create_blob_dataset_location(
        self, upload_base_path, dataset_id, is_public, data=None
    ):
        url = self.api_prefix + "data"
        req_data = {
            "upload_base_path": upload_base_path,
            "dataset_id": dataset_id,
        }

        if is_public is not None:
            req_data["is_public"] = is_public

        if data is not None:
            req_data["data"] = data

        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )

    def update_blob_dataset_num_datapoints(self, dataset_id, num_datapoints):
        url = self.api_prefix + "data/num-datapoints"
        req_data = {
            "num_datapoints": num_datapoints,
            "dataset_id": dataset_id,
        }

        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )

    def get_upload_base_path(self):
        url = self.api_prefix + "get-upload-base-path"
        return make_web_request(self.api_key, url, "get")

    def upload_dataset_locally(self, upload_base_path, dataset_id, is_public, data):
        url = self.api_prefix + "local-data"
        req_data = {}
        req_data["upload_base_path"] = upload_base_path
        req_data["dataset_id"] = dataset_id
        req_data["data"] = SerializableGenerator(data)
        if is_public is not None:
            req_data["is_public"] = is_public
        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )

    def get_existing_dataset(self, dataset_id, upload_base_path, is_public):
        url = self.api_prefix + "existing-data"
        req_data = {"dataset_id": dataset_id}
        req_data["upload_base_path"] = upload_base_path
        if is_public is not None:
            req_data["is_public"] = is_public
        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )
