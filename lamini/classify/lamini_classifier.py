import logging
from typing import List, Optional, Union

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.api.utils.supported_models import LLAMA_31_8B_INST


logger = logging.getLogger(__name__)


class LaminiClassifier:
    def __init__(
        self,
        classifier_name: str,
        model_name: str = LLAMA_31_8B_INST,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v2/classifier"
        self.classifier_name = classifier_name
        self.model_name = model_name
        self.classifier_id = None
        self.initialize_job_id = None
        self.train_job_id = None

    def initialize(self, classes: dict):
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/initialize",
            "post",
            {
                "classes": classes,
                "name": self.classifier_name,
                "model_name": self.model_name,
            },
        )
        self.initialize_job_id = resp["job_id"]
        return resp

    def initialize_status(self):
        resp = make_web_request(
            self.api_key,
            self.api_url + f"/v1/data_generation/{self.initialize_job_id}/status",
            "get",
        )
        return resp

    def train(self):
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/train",
            "post",
            {"name": self.classifier_name},
        )
        self.train_job_id = resp["job_id"]
        return resp

    def train_status(self):
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/{self.train_job_id}/status",
            "get",
        )
        self.classifier_id = resp["model_id"]
        return resp

    # Add alias for tune
    tune = train
    prompt_train = initialize
    create = initialize

    def add(self, dataset_name: str, data: dict):
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/add",
            "post",
            {
                "data": data,
                "dataset_name": dataset_name,
                "project_name": self.classifier_name,
            },
        )
        return resp

    def classify(
        self,
        prompt: Union[str, List[str]],
    ):
        if self.classifier_id is None:
            raise Exception(
                "LaminiClassifier.classifier_id must be set in order to classify. Manually set this or train a new classifier."
            )
        params = {"prompt": prompt}
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/{self.classifier_id}/classify",
            "post",
            params,
        )
        return resp
