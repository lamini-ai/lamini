import time
from typing import List, Union

import lamini
import requests
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class Classifier:
    def __init__(
        self, model_id: int = None, api_key: str = None, api_url: str = None, config={}
    ):
        self.model_id = model_id
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/classifier"

    def classify(
        self,
        prompt: Union[str, List[str]],
        top_n: int = None,
        threshold: float = None,
        metadata: bool = None,
    ):
        if self.model_id is None:
            raise Exception(
                "model_id must be set in order to classify. Upload a model or set an existing model_id"
            )
        params = {"prompt": prompt}
        if top_n:
            params["top_n"] = top_n
        if threshold:
            params["threshold"] = threshold
        if metadata:
            params["metadata"] = metadata
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/{self.model_id}/classification",
            "post",
            params,
        )
        return resp["classification"]

    def predict(self, prompt: Union[str, List[str]]):
        if self.model_id is None:
            raise Exception(
                "model_id must be set in order to classify. Upload a model or set an existing model_id"
            )
        params = {"prompt": prompt}
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/{self.model_id}/prediction",
            "post",
            params,
        )
        return resp["prediction"]

    def upload(self, file_path: str):
        files = {"file": open(file_path, "rb")}
        headers = {
            "Authorization": "Bearer " + self.api_key,
        }

        r = requests.post(self.api_prefix, files=files, headers=headers)
        self.model_id = r.json()["model_id"]
