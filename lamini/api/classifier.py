from typing import List, Optional, Union

import lamini
import requests
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class Classifier:
    """Handler for classification functions of an already trained LLM for classification tasks
    on the Lamini Platform

    Parameters
    ----------
    model_id: int = None
        Tuned Model designation on the Lamini platform

    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the default.
            default = "https://app.lamini.ai"

    """

    def __init__(
        self,
        model_id: int = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/classifier"

    def classify(
        self,
        prompt: Union[str, List[str]],
        top_n: int = None,
        threshold: float = None,
        metadata: bool = None,
    ) -> str:
        """Send a classification request for self.model_id with the provided prompt.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Text prompt for the LLM classifier

        top_n: int = None
            Top N responses from the LLM Classifier, n indicates the limit

        threshold: float = None
            Classifier threshold to indicate a prediction is 'confident' enough
            for a predicted class

        metadata: bool = None
            Boolean flag to request for metadata return from the request

        Raises
        ------
        Exception
            Raised if self.model_id was not set on instantiation. If no model_id
            was provided then no model can be requested for a prediction.

        Returns
        -------
        resp["classification"]: str
            Returned predicted class as a string
        """

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

    def predict(self, prompt: Union[str, List[str]]) -> str:
        """Send a prediction request for self.model_id with the provided prompt.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Text prompt for the LLM classifier

        Raises
        ------
        Exception
            Raised if self.model_id was not set on instantiation. If no model_id
            was provided then no model can be requested for a prediction.

        Returns
        -------
        resp["prediction"]: str
            Returned predicted class as a string
        """

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

    def upload(self, file_path: str) -> None:
        """Upload file to Lamini platform


        Parameters
        ----------
        file_path: str
            Path to file to upload

        Returns
        -------
        None
        """

        files = {"file": open(file_path, "rb")}
        headers = {
            "Authorization": "Bearer " + self.api_key,
        }

        r = requests.post(self.api_prefix, files=files, headers=headers)
        self.model_id = r.json()["model_id"]
