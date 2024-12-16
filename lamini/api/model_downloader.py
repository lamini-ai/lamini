"""
A class to handle downloading models from the Lamini Platform.
"""

import enum
from typing import List

from lamini.api.rest_requests import make_web_request


class DownloadedModel:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    model_id = None
    model_name = None
    model_type = None
    user_id = None
    is_public = None
    creation_ts = None
    prev_download_ts = None
    prev_download_error = None
    download_attempts = None
    status = None

    def __repr__(self):
        return f"<DownloadedModel({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})>"


class ModelType(enum.Enum):
    """This must be consistent with the db/migrations table definition's MODEL_TYPE type."""

    transformer = "transformer"
    embedding = "embedding"


class ModelDownloader:
    """Handler for requesting Lamini Platform to download a hugging face model.

    Parameters
    ----------
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
        api_key: str,
        api_url: str,
    ):
        self.api_key = api_key
        self.api_endpoint = api_url + "/v1alpha/downloaded_models/"

    def download(self, hf_model_name: str, model_type: ModelType) -> DownloadedModel:
        """Request to Lamini platform for an embedding encoding of the provided
        prompt


        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to encoding into an embedding

        Returns
        -------
        DownloadedModel:
            A object describing the state of the model.
        """

        params = {"hf_model_name": hf_model_name, "model_type": model_type.value}
        resp = make_web_request(self.api_key, self.api_endpoint, "post", params)
        return DownloadedModel(**resp)

    def list(self) -> List[DownloadedModel]:
        """List all models on the Lamini Platform.

        Returns
        -------
        List[DownloadedModel]:
            A object describing the state of the model.
        """
        resp = make_web_request(self.api_key, self.api_endpoint, "get")
        res = []
        for model in resp:
            res.append(DownloadedModel(**model))
        return res
