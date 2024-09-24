from typing import List, Union

import lamini
import numpy as np
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class Embedding:
    """Handler for embedding requests to the Lamini Platform


    Parameters
    ----------
    model_name: str = None
        LLM hugging face ID, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
        model_name: str = None,
        api_key: str = None,
        api_url: str = None,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.model_name = model_name

    def generate(self, prompt: Union[str, List[str]]) -> List[np.ndarray]:
        """Request to Lamini platform for an embedding encoding of the provided
        prompt


        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to encoding into an embedding

        Returns
        -------
        List[np.ndarray]
            Formatted returned embedding from the Lamini platform
        """

        params = {"prompt": prompt, "model_name": self.model_name}
        resp = make_web_request(
            self.api_key, self.api_prefix + "embedding", "post", params
        )
        embeddings = resp["embedding"]

        if isinstance(prompt, str):
            return np.reshape(embeddings, (1, -1))
        return [np.reshape(embedding, (1, -1)) for embedding in embeddings]
