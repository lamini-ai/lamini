from typing import List, Optional, Union

import lamini
import numpy as np
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class Embedding:
    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
        model_name: str = None,
        config={},
    ):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/inference/"
        self.model_name = model_name

    def generate(self, prompt: Union[str, List[str]]):
        params = {"prompt": prompt, "model_name": self.model_name}
        resp = make_web_request(
            self.api_key, self.api_prefix + "embedding", "post", params
        )
        embeddings = resp["embedding"]

        if isinstance(prompt, str):
            return np.reshape(embeddings, (1, -1))
        return [np.reshape(embedding, (1, -1)) for embedding in embeddings]
