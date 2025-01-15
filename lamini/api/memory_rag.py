from typing import List, Optional, Union

import requests

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class MemoryRAG:

    def __init__(
        self,
        model_id: int = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    ):
        self.model_id = model_id
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/alpha/memory-rag"
        self.model_name = model_name

    def memory_index(
        self,
        documents: List,
    ) -> str:
        if self.model_id is None:
            raise Exception("model_id must be set in order to use memory_index")
        payload = {"model_name": self.model_id}

        files = [
            (
                "files",
                (
                    file_path,
                    open(file_path, "rb"),
                ),
            )
            for file_path in documents
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.request(
            "POST",
            self.api_prefix + f"/train",
            headers=headers,
            data=payload,
            files=files,
        )

        return response.json()

    def status(self, job_id: str) -> str:
        if self.model_id is None:
            raise Exception("model_id must be set in order to query")
        params = {"job_id": job_id}
        resp = make_web_request(
            self.api_key, self.api_prefix + f"/status", "post", params
        )
        return resp

    def query(self, prompt: str, k: int = 3) -> str:
        if self.model_id is None:
            raise Exception("model_id must be set in order to query")
        params = {"prompt": prompt, "model_name": self.model_name, "job_id": self.model_id, "rag_query_size": k}
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/completions",
            "post",
            params,
        )
        return resp
    
    def add_index(self, prompt: str) -> str:
        params = {"prompt": prompt, "job_id": self.model_id}
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/add-index",
            "post",
            params,
        )
        return resp
