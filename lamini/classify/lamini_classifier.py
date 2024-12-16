import logging
from typing import List, Optional, Union

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.api.utils.supported_models import LLAMA_31_8B_INST
import os

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
        self.train_job_id = None

    def initialize(self, classes: dict, examples: dict):
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/initialize",
            "post",
            {
                "classes": classes,
                "examples": examples,
                "name": self.classifier_name,
                "model_name": self.model_name,
            },
        )
        self.train_job_id = resp["job_id"]
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

    def download_dataset(self, dataset_name: str, output_dir_path: str = None):
        """Download a dataset and save it to disk"""
        project_name = self.classifier_name
        url = f"{self.api_prefix}/{project_name}/{dataset_name}/download"
        filename = f"{dataset_name}.jsonl"

        # Make the request to the API
        resp = make_web_request(
            self.api_key,
            url,
            "get",
            stream=True  # Enable streaming for large files
        )
        
        # Check if the request was successful
        if resp.status_code != 200:
            logger.error(f"Failed to download file: {resp.status_code} - {resp.text}")
            resp.raise_for_status()  # Raise exception for non-200 status codes

        # Set default output path if not provided
        if output_dir_path is None:
            output_dir_path = "./"
        else:
            # Ensure directories in output_dir_path exist
            os.makedirs(os.path.dirname(output_dir_path), exist_ok=True)
            
        # Set full path to the output file
        output_path = os.path.join(output_dir_path, filename)
        logger.debug(f"Downloading to {output_path}")

        # Write the file content to disk
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        logger.debug(f"File successfully downloaded to {output_path}")
        return output_path

    def delete_dataset(self, dataset_name: str):
        project_name = self.classifier_name
        url = f"{self.api_prefix}/{project_name}/{dataset_name}"
        resp = make_web_request(
            self.api_key,
            url,
            "delete",
        )
        return resp
