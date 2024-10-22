import logging
from typing import Any, Dict, Optional

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.api.utils.upload_client import SerializableGenerator

logger = logging.getLogger(__name__)


class Train:
    """Handler for the training jobs on the Lamini Platform

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
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"

    def train(
        self,
        model_name: str,
        dataset_id: str,
        upload_file_path: Optional[str] = None,
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        is_public: Optional[bool] = None,
        custom_model_name: Optional[str] = None,
    ) -> str:
        """Make a web request to start a training job using the dataset ID provided

        Parameters
        ----------
        model_name: str
            Which model to use from hugging face

        dataset_id: str
            Dataset ID to use for the training job

        upload_file_path: Optional[str] = None

        finetune_args: Optional[dict] = None
            Arguments that are passed into the Trainer.train function

        gpu_config: Optional[dict] = None
            Configuration for the GPUs on the platform

        is_public: Optional[bool] = None
            Allow public access to the model and dataset

        custom_model_name: Optional[str] = None
            A human-readable name for the model.

        Returns
        -------
        job: str
            Job ID on the Platform
        """

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
        if custom_model_name is not None:
            req_data["custom_model_name"] = custom_model_name
        url = self.api_prefix + "train"

        job = make_web_request(self.api_key, url, "post", req_data)
        self.job_id = job["job_id"]
        print(
            f"Tuning job submitted! Check status of job {self.job_id} here: {self.api_url}/train/{self.job_id}"
        )

        return job

    # Add alias for tune
    tune = train

    def cancel_job(self, job_id: str = None) -> Dict[str, Any]:
        """Cancel the job ID provided on the Lamini Platform

        Parameters
        ----------
        job_id: str=None
            Job to be cancelled

        Returns
        -------
        Dict[str, Any]
            Result from the web request
        """

        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/cancel"

        return make_web_request(self.api_key, url, "post", {})

    def cancel_all_jobs(
        self,
    ) -> Dict[str, Any]:
        """Cancel all jobs for this user on the Lamini Platform

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Result from the web request
        """

        url = self.api_prefix + "train/jobs/cancel"

        return make_web_request(self.api_key, url, "post", {})

    def resume_job(self, job_id: str = None) -> Dict[str, Any]:
        """Resume the job ID on the Lamini Platform

        Parameters
        ----------
        job_id: str=None
            Job to be resumed

        Returns
        -------
        Dict[str, Any]
            Result from the web request
        """

        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/resume"

        return make_web_request(self.api_key, url, "post", {})

    def check_job_status(self, job_id: str = None) -> str:
        """Check the specified job on the Lamini platform

        Parameters
        ----------
        job_id: str=None
            Job to check status

        Returns
        -------
        str
            Returned status of the platform job
        """

        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id)

        return make_web_request(self.api_key, url, "get")

    def get_jobs(self) -> Dict[str, Any]:
        """Get all jobs for this user on the Lamini Platform

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        url = self.api_prefix + "train/jobs"

        return make_web_request(self.api_key, url, "get")

    def evaluate(self, job_id: str = None) -> Dict[str, Any]:
        """Run an evaluation job on the specified training job

        Parameters
        ----------
        job_id: str=None
            Job to evaluate

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/eval"

        return make_web_request(self.api_key, url, "get")

    def create_blob_dataset_location(
        self, upload_base_path: str, is_public: bool
    ) -> Dict[str, Any]:
        """Create a blob dataset on the Lamini Platform

        Parameters
        ----------
        upload_base_path: str
            Path for dataset base location

        is_public: bool
            Flag to mark this dataset blog as public

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        url = self.api_prefix + "data"
        req_data = {
            "upload_base_path": upload_base_path,
        }

        if is_public is not None:
            req_data["is_public"] = is_public

        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )

    def update_blob_dataset_num_datapoints(
        self, dataset_id: str, num_datapoints: int
    ) -> Dict[str, Any]:
        """Update an existing blob dataset and datapoints on the Lamini Platform

        Parameters
        ----------
        dataset_id: str
            Dataset to update

        num_datapoints: int
            Number of datapoints to update

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

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

    def get_upload_base_path(self) -> Dict[str, Any]:
        """Get the base path for uploads to the Lamini Platform

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        url = self.api_prefix + "get-upload-base-path"
        return make_web_request(self.api_key, url, "get")

    def upload_dataset_locally(
        self, upload_base_path: str, is_public: bool, data: Any
    ) -> Dict[str, Any]:
        """Upload a local dataset to the Lamini Platform

        Parameters
        ----------
        upload_base_path: str
            Base path on Lamini Platform

        is_public: bool
            Flag to make this data public

        data: Any
            Serializable data set to send in web request

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        url = self.api_prefix + "local-data"
        req_data = {}
        req_data["upload_base_path"] = upload_base_path
        req_data["data"] = SerializableGenerator(data)
        if is_public is not None:
            req_data["is_public"] = is_public
        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )

    def get_existing_dataset(
        self, dataset_id: str, upload_base_path: str
    ) -> Dict[str, Any]:
        """Retrieve the existing dataset on the Lamini Platform

        Parameters
        ----------
        dataset_id: str
            Dataset for which to retrieve

        upload_base_path: str
            Base path on Lamini Platform

        Returns
        -------
        Dict[str, Any]:
            Returned information from the request
        """

        url = self.api_prefix + "existing-data"
        req_data = {"dataset_id": dataset_id}
        req_data["upload_base_path"] = upload_base_path
        return make_web_request(
            self.api_key,
            url,
            "post",
            req_data,
        )
