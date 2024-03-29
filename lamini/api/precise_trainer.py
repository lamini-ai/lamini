import logging
from typing import Dict, Iterable, List, Optional, Union

from lamini.api.lamini import Lamini
from lamini.api.lamini_config import get_config
from lamini.api.train import Train

logger = logging.getLogger(__name__)


class PreciseTrainer:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        local_cache_file: Optional[str] = None,
        config: dict = {},
    ):
        self.config = get_config(config)
        self.model_name = model_name
        self.trainer = Train(api_key, api_url, config=config)
        self.lamini_api = Lamini(api_key, api_url, config=config)
        self.upload_file_path = None
        self.upload_base_path = None
        self.local_cache_file = local_cache_file
        self.model_config = self.config.get("model_config", None)

    def upload_file(
        self, file_path, input_key: str = "input", output_key: str = "output"
    ):
        items = self.lamini_api._upload_file_impl(file_path, input_key, output_key)
        try:
            self.lamini_api.upload_data(items)
        except Exception as e:
            print(f"Error reading data file: {e}")
            raise e

    def train(
        self,
        data: Optional[
            Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ] = None,
        finetune_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        dataset_id: Optional[str] = None,
    ):
        if dataset_id:
            self.upload_base_path = self.trainer.get_upload_base_path()[
                "upload_base_path"
            ]
            output = self.trainer.get_existing_dataset(
                dataset_id, self.upload_base_path, is_public
            )
            self.upload_file_path = output["dataset_location"]

        if dataset_id is None and data is not None:
            dataset_id = self.lamini_api.upload_data(data, is_public)
            if (
                self.upload_base_path == "azure"
            ):  # if data is uploaded to azure, dont send it with the request
                data = None
        job = self.trainer.precise_train(
            data,
            self.model_name,
            self.upload_file_path or self.lamini_api.upload_file_path,
            finetune_args,
            is_public,
            use_cached_model,
            dataset_id,
        )
        job["dataset_id"] = dataset_id
        return job
