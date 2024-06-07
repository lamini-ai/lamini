import logging
from typing import Dict, Iterable, List, Optional, Union

from lamini.api.lamini import Lamini
from lamini.api.lamini_config import get_config
from lamini.api.train import Train

logger = logging.getLogger(__name__)


class MemoryTrainer:
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
            return self.lamini_api.upload_data(items)
        except Exception as e:
            print(f"Error reading data file: {e}")
            raise e

    def train(
        self,
        data_or_dataset_id: Union[
            str, Iterable[Dict[str, Union[int, float, str, bool, Dict, List]]]
        ],
        finetune_args: Optional[dict] = None,
        gpu_config: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
    ):
        if isinstance(data_or_dataset_id, str):
            dataset_id = data_or_dataset_id
        else:
            dataset_id = self.lamini_api.upload_data(
                data_or_dataset_id, is_public=is_public
            )
        self.upload_base_path = self.trainer.get_upload_base_path()["upload_base_path"]
        output = self.trainer.get_existing_dataset(
            dataset_id, self.upload_base_path, is_public
        )
        self.upload_file_path = output["dataset_location"]

        job = self.trainer.precise_train(
            self.model_name,
            dataset_id,
            self.upload_file_path or self.lamini_api.upload_file_path,
            finetune_args,
            gpu_config,
            is_public,
            use_cached_model,
        )
        job["dataset_id"] = dataset_id
        return job

    # Add alias for tune
    tune = train
