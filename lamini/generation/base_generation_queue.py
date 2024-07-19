import json
import logging
import os

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.utils.reservations import create_reservation_api

logger = logging.getLogger(__name__)


class BaseGenerationQueue:
    def __init__(self, api_key, api_url, config):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.reservation_api = None
        self.reservation_polling_task = None
        # TODO: dedup code with base_async_inference_queue once stable
        self.connector = aiohttp.TCPConnector(limit=self.get_max_workers())
        self.client = aiohttp.ClientSession(connector=self.connector)
        self.reservation_api = create_reservation_api(
            self.api_key, self.api_url, self.config
        )

    def get_max_workers(self):
        return lamini.max_workers

    def get_batch_size(self):
        return int(lamini.batch_size)

    def get_retry_limit(self):
        return int(lamini.retry_limit)

    def __del__(self):
        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
