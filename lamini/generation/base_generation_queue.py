import logging
from typing import Optional

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.utils.reservations import Reservations

logger = logging.getLogger(__name__)


class BaseGenerationQueue:
    def __init__(
        self,
        api_key: Optional[str],
        api_url: Optional[str],
        variable_capacity: Optional[bool] = False,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.reservation_polling_task = None
        self.connector = aiohttp.TCPConnector(limit=self.get_max_workers())
        self.client = aiohttp.ClientSession(connector=self.connector)
        self.reservation_api = Reservations(
            self.api_key, self.api_url, variable_capacity
        )

    def get_max_workers(self):
        """Return the Lamini API max number of workers

        Parameters
        ----------
        None

        Returns
        -------
        int
            lamini.max_workers
        """
        return int(lamini.max_workers)

    def get_batch_size(self):
        """Return the Lamini API batch size

        Parameters
        ----------
        None

        Returns
        -------
        int
            lamini.batch_size
        """
        return int(lamini.batch_size)

    def get_retry_limit(self):
        return int(lamini.retry_limit)

    def get_dynamic_max_batch_size(self):
        if lamini.static_batching:
            return self.get_batch_size()

        return self.reservation_api.dynamic_max_batch_size

    def __del__(self):
        """Handle cancelling reservation_polling_task if one is present
        upon deletion of this object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
