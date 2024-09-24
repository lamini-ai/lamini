import asyncio
import datetime
import logging
import time
from typing import Optional

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request

logger = logging.getLogger(__name__)


class Reservations:
    """Handler for API reservations endpoint.


    Parameters
    ----------
    api_key: Optinal[str] = None
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str] = None
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults

    variable_capacity: Optional[bool] = False

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        variable_capacity: Optional[bool] = False,
    ) -> None:
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/reservation"
        self.current_reservation = None
        self.capacity_remaining = 0
        self.capacity_needed = 0
        self.dynamic_max_batch_size = 0
        self.condition = asyncio.Condition()
        self.is_working = False
        self.polling_task = None
        self.poll_for_reservation = asyncio.Event()
        self.is_polling = False
        self.variable_capacity = variable_capacity

    def initialize_reservation(
        self, capacity: int, model_name: str, batch_size: int, max_tokens: Optional[int]
    ) -> None:
        """Submit post request to the reservations endpoint and store the
        reservation metadata within this object.

        Parameters
        ----------
        capacity: int
            Reservation capactiy

        model_name: str
            Model to use for the reserved request

        batch_size: int
            Batch size for the inference call

        max_tokens: Optional[int]
            Max tokens for the inference call

        Returns
        -------
        None

        Raises
        ------
        Exception
            General exception for reservation issues. The exception is logged
            but execution is continued.
        """
        if lamini.bypass_reservation:
            self.current_reservation = None
            self.capacity_remaining = 0
            self.dynamic_max_batch_size = 0
            self.capacity_needed = 0
            self.model_name = model_name
            self.max_tokens = None
            return

        try:
            logger.info(
                f"Attempt reservation {capacity} {model_name} {batch_size} {max_tokens}"
            )
            reservation = make_web_request(
                self.api_key,
                self.api_prefix,
                "post",
                {
                    "capacity": max(capacity, batch_size),
                    "model_name": model_name,
                    "max_tokens": max_tokens,
                    "batch_size": batch_size,
                },
            )
            logger.info("Made initial reservation " + str(reservation))
            self.current_reservation = reservation
            self.capacity_needed = capacity
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.capacity_remaining = reservation["capacity_remaining"]
            self.dynamic_max_batch_size = min(
                reservation["dynamic_max_batch_size"], reservation["capacity_remaining"]
            )
            if self.variable_capacity:
                self.capacity_needed = self.dynamic_max_batch_size * lamini.max_workers
            self.is_working = True
            self.batch_size = batch_size

        except Exception as e:
            logger.warning(f"Error making reservation, continuing without one. {e}")
            self.current_reservation = None
            self.capacity_remaining = 0
            self.dynamic_max_batch_size = 0
            self.capacity_needed = 0
            self.model_name = model_name
            self.max_tokens = None

    def pause_for_reservation_start(self) -> None:
        """Barrier until specified start time for the reservation

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.current_reservation is None:
            return
        current_time = datetime.datetime.utcnow()
        start_time = datetime.datetime.fromisoformat(
            self.current_reservation["start_time"]
        )
        sleep_time = start_time - current_time
        if sleep_time.total_seconds() > 0:
            time.sleep(sleep_time.total_seconds())

    async def wait_and_poll_for_reservation(
        self, client: aiohttp.ClientSession
    ) -> None:
        """Wait for current reservation to finish and then make a new reservation. If
        this reservation is working (indicated by the self.is_working flag), then
        set the kickoff and timer based polling jobs.

        Parameters
        ----------
        client: aiohttp.ClientSession
            Http Client Handler

        Returns
        -------
        None
        """

        await self.poll_for_reservation.wait()
        self.is_polling = True
        self.poll_for_reservation.clear()
        reservation = await make_async_web_request(
            client,
            self.api_key,
            self.api_prefix,
            "post",
            {
                "capacity": max(self.capacity_needed, self.batch_size),
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
                "batch_size": self.batch_size,
            },
        )
        logger.info("Made reservation " + str(reservation))
        self.current_reservation = reservation
        self.capacity_remaining = reservation["capacity_remaining"]
        self.dynamic_max_batch_size = reservation["dynamic_max_batch_size"]
        if self.variable_capacity:
            self.capacity_needed = self.dynamic_max_batch_size * lamini.max_workers
        async with self.condition:
            self.condition.notify(len(self.condition._waiters))
        self.is_polling = False
        self.polling_task = asyncio.create_task(
            self.kickoff_reservation_polling(client)
        )

    async def timer_based_polling(self, wakeup_time: int) -> None:
        """Wait for the provided wakeup_time to run the polling for the
        current reservation.

        Parameters
        ----------
        wakeup_time: int
            ISO format datetime

        Returns
        -------
        None
        """

        try:
            current_time = datetime.datetime.utcnow()
            end_time = datetime.datetime.fromisoformat(wakeup_time)
            sleep_time = end_time - current_time
            if sleep_time.total_seconds() > 0:
                logger.debug("timer_based_polling sleep time: " + str(sleep_time))
                await asyncio.sleep(sleep_time.total_seconds())
                self.poll_for_reservation.set()
        except asyncio.CancelledError:
            logger.debug("Task was cancelled")

    async def kickoff_reservation_polling(self, client: aiohttp.ClientSession) -> None:
        """If a current reservation is present, then kickoff the polling for this
        reservation. If an error occurrs, the reservation is set to None and the
        polling task is cancelled.

        Parameters
        ----------
        client: aiohttp.ClientSession
            Http Session handler

        Returns
        -------
        None
        """

        if self.current_reservation is None:
            return None
        try:
            await self.wait_and_poll_for_reservation(client)
        except:
            self.current_reservation = None
            if self.polling_task is not None:
                self.polling_task.cancel()
            return None

    async def async_pause_for_reservation_start(self) -> None:
        """Sleep until start of the current reseravtion

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.current_reservation is None:
            return
        current_time = datetime.datetime.utcnow()
        start_time = datetime.datetime.fromisoformat(
            self.current_reservation["start_time"]
        )
        sleep_time = start_time - current_time
        if sleep_time.total_seconds() > 0:
            await asyncio.sleep(sleep_time.total_seconds())

    def update_capacity_use(self, queries: int) -> None:
        """Decrease the self.capacity_remaining param by the int queries

        Parameters
        ----------
        queries: int
            Quantity of queries to decrease from self.capacity_remaining

        Returns
        -------
        None
        """

        if self.current_reservation is None:
            return
        self.capacity_remaining -= queries

    def update_capacity_needed(self, queries: int) -> None:
        """Decrease the self.capacity_needed param by the int queries

        Parameters
        ----------
        queries: int
            Quantity of queries to decrease from self.capacity_needed

        Returns
        -------
        None
        """

        if self.current_reservation is None:
            return
        self.capacity_needed -= queries

    def __del__(self) -> None:
        """Handler for object deletion, jobs cancelled when __del__ is called"""
        if self.polling_task is not None:
            self.polling_task.cancel()
