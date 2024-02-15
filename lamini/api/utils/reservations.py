import asyncio
import datetime
import logging
import time
from typing import Optional

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request

logger = logging.getLogger(__name__)

reservation_api = None


def create_reservation_api(api_key, api_url, config):
    global reservation_api
    if reservation_api is None:
        reservation_api = Reservations(api_key, api_url, config)
    return reservation_api


def get_reservation_api():
    global reservation_api
    assert reservation_api is not None
    return reservation_api


class Reservations:
    def __init__(self, api_key: str = None, api_url: str = None, config={}):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/reservation"
        self.current_reservation = None
        self.capacity_remaining = 0
        self.capacity_needed = 0
        self.condition = asyncio.Condition()
        self.is_working = False
        self.polling_task = None
        self.poll_for_reservation = asyncio.Event()
        self.is_polling = False

    def initialize_reservation(
        self, capacity: int, model_name: str, max_tokens: Optional[int]
    ):
        try:
            reservation = make_web_request(
                self.api_key,
                self.api_prefix,
                "post",
                {
                    "capacity": capacity,
                    "model_name": model_name,
                    "max_tokens": max_tokens,
                },
            )
            logger.debug("Made reservation " + str(reservation))
            self.current_reservation = reservation
            self.capacity_needed = capacity
            self.model_name = model_name
            self.max_tokens = max_tokens
            self.capacity_remaining = reservation["capacity_remaining"]
            self.is_working = True
        except Exception as e:
            logger.debug(f"Error making reservation, continuing without one. {e}")
            self.current_reservation = None
            self.capacity_remaining = 0
            self.capacity_needed = 0
            self.model_name = model_name
            self.max_tokens = None

    def pause_for_reservation_start(self):
        if self.current_reservation is None:
            return
        current_time = datetime.datetime.utcnow()
        start_time = datetime.datetime.fromisoformat(
            self.current_reservation["start_time"]
        )
        sleep_time = start_time - current_time
        if sleep_time.total_seconds() > 0:
            time.sleep(sleep_time.total_seconds())

    async def wait_and_poll_for_reservation(self, client):
        await self.poll_for_reservation.wait()
        self.is_polling = True
        self.poll_for_reservation.clear()
        reservation = await make_async_web_request(
            client,
            self.api_key,
            self.api_prefix,
            "post",
            {
                "capacity": self.capacity_needed,
                "model_name": self.model_name,
                "max_tokens": self.max_tokens,
            },
        )
        logger.debug("Made reservation " + str(reservation))
        self.current_reservation = reservation
        self.capacity_remaining = reservation["capacity_remaining"]
        async with self.condition:
            self.condition.notify_all()
        self.is_polling = False
        if self.is_working:
            self.polling_task = asyncio.create_task(
                self.kickoff_reservation_polling(client)
            )

    async def kickoff_reservation_polling(self, client):
        if self.current_reservation is None:
            return None
        try:
            await self.wait_and_poll_for_reservation(client)
        except:
            self.current_reservation = None
            if self.polling_task is not None:
                self.polling_task.cancel()
            return None

    async def async_pause_for_reservation_start(self):
        if self.current_reservation is None:
            return
        current_time = datetime.datetime.utcnow()
        start_time = datetime.datetime.fromisoformat(
            self.current_reservation["start_time"]
        )
        sleep_time = start_time - current_time
        if sleep_time.total_seconds() > 0:
            await asyncio.sleep(sleep_time.total_seconds())

    def update_capacity_use(self, queries: int):
        if self.current_reservation is None:
            return
        self.capacity_remaining -= queries

    def update_capacity_needed(self, queries: int):
        if self.current_reservation is None:
            return
        self.capacity_needed -= queries

    def __del__(self):
        if self.polling_task is not None:
            self.polling_task.cancel()
