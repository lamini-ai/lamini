import json
import logging
import traceback

from lamini.api.rest_requests import make_async_web_request
from lamini.api.utils.reservations import get_reservation_api

logger = logging.getLogger(__name__)


async def process_generation_batch(args: dict):
    client = args["client"]
    key = args["key"]
    batch = args["batch"]
    url = get_url_from_args(args)
    # this will block until there is space in capacity
    reservation_api = get_reservation_api()
    await reservation_api.async_pause_for_reservation_start()

    def can_submit_query():
        if reservation_api.current_reservation is None:
            return True
        if reservation_api.capacity_remaining < len(batch["prompt"]):
            return False
        # Now we can consume credits and send batch
        reservation_api.update_capacity_use(len(batch["prompt"]))
        logger.debug(
            f"yes reservation_api.capacity_remaining {reservation_api.capacity_remaining}"
        )
        return True

    if not can_submit_query():
        async with reservation_api.condition:
            await reservation_api.condition.wait_for(can_submit_query)

    # Separate thread updates existing reservations
    try:
        reservation_id = None
        if reservation_api.current_reservation is not None:
            reservation_id = reservation_api.current_reservation["reservation_id"]
        json = get_body_from_args(batch, reservation_id)
        result = await query_api(client, key, url, json, batch)
    except Exception as e:
        logger.debug(
            f"Error in process_generation_batch, type: {type(e)}, message: {e}"
        )
        for prompt_obj in batch["prompt"]:
            if prompt_obj.error is None:
                prompt_obj.error = []
            prompt_obj.error.append(e)
        raise e
    finally:
        if (
            reservation_api.capacity_needed > 0
            and reservation_api.capacity_remaining < len(batch["prompt"])
            and not reservation_api.is_polling
        ):
            logger.debug(
                f"capacity remaining after query: {reservation_api.capacity_remaining}"
            )
            reservation_api.poll_for_reservation.set()
    for i, prompt_obj in enumerate(batch["prompt"]):
        prompt_obj.response = result[i]


async def query_api(client, key, url, json, batch):
    if batch["type"] == "embedding":
        # TODO: Replace make_async_web_request() with Completion.generate()
        result = await make_async_web_request(client, key, url, "post", json)
        result = result["embedding"]
    else:
        result = await make_async_web_request(client, key, url, "post", json)
    return result


def get_url_from_args(args: dict) -> str:
    api_prefix = args["api_prefix"]
    batch = args["batch"]
    if batch["type"] == "embedding":
        url = api_prefix + "inference/embedding"
    else:
        url = api_prefix + "completions"
    return url


def get_body_from_args(batch: dict, reservation_id: str) -> dict:
    if batch["type"] == "embedding":
        json = {
            "model_name": batch["model_name"],
            "prompt": [p.get_prompt() for p in batch["prompt"]],
        }
    else:
        json = {
            "reservation_id": reservation_id,
            "model_name": batch["model_name"],
            "prompt": [p.get_prompt() for p in batch["prompt"]],
            "output_type": batch["output_type"],
            "max_tokens": batch["max_tokens"],
            "max_new_tokens": batch.get("max_new_tokens", None),
            "model_config": batch.get("model_config", None),
        }
    return json
