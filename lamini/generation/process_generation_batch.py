import json
import logging
import traceback

from lamini.api.rest_requests import make_async_web_request
from lamini.api.utils.reservations import get_reservation_api

logger = logging.getLogger(__name__)


async def process_generation_batch(args):
    client = args["client"]
    key = args["key"]
    api_prefix = args["api_prefix"]
    batch = args["batch"]
    url = api_prefix + "completions"
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
        json = {
            "reservation_id": reservation_id,
            "model_name": batch["model_name"],
            "prompt": [p.get_prompt() for p in batch["prompt"]],
            "out_type": batch["out_type"],
            "max_tokens": batch["max_tokens"],
            "max_new_tokens": batch.get("max_new_tokens", None),
            "model_config": batch.get("model_config", None),
        }
        result = await make_async_web_request(client, key, url, "post", json)
    except Exception as e:
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
