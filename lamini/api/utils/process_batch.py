import json
import logging

from lamini.api.rest_requests import make_async_web_request
from lamini.api.utils.reservations import get_reservation_api

logger = logging.getLogger(__name__)


async def process_batch(args):
    client = args["client"]
    key = args["key"]
    api_prefix = args["api_prefix"]
    batch = args["batch"]
    local_cache_file = args["local_cache_file"]
    local_cache = args["local_cache"]
    callback = args["callback"]
    url = api_prefix + "completions"
    batch_k = str(batch)
    if local_cache and batch_k in local_cache:
        return local_cache[batch_k]

    # this will block until there is space in capacity
    reservation_api = get_reservation_api()
    await reservation_api.async_pause_for_reservation_start()

    def can_submit_query():
        if reservation_api.current_reservation is None:
            return True
        if reservation_api.capacity_remaining <= 0:
            return False
        # Now we can consume credits and send batch
        reservation_api.update_capacity_use(len(batch["prompt"]))
        return reservation_api.capacity_remaining >= 0

    if not can_submit_query():
        async with reservation_api.condition:
            await reservation_api.condition.wait_for(can_submit_query)

    # Separate thread updates existing reservations
    if reservation_api.current_reservation is not None:
        batch = {
            "reservation_id": reservation_api.current_reservation["reservation_id"],
            **batch,
        }

    logger.debug(f"Sending batch {args['index']}")
    result = await make_async_web_request(client, key, url, "post", batch)
    logger.debug(f"Received batch response")
    logger.debug(f"reservation_api.capacity_needed {reservation_api.capacity_needed}")
    logger.debug(
        f"reservation_api.capacity_remaining {reservation_api.capacity_remaining}"
    )
    reservation_api.update_capacity_needed(len(batch["prompt"]))
    if (
        reservation_api.capacity_needed > 0
        and reservation_api.capacity_remaining <= 0
        and not reservation_api.is_polling
    ):
        logger.debug(
            f"capacity remaining after query: {reservation_api.capacity_remaining}"
        )
        reservation_api.poll_for_reservation.set()

    if local_cache_file and result:
        append_local_cache(local_cache_file, batch_k, result)
    if callback:
        callback(batch, result)
    return result


def append_local_cache(local_cache_file, batch, res):
    batch_k = json.dumps(str(batch))
    batch_v = json.dumps(res)
    cache_line = f"{batch_k}: {batch_v},\n\n"

    with open(local_cache_file, "a") as file:
        file.write(cache_line)
