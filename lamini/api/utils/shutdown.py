import asyncio
import logging


async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]
    logging.info(f"Exiting program...")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
