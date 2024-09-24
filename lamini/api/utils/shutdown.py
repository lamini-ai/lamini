import asyncio
import logging


async def shutdown(signal, loop) -> None:
    """Cleanup tasks tied to the service's shutdown.

    Parameters
    ----------
    signal: signal object containing signal source

    loop: current asyncio loop

    Returns
    -------
    None
    """

    logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    [task.cancel() for task in tasks]
    logging.info(f"Exiting program...")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
