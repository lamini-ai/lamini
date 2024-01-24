import asyncio
from threading import Thread, current_thread


def sync(awaitable):
    """
    Get result of calling function on the given args. If it is awaitable, will
    block until it is finished. Runs in a new thread in such cases.
    Credit Piotr: https://github.com/truera/trulens/pull/793/files#diff-23a219ce07a4edb8892fe8ecf21aba06d5ebe012c80c3386f9a9e1fe80d23254
    """

    # Check if there is a running loop.
    try:
        asyncio.get_running_loop()
        in_loop = True
    except Exception:
        in_loop = False

    if not in_loop:
        # If not, we can create one here and run it until completion.
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(awaitable)

    # Otherwise we cannot create a new one in this thread so we create a
    # new thread to run the awaitable until completion.

    def run_in_new_loop():
        th = current_thread()
        # Attach return value and possibly exception to thread object so we
        # can retrieve from the starter of the thread.
        th.ret = None
        th.error = None
        try:
            loop = asyncio.new_event_loop()
            th.ret = loop.run_until_complete(awaitable)
        except Exception as e:
            th.error = e

    thread = Thread(target=run_in_new_loop)

    # Start thread and wait for it to finish.
    thread.start()
    thread.join()

    # Get the return or error, return the return or raise the error.
    if thread.error is not None:
        raise thread.error
    else:
        return thread.ret
