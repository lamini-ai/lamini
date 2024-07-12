from typing import Iterator


async def async_iter(normal_iter: Iterator):
    """Adapt an normal iterator to an async iterator"""
    for item in normal_iter:
        yield item
