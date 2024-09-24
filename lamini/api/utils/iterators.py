from typing import Iterator, AsyncGenerator, Any


async def async_iter(normal_iter: Iterator) -> AsyncGenerator[Any, None]:
    """Adapt an normal iterator to an async iterator

    Parameters
    ----------
    normal_iter: Iterator
        Iterator to wrap with a yield generator

    Yields
    -------
    item: Any
        Items within the provided normal iterator
    """

    for item in normal_iter:
        yield item
