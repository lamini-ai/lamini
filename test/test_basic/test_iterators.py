import pytest

from lamini.api.utils.iterators import async_iter


@pytest.mark.asyncio
async def test_async_iter_basic():
    normal_iter = iter([1, 2, 3, 4])
    async_gen = async_iter(normal_iter)

    result = []
    async for item in async_gen:
        result.append(item)

    assert result == [1, 2, 3, 4]
