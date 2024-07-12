import asyncio
import pytest


from lamini.generation.generation_queue_3_10 import (
    AppendableAsyncGenerator,
    next_n_w_step_func,
    limit_concurrency,
    map_unordered,
    arange,
    async_chunks,
    chunks,
    return_args_and_exceptions,
    _return_args_and_exceptions,
)


async def original_generator():
    for i in range(5):
        yield i


async def foo(arg):
    return arg


@pytest.mark.asyncio
async def test_limit_concurrency_basic():
    async_gen = AppendableAsyncGenerator(original_generator())
    aws = (foo(x) async for x in async_gen)

    results = set()
    async for result in limit_concurrency(aws, limit=2):
        r = await result
        results.add(r)

    assert results == set([0, 1, 2, 3, 4])


async def delayed_foo(arg):
    await asyncio.sleep(1)
    return arg


@pytest.mark.asyncio
async def test_limit_concurrency_with_append():
    async_gen = AppendableAsyncGenerator(original_generator())
    results = set()

    aws = (foo(x) async for x in async_gen)
    async for result in limit_concurrency(aws, limit=1):
        r = await result
        if r == 2:
            async_gen.append(6)
        results.add(r)
    assert results == set([0, 1, 2, 3, 4, 6])


@pytest.mark.asyncio
async def test_limit_concurrency_with_append_more_worker_than_items():
    async_gen = AppendableAsyncGenerator(original_generator())
    results = set()

    aws = (foo(x) async for x in async_gen)
    # Limit is 6, larger than 5, so the appended item will be ignored.
    async for result in limit_concurrency(aws, limit=6):
        r = await result
        if r == 2:
            async_gen.append(6)
        results.add(r)
    assert results == set([0, 1, 2, 3, 4])


concurrency_count = 0
max_concurrency = 0


async def delayed_counter(arg):
    global concurrency_count, max_concurrency
    concurrency_count += 1
    max_concurrency = max(max_concurrency, concurrency_count)
    await asyncio.sleep(1)
    concurrency_count -= 1
    return arg


@pytest.mark.asyncio
async def test_limit_concurrency_respects_limit():
    async_gen = AppendableAsyncGenerator(original_generator())

    aws = (delayed_counter(x) async for x in async_gen)
    # Since there is only 5 items, the maximal number of workers is 5, not 6
    async for result in limit_concurrency(aws, limit=6):
        pass

    assert max_concurrency == 5


@pytest.mark.asyncio
async def test_arange():
    result = []
    async for i in arange(3):
        result.append(i)
    assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_chunks():
    def size_fn():
        return 2

    it = iter([1, 2, 3, 4, 5])
    chunks_result = list(chunks(it, size_fn))
    assert chunks_result == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_async_chunks():
    async def async_gen():
        for i in range(5):
            yield i

    def size_fn():
        return 2

    async_iterator = async_gen()
    result = []
    async for chunk in async_chunks(async_iterator, size_fn):
        result.append(chunk)
    assert result == [[0, 1], [2, 3], [4]]


@pytest.mark.asyncio
async def test_map_unordered():
    async def async_double(x):
        await asyncio.sleep(0.1)
        return x * 2

    async def gen():
        for i in range(5):
            yield i

    result = []
    async for item in map_unordered(async_double, gen(), limit=2):
        result.append(item)
    assert sorted(result) == [0, 2, 4, 6, 8]


@pytest.mark.asyncio
async def test_return_args_and_exceptions():
    async def async_double(x):
        await asyncio.sleep(0.1)
        return x * 2

    wrapped_func = return_args_and_exceptions(async_double)

    result = await _return_args_and_exceptions(wrapped_func, 3)
    assert result == (3, (3, 6))


@pytest.mark.asyncio
async def test_next_n_w_step_func():
    async def async_gen():
        for i in range(5):
            yield i

    def step_func():
        return 2

    result = []
    async for chunk in next_n_w_step_func(async_gen(), step_func):
        result.append(chunk)
    assert result == [[0, 1], [2, 3], [4]]
