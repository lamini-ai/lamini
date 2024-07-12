import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lamini.api.lamini_config import get_config
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.token_optimizer import TokenOptimizer
from lamini.api.utils.iterators import async_iter
from lamini.generation.generation_node import GenerationNode


class Node(GenerationNode):
    def __init__(self):
        super(Node, self).__init__(
            model_name="test_model",
            max_tokens=100,
            max_new_tokens=50,
            api_key="test_api_key",
            api_url="test_api_url",
            config={},
        )


@pytest.fixture
def gen_node():
    return Node()


@pytest.mark.asyncio
async def test_transform_prompt_sync(gen_node):
    p = PromptObject(prompt="Hello")
    transformed_prompts = [
        prompt async for prompt in gen_node.transform_prompt(iter([p]))
    ]
    assert transformed_prompts == [p]


@pytest.fixture
async def async_prompt_gen_preprocess():
    yield PromptObject(prompt="Hello", response="Hi")


@pytest.mark.asyncio
async def test_transform_prompt_async(gen_node, async_prompt_gen_preprocess):
    def greet_world(p: PromptObject):
        p.prompt = "Hello World!"

    gen_node.preprocess = greet_world

    transformed_prompts = [
        prompt
        async for prompt in gen_node.transform_prompt(async_prompt_gen_preprocess)
    ]
    assert len(transformed_prompts) == 1
    assert transformed_prompts[0].prompt == "Hello World!"


@pytest.fixture
async def async_prompt_gen():
    yield PromptObject(prompt="Hello", response="Hi")


@pytest.mark.asyncio
async def test_process_results_postprocess_modify_prompt_in_place(
    gen_node, async_prompt_gen
):
    def greet_world(p: PromptObject):
        p.response = "Hey!"

    gen_node.postprocess = greet_world

    results = [result async for result in gen_node.process_results(async_prompt_gen)]
    assert len(results) == 1
    assert results[0].prompt == "Hello"
    assert results[0].response == "Hey!"


@pytest.mark.asyncio
async def test_process_results_postprocess_return(gen_node, async_prompt_gen):
    def greet_world(p: PromptObject):
        return PromptObject(prompt="test", response="Hey!")

    gen_node.postprocess = greet_world

    results = [result async for result in gen_node.process_results(async_prompt_gen)]
    assert len(results) == 1
    assert results[0].prompt == "test"
    assert results[0].response == "Hey!"


@pytest.fixture
async def async_prompt_gen_errored():
    # This has no response, will be considered as errorred.
    yield PromptObject(prompt="Hello")


@pytest.mark.asyncio
async def test_process_results_errored_prompt(gen_node, async_prompt_gen_errored):
    results = [
        result async for result in gen_node.process_results(async_prompt_gen_errored)
    ]
    assert len(results) == 0


if __name__ == "__main__":
    pytest.main()
