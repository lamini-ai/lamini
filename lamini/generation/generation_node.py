import logging
import sys
from typing import AsyncIterator, Generator, Iterator, Optional, Union

from lamini.api.lamini_config import get_config
from lamini.generation.base_node_object import BaseGenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class GenerationNode(BaseGenerationNode):
    def __init__(
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
    ):
        self.config = get_config(config)
        self.model_name = model_name
        self.token_optimizer = TokenOptimizer(model_name)
        if sys.version_info >= (3, 10):
            logger.info("Using 3.10 InferenceQueue Interface")
            from lamini.generation.generation_queue_3_10 import (
                get_global_inference_queue as get_global_inference_queue_3_10,
            )

            self.async_inference_queue = get_global_inference_queue_3_10(
                api_key, api_url, config=config
            )
        else:
            raise Exception("Must use Python 3.10 or greater for this feature")

        self.model_config = self.config.get("model_config", None)
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt, *args, **kwargs):
        prompt = self.transform_prompt(prompt)
        results = self.generate(prompt, *args, **kwargs)
        results = self.process_results(results)
        return results

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        assert isinstance(prompt, Iterator) or isinstance(prompt, AsyncIterator)
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
            output_type=output_type,
            max_tokens=max_tokens or self.max_tokens,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
        return self.async_inference_queue.submit(req_data, self.token_optimizer)

    def make_llm_req_map(
        self,
        model_name,
        prompt,
        output_type,
        max_tokens,
        max_new_tokens,
    ):
        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        req_data["type"] = "completion"
        return req_data

    async def transform_prompt(
        self, prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]
    ):
        if isinstance(prompt, Iterator):
            for a in prompt:
                if hasattr(self, "preprocess"):
                    mod_a = self.preprocess(a)
                    if isinstance(mod_a, Generator):
                        for a in mod_a:
                            if a is not None:
                                assert isinstance(a, PromptObject)
                                yield a
                        continue
                    if mod_a is not None:
                        a = mod_a
                assert a is None or isinstance(a, PromptObject)
                yield a
        elif isinstance(prompt, AsyncIterator):
            async for a in prompt:
                if hasattr(self, "preprocess"):
                    mod_a = self.preprocess(a)
                    if isinstance(mod_a, Generator):
                        for a in mod_a:
                            if a is not None:
                                assert isinstance(a, PromptObject)
                                yield a
                        continue
                    if mod_a is not None:
                        a = mod_a
                assert a is None or isinstance(a, PromptObject)
                yield a
        else:
            raise Exception("Invalid prompt type")

    async def process_results(self, results: AsyncIterator[PromptObject]):
        async for a in results:
            if a is None or a.response is None:
                continue
            if hasattr(self, "postprocess"):
                mod_a = self.postprocess(a)
                if isinstance(mod_a, Generator):
                    for a in mod_a:
                        if a is not None:
                            assert isinstance(a, PromptObject)
                            yield a
                    continue
                if mod_a is not None:
                    a = mod_a
            assert a is None or isinstance(a, PromptObject)
            yield a
