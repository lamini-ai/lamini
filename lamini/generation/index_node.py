import logging
from typing import AsyncIterator, Generator, Optional

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode
from lamini.index.lamini_index import LaminiIndex

logger = logging.getLogger(__name__)


class IndexNode(EmbeddingNode):
    def __init__(
        self,
        index: LaminiIndex,
        index_top_k: int = 1,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
        max_tokens: Optional[int] = None,
    ):
        super(IndexNode, self).__init__(
            model_name=model_name, api_key=api_key, api_url=api_url, config=config
        )
        self.max_tokens = max_tokens
        self.index = index
        self.index_top_k = index_top_k

    def __call__(self, prompt, *args, **kwargs):
        prompt = self.transform_prompt(prompt)
        results = self.generate(prompt, *args, **kwargs)
        results = self.query_index(results)
        results = self.process_results(results)
        return results

    async def query_index(self, results: AsyncIterator[PromptObject]):
        async for a in results:
            if a is None or a.response is None:
                continue
            if hasattr(self, "query_index_impl"):
                mod_a = self.query_index_impl(a)
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

    def query_index_impl(self, prompt: PromptObject):
        index_result = self.index.query_with_embedding(
            prompt.response, k=self.index_top_k
        )
        prompt.response = index_result
        return prompt
