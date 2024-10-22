import asyncio
import logging
import sys
from typing import AsyncIterator, Iterator, Optional

import lamini
from lamini.generation.base_node_object import BaseGenerationNode

logger = logging.getLogger(__name__)


class GenerationPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_url = api_url

    def forward(self, prompt: AsyncIterator) -> AsyncIterator:
        """NOTE: You must implement this function.

        Typical use:
        class MyOwnPipeline(GenerationPipeline):
            def forward(...):
                ...
        pipeline_results = MyOwnPipeline().call(...)
        In your subclass, use GenerationNode, EmbeddingNode and their subclasses to
         define pipeline nodes. The input and output of pipeline nodes should all
         be async iterators.
        """

    async def __call(
        self,
        prompt: AsyncIterator,
    ) -> AsyncIterator:
        if sys.version_info >= (3, 10):
            logger.info("Using 3.10 InferenceQueue Interface")
            from lamini.generation.generation_queue_3_10 import (
                get_global_inference_queue as get_global_inference_queue_3_10,
            )

            self.async_inference_queue = get_global_inference_queue_3_10(
                self.api_key,
                self.api_url,
            )
            self.reservation_api = self.async_inference_queue.reservation_api
        else:
            raise Exception("Must use Python 3.10 or greater for this feature")
        model_names = []
        max_tokens = []
        for _, val in vars(self).items():
            if isinstance(val, BaseGenerationNode):
                val.async_inference_queue = self.async_inference_queue
                try:
                    model_names.append(val.model_name)
                    max_tokens.append(val.max_tokens)
                except:
                    continue
        assert len(model_names) > 0
        assert isinstance(prompt, Iterator) or isinstance(prompt, AsyncIterator)
        iterator = self.forward(prompt)
        assert isinstance(iterator, AsyncIterator)

        if not any(max_tokens):
            max_tokens = None
        else:
            max_tokens = max(max_tokens)
        self.reservation_api.initialize_reservation(
            capacity=lamini.batch_size * lamini.max_workers,
            model_name=model_names[0],
            batch_size=lamini.batch_size,
            max_tokens=max_tokens,
        )
        self.reservation_api.pause_for_reservation_start()

        self.reservation_polling_task = asyncio.create_task(
            self.reservation_api.kickoff_reservation_polling(
                self.async_inference_queue.client
            )
        )
        return iterator

    async def __cleanup(self):
        self.reservation_api.is_working = False
        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
        if self.reservation_api.polling_task is not None:
            self.reservation_api.polling_task.cancel()
        await self.async_inference_queue.client.close()

    async def call_with_result(
        self,
        prompt: AsyncIterator,
    ):
        iterator = await self.__call(prompt)
        finished = False
        results = []
        while not finished:
            try:
                r = None
                while r is None:
                    r = await anext(iterator)
            except StopAsyncIteration:
                finished = True
            else:
                results.append(r)

        await self.__cleanup()
        return results

    async def call(
        self,
        prompt: AsyncIterator,
    ):
        iterator = await self.__call(prompt)
        finished = False
        while not finished:
            try:
                r = None
                while r is None:
                    r = await anext(iterator)
            except StopAsyncIteration:
                finished = True
            else:
                yield r

        await self.__cleanup()
