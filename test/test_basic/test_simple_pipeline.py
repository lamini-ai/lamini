import datetime
import random
import unittest
from unittest.mock import patch

import lamini
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.split_response_node import SplitResponseNode


class ExamplePipeline(GenerationPipeline):
    def __init__(self, api_key):
        super(ExamplePipeline, self).__init__(api_key=api_key)

        self.question_generation = GenerationNode(
            model_name="hf-internal-testing/tiny-random-gpt2", max_tokens=150
        )
        self.answer_generation = GenerationNode(
            model_name="hf-internal-testing/tiny-random-gpt2",
            max_tokens=400,
        )

    def forward(self, x):
        x = self.question_generation(x)
        x = self.answer_generation(x)
        return x


async def async_mock_side_effect(args):
    batch = args["batch"]
    for i, prompt_obj in enumerate(batch["prompt"]):
        if prompt_obj.response is None:
            prompt_obj.response = "hello"
        else:
            prompt_obj.response += " hello"


class TestSimplePipeline(unittest.IsolatedAsyncioTestCase):

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.generation.generation_queue_3_10.process_generation_batch")
    async def test_pipeline(
        self,
        mock_process_generation_batch,
        mock_reservation_api_poll,
        mock_reservation_api_init,
    ):
        mock_process_generation_batch.side_effect = async_mock_side_effect
        mock_reservation_api_poll.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }
        mock_reservation_api_init.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }

        pipeline = ExamplePipeline("test")

        prompts = list(
            PromptObject(
                f"<s>[INST] There will be a nuclear war in {random.random() * 10} days. Generate a question a small child in Palo Alto facing this problem will ask.[/INST]",
                data="test",
            )
            for _ in range(10)
        )
        prompts_iterator = iter(prompts)
        results = pipeline.call(prompts_iterator)
        print(results)
        async for p in results:
            assert isinstance(p, PromptObject)
            print(p)
            assert p.response == "hello hello"

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.generation.generation_queue_3_10.process_generation_batch")
    async def test_pipeline_call_with_result(
        self,
        mock_process_generation_batch,
        mock_reservation_api_poll,
        mock_reservation_api_init,
    ):
        mock_process_generation_batch.side_effect = async_mock_side_effect
        mock_reservation_api_poll.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }
        mock_reservation_api_init.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }

        pipeline = ExamplePipeline("test")

        prompts = list(
            PromptObject(
                f"<s>[INST] There will be a nuclear war in {random.random() * 10} days. Generate a question a small child in Palo Alto facing this problem will ask.[/INST]",
                data="test",
            )
            for _ in range(10)
        )
        prompts_iterator = iter(prompts)
        results = await pipeline.call_with_result(prompts_iterator)
        print(results)
        assert len(results) == 10
