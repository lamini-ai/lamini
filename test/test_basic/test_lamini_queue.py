import datetime
import random
import unittest
from unittest.mock import patch

import lamini


def async_mock_completions_func(key, prefix, post, args):
    print("args: ", args)
    if isinstance(args["prompt"], list):
        return [{"output": "response"}]
    return {"output": "response"}


async def async_mock_process_batch_func(args):
    return [{"output": "response"} for _ in range(len(args["batch"]["prompt"]))]


class TestLaminiQueue(unittest.TestCase):

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.async_inference_queue_3_10.process_batch")
    def test_queue(
        self,
        async_mock_proces_batch,
        mock_reservation_api_poll,
        mock_reservation_api_init,
    ):
        async_mock_proces_batch.side_effect = async_mock_process_batch_func
        mock_reservation_api_poll.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
            "dynamic_max_batch_size": 5,
        }
        mock_reservation_api_init.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
            "dynamic_max_batch_size": 5,
        }

        con = {"production.key": ""}
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        prompts = list(
            f"<s>[INST] There will be a nuclear war in {random.random() * 10} days. Generate a question a small child in Palo Alto facing this problem will ask.[/INST]"
            for _ in range(10)
        )
        results = engine.generate(prompts)
        print("results: ", results)
        assert len(results) == len(prompts)

    @patch("lamini.api.utils.completion.make_web_request")
    @patch("lamini.api.utils.async_inference_queue_3_10.process_batch")
    def test_non_queue(
        self,
        async_mock_proces_batch,
        async_mock_completions,
    ):
        async_mock_proces_batch.side_effect = async_mock_process_batch_func
        async_mock_completions.side_effect = async_mock_completions_func
        con = {"production.key": ""}
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        prompt = "string"
        response = engine.generate(prompt)
        print("response1: ", response)
        assert response == "response"
        self.assertEqual(async_mock_proces_batch.call_count, 0)

        prompt = ["string"]
        response = engine.generate(prompt)
        print("response2: ", response)
        assert response == ["response"]

        self.assertEqual(async_mock_proces_batch.call_count, 0)
