import datetime
import os
import unittest
import uuid
from unittest.mock import patch

import lamini
from lamini import MistralRunner
from lamini.api.lamini import Lamini
from lamini.api.utils.async_inference_queue import AsyncInferenceQueue

RESERVATION_RETURN_VALUE = {
    "reservation_id": 112,
    "capacity": 5,
    "start_time": datetime.datetime.now().isoformat(),
    "end_time": (datetime.datetime.now() + datetime.timedelta(seconds=60)).isoformat(),
    "capacity_remaining": 5,
    "dynamic_max_batch_size": 5,
}


def mock_side_effect(key, api_prefix, post, capacity):
    return {
        "reservation_id": 112,
        "capacity": capacity["capacity"],
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": (
            datetime.datetime.now() + datetime.timedelta(seconds=60)
        ).isoformat(),
        "capacity_remaining": capacity["capacity"],
        "dynamic_max_batch_size": 5,
    }


async def async_mock_side_effect(client, key, api_prefix, post, capacity):
    return {
        "reservation_id": 112,
        "capacity": capacity["capacity"],
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": (
            datetime.datetime.now() + datetime.timedelta(seconds=60)
        ).isoformat(),
        "capacity_remaining": capacity["capacity"],
        "dynamic_max_batch_size": 5,
    }


class TestQueueCallback(unittest.TestCase):
    def setUp(self):
        lamini.batch_size = 3
        self.local_cache_file = f"local_cache_{uuid.uuid4()}.txt"
        self.iq = AsyncInferenceQueue(api_key="test_k", api_url="test_u", config={})

    def tearDown(self):
        if os.path.exists(self.local_cache_file):
            os.remove(self.local_cache_file)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_lamini_callback(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        # if local_cache_file flag not used in runner,
        # then there should be no local cache file at the end
        lamini = Lamini("hf-internal-testing/tiny-random-gpt2", api_key="test")
        mock_make_completions_call.return_value = [
            {"output": "letter 1"},
            {"output": "letter 2"},
        ]
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        metadata = [{"meta": "data"}, {"meta", "data"}]

        def callback(args):
            assert args["metadata"][0]["meta"] == "data"
            assert args["prompt"] == ["What is A?", "What is B?"]
            # TODO: When running on github, reservation_id is not in args
            # But running locally it exists. Fix this inconsistency.
            # assert args["reservation_id"] == 112

        prompt = ["What is A?", "What is B?"]
        answer = lamini.generate(prompt, callback=callback, metadata=metadata)
        self.assertEqual(mock_init_reservation.call_count, 1)
        self.assertEqual(mock_current_reservation.call_count, 0)
        self.assertEqual(mock_make_completions_call.call_count, 1)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_runner_callback(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        mock_make_completions_call.return_value = [{"output": "letter 1"}]
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        runner = MistralRunner(api_key="test", local_cache_file=self.local_cache_file)
        metadata = [
            {"meta": "data1"},
            {"meta": "data2"},
            {"meta": "data3"},
        ]

        def callback(args):
            assert args == {
                "index": 0,
                # TODO: When test_runner_callback is run by itself,
                # pytest sdk/test/test_basic/test_queue_callback.py -k test_runner_callback
                # We need to add "reservation_id": 112,
                # When running pytest sdk/test/test_basic/test_queue_callback.py
                # We need to remove "reservation_id": 112,
                # So it seems 2 tests have interdepencies, make tests independent.
                "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
                "prompt": [
                    "<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is A? [/INST]",
                    "<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is B? [/INST]",
                    "<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is C? [/INST]",
                ],
                "output_type": None,
                "max_tokens": None,
                "completion": [{"output": "letter 1"}],
                "metadata": [{"meta": "data1"}, {"meta": "data2"}, {"meta": "data3"}],
            }

        prompt = ["What is A?", "What is B?", "What is C?"]
        answer = runner.call(prompt, callback=callback, metadata=metadata)
        self.assertEqual(mock_init_reservation.call_count, 1)
        self.assertEqual(mock_current_reservation.call_count, 0)
        self.assertEqual(mock_make_completions_call.call_count, 1)
