import datetime
import os
import unittest
import uuid
from unittest.mock import patch

import lamini
from lamini import MistralRunner
from lamini.api.lamini import Lamini
from lamini.api.utils.async_inference_queue import AsyncInferenceQueue
import tempfile

RESERVATION_RETURN_VALUE = {
    "reservation_id": 112,
    "capacity": 5,
    "start_time": datetime.datetime.now().isoformat(),
    "end_time": (datetime.datetime.now() + datetime.timedelta(seconds=60)).isoformat(),
    "capacity_remaining": 5,
    "dynamic_max_batch_size": 1,
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
        "dynamic_max_batch_size": 1,
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
        "dynamic_max_batch_size": 1,
    }


class TestLocalCacheFile(unittest.TestCase):
    def setUp(self):
        lamini.batch_size = 1
        self.local_cache_file = f"local_cache_{uuid.uuid4()}.txt"
        self.iq = AsyncInferenceQueue(api_key="test_k", api_url="test_u", config={})

    def tearDown(self):
        if os.path.exists(self.local_cache_file):
            os.remove(self.local_cache_file)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_runner_no_local_cache(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        # if local_cache_file flag not used in runner,
        # then there should be no local cache file at the end
        runner = MistralRunner(api_key="test")
        mock_make_completions_call.return_value = [
            {"output": "letter 1"},
            {"output": "letter 2"},
        ]
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect

        prompt = ["What is A?", "what is B"]
        answer = runner.call(prompt)
        self.assertFalse(os.path.exists(self.local_cache_file))
        self.assertEqual(mock_make_completions_call.call_count, 2)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_runner(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        def custom_side_effect(client, key, url, post, batch):
            batch_info = str(batch)

            if "What is A?" in batch_info:
                return [{"output": "letter 1"}]
            elif "What is B?" in batch_info:
                return [{"output": "letter 2"}]
            elif "What is C?" in batch_info:
                return [{"output": "letter 3"}]
            elif "What is D?" in batch_info:
                return [{"output": "letter 4"}]

        mock_make_completions_call.side_effect = custom_side_effect
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        self.assertFalse(os.path.exists(self.local_cache_file))

        # First runner.call(prompt) writes results to local cache file

        runner = MistralRunner(api_key="test", local_cache_file=self.local_cache_file)
        prompt = ["What is A?", "What is B?", "What is C?"]
        answer = runner.call(prompt)
        self.assertTrue(os.path.exists(self.local_cache_file))
        expected_answer = [
            "letter 1",
            "letter 2",
            "letter 3",
        ]
        self.assertEqual(answer, expected_answer)
        self.assertEqual(mock_make_completions_call.call_count, 3)

        content = self.iq.read_local_cache(self.local_cache_file)
        expected_content = {
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is A? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 1"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is B? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 2"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is C? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 3"}
            ],
        }
        self.assertEqual(content, expected_content)

        # Second runner.call(prompt) reads result from local cache file and does not invoke process_batch

        mock_make_completions_call.reset_mock()
        answer = runner.call(prompt)
        expected_answer = [
            "letter 1",
            "letter 2",
            "letter 3",
        ]
        self.assertEqual(answer, expected_answer)
        mock_make_completions_call.assert_not_called()
        content = self.iq.read_local_cache(self.local_cache_file)
        self.assertEqual(content, expected_content)

        # Third runner.call(prompt) has an additional prompt

        mock_make_completions_call.reset_mock()
        prompt.append("What is D?")
        answer = runner.call(prompt)
        expected_answer = [
            "letter 1",
            "letter 2",
            "letter 3",
            "letter 4",
        ]
        self.assertEqual(answer, expected_answer)
        self.assertEqual(mock_make_completions_call.call_count, 1)
        content = self.iq.read_local_cache(self.local_cache_file)
        expected_content = {
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is A? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 1"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is B? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 2"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is C? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 3"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is D? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 4"}
            ],
        }
        self.assertEqual(content, expected_content)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_runner_exception(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        def custom_side_effect(client, key, url, post, batch):
            batch_info = str(batch)

            if "What is A?" in batch_info:
                return [{"output": "letter 1"}]
            elif "What is B?" in batch_info:
                raise Exception("bad stuff")
            elif "What is C?" in batch_info:
                return [{"output": "letter 3"}]

        # First runner.call:
        # one job raises an exception, results for the other jobs still saved in local cache

        mock_make_completions_call.side_effect = custom_side_effect
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        self.assertFalse(os.path.exists(self.local_cache_file))

        runner = MistralRunner(
            api_key="test",
            local_cache_file=self.local_cache_file,
        )
        prompt = [
            "What is A?",
            "What is B?",
            "What is C?",
        ]
        lamini.batch_size = 1

        with self.assertRaises(Exception):
            runner.call(prompt)
        self.assertEqual(mock_make_completions_call.call_count, 3)
        self.assertTrue(os.path.exists(self.local_cache_file))

        content = self.iq.read_local_cache(self.local_cache_file)
        expected_content = {
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is A? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 1"}
            ],
            "{'model_name': 'mistralai/Mistral-7B-Instruct-v0.2', 'prompt': ['<s>[INST] Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. What is C? [/INST]'], 'output_type': None, 'max_tokens': None}": [
                {"output": "letter 3"}
            ],
        }
        self.assertEqual(content, expected_content)

        # Second runner.call:
        # one job still raises the exception
        # results for the other jobs loaded from local cache
        # at the end, local cache does not change

        mock_make_completions_call.reset_mock()
        with self.assertRaises(Exception):
            runner.call(prompt)
        self.assertEqual(mock_make_completions_call.call_count, 1)
        self.assertTrue(os.path.exists(self.local_cache_file))
        content = self.iq.read_local_cache(self.local_cache_file)
        self.assertEqual(content, expected_content)

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_api_generate(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        def custom_side_effect(client, key, url, post, batch):
            batch_info = str(batch)

            if "What is A?" in batch_info:
                return [{"output": "letter 1"}]
            elif "What is B?" in batch_info:
                return [{"output": "letter 2"}]

        mock_make_completions_call.side_effect = custom_side_effect
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        self.assertFalse(os.path.exists(self.local_cache_file))

        api = Lamini(
            model_name="hf-internal-testing/tiny-random-gpt2",
            local_cache_file=self.local_cache_file,
        )
        prompt = ["What is A?", "What is B?"]
        answer = api.generate(prompt=prompt)
        expected_answer = ["letter 1", "letter 2"]
        self.assertEqual(answer, expected_answer)
        self.assertEqual(mock_make_completions_call.call_count, 2)

        # the second api.generate uses local cache
        mock_make_completions_call.reset_mock()
        answer = api.generate(prompt=prompt)
        expected_answer = ["letter 1", "letter 2"]
        self.assertEqual(answer, expected_answer)
        mock_make_completions_call.assert_not_called()

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.api.utils.process_batch.make_async_web_request")
    def test_api_generate_with_output_type(
        self,
        mock_make_completions_call,
        mock_current_reservation,
        mock_init_reservation,
    ):
        def custom_side_effect(client, key, url, post, batch):
            batch_info = str(batch)

            if "What is A?" in batch_info:
                return [{"ans1": "letter 1", "ans2": "letter 3"}]
            elif "What is B?" in batch_info:
                return [{"ans1": "letter 2", "ans2": "letter 4"}]

        mock_make_completions_call.side_effect = custom_side_effect
        mock_current_reservation.side_effect = async_mock_side_effect
        mock_init_reservation.side_effect = mock_side_effect
        self.assertFalse(os.path.exists(self.local_cache_file))

        api = Lamini(
            model_name="hf-internal-testing/tiny-random-gpt2",
            local_cache_file=self.local_cache_file,
        )
        prompt = ["What is A?", "What is B?"]
        answer = api.generate(prompt=prompt, output_type={"ans1": "str", "ans2": "str"})
        expected_answer = [
            {"ans1": "letter 1", "ans2": "letter 3"},
            {"ans1": "letter 2", "ans2": "letter 4"},
        ]
        self.assertEqual(answer, expected_answer)
        self.assertEqual(mock_make_completions_call.call_count, 2)

        # the second api.generate uses local cache
        mock_make_completions_call.reset_mock()
        answer = api.generate(prompt=prompt, output_type={"ans1": "str", "ans2": "str"})
        expected_answer = [
            {"ans1": "letter 1", "ans2": "letter 3"},
            {"ans1": "letter 2", "ans2": "letter 4"},
        ]
        self.assertEqual(answer, expected_answer)
        mock_make_completions_call.assert_not_called()
