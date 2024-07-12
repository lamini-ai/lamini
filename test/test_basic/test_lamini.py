import unittest
from unittest.mock import patch

import jsonlines
import lamini


def mock_upload_dataset_locally(dataset_id, upload_base_path, data, is_public):
    return {"dataset_location": "/tmp/read"}


def mock_get_upload_base_path():
    return {"upload_base_path": "/tmp"}


class TestLamini(unittest.TestCase):
    def setUp(self):
        self.engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )

    def test_make_llm_req_data_single_input(self):
        prompt = "What is the hottest day of the year?"
        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name,
            prompt,
            output_type,
            max_tokens,
            None,
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the hottest day of the year?",
            "output_type": {"Answer": "An answer to the question"},
            "max_tokens": None,
        }
        self.assertEqual(req_data, wanted_data)

    def test_passing_model_config(self):
        con = {"production.key": ""}
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        self.assertIsNone(engine.model_config)

        con = {
            "model_config": {"rope_scaling.type": "dynamic", "rope_scaling.factor": 3.0}
        }
        engine = lamini.Lamini(
            api_key="test_k",
            model_name="hf-internal-testing/tiny-random-gpt2",
            config=con,
        )
        self.assertEqual(engine.model_config["rope_scaling.type"], "dynamic")
        self.assertEqual(engine.model_config["rope_scaling.factor"], 3.0)

    # multiple input values and output types
    def test_make_llm_req_data_multiple_input(self):
        prompt = "What is the hottest day of the year?"
        output_type = {
            "Answer": "An answer to the question",
            "Answer2": "An answer to the question2",
        }
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the hottest day of the year?",
            "output_type": {
                "Answer": "An answer to the question",
                "Answer2": "An answer to the question2",
            },
            "max_tokens": None,
        }
        self.assertEqual(req_data, wanted_data)

    def test_make_llm_req_data_user_specified_output_type(self):
        # user can specify boolean with both "#bool" and "#boolean"

        prompt = "What is the answer?"
        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        wanted_data = {
            "model_name": "hf-internal-testing/tiny-random-gpt2",
            "prompt": "What is the answer?",
            "output_type": {"Answer": "An answer to the question"},
            "max_tokens": None,
        }
        self.assertEqual(req_data, wanted_data)

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify integer with both "#int" and "#integer"

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        wanted_data["output_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify string with both "#str" and "#string"

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        wanted_data["output_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        self.assertEqual(req_data, wanted_data)

        # user can specify number with both "#float" and "#number"

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        wanted_data["output_type"]["Answer"] = "An answer to the question"
        self.assertEqual(req_data, wanted_data)

        output_type = {"Answer": "An answer to the question"}
        model_name = "hf-internal-testing/tiny-random-gpt2"
        max_tokens = None
        req_data = self.engine.completion.make_llm_req_map(
            model_name, prompt, output_type, max_tokens, None
        )
        self.assertEqual(req_data, wanted_data)

    @patch("lamini.api.utils.async_inference_queue_3_10.AsyncInferenceQueue.submit")
    def test_lamini_generate_no_retries_batch(self, mock_submit):
        mock_submit.return_value = [{"output": "my result"}, {"output": "my result2"}]
        api = lamini.Lamini(
            api_key="test", model_name="hf-internal-testing/tiny-random-gpt2"
        )
        prompt = ["hello!", "hello2!"]
        res = api.generate(prompt=prompt)
        expected_res = ["my result", "my result2"]
        self.assertEqual(res, expected_res)
        self.assertEqual(mock_submit.call_count, 1)

    @patch("lamini.api.utils.completion.Completion.generate")
    def test_lamini_generate_no_retries(self, mock_submit):
        mock_submit.return_value = {"output": "my result"}
        api = lamini.Lamini(
            api_key="test", model_name="hf-internal-testing/tiny-random-gpt2"
        )
        prompt = "hello!"
        res = api.generate(prompt=prompt)
        expected_res = "my result"
        self.assertEqual(res, expected_res)
        self.assertEqual(mock_submit.call_count, 1)

    @patch(
        "lamini.api.train.Train.upload_dataset_locally",
        side_effect=mock_upload_dataset_locally,
    )
    @patch(
        "lamini.api.train.Train.get_upload_base_path",
        side_effect=mock_get_upload_base_path,
    )
    def test_upload_data(
        self, mock_create_blob_dataset_location, mock_get_upload_base_path
    ):

        def sample_data():
            for i in range(3):
                yield {"input": f"hello{i}", "output": f"world{i}"}

        self.engine.upload_data(sample_data())
        print(" self.engine.upload_file_path:", self.engine.upload_file_path)

        assert self.engine.upload_file_path == "/tmp/read"
