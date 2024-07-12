import os
import unittest
from unittest.mock import patch

from lamini import BasicModelRunner


class TestBasicModelRunner(unittest.TestCase):
    @patch("lamini.api.lamini.Lamini.generate")
    def test_basic_model_runner(self, mock_generate):
        mock_generate.return_value = "My favorite food is pizza"
        model = BasicModelRunner(
            api_key="test", model_name="hf-internal-testing/tiny-random-gpt2"
        )
        answer = model("What is your favorite food?")
        self.assertEqual(answer, "My favorite food is pizza")

    def test_final_prompt(self):
        model = BasicModelRunner(
            api_key="test", model_name="hf-internal-testing/tiny-random-gpt2"
        )
        result = model.create_final_prompts("What is your favorite food?", None)
        self.assertEqual(result, "What is your favorite food?")

    def test_load_data(self):
        model = BasicModelRunner(
            api_key="test", model_name="hf-internal-testing/tiny-random-gpt2"
        )
        dir_path = os.path.dirname(os.path.realpath(__file__))

        model.load_data_from_jsonlines(
            dir_path + "/test_data.jsonlines", input_key="question", output_key="answer"
        )
        print(len(model.data))
