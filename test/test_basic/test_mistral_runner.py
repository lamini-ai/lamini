import unittest
from unittest.mock import patch

from lamini import MistralRunner
from lamini.api.lamini import Lamini


class TestMistralRunner(unittest.TestCase):
    @patch("lamini.api.lamini.Lamini.generate")
    def test_mistral_runner(self, mock_generate):
        mock_generate.return_value = "My favorite food is pizza"
        model = MistralRunner(
            api_key="test", model_name="mistralai/Mistral-7B-Instruct-v0.2"
        )
        answer = model.call("What is your favorite food?")
        self.assertEqual(answer, "My favorite food is pizza")
