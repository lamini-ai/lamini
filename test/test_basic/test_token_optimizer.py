import datetime
import random
import unittest
from unittest.mock import patch

import lamini
from lamini.generation.token_optimizer import TokenOptimizer


class TestTokenOptimizer(unittest.TestCase):

    def test_token_optimizer(
        self,
    ):
        token_optimizer = TokenOptimizer("hf-internal-testing/tiny-random-gpt2")
        max_tokens = token_optimizer.calculate_heuristic_max_tokens_from_prompt(
            ["prompt"], 10
        )
        print("max_tokens: ", max_tokens)
        assert max_tokens == 12
