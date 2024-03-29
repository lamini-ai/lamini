import logging
import math
from typing import List

logger = logging.getLogger(__name__)


class TokenOptimizer:

    def __init__(self, model_name):
        self.model_name = model_name

    def calculate_heuristic_max_tokens_from_prompt(
        self, prompt: List[str], max_new_tokens: int
    ):
        assert isinstance(prompt, list) and len(prompt) > 0
        longest_prompt = max(prompt, key=len)
        token_count_estimate = math.ceil(len(longest_prompt) / 4)
        max_tokens = token_count_estimate + max_new_tokens
        return max_tokens
