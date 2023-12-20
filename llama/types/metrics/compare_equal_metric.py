from llama.metrics.metric import Metric
from llama.types.type import Type
from llama.types.context import Context


class Match(Type):
    a: str = Context("the first string")
    b: str = Context("the second string")


class MatchResult(Type):
    similarity: float = Context(
        "a number between 0.0 and 1.0 describing similarity of the input strings.  0.0 is no similarity, and 1.0 is a perfect match."
    )


class CompareEqualMetric(Metric):
    def __init__(self, a, b):
        self.input = Match(a=a, b=b)

        print("match", self.input)

    def get_metric_type(self):
        return MatchResult
