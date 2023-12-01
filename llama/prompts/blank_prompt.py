import logging
from llama.prompts.prompt import BasePrompt
from llama import Type, Context

logger = logging.getLogger(__name__)


class Input(Type):
    input: str = Context("input")


class Output(Type):
    output: str = Context("output")


class BlankPrompt(BasePrompt):
    prompt_template = """{input:input}"""

    input = Input
    output = Output
