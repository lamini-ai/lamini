from llama.prompts.prompt import BasePrompt
from llama import Type, Context
from typing import Optional


class MistralInput(Type):
    system: str = Context(" ")
    user: str = Context(" ")


class MistralOutput(Type):
    output: str = Context(" ")


class MistralPrompt(BasePrompt):
    prompt_template = "[INST] {input:system} {input:user} [/INST]"
