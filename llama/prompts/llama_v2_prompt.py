from llama.prompts.prompt import BasePrompt
from llama import Type, Context
from typing import Optional


class LlamaV2Input(Type):
    system: str = Context(" ")
    user: str = Context(" ")
    cue: Optional[str] = Context(" ")


class LlamaV2Output(Type):
    output: str = Context(" ")


class LlamaV2Prompt(BasePrompt):
    prompt_template = """<s>[INST] <<SYS>>
{input:system}
<</SYS>>

{input:user} [/INST]{input:cue}"""
