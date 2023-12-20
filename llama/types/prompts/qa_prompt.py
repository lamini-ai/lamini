import logging
from llama.prompts.prompt import BasePrompt

logger = logging.getLogger(__name__)


class QAPrompt(BasePrompt):
    prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction: {input:question}

### Response:"""
