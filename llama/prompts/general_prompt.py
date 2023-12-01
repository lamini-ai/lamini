import logging
from llama.prompts.prompt import BasePrompt

logger = logging.getLogger(__name__)


class GeneralPrompt(BasePrompt):
    prompt_template = """Given:
question: {input:question}
Generate:
answer, after "answer:"
answer: """
