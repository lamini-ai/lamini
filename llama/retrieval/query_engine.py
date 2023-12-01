import logging

from llama.runners.llama_v2_runner import LlamaV2Runner

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(
        self,
        index,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        config={},
        k=5,
        system_prompt=None,
    ):
        self.index = index
        self.model = LlamaV2Runner(
            model_name=model_name, config=config, system_prompt=system_prompt
        )
        self.k = k

    def answer_question(self, question):
        prompt = self._build_prompt(question)

        logger.info(
            "------------------------------ Prompt ------------------------------"
        )
        logger.info(prompt)
        logger.info(
            "----------------------------- End Prompt -----------------------------"
        )

        return self.model(prompt)

    def _build_prompt(self, question):
        most_similar = self.index.query(question, k=self.k)

        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question

        return prompt
