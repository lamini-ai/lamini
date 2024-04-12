class MMLUEvaluator:

    def get_prompt(self, question: str) -> str:
        prompt = "<s>[INST] You'll be presented with a task or question.\n"
        prompt += "Provide brief thoughts in 1-2 sentences, no longer than 100 words each. Then, respond with a single letter or number representing the multiple-choice option.\n"
        prompt += "Output your answer as a JSON object in the format {\"explanation\" : str, \"answer\" : str}\n"
        prompt += "Use single quotes within your explanation. End your explanation with a double quote.\n"
        prompt += f"========== question =========\n{question}\n\n"
        prompt += "=" * 20 + "\n\n"
        prompt += " [/INST]"
        # prompt = f"<s>[INST] <<SYS>>\n\n<</SYS>>\n\n{question} [/INST]"
        return prompt
