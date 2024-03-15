from typing import Any


class PromptObject:
    def __init__(self, prompt: str, response: str = None, data: Any = None) -> None:
        assert isinstance(prompt, str)
        self.prompt = prompt
        self.response = response
        self.error = None
        self.data = data

    def get_prompt(self) -> str:
        prompt = self.prompt
        assert isinstance(prompt, str)
        return prompt

    def __repr__(self):
        assert isinstance(self.prompt, str)
        return f"PromptObject(prompt={self.prompt}, response={self.response}, data={self.data}, error={self.error}, id={id(self)})"

    def __dict__(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "error": self.error,
            "data": self.data,
        }
