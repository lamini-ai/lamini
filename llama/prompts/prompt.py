from abc import ABCMeta


class BasePrompt(object, metaclass=ABCMeta):
    prompt_template = ""
