class LlamaError(Exception):
    def __init__(
        self,
        message=None,
    ):
        super(LlamaError, self).__init__(message)


class ModelNotFound(LlamaError):
    """The model name is invalid. Make sure it's a valid model in Huggingface or a finetuned model"""


class APIError(LlamaError):
    """There is an internal error in the Lamini API"""


class AuthenticationError(LlamaError):
    """The Lamini API key is invalid"""


class RateLimitError(LlamaError):
    """The QPS of requests to the API is too high"""


class UserError(LlamaError):
    """The user has made an invalid request"""


class APIUnprocessableContentError(LlamaError):
    """Invalid request format. Consider upgrading lamini library version"""


class UnavailableResourceError(LlamaError):
    """Model is still downloading"""


class ServerTimeoutError(LlamaError):
    """Model is still downloading"""
