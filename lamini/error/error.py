class LaminiError(Exception):
    def __init__(
        self,
        message=None,
    ):
        super(LaminiError, self).__init__(message)


class ProjectNotFoundError(LaminiError):
    """The project was not found in the database."""


class DuplicateResourceError(LaminiError):
    """The project was not found in the database."""


class ModelNotFoundError(LaminiError):
    """The model name is invalid. Make sure it's a valid model in Huggingface or a finetuned model"""


class JobNotFoundError(LaminiError):
    """No jobs were found which match the specified criteria."""


class APIError(LaminiError):
    """There is an internal error in the Lamini API"""


class AuthenticationError(LaminiError):
    """The Lamini API key is invalid"""


class RateLimitError(LaminiError):
    """The QPS of requests to the API is too high"""


class UserError(LaminiError):
    """The user has made an invalid request"""


class APIUnprocessableContentError(LaminiError):
    """Invalid request format. Consider upgrading lamini library version"""


class UnavailableResourceError(LaminiError):
    """Model is still downloading"""


class ServerTimeoutError(LaminiError):
    """Model is still downloading"""


class DownloadingModelError(LaminiError):
    """Downloading model"""


class RequestTimeoutError(LaminiError):
    """Request Timeout. Please try again."""


class OutdatedServerError(LaminiError):
    """Outdated Server Version"""
