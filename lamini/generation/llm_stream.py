import time
from typing import Any, Dict, Iterator, List, Optional, TypeVar, Union


import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request

T = TypeVar("T")


class LLMStream:
    """Handler for formatting and POST request for the batch submission API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.current_minibatch_result = None
        self.current_minibatch_index = 0
        self.polling_interval = 1
        self.error_count = 0
        self.max_errors = 3

    def generate(
        self,
        prompts: Union[Iterator, List],
        model_name: str,
        output_type: Optional[dict] = None,
        max_new_tokens: Optional[int] = None,
    ):
        if isinstance(prompts, list):
            prompts = iter(prompts)
        minibatch_stream = self.minibatch(prompts, lambda: lamini.batch_size)
        for minibatch in minibatch_stream:
            # print("MINIBATCH", minibatch)
            assert isinstance(minibatch, list)
            req_data = self.make_llm_req_map(
                prompt=minibatch,
                model_name=model_name,
                output_type=output_type,
                max_new_tokens=max_new_tokens,
            )
            resp = make_web_request(
                self.api_key,
                self.api_prefix + "batch_completions",
                "post",
                req_data,
            )
            # print("rrrrrr", resp)
            result_stream = self.get_minibatch_result_stream(resp["id"])
            # print("streammmm", result_stream)
            for result in result_stream:
                yield result

            self.current_minibatch_result = None
            self.current_minibatch_index = 0

    def get_minibatch_result_stream(self, id: str):
        # Keep polling until results are yielded
        while (
            self.current_minibatch_result is None
            or self.current_minibatch_result == {}
            or not all(self.current_minibatch_result["finish_reason"])
        ):
            time.sleep(self.polling_interval)
            try:
                self.current_minibatch_result = make_web_request(
                    self.api_key,
                    self.api_prefix + f"batch_completions/{id}/result",
                    "get",
                )
                if self.current_minibatch_result == {}:
                    continue

                # Yield all most recently available results
                available_results = len(
                    self.current_minibatch_result["finish_reason"]
                ) - self.current_minibatch_result["finish_reason"].count(None)
                for i in range(self.current_minibatch_index, available_results):
                    result = {
                        "output": self.current_minibatch_result["outputs"][i],
                        "finish_reason": self.current_minibatch_result["finish_reason"][
                            i
                        ],
                    }
                    yield result

                self.current_minibatch_index = available_results

            except Exception as e:
                self.error_count += 1
                if self.error_count > self.max_errors:
                    raise e

    def check_result(
        self,
        id: str,
    ) -> Dict[str, Any]:
        """Check for the result of a batch request with the appropriate batch id."""
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"batch_completions/{id}/result",
            "get",
        )
        return resp

    def make_llm_req_map(
        self,
        model_name: str,
        prompt: Union[str, List[str]],
        output_type: Optional[dict] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        return req_data

    def minibatch(
        self,
        iterator: Iterator[T],
        size_fn,
    ) -> Iterator[list[T]]:
        """Yield successive n-sized chunks from lst."""
        finished = False

        while not finished:
            results: list[T] = []
            size = size_fn()

            for _ in range(size):
                try:
                    result = None
                    while result is None:
                        result = next(iterator)
                except StopIteration:
                    finished = True
                else:
                    results.append(result)

            if results:
                yield results
