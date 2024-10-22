import asyncio
from typing import Any, Dict

from lamini.api.rest_requests import make_async_web_request
from lamini.api.utils.batch_completions import BatchCompletions
from lamini.api.utils.batch_embeddings import BatchEmbeddings


class PipelineClient:

    async def embedding(self, client, key, url, json):
        result = await make_async_web_request(client, key, url, "post", json)
        result = result["embedding"]
        return result

    async def completions(self, client, key, url, json: dict) -> Dict[str, Any]:
        result = await make_async_web_request(client, key, url, "post", json)
        return result

    async def batch_completions(
        self,
        client,
        json: dict,
    ) -> Dict[str, Any]:
        batch_api = BatchCompletions()
        submit_response = await batch_api.async_submit(
            prompt=json["prompt"],
            model_name=json["model_name"],
            output_type=json["output_type"],
            max_new_tokens=json["max_new_tokens"],
        )
        while True:
            await asyncio.sleep(5)
            result = await batch_api.async_check_result(submit_response["id"])
            if result and all(result["finish_reason"]):
                break
        return result

    async def batch_embeddings(
        self,
        client,
        json: dict,
    ) -> Dict[str, Any]:
        batch_api = BatchEmbeddings()
        submit_response = await batch_api.async_submit(
            prompt=json["prompt"], model_name=json["model_name"]
        )
        while True:
            await asyncio.sleep(1)
            result = await batch_api.async_check_result(submit_response["id"])
            if result:
                break
        result = result["embedding"]
        return result
