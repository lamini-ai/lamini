from typing import List, Optional, Union
import time
import json
from llama.program.util.run_ai import (
    get_configured_url,
    get_ui_url,
    get_model_config,
    make_web_request,
)
from llama.program.util.config import edit_config
from concurrent.futures import ThreadPoolExecutor
import logging
import lamini

logger = logging.getLogger(__name__)


class Lamini:
    def __init__(
        self,
        id: str,
        model_name: str,
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None,
        config: dict = {},
    ):
        self.id = id
        self.model_name = model_name
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.config = edit_config(config)
        url = get_configured_url()
        self.model_config = get_model_config()
        self.api_prefix = url + "/v2/lamini/"
        self.job_id = None
        self.upload_file_path = None

    """
    https://lamini-ai.github.io/API/completions/

    - arguments are maps
    - user can optionally specify type in output_type like "Answer#bool"
      valid types are:
        'string'/'str', 'integer'/'int', 'float', 'bool'/'boolean'
    - input type is runtime type of input value.
      if runtime type is not a valid type above, then default to string

    Ex 1:
      input_value = {"question": "What is the hottest day of the year?"}
      output_type = {"Answer": "An answer to the question"}
    Ex 2:
      input_value = {"question": "What is the hottest day of the year?",
                     "question2": "What is for lunch?"}
      output_type = {"Answer": "An answer to the question",
                     "Answer2": "An answer to the question2"}
    """

    def __call__(
        self,
        input: Union[dict, List[dict]],
        output_type,
        stop_tokens: Union[str, List[str]] = None,
        model_name=None,
        enable_peft=None,
        random=None,
        max_tokens=None,
        streaming=None,
    ):
        if isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens]
        if isinstance(input, List):
            results = []
            batch_size = lamini.batch_size

            def work(chunk):
                req_data = self.make_llm_req_map(
                    self.id,
                    model_name or self.model_name,
                    chunk,
                    output_type,
                    self.prompt_template,
                    stop_tokens,
                    enable_peft,
                    random,
                    max_tokens,
                    streaming,
                )
                url = self.api_prefix + "completions"
                return make_web_request("post", url, self.api_key, req_data)

            with ThreadPoolExecutor(max_workers=lamini.max_workers) as executor:
                chunks = [
                    input[i : i + batch_size] for i in range(0, len(input), batch_size)
                ]
                results = executor.map(work, chunks)
                results = [item for sublist in results for item in sublist]

            return results

        req_data = self.make_llm_req_map(
            self.id,
            model_name or self.model_name,
            input,
            output_type,
            self.prompt_template,
            stop_tokens,
            enable_peft,
            random,
            max_tokens,
            streaming,
        )
        url = self.api_prefix + "completions"
        return make_web_request("post", url, self.api_key, req_data)

    # https://lamini-ai.github.io/API/data/
    # data must be a single map or a list of maps
    # cannot be a list of input/output map pairs
    def save_data(self, data):
        paired = []
        contextual = []
        result = None
        if type(data) == list:
            for d in data:
                if type(d) == list and len(d) == 2:
                    paired.append(d)
                else:
                    contextual.append(d)
            if len(paired) > 0:
                result = self.save_data_pairs(paired)
            data = contextual
        if len(contextual) == 0 and result is not None:
            return result
        req_data = self.make_save_data_req_map(data)
        url = self.api_prefix + "data"

        return make_web_request("post", url, self.api_key, req_data)

    # https://lamini-ai.github.io/API/data/
    # data can only be a list of input/output map pairs, like
    # [[input_map1, output_map1], [input_map2, output_map2], ...]
    def save_data_pairs(self, data):
        req_data = self.make_save_data_pairs_req_map(data)
        url = self.api_prefix + "data_pairs"
        return make_web_request("post", url, self.api_key, req_data)

    def upload_data(self, data_pairs, azure_dir_name="default"):
        url = self.api_prefix + "upload_data"
        self.upload_file_path = make_web_request(
            "post",
            url,
            self.api_key,
            {"id": self.id, "azure_dir_name": azure_dir_name, "data": data_pairs},
        )

    def upload_file(self, data_path, azure_dir_name="default"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        f.close()
        self.upload_file_path = self.upload_data(data, azure_dir_name)

    # https://lamini-ai.github.io/API/train/
    # just submit the job, no polling
    def train_async(
        self,
        data: Optional[List] = None,
        finetune_args: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
    ):
        req_data = {"id": self.id, "model_name": self.model_name}
        if data is not None:
            req_data["data"] = data
        if self.upload_file_path is not None:
            req_data["upload_file_path"] = self.upload_file_path
        if self.prompt_template is not None:
            req_data["prompt_template"] = self.prompt_template
        if finetune_args is not None:
            req_data["finetune_args"] = finetune_args
        if enable_peft is not None:
            req_data["enable_peft"] = enable_peft
        if peft_args is not None:
            req_data["peft_args"] = peft_args
        if is_public is not None:
            req_data["is_public"] = is_public
        if use_cached_model is not None:
            req_data["use_cached_model"] = use_cached_model
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        url = self.api_prefix + "train"

        job = make_web_request("post", url, self.api_key, req_data)
        self.job_id = job["job_id"]
        ui_url = get_ui_url()
        print(
            f"Training job submitted! Check status of job {self.job_id} here: {ui_url}/train/{self.job_id}"
        )

        return job

    # https://lamini-ai.github.io/API/train/
    # continuously poll until the job is completed
    def train(
        self,
        data: Optional[List] = None,
        finetune_args: Optional[dict] = None,
        enable_peft: Optional[bool] = None,
        peft_args: Optional[dict] = None,
        is_public: Optional[bool] = None,
        use_cached_model: Optional[bool] = None,
        **kwargs,
    ):
        job = self.train_async(
            data,
            finetune_args=finetune_args,
            enable_peft=enable_peft,
            peft_args=peft_args,
            is_public=is_public,
            use_cached_model=use_cached_model,
        )

        try:
            status = self.check_job_status(job["job_id"])
            if status["status"] == "FAILED":
                print(f"Job failed: {status}")
                return status

            while status["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
                if kwargs.get("verbose", False):
                    print(f"job not done. waiting... {status}")
                time.sleep(30)
                status = self.check_job_status(job["job_id"])
                if status["status"] == "FAILED":
                    print(f"Job failed: {status}")
                    return status
                elif status["status"] == "CANCELLED":
                    print(f"Job canceled: {status}")
                    return status
            print(
                f"Finetuning process completed, model name is: {status['model_name']}"
            )
        except KeyboardInterrupt as e:
            print("Cancelling job")
            return self.cancel_job(job["job_id"])

        return status

    # https://lamini-ai.github.io/API/train_job_cancel/
    def cancel_job(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/cancel"

        return make_web_request("post", url, self.api_key, {})

    def cancel_all_jobs(
        self,
    ):
        url = self.api_prefix + "train/jobs/cancel"

        return make_web_request("post", url, self.api_key, {})

    # https://lamini-ai.github.io/API/train_job_status/
    def check_job_status(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id)

        return make_web_request("get", url, self.api_key, {})

    def get_jobs(self):
        url = self.api_prefix + "train/jobs"

        return make_web_request("get", url, self.api_key, {})

    # https://lamini-ai.github.io/API/eval_results/#request
    def evaluate(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
        url = self.api_prefix + "train/jobs/" + str(job_id) + "/eval"

        return make_web_request("get", url, self.api_key, {})

    # https://lamini-ai.github.io/API/delete_data/
    def delete_data(self):
        url = self.api_prefix + "delete_data"

        return make_web_request("post", url, self.api_key, {"id": self.id})

    # check if two maps have the same keys and value types
    def same_type(self, t1, t2):
        if t1.keys() != t2.keys():
            return False

        for k in t1.keys():
            if type(t1[k]) != type(t2[k]):
                return False

        return True

    def is_correct_type(self, t):
        return isinstance(t, dict)

    def make_save_data_req_map(self, data):
        req_data = {}
        req_data["id"] = self.id
        req_data["data"] = []

        if type(data) != list:
            data = [data]

        assert len(data) > 0
        type_reference = data[0]
        if type(type_reference) == list and len(type_reference) == 1:
            type_reference = type_reference[0]

        for d in data:
            if type(d) == list and len(d) == 1:
                d = d[0]

            if not self.is_correct_type(d):
                raise TypeError(
                    "data type must be a list of type dict or a single dict"
                )

            if not self.same_type(type_reference, d):
                raise TypeError(
                    "All data must have the same keys and value types. If you are trying to save input and output data with different types, use save_data_pairs instead."
                )
            req_data["data"].append(d)

        return req_data

    def make_save_data_pairs_req_map(self, data):
        req_data = {}
        req_data["id"] = self.id
        req_data["data"] = []
        type_err_msg = "data must be in the form [[input_map1, output_map1], [input_map2, output_map2], ...]. Each element in the data array must have the same type."

        if type(data) != list:
            raise TypeError(type_err_msg)

        for d in data:
            if len(d) != 2:
                raise TypeError(type_err_msg)

            input_data = d[0]
            output_data = d[1]

            if (
                type(input_data) != dict
                or type(output_data) != dict
                or not self.same_type(input_data, data[0][0])
                or not self.same_type(output_data, data[0][1])
            ):
                raise TypeError(type_err_msg)

            req_data["data"].append(d)

        return req_data

    def make_llm_req_map(
        self,
        id,
        model_name,
        input_value,
        output_type,
        prompt_template,
        stop_tokens,
        enable_peft,
        random,
        max_tokens,
        streaming,
    ):
        req_data = {}
        req_data["id"] = id
        req_data["model_name"] = model_name
        req_data["in_value"] = input_value
        req_data["out_type"] = output_type
        if streaming is not None:
            req_data["streaming"] = streaming
        if prompt_template is not None:
            req_data["prompt_template"] = prompt_template
        if stop_tokens is not None:
            req_data["stop_tokens"] = stop_tokens
        if enable_peft is not None:
            req_data["enable_peft"] = enable_peft
        if random is not None:
            req_data["random"] = random
        if max_tokens is not None:
            req_data["max_tokens"] = max_tokens
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        return req_data
