import json
from typing import Dict, List, Optional

import requests

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class MemoryRAG:

    def __init__(
        self,
        job_id: int = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        self.job_id = job_id
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/alpha/memory-rag"
        self.model_name = model_name

    def memory_index(
        self,
        documents: List,
    ) -> str:
        if self.model_name is None:
            raise Exception("model_name must be set in order to use memory_index")
        payload = {"model_name": self.model_name}

        files = [
            (
                "files",
                (
                    file_path,
                    open(file_path, "rb"),
                ),
            )
            for file_path in documents
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.request(
            "POST",
            self.api_prefix + f"/train",
            headers=headers,
            data=payload,
            files=files,
        )

        json_response = response.json()
        self.job_id = json_response["job_id"]
        return json_response

    def status(self) -> str:
        if self.job_id is None:
            raise Exception("job_id must be set in order to get status")
        params = {"job_id": self.job_id}
        resp = make_web_request(
            self.api_key, self.api_prefix + f"/status", "post", params
        )
        return resp

    def query(self, prompt: str, k: int = 3) -> str:
        if self.job_id is None:
            raise Exception("job_id must be set in order to query")
        params = {
            "prompt": prompt,
            "model_name": self.model_name,
            "job_id": self.job_id,
            "rag_query_size": k,
        }
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/completions",
            "post",
            params,
        )
        return resp

    def add_index(self, prompt: str) -> str:
        if self.job_id is None:
            raise Exception("job_id must be set in order to add to index")
        params = {"prompt": prompt, "job_id": self.job_id}
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/add-index",
            "post",
            params,
        )
        return resp

    def get_logs(self) -> List[str]:
        """Get training logs for a memory RAG job.

        Args:
            job_id: The ID of the memory RAG job

        Returns:
            List of log lines
        """
        if self.job_id is None:
            raise Exception("job_id must be set in order to get job logs")
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/training_log/{self.job_id}",
            "get",
        )
        return resp
    
    def document_experiment(
        self,
        documents: List[str],
        metadata: Dict = None,
        pipeline_config: Dict = None,
        model_config: Dict = None,
        num_questions: int = 3,
        chunk_size: int = 1024,
        role: str = None,
        question_task: str = None,
        answer_task: str = None,
        spot_check: Dict = None,
        project_id: str = None,
    ) -> Dict:

        if not documents:
            raise Exception("At least one document must be provided")

        # Prepare payload
        payload = {
            "metadata": metadata or {},
            "pipeline_config": pipeline_config or {"validate_answers": False, "dedupe": False},
            "model_config": model_config or {
                "question_model": self.model_name,
                "answer_model": self.model_name,
                "validator_model": self.model_name
            },
            "num_questions": num_questions,
            "chunk_size": chunk_size,
            "role": role or "You are an AI assistant analyzing a document.",
            "question_task": question_task or "Generate specific questions that can be answered using only the facts in this content section.",
            "answer_task": answer_task,
            "project_id": project_id,
        }

        if spot_check:
            payload["spot_check"] = spot_check

        # Prepare files
        files = [
            ("files", (doc, open(doc, "rb")))
            for doc in documents
        ]
        
        files.append(("payload", (None, json.dumps(payload))))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.request(
            "POST",
            self.api_prefix + "/DocumentExperiment",
            headers=headers,
            files=files
        )
        
        json_response = response.json()
        self.job_id = json_response["job_id"]
        return json_response

    def get_spot_check_results(self) -> Dict:

        if self.job_id is None:
            raise Exception("job_id must be set to get spot check results")

        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"/spot_check/{self.job_id}",
            "get"
        )
        return resp
    
    def sql_experiment(
        self,
        metadata: Dict = None,
        pipeline_config: Dict = None,
        model_config: Dict = None,
        num_questions: int = 3,
        role: str = None,
        schemas: Dict = None,
        seed_questions: Dict = None,
        query_task: str = None,
        question_task: str = None,
        spot_check: Dict = None,
        project_id: str = None,
    ) -> Dict:
        
        payload = {
            "metadata": metadata or {},
            "pipeline_config": pipeline_config or {"validate_answers": True, "dedupe": True},
            "model_config": model_config or {
                "question_model": self.model_name,
                "answer_model": self.model_name,
                "validator_model": self.model_name
            },
            "num_questions": num_questions,
            "role": role,
            "schemas": schemas,
            "seed_questions": seed_questions,
            "query_task": query_task,
            "question_task": question_task,
            "project_id": project_id,
        }

        if spot_check:
            payload["spot_check"] = spot_check

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.request(
            "POST",
            self.api_prefix + "/SQLExperiment",
            headers=headers,
            json=payload
        )
        
        json_response = response.json()
        self.job_id = json_response["job_id"]
        return json_response

