import os
import json
import openai
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BaseOpenAIClient:
    """Simplified base class that only handles OpenAI client setup and basic async execution"""

    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        # self.api_key = ""
        # self.api_url = "https://staging.lamini.ai"

        self.client = openai.OpenAI(
            api_key=self.api_key, base_url=f"{self.api_url}/inf"
        )

    async def initialize_async_client(self):
        """Create async client with same configuration"""
        return openai.AsyncOpenAI(api_key=self.api_key, base_url=f"{self.api_url}/inf")

    async def execute_completion(
        self,
        model: str,
        prompt: str,
        response_schema: Dict[str, Any],
        schema_name: str = None,
    ) -> Dict[str, Any]:
        """Basic completion execution with error handling
        Args:
            model: Model identifier
            prompt: The prompt text
            response_schema: The schema structure
            schema_name: Optional name for the schema. If not provided, will use 'response_format'
        """
        try:
            async with await self.initialize_async_client() as async_client:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name or "response_format",
                            "schema": response_schema,
                            "strict": True,
                        },
                    },
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(
                f"Error in completion: {e}. Prompt: {prompt}. Response schema: {response_schema}. Model: {model}."
            )
            return {}


@dataclass
class Step:
    """Stores prompt and metadata information for a pipeline step"""

    instruction_prompt: str
    metadata_key: str


class OpenAIStep:
    """Replacement for StepGenerator that uses BaseOpenAIClient"""

    def __init__(
        self,
        step_id: str,
        model_name: str,
        output_type: Dict[str, Any],
        client: BaseOpenAIClient,
    ):
        self.step_id = step_id
        self.model_name = model_name
        self.output_type = output_type
        self.client = client
        self.steps: List[Step] = []

    def add_step(self, instruction_prompt: str, metadata_key: str) -> None:
        """Add a step to the pipeline"""
        self.steps.append(Step(instruction_prompt, metadata_key))

    def make_prompt(self, data: Dict[str, Any]) -> str:
        """Create formatted prompt from steps"""
        prompt_parts = []
        for step in self.steps:
            prompt_parts.extend(
                [
                    step.instruction_prompt,
                    "\n\n+++++++++++++++++++++++++++++++++\n\n",
                    str(data.get(step.metadata_key, "")),
                    "\n\n+++++++++++++++++++++++++++++++++\n\n",
                ]
            )
        return "".join(prompt_parts)

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data point through this step"""
        try:
            # Build prompt
            prompt = self.make_prompt(data)

            # Create response schema from output_type
            response_schema = {
                "type": "object",
                "properties": {
                    k: {"type": "string" if v == "str" else v}
                    for k, v in self.output_type.items()
                },
                "required": list(self.output_type.keys()),
            }

            # Execute completion
            result = await self.client.execute_completion(
                model=self.model_name,
                prompt=prompt,
                response_schema=response_schema,
                schema_name=f"{self.step_id}_response",
            )

            # Update data with results
            output_data = data.copy()
            output_data[f"{self.step_id}_prompt"] = prompt
            output_data.update(result)

            return output_data

        except Exception as e:
            logger.error(
                f"Error in step {self.step_id}: {str(e)}. Prompt: {prompt}. Response schema: {response_schema}. Model: {self.model_name}."
            )
            raise


class OpenAIPipeline:
    """Replacement for Pipeline class using BaseOpenAIClient"""

    def __init__(self):
        self.steps: List[OpenAIStep] = []

    def add_node(self, node: OpenAIStep) -> None:
        """Add a processing step to the pipeline"""
        self.steps.append(node)
        setattr(self, node.step_id, node)

    async def process_single(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data point through all steps"""
        current_data = initial_data
        for step in self.steps:
            current_data = await step.process(current_data)
        return current_data

    async def process_batch(
        self, data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process multiple data points through the pipeline"""
        results = []
        for data in data_points:
            result = await self.process_single(data)
            results.append(result)
        return results
