import json
from lamini.experiment.generators.base_generator import BaseGenerator
from typing import Optional, Dict
import os
from openai import OpenAI


class GlossaryGenerator(BaseGenerator):
    """
    A generator class for creating glossaries from database schemas and queries.

    Attributes:
        use_gpt (bool): Indicates if GPT is being used.
        client: The client to communicate with the language model.
    """

    def __init__(
        self,
        model=None,
        client=None,
        name=None,
        role=None,
        instruction=None,
        output_type=None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the GlossaryGenerator with the provided parameters.

        Args:
            model (str): The model to be used.
            client: The client used for the language model.
            name (str): The name of the generator.
            role (str): The role description of the generator.
            instruction (str): Instruction for generating glossary.
            output_type (dict): Expected output type.
            api_key (str): API key for authentication.
            api_url (str): API endpoint URL.
            **kwargs: Additional keyword arguments.
        """
        self.use_gpt = False
        if model == "gpt-4o":
            # Configure the OpenAI client for GPT-based models
            if not api_key:
                api_key = os.getenv("LAMINI_API_KEY")
            if not api_key:
                raise ValueError("Please set LAMINI_API_KEY environment variable")
            if not api_url:
                api_url = "https://api.lamini.ai/inf"
            client = OpenAI(api_key=api_key, base_url=api_url)
            self.use_gpt = True
            self.client = client

        # Default instruction for generating a glossary.

        instruction = """
        Task: Given the following database schema and a list of questions and SQL pairs, and user provided glossary,
        generate a new glossary of terms and abbreviations that appear in the schema and queries.
        Schema: {schema}
        Queries: {queries}
        Original Glossary: {input_glossary}\n
        """

        instruction += """
        Please ONLY provide updated glossary in JSON format with a key 'glossary' that maps to a list of dictionaries,
        where each dictionary has 'input' and 'output' keys. Do not have any markdown formatting for output JSON.
        """

        output_type = output_type or {"glossary": "array"}

        # All input fields are expected to be strings.
        self.input = {"schema": "str", "queries": "str", "input_glossary": "str"}
        super().__init__(
            client=client,
            model=model,
            name=name or "GlossaryGenerator",
            role=role
            or "You are an expert at generating glossaries from schemas and queries.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def __call__(self, prompt_obj, debug=False):
        """
        Calls the glossary generator with the given input data.

        Args:
            prompt_obj: The prompt object containing input data.
            debug (bool): Flag to enable debug mode.

        Returns:
            The processed prompt object with a response.
        """

        if "input_glossary" not in prompt_obj.data:
            prompt_obj.data["input_glossary"] = ""
        if self.use_gpt:
            prompt = f"""
            {self.role}

            Schema: {prompt_obj.data['schema']}
            Queries: {prompt_obj.data['queries']}
            Original Glossary: {prompt_obj.data['input_glossary']}

            Generate a glossary of terms and abbreviations in JSON format.
            The output should be a JSON object with a key 'glossary' that maps to a list of dictionaries,
            where each dictionary has 'input' and 'output' keys.
            """
            output = self.client.completions.create(
                model="gpt-4o", prompt=prompt, max_tokens=10000
            )
            response_text = output.choices[0].text.strip()

            # Remove markdown formatting if present
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines)

            try:
                json_response = json.loads(response_text)
                prompt_obj.response = json_response
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print("Response text:", response_text)
                prompt_obj.response = {"glossary": []}
            return self.postprocess(prompt_obj)
        else:
            return super().__call__(prompt_obj, debug=debug)

    def postprocess(self, prompt_obj):
        """
        Processes the response from the language model.

        Args:
            prompt_obj: The prompt object containing the response.

        Returns:
            The prompt object with a processed response.
        """
        if not prompt_obj.response:
            self.logger.warning("Empty response from model for schema input.")
            prompt_obj.response = {"glossary": []}
            return prompt_obj

        generated_glossary = []
        if isinstance(prompt_obj.response, str):
            response_text = prompt_obj.response.strip()
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines)
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict) and "glossary" in parsed:
                    generated_glossary = parsed["glossary"]
                elif isinstance(parsed, list):
                    generated_glossary = parsed
                else:
                    generated_glossary = []
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding generated glossary: {e}")
                generated_glossary = []
        elif (
            isinstance(prompt_obj.response, dict) and "glossary" in prompt_obj.response
        ):
            generated_glossary = prompt_obj.response["glossary"]
        elif isinstance(prompt_obj.response, list):
            generated_glossary = prompt_obj.response
        else:
            generated_glossary = []

        formatted_generated = []
        for entry in generated_glossary:
            inp = entry.get("input", "")
            out = entry.get("output", "")
            if "\n" in inp or "\n" in out:
                inputs = [i.strip() for i in inp.split("\n") if i.strip()]
                outputs = [o.strip() for o in out.split("\n") if o.strip()]
                for i, o in zip(inputs, outputs):
                    formatted_generated.append({"input": i, "output": o})
            else:
                formatted_generated.append(
                    {"input": inp.strip(), "output": out.strip()}
                )

        existing_glossary = prompt_obj.data.get("input_glossary", [])
        if isinstance(existing_glossary, str):
            existing_glossary = existing_glossary.strip()
            if existing_glossary.startswith("["):
                try:
                    existing_glossary = json.loads(existing_glossary)
                except Exception as e:
                    self.logger.error(
                        f"Error parsing existing glossary as JSON array: {e}"
                    )
                    existing_glossary = []
            else:
                items = []
                for line in existing_glossary.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            items.append(item)
                        except Exception as e:
                            self.logger.error(
                                f"Error parsing line in existing glossary: {e}"
                            )
                existing_glossary = items
        elif not isinstance(existing_glossary, list):
            existing_glossary = []

        new_glossary = existing_glossary + formatted_generated
        prompt_obj.response = {"glossary": new_glossary}
        return prompt_obj
