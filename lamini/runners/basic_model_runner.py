from lamini.runners.base_runner import BaseRunner


class BasicModelRunner(BaseRunner):
    """This class is deprecated, use lamini.BaseRunner or lamini.Lamini instead."""

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-410m-deduped",
        system_prompt: str = None,
        prompt_template=None,
        api_key=None,
        api_url=None,
        config={},
        local_cache_file=None,
    ):
        super().__init__(
            config=config,
            system_prompt=system_prompt,
            model_name=model_name,
            prompt_template=prompt_template,
            api_key=api_key,
            api_url=api_url,
            local_cache_file=local_cache_file,
        )
