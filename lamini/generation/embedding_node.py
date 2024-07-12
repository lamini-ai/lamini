import logging
import sys
from typing import AsyncIterator, Iterator, Optional, Union

from lamini.api.lamini_config import get_config
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode

logger = logging.getLogger(__name__)


class EmbeddingNode(GenerationNode):
    """ 
    This child class of GenerationNode is for use of specific calls
    for an embedding generated response. The main change is a reduction
    in the needed parameters for an Embedding response which is seen through
    the differences in this object overriden make_llm_req_map function when
    compared to the GenerationNode's make_llm_req_map function.
        The GenerationNode defintion:
            https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/generation/generation_node.py
        i.e. The EmbeddingNode does not need to supply the following keys:
            - output_type
            - max_tokens
            - max_new_tokens
            - model_config
            "type" is the main difference with a hard coded "embedding" 
            
    Parameters
    ----------
    model_name: Optional[str]
        Model name as referred to on HuggingFace https://huggingface.co/models
        
    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored 
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults
        
    config: dict
        Dictionary that is handled from the following script:
            https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py
        Configurations currently hold the following keys and data as a yaml format:
            local:
                url: <url>
            staging:
                url: <url>
            production:
                url: <url>
            
            local:
                key: <auth-key>
            staging:
                key: <auth-key>
            production:
                key:
                    <auth-key  
        

    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
    ):
        super(EmbeddingNode, self).__init__(
            model_name=model_name, api_key=api_key, api_url=api_url, config=config
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
    ):
        """ Call to the Lamini API to generate the output of the model_name
        requested. Before submitting the request to the async inference queue,
        the req_data is constructed from self.make_llm_req_map. This function
        is expected to be used in conjunction with async function calls or iterators.

        Parameters
        ----------
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]
            An iterator (Async or not) used as the input for the async call
            of get_query_prompt to encapsulate the text for model input 
            within a PromptObject

        model_name: Optional[str]
            Model name that will be passed to the API call
        
        Returns
        -------
        Generator: 
            A generator is returned from the parent class' generate call, 
            which in turn is returned from this function. For more information
            on the generator returned, refer to:
                https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_node.py#L48
                The parent EmbeddingNode of this class is itself a child of the GenerationNode,
                which is why you should refer to GenerationNode for parent functionality
        """

        assert isinstance(prompt, Iterator) or isinstance(prompt, AsyncIterator)
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
        )
        return self.async_inference_queue.submit(req_data)

    def make_llm_req_map(
        self,
        model_name: str,
        prompt: str,
    ):
        """Construct the request dictionary for API post requests.
        Only model_name and prompt are parameters as an EmbeddingNode
        is expected to be only sending "embedding" type requests. 

        Parameters
        ----------
        model_name: str
            Model name as referred to on HuggingFace https://huggingface.co/models
        
        prompt: str
            Prompt string that is sent to the model

        Returns
        -------
        req_data: Dict[str, Any]
            A dictionary is returned with the necessary keys for the API Post
            request.    
        """

        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["type"] = "embedding"
        return req_data
