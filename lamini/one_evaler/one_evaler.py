import logging
from typing import Dict, List, Optional, Union
import uuid

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.api.utils.supported_models import LLAMA_31_8B_INST

logger = logging.getLogger(__name__)

class LaminiOneEvaler:
    """
    Lamini One Evaler SDK pkg.
    """
    def __init__(
        self,
        test_model_id: str,
        eval_data: List[Dict[str, str]],
        test_eval_type: str,
        eval_data_id: Optional[str] = '',
        base_model_id: Optional[str] = '',
        base_eval_type: Optional[str] = 'classifier',
        sbs: bool= False,
        fuzzy: bool = False,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.test_model_id = test_model_id
        self.eval_data = eval_data        
        if not eval_data_id:
            self.eval_data_id=str(uuid.uuid4())
        else:
            self.eval_data_id = eval_data_id
        self.test_eval_type = test_eval_type
        self.base_model_id=base_model_id
        self.base_eval_type=base_eval_type
        #TEMP restriction
        if self.test_eval_type!='classifier' or self.base_eval_type!='classifier':
            raise ValueError("Currently only classifier can use one eval.")
        self.sbs=sbs
        self.fuzzy=fuzzy

    def run(self):
        """
        Run Lamini One Evaler.
        Currently only support classifier. 
        """
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"eval/run",
            "post",
            {  
                "test_model_id": self.test_model_id,
                "test_eval_type": self.test_eval_type,
                "eval_data":self.eval_data,
                "eval_data_id":self.eval_data_id,
                "base_model_id":self.base_model_id,
                "base_eval_type": self.base_eval_type,
                "sbs": self.sbs,
                "fuzzy_comparison":self.fuzzy

            },
        )
        self.eval_job_id = resp["eval_job_id"]
        return resp
