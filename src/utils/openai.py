from copy import deepcopy
from typing import Any, Dict, List

from common import OPENAI_REQUEST
from typing import Optional


def generate_openai_api_request(
    custom_id: str,
    model: str,
    messages: List[Dict[str, str]],
    logprobs: bool = True,
    temperature: Optional[float] = None,
    num_choices: int = 1,
    engine: str = "openai",
) -> Dict[str, Any]:
    # url = "/chat/completions" if engine == "azure" else "/v1/chat/completions"
    url = "/chat/completions"
    return {
        **deepcopy(OPENAI_REQUEST),
        "custom_id": custom_id,
        "url": url,
        "body": {
            "model": model,
            "messages": messages,
            "logprobs": logprobs,
            "temperature": temperature,
            "n": num_choices,
        },
    }
