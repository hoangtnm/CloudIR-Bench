import datetime
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import vllm
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI

load_dotenv()

# import ollama
# from utils.reasoning import DeepSeekR1ReasoningParser


class LLMClient:
    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.0-flash",
        do_reasoning: bool = False,
        engine: str = "api",
    ) -> None:
        self.api_keys = api_keys
        self.model = model
        self.do_reasoning = do_reasoning
        self.engine = engine

    def _retry_until_success(self, func, *args, interval=10, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e, flush=True)
                time.sleep(interval)

    def _generate_gemini(
        self,
        model: str,
        api_key: str,
        prompt: str,
        num_candidates: int = 1,
        temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        outputs = []

        def call_gemini():
            gemini_outputs = []
            client = genai.Client(api_key=api_key)
            config = GenerateContentConfig(
                candidate_count=num_candidates,
                temperature=temperature,
                # response_logprobs=True,
                # logprobs=1,
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            # model_version = response.model_version
            candidates = response.candidates
            for choice in candidates:  # type: ignore
                avg_logprobs = choice.avg_logprobs
                text = choice.content.parts[0].text.strip()  # type: ignore
                gemini_outputs.append(
                    {
                        "content": text,
                        "reasoning_content": None,
                        "avg_logprobs": avg_logprobs,
                    }
                )
            return gemini_outputs

        outputs = self._retry_until_success(call_gemini)
        return outputs

    def _generate_openai(
        self,
        model: str,
        chat_legacy: bool,
        api_key: str,
        prompt: str,
        num_candidates: int = 1,
        temperature: Optional[float] = None,
        api_endpoint: Optional[str] = None,
        logprobs: bool = False,
    ) -> List[Dict[str, Any]]:
        client = OpenAI(api_key=api_key, base_url=api_endpoint)
        messages = [{"role": "user", "content": prompt}]

        def call_openai():
            if chat_legacy:
                return client.completions.create(
                    model=model,
                    prompt=prompt,
                    logprobs=int(logprobs),
                    n=num_candidates,
                    temperature=temperature,
                )
            return client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=logprobs,
                n=num_candidates,
                temperature=temperature,
            )

        completion = self._retry_until_success(call_openai)
        outputs = []
        for choice in completion.choices:
            choice = choice.to_dict()
            logprobs = choice["logprobs"]
            reasoning_content = None

            if chat_legacy:
                content = choice["text"].strip()
                tokens = logprobs["tokens"]
                token_logprobs = logprobs["token_logprobs"]
                logprobs_content = [
                    {"token": token, "logprob": logprob} for token, logprob in zip(tokens, token_logprobs)
                ]
                logprobs = {"content": logprobs_content}
            else:
                message = choice["message"]
                content = message["content"].strip()
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    reasoning_content = reasoning_content.strip()
            outputs.append(
                {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "logprobs": logprobs,
                }
            )
        return outputs

    def generate(
        self,
        chat_legacy: bool,
        prompt: str,
        num_candidates: int = 1,
        temperature: Optional[float] = None,
        logprobs: bool = False,
        interval: float = 5.0,
    ) -> List[Dict[str, Any]]:
        start = datetime.datetime.now()
        api_key = np.random.choice(self.api_keys)
        outputs = []

        if self.model.startswith("gemini"):
            outputs = self._generate_gemini(
                model=self.model,
                api_key=api_key,
                prompt=prompt,
                num_candidates=num_candidates,
                temperature=temperature,
            )

        else:
            if self.model.startswith(("gpt", "davinci")):
                openai_api_key = os.getenv("OPENAI_API_KEY")
                openai_api_endpoint = None
            else:
                openai_api_key = "EMPTY"
                openai_api_endpoint = "http://localhost:8000/v1"

            outputs = self._generate_openai(
                model=self.model,
                chat_legacy=chat_legacy,
                api_key=openai_api_key,
                prompt=prompt,
                num_candidates=num_candidates,
                temperature=temperature,
                api_endpoint=openai_api_endpoint,
                logprobs=logprobs,
            )
        end = datetime.datetime.now()
        elapsed = (end - start).total_seconds()
        if elapsed < interval:
            time.sleep(interval)

        return outputs

    def generate_batch(
        self,
        prompts: List[str],
        num_candidates: int = 1,
        temperature: float = 1.0,
    ):
        if self.model.startswith("gemini"):
            raise ValueError(f"{self.model} is not supported for generate_batch")

        if self.engine == "vllm":
            sampling_params = vllm.SamplingParams(n=num_candidates, temperature=temperature, top_p=0.95)
            llm = vllm.LLM(model=self.model)
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                # prompt = output.prompt
                generated_text = output.outputs[0].text
