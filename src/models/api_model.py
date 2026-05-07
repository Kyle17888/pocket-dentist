import os
import random
import threading
import time
from typing import TypeVar

from openai import OpenAI, RateLimitError

from src.models.base_model import BaseModel
from src.utils.common_utils import encode_image

JSON_T = dict | list
JSON_T_VAR = TypeVar("JSON_T_VAR", bound=JSON_T)

# Maximum number of retries on 429 rate-limit errors before giving up.
_MAX_RETRIES = 8

class APIModel(BaseModel):
    def __init__(self, model_name: str, temperature: float, base_url: str, api_key: str, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # max_retries=0: disable the SDK's built-in retry so we control it ourselves
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._usage_lock = threading.Lock()

    @staticmethod
    def _backoff_sleep(attempt: int):
        """Exponential backoff with full jitter, floor 1s: sleep = U(1, min(60, 2^(attempt+1)))."""
        cap = min(60.0, 2.0 ** (attempt + 1))  # 2, 4, 8, 16, 32, 60 ...
        wait = random.uniform(1.0, cap)
        time.sleep(wait)

    def _call_with_retry(self, call_fn):
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return call_fn()
            except RateLimitError as e:
                if attempt == _MAX_RETRIES:
                    raise
                self._backoff_sleep(attempt)
        raise RuntimeError("Unreachable")

    def _accumulate_usage(self, usage):
        if usage is None:
            return
        with self._usage_lock:
            self._usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self._usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self._usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

    def get_usage(self) -> dict:
        return dict(self._usage)

    def generate_from_image_and_text(self, image_path: str, prompt: str, output_type: type[JSON_T_VAR] = dict, few_shot_messages: list = None):
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        query_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encode_image(image_path)}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt if "ovis" not in self.model_name.lower() else f"<image>\n{prompt}"
                }
            ]
        }
        messages = (few_shot_messages or []) + [query_message]
        response = self._call_with_retry(lambda: self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        ))
        self._accumulate_usage(response.usage)
        assert (res_text:=response.choices[0].message.content)
        if output_type is str:
            return res_text
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res

    def generate_from_images_and_text(self, image_paths: list[str], prompt: str, output_type: type[JSON_T_VAR] = dict, few_shot_messages: list = None):
        """Generate response from multiple images and text prompt (for multi-image datasets like COde)."""
        content = []
        for image_path in image_paths:
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{encode_image(image_path)}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt if "ovis" not in self.model_name.lower() else f"<image>\n{prompt}"
        })
        query_message = {"role": "user", "content": content}
        messages = (few_shot_messages or []) + [query_message]
        response = self._call_with_retry(lambda: self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        ))
        self._accumulate_usage(response.usage)
        assert (res_text:=response.choices[0].message.content)
        if output_type is str:
            return res_text
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res

    def generate_from_text(self, prompt: str, output_type: type[JSON_T_VAR] = dict, temperature: float = None):
        response = self._call_with_retry(lambda: self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature else self.temperature,
            max_completion_tokens=self.max_tokens,
        ))
        self._accumulate_usage(response.usage)
        assert (res_text:=response.choices[0].message.content)
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res
