import json
from abc import ABC, abstractmethod
from typing import TypeVar

from json_repair import repair_json

JSON_T = dict | list
JSON_T_VAR = TypeVar("JSON_T_VAR", bound=JSON_T)
def parse_json(json_str: str, ret_t: type[JSON_T_VAR] = dict) -> JSON_T_VAR:
    if json_str.startswith("```json") and json_str.endswith("```"):
        json_str = json_str[7:-3].strip()

    try:
        json_str = repair_json(json_str, ensure_ascii=False)
        res_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {json_str}") from e
    
    assert isinstance(res_json, ret_t), f"Expected type {ret_t}, but got {type(res_json)}"
    return res_json

class BaseModel(ABC):
    def __init__(self):
        pass
    
    @staticmethod
    def j2t(json_dict: JSON_T) -> str:
        return json.dumps(json_dict, ensure_ascii=False, indent=2)
    
    @staticmethod
    def t2j(json_str: str, t: type[JSON_T_VAR] = dict) -> JSON_T_VAR:
        return parse_json(json_str, t)
    
    @abstractmethod
    def generate_from_image_and_text(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def generate_from_text(self, *args, **kwargs):
        pass
