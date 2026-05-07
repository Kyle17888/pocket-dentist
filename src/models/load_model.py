from argparse import Namespace
from typing import Any, Dict, Optional

from src.models.api_model import APIModel
from src.models.base_model import BaseModel
from src.utils.common_utils import strip_trailing_slash


def load_model(args: Namespace, model_cfg: Optional[Dict[str, Any]]) -> BaseModel:
    if args.task == "generation" or args.task == "prediction":
        if args.client_type == "local":
            if args.served_model_name == "baichuan-inc/Baichuan-Omni-1d5":
                from src.models.local_model import BaichuanOmni1d5Model
                model = BaichuanOmni1d5Model(strip_trailing_slash(model_cfg["model_dir"]), args.temperature, args.do_sample, args.max_new_tokens)
            else:
                raise ValueError(f"Unsupported model: {args.served_model_name}")
        else:
            model = APIModel(args.served_model_name, args.temperature, args.api_base_url, args.api_key, max_tokens=args.max_new_tokens)
    
    elif args.task == "evaluation":
        if args.client_type == "local":
            raise ValueError(f"Unsupported local client for evaluation task.")
        else:
            model = APIModel(args.served_model_name, args.temperature, args.api_base_url, args.api_key, max_tokens=args.max_new_tokens)
    return model