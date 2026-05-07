import json
import os
from argparse import Namespace
from typing import Any, Dict, Literal

import pandas as pd
from tqdm import tqdm

from src.utils.common_utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))


def load_data(data_path: str) -> Dict[str, Any]:
    """Load any JSON file by path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return json.load(open(data_path, "r", encoding="utf-8"))

def save_json_data(data: Dict[str, Any], save_dir: str, save_file_name: str, title: str = "Saved") -> None:
    save_path = os.path.join(strip_trailing_slash(save_dir), save_file_name)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(data, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        tqdm.write(f"[{title}] Saved data to '{save_path}'")
    except Exception as e:
        tqdm.write(f"[{title}] Failed to save data to '{save_path}': {e}")

def save_csv_data(data: pd.DataFrame, save_dir: str, save_file_name: str, title: str = "Saved") -> None:
    save_path = os.path.join(strip_trailing_slash(save_dir), save_file_name)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)
        tqdm.write(f"[{title}] Saved data to '{save_path}'")
    except Exception as e:
        tqdm.write(f"[{title}] Failed to save data to '{save_path}': {e}")
