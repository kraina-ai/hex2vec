from typing import Dict, Union
from keplergl import KeplerGl
from pathlib import Path
import pickle as pkl
import json
from src.settings import KEPLER_CONFIG_DIR


def save_config(kepler: KeplerGl, config_name: str) -> Path:
    with open(KEPLER_CONFIG_DIR.joinpath(f"{config_name}.json"), "wt") as f:
        json.dump(kepler.config, f)
    return


def load_config(config_name: str) -> Union[Dict, None]:
    try:
        with open(KEPLER_CONFIG_DIR.joinpath(f"{config_name}.json"), "rt") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
