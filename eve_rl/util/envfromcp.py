from typing import Dict
import torch
from eve.util.eveobject import EveObject
import eve


def get_env_from_checkpoint(
    checkpoint_path: str, mode: str = "eval", to_exchange: Dict[str, EveObject] = None
) -> eve.Env:
    cp = torch.load(checkpoint_path)
    env_dict = cp[f"env_{mode}"]
    return eve.Env.from_config_dict(env_dict, to_exchange)
