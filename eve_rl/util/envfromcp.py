from importlib import import_module
import torch


def get_env_from_checkpoint(checkpoint_path: str, mode: str = "eval"):
    cp = torch.load(checkpoint_path)
    env_dict = cp[f"env_{mode}"]
    eve = import_module("eve")
    return eve.Env.from_config_dict(env_dict)
