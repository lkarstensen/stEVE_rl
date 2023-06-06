from typing import Dict, List, Tuple, Union
import numpy as np


def flatten_obs(
    obs: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, Union[Tuple, List, Dict]]:
    if isinstance(obs, np.ndarray):
        return obs.flatten(), obs.shape
    if isinstance(obs, list):
        obs_flat_list = []
        flat_obs_to_obs = []
        idx = 0
        for obs_entry in obs:
            shape = obs_entry.shape
            obs_flat = obs_entry.flatten()
            obs_flat_list.append(obs_flat)
            flat_obs_to_obs.append((shape, (idx, idx + obs_flat.size)))
            idx += obs_flat.size
        obs_flat_np = np.concatenate(obs_flat_list)
        return obs_flat_np, flat_obs_to_obs
    if isinstance(obs, Dict):
        obs_flat_list = []
        flat_obs_to_obs = {}
        current_idx = 0
        for name, obs_entry in obs.items():
            shape = obs_entry.shape
            obs_flat = obs_entry.flatten()
            obs_flat_list.append(obs_flat)
            flat_obs_to_obs[name] = (shape, (current_idx, current_idx + obs_flat.size))
            current_idx += obs_flat.size
        obs_flat_np = np.concatenate(obs_flat_list)
        return obs_flat_np, flat_obs_to_obs

    raise ValueError("Wrong Observation Type")
