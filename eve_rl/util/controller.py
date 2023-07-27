from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

import eve
from ..agent import SingleEvalOnly
from .flattenobs import flatten_obs


class Controller:
    def __init__(
        self,
        checkpoint: str,
        branch_centerlines: List[np.ndarray],
        image_rot_zx: Optional[Tuple[float, float]] = None,
        image_center: Optional[Tuple[float, float, float]] = None,
        field_of_view: Optional[Tuple[float, float]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        image_rot_zx = image_rot_zx or [0, 0]
        image_center = image_center or [0,0,0]

        cp = torch.load(checkpoint)
        env_config = cp["env_eval"]
        device_configs = env_config["intervention"]["devices"]

        devices = [
            eve.intervention.device.Device.from_config_dict(device_config)
            for device_config in device_configs
        ]
        threshold = env_config["intervention"]["target"]["threshold"]
        image_frequency = env_config["intervention"]["fluoroscopy"]["image_frequency"]

        vessel_tree = eve.intervention.vesseltree.VesselTreeDummy(branch_centerlines)

        fluoroscopy = eve.intervention.fluoroscopy.FluoroscopyDummyWithVesselTree(
            vessel_tree=vessel_tree,
            image_frequency=image_frequency,
            image_rot_zx=image_rot_zx,
            image_center=image_center,
            field_of_view=field_of_view,
        )
        target = eve.intervention.target.TargetDummy(
            fluoroscopy=fluoroscopy,
            threshold=threshold,
        )

        intervention = eve.intervention.InterventionDummy(
            vessel_tree=vessel_tree,
            devices=devices,
            fluoroscopy=fluoroscopy,
            target=target,
        )

        to_exchange = {
            eve.intervention.Intervention: intervention,
            eve.pathfinder.Pathfinder: eve.pathfinder.PathfinderDummy(),
            eve.interimtarget.InterimTarget: None,
            eve.info.Info: None,
        }

        self.env: eve.EnvObsInfoOnly = eve.EnvObsInfoOnly.from_config_dict(
            env_config, to_exchange
        )

        self.agent = SingleEvalOnly.from_checkpoint(
            checkpoint, device=device, env_eval=self.env
        )
        self.intervention = intervention
        self.fluoroscopy = fluoroscopy
        self.target = target
        self.last_action = self.env.action_space.sample() * 0.0

    def step(
        self,
        tracking: np.ndarray,
        target: np.ndarray,
        device_lengths_inserted: Optional[List[float]] = None,
    ) -> np.ndarray:
        self._update_tracking_target_lengths(tracking, target, device_lengths_inserted)

        obs, _ = self.env.step(self.last_action)
        self.last_action = self._get_action(obs)
        return self.last_action, obs

    def reset(
        self,
        tracking: np.ndarray,
        target: np.ndarray,
        device_lengths_inserted: Optional[List[float]] = None,
    ) -> np.ndarray:
        self._update_tracking_target_lengths(tracking, target, device_lengths_inserted)
        self.last_action *= 0.0
        obs, _ = self.env.reset()
        self.agent.algo.reset()
        self.last_action = self._get_action(obs)
        return self.last_action, obs

    def _update_tracking_target_lengths(
        self,
        tracking: np.ndarray,
        target: np.ndarray,
        device_lengths_inserted: Optional[List[float]],
    ):
        if tracking.shape[1] == 3:
            self.fluoroscopy.tracking3d = tracking
        elif tracking.shape[1] == 2:
            self.fluoroscopy.tracking2d = tracking
        else:
            raise ValueError("Wrong tracking shape")
        self.intervention.device_lengths_inserted = device_lengths_inserted
        if target.shape[0] == 3:
            self.target.coordinates3d = target
        elif target.shape[0] == 2:
            self.target.coordinates2d = target
        else:
            raise ValueError("Wrong target shape")

    def _get_action(self, obs):
        obs_flat, _ = flatten_obs(obs)
        action = self.agent.algo.get_eval_action(obs_flat)
        action = action.reshape(self.last_action.shape)
        if self.agent.normalize_actions:
            action *= self.intervention.action_space.high
        self.last_action = action
        return action
