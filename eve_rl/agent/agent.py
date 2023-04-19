from abc import ABC, abstractmethod
from math import inf
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import torch.multiprocessing as mp
import gymnasium as gym
import torch

from ..replaybuffer.replaybuffer import Episode
from ..util import EveRLObject
from ..algo import Algo
from ..replaybuffer import ReplayBuffer


@dataclass
class EpisodeCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0
    lock = mp.Lock()

    def __iadd__(self, other):
        self.heatup += other.heatup
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        return self


@dataclass
class StepCounter:
    heatup: int = 0
    exploration: int = 0
    evaluation: int = 0
    update: int = 0
    lock = mp.Lock()

    def __iadd__(self, other):
        self.heatup += other.heatup
        self.exploration += other.exploration
        self.evaluation += other.evaluation
        self.update += other.update
        return self


class StepCounterShared(StepCounter):
    # pylint: disable=super-init-not-called
    def __init__(self):
        self._heatup: mp.Value = mp.Value("i", 0)
        self._exploration: mp.Value = mp.Value("i", 0)
        self._evaluation: mp.Value = mp.Value("i", 0)
        self._update: mp.Value = mp.Value("i", 0)

    @property
    def heatup(self) -> int:
        return self._heatup.value

    @heatup.setter
    def heatup(self, value: int) -> int:
        self._heatup.value = value

    @property
    def exploration(self) -> int:
        return self._exploration.value

    @exploration.setter
    def exploration(self, value: int) -> int:
        self._exploration.value = value

    @property
    def evaluation(self) -> int:
        return self._evaluation.value

    @evaluation.setter
    def evaluation(self, value: int) -> int:
        self._evaluation.value = value

    @property
    def update(self) -> int:
        return self._update.value

    @update.setter
    def update(self, value: int) -> int:
        self._update.value = value

    def __iadd__(self, other):
        self._heatup.value = self._heatup.value + other.heatup
        self._exploration.value = self._exploration.value + other.exploration
        self._evaluation.value = self._evaluation.value + other.evaluation
        self._update.value = self._update.value + other.update
        return self


class EpisodeCounterShared(EpisodeCounter):
    # pylint: disable=super-init-not-called
    def __init__(self):
        self._heatup: mp.Value = mp.Value("i", 0)
        self._exploration: mp.Value = mp.Value("i", 0)
        self._evaluation: mp.Value = mp.Value("i", 0)

    @property
    def heatup(self) -> int:
        return self._heatup.value

    @heatup.setter
    def heatup(self, value: int) -> int:
        self._heatup.value = value

    @property
    def exploration(self) -> int:
        return self._exploration.value

    @exploration.setter
    def exploration(self, value: int) -> int:
        self._exploration.value = value

    @property
    def evaluation(self) -> int:
        return self._evaluation.value

    @evaluation.setter
    def evaluation(self, value: int) -> int:
        self._evaluation.value = value

    def __iadd__(self, other):
        self._heatup.value = self._heatup.value + other.heatup
        self._exploration.value = self._exploration.value + other.exploration
        self._evaluation.value = self._evaluation.value + other.evaluation
        return self


class Agent(EveRLObject, ABC):
    step_counter: StepCounter
    episode_counter: EpisodeCounter
    algo: Algo
    env_train: gym.Env
    env_eval: gym.Env
    replay_buffer: ReplayBuffer
    logger: logging.Logger
    update_error: bool

    @abstractmethod
    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ) -> List[Episode]:
        ...

    @abstractmethod
    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        ...

    @abstractmethod
    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[List[float]]:
        ...

    @abstractmethod
    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Episode]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def save_checkpoint(self, file_path) -> None:
        algo_config = self.algo.get_config_dict()
        replay_config = self.replay_buffer.get_config_dict()
        checkpoint_dict = {
            "algo": {
                "network": self.algo.state_dicts_network(),
                "config": algo_config,
            },
            "replay_buffer": {"config": replay_config},
            "steps": {
                "heatup": self.step_counter.heatup,
                "exploration": self.step_counter.exploration,
                "update": self.step_counter.update,
                "evaluation": self.step_counter.evaluation,
            },
            "episodes": {
                "heatup": self.episode_counter.heatup,
                "exploration": self.episode_counter.exploration,
                "evaluation": self.episode_counter.evaluation,
            },
        }

        torch.save(checkpoint_dict, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = torch.load(file_path)

        state_dicts_network = checkpoint["algo"]["network"]
        self.algo.load_state_dicts_network(state_dicts_network)

        self.step_counter.heatup = checkpoint["steps"]["heatup"]
        self.step_counter.exploration = checkpoint["steps"]["exploration"]
        self.step_counter.evaluation = checkpoint["steps"]["evaluation"]
        self.step_counter.update = checkpoint["steps"]["update"]

        self.episode_counter.heatup = checkpoint["episodes"]["heatup"]
        self.episode_counter.exploration = checkpoint["episodes"]["exploration"]
        self.episode_counter.evaluation = checkpoint["episodes"]["evaluation"]

    def _log_task(
        self,
        task: str,
        steps: Optional[int] = None,
        step_limit: Optional[int] = None,
        episodes: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ):
        if task == "update":
            log_text = f"update (amount/limit): steps {steps}/{step_limit}"
        elif task == "explore":
            log_text = f"explore (amount/limit): steps {steps}/{step_limit} | episodes {episodes}/{episode_limit}"
        elif task == "evaluate":
            seed_text = f"{len(seeds)=}" if seeds is not None else "seeds=None"
            options_text = f"{len(options)=}" if options is not None else "options=None"
            log_text = f"evaluate (amount/limit): steps {steps}/{step_limit} | episodes {episodes}/{episode_limit} | {seed_text}/{options_text}"
        elif task == "heatup":
            log_text = f"heatup (amount/limit): steps {steps}/{step_limit} | episodes {episodes}/{episode_limit} | {custom_action_low=}/{custom_action_high=}"
        else:
            raise ValueError(f"{task=} is not possible")
        self.logger.debug(log_text)

    def _log_task_completion(
        self, task: str, steps: int, t_duration: float, episodes: Optional[int] = None
    ):
        current_steps = getattr(self.step_counter, task)
        if task == "update":
            log_text = f"{task:<11}: {t_duration:>6.1f}s | {steps/t_duration:>5.1f} steps/s | {steps:>7} steps | Total: {current_steps:>8} steps"
        else:
            current_episodes = getattr(self.episode_counter, task)
            log_text = f"{task:<11}: {t_duration:>6.1f}s | {steps/t_duration:>5.1f} steps/s | {steps:>7} steps / {episodes:>4} episodes | Total: {current_steps:>8} steps / {current_episodes:>5} episodes"
        self.logger.info(log_text)

    def _log_and_convert_limits(
        self,
        task: str,
        steps: Optional[int] = None,
        step_limit: Optional[int] = None,
        episodes: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, int]:
        steps = int(steps) if steps not in [None, inf] else steps
        episodes = int(episodes) if episodes not in [None, inf] else episodes
        step_limit = int(step_limit) if step_limit not in [None, inf] else step_limit
        episode_limit = (
            int(episode_limit) if episode_limit not in [None, inf] else episode_limit
        )

        if (
            steps is None
            and episodes is None
            and step_limit is None
            and episode_limit is None
            and seeds is None
            and options is None
        ):
            raise ValueError(
                f"{steps=}, {episodes=}, {step_limit=}, {episode_limit=}, {seeds=} and {options=} for {task}. At least one must be given."
            )
        steps = steps if steps is not None else inf
        episodes = episodes if episodes is not None else inf
        step_limit = step_limit if step_limit is not None else inf
        episode_limit = episode_limit if episode_limit is not None else inf

        if steps < 0 or step_limit < 0 or episodes < 0 or episode_limit < 0:
            raise ValueError(
                f"{steps=}, {episodes=}, {step_limit=} and {episode_limit=} for {task} must be positive integers."
            )

        current_steps = getattr(self.step_counter, task)
        step_limit = min(step_limit, current_steps + steps)

        if task != "update":
            current_episodes = getattr(self.episode_counter, task)
            episode_limit = min(episode_limit, current_episodes + episodes)

        if seeds is not None and options is not None:
            if len(seeds) != len(options):
                raise ValueError(
                    f"if seeds and options are given, they must be the same length. {len(seeds)=}, {len(options)=}"
                )
        return step_limit, episode_limit