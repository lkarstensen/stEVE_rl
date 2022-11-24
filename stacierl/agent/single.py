import logging
from typing import Callable, List, Tuple

from stacierl.replaybuffer.replaybuffer import Episode
from .agent import Agent, StepCounter, EpisodeCounter
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from eve import Env
import torch
from math import inf

import numpy as np
import os


class Single(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: Env,
        env_eval: Env,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_action: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_action = normalize_action

        self._step_counter = StepCounter()
        self._episode_counter = EpisodeCounter()
        self.to(device)

    def heatup(
        self,
        steps: int = inf,
        episodes: int = inf,
        custom_action_low: List[float] = None,
        custom_action_high: List[float] = None,
    ) -> List[Episode]:

        step_limit = self.step_counter.heatup + steps
        episode_limit = self.episode_counter.heatup + episodes
        episodes_data = []

        def random_action(*args, **kwargs):
            env_low = self.env_train.action_space.low.reshape(-1)
            env_high = self.env_train.action_space.high.reshape(-1)

            if custom_action_low is not None:
                action_low = np.array(custom_action_low).reshape(-1)
            else:
                action_low = env_low.reshape(-1)
            if custom_action_high is not None:
                action_high = np.array(custom_action_high).reshape(-1)
            else:
                action_high = env_high.reshape(-1)
            action = np.random.uniform(action_low, action_high)

            if self.normalize_action:
                action = 2 * (action - env_low) / (env_high - env_low) - 1

            return action

        while True:
            with self.episode_counter.lock:
                self.episode_counter.heatup += 1

            episode, step_counter = self._play_episode(
                env=self.env_train,
                action_function=random_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.heatup += step_counter

            self.logger.info(f"Heatup Steps: {int(self.step_counter.heatup)}")
            self.replay_buffer.push(episode)
            episodes_data.append(episode)
            if (
                self.step_counter.heatup > step_limit
                or self.episode_counter.heatup > episode_limit
            ):
                break
        return episodes_data

    def explore(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        step_limit = self.step_counter.exploration + steps
        episode_limit = self.episode_counter.exploration + episodes
        episodes_data = []

        while True:
            with self.episode_counter.lock:
                self.episode_counter.exploration += 1

            episode, step_counter = self._play_episode(
                env=self.env_train,
                action_function=self.algo.get_exploration_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.exploration += step_counter

            self.logger.info(f"Explore Steps: {self.step_counter.exploration}")
            self.replay_buffer.push(episode)
            episodes_data.append(episode)
            if (
                self.step_counter.exploration > step_limit
                or self.episode_counter.exploration > episode_limit
            ):
                break
        return episodes_data

    def update(self, steps) -> List[List[float]]:
        step_limit = self.step_counter.update + steps
        results = []
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return results

        while self.step_counter.update < step_limit:
            with self.step_counter.lock:
                self.step_counter.update += 1
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            results.append(result)
        self.logger.info(f"Update Steps: {self.step_counter.update}")

        return results

    def evaluate(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        step_limit = self.step_counter.evaluation + steps
        episode_limit = self.episode_counter.evaluation + episodes
        episodes_data = []

        while True:
            with self.episode_counter.lock:
                self.episode_counter.evaluation += 1

            episode, step_counter = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_eval_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            with self.step_counter.lock:
                self.step_counter.evaluation += step_counter

            self.logger.info(f"Eval Episodes: {self.episode_counter.evaluation}")
            episodes_data.append(episode)
            if (
                self.step_counter.evaluation > step_limit
                or self.episode_counter.evaluation > episode_limit
            ):
                break
        return episodes_data

    def _play_episode(
        self,
        env: Env,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
    ) -> Tuple[Episode, int]:
        done = False
        step_counter = 0

        self.algo.reset()
        state = env.reset()
        flat_state = env.observation_space.to_flat_array(state)
        episode = Episode(state, flat_state)

        while done == False:
            action = action_function(flat_state)

            for _ in range(consecutive_actions):
                env_action = action.reshape(env.action_space.low.shape)
                if self.normalize_action:
                    env_action = (env_action + 1) / 2 * (
                        env.action_space.high - env.action_space.low
                    ) + env.action_space.low
                state, reward, done, info, success = env.step(env_action)
                flat_state = env.observation_space.to_flat_array(state)
                step_counter += 1
                env.render()
                episode.add_transition(
                    state, flat_state, action, reward, done, info, success
                )
                if done:
                    break

        return episode, step_counter

    def to(self, device: torch.device):
        self.device = device
        self.algo.to(device)

    def close(self):
        self.env_train.close()
        self.env_eval.close()

    def save_checkpoint(self, directory: str, name: str) -> None:
        path = directory + "/" + name + ".pt"

        optimizer_state_dicts = self.algo.model.optimizer_states_container.to_dict()
        network_state_dicts = self.algo.model.network_states_container.to_dict()

        checkpoint_dict = {
            "optimizer_state_dicts": optimizer_state_dicts,
            "network_state_dicts": network_state_dicts,
            "heatup_steps": self.step_counter.heatup,
            "exploration_steps": self.step_counter.exploration,
            "update_steps": self.step_counter.update,
            "evaluation_steps": self.step_counter.evaluation,
        }

        torch.save(checkpoint_dict, path)

    def load_checkpoint(self, directory: str, name: str) -> None:
        name, _ = os.path.splitext(name)
        path = os.path.join(directory, name + ".pt")
        checkpoint = torch.load(path, map_location=self.device)

        network_states_container = self.algo.model.network_states_container
        network_states_container.from_dict(checkpoint["network_state_dicts"])

        optimizer_states_container = self.algo.model.optimizer_states_container
        optimizer_states_container.from_dict(checkpoint["optimizer_state_dicts"])

        self.algo.model.set_network_states(network_states_container)
        self.algo.model.set_optimizer_states(optimizer_states_container)

        self.step_counter = StepCounter(
            checkpoint["heatup_steps"],
            checkpoint["exploration_steps"],
            checkpoint["evaluation_steps"],
            checkpoint["update_steps"],
        )

    @property
    def step_counter(self) -> StepCounter:
        return self._step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self._episode_counter

    @step_counter.setter
    def step_counter(self, new_counter: StepCounter) -> None:
        self._step_counter = new_counter

    @episode_counter.setter
    def episode_counter(self, new_counter: EpisodeCounter) -> None:
        self._episode_counter = new_counter

    def copy(self):
        ...
