import logging
from typing import Callable, List, Tuple
from .agent import Agent, StepCounter, EpisodeCounter
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from ..util import Environment
import torch
from math import inf

import numpy as np
import os


class Single(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: Environment,
        env_eval: Environment,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps

        self._step_counter = StepCounter()
        self._episode_counter = EpisodeCounter()
        self.to(device)

    def heatup(
        self, steps: int = inf, episodes: int = inf, custom_action_low: List[float] = None
    ) -> Tuple[List[float], List[float]]:

        step_limit = self._step_counter.heatup + steps
        episode_limit = self._episode_counter.heatup + episodes
        episode_rewards = []
        successes = []

        def random_action(*args, **kwargs):
            if custom_action_low is not None:
                action_low = custom_action_low
            else:
                action_low = self.env_train.action_space.low
            action_high = self.env_train.action_space.high
            action = np.random.uniform(action_low, action_high)
            return action

        while (
            self._step_counter.heatup < step_limit and self._episode_counter.heatup < episode_limit
        ):
            episode_transitions, episode_reward, step_counter, success = self._play_episode(
                env=self.env_eval,
                action_function=random_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            self._step_counter.heatup += step_counter
            self._episode_counter.heatup += 1
            self.replay_buffer.push(episode_transitions)
            successes.append(success)
            episode_rewards.append(episode_reward)

        return episode_rewards, successes

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        step_limit = self._step_counter.exploration + steps
        episode_limit = self._episode_counter.exploration + episodes
        episode_rewards = []
        successes = []

        while (
            self._step_counter.exploration < step_limit
            and self._episode_counter.exploration < episode_limit
        ):
            episode_transitions, episode_reward, step_counter, success = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_exploration_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            self._step_counter.exploration += step_counter
            self._episode_counter.exploration += 1
            self.replay_buffer.push(episode_transitions)
            successes.append(success)
            episode_rewards.append(episode_reward)

        return episode_rewards, successes

    def update(self, steps) -> List[List[float]]:
        step_limit = self._step_counter.update + steps
        results = []
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return results

        while self._step_counter.update < step_limit:
            self._step_counter.update += 1
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            results.append(result)

        return results

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        step_limit = self._step_counter.evaluation + steps
        episode_limit = self._episode_counter.evaluation + episodes
        episode_rewards = []
        successes = []

        while (
            self._step_counter.evaluation < step_limit
            and self._episode_counter.evaluation < episode_limit
        ):
            _, episode_reward, step_counter, success = self._play_episode(
                env=self.env_eval,
                action_function=self.algo.get_eval_action,
                consecutive_actions=self.consecutive_action_steps,
            )

            self._step_counter.evaluation += step_counter
            self._episode_counter.evaluation += 1
            successes.append(success)
            episode_rewards.append(episode_reward)

        return episode_rewards, successes

    def _play_episode(
        self,
        env: Environment,
        action_function: Callable[[np.ndarray], np.ndarray],
        consecutive_actions: int,
    ):
        done = False
        step_counter = 0
        episode_transitions = Episode()
        episode_reward = 0

        self.algo.reset()
        state = env.reset()
        flat_state = env.observation_space.to_flat_array(state)
        episode_transitions.add_reset_state(flat_state)

        while done == False:
            action = action_function(flat_state)
            for _ in range(consecutive_actions):
                state, reward, done, _, success = env.step(action)
                step_counter += 1
                flat_state = env.observation_space.to_flat_array(state)
                env.render()
                episode_transitions.add_transition(flat_state, action, reward, done)
                episode_reward += reward
                if done:
                    break

        return episode_transitions, episode_reward, step_counter, success

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
