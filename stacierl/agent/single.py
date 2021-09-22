import logging
from typing import List, Tuple
from .agent import Agent, StepCounter, EpisodeCounter
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from ..environment import Environment
import torch
from math import inf


class Single(Agent):
    def __init__(
        self,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.device = device
        self.algo = algo
        self.env = env
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps

        self._step_counter = StepCounter()
        self._episode_counter = EpisodeCounter()
        self.to(device)
        self._episode_transitions = None
        self._episode_reward = None
        self._last_play_mode = None
        self._state = None

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        (explored_steps, explored_episodes, rewards, successes,) = self._play(
            steps, episodes, consecutive_actions=self.consecutive_action_steps, mode="exploration"
        )
        self._step_counter.heatup += explored_steps
        self._episode_counter.heatup += explored_episodes

        average_reward = (sum(rewards) / len(rewards)) if rewards else None
        average_success = (sum(successes) / len(successes)) if successes else None

        return average_reward, average_success

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        (explored_steps, explored_episodes, rewards, successes,) = self._play(
            steps, episodes, consecutive_actions=self.consecutive_action_steps, mode="exploration"
        )
        self.step_counter.exploration += explored_steps
        self.episode_counter.exploration += explored_episodes

        average_reward = (sum(rewards) / len(rewards)) if rewards else None
        average_success = (sum(successes) / len(successes)) if successes else None

        return average_reward, average_success

    def update(self, steps) -> List[float]:
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        for _ in range(steps):
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            self.step_counter.update += 1

        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        (steps, episodes, rewards, successes,) = self._play(
            steps, episodes, consecutive_actions=self.consecutive_action_steps, mode="exploration"
        )

        self.step_counter.evaluation += steps
        self.episode_counter.evaluation += episodes
        average_reward = (sum(rewards) / len(rewards)) if rewards else None
        average_success = (sum(successes) / len(successes)) if successes else None

        return average_reward, average_success

    def _play(self, steps: int, episodes: int, consecutive_actions: int, mode: str):
        assert mode in ["exploration", "eval"]
        if steps == inf and episodes == inf:
            raise ValueError(
                "One of the two (steps or episodes) needs to be given and may not be inf"
            )
        step_counter = 0
        episode_counter = 0
        episode_rewards = []
        successes = []

        if mode != self._last_play_mode or self._state is None:
            self._episode_transitions = Episode()
            self._episode_reward = 0.0
            state = self.env.reset()
            self._state = self.env.observation_space.to_flat_array(state)
            self.algo.reset()
            self._last_play_mode = mode

        while step_counter < steps and episode_counter < episodes:
            if mode == "exploration":
                action = self.algo.get_exploration_action(self._state)
            else:
                action = self.algo.get_eval_action(self._state)
            self.logger.debug(f"Action: {action}")
            for _ in range(consecutive_actions):
                next_state, reward, done, _, success = self.env.step(action)
                self.logger.debug(f"Next state:\n{next_state}")
                next_state = self.env.observation_space.to_flat_array(next_state)
                self.env.render()
                self._episode_transitions.add_transition(
                    self._state, action, reward, next_state, done
                )
                self._state = next_state
                self._episode_reward += reward
                step_counter += 1
                if done or step_counter >= steps:
                    break
            if done:
                if mode == "exploration":
                    self.replay_buffer.push(self._episode_transitions)
                self._episode_transitions = Episode()
                episode_rewards.append(self._episode_reward)
                self._episode_reward = 0.0
                successes.append(success)
                episode_counter += 1
                state = self.env.reset()
                self._state = self.env.observation_space.to_flat_array(state)
                self.algo.reset()
        return step_counter, episode_counter, episode_rewards, successes

    def to(self, device: torch.device):
        self.device = device
        self.algo.to(device)

    def close(self):
        self.env.close()

    @property
    def step_counter(self) -> StepCounter:
        return self._step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self._episode_counter
