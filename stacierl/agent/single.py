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

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps, episodes = self._check_steps_and_episodes(steps, episodes)
        step_counter = 0
        episode_counter = 0
        rewards = []
        successes = []
        while step_counter < steps and episode_counter < episodes:
            (
                episode_transitions,
                explored_steps,
                episode_reward,
                episode_success,
            ) = self._play_episode(
                consecutive_actions=self.consecutive_action_steps, mode="exploration"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            step_counter += explored_steps
            episode_counter += 1
            self.replay_buffer.push(episode_transitions)

        average_reward = (sum(rewards) / len(rewards)) if rewards else None
        average_success = (sum(successes) / len(successes)) if successes else None

        return average_reward, average_success

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps, episodes = self._check_steps_and_episodes(steps, episodes)

        step_limit = self.step_counter.exploration + steps
        episode_limit = self.episode_counter.exploration + episodes
        rewards = []
        successes = []
        while (
            self.step_counter.exploration < step_limit
            and self.episode_counter.exploration < episode_limit
        ):
            (
                episode_transitions,
                explored_steps,
                episode_reward,
                episode_success,
            ) = self._play_episode(
                consecutive_actions=self.consecutive_action_steps, mode="exploration"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            self.step_counter.exploration += explored_steps
            self.episode_counter.exploration += 1
            self.replay_buffer.push(episode_transitions)

        average_reward = sum(rewards) / len(rewards)
        average_success = sum(successes) / len(successes)

        return average_reward, average_success

    def update(self, steps) -> List[float]:
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        for _ in range(steps):
            batch = self.replay_buffer.sample()
            result = self.algo.update(batch)
            self.step_counter.update += 1

        return result

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps, episodes = self._check_steps_and_episodes(steps, episodes)
        step_limit = self.step_counter.eval + steps
        episode_limit = self.episode_counter.eval + episodes
        rewards = []
        successes = []
        while self.step_counter.eval < step_limit and self.episode_counter.eval < episode_limit:
            _, steps, episode_reward, episode_success = self._play_episode(
                consecutive_actions=1, mode="eval"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            self.step_counter.eval += steps
            self.episode_counter.eval += 1

        average_reward = sum(rewards) / len(rewards)
        average_success = sum(successes) / len(successes)

        return average_reward, average_success

    def _play_episode(self, consecutive_actions: int, mode: str):
        assert mode in ["exploration", "eval"]
        episode_transitions = Episode()
        self.algo.reset()
        state = self.env.reset()
        self.logger.debug(f"Reset state:\n{state}")
        state = self.env.observation_space.to_flat_array(state)
        episode_reward = 0
        step_counter = 0
        while True:
            if mode == "exploration":
                action = self.algo.get_exploration_action(state)
            else:
                action = self.algo.get_eval_action(state)
            self.logger.debug(f"Action: {action}")
            for _ in range(consecutive_actions):
                next_state, reward, done, info, success = self.env.step(action)
                self.logger.debug(f"Next state:\n{next_state}")
                next_state = self.env.observation_space.to_flat_array(next_state)
                self.env.render()
                episode_transitions.add_transition(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step_counter += 1
                if done:
                    break
            if done:
                break
        return episode_transitions, step_counter, episode_reward, success

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

    @staticmethod
    def _check_steps_and_episodes(steps: int = None, episodes: int = None) -> Tuple[int, int]:
        if steps is None and episodes is None:
            raise ValueError("One of the two (steps or episodes) needs to be given")
        steps = inf if steps is None else steps
        episodes = inf if episodes is None else episodes
        return steps, episodes
