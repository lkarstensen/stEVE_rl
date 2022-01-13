import logging
from typing import List, Tuple
from .agent import Agent, StepCounter, EpisodeCounter
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from ..util import Environment
import torch
from math import inf

# added because of heatup
from torch.distributions import Normal
import numpy as np


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

        self._episode_transitions = Episode()
        self._episode_reward = 0.0
        self._last_play_mode = None
        self._state = None

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        # random action selection
        action_dim = 2
        
        step_limit = self._step_counter.heatup + steps
        episode_limit = self._episode_counter.heatup + episodes
        episode_rewards = []
        successes = []

        if self._last_play_mode != "exploration" or self._state is None:
            self._reset_env()
            self._last_play_mode = "exploration"

        while (
            self._step_counter.heatup < step_limit and self._episode_counter.heatup < episode_limit
        ):
            self._step_counter.heatup += self.consecutive_action_steps
            #action = self.algo.get_exploration_action(self._state)
            
            # random action selection
            with torch.no_grad():
                normal = Normal(0,1)
                action = (normal.sample((action_dim,))).numpy()
                #action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                action = action + np.random.normal(0, 0.25)
        
            done, success = self._play_step(
                action, consecutive_actions=self.consecutive_action_steps
            )
            if done:
                self.episode_counter.heatup += 1
                successes.append(success)
                episode_rewards.append(self._episode_reward)
                self._reset_env()

        return episode_rewards, successes

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        step_limit = self._step_counter.exploration + steps
        episode_limit = self._episode_counter.exploration + episodes
        episode_rewards = []
        successes = []

        if self._last_play_mode != "exploration" or self._state is None:
            self._reset_env()
            self._last_play_mode = "exploration"

        while (
            self._step_counter.exploration < step_limit
            and self._episode_counter.exploration < episode_limit
        ):
            self._step_counter.exploration += self.consecutive_action_steps
            action = self.algo.get_exploration_action(self._state)
            done, success = self._play_step(
                action, consecutive_actions=self.consecutive_action_steps
            )

            if done:
                self.episode_counter.exploration += 1
                successes.append(success)
                episode_rewards.append(self._episode_reward)
                self._reset_env()

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

        if self._last_play_mode != "evaluation" or self._state is None:
            self._reset_env()
            self._last_play_mode = "evaluation"

        while (
            self._step_counter.evaluation < step_limit
            and self._episode_counter.evaluation < episode_limit
        ):
            self._step_counter.evaluation += 1
            action = self.algo.get_eval_action(self._state)
            done, success = self._play_step(action, consecutive_actions=1)

            if done:
                self._episode_counter.evaluation += 1
                successes.append(success)
                episode_rewards.append(self._episode_reward)
                self._reset_env()

        return episode_rewards, successes

    def _play_step(self, action, consecutive_actions: int):

        self.logger.debug(f"Action: {action}")
        for _ in range(consecutive_actions):
            state, reward, done, _, success = self.env.step(action)
            
            #if self._last_play_mode == "exploration":
            #    self.replay_buffer.push(self._state, action, reward, self.env.observation_space.to_flat_array(state), done)
            
            self._state = self.env.observation_space.to_flat_array(state)
            self.env.render()
            self._episode_transitions.add_transition(self._state, action, reward, done)
            self._episode_reward += reward
            if done:
                break
        return done, success

    def _reset_env(self):
        if self._last_play_mode == "exploration":
            self.replay_buffer.push(self._episode_transitions)
        self._episode_transitions = Episode()
        self._episode_reward = 0.0
        self.algo.reset()
        state = self.env.reset()
        self._state = self.env.observation_space.to_flat_array(state)
        self._episode_transitions.add_reset_state(self._state)

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

    @step_counter.setter
    def step_counter(self, new_counter: StepCounter) -> StepCounter:
        self._step_counter = new_counter

    @episode_counter.setter
    def episode_counter(self, new_counter: EpisodeCounter) -> EpisodeCounter:
        self._episode_counter = new_counter
