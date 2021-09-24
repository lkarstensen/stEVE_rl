import logging
from typing import List, Tuple

from .agent import Agent, StepCounterShared, EpisodeCounterShared
from .single import EpisodeCounter, StepCounter, Algo, ReplayBuffer
from .singelagentprocess import SingleAgentProcess
from ..environment import EnvFactory, DummyEnvFactory
from math import ceil, inf
import numpy as np
import torch


class Synchron(Agent):
    def __init__(
        self,
        n_worker: int,
        n_trainer: int,
        algo: Algo,
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        share_trainer_model=False,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.n_trainer = n_trainer
        self.share_trainer_model = share_trainer_model
        self.worker: List[SingleAgentProcess] = []
        self.trainer: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer
        self._step_counter = StepCounterShared()
        self._episode_counter = EpisodeCounterShared()
        self._step_counter_set_point = StepCounter()
        self._episode_counter_set_point = EpisodeCounter()

        for i in range(n_worker):
            self.worker.append(
                SingleAgentProcess(
                    i,
                    algo.copy(),
                    env_factory,
                    replay_buffer.copy(),
                    worker_device,
                    consecutive_action_steps,
                    name="worker_" + str(i),
                    parent_agent=self,
                    step_counter=self._step_counter,
                    episode_counter=self._episode_counter,
                )
            )

        for i in range(n_trainer):
            if share_trainer_model:
                new_algo = algo.copy_shared_memory()
            else:
                new_algo = algo.copy()
            self.trainer.append(
                SingleAgentProcess(
                    i,
                    new_algo,
                    DummyEnvFactory(),
                    replay_buffer.copy(),
                    trainer_device,
                    0,
                    name="trainer_" + str(i),
                    parent_agent=self,
                    step_counter=self._step_counter,
                    episode_counter=self._episode_counter,
                )
            )
        self.logger.debug("Synchron Agent initialized")

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        if steps < inf:
            self._step_counter_set_point.heatup += steps
            steps = self._step_counter_set_point.heatup - self._step_counter.heatup
        if episodes < inf:
            self._episode_counter_set_point.heatup += episodes
            episodes = self._episode_counter_set_point.heatup - self._episode_counter.heatup

        self.logger.debug(f"heatup: {steps} steps / {episodes} episodes")

        for agent in self.worker:
            agent.heatup(steps, episodes)
        result = self._get_worker_results()
        return result

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        if steps < inf:
            self._step_counter_set_point.exploration += steps
            steps = self._step_counter_set_point.exploration - self._step_counter.exploration
        if episodes < inf:
            self._episode_counter_set_point.exploration += episodes
            episodes = (
                self._episode_counter_set_point.exploration - self._episode_counter.exploration
            )
        self.logger.debug(f"explore: {steps} steps / {episodes} episodes")

        for agent in self.worker:
            agent.explore(steps, episodes)
        result = self._get_worker_results()
        return result

    def update(self, steps) -> List[float]:
        self._step_counter_set_point.update += steps
        steps = self._step_counter_set_point.update - self._step_counter.update
        self.logger.debug(f"update: {steps} steps")
        for agent in self.trainer:
            agent.update(steps)

        result = self._get_trainer_results()

        if self.share_trainer_model:
            self.trainer[0].put_state_dict()
            new_state_dict = self.trainer[0].get_state_dict()
        else:
            for agent in self.trainer:
                agent.put_state_dict()
            new_state_dict = None
            for agent in self.trainer:
                state_dicts = agent.get_state_dict() / self.n_trainer
                if new_state_dict is None:
                    new_state_dict = state_dicts
                else:
                    new_state_dict += state_dicts

        for agent in self.worker:
            agent.set_state_dict(new_state_dict)
        if not self.share_trainer_model:
            for agent in self.trainer:
                agent.set_state_dict(new_state_dict)
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        if steps < inf:
            self._step_counter_set_point.evaluation += steps
            steps = self._step_counter_set_point.evaluation - self._step_counter.evaluation
        if episodes < inf:
            self._episode_counter_set_point.evaluation += episodes
            episodes = self._episode_counter_set_point.evaluation - self._episode_counter.evaluation
        self.logger.debug(f"evaluate: {steps} steps / {episodes} episodes")

        for agent in self.worker:
            agent.evaluate(steps, episodes)

        result = self._get_worker_results()
        return result

    def close(self):
        for agent in self.worker + self.trainer:
            agent.close()
        self.replay_buffer.close()

    def _get_worker_results(self):
        successes = []
        rewards = []
        for agent in self.worker:
            reward, success = agent.get_result()
            rewards += reward
            successes += success
        return rewards, successes

    def _get_trainer_results(self):
        results = []
        for agent in self.trainer:
            results.append(agent.get_result())
            n_max = len(max(results, key=len))
            results = [result + [None] * (n_max - len(result)) for result in results]
            results = [
                val for result_tuple in zip(*results) for val in result_tuple if val is not None
            ]
        return results

    @property
    def step_counter(self) -> StepCounterShared:
        return self._step_counter

    @property
    def episode_counter(self) -> EpisodeCounterShared:
        return self._episode_counter
