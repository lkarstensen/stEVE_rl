import logging
from typing import List, Tuple

from .agent import Agent
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
                )
            )

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        self.logger.debug(f"heatup: {steps} steps / {episodes} episodes")
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.heatup(steps_per_agent, episodes_per_agent)
        result = self._get_worker_results()
        return tuple(result)

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        self.logger.debug(f"explore: {steps} steps / {episodes} episodes")
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.explore(steps_per_agent, episodes_per_agent)
        result = self._get_worker_results()
        return tuple(result)

    def update(self, steps):

        self.logger.debug(f"update: {steps} steps")
        steps_per_agent = ceil(steps / self.n_trainer)
        for agent in self.trainer:
            agent.update(steps_per_agent)

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
        result = list(result) if result is not None else result
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:

        self.logger.debug(f"evaluate: {steps} steps / {episodes} episodes")
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.evaluate(steps_per_agent, episodes_per_agent)

        result = self._get_worker_results()
        return tuple(result)

    def close(self):
        for agent in self.worker + self.trainer:
            agent.close()
        self.replay_buffer.close()

    def _divide_steps_and_episodes(self, steps, episodes, n_agents) -> Tuple[int, int]:
        steps = ceil(steps / n_agents) if steps != inf else inf
        episodes = ceil(episodes / n_agents) if episodes != inf else inf
        return steps, episodes

    def _get_worker_results(self):
        results = []
        for agent in self.worker:
            result = agent.get_result()
            if result is not None and not None in result:
                results.append(result)
        results = np.mean(np.array(results), axis=0) if results else result
        return results

    def _get_trainer_results(self):
        results = []
        for agent in self.trainer:
            result = agent.get_result()
            if result is not None and not None in result:
                results.append(result)
        results = np.mean(np.array(results), axis=0) if results else result
        return results

    @property
    def step_counter(self) -> StepCounter:
        step_counter = StepCounter()
        for agent in self.worker:
            step_counter += agent.step_counter
        for agent in self.trainer:
            step_counter += agent.step_counter
        return step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        episode_counter = EpisodeCounter()
        for agent in self.worker:
            episode_counter += agent.episode_counter
        for agent in self.trainer:
            episode_counter += agent.episode_counter

        return episode_counter
