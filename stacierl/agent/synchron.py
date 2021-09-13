from typing import List, Tuple

from .agent import Agent, dataclass
from .single import Single, EpisodeCounter, StepCounter, Algo, ReplayBuffer, Environment
from .parallel import SingleAgentProcess
from ..environment import EnvFactory
from torch import multiprocessing as mp
from math import ceil
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

        self.n_worker = n_worker
        self.n_trainer = n_trainer
        self.share_trainer_model = share_trainer_model
        self.worker: List[SingleAgentProcess] = []
        self.trainer: List[SingleAgentProcess] = []

        for i in range(n_worker):
            self.worker.append(
                SingleAgentProcess(
                    i,
                    algo.copy(),
                    env_factory,
                    replay_buffer.copy(),
                    worker_device,
                    consecutive_action_steps,
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
                    env_factory,
                    replay_buffer.copy(),
                    trainer_device,
                    consecutive_action_steps,
                )
            )

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.heatup(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.worker:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.explore(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.worker:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def update(self, steps, batch_size):
        steps_per_agent = ceil(steps / self.n_trainer)
        for agent in self.trainer:
            agent.update(steps_per_agent, batch_size)

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

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.evaluate(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.worker:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def close(self):
        for agent in self.worker:
            agent.close()
        for agent in self.trainer:
            agent.close()

    def _divide_steps_and_episodes(self, steps, episodes, n_agents) -> Tuple[int, int]:

        steps = ceil(steps / n_agents) if steps is not None else None

        episodes = ceil(episodes / n_agents) if episodes is not None else None

        return steps, episodes

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
