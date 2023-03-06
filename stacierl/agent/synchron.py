from copy import deepcopy
from typing import Any, Dict, List, Tuple
from math import inf
import logging
import torch

from eve.env import DummyEnv
from .agent import Agent, Episode, StepCounterShared, EpisodeCounterShared
from .single import Algo, ReplayBuffer, gym
from .singelagentprocess import SingleAgentProcess


class Synchron(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: gym.Env,
        env_eval: gym.Env,
        replay_buffer: ReplayBuffer,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
    ) -> None:

        self.algo = algo
        self.algo.to(torch.device("cpu"))
        self.env_train = env_train
        self.env_eval = env_eval
        self.worker_device = worker_device
        self.trainer_device = trainer_device
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.worker: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer
        self.step_counter = StepCounterShared()
        self.episode_counter = EpisodeCounterShared()

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        self.trainer = self._create_trainer_agent()

        self.logger.debug("Synchron Agent initialized")

    def heatup(
        self,
        steps: int = inf,
        episodes: int = inf,
        custom_action_low: List[float] = None,
        custom_action_high: List[float] = None,
    ) -> List[Episode]:
        log_info = f"heatup: {steps} steps / {episodes} episodes"
        self.logger.info(log_info)
        for agent in self.worker:
            agent.heatup(
                steps,
                episodes,
                custom_action_low,
                custom_action_high,
            )
        result = self._get_worker_results()
        return result

    def explore(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        log_info = f"explore: {steps} steps / {episodes} episodes"
        self.logger.info(log_info)

        for agent in self.worker:
            agent.explore(steps, episodes)
        result = self._get_worker_results()
        return result

    def update(self, steps) -> List[float]:
        log_info = f"update: {steps} steps"
        self.logger.info(log_info)
        self.trainer.update(steps)
        result = self._get_trainer_results()
        self._update_state_dicts_network()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        log_info = f"evaluate: {steps} steps / {episodes} episodes"
        self.logger.info(log_info)

        for agent in self.worker:
            agent.evaluate(steps, episodes)

        result = self._get_worker_results()
        return result

    def explore_and_update_parallel(
        self, update_steps: int, explore_steps: int = inf, explore_episodes: int = inf
    ) -> Tuple[List[Episode], List[float]]:
        log_info = f"explore: {explore_steps} steps / {explore_episodes} episodes, update: {update_steps} steps "
        self.logger.info(log_info)
        self.trainer.update(update_steps)
        for agent in self.worker:
            agent.explore(explore_steps, explore_episodes)
        update_result = self._get_trainer_results()
        explore_result = self._get_worker_results()
        self._update_state_dicts_network()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

        return explore_result, update_result

    def close(self):
        for agent in self.worker:
            agent.close()
        self.trainer.close()
        self.replay_buffer.close()

    def _update_state_dicts_network(self):
        state_dicts = self.algo.state_dicts_network()
        self.trainer.state_dicts_network(state_dicts)

    def _worker_load_state_dicts_network(self, state_dicts: Dict[str, Any]):
        for agent in self.worker:
            agent.load_state_dicts_network(state_dicts)

    def _get_worker_results(self):
        episodes = []
        for i, agent in enumerate(self.worker):
            try:
                episodes += agent.get_result()
            except Exception as exception:  # pylint: disable=broad-exception-caught
                log_warn = (
                    f"Restaring Agent {agent.name} because of Exception {exception}"
                )
                self.logger.warning(log_warn)
                agent.close()
                new_agent = self._create_worker_agent(i)
                new_agent.load_state_dicts_network(self.algo.state_dicts_network())
                self.worker[i] = new_agent
                continue

        return episodes

    def _get_trainer_results(self):
        try:
            results = self.trainer.get_result()
        except Exception as exception:  # pylint: disable=broad-exception-caught
            log_warn = f"Restaring Trainer because of Exception {exception}"
            self.logger.warning(log_warn)
            self.trainer.close()
            self.trainer = self._create_trainer_agent()
            self.trainer.load_state_dicts_network(self.algo.state_dicts_network())
            results = None
        return results

    def _create_worker_agent(self, i):
        return SingleAgentProcess(
            i,
            self.algo.copy(),
            deepcopy(self.env_train),
            deepcopy(self.env_eval),
            self.replay_buffer.copy(),
            self.worker_device,
            self.consecutive_action_steps,
            self.normalize_actions,
            name="worker_" + str(i),
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
        )

    def _create_trainer_agent(self):
        return SingleAgentProcess(
            0,
            self.algo.copy(),
            DummyEnv(),
            DummyEnv(),
            self.replay_buffer.copy(),
            self.trainer_device,
            0,
            self.normalize_actions,
            name="trainer_synchron",
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
        )

    def load_checkpoint(self, file_path: str) -> None:
        super().load_checkpoint(file_path)

        self._worker_load_state_dicts_network(self.algo.state_dicts_network())
        self.trainer.load_state_dicts_network(self.algo.state_dicts_network())