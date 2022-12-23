from copy import deepcopy
import logging
from typing import List, Tuple

from .agent import Agent, Episode, StepCounterShared, EpisodeCounterShared
from .single import EpisodeCounter, StepCounter, Algo, ReplayBuffer, Env
from .singelagentprocess import SingleAgentProcess
from eve.env import DummyEnv
from math import ceil, inf
import torch
import os


class Synchron(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: Env,
        env_eval: Env,
        replay_buffer: ReplayBuffer,
        n_worker: int,
        n_trainer: int,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        share_trainer_model=False,
    ) -> None:

        # needed for saving config only
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.worker_device = worker_device
        self.trainer_device = trainer_device
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.n_trainer = n_trainer
        self.share_trainer_model = share_trainer_model
        self.worker: List[SingleAgentProcess] = []
        self.trainer: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer
        self._step_counter = StepCounterShared()
        self._episode_counter = EpisodeCounterShared()

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        for i in range(n_trainer):
            self.trainer.append(self._create_trainer_agent(i))
        self.logger.debug("Synchron Agent initialized")

    @property
    def step_counter(self) -> StepCounter:
        return self._step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self._episode_counter

    def heatup(
        self,
        steps: int = inf,
        episodes: int = inf,
        custom_action_low: List[float] = None,
        custom_action_high: List[float] = None,
    ) -> List[Episode]:
        self.logger.info(f"heatup: {steps} steps / {episodes} episodes")
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
        self.logger.info(f"explore: {steps} steps / {episodes} episodes")

        for agent in self.worker:
            agent.explore(steps, episodes)
        result = self._get_worker_results()
        return result

    def update(self, steps) -> List[float]:
        self.logger.info(f"update: {steps} steps")
        for agent in self.trainer:
            agent.update(steps)
        result = self._get_trainer_results()
        new_network_states = self._get_network_states_container()
        self._set_network_states_container(new_network_states)
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        self.logger.info(f"evaluate: {steps} steps / {episodes} episodes")

        for agent in self.worker:
            agent.evaluate(steps, episodes)

        result = self._get_worker_results()
        return result

    def explore_and_update_parallel(
        self, update_steps: int, explore_steps: int = inf, explore_episodes: int = inf
    ) -> Tuple[List[Episode], List[float]]:
        self.logger.info(
            f"explore: {explore_steps} steps / {explore_episodes} episodes, update: {update_steps} steps "
        )

        for agent in self.trainer:
            agent.update(update_steps)
        for agent in self.worker:
            agent.explore(explore_steps, explore_episodes)
        update_result = self._get_trainer_results()
        explore_result = self._get_worker_results()
        self.latest_network_states = self._get_network_states_container()
        self._set_network_states_container(self.latest_network_states)

        return explore_result, update_result

    def close(self):
        for agent in self.worker + self.trainer:
            agent.close()
        self.replay_buffer.close()

    def _get_network_states_container(self):
        if self.share_trainer_model:
            new_network_states = self.trainer[0].get_network_states_container()
        else:
            new_network_states = None
            for agent in self.trainer:
                network_states = agent.get_network_states_container()
                if new_network_states is None:
                    new_network_states = network_states
                else:
                    new_network_states += network_states
            new_network_states /= self.n_trainer
        return new_network_states

    def _set_network_states_container(self, new_network_states):
        for agent in self.worker:
            agent.set_network_states(new_network_states)
        if self.share_trainer_model:
            self.trainer[0].set_network_states(new_network_states)
        else:
            for agent in self.trainer:
                agent.set_network_states(new_network_states)

    def _get_optimizer_states_container(self):
        if self.share_trainer_model:
            optimizer_states_container = self.trainer[
                0
            ].get_optimizer_states_container()
        else:
            new_optimizer_states_container = None
            for trainer in self.trainer:
                optimizer_states_container = trainer.get_optimizer_states_container()
                if new_optimizer_states_container is None:
                    new_optimizer_states_container = optimizer_states_container
                else:
                    new_optimizer_states_container += optimizer_states_container
            optimizer_states_container /= self.n_trainer
        return optimizer_states_container

    def _divide_steps_and_episodes(self, steps, episodes, n_agents) -> Tuple[int, int]:
        steps = ceil(steps / n_agents) if steps != inf else inf
        episodes = ceil(episodes / n_agents) if episodes != inf else inf
        return steps, episodes

    def _get_worker_results(self):
        episodes = []
        for i, agent in enumerate(self.worker):
            try:
                episodes += agent.get_result()
            except Exception as e:
                self.logger.warning(
                    f"Restaring Agent {agent.name} because of Exception {e}"
                )
                agent.close()
                new_agent = self._create_worker_agent(i)
                new_agent.set_network_states(self.latest_network_states)
                self.worker[i] = new_agent
                continue

        return episodes

    def _get_trainer_results(self):
        results = []
        for i, agent in enumerate(self.trainer):
            try:
                results.append(agent.get_result())
            except Exception as e:
                self.logger.warning(
                    f"Restaring Agent {agent.name} because of Exception {e}"
                )
                agent.close()
                new_agent = self._create_trainer_agent(i)
                self.trainer[i] = new_agent
                continue

        n_max = len(max(results, key=len))
        results = [result + [None] * (n_max - len(result)) for result in results]
        results = [
            val
            for result_tuple in zip(*results)
            for val in result_tuple
            if val is not None
        ]
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

    def _create_trainer_agent(self, i):
        if self.share_trainer_model:
            new_algo = self.algo.copy_shared_memory()
        else:
            new_algo = self.algo.copy()
        return SingleAgentProcess(
            i,
            new_algo,
            DummyEnv(),
            DummyEnv(),
            self.replay_buffer.copy(),
            self.trainer_device,
            0,
            self.normalize_actions,
            name="trainer_" + str(i),
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
        )

    def save_checkpoint(self, file_path: str) -> None:
        new_optimizer_states_container = self._get_optimizer_states_container()
        new_network_states_container = self._get_network_states_container()
        optimizer_state_dicts = new_optimizer_states_container.to_dict()
        network_state_dicts = new_network_states_container.to_dict()
        scheduler_states = self.trainer[0].get_scheduler_states_container()

        step_counter = self.step_counter

        checkpoint_dict = {
            "scheduler_state_dicts": scheduler_states,
            "optimizer_state_dicts": optimizer_state_dicts,
            "network_state_dicts": network_state_dicts,
            "heatup_steps": step_counter.heatup,
            "exploration_steps": step_counter.exploration,
            "update_steps": step_counter.update,
            "evaluation_steps": step_counter.evaluation,
        }

        torch.save(checkpoint_dict, file_path)

    def load_checkpoint(self, file_path: str) -> None:
        checkpoint = torch.load(file_path)

        network_states_container = self.trainer[0].get_network_states_container()
        network_states_container.from_dict(checkpoint["network_state_dicts"])

        optimizer_states_container = self.trainer[0].get_optimizer_states_container()
        optimizer_states_container.from_dict(checkpoint["optimizer_state_dicts"])

        scheduler_states_container = checkpoint["scheduler_state_dicts"]
        # self.trainer[0].get_scheduler_states_container()
        # scheduler_states_container.from_dict(checkpoint["scheduler_state_dicts"])

        self.step_counter.heatup = checkpoint["heatup_steps"]
        self.step_counter.exploration = checkpoint["exploration_steps"]
        self.step_counter.evaluation = checkpoint["evaluation_steps"]
        self.step_counter.update = checkpoint["update_steps"]

        for worker in self.worker:
            worker.set_network_states(network_states_container)

        for trainer in self.trainer:
            trainer.set_optimizer_states(optimizer_states_container)
            trainer.set_network_states(network_states_container)
            trainer.set_scheduler_states(scheduler_states_container)

    def copy(self):
        return self.__class__(
            self.algo.copy(),
            self.env_train.copy(),
            self.env_eval.copy(),
            self.replay_buffer.copy(),
            self.n_worker,
            self.n_trainer,
            self.worker_device,
            self.trainer_device,
            self.consecutive_action_steps,
            self.share_trainer_model,
        )
