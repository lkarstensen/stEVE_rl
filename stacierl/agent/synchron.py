import logging
from typing import List, Tuple

from .agent import Agent
from .single import EpisodeCounter, StepCounter, Algo, ReplayBuffer
from .singelagentprocess import SingleAgentProcess
from ..util import Environment, DummyEnvironment
from math import ceil, inf
import torch
import os


class Synchron(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: Environment,
        env_eval: Environment,
        replay_buffer: ReplayBuffer,
        n_worker: int,
        n_trainer: int,
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
        self._step_counter_set_point = StepCounter()
        self._episode_counter_set_point = EpisodeCounter()

        for i in range(n_worker):
            self.worker.append(
                SingleAgentProcess(
                    i,
                    algo.copy(),
                    env_train.copy(),
                    env_eval.copy(),
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
                    DummyEnvironment(),
                    DummyEnvironment(),
                    replay_buffer.copy(),
                    trainer_device,
                    0,
                    name="trainer_" + str(i),
                    parent_agent=self,
                )
            )
        self.logger.debug("Synchron Agent initialized")

    def heatup(
        self, steps: int = inf, episodes: int = inf, custom_action_low: List[float] = None
    ) -> Tuple[List[float], List[float]]:
        self.logger.debug(f"heatup: {steps} steps / {episodes} episodes")
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.heatup(steps_per_agent, episodes_per_agent, custom_action_low)
        result = self._get_worker_results()
        return result

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        self.logger.debug(f"explore: {steps} steps / {episodes} episodes")

        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.explore(steps_per_agent, episodes_per_agent)
        result = self._get_worker_results()
        return result

    def update(self, steps) -> List[float]:
        self.logger.debug(f"update: {steps} steps")
        steps_per_agent = ceil(steps / self.n_trainer)
        for agent in self.trainer:
            agent.update(steps_per_agent)
        result = self._get_trainer_results()
        new_network_states = self._get_network_states_container()
        self._set_network_states_container(new_network_states)
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[List[float], List[float]]:
        self.logger.debug(f"evaluate: {steps} steps / {episodes} episodes")

        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes, self.n_worker
        )
        for agent in self.worker:
            agent.evaluate(steps_per_agent, episodes_per_agent)

        result = self._get_worker_results()
        return result

    def explore_and_update_parallel(
        self, update_steps: int, explore_steps: int, explore_episodes: int
    ) -> Tuple[Tuple[List[float], List[float]], List[float]]:
        update_steps_per_agent = ceil(update_steps / self.n_trainer)
        explore_steps_per_agent, explore_episodes_per_agent = self._divide_steps_and_episodes(
            explore_steps, explore_episodes, self.n_worker
        )
        for agent in self.trainer:
            agent.update(update_steps_per_agent)
        for agent in self.worker:
            agent.explore(explore_steps_per_agent, explore_episodes_per_agent)
        update_result = self._get_trainer_results()
        explore_result = self._get_worker_results()
        new_network_states = self._get_network_states_container()
        self._set_network_states_container(new_network_states)

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
            optimizer_states_container = self.trainer[0].get_optimizer_states_container()
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
        results = [val for result_tuple in zip(*results) for val in result_tuple if val is not None]
        return results

    @property
    def step_counter(self) -> StepCounter:
        step_counter = StepCounter()
        for agent in self.worker + self.trainer:
            step_counter += agent.step_counter
        step_counter.heatup = int(step_counter.heatup)
        step_counter.exploration = int(step_counter.exploration)
        step_counter.evaluation = int(step_counter.evaluation)
        step_counter.update = int(step_counter.update)
        return step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        episode_counter = EpisodeCounter()
        for agent in self.worker + self.trainer:
            episode_counter += agent.episode_counter
        episode_counter.heatup = int(episode_counter.heatup)
        episode_counter.exploration = int(episode_counter.exploration)
        episode_counter.evaluation = int(episode_counter.evaluation)
        return episode_counter

    def save_checkpoint(self, directory: str, name: str) -> None:
        path = directory + "/" + name + ".pt"
        new_optimizer_states_container = self._get_optimizer_states_container()
        new_network_states_container = self._get_network_states_container()
        optimizer_state_dicts = new_optimizer_states_container.to_dict()
        network_state_dicts = new_network_states_container.to_dict()

        step_counter = self.step_counter

        checkpoint_dict = {
            "optimizer_state_dicts": optimizer_state_dicts,
            "network_state_dicts": network_state_dicts,
            "heatup_steps": step_counter.heatup,
            "exploration_steps": step_counter.exploration,
            "update_steps": step_counter.update,
            "evaluation_steps": step_counter.evaluation,
        }

        torch.save(checkpoint_dict, path)

    def load_checkpoint(self, directory: str, name: str) -> None:
        name, _ = os.path.splitext(name)
        path = os.path.join(directory, name + ".pt")
        checkpoint = torch.load(path)

        network_states_container = self.trainer[0].get_network_states_container()
        network_states_container.from_dict(checkpoint["network_state_dicts"])

        optimizer_states_container = self.trainer[0].get_optimizer_states_container()
        optimizer_states_container.from_dict(checkpoint["optimizer_state_dicts"])

        single_worker_step_counter = StepCounter(
            checkpoint["heatup_steps"] / self.n_worker,
            checkpoint["exploration_steps"] / self.n_worker,
            checkpoint["evaluation_steps"] / self.n_worker,
            0,
        )
        single_trainer_step_counter = StepCounter(
            0,
            0,
            0,
            checkpoint["update_steps"] / self.n_trainer,
        )

        for worker in self.worker:
            worker.set_network_states(network_states_container)
            worker.step_counter = single_worker_step_counter

        for trainer in self.trainer:
            trainer.set_optimizer_states(optimizer_states_container)
            trainer.set_network_states(network_states_container)
            trainer.step_counter = single_trainer_step_counter
