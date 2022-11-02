from typing import List, Tuple

from .agent import Agent, Episode
from .single import EpisodeCounter, StepCounter, Algo, ReplayBuffer, Env
from .singelagentprocess import SingleAgentProcess
from torch import multiprocessing as mp
from math import ceil, inf
import numpy as np
import torch
import os


class Parallel(Agent):
    def __init__(
        self,
        algo: Algo,
        env_train: Env,
        env_eval: Env,
        replay_buffer: ReplayBuffer,
        n_agents: int,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        shared_model=False,
    ) -> None:

        # needed for saving config only
        self.algo = algo
        self.env_train = env_train
        self.env_eval = env_eval
        self.n_agents = n_agents
        self.consecutive_action_steps = consecutive_action_steps
        self.device = device

        self.shared_model = shared_model
        self.agents: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer

        for i in range(n_agents):
            if shared_model:
                new_algo = algo.copy_shared_memory()
            else:
                new_algo = algo.copy()

            if env_train is env_eval:
                new_env_train = new_env_eval = env_train.copy()
            else:
                new_env_train = env_train.copy()
                new_env_eval = env_eval.copy()

            self.agents.append(
                SingleAgentProcess(
                    i,
                    new_algo,
                    new_env_train,
                    new_env_eval,
                    replay_buffer.copy(),
                    device,
                    consecutive_action_steps,
                    name="agent_" + str(i),
                    parent_agent=self,
                )
            )
        if not shared_model:
            self.update(0)

    def heatup(
        self,
        steps: int = inf,
        episodes: int = inf,
        custom_action_low: List[float] = None,
    ) -> List[Episode]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes
        )
        for agent in self.agents:
            agent.heatup(steps_per_agent, episodes_per_agent, custom_action_low)
        result = self._get_play_results()
        return result

    def explore(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes
        )
        for agent in self.agents:
            agent.explore(steps_per_agent, episodes_per_agent)
        result = self._get_play_results()
        return result

    def update(self, steps) -> List[float]:

        steps_per_agent = ceil(steps / self.n_agents)
        for agent in self.agents:
            agent.update(steps_per_agent)
        result = self._get_update_results()
        if not self.shared_model:

            new_network_states_container = None
            for agent in self.agents:
                network_states_container = (
                    agent.get_network_states_container() / self.n_agents
                )
                if new_network_states_container is None:
                    new_network_states_container = network_states_container
                else:
                    new_network_states_container += network_states_container

            for agent in self.agents:
                agent.set_network_states(new_network_states_container)

        result = list(result) if np.any(result) else result
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> List[Episode]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(
            steps, episodes
        )
        for agent in self.agents:
            agent.evaluate(steps_per_agent, episodes_per_agent)
        result = self._get_play_results()
        return tuple(result)

    def close(self):
        for agent in self.agents:
            agent.close()
        self.replay_buffer.close()

    def _divide_steps_and_episodes(self, steps, episodes) -> Tuple[int, int]:

        steps = ceil(steps / self.n_agents) if steps != inf else inf

        episodes = ceil(episodes / self.n_agents) if episodes != inf else inf

        return steps, episodes

    def _get_play_results(self):
        episodes = []
        for agent in self.agents:
            episodes += agent.get_result()
        return episodes

    def _get_update_results(self):
        results = []
        for agent in self.agents:
            results.append(agent.get_result())
        n_max = len(max(results, key=len))
        results = [result + [None] * (n_max - len(result)) for result in results]
        results = [
            val
            for result_tuple in zip(*results)
            for val in result_tuple
            if val is not None
        ]
        return results

    @property
    def step_counter(self) -> StepCounter:
        step_counter = StepCounter()
        for agent in self.agents:
            step_counter += agent.step_counter
        step_counter.heatup = int(step_counter.heatup)
        step_counter.exploration = int(step_counter.exploration)
        step_counter.evaluation = int(step_counter.evaluation)
        step_counter.update = int(step_counter.update)
        return step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        episode_counter = EpisodeCounter()
        for agent in self.agents:
            episode_counter += agent.episode_counter
        episode_counter.heatup = int(episode_counter.heatup)
        episode_counter.exploration = int(episode_counter.exploration)
        episode_counter.evaluation = int(episode_counter.evaluation)
        return episode_counter

    def save_checkpoint(self, directory: str, name: str) -> None:
        path = directory + "/" + name + ".pt"

        new_optimizer_states_container = None
        for agent in self.agents:
            optimizer_states_container = (
                agent.get_optimizer_states_container() / self.n_agents
            )
            if new_optimizer_states_container is None:
                new_optimizer_states_container = optimizer_states_container
            else:
                new_optimizer_states_container += optimizer_states_container
        optimizer_state_dicts = new_optimizer_states_container.to_dict()

        new_network_states_container = None
        for agent in self.agents:
            network_states_container = (
                agent.get_network_states_container() / self.n_agents
            )
            if new_network_states_container is None:
                new_network_states_container = network_states_container
            else:
                new_network_states_container += network_states_container
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

        network_states_container = self.agents[0].get_network_states_container()
        network_states_container.from_dict(checkpoint["network_state_dicts"])

        optimizer_states_container = self.agents[0].get_optimizer_states_container()
        optimizer_states_container.from_dict(checkpoint["optimizer_state_dicts"])

        single_agent_step_counter = StepCounter(
            checkpoint["heatup_steps"] / self.n_agents,
            checkpoint["exploration_steps"] / self.n_agents,
            checkpoint["evaluation_steps"] / self.n_agents,
            checkpoint["update_steps"] / self.n_agents,
        )

        for agent in self.agents:
            agent.set_network_states(network_states_container)
            agent.set_optimizer_states(optimizer_states_container)
            agent.step_counter = single_agent_step_counter

    def copy(self):
        ...
