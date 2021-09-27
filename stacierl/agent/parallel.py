from typing import List, Tuple

from .agent import Agent
from .single import EpisodeCounter, StepCounter, Algo, ReplayBuffer
from .singelagentprocess import SingleAgentProcess
from ..util import Environment
from torch import multiprocessing as mp
from math import ceil, inf
import numpy as np
import torch


class Parallel(Agent):
    def __init__(
        self,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        n_agents: int,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        shared_model=False,
    ) -> None:

        self.n_agents = n_agents
        self.shared_model = shared_model
        self.agents: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer

        for i in range(n_agents):
            if shared_model:
                new_algo = algo.copy_shared_memory()
            else:
                new_algo = algo.copy()
            self.agents.append(
                SingleAgentProcess(
                    i,
                    new_algo,
                    env.copy(),
                    replay_buffer.copy(),
                    device,
                    consecutive_action_steps,
                    name="agent_" + str(i),
                    parent_agent=self,
                )
            )

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.heatup(steps_per_agent, episodes_per_agent)
        result = self._get_play_results()
        return result

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.explore(steps_per_agent, episodes_per_agent)
        result = self._get_play_results()
        return result

    def update(self, steps):

        steps_per_agent = ceil(steps / self.n_agents)
        for agent in self.agents:
            agent.update(steps_per_agent)
        result = self._get_update_results()
        if not self.shared_model:
            for agent in self.agents:
                agent.put_state_dict()

            new_state_dict = None
            for agent in self.agents:
                state_dicts = agent.get_state_dict() / self.n_agents
                if new_state_dict is None:
                    new_state_dict = state_dicts
                else:
                    new_state_dict += state_dicts

            for agent in self.agents:
                agent.set_state_dict(new_state_dict)

        result = list(result) if np.any(result) else result
        return result

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
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
        successes = []
        rewards = []
        for agent in self.agents:
            reward, success = agent.get_result()
            rewards += reward
            successes += success
        return rewards, successes

    def _get_update_results(self):
        results = []
        for agent in self.agents:
            results.append(agent.get_result())
        n_max = len(max(results, key=len))
        results = [result + [None] * (n_max - len(result)) for result in results]
        results = [val for result_tuple in zip(*results) for val in result_tuple if val is not None]
        return results

    @property
    def step_counter(self) -> StepCounter:
        step_counter = StepCounter()
        for agent in self.agents:
            step_counter += agent.step_counter
        return step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        episode_counter = EpisodeCounter()
        for agent in self.agents:
            episode_counter += agent.episode_counter

        return episode_counter
