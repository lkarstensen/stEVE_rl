import queue
from typing import List, Tuple

from tiltmaze.env import Env
from stacierl.replaybuffer import replaybuffer
from .agent import Agent
from .singleagent import SingleAgent, dict_state_to_flat_np_state
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from ..environment import Environment, EnvFactory
from torch import multiprocessing as mp
from math import inf, ceil
import numpy as np


class SingleAgentProcess(mp.Process, SingleAgent):
    def __init__(
        self,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        consecutive_action_steps: int = 1,
    ) -> None:
        mp.Process.__init__(
            self,
        )
        SingleAgent.__init__(self, algo, env, replay_buffer, consecutive_action_steps)

        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()

        self.explore_step_counter_mp = mp.Value("i", 0)
        self.explore_episode_counter_mp = mp.Value("i", 0)
        self.update_step_counter_mp = mp.Value("i", 0)
        self.eval_step_counter_mp = mp.Value("i", 0)
        self.eval_episode_counter_mp = mp.Value("i", 0)

    def run(self):
        while not self._shutdown_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            task_name = task[0]
            if task_name == "heatup":
                result = self._heatup(task[1], task[2])
            elif task_name == "explore":
                result = self._explore(task[1], task[2])
                self.explore_episode_counter_mp.value = self.explore_episode_counter
                self.explore_step_counter_mp.value = self.explore_step_counter
            elif task_name == "evaluate":
                result = self._evaluate(task[1], task[2])
                self.eval_step_counter_mp.value = self.eval_step_counter
                self.eval_episode_counter_mp.value = self.eval_episode_counter
            elif task_name == "update":
                result = self._update(task[1], task[2])
                self.update_step_counter_mp.value = self.update_step_counter
            self._result_queue.put(result)

        while True:
            try:
                self._task_queue.get(block=False)
            except queue.Empty:
                break

        while True:
            try:
                self._result_queue.get(block=False)
            except queue.Empty:
                break

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["heatup", steps, episodes])

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["explore", steps, episodes])

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["evaluate", steps, episodes])

    def update(self, steps, batch_size):
        self._task_queue.put(["update", steps, batch_size])

    def get_results(self):
        return self._result_queue.get()

    def shutdown(self):
        self._shutdown_event.set()


class ParallelAgent(Agent):
    def __init__(
        self,
        n_agents: int,
        algo: Algo,
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        consecutive_action_steps: int = 1,
    ) -> None:

        self.algo = algo
        self.env_factory = env_factory
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.n_agents = n_agents

        self.agents: List[SingleAgentProcess] = []

        for _ in range(n_agents):
            self.agents.append(
                SingleAgentProcess(
                    self.algo.copy_shared_memory(),
                    self.env_factory.create_env(),
                    replay_buffer.copy(),
                    consecutive_action_steps,
                )
            )

        for agent in self.agents:
            agent.start()

    def _update(self, steps, batch_size):
        steps_per_agent = ceil(steps / self.n_agents)
        for agent in self.agents:
            agent.update(steps_per_agent, batch_size)
        results = []
        for agent in self.agents:
            results.append(agent.get_results())

    def _heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.heatup(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_results())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def _explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.explore(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_results())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def _evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.evaluate(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_results())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def close(self):
        for agent in self.agents:
            agent.shutdown()

    def _divide_steps_and_episodes(self, steps, episodes) -> Tuple[int, int]:
        if steps < inf:
            steps_per_agent = ceil(steps / self.n_agents)
        else:
            steps_per_agent = inf
        if episodes < inf:
            episodes_per_agent = ceil(episodes / self.n_agents)
        else:
            episodes_per_agent = inf
        return steps_per_agent, episodes_per_agent

    @property
    def explore_step_counter(self) -> int:
        result = 0
        for agent in self.agents:
            result += agent.explore_step_counter_mp.value
        return result

    @property
    def explore_episode_counter(self) -> int:
        result = 0
        for agent in self.agents:
            result += agent.explore_episode_counter_mp.value
        return result

    @property
    def eval_step_counter(self) -> int:
        result = 0
        for agent in self.agents:
            result += agent.eval_step_counter_mp.value
        return result

    @property
    def eval_episode_counter(self) -> int:
        result = 0
        for agent in self.agents:
            result += agent.eval_episode_counter_mp.value
        return result

    @property
    def update_step_counter(self) -> int:
        result = 0
        for agent in self.agents:
            result += agent.update_step_counter_mp.value
        return result
