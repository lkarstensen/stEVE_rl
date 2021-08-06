from typing import List, Tuple

from .agent import Agent, dataclass
from .single import Single, EpisodeCounter, StepCounter, Algo, ReplayBuffer, Environment
from ..environment import EnvFactory
from torch import multiprocessing as mp
from math import ceil
import numpy as np
import torch


class SingleAgentProcess(Agent):
    def __init__(
        self,
        id: int,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
    ) -> None:

        self.id = id
        self._shutdown_event = mp.Event()
        self._task_queue = mp.SimpleQueue()
        self._result_queue = mp.SimpleQueue()
        self._model_queue = mp.SimpleQueue()

        self.device = device

        self._step_counter_exploration: mp.Value = mp.Value("i", 0)
        self._step_counter_eval: mp.Value = mp.Value("i", 0)
        self._step_counter_update: mp.Value = mp.Value("i", 0)
        self._episode_counter_exploration: mp.Value = mp.Value("i", 0)
        self._episode_counter_eval: mp.Value = mp.Value("i", 0)
        self._process = mp.Process(
            target=self.run,
            args=[
                id,
                algo,
                env,
                replay_buffer,
                device,
                consecutive_action_steps,
                self._task_queue,
                self._result_queue,
                self._model_queue,
                self._shutdown_event,
                self._step_counter_exploration,
                self._episode_counter_exploration,
                self._step_counter_eval,
                self._episode_counter_eval,
                self._step_counter_update,
            ],
        )
        self._process.start()

    def run(
        self,
        id: int,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        model_queue: mp.SimpleQueue,
        shutdown_event: mp.Event,
        step_counter_exploration: mp.Value,
        episode_counter_exploration: mp.Value,
        step_counter_eval: mp.Value,
        episode_counter_eval: mp.Value,
        step_counter_update: mp.Value,
    ):
        agent = Single(algo, env, replay_buffer, device, consecutive_action_steps)
        while not shutdown_event.is_set():
            task = task_queue.get()

            task_name = task[0]
            if task_name == "heatup":
                result = agent.heatup(task[1], task[2])
            elif task_name == "explore":
                result = agent.explore(task[1], task[2])
                episode_counter_exploration.value = agent.episode_counter.exploration
                step_counter_exploration.value = agent.step_counter.exploration
            elif task_name == "evaluate":
                result = agent.evaluate(task[1], task[2])
                episode_counter_eval.value = agent.episode_counter.eval
                step_counter_eval.value = agent.step_counter.eval
            elif task_name == "update":
                agent.update(task[1], task[2])
                step_counter_update.value = agent.step_counter.update
                continue
            elif task_name == "put_state_dict":
                agent.to(torch.device("cpu"))
                model_queue.put(agent.algo.model.all_state_dicts())
                agent.to(device)
                continue
            elif task_name == "set_state_dict":
                agent.to(torch.device("cpu"))
                result = agent.algo.model.load_all_state_dicts(task[1])
                agent.to(device)
                continue
            else:
                continue
            result_queue.put(result)

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["heatup", steps, episodes])

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["explore", steps, episodes])

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        self._task_queue.put(["evaluate", steps, episodes])

    def update(self, steps, batch_size):
        self._task_queue.put(["update", steps, batch_size])

    def set_state_dict(self, all_state_dicts):
        self._task_queue.put(["set_state_dict", all_state_dicts])

    def put_state_dict(self):
        self._task_queue.put(["put_state_dict"])

    def get_result(self):
        return self._result_queue.get()

    def get_state_dict(self):
        return self._model_queue.get()

    def close(self):
        self._shutdown_event.set()
        self._task_queue.put(["shutdown"])

    @property
    def step_counter(self) -> StepCounter:
        return StepCounter(
            self._step_counter_exploration.value,
            self._step_counter_eval.value,
            self._step_counter_update.value,
        )

    @property
    def episode_counter(self) -> EpisodeCounter:
        return EpisodeCounter(
            self._episode_counter_exploration.value, self._episode_counter_eval.value
        )


class Parallel(Agent):
    def __init__(
        self,
        n_agents: int,
        algo: Algo,
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        model_update_per_agent_tau=0.35,
    ) -> None:

        self.device = device
        self.algo = algo
        self.dummy_algo = self.algo.copy()
        self.env_factory = env_factory
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps
        self.n_agents = n_agents
        self.model_update_per_agent_tau = model_update_per_agent_tau

        self.agents: List[SingleAgentProcess] = []

        for i in range(n_agents):
            self.agents.append(
                SingleAgentProcess(
                    i,
                    self.algo.copy(),
                    self.env_factory.create_env(),
                    replay_buffer.copy(),
                    self.device,
                    consecutive_action_steps,
                )
            )

    def heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.heatup(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.explore(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def update(self, steps, batch_size):
        steps_per_agent = ceil(steps / self.n_agents)
        for agent in self.agents:
            agent.update(steps_per_agent, batch_size)
            agent.put_state_dict()
        for agent in self.agents:
            state_dict = agent.get_state_dict()
            self.dummy_algo.model.load_all_state_dicts(state_dict)
            all_parameters = self.dummy_algo.model.all_parameters()
            self.algo.model.soft_tau_update_all(all_parameters, self.model_update_per_agent_tau)

        all_state_dicts = self.algo.model.all_state_dicts()
        for agent in self.agents:
            agent.set_state_dict(all_state_dicts)

    def evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        steps_per_agent, episodes_per_agent = self._divide_steps_and_episodes(steps, episodes)
        for agent in self.agents:
            agent.evaluate(steps_per_agent, episodes_per_agent)
        results = []
        for agent in self.agents:
            results.append(agent.get_result())

        results = np.array(results)
        results = np.mean(results, axis=0)
        return tuple(results)

    def close(self):
        for agent in self.agents:
            agent.close()

    def _divide_steps_and_episodes(self, steps, episodes) -> Tuple[int, int]:

        steps = ceil(steps / self.n_agents) if steps is not None else None

        episodes = ceil(episodes / self.n_agents) if episodes is not None else None

        return steps, episodes

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
