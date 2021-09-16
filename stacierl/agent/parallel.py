import logging

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
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
        name: str,
    ) -> None:
        log_level = logging.root.level
        for handler in logging.root.handlers:
            # check the handler is a file handler
            # (rotating handler etc. inherit from this, so it will still work)
            # stream handlers write to stderr, so their filename is not useful to us
            if isinstance(handler, logging.FileHandler):
                # h.stream should be an open file handle, it's name is the path
                log_file = f"{handler.baseFilename}-{name}"
                log_format = handler.formatter._fmt
                break
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
                env_factory,
                replay_buffer,
                device,
                consecutive_action_steps,
                log_file,
                log_level,
                log_format,
            ],
            name=name,
        )
        self._process.start()

    def run(
        self,
        id: int,
        algo: Algo,
        env_factory: EnvFactory,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
        log_file: str,
        log_level,
        log_format,
    ):
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format=log_format,
        )
        logging.info("logging initialized")

        env = env_factory.create_env()
        agent = Single(algo, env, replay_buffer, device, consecutive_action_steps)
        while not self._shutdown_event.is_set():
            task = self._task_queue.get()

            task_name = task[0]
            if task_name == "heatup":
                result = agent.heatup(task[1], task[2])
            elif task_name == "explore":
                result = agent.explore(task[1], task[2])
                self._episode_counter_exploration.value = agent.episode_counter.exploration
                self._step_counter_exploration.value = agent.step_counter.exploration
            elif task_name == "evaluate":
                result = agent.evaluate(task[1], task[2])
                self._episode_counter_eval.value = agent.episode_counter.eval
                self._step_counter_eval.value = agent.step_counter.eval
            elif task_name == "update":
                agent.update(task[1], task[2])
                self._step_counter_update.value = agent.step_counter.update
                continue
            elif task_name == "put_state_dict":
                state_dicts = agent.algo.model.nets.state_dicts
                state_dicts.to(torch.device("cpu"))
                self._model_queue.put(state_dicts)
                continue
            elif task_name == "set_state_dict":
                state_dicts = task[1]
                state_dicts.to(device)
                agent.algo.model.nets.load_state_dicts(state_dicts)
                continue
            elif task_name == "shutdown":
                agent.close()
                continue
            else:
                continue
            self._result_queue.put(result)

        env.close()
        replay_buffer.close()

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
                    env_factory,
                    replay_buffer.copy(),
                    device,
                    consecutive_action_steps,
                    name="agent_" + str(i),
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
        self.replay_buffer.close()

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
