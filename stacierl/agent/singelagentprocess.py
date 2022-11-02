from copy import deepcopy
import logging
from math import inf

from typing import Any, Dict, List, Optional, Tuple

from .agent import (
    Agent,
    EpisodeCounterShared,
    StepCounterShared,
    StepCounter,
    EpisodeCounter,
)
from .single import Single, Algo, ReplayBuffer, Env
from ..algo.model import NetworkStatesContainer, OptimizerStatesContainer
from torch import multiprocessing as mp
import torch

import queue

import logging.config
from random import randint


def file_handler_callback(handler: logging.FileHandler):
    handler_dict = {
        handler.name: {
            "level": handler.level,
            "class": "logging.FileHandler",
            "filename": handler.baseFilename,
            "mode": handler.mode,
        }
    }
    if handler.formatter is not None:
        formatter_name = handler.name or randint(1, 99999)
        handler_dict[handler.name]["formatter"] = str(formatter_name)
        formatter_dict = {str(formatter_name): {"format": handler.formatter._fmt}}
    else:
        formatter_dict = None
    return handler_dict, formatter_dict, handler.name


handler_callback = {logging.FileHandler: file_handler_callback}


def get_logging_config_dict():

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {},
        "handlers": {},
        "loggers": {
            "": {
                "handlers": [],
                "level": logging.WARNING,
                "propagate": False,
            },  # root logger
        },
    }

    config["loggers"][""]["level"] = logging.root.level
    config["loggers"][""]["propagate"] = logging.root.propagate
    for handler in logging.root.handlers:
        handler_dict, formatter_dict, name = handler_callback[type(handler)](handler)
        if formatter_dict is not None:
            config["formatters"].update(formatter_dict)
        config["handlers"].update(handler_dict)
        config["loggers"][""]["handlers"].append(name)
    return config


def run(
    id: int,
    algo: Algo,
    env_train: Env,
    env_eval: Env,
    replay_buffer: ReplayBuffer,
    device: torch.device,
    consecutive_action_steps: int,
    log_config_dict: Dict,
    task_queue,
    result_queue,
    model_queue,
    step_counter,
    episode_counter,
    shutdown_event,
    name,
):
    try:
        torch.set_num_threads(1)
        for handler_name, handler_config in log_config_dict["handlers"].items():
            if "filename" in handler_config.keys():
                filename = handler_config["filename"]
                if ".log" in filename:
                    filename = filename.replace(".log", f"-{name}.log")
                else:
                    filename += f"-{name}"
                log_config_dict["handlers"][handler_name]["filename"] = filename
        logging.config.dictConfig(log_config_dict)
        logger = logging.getLogger(__name__)
        logger.info("logger initialized")
        agent = Single(
            algo, env_train, env_eval, replay_buffer, device, consecutive_action_steps
        )
        agent.step_counter = step_counter
        agent.episode_counter = episode_counter
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue

            task_name = task[0]
            if task_name == "heatup":
                result = agent.heatup(task[1], task[2], task[3])
            elif task_name == "explore":
                result = agent.explore(task[1], task[2])
            elif task_name == "evaluate":
                result = agent.evaluate(task[1], task[2])
            elif task_name == "update":
                try:
                    result = agent.update(task[1])
                except ValueError as error:
                    logger.warning(f"Update Error: {error}")
                    shutdown_event.set()
                    result = error
            elif task_name == "put_network_states_container":
                network_states_container = deepcopy(
                    agent.algo.model.network_states_container
                )
                network_states_container.to(torch.device("cpu"))
                model_queue.put(network_states_container)
                continue
            elif task_name == "set_network_states_container":
                network_states_container = task[1]
                network_states_container.to(device)
                agent.algo.model.set_network_states(network_states_container)
                continue
            elif task_name == "put_optimizer_states_container":
                optimizer_states_container = deepcopy(
                    agent.algo.model.optimizer_states_container
                )
                optimizer_states_container.to(torch.device("cpu"))
                model_queue.put(optimizer_states_container)
                continue
            elif task_name == "set_optmizer_states_container":
                optimizer_states_container = task[1]
                optimizer_states_container.to(device)
                agent.algo.model.set_optimizer_states(optimizer_states_container)
                continue
            elif task_name == "shutdown":
                agent.close()
                continue
            else:
                continue
            result_queue.put(result)
    except Exception as e:
        logger.warning(e)
        result_queue.put(e)
    agent.close()


class SingleAgentProcess(Agent):
    def __init__(
        self,
        id: int,
        algo: Algo,
        env_train: Env,
        env_eval: Env,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        consecutive_action_steps: int,
        name: str,
        parent_agent: Agent,
    ) -> None:

        self.id = id
        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._model_queue = mp.Queue()

        self.device = device
        self.parent_agent = parent_agent

        self._step_counter = StepCounterShared()
        self._episode_counter = EpisodeCounterShared()
        logging_config = get_logging_config_dict()
        self._process = mp.Process(
            target=run,
            args=[
                id,
                algo,
                env_train,
                env_eval,
                replay_buffer,
                device,
                consecutive_action_steps,
                logging_config,
                self._task_queue,
                self._result_queue,
                self._model_queue,
                self._step_counter,
                self._episode_counter,
                self._shutdown_event,
                name,
            ],
            name=name,
        )
        self._process.start()

    def heatup(
        self,
        steps: int = inf,
        episodes: int = inf,
        custom_action_low: List[float] = None,
    ) -> None:
        self._task_queue.put(["heatup", steps, episodes, custom_action_low])

    def explore(self, steps: int = inf, episodes: int = inf) -> None:
        self._task_queue.put(["explore", steps, episodes])

    def evaluate(self, steps: int = inf, episodes: int = inf) -> None:
        self._task_queue.put(["evaluate", steps, episodes])

    def update(self, steps) -> None:
        self._task_queue.put(["update", steps])

    def get_result(self) -> List[Any]:
        result = self._result_queue.get()
        if isinstance(result, Exception):
            self.parent_agent.close()
            raise result
        return result

    def set_network_states(self, states_container: NetworkStatesContainer):
        self._task_queue.put(["set_network_states_container", states_container])

    def get_network_states_container(self) -> NetworkStatesContainer:
        self._task_queue.put(["put_network_states_container"])
        return self._model_queue.get()

    def set_optimizer_states(self, states_container: OptimizerStatesContainer):
        self._task_queue.put(["set_optimizer_states_container", states_container])

    def get_optimizer_states_container(self) -> OptimizerStatesContainer:
        self._task_queue.put(["put_optimizer_states_container"])
        return self._model_queue.get()

    def close(self) -> None:
        self._shutdown_event.set()
        self._task_queue.put(["shutdown"])
        self._process.join()
        self._clear_queues()
        self._process.close()

    def _clear_queues(self):
        if self._process.is_alive():
            return
        for queue_ in [self._result_queue, self._model_queue, self._task_queue]:
            while True:
                try:
                    queue_.get_nowait()
                except queue.Empty:
                    break

    @property
    def step_counter(self) -> StepCounterShared:
        return self._step_counter

    @step_counter.setter
    def step_counter(self, new_counter: StepCounter) -> None:
        self._step_counter.heatup = new_counter.heatup
        self._step_counter.exploration = new_counter.exploration
        self._step_counter.evaluation = new_counter.evaluation
        self._step_counter.update = new_counter.update

    @property
    def episode_counter(self) -> EpisodeCounterShared:
        return self._episode_counter

    @episode_counter.setter
    def episode_counter(self, new_counter: EpisodeCounter) -> None:
        self._episode_counter.heatup = new_counter.heatup
        self._episode_counter.exploration = new_counter.exploration
        self._episode_counter.evaluation = new_counter.evaluation

    def load_checkpoint(self, directory: str, name: str) -> None:
        ...

    def save_checkpoint(self, directory: str, name: str) -> None:
        ...

    def copy(self):
        ...
