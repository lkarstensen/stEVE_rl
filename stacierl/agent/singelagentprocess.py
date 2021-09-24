import logging
from math import inf

from typing import Dict, Optional, Tuple

from .agent import Agent, EpisodeCounterShared, StepCounterShared
from .single import Single, EpisodeCounter, StepCounter, Algo, ReplayBuffer
from ..environment import EnvFactory
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
            "": {"handlers": [], "level": logging.WARNING, "propagate": False},  # root logger
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
    env_factory: EnvFactory,
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
    env = env_factory.create_env()
    agent = Single(algo, env, replay_buffer, device, consecutive_action_steps)
    agent.step_counter = step_counter
    agent.episode_counter = episode_counter
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        task_name = task[0]
        if task_name == "heatup":
            result = agent.heatup(task[1], task[2])
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
        elif task_name == "put_state_dict":
            state_dicts = agent.algo.state_dicts
            state_dicts.to(torch.device("cpu"))
            model_queue.put(state_dicts)
            continue
        elif task_name == "set_state_dict":
            state_dicts = task[1]
            state_dicts.to(device)
            agent.algo.load_state_dicts(state_dicts)
            continue
        elif task_name == "shutdown":
            agent.close()
            continue
        else:
            continue
        result_queue.put(result)

    env.close()
    replay_buffer.close()


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
        parent_agent: Agent,
        step_counter: Optional[StepCounterShared] = None,
        episode_counter: Optional[EpisodeCounterShared] = None,
    ) -> None:

        # log_level = logging.root.level
        # for handler in logging.root.handlers:
        #     # check the handler is a file handler
        #     # (rotating handler etc. inherit from this, so it will still work)
        #     # stream handlers write to stderr, so their filename is not useful to us
        #     if isinstance(handler, logging.FileHandler):
        #         # h.stream should be an open file handle, it's name is the path
        #         log_file = f"{handler.baseFilename}-{name}"
        #         log_format = handler.formatter._fmt
        #         break
        self.id = id
        self._shutdown_event = mp.Event()
        self._task_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._model_queue = mp.Queue()

        self.device = device
        self.parent_agent = parent_agent

        self._step_counter = step_counter or StepCounterShared()
        self._episode_counter = episode_counter or EpisodeCounterShared()
        logging_config = get_logging_config_dict()
        self._process = mp.Process(
            target=run,
            args=[
                id,
                algo,
                env_factory,
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

    def heatup(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        self._task_queue.put(["heatup", steps, episodes])

    def explore(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        self._task_queue.put(["explore", steps, episodes])

    def evaluate(self, steps: int = inf, episodes: int = inf) -> Tuple[float, float]:
        self._task_queue.put(["evaluate", steps, episodes])

    def update(self, steps):
        self._task_queue.put(["update", steps])

    def set_state_dict(self, all_state_dicts):
        self._task_queue.put(["set_state_dict", all_state_dicts])

    def put_state_dict(self):
        self._task_queue.put(["put_state_dict"])

    def get_result(self):
        result = self._result_queue.get()
        if isinstance(result, Exception):
            self.parent_agent.close()
            raise result
        return result

    def get_state_dict(self):
        return self._model_queue.get()

    def close(self):
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

    @property
    def episode_counter(self) -> EpisodeCounterShared:
        return self._episode_counter
