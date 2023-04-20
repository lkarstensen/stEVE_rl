from copy import deepcopy
from importlib import import_module
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from math import inf
import logging
import torch
import queue

from ..util import DummyEnv
from .agent import Agent, Episode, StepCounterShared, EpisodeCounterShared
from .single import Algo, ReplayBuffer, gym
from .singelagentprocess import SingleAgentProcess
from ..util import ConfigHandler


class SynchronEvalOnly(Agent):
    def __init__(
        self,
        algo: Algo,
        env_eval: gym.Env,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
    ) -> None:
        self.algo = algo
        self.algo.to(torch.device("cpu"))
        self.env_eval = env_eval
        self.worker_device = worker_device
        self.normalize_actions = normalize_actions
        self.timeout_worker_after_reaching_limit = timeout_worker_after_reaching_limit

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.worker: List[SingleAgentProcess] = []
        self.step_counter = StepCounterShared()
        self.episode_counter = EpisodeCounterShared()

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        self._eval_seeds = None
        self._eval_options = None

        self.logger.info("Synchron Agent initialized")

    def evaluate(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        options: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        steps_start = self.step_counter.evaluation
        episodes_start = self.episode_counter.evaluation
        self._log_task(
            "evaluate", steps, step_limit, episodes, episode_limit, seeds, options
        )
        step_limit, episode_limit = self._log_and_convert_limits(
            "evaluation", steps, step_limit, episodes, episode_limit, seeds, options
        )
        if seeds is not None:
            self._eval_seeds = self._split(seeds, self.n_worker)
        if options is not None:
            self._eval_options = self._split(options, self.n_worker)

        for agent in self.worker:
            i = agent.agent_id
            agent_seeds = self._eval_seeds[i] if seeds is not None else None
            agent_options = self._eval_options[i] if options is not None else None
            agent.evaluate(
                step_limit=step_limit,
                episode_limit=episode_limit,
                seeds=agent_seeds,
                options=agent_options,
            )

        self._eval_seeds = None
        self._eval_options = None
        result = self._get_worker_results(step_limit, episode_limit, "evaluation")

        n_steps = self.step_counter.evaluation - steps_start
        n_episodes = self.episode_counter.evaluation - episodes_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("evaluation", n_steps, t_duration, n_episodes)
        return result

    def close(self):
        for agent in self.worker:
            agent.close()

    def _worker_load_state_dicts_network(self, state_dicts: Dict[str, Any]):
        for agent in self.worker:
            agent.load_state_dicts_network(state_dicts)

    def _get_worker_results(self, step_limit: int, episode_limit: int, task: str):
        episode_results = []
        results_pending = self.worker.copy()
        t_limit_result = inf
        while results_pending and perf_counter() < t_limit_result:
            results_pending, t_limit_result, episode_results = self._worker_result_loop(
                step_limit,
                episode_limit,
                task,
                episode_results,
                results_pending,
                t_limit_result,
            )

        for agent in results_pending:
            log_warn = f"Restaring Agent {agent.name} because of Timeout"
            self.logger.warning(log_warn)
            self._restart_worker_agent(agent)

        return episode_results

    def _worker_result_loop(
        self,
        step_limit,
        episode_limit,
        task,
        episode_results,
        results_pending,
        t_limit_result,
    ):
        remove = []
        add = []
        for agent in results_pending:
            result = agent.get_result(timeout=0.1)
            if isinstance(result, queue.Empty):
                if not agent.is_alive():
                    log_warn = (
                        f"Restaring Agent {agent.name} because it is not alive anymore"
                    )
                    self.logger.warning(log_warn)
                    new = self._restart_worker_agent(
                        agent, task, step_limit, episode_limit
                    )
                    remove.append(agent)
                    add.append(new)

            elif isinstance(result, Exception):
                log_warn = f"Restaring Agent {agent.name} because of Exception {result}"
                self.logger.warning(log_warn)
                new = self._restart_worker_agent(agent, task, step_limit, episode_limit)
                remove.append(agent)
                add.append(new)

            else:
                episode_results += result
                remove.append(agent)

        for agent in remove:
            results_pending.remove(agent)
        for agent in add:
            results_pending.append(agent)

        if t_limit_result == inf:
            steps = getattr(self.step_counter, task)
            episodes = getattr(self.episode_counter, task)
            if steps >= step_limit or episodes >= episode_limit:
                log_debug = (
                    task
                    + f": {steps=} >= {step_limit=} or {episodes=} >= {episode_limit=}. Setting time limit for workers to finish to {self.timeout_worker_after_reaching_limit}s"
                )
                self.logger.debug(log_debug)
                t_limit_result = (
                    perf_counter() + self.timeout_worker_after_reaching_limit
                )

        return results_pending, t_limit_result, episode_results

    def _restart_worker_agent(
        self,
        agent: SingleAgentProcess,
        task: str = None,
        step_limit: int = None,
        episode_limit: int = None,
    ):
        log_debug = (
            f"Restarting Agent with {task=}, {step_limit=} and {episode_limit=}."
        )
        self.logger.debug(log_debug)
        agent.close()
        new_agent = self._create_worker_agent(agent.agent_id)
        new_agent.load_state_dicts_network(self.algo.state_dicts_network())
        self.worker[agent.agent_id] = new_agent
        i = agent.agent_id
        seeds = self._eval_seeds[i] if self._eval_seeds is not None else None
        options = self._eval_options[i] if self._eval_options is not None else None
        new_agent.evaluate(
            step_limit=step_limit,
            episode_limit=episode_limit,
            seeds=seeds,
            options=options,
        )
        return new_agent

    def _create_worker_agent(self, i):
        return SingleAgentProcess(
            i,
            self.algo.to_play_only(),
            DummyEnv(),
            deepcopy(self.env_eval),
            None,
            self.worker_device,
            1,
            self.normalize_actions,
            name="worker_" + str(i),
            parent_agent=self,
            step_counter=self.step_counter,
            episode_counter=self.episode_counter,
            nice_level=10,
        )

    def load_checkpoint(self, file_path: str) -> None:
        super().load_checkpoint(file_path)
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

    @staticmethod
    def _split(to_split: List, n_parts: int):
        floor, mod = divmod(len(to_split), n_parts)
        split_lists = [
            to_split[i * floor + min(i, mod) : (i + 1) * floor + min(i + 1, mod)]
            for i in range(n_parts)
        ]
        return split_lists

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
        env_eval: Optional[gym.Env] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo: Algo = confighandler.load_config_dict(cp["algo"])
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_eval = env_eval or eve_cfh.load_config_dict(cp["env_eval"])
        agent = cls(
            algo.to_play_only(),
            env_eval,
            n_worker,
            worker_device,
            normalize_actions,
            timeout_worker_after_reaching_limit,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent


class Synchron(SynchronEvalOnly, Agent):
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
        timeout_worker_after_reaching_limit: float = 90,
    ) -> None:
        self.algo = algo
        self.algo.to(torch.device("cpu"))
        self.env_train = env_train
        self.env_eval = env_eval
        self.worker_device = worker_device
        self.trainer_device = trainer_device
        self.consecutive_action_steps = consecutive_action_steps
        self.normalize_actions = normalize_actions
        self.timeout_worker_after_reaching_limit = timeout_worker_after_reaching_limit

        self.logger = logging.getLogger(self.__module__)
        self.n_worker = n_worker
        self.worker: List[SingleAgentProcess] = []
        self.replay_buffer = replay_buffer
        self.step_counter = StepCounterShared()
        self.episode_counter = EpisodeCounterShared()

        for i in range(n_worker):
            self.worker.append(self._create_worker_agent(i))

        self.trainer = self._create_trainer_agent()

        self.update_error = False

        self._eval_seeds = None
        self._eval_options = None

        self.logger.info("Synchron Agent initialized")

    def heatup(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
        custom_action_low: Optional[List[float]] = None,
        custom_action_high: Optional[List[float]] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        steps_start = self.step_counter.heatup
        episodes_start = self.episode_counter.heatup
        self._log_task(
            "heatup",
            steps,
            step_limit,
            episodes,
            episode_limit,
            custom_action_low=custom_action_low,
            custom_action_high=custom_action_high,
        )
        step_limit, episode_limit = self._log_and_convert_limits(
            "heatup", steps, step_limit, episodes, episode_limit
        )
        for agent in self.worker:
            agent.heatup(
                step_limit=step_limit,
                episode_limit=episode_limit,
                custom_action_low=custom_action_low,
                custom_action_high=custom_action_high,
            )
        result = self._get_worker_results(step_limit, episode_limit, "heatup")
        n_steps = self.step_counter.heatup - steps_start
        n_episodes = self.episode_counter.heatup - episodes_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("heatup", n_steps, t_duration, n_episodes)
        return result

    def explore(
        self,
        *,
        steps: Optional[int] = None,
        episodes: Optional[int] = None,
        step_limit: Optional[int] = None,
        episode_limit: Optional[int] = None,
    ) -> List[Episode]:
        t_start = perf_counter()
        steps_start = self.step_counter.exploration
        episodes_start = self.episode_counter.exploration
        self._log_task("explore", steps, step_limit, episodes, episode_limit)
        step_limit, episode_limit = self._log_and_convert_limits(
            "exploration", steps, step_limit, episodes, episode_limit
        )

        for agent in self.worker:
            agent.explore(step_limit=step_limit, episode_limit=episode_limit)
        result = self._get_worker_results(step_limit, episode_limit, "exploration")

        n_episodes = self.episode_counter.exploration - episodes_start
        n_steps = self.step_counter.exploration - steps_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("exploration", n_steps, t_duration, n_episodes)
        return result

    def update(
        self, *, steps: Optional[int] = None, step_limit: Optional[int] = None
    ) -> List[float]:
        t_start = perf_counter()
        steps_start = self.step_counter.update
        self._log_task("update", steps, step_limit)
        step_limit, _ = self._log_and_convert_limits("update", steps, step_limit)

        self.trainer.update(steps=steps, step_limit=step_limit)
        result = self._get_trainer_results()
        self._update_state_dicts_network()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

        n_steps = self.step_counter.update - steps_start
        t_duration = perf_counter() - t_start
        self._log_task_completion("update", n_steps, t_duration)
        return result

    def explore_and_update_parallel(
        self,
        update_steps: Optional[int] = None,
        update_step_limit: Optional[int] = None,
        explore_steps: Optional[int] = None,
        explore_episodes: Optional[int] = None,
        explore_step_limit: Optional[int] = None,
        explore_episode_limit: Optional[int] = None,
    ) -> Tuple[List[Episode], List[float]]:
        t_start = perf_counter()
        update_steps_start = self.step_counter.update
        explore_steps_start = self.step_counter.exploration
        explore_episodes_start = self.episode_counter.exploration
        self._log_task(
            "explore",
            explore_steps,
            explore_step_limit,
            explore_episodes,
            explore_episode_limit,
        )
        self._log_task("update", update_steps, update_step_limit)
        update_step_limit, _ = self._log_and_convert_limits(
            "update", update_steps, update_step_limit
        )
        self.trainer.update(step_limit=update_step_limit)

        explore_step_limit, explore_episode_limit = self._log_and_convert_limits(
            "exploration",
            explore_steps,
            explore_step_limit,
            explore_episodes,
            explore_episode_limit,
        )
        for agent in self.worker:
            agent.explore(
                step_limit=explore_step_limit, episode_limit=explore_episode_limit
            )

        got_worker_results = False
        got_trainer_results = False
        explore_results = []
        update_result = None
        results_pending = self.worker.copy()
        t_limit_result = inf
        while True:
            if not got_worker_results:
                (
                    results_pending,
                    t_limit_result,
                    explore_results,
                ) = self._worker_result_loop(
                    explore_step_limit,
                    explore_episode_limit,
                    "exploration",
                    explore_results,
                    results_pending,
                    t_limit_result,
                )
                if not results_pending or perf_counter() > t_limit_result:
                    for agent in results_pending:
                        log_warn = f"Restaring Agent {agent.name} because of Timeout"
                        self.logger.warning(log_warn)
                        self._restart_worker_agent(agent)
                    results_pending = []
                    got_worker_results = True
                    n_steps_explore = (
                        self.step_counter.exploration - explore_steps_start
                    )
                    n_episodes_explore = (
                        self.episode_counter.exploration - explore_episodes_start
                    )
                    t_duration_explore = perf_counter() - t_start

            if not got_trainer_results:
                update_result = self._get_trainer_results(0.5)
                if update_result is not None:
                    got_trainer_results = True
                    n_steps_update = self.step_counter.update - update_steps_start
                    t_duration_update = perf_counter() - t_start

            if got_worker_results and got_trainer_results:
                break

        self._log_task_completion(
            "exploration", n_steps_explore, t_duration_explore, n_episodes_explore
        )
        self._log_task_completion("update", n_steps_update, t_duration_update)
        self._update_state_dicts_network()
        self._worker_load_state_dicts_network(self.algo.state_dicts_network())

        return explore_results, update_result

    def close(self):
        for agent in self.worker:
            agent.close()
        self.trainer.close()
        self.replay_buffer.close()

    def _update_state_dicts_network(self):
        state_dicts = self.algo.state_dicts_network()
        self.trainer.state_dicts_network(state_dicts)

    def _restart_worker_agent(
        self,
        agent: SingleAgentProcess,
        task: str = None,
        step_limit: int = None,
        episode_limit: int = None,
    ):
        log_debug = (
            f"Restarting Agent with {task=}, {step_limit=} and {episode_limit=}."
        )
        self.logger.debug(log_debug)
        agent.close()
        new_agent = self._create_worker_agent(agent.agent_id)
        new_agent.load_state_dicts_network(self.algo.state_dicts_network())
        self.worker[agent.agent_id] = new_agent
        if task == "heatup":
            new_agent.heatup(step_limit=step_limit, episode_limit=episode_limit)
        elif task == "exploration":
            new_agent.explore(step_limit=step_limit, episode_limit=episode_limit)
        elif task == "evaluation":
            i = agent.agent_id
            seeds = self._eval_seeds[i] if self._eval_seeds is not None else None
            options = self._eval_options[i] if self._eval_options is not None else None
            new_agent.evaluate(
                step_limit=step_limit,
                episode_limit=episode_limit,
                seeds=seeds,
                options=options,
            )
        return new_agent

    def _get_trainer_results(self, timeout: Optional[float] = None):
        result = self.trainer.get_result(timeout=timeout)
        if isinstance(result, queue.Empty):
            return None
        if isinstance(result, Exception):
            log_warn = f"Restaring Trainer because of Exception {result}"
            self.logger.warning(log_warn)
            self.trainer.close()
            self.trainer = self._create_trainer_agent()
            self.trainer.load_state_dicts_network(self.algo.state_dicts_network())
            self.update_error = True
            return []
        return result

    def _create_worker_agent(self, i):
        return SingleAgentProcess(
            i,
            self.algo.to_play_only(),
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
            nice_level=10,
        )

    def _create_trainer_agent(self):
        return SingleAgentProcess(
            0,
            deepcopy(self.algo),
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
            nice_level=0,
        )

    def load_checkpoint(self, file_path: str) -> None:
        super().load_checkpoint(file_path)
        self.trainer.load_state_dicts_network(self.algo.state_dicts_network())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        n_worker: int,
        worker_device: torch.device = torch.device("cpu"),
        trainer_device: torch.device = torch.device("cpu"),
        consecutive_action_steps: int = 1,
        normalize_actions: bool = True,
        timeout_worker_after_reaching_limit: float = 90,
        env_train: Optional[gym.Env] = None,
        env_eval: Optional[gym.Env] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        cp = torch.load(checkpoint_path)
        confighandler = ConfigHandler()
        algo = confighandler.load_config_dict(cp["algo"])
        replay_buffer = replay_buffer or confighandler.load_config_dict(
            cp["replay_buffer"]
        )
        eve = import_module("eve.util")
        eve_cfh = eve.ConfigHandler()
        env_train = env_train or eve_cfh.load_config_dict(cp["env_train"])
        env_eval = env_eval or eve_cfh.load_config_dict(cp["env_eval"])
        agent = cls(
            algo,
            env_train,
            env_eval,
            replay_buffer,
            n_worker,
            worker_device,
            trainer_device,
            consecutive_action_steps,
            normalize_actions,
            timeout_worker_after_reaching_limit,
        )
        agent.load_checkpoint(checkpoint_path)
        return agent
