from typing import List, Optional
from math import inf
import csv
import logging
import os
from ..util import EveRLObject
from ..agent.agent import Agent, StepCounter, EpisodeCounter


class Runner(EveRLObject):
    def __init__(
        self,
        agent: Agent,
        heatup_action_low: List[float],
        heatup_action_high: List[float],
        agent_parameter_for_result_file: dict,
        checkpoint_folder: str,
        results_file: str,
        info_results: Optional[List[str]] = None,
    ) -> None:
        self.agent = agent
        self.heatup_action_low = heatup_action_low
        self.heatup_action_high = heatup_action_high
        self.agent_parameter_dict = agent_parameter_for_result_file
        self.checkpoint_folder = checkpoint_folder
        self.results_file = results_file
        self.info_results = info_results or []
        self.logger = logging.getLogger(self.__module__)

        self._results = {
            "episodes explore": 0,
            "steps explore": 0,
        }
        for info_result in self.info_results:
            self._results[info_result] = 0.0
        self._results["reward"] = 0.0
        self._results["best success"] = 0.0
        self._results["best explore steps"] = 0.0

        with open(results_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(agent_parameter_for_result_file.keys())
            writer.writerow(agent_parameter_for_result_file.values())
            writer.writerow([])
            writer.writerow(self._results.keys())

        self.best_eval = {"steps": 0, "success": -inf}

    @property
    def step_counter(self) -> StepCounter:
        return self.agent.step_counter

    @property
    def episode_counter(self) -> EpisodeCounter:
        return self.agent.episode_counter

    def heatup(self, steps: int):
        self.agent.heatup(
            steps=steps,
            custom_action_low=self.heatup_action_low,
            custom_action_high=self.heatup_action_high,
        )

    def explore(self, n_episodes: int):
        self.agent.explore(episodes=n_episodes)

    def update(self, n_steps: int):
        self.agent.update(n_steps)

    def eval(
        self, *, episodes: Optional[int] = None, seeds: Optional[List[int]] = None
    ):
        explore_steps = self.step_counter.exploration
        checkpoint_file = os.path.join(
            self.checkpoint_folder, f"checkpoint{explore_steps}"
        )
        self.agent.save_checkpoint(checkpoint_file)
        episodes = self.agent.evaluate(episodes=episodes, seeds=seeds)

        self._results["episodes explore"] = self.episode_counter.exploration
        self._results["steps explore"] = explore_steps
        successes = [episode.infos[-1]["success"] for episode in episodes]
        success = sum(successes) / len(successes)

        for info_result in self.info_results:
            result = [episode.infos[-1][info_result] for episode in episodes]
            result = sum(result) / len(result)
            self._results[info_result] = round(result, 3)

        rewards = [episode.episode_reward for episode in episodes]
        reward = sum(rewards) / len(rewards)
        self._results["reward"] = round(reward, 3)

        if success > self.best_eval["success"]:
            checkpoint_file = os.path.join(self.checkpoint_folder, "checkpoint_best")
            self.agent.save_checkpoint(checkpoint_file)
            self.best_eval["success"] = success
            self.best_eval["steps"] = explore_steps

        self._results["best success"] = self.best_eval["success"]
        self._results["best explore steps"] = self.best_eval["steps"]

        log_info = (
            f"Success: {success}, Reward: {reward}, Exploration steps: {explore_steps}"
        )
        self.logger.info(log_info)
        with open(self.results_file, "a+", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(self._results.values())

        return success, reward

    def explore_and_update(
        self,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        *,
        explore_steps: int = None,
        explore_steps_limit: int = None,
    ):
        if explore_steps is not None and explore_steps_limit is not None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit should be given. Not both."
            )
        if explore_steps is None and explore_steps_limit is None:
            raise ValueError(
                "Either explore_steps ors explore_steps_limit needs to be given. Not both."
            )

        explore_steps_limit = (
            explore_steps_limit or self.step_counter.exploration + explore_steps
        )
        while self.step_counter.exploration < explore_steps_limit:
            update_steps = (
                self.step_counter.exploration * update_steps_per_explore_step
                - self.step_counter.update
            )
            self.agent.explore_and_update(
                explore_episodes=explore_episodes_between_updates,
                update_steps=update_steps,
            )

    def training_run(
        self,
        heatup_steps: int,
        training_steps: int,
        explore_steps_between_eval: int,
        explore_episodes_between_updates: int,
        update_steps_per_explore_step: float,
        eval_episodes: Optional[int] = None,
        eval_seeds: Optional[List[int]] = None,
    ):
        self.heatup(heatup_steps)
        next_eval_step_limt = explore_steps_between_eval
        while self.agent.step_counter.exploration < training_steps:
            self.explore_and_update(
                explore_episodes_between_updates,
                update_steps_per_explore_step,
                explore_steps_limit=next_eval_step_limt,
            )
            self.eval(episodes=eval_episodes, seeds=eval_seeds)
            next_eval_step_limt += explore_steps_between_eval

        reward, success = self.eval(episodes=eval_episodes, seeds=eval_seeds)
        return reward, success
