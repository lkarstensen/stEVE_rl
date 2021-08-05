from typing import Tuple
from stacierl.replaybuffer import replaybuffer
from .agent import Agent, dict_state_to_flat_np_state
from ..algo import Algo
from ..replaybuffer import ReplayBuffer, Episode
from ..environment import EnvFactory, Environment
import torch


class Single(Agent):
    def __init__(
        self,
        algo: Algo,
        env: Environment,
        replay_buffer: ReplayBuffer,
        consecutive_action_steps: int = 1,
        device:torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.algo = algo
        self.env = env
        self.replay_buffer = replay_buffer
        self.consecutive_action_steps = consecutive_action_steps

        self.explore_step_counter = 0
        self.update_step_counter = 0
        self.eval_step_counter = 0
        self.explore_episode_counter = 0
        self.eval_episode_counter = 0
        self.algo.to(device)

    def _heatup(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        step_counter = 0
        episode_counter = 0
        rewards = []
        successes = []
        while step_counter < steps and episode_counter < episodes:
            (
                episode_transitions,
                explored_steps,
                episode_reward,
                episode_success,
            ) = self._play_episode(
                consecutive_actions=self.consecutive_action_steps, mode="exploration"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            step_counter += explored_steps
            episode_counter += 1
            self.replay_buffer.push(episode_transitions)

        average_reward = sum(rewards) / len(rewards)
        average_success = sum(successes) / len(successes)

        return average_reward, average_success

    def _explore(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:

        step_limit = self.explore_step_counter + steps
        episode_limit = self.explore_episode_counter + episodes
        rewards = []
        successes = []
        while (
            self.explore_step_counter < step_limit and self.explore_episode_counter < episode_limit
        ):
            (
                episode_transitions,
                explored_steps,
                episode_reward,
                episode_success,
            ) = self._play_episode(
                consecutive_actions=self.consecutive_action_steps, mode="exploration"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            self.explore_step_counter += explored_steps
            self.explore_episode_counter += 1
            self.replay_buffer.push(episode_transitions)

        average_reward = sum(rewards) / len(rewards)
        average_success = sum(successes) / len(successes)

        return average_reward, average_success

    def _update(self, steps, batch_size) -> None:
        if len(self.replay_buffer) < batch_size:
            return

        for _ in range(steps):
            batch = self.replay_buffer.sample(batch_size)
            self.algo.update(batch)
            self.update_step_counter += 1

    def _evaluate(self, steps: int = None, episodes: int = None) -> Tuple[float, float]:
        step_limit = self.eval_step_counter + steps
        episode_limit = self.eval_episode_counter + episodes
        rewards = []
        successes = []
        while self.eval_step_counter < step_limit and self.eval_episode_counter < episode_limit:
            _, steps, episode_reward, episode_success = self._play_episode(
                consecutive_actions=1, mode="eval"
            )
            successes.append(episode_success)
            rewards.append(episode_reward)
            self.eval_step_counter += steps
            self.eval_episode_counter += 1

        average_reward = sum(rewards) / len(rewards)
        average_success = sum(successes) / len(successes)

        return average_reward, average_success

    def _play_episode(self, consecutive_actions: int, mode: str):
        assert mode in ["exploration", "eval"]
        episode_transitions = Episode()
        state = self.env.reset()
        state = dict_state_to_flat_np_state(state)
        episode_reward = 0
        step_counter = 0
        hidden_state = self.algo.get_initial_hidden_state()
        while True:
            if mode == "exploration":
                action, hidden_next_state = self.algo.get_exploration_action(state, hidden_state)
            else:
                action, hidden_next_state = self.algo.get_eval_action(state, hidden_state)
            for _ in range(consecutive_actions):
                next_state, reward, done, info, success = self.env.step(action)
                next_state = dict_state_to_flat_np_state(next_state)
                self.env.render()
                episode_transitions.add_transition(
                    state, action, reward, next_state, done, hidden_state
                )
                state = next_state
                hidden_state = hidden_next_state
                episode_reward += reward
                step_counter += 1
                if done:
                    break
            if done:
                break
        return episode_transitions, step_counter, episode_reward, success
