import logging
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import PackedSequence
from .algo import Algo, ModelStateDicts
from .. import model
import numpy as np
from ..replaybuffer import Batch
from ..environment import ActionSpace


class SAC(Algo):
    def __init__(
        self,
        model: model.SAC,
        action_space: ActionSpace,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_scaling: float = 1,
        action_scaling: float = 1,
        exploration_action_noise: float = 0.2,
    ):
        self.logger = logging.getLogger(self.__module__)
        # HYPERPARAMETERS
        self.action_space = action_space
        self.gamma = gamma
        self.tau = tau
        self.exploration_action_noise = exploration_action_noise
        # Model
        self._model = model

        # REST
        self.reward_scaling = reward_scaling
        self.action_scaling = action_scaling
        self._device = torch.device("cpu")
        self.update_step = 0

        # ENTROPY TEMPERATURE
        self.alpha = torch.zeros(1)
        n_actions = 1
        for dim in action_space.shape:
            n_actions *= dim

        self.target_entropy = -torch.ones(1) * n_actions

    @property
    def model(self) -> model.SAC:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def get_exploration_action(self, flat_state: np.ndarray) -> np.ndarray:

        action = self.get_eval_action(flat_state)
        action += np.random.normal(0, self.exploration_action_noise)
        return action

    def get_eval_action(self, flat_state: np.ndarray) -> np.ndarray:
        action = self.model.get_play_action(flat_state)
        return action * self.action_scaling

    def update(self, batch: Batch) -> List[float]:

        (states, actions, rewards, next_states, dones) = batch
        # actions /= self.action_scaling
        if isinstance(rewards, PackedSequence):
            rewards = rewards.data
        if isinstance(dones, PackedSequence):
            dones = dones.data

        states = states.to(dtype=torch.float32, device=self._device)
        actions = actions.to(dtype=torch.float32, device=self._device)
        rewards = rewards.to(dtype=torch.float32, device=self._device)
        next_states = next_states.to(dtype=torch.float32, device=self._device)
        dones = dones.to(dtype=torch.float32, device=self._device)

        next_actions, next_log_pi = self.model.get_update_action(next_states)
        next_q1, next_q2 = self.model.get_target_q_values(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = (
            rewards + (1 - dones) * self.gamma * next_q_target
        )  # self.reward_scaling * rewards

        # Q LOSS
        curr_q1, curr_q2 = self.model.get_q_values(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.model.update_target_q(self.tau)

        # Policy loss
        new_actions, log_pi = self.model.get_update_action(states)
        q1, q2 = self.model.get_q_values(states, new_actions)
        min_q = torch.min(q1, q2)
        policy_loss = (self.alpha * log_pi - min_q).mean()

        alpha_loss = (self.model.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.model.q1_update_zero_grad()
        self.model.q2_update_zero_grad()
        self.model.policy_update_zero_grad()
        self.model.alpha_update_zero_grad()
        q1_loss.backward(retain_graph=True)
        q2_loss.backward(retain_graph=True)
        policy_loss.backward(retain_graph=True)
        alpha_loss.backward()
        self.model.q1_update_step()
        self.model.q2_update_step()
        self.model.policy_update_step()
        self.model.alpha_update_step()

        self.alpha = self.model.log_alpha.exp()

        self.update_step += 1
        return [
            q1_loss.detach().cpu().numpy(),
            q2_loss.detach().cpu().numpy(),
            policy_loss.detach().cpu().numpy(),
        ]

    # def save_model(self, path: str):
    #     print("... saving model ...")
    #     torch.save(self.model.q_net_1.state_dict(), path + "_q_net1")
    #     torch.save(self.model.q_net_2.state_dict(), path + "_q_net2")

    #     torch.save(self.model.target_q_net_1.state_dict(), path + "_target_q_net1")
    #     torch.save(self.model.target_q_net_2.state_dict(), path + "_target_q_net2")

    #     torch.save(self.model.policy_net.state_dict(), path + "_policy_net")

    #     torch.save(self.alpha, path + "_alpha.pt")

    # def load_model(self, path: str):
    #     print("... loading model ...")
    #     self.model.q_net_1.load_state_dict(torch.load(path + "_q_net1"))
    #     self.model.q_net_2.load_state_dict(torch.load(path + "_q_net2"))

    #     self.model.target_q_net_1.load_state_dict(torch.load(path + "_target_q_net1"))
    #     self.model.target_q_net_2.load_state_dict(torch.load(path + "_target_q_net2"))

    #     self.model.policy_net.load_state_dict(torch.load(path + "_policy_net"))

    #     self.alpha = torch.load(path + "_alpha.pt")

    #     self.model.q_net_1.eval()
    #     self.model.q_net_2.eval()
    #     self.model.target_q_net_1.eval()
    #     self.model.target_q_net_2.eval()
    #     self.model.policy_net.eval()

    def copy(self):
        copy = self.__class__(
            self.model.copy(),
            self.action_space,
            self.gamma,
            self.tau,
            self.reward_scaling,
            self.action_scaling,
            self.exploration_action_noise,
        )
        return copy

    def copy_shared_memory(self):
        copy = self.__class__(
            self.model.copy_shared_memory(),
            self.action_space,
            self.gamma,
            self.tau,
            self.reward_scaling,
            self.action_scaling,
            self.exploration_action_noise,
        )
        return copy

    def to(self, device: torch.device):
        self._device = device
        self.alpha = self.alpha.to(device)
        self.target_entropy = self.target_entropy.to(device)
        self.model.to(device)

    @property
    def state_dicts(self) -> ModelStateDicts:
        return self.model.state_dicts

    def load_state_dicts(self, state_dicts: ModelStateDicts) -> None:
        self.model.load_state_dicts(state_dicts)

    def reset(self) -> None:
        self.model.reset()
