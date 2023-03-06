from copy import deepcopy
from typing import Any, Dict, Iterator, Tuple
import numpy as np
from torch.distributions.normal import Normal
import torch
import torch.optim as optim

from .sacmodel import SACModel
from ... import network
from ...optimizer import Optimizer


class Vanilla(SACModel):
    def __init__(
        self,
        lr_alpha: float,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        q1_optimizer: Optimizer,
        q2_optimizer: Optimizer,
        policy_optimizer: Optimizer,
        q1_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        q2_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        policy_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        self.lr_alpha = lr_alpha

        self.q1 = q1
        self.q2 = q2
        self.policy = policy

        self.q1_optimizer = q1_optimizer
        self.q2_optimizer = q2_optimizer
        self.policy_optimizer = policy_optimizer

        self.q1_scheduler = q1_scheduler
        self.q2_scheduler = q2_scheduler
        self.policy_scheduler = policy_scheduler

        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def get_play_action(
        self, flat_state: np.ndarray = None, evaluation=False
    ) -> np.ndarray:
        with torch.no_grad():
            flat_state = (
                torch.as_tensor(
                    flat_state, dtype=torch.float32, device=self.policy.device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            mean, log_std = self.policy.forward(flat_state)
            std = log_std.exp()

            if evaluation:
                action = torch.tanh(mean)
                rescaled_action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                return rescaled_action
            else:
                normal = Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
                action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                return action

    def get_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        q1 = self.q1.forward(states, actions)
        self.reset()
        q2 = self.q2.forward(states, actions)
        return q1, q2

    def get_target_q_values(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.reset()
            q1 = self.target_q1.forward(states, actions)
            self.reset()
            q2 = self.target_q2.forward(states, actions)
            return q1, q2

    # epsilon makes sure that log(0) does not occur
    def get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        mean_batch, log_std = self.policy.forward(state_batch)
        std_batch = log_std.exp()

        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        # log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
        #        torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)

        return action_batch, log_pi_batch

    def q1_update_zero_grad(self):
        self.q1_optimizer.zero_grad()

    def q2_update_zero_grad(self):
        self.q2_optimizer.zero_grad()

    def policy_update_zero_grad(self):
        self.policy_optimizer.zero_grad()

    def alpha_update_zero_grad(self):
        self.alpha_optimizer.zero_grad()

    def q1_update_step(self):
        self.q1_optimizer.step()

    def q2_update_step(self):
        self.q2_optimizer.step()

    def policy_update_step(self):
        self.policy_optimizer.step()

    def alpha_update_step(self):
        self.alpha_optimizer.step()

    def q1_scheduler_step(self):
        if self.q1_scheduler is not None:
            self.q1_scheduler.step()

    def q2_scheduler_step(self):
        if self.q2_scheduler is not None:
            self.q2_scheduler.step()

    def policy_scheduler_step(self):
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()

    def to(self, device: torch.device):
        super().to(device)
        self.q1.to(device)
        self.target_q1.to(device)
        self.q1_optimizer.param_groups = []
        self.q1_optimizer.add_param_group({"params": self.q1.parameters()})

        self.q2.to(device)
        self.target_q2.to(device)
        self.q2_optimizer.param_groups = []
        self.q2_optimizer.add_param_group({"params": self.q2.parameters()})

        self.policy.to(device)
        self.policy_optimizer.param_groups = []
        self.policy_optimizer.add_param_group({"params": self.policy.parameters()})

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self.alpha_optimizer.param_groups = []
        self.alpha_optimizer.add_param_group({"params": [self.log_alpha]})

    def update_target_q(self, tau):
        for target_param, param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def reset(self) -> None:
        for net in self:
            net.reset()

    def __iter__(self) -> Iterator[network.Network]:
        return iter([self.q1, self.q2, self.target_q1, self.target_q2, self.policy])

    def close(self):
        del self.q1
        del self.q1_optimizer
        del self.q2
        del self.q2_optimizer
        del self.policy
        del self.policy_optimizer
        del self.alpha_optimizer

    def state_dicts_network(self, destination: Dict[str, Any] = None) -> Dict[str, Any]:

        ret = state_dicts = {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "policy": self.policy.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }

        if destination is not None:

            for net in ["q1", "q2", "target_q1", "target_q2", "policy"]:
                state_dict = state_dicts[net]
                dest = destination[net]

                for tensor, dest_tensor in zip(state_dict.values(), dest.values()):
                    dest_tensor.copy_(tensor)

            destination["log_alpha"].copy_(state_dicts["log_alpha"])
            ret = destination

        return ret

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        self.q1.load_state_dict(state_dicts["q1"])
        self.q2.load_state_dict(state_dicts["q2"])

        self.target_q1.load_state_dict(state_dicts["target_q1"])
        self.target_q2.load_state_dict(state_dicts["target_q2"])

        self.policy.load_state_dict(state_dicts["policy"])

        self.log_alpha.data.copy_(state_dicts["log_alpha"])

    # def set_network_states(self, network_states_container: SACNetworkStateContainer):
    #     self.q1.load_state_dict(network_states_container.q1)
    #     self.q2.load_state_dict(network_states_container.q2)
    #     self.target_q1.load_state_dict(network_states_container.target_q1)
    #     self.target_q2.load_state_dict(network_states_container.target_q2)
    #     self.policy.load_state_dict(network_states_container.policy)
    #     self.log_alpha.data.copy_(network_states_container.log_alpha["log_alpha"])

    # @property
    # def network_states_container(self) -> SACNetworkStateContainer:
    #     network_states_container = SACNetworkStateContainer(
    #         self.q1.state_dict(),
    #         self.q2.state_dict(),
    #         self.target_q1.state_dict(),
    #         self.target_q2.state_dict(),
    #         self.policy.state_dict(),
    #         {"log_alpha": self.log_alpha.detach()},
    #     )
    #     return network_states_container

    # @property
    # def optimizer_states_container(self) -> SACOptimizerStateContainer:
    #     optimizer_states_container = SACOptimizerStateContainer(
    #         self.q1_optimizer.state_dict(),
    #         self.q2_optimizer.state_dict(),
    #         self.policy_optimizer.state_dict(),
    #         self.alpha_optimizer.state_dict(),
    #     )

    #     return optimizer_states_container

    # def set_optimizer_states(
    #     self, optimizer_states_container: SACOptimizerStateContainer
    # ):
    #     self.q1_optimizer.load_state_dict(optimizer_states_container.q1)
    #     self.q2_optimizer.load_state_dict(optimizer_states_container.q2)
    #     self.policy_optimizer.load_state_dict(optimizer_states_container.policy)
    #     self.alpha_optimizer.load_state_dict(optimizer_states_container.alpha)
