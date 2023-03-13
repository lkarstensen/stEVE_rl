from copy import deepcopy
from typing import Any, Dict, Iterator
from torch import optim
import torch
from .model import Model
from ... import network
from ...optimizer import Optimizer


class SACModel(Model):
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
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    # epsilon makes sure that log(0) does not occur

    def to(self, device: torch.device):
        super().to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)

        for net, optimizer in zip(
            [self.q1, self.q2, self.policy],
            [self.q1_optimizer, self.q2_optimizer, self.policy_optimizer],
        ):

            old_params = net.parameters()
            net.to(device)
            new_params = net.parameters()

            for param_group in optimizer.param_groups:
                optim_ids = [id(param) for param in param_group["params"]]
                for old_param, new_param in zip(old_params, new_params):
                    if id(old_param) in optim_ids:
                        idx = optim_ids.index(id(old_param))
                        param_group["params"][idx] = new_param

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self.alpha_optimizer.param_groups[0]["params"] = [self.log_alpha]

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

    def copy_play_only(self):
        return SACModelPolicyOnly(deepcopy(self.policy))

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


class SACModelPolicyOnly(Model):
    def __init__(
        self,
        policy: network.GaussianPolicy,
    ) -> None:
        self.policy = policy

    def to(self, device: torch.device):
        super().to(device)
        self.policy.to(device)

    def state_dicts_network(self, destination: Dict[str, Any] = None) -> Dict[str, Any]:

        ret = state_dicts = {
            "q1": None,
            "q2": None,
            "target_q1": None,
            "target_q2": None,
            "policy": self.policy.state_dict(),
            "log_alpha": None,
        }

        if destination is not None:

            for net in ["policy"]:
                state_dict = state_dicts[net]
                dest = destination[net]

                for tensor, dest_tensor in zip(state_dict.values(), dest.values()):
                    dest_tensor.copy_(tensor)
            ret = destination

        return ret

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        self.policy.load_state_dict(state_dicts["policy"])

    def reset(self) -> None:
        self.policy.reset()

    def close(self):
        del self.policy

    def copy_play_only(self):
        return deepcopy(self)
