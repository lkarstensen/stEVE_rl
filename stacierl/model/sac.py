from typing import List, Optional, Tuple
import numpy as np

from torch.distributions.normal import Normal
from .model import Model
from .. import network
import torch.optim as optim
import torch


class SAC(Model):
    def __init__(
        self,
        q_net_1: network.QNetwork,
        q_net_2: network.QNetwork,
        target_q_net_2: network.QNetwork,
        target_q_net_1: network.QNetwork,
        policy_net: network.GaussianPolicy,
        learning_rate: float = 0.0007525,
    ) -> None:
        self.q_net_1 = q_net_1
        self.q_net_2 = q_net_2
        self.policy_net = policy_net
        self.target_q_net_1 = target_q_net_1
        self.target_q_net_2 = target_q_net_2
        self.learning_rate = learning_rate
        self._init_optimizer(learning_rate)
        self._init_alpha(learning_rate)

    def _init_optimizer(self, learning_rate):
        self.q1_optimizer = optim.Adam(self.q_net_1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q_net_2.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate / 2)

    def _init_alpha(self, learning_rate):
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_rate)

    def get_action(
        self, flat_state: np.ndarray, hidden_state: Optional[torch.tensor] = None
    ) -> Tuple[np.ndarray, Optional[torch.tensor]]:
        with torch.no_grad():
            flat_state = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)

            mean, log_std, hidden_state_out = self.policy_net(flat_state, hidden_state)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).numpy()
            return action, hidden_state_out

    @property
    def initial_hidden_state(self) -> Optional[torch.Tensor]:
        return None

    def to(self, device: torch.device):
        self.device = device
        self.q_net_1.to(device)
        self.q_net_2.to(device)
        self.target_q_net_1.to(device)
        self.target_q_net_2.to(device)
        self.policy_net.to(device)
        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        # torch.ones(1, requires_grad=True, device=device.type)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def copy(self):
        copy = self.__class__(
            self.q_net_1.copy(),
            self.target_q_net_1.copy(),
            self.q_net_2.copy(),
            self.target_q_net_2.copy(),
            self.policy_net.copy(),
            self.learning_rate,
        )

        return copy

    def copy_shared_memory(self):
        self.q_net_1.share_memory()
        self.target_q_net_1.share_memory()
        self.q_net_2.share_memory()
        self.target_q_net_2.share_memory()
        self.policy_net.share_memory()
        copy = self.__class__(
            self.q_net_1,
            self.target_q_net_1,
            self.q_net_2,
            self.target_q_net_2,
            self.policy_net,
            self.learning_rate,
            device=self.device,
        )

        return copy

    def all_state_dicts(self):
        all_state_dicts = {
            "q_net_1": self.q_net_1.state_dict(),
            "q_net_2": self.q_net_2.state_dict(),
            "target_q_net_1": self.target_q_net_1.state_dict(),
            "target_q_net_2": self.target_q_net_2.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }
        return all_state_dicts

    def all_parameters(self):
        all_parameters = {
            "q_net_1": self.q_net_1.parameters(),
            "q_net_2": self.q_net_2.parameters(),
            "target_q_net_1": self.target_q_net_1.parameters(),
            "target_q_net_2": self.target_q_net_2.parameters(),
            "policy_net": self.policy_net.parameters(),
        }
        return all_parameters

    def load_all_state_dicts(self, all_state_dicts: dict):
        self.q_net_1.load_state_dict(all_state_dicts["q_net_1"])
        self.q_net_2.load_state_dict(all_state_dicts["q_net_2"])
        self.target_q_net_1.load_state_dict(all_state_dicts["target_q_net_1"])
        self.target_q_net_2.load_state_dict(all_state_dicts["target_q_net_2"])
        self.policy_net.load_state_dict(all_state_dicts["policy_net"])

    def soft_tau_update_all(self, all_parameters: dict, tau: float):

        for net, net_key in zip(
            [self.q_net_1, self.q_net_2, self.target_q_net_1, self.target_q_net_2, self.policy_net],
            ["q_net_1", "q_net_2", "target_q_net_1", "target_q_net_2", "policy_net"],
        ):
            for self_param, extern_param in zip(net.parameters(), all_parameters[net_key]):
                self_param.data.copy_(tau * extern_param + (1 - tau) * self_param)
