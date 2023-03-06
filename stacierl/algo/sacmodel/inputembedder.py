from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple
import numpy as np

from torch.distributions.normal import Normal
import torch


from .vanilla import Vanilla
from ... import network
from ...network import NetworkDummy, Network
from ...optimizer import Optimizer


@dataclass
class Embedder:
    network: Network
    update: bool
    optimizer: Optimizer = None
    scheduler: torch.optim.lr_scheduler._LRScheduler = None

    def __post_init__(self):
        if self.update and self.optimizer is None:
            raise ValueError("If Embedder should update Network, it needs an optimizer")

    def copy(self):
        ...


class InputEmbedding(Vanilla):
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
        q1_common_input_embedder: Optional[Embedder] = None,
        q2_common_input_embedder: Optional[Embedder] = None,
        policy_common_input_embedder: Optional[Embedder] = None,
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
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.q1_common_input_embedder = q1_common_input_embedder
        self.q2_common_input_embedder = q2_common_input_embedder
        self.policy_common_input_embedder = policy_common_input_embedder

        self.q1_common_input_embedder = self._init_common_embedder(
            self.q1_common_input_embedder
        )
        self.q2_common_input_embedder = self._init_common_embedder(
            self.q2_common_input_embedder
        )
        self.policy_common_input_embedder = self._init_common_embedder(
            self.policy_common_input_embedder
        )

    def _init_common_embedder(self, common_input_embedder: Embedder):

        if common_input_embedder is None:
            network = NetworkDummy(self.policy.n_observations)
            update = False
            embedder = Embedder(network, update)
        else:

            embedder = common_input_embedder

        return embedder

    def get_play_action(
        self, flat_state: np.ndarray = None, evaluation=False
    ) -> np.ndarray:
        with torch.no_grad():
            flat_state = torch.from_numpy(flat_state).unsqueeze(0).unsqueeze(0)
            flat_state = flat_state.to(self.policy.device)
            embedded_state = self._get_embedded_state(
                flat_state,
                self.policy_common_input_embedder,
                use_hidden_state=True,
            )
            mean, log_std = self.policy.forward(embedded_state, use_hidden_state=True)
            std = log_std.exp()

            if evaluation:
                action = torch.tanh(mean)
                action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                return action
            else:
                normal = Normal(mean, std)
                z = normal.sample()
                action = torch.tanh(z)
                action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
                return action

    def get_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.q1(embedded_state, action_batch, use_hidden_state=False)
        # self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q2_common_input_embedder,
            use_hidden_state=False,
        )
        q2 = self.q2(embedded_state, action_batch, use_hidden_state=False)
        return q1, q2

    def get_target_q_values(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # with torch.no_grad():
        # self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.target_q1(embedded_state, action_batch, use_hidden_state=False)
        # self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q2_common_input_embedder,
            use_hidden_state=False,
        )
        q2 = self.target_q2(embedded_state, action_batch, use_hidden_state=False)
        return q1, q2

    # epsilon makes sure that log(0) does not occur
    def get_update_action(
        self, state_batch: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.policy_common_input_embedder,
            use_hidden_state=False,
        )
        mean_batch, log_std = self.policy.forward(
            embedded_state, use_hidden_state=False
        )

        std_batch = log_std.exp()
        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = torch.sum(normal.log_prob(z), dim=-1, keepdim=True) - torch.sum(
            torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True
        )

        return action_batch, log_pi_batch

    def _get_embedded_state(
        self,
        state_batch: torch.Tensor,
        common_embedder: Embedder,
        use_hidden_state: bool,
    ):
        hydra_state = state_batch

        if common_embedder is None:
            embedded_state = hydra_state
        else:
            if common_embedder.update:
                embedded_state = common_embedder.network.forward(
                    hydra_state, use_hidden_state=use_hidden_state
                )
            else:
                with torch.no_grad():
                    embedded_state = common_embedder.network.forward(
                        hydra_state, use_hidden_state=use_hidden_state
                    )
        return embedded_state

    def q1_update_zero_grad(self):
        self.q1_optimizer.zero_grad()
        if self.q1_common_input_embedder.update:
            self.q1_common_input_embedder.optimizer.zero_grad()

    def q2_update_zero_grad(self):
        self.q2_optimizer.zero_grad()
        if self.q2_common_input_embedder.update:
            self.q2_common_input_embedder.optimizer.zero_grad()

    def policy_update_zero_grad(self):
        self.policy_optimizer.zero_grad()
        if self.policy_common_input_embedder.update:
            self.policy_common_input_embedder.optimizer.zero_grad()

    def alpha_update_zero_grad(self):
        self.alpha_optimizer.zero_grad()

    def q1_update_step(self):
        self.q1_optimizer.step()
        if self.q1_common_input_embedder.update:
            self.q1_common_input_embedder.optimizer.step()

    def q2_update_step(self):
        self.q2_optimizer.step()
        if self.q2_common_input_embedder.update:
            self.q2_common_input_embedder.optimizer.step()

    def policy_update_step(self):
        self.policy_optimizer.step()
        if self.policy_common_input_embedder.update:
            self.policy_common_input_embedder.optimizer.step()

    def alpha_update_step(self):
        self.alpha_optimizer.step()

    def alpha_update(self, loss: torch.Tensor):
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

    def q1_scheduler_step(self):
        if self.q1_scheduler is not None:
            self.q1_scheduler.step()
        if self.q1_common_input_embedder.scheduler is not None:
            self.q1_common_input_embedder.scheduler.step()

    def q2_scheduler_step(self):
        if self.q2_scheduler is not None:
            self.q2_scheduler.step()
        if self.q2_common_input_embedder.scheduler is not None:
            self.q2_common_input_embedder.scheduler.step()

    def policy_scheduler_step(self):
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()
        if self.policy_common_input_embedder.scheduler is not None:
            self.policy_common_input_embedder.scheduler.step()

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

        self.q1_common_input_embedder.network.to(device)
        if self.q1_common_input_embedder.update:
            self.q1_common_input_embedder.optimizer.param_groups = []
            self.q1_common_input_embedder.optimizer.add_param_group(
                {"params": self.q1_common_input_embedder.network.parameters()}
            )

        self.q2_common_input_embedder.network.to(device)
        if self.q2_common_input_embedder.update:
            self.q2_common_input_embedder.optimizer.param_groups = []
            self.q2_common_input_embedder.optimizer.add_param_group(
                {"params": self.q2_common_input_embedder.network.parameters()}
            )

        self.policy_common_input_embedder.network.to(device)
        if self.policy_common_input_embedder.update:
            self.policy_common_input_embedder.optimizer.param_groups = []
            self.policy_common_input_embedder.optimizer.add_param_group(
                {"params": self.policy_common_input_embedder.network.parameters()}
            )

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

    def __iter__(self) -> Iterator[Network]:
        nets = [
            self.q1,
            self.q2,
            self.target_q1,
            self.target_q2,
            self.policy,
            self.q1_common_input_embedder.network,
            self.q2_common_input_embedder.network,
            self.policy_common_input_embedder.network,
        ]
        return iter(nets)

    def close(self):
        del self.q1
        del self.q1_optimizer
        del self.q2
        del self.q2_optimizer
        del self.policy
        del self.policy_optimizer
        del self.q1_common_input_embedder
        del self.q2_common_input_embedder
        del self.policy_common_input_embedder
        del self.alpha_optimizer

    def state_dicts_network(self, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        ret = state_dicts = {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "policy": self.policy.state_dict(),
            "q1_common": self.q1_common_input_embedder.network.state_dict(),
            "q2_common": self.q2_common_input_embedder.network.state_dict(),
            "policy_common": self.policy_common_input_embedder.network.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }

        if destination is not None:

            for net in [
                "q1",
                "q2",
                "target_q1",
                "target_q2",
                "policy",
                "q1_common",
                "q2_common",
                "policy_common",
            ]:
                state_dict = state_dicts[net]
                dest = destination[net]

                for tensor, dest_tensor in zip(state_dict.values(), dest.values()):
                    dest_tensor.copy_(tensor)

            destination["log_alpha"].copy_(state_dicts["log_alpha"])
            ret = destination

        return ret

    def load_state_dicts_network(self, state_dicts: Dict[str, Any]) -> None:
        self.q1.load_state_dict(state_dicts["q1"])
        self.q1_common_input_embedder.network.load_state_dict(state_dicts["q1_common"])

        self.q2.load_state_dict(state_dicts["q2"])
        self.q2_common_input_embedder.network.load_state_dict(state_dicts["q2_common"])

        self.target_q1.load_state_dict(state_dicts["target_q1"])
        self.target_q2.load_state_dict(state_dicts["target_q2"])

        self.policy.load_state_dict(state_dicts["policy"])
        self.policy_common_input_embedder.network.load_state_dict(
            state_dicts["policy_common"]
        )

        self.log_alpha.data.copy_(state_dicts["log_alpha"])

    # @property
    # def optimizer_states_container(self) -> SACEmbeddedOptimizerStateContainer:

    #     q1 = self.q1_optimizer.state_dict()
    #     if self.q1_common_input_embedder.update:
    #         q1_common = self.q1_common_input_embedder.optimizer.state_dict()
    #     else:
    #         q1_common = None

    #     q2 = self.q2_optimizer.state_dict()
    #     if self.q2_common_input_embedder.update:
    #         q2_common = self.q2_common_input_embedder.optimizer.state_dict()
    #     else:
    #         q2_common = None

    #     policy = self.policy_optimizer.state_dict()
    #     if self.policy_common_input_embedder.update:
    #         policy_common = self.policy_common_input_embedder.optimizer.state_dict()
    #     else:
    #         policy_common = None

    #     optimizer_states_container = SACEmbeddedOptimizerStateContainer(
    #         q1,
    #         q2,
    #         policy,
    #         q1_common,
    #         q2_common,
    #         policy_common,
    #         self.alpha_optimizer.state_dict(),
    #     )

    #     return optimizer_states_container

    # def set_optimizer_states(
    #     self, optimizer_states_container: SACEmbeddedOptimizerStateContainer
    # ):
    #     self.q1_optimizer.load_state_dict(optimizer_states_container.q1)
    #     if optimizer_states_container.q1_common is not None:
    #         self.q1_common_input_embedder.optimizer.load_state_dict(
    #             optimizer_states_container.q1_common
    #         )

    #     self.q2_optimizer.load_state_dict(optimizer_states_container.q2)
    #     if optimizer_states_container.q2_common is not None:
    #         self.q2_common_input_embedder.optimizer.load_state_dict(
    #             optimizer_states_container.q2_common
    #         )

    #     self.policy_optimizer.load_state_dict(optimizer_states_container.policy)
    #     if optimizer_states_container.policy_common is not None:
    #         self.policy_common_input_embedder.optimizer.load_state_dict(
    #             optimizer_states_container.policy_common
    #         )

    #     self.alpha_optimizer.load_state_dict(optimizer_states_container.alpha)

    # @property
    # def scheduler_states_container(self) -> SACEmbeddedSchedulerStateContainer:
    #     if self.q1_scheduler is not None:
    #         q1 = self.q1_scheduler.state_dict()
    #     else:
    #         q1 = None

    #     if self.q1_common_input_embedder.scheduler is not None:
    #         q1_common = self.q1_common_input_embedder.scheduler.state_dict()
    #     else:
    #         q1_common = None

    #     if self.q2_scheduler is not None:
    #         q2 = self.q2_scheduler.state_dict()
    #     else:
    #         q2 = None

    #     if self.q2_common_input_embedder.scheduler is not None:
    #         q2_common = self.q2_common_input_embedder.scheduler.state_dict()
    #     else:
    #         q2_common = None

    #     if self.policy_scheduler is not None:
    #         policy = self.policy_scheduler.state_dict()
    #     else:
    #         policy = None

    #     if self.policy_common_input_embedder.scheduler is not None:
    #         policy_common = self.policy_common_input_embedder.scheduler.state_dict()
    #     else:
    #         policy_common = None

    #     scheduler_states_container = SACEmbeddedSchedulerStateContainer(
    #         q1,
    #         q2,
    #         policy,
    #         q1_common,
    #         q2_common,
    #         policy_common,
    #     )

    #     return scheduler_states_container

    # def set_scheduler_states(
    #     self, scheduler_states_container: SACEmbeddedSchedulerStateContainer
    # ):
    #     if scheduler_states_container.q1 is not None:
    #         self.q1_scheduler.load_state_dict(scheduler_states_container.q1)
    #     if scheduler_states_container.q1_common is not None:
    #         self.q1_common_input_embedder.scheduler.load_state_dict(
    #             scheduler_states_container.q1_common
    #         )

    #     if scheduler_states_container.q2 is not None:
    #         self.q2_scheduler.load_state_dict(scheduler_states_container.q2)
    #     if scheduler_states_container.q2_common is not None:
    #         self.q2_common_input_embedder.scheduler.load_state_dict(
    #             scheduler_states_container.q2_common
    #         )
    #     if scheduler_states_container.policy is not None:
    #         self.policy_scheduler.load_state_dict(scheduler_states_container.policy)
    #     if scheduler_states_container.policy_common is not None:
    #         self.policy_common_input_embedder.scheduler.load_state_dict(
    #             scheduler_states_container.policy_common
    #         )
