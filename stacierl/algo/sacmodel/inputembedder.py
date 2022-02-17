from abc import ABC, abstractmethod
from typing import Dict, Iterator, Optional, Tuple
import numpy as np
from torch.distributions.normal import Normal

from stacierl.util.stacierluserobject import StacieRLUserObject


from .vanilla import Vanilla, NetworkStatesContainer, OptimizerStatesContainer
from ... import network
from ...network import NetworkDummy, Network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy

from ...util import ObservationSpace, ActionSpace


@dataclass
class Embedder(StacieRLUserObject):
    network: Network
    update: bool

    def copy(self):
        ...
@dataclass
class SACEmbeddedNetworkStateContainer(NetworkStatesContainer):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    target_q1: Dict[str, torch.Tensor]
    target_q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    q1_common: Dict[str, torch.Tensor]
    q2_common: Dict[str, torch.Tensor]
    policy_common: Dict[str, torch.Tensor]
    log_alpha: Dict[str, torch.Tensor]

    def __iter__(self):
        iter_list = [
            self.q1,
            self.q2,
            self.target_q1,
            self.target_q2,
            self.policy,
            self.q1_common,
            self.q2_common,
            self.policy_common,
            self.log_alpha,
        ]
        return iter(iter_list)

    def copy(self):
        return SACEmbeddedNetworkStateContainer(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.target_q1),
            deepcopy(self.target_q2),
            deepcopy(self.policy),
            deepcopy(self.q1_common),
            deepcopy(self.q2_common),
            deepcopy(self.policy_common),
            deepcopy(self.log_alpha),
        )

    def to_dict(self) -> Dict:
        model_state_dict = {
            "q1": self.q1,
            "q2": self.q2,
            "target_q1": self.target_q1,
            "target_q2": self.target_q2,
            "policy": self.policy,
            "q1_common": self.q1_common,
            "q2_common": self.q2_common,
            "policy_common": self.policy_common,
            "log_alpha": self.log_alpha,
        }

        return model_state_dict

    def from_dict(self, model_state_dict: Dict):
        self.q1 = model_state_dict["q1"]
        self.q2 = model_state_dict["q2"]
        self.target_q1 = model_state_dict["target_q1"]
        self.target_q2 = model_state_dict["target_q2"]
        self.policy = model_state_dict["policy"]
        self.q1_common = model_state_dict["q1_common"]
        self.q2_common = model_state_dict["q2_common"]
        self.policy_common = model_state_dict["policy_common"]
        self.log_alpha = model_state_dict["log_alpha"]


@dataclass
class SACEmbeddedOptimizerStateContainer(OptimizerStatesContainer):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    q1_common: Dict[str, torch.Tensor] or None
    q2_common: Dict[str, torch.Tensor] or None
    policy_common: Dict[str, torch.Tensor] or None
    alpha: Dict[str, torch.Tensor]

    def __iter__(self):
        iter_list = [
            self.q1,
            self.q2,
            self.policy,
            self.alpha,
        ]
        if self.q1_common is not None:
            iter_list.append(self.q1_common)
        if self.q2_common is not None:
            iter_list.append(self.q2_common)
        if self.policy_common is not None:
            iter_list.append(self.policy_common)
        return iter(iter_list)

    def copy(self):
        return SACEmbeddedOptimizerStateContainer(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.policy),
            deepcopy(self.q1_common),
            deepcopy(self.q2_common),
            deepcopy(self.policy_common),
            deepcopy(self.alpha),
        )

    def to_dict(self) -> Dict:
        model_state_dict = {
            "q1": self.q1,
            "q2": self.q2,
            "policy": self.policy,
            "q1_common": self.q1_common,
            "q2_common": self.q2_common,
            "policy_common": self.policy_common,
            "alpha": self.alpha,
        }

        return model_state_dict

    def from_dict(self, optimizer_state_dict: Dict):
        self.q1 = optimizer_state_dict["q1"]
        self.q2 = optimizer_state_dict["q2"]
        self.policy = optimizer_state_dict["policy"]
        self.q1_common = optimizer_state_dict["q1_common"]
        self.q2_common = optimizer_state_dict["q2_common"]
        self.policy_common = optimizer_state_dict["policy_common"]
        self.alpha = optimizer_state_dict["alpha"]


class InputEmbedding(Vanilla):
    def __init__(
        self,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        learning_rate: float,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        q1_common_input_embedder: Optional[Embedder] = None,
        q2_common_input_embedder: Optional[Embedder] = None,
        policy_common_input_embedder: Optional[Embedder] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.obs_space = obs_space
        self.action_space = action_space

        self.q1_common_input_embedder = q1_common_input_embedder
        self.q2_common_input_embedder = q2_common_input_embedder
        self.policy_common_input_embedder = policy_common_input_embedder

        self.q1 = q1
        self.q2 = q2
        self.target_q1 = q1.copy()
        self.target_q2 = q2.copy()
        self.policy = policy
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self.q1_common_input_embedder = self._init_common_embedder(self.q1_common_input_embedder)
        self.q2_common_input_embedder = self._init_common_embedder(self.q2_common_input_embedder)
        self.policy_common_input_embedder = self._init_common_embedder(
            self.policy_common_input_embedder
        )

        n_actions = 1
        for dim in self.action_space.shape:
            n_actions *= dim

        self.q1.set_input(self.q1_common_input_embedder.network.n_outputs, n_actions)
        self.q2.set_input(self.q2_common_input_embedder.network.n_outputs, n_actions)
        self.target_q1.set_input(self.q1_common_input_embedder.network.n_outputs, n_actions)
        self.target_q2.set_input(self.q2_common_input_embedder.network.n_outputs, n_actions)
        self.policy.set_input(self.policy_common_input_embedder.network.n_outputs)
        self.policy.set_output(n_actions)

        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(param)

        self._init_optimizer()

    def _init_common_embedder(self, common_input_embedder: Embedder):
        n_observations = 0
        for observation in self.obs_space.low.values():
            n_observations += observation.size

        if common_input_embedder is None:
            network = NetworkDummy()
            network.set_input(n_observations)
            update = False
        else:
            network = common_input_embedder.network
            update = common_input_embedder.update
            if network.input_is_set:
                if network.n_inputs != n_observations:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Common Embedder Network {network} is wrongly assigned for q1_common_embedder."
                    )
            else:
                network.set_input(n_observations)

        return Embedder(network, update)

    def _init_optimizer(self):
        self.q1_optimizers = self._init_leg_optimizer(
            self.q1_common_input_embedder,
            self.q1,
        )
        self.q2_optimizers = self._init_leg_optimizer(
            self.q2_common_input_embedder,
            self.q2,
        )
        self.policy_optimizers = self._init_leg_optimizer(
            self.policy_common_input_embedder,
            self.policy,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def _init_leg_optimizer(
        self,
        common_embedder: Embedder,
        main_net: Network,
    ):
        optimizers = [optim.Adam(main_net.parameters(), lr=self.learning_rate)]

        if isinstance(common_embedder.network, NetworkDummy):
            common_embedder = None
        else:
            if common_embedder.update:
                optimizer = optim.Adam(common_embedder.network.parameters(), lr=self.learning_rate)
                optimizers.append(optimizer)
        return optimizers

    def get_play_action(self, flat_state: np.ndarray = None, evaluation=False) -> np.ndarray:
        with torch.no_grad():
            flat_state = torch.from_numpy(flat_state).unsqueeze(0).unsqueeze(0)
            flat_state = flat_state.to(self.device)
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
        mean_batch, log_std = self.policy.forward(embedded_state, use_hidden_state=False)

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
        for optimizer in self.q1_optimizers:
            optimizer.zero_grad()

    def q2_update_zero_grad(self):
        for optimizer in self.q2_optimizers:
            optimizer.zero_grad()

    def policy_update_zero_grad(self):
        for optimizer in self.policy_optimizers:
            optimizer.zero_grad()

    def alpha_update_zero_grad(self):
        self.alpha_optimizer.zero_grad()

    def q1_update_step(self):
        for optimizer in self.q1_optimizers:
            optimizer.step()

    def q2_update_step(self):
        for optimizer in self.q2_optimizers:
            optimizer.step()

    def policy_update_step(self):
        for optimizer in self.policy_optimizers:
            optimizer.step()

    def alpha_update_step(self):
        self.alpha_optimizer.step()

    def alpha_update(self, loss: torch.Tensor):
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()

    def to(self, device: torch.device):
        self.device = device
        self.q1.to(device)
        self.q2.to(device)
        self.target_q1.to(device)
        self.target_q2.to(device)
        self.policy.to(device)
        self.q1_common_input_embedder.network.to(device)
        self.q2_common_input_embedder.network.to(device)
        self.policy_common_input_embedder.network.to(device)

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self._init_optimizer()

    def update_target_q(self, tau):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def copy(self):
        q1_embed_network_current = self.q1_common_input_embedder.network
        q1_embed_update = self.q1_common_input_embedder.update
        q2_embed_network_current = self.q2_common_input_embedder.network
        q2_embed_update = self.q2_common_input_embedder.update
        policy_embed_network_current = self.policy_common_input_embedder.network
        policy_embed_update = self.policy_common_input_embedder.update
        q1_embed_network_copy = q1_embed_network_current.copy()
        if q2_embed_network_current is q1_embed_network_current:
            q2_embed_network_copy = q1_embed_network_copy
        else:
            q2_embed_network_copy = q2_embed_network_current.copy()

        if policy_embed_network_current is q1_embed_network_current:
            policy_embed_network_copy = q1_embed_network_copy
        elif policy_embed_network_current is q2_embed_network_current:
            policy_embed_network_copy = q2_embed_network_copy
        else:
            policy_embed_network_copy = policy_embed_network_current.copy()

        copy = self.__class__(
            self.q1.copy(),
            self.q2.copy(),
            self.policy.copy(),
            self.learning_rate,
            self.obs_space,
            self.action_space,
            Embedder(q1_embed_network_copy, q1_embed_update),
            Embedder(q2_embed_network_copy, q2_embed_update),
            Embedder(policy_embed_network_copy, policy_embed_update),
        )
        return copy

    def copy_shared_memory(self):
        self.q1.share_memory()
        self.q2.share_memory()
        self.target_q1.share_memory()
        self.target_q2.share_memory()
        self.policy.share_memory()
        self.q1_common_input_embedder.network.share_memory()
        self.q2_common_input_embedder.network.share_memory()
        self.policy_common_input_embedder.network.share_memory()

        copy = self.__class__(
            self.q1,
            self.q2,
            self.policy,
            self.learning_rate,
            self.obs_space,
            self.action_space,
            self.q1_common_input_embedder,
            self.q2_common_input_embedder,
            self.policy_common_input_embedder,
        )
        copy.target_q1 = self.target_q1
        copy.target_q2 = self.target_q2

        return copy

    def set_network_states(self, network_states_container: SACEmbeddedNetworkStateContainer):
        self.q1.load_state_dict(network_states_container.q1)
        self.q1_common_input_embedder.network.load_state_dict(network_states_container.q1_common)

        self.q2.load_state_dict(network_states_container.q2)
        self.q2_common_input_embedder.network.load_state_dict(network_states_container.q2_common)

        self.target_q1.load_state_dict(network_states_container.target_q1)
        self.target_q2.load_state_dict(network_states_container.target_q2)

        self.policy.load_state_dict(network_states_container.policy)
        self.policy_common_input_embedder.network.load_state_dict(
            network_states_container.policy_common
        )

        self.log_alpha.data.copy_(network_states_container.log_alpha["log_alpha"])

    @property
    def network_states_container(self) -> SACEmbeddedNetworkStateContainer:
        network_states_container = SACEmbeddedNetworkStateContainer(
            self.q1.state_dict(),
            self.q2.state_dict(),
            self.target_q1.state_dict(),
            self.target_q2.state_dict(),
            self.policy.state_dict(),
            self.q1_common_input_embedder.network.state_dict(),
            self.q2_common_input_embedder.network.state_dict(),
            self.policy_common_input_embedder.network.state_dict(),
            {"log_alpha": self.log_alpha.detach()},
        )
        return network_states_container

    @property
    def optimizer_states_container(self) -> SACEmbeddedOptimizerStateContainer:

        q1 = self.q1_optimizers[0].state_dict()
        if len(self.q1_optimizers) > 1:
            q1_common = self.q1_optimizers[1].state_dict()
        else:
            q1_common = None

        q2 = self.q2_optimizers[0].state_dict()
        if len(self.q2_optimizers) > 1:
            q2_common = self.q2_optimizers[1].state_dict()
        else:
            q2_common = None

        policy = self.policy_optimizers[0].state_dict()

        if len(self.policy_optimizers) > 1:
            policy_common = self.policy_optimizers[1].state_dict()
        else:
            policy_common = None

        optimizer_states_container = SACEmbeddedOptimizerStateContainer(
            q1, q2, policy, q1_common, q2_common, policy_common, self.alpha_optimizer.state_dict()
        )

        return optimizer_states_container

    def set_optimizer_states(self, optimizer_states_container: SACEmbeddedOptimizerStateContainer):
        self.q1_optimizers[0].load_state_dict(optimizer_states_container.q1)
        if optimizer_states_container.q1_common:
            self.q1_optimizers[1].load_state_dict(optimizer_states_container.q1_common)

        self.q2_optimizers[0].load_state_dict(optimizer_states_container.q2)
        if optimizer_states_container.q2_common:
            self.q2_optimizers[1].load_state_dict(optimizer_states_container.q2_common)

        self.policy_optimizers[0].load_state_dict(optimizer_states_container.policy)
        if optimizer_states_container.policy_common:
            self.policy_optimizers[1].load_state_dict(optimizer_states_container.policy_common)

        self.alpha_optimizer.load_state_dict(optimizer_states_container.alpha)

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
