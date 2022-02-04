from typing import Dict, Iterator, Optional, Tuple
import numpy as np
from torch.distributions.normal import Normal
from .vanilla import Vanilla, SACStateDicts
from ... import network
from ...network import NetworkDummy, Network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy

from ...util import ObservationSpace, ActionSpace


@dataclass
class Embedder:
    network: Network
    update: bool


@dataclass
class HydraNetwork:
    network: Network
    split_state_id: int
    requires_grad: bool


@dataclass
class SACEmbeddedStateDicts(SACStateDicts):
    q1: Dict[str, torch.Tensor]
    q2: Dict[str, torch.Tensor]
    target_q1: Dict[str, torch.Tensor]
    target_q2: Dict[str, torch.Tensor]
    policy: Dict[str, torch.Tensor]
    q1_common: Dict[str, torch.Tensor]
    q2_common: Dict[str, torch.Tensor]
    policy_common: Dict[str, torch.Tensor]

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
        ]
        return iter(iter_list)

    def copy(self):
        return SACEmbeddedStateDicts(
            deepcopy(self.q1),
            deepcopy(self.q2),
            deepcopy(self.target_q1),
            deepcopy(self.target_q2),
            deepcopy(self.policy),
            deepcopy(self.q1_common),
            deepcopy(self.q2_common),
            deepcopy(self.policy_common),
        )


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

        self.dict_to_flat_np_map = self.obs_space.dict_to_flat_np_map

        self.q1_common_input_embedder = self._init_common_embedder(self.q1_common_input_embedder)
        self.q2_common_input_embedder = self._init_common_embedder(self.q2_common_input_embedder)
        self.policy_common_input_embedder = self._init_common_embedder(self.policy_common_input_embedder)
        
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

        #self._init_optimizer()

    def _init_common_embedder(
        self, common_input_embedder: Embedder
    ):
        hydra_out = 18

        if common_input_embedder is None:
            network = NetworkDummy()
            network.set_input(hydra_out)
            update = False
        else:
            network = common_input_embedder.network
            update = common_input_embedder.update
            if network.input_is_set:
                if network.n_inputs != hydra_out:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Common Embedder Network {network} is wrongly assigned for q1_common_embedder."
                    )
            else:
                network.set_input(hydra_out)

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
        #self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.q1(embedded_state, action_batch, use_hidden_state=False)
        #self.reset()
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
        #self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self.q1_common_input_embedder,
            use_hidden_state=False,
        )
        q1 = self.target_q1(embedded_state, action_batch, use_hidden_state=False)
        #self.reset()
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
        #self.reset()
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
                torch.log(1 - action_batch.pow(2) + epsilon), dim=-1, keepdim=True)
                
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
        ...

    def copy_shared_memory(self):
        ...

    def load_state_dicts(self, state_dicts: SACEmbeddedStateDicts):
        ...

    @property
    def state_dicts(self) -> SACEmbeddedStateDicts:
        ...

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
