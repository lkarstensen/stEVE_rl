from typing import Dict, Generator, Iterator, List, NamedTuple, Optional, Tuple
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
)
from .sac import SAC, SACStateDicts
from .. import network
from ..network import NetworkDummy, Network
import torch.optim as optim
import torch
from dataclasses import dataclass
from copy import deepcopy

from ..environment import ObservationSpace, ActionSpace


class InputEmbedder(NamedTuple):
    embedding_network_name: str
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
    q1_hydra: Dict[str, Dict[str, torch.Tensor]]
    q2_hydra: Dict[str, Dict[str, torch.Tensor]]
    policy_hydra: Dict[str, Dict[str, torch.Tensor]]

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
        iter_list += list(self.q1_hydra.values())
        iter_list += list(self.q2_hydra.values())
        iter_list += list(self.policy_hydra.values())
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
            deepcopy(self.q1_hydra),
            deepcopy(self.q2_hydra),
            deepcopy(self.policy_hydra),
        )


class SACembedder(SAC):
    def __init__(
        self,
        q1: network.QNetwork,
        q2: network.QNetwork,
        policy: network.GaussianPolicy,
        learning_rate: float,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        embedding_networks: Optional[Dict[str, Network]] = None,
        q1_common_input_embedder: Optional[InputEmbedder] = None,
        q2_common_input_embedder: Optional[InputEmbedder] = None,
        policy_common_input_embedder: Optional[InputEmbedder] = None,
        q1_hydra_input_embedder: Optional[Dict[str, InputEmbedder]] = None,
        q2_hydra_input_embedder: Optional[Dict[str, InputEmbedder]] = None,
        policy_hydra_input_embedder: Optional[Dict[str, InputEmbedder]] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.obs_space = obs_space
        self.dict_to_flat_np_map = self.obs_space.dict_to_flat_np_map
        self.action_space = action_space
        self.all_embed_networks = embedding_networks
        self.q1_hydra_input_embedders = q1_hydra_input_embedder or {}
        self.q2_hydra_input_embedders = q2_hydra_input_embedder or {}
        self.policy_hydra_input_embedders = policy_hydra_input_embedder or {}

        self.q1_common_input_embedder = q1_common_input_embedder
        self.q2_common_input_embedder = q2_common_input_embedder
        self.policy_common_input_embedder = policy_common_input_embedder

        self.q1 = q1
        self.q2 = q2
        self.target_q1 = q1.copy()
        self.target_q2 = q2.copy()
        self.policy = policy
        self.log_alpha = torch.zeros(1, requires_grad=True)

        self._q1_hydra_networks: Dict[str, Network] = {}
        self._q2_hydra_networks: Dict[str, Network] = {}
        self._policy_hydra_networks: Dict[str, Network] = {}

        self._init_hydra_network(self.q1_hydra_input_embedders, self._q1_hydra_networks)
        self._init_hydra_network(self.q2_hydra_input_embedders, self._q2_hydra_networks)
        self._init_hydra_network(self.policy_hydra_input_embedders, self._policy_hydra_networks)

        self._q1_common_network = self._init_common_embedder(
            self.q1_common_input_embedder, self._q1_hydra_networks
        )
        self._q2_common_network = self._init_common_embedder(
            self.q2_common_input_embedder, self._q2_hydra_networks
        )
        self._policy_common_network = self._init_common_embedder(
            self.policy_common_input_embedder, self._policy_hydra_networks
        )
        n_actions = 1
        for dim in self.action_space.shape:
            n_actions *= dim
        self.q1.set_input(self._q1_common_network.n_outputs, n_actions)
        self.q2.set_input(self._q2_common_network.n_outputs, n_actions)
        self.target_q1.set_input(self._q1_common_network.n_outputs, n_actions)
        self.target_q2.set_input(self._q2_common_network.n_outputs, n_actions)
        self.policy.set_input(self._policy_common_network.n_outputs)

    def _init_hydra_network(
        self,
        hydra_input_embedder: Dict[str, InputEmbedder],
        hydra_embed_networks: Dict[str, Network],
    ):

        for state_key, input_embedder in hydra_input_embedder.items():
            network_name = input_embedder.embedding_network_name
            network = self.all_embed_networks[network_name]
            ids = self.dict_to_flat_np_map[state_key]
            n_observations = ids[1] - ids[0]
            if network.input_is_set:
                if network.n_inputs != n_observations:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Network {network_name} is wrongly assigned for state {state_key}."
                    )
            else:
                network.set_input(n_observations)
            hydra_embed_networks.update({state_key: network})

        for state_key in self.dict_to_flat_np_map.keys():
            if not state_key in hydra_embed_networks.keys():
                dummy = NetworkDummy()
                hydra_embed_networks.update({state_key: dummy})
                ids = self.dict_to_flat_np_map[state_key]
                n_observations = ids[1] - ids[0]
                dummy.set_input(n_observations)

    def _init_common_embedder(
        self, common_input_embedder: InputEmbedder, hydra_embed_networks: Dict[str, Network]
    ):
        hydra_out = 0
        for network in hydra_embed_networks.values():
            hydra_out += network.n_outputs
        if common_input_embedder is None:
            network = NetworkDummy()
            network.set_input(hydra_out)
        else:
            network_name = self.q1_common_input_embedder.embedding_network_name
            network = self.all_embed_networks[network_name]
            if network.input_is_set:
                if network.n_inputs != hydra_out:
                    raise RuntimeError(
                        f"Input Embedder assignment seems to be wrong. Input Embedders always need the same number of inputs. Common Embedder Network {network_name} is wrongly assigned for q1_common_embedder."
                    )
            else:
                network.set_input(hydra_out)
        return network

    def _init_optimizer(self):
        self.q1_optimizers = self._init_leg_optimizer(
            self._q1_hydra_networks,
            self.q1_hydra_input_embedders,
            self._q1_common_network,
            self.q1_common_input_embedder,
            self.q1,
        )
        self.q2_optimizers = self._init_leg_optimizer(
            self._q2_hydra_networks,
            self.q2_hydra_input_embedders,
            self._q2_common_network,
            self.q2_common_input_embedder,
            self.q2,
        )
        self.policy_optimizers = self._init_leg_optimizer(
            self._policy_hydra_networks,
            self.policy_hydra_input_embedders,
            self._policy_common_network,
            self.policy_common_input_embedder,
            self.policy,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

    def _init_leg_optimizer(
        self, hydra_nets, hydra_embedder, common_net, common_embedder, main_net
    ):
        optimizers = [optim.Adam(main_net.parameters(), lr=self.learning_rate)]
        for state_key, network in hydra_nets.items():
            if isinstance(network, NetworkDummy):
                continue
            if not hydra_embedder[state_key].requires_grad:
                continue
            optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
            optimizers.append(optimizer)

        if not isinstance(common_net, NetworkDummy) and common_embedder.requires_grad:
            optimizer = optim.Adam(self._q1_common_network.parameters(), lr=self.learning_rate)
            optimizers.append(optimizer)
        return optimizers

    def get_play_action(self, flat_state: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            flat_state = torch.from_numpy(flat_state).unsqueeze(0)
            flat_state = flat_state.to(self.device)
            flat_state = pack_sequence([flat_state])
            embedded_state = self._get_embedded_state(
                flat_state,
                self._policy_hydra_networks,
                self.policy_hydra_input_embedders,
                self._policy_common_network,
                self.policy_common_input_embedder,
            )

            mean, log_std = self.policy.forward(embedded_state)

            mean, _ = pad_packed_sequence(mean, batch_first=True)
            log_std, _ = pad_packed_sequence(log_std, batch_first=True)
            std = log_std.exp()

            normal = Normal(mean, std)
            z = normal.sample()
            action = torch.tanh(z)
            action = action.cpu().detach().squeeze(0).squeeze(0).numpy()
            return action

    def get_q_values(
        self,
        state_batch: PackedSequence,
        action_batch: PackedSequence,
    ) -> Tuple[PackedSequence, PackedSequence]:
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q1_hydra_networks,
            self.q1_hydra_input_embedders,
            self._q1_common_network,
            self.q1_common_input_embedder,
        )
        q1 = self.q1(embedded_state, action_batch)
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._q2_hydra_networks,
            self.q2_hydra_input_embedders,
            self._q2_common_network,
            self.q2_common_input_embedder,
        )
        q2 = self.q2(embedded_state, action_batch)
        return q1, q2

    def get_target_q_values(
        self,
        state_batch: PackedSequence,
        action_batch: PackedSequence,
    ) -> Tuple[PackedSequence, PackedSequence]:
        with torch.no_grad():
            self.reset()
            embedded_state = self._get_embedded_state(
                state_batch,
                self._q1_hydra_networks,
                self.q1_hydra_input_embedders,
                self._q1_common_network,
                self.q1_common_input_embedder,
            )
            q1 = self.target_q1(embedded_state, action_batch)
            self.reset()
            embedded_state = self._get_embedded_state(
                state_batch,
                self._q2_hydra_networks,
                self.q2_hydra_input_embedders,
                self._q2_common_network,
                self.q2_common_input_embedder,
            )
            q2 = self.target_q2(embedded_state, action_batch)
            return q1, q2

    # epsilon makes sure that log(0) does not occur
    def get_update_action(
        self, state_batch: PackedSequence, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reset()
        embedded_state = self._get_embedded_state(
            state_batch,
            self._policy_hydra_networks,
            self.policy_hydra_input_embedders,
            self._policy_common_network,
            self.policy_common_input_embedder,
        )
        mean_batch, log_std = self.policy.forward(embedded_state)
        mean_batch, seq_length = pad_packed_sequence(mean_batch, batch_first=True)
        log_std, _ = pad_packed_sequence(log_std, batch_first=True)

        std_batch = log_std.exp()
        normal = Normal(mean_batch, std_batch)
        z = normal.rsample()
        action_batch = torch.tanh(z)

        log_pi_batch = normal.log_prob(z) - torch.log(1 - action_batch.pow(2) + epsilon)
        log_pi_batch = log_pi_batch.sum(-1, keepdim=True)

        action_batch = pack_padded_sequence(
            action_batch, seq_length, batch_first=True, enforce_sorted=False
        )
        log_pi_batch = pack_padded_sequence(
            log_pi_batch, seq_length, batch_first=True, enforce_sorted=False
        )
        return action_batch, log_pi_batch

    def _get_embedded_state(
        self,
        state_batch: PackedSequence,
        hydra_nets: Dict[str, Network],
        hydra_embedder: Dict[str, InputEmbedder],
        common_net: Network,
        common_embedder: InputEmbedder,
    ):
        unpacked_state, seq_lengths = pad_packed_sequence(state_batch, batch_first=True)
        embedded_state = None
        for state_key, network in hydra_nets.items():
            ids = self.dict_to_flat_np_map[state_key]
            reduced_state = unpacked_state[:, :, ids[0] : ids[1]]
            reduced_state = pack_padded_sequence(
                reduced_state, seq_lengths, batch_first=True, enforce_sorted=False
            )
            if state_key in hydra_embedder.keys() and hydra_embedder[state_key].requires_grad:
                reduced_output = network.forward(reduced_state)
            else:
                with torch.no_grad():
                    reduced_output = network.forward(reduced_state)
            reduced_output, _ = pad_packed_sequence(reduced_output, batch_first=True)
            if embedded_state is None:
                embedded_state = reduced_output
            else:
                embedded_state = torch.cat([embedded_state, reduced_output], dim=-1)

        embedded_state = pack_padded_sequence(
            embedded_state, seq_lengths, batch_first=True, enforce_sorted=False
        )
        if common_embedder is not None and common_embedder.requires_grad:
            embedded_state = common_net.forward(embedded_state)
        else:
            with torch.no_grad():
                embedded_state = common_net.forward(embedded_state)
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
        self._q1_common_network.to(device)
        self._q2_common_network.to(device)
        self._policy_common_network.to(device)
        for net in self._q1_hydra_networks.values():
            net.to(device)
        for net in self._q2_hydra_networks.values():
            net.to(device)
        for net in self._policy_hydra_networks.values():
            net.to(device)

        self.log_alpha = self.log_alpha.detach().to(device=device).requires_grad_()
        self._init_optimizer()

    def update_target_q(self, tau):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def copy(self):
        if self.all_embed_networks is not None:
            embed_copy = {name: net.copy() for name, net in self.all_embed_networks.items()}
        else:
            embed_copy = None

        copy = self.__class__(
            self.q1.copy(),
            self.q2.copy(),
            self.policy.copy(),
            self.learning_rate,
            self.obs_space,
            self.action_space,
            embed_copy,
            self.q1_common_input_embedder,
            self.q2_common_input_embedder,
            self.policy_common_input_embedder,
            self.q1_hydra_input_embedders,
            self.q2_hydra_input_embedders,
            self.policy_hydra_input_embedders,
        )

        return copy

    def copy_shared_memory(self):
        copy = self.copy()

        self.q1.share_memory()
        self.q2.share_memory()
        self.target_q1.share_memory()
        self.target_q2.share_memory()
        self.policy.share_memory()
        if self.all_embed_networks is not None:
            for net in self.all_embed_networks.values():
                net.share_memory()

        copy.q1 = self.q1
        copy.q2 = self.q2
        copy.target_q1 = self.target_q1
        copy.target_q2 = self.target_q2
        copy.policy = self.policy
        if self.all_embed_networks is not None:
            for name, net in self.all_embed_networks.items():
                copy.all_embed_networks[name] = net
        return copy

    def load_state_dicts(self, state_dicts: SACEmbeddedStateDicts):
        self.q1.load_state_dict(state_dicts.q1)
        self.q2.load_state_dict(state_dicts.q2)
        self.target_q1.load_state_dict(state_dicts.target_q1)
        self.target_q2.load_state_dict(state_dicts.target_q2)
        self._q1_common_network.load_state_dict(state_dicts.q1_common)
        self._q2_common_network.load_state_dict(state_dicts.q2_common)
        self._policy_common_network.load_state_dict(state_dicts.policy_common)
        for state, net in self._q1_hydra_networks.items():
            net.load_state_dict(state_dicts.q1_hydra[state])
        for state, net in self._q2_hydra_networks.items():
            net.load_state_dict(state_dicts.q2_hydra[state])
        for state, net in self._policy_hydra_networks.items():
            net.load_state_dict(state_dicts.policy_hydra[state])

    @property
    def state_dicts(self) -> SACEmbeddedStateDicts:
        q1_hydra = {state: net.state_dict() for state, net in self._q1_hydra_networks.items()}
        q2_hydra = {state: net.state_dict() for state, net in self._q2_hydra_networks.items()}
        policy_hydra = {
            state: net.state_dict() for state, net in self._policy_hydra_networks.items()
        }

        state_dicts = SACEmbeddedStateDicts(
            self.q1.state_dict(),
            self.q2.state_dict(),
            self.target_q1.state_dict(),
            self.target_q2.state_dict(),
            self.policy.state_dict(),
            self._q1_common_network.state_dict(),
            self._q2_common_network.state_dict(),
            self._policy_common_network.state_dict(),
            q1_hydra,
            q2_hydra,
            policy_hydra,
        )
        return state_dicts

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
            self._q1_common_network,
            self._q2_common_network,
            self._policy_common_network,
        ]
        return iter(
            nets
            + list(self._q1_hydra_networks.values())
            + list(self._q2_hydra_networks.values())
            + list(self._policy_hydra_networks.values())
        )
