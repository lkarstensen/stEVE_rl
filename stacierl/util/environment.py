from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional
import numpy as np
from . import StacieRLUserObject


class ActionSpace(ABC):
    @property
    @abstractmethod
    def shape(self) -> Tuple[float]:
        ...

    @property
    @abstractmethod
    def low(self) -> Tuple[float]:
        ...

    @property
    @abstractmethod
    def high(self) -> Tuple[float]:
        ...


class ObservationSpace(ABC):
    @property
    @abstractmethod
    def shape(self) -> Dict[str, Tuple[float]]:
        ...

    @property
    @abstractmethod
    def low(self) -> Dict[str, np.ndarray]:
        ...

    @property
    @abstractmethod
    def high(self) -> Dict[str, np.ndarray]:
        ...

    @property
    @abstractmethod
    def keys(self) -> List[str]:
        ...

    @property
    def dict_to_flat_np_map(self) -> Dict[str, Tuple[int, int]]:
        map = {}
        id = 0
        for key in self.keys:
            key_shape = self.shape[key]
            n_values = 1
            for dim in key_shape:
                n_values *= dim
            map.update({key: (id, id + n_values)})
            id += n_values
        return map

    def to_flat_array(self, state: Dict[str, np.ndarray]) -> np.ndarray:

        flat_state = np.array([], dtype=np.float32)
        for key in self.keys:
            new_state = state[key]
            new_state = new_state.reshape((-1,))
            flat_state = np.append(flat_state, new_state)
        return flat_state


class Environment(StacieRLUserObject, ABC):
    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict[str, Any]]]:
        ...

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        ...

    @abstractmethod
    def render(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> ObservationSpace:
        ...


class DummyEnvironment(Environment):
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Optional[Dict[str, Any]]]:
        ...

    def reset(self) -> Dict[str, np.ndarray]:
        ...

    def render(self) -> None:
        ...

    def close(self) -> None:
        ...

    @property
    def action_space(self) -> ActionSpace:
        return DummyActionSpace()

    @property
    def observation_space(self) -> ObservationSpace:
        return DummyObservationSpace()


class DummyActionSpace(ActionSpace):
    def __init__(self) -> None:
        ...

    def shape(self):
        return ()

    def low(self):
        return ()

    def high(self):
        return ()


class DummyObservationSpace(ObservationSpace):
    def __init__(self) -> None:
        ...

    @property
    def shape(self) -> Dict[str, Tuple[float]]:
        return {}

    @property
    def low(self) -> Dict[str, np.ndarray]:
        return {}

    @property
    def high(self) -> Dict[str, np.ndarray]:
        return {}

    @property
    def keys(self) -> List[str]:
        return []
