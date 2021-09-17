from . import Environment, ActionSpace, ObservationSpace
from typing import Any, Tuple, Dict, Optional
import numpy as np


class DummyyEnv(Environment):
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
