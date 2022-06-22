from abc import ABC
from ..replaybuffer import ReplayBuffer
from enum import Enum
from dataclasses import dataclass
from typing import Any

class FilterMethod(str, Enum):
    EXACT = '=='
    GREATEREQUAL = '>='
    LESSEQUAL = '<='
    NOTEQUAL = '!='

@dataclass
class FilterElement:
    path: str
    value: Any
    method: FilterMethod

class Wrapper(ReplayBuffer, ABC):
    ...
