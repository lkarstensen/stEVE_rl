from abc import ABC
from ..replaybuffer_db import ReplayBufferDB
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

class Wrapper(ReplayBufferDB, ABC):
    ...
