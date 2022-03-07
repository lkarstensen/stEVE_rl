from abc import ABC

from ..replaybuffer_db import ReplayBufferDB


class Wrapper(ReplayBufferDB, ABC):
    ...
