from abc import ABC, abstractmethod


class StacieRLUserObject(ABC):
    @abstractmethod
    def copy(self):
        ...
