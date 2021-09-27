from abc import ABC, abstractmethod


class StacieRLUserObject:
    @abstractmethod
    def copy(self):
        ...
