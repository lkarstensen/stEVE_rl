from abc import ABC, abstractmethod

class ConfigHandler(ABC):
    
    @abstractmethod
    def load_config_data(self, file):
        ...

    @abstractmethod
    def save_dict_to_file(self, dictionary: dict):
        ...
