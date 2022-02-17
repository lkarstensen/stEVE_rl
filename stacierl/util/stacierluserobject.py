from abc import ABC, abstractmethod
import json
from attr import attr
import numpy as np
from torch import device 
import inspect
from enum import Enum

import stacierl
#from stacierl.util.environment import ActionSpace, ObservationSpace


class StacieRLUserObject(ABC):
    @abstractmethod
    def copy(self):
        ...
        
    def _get_repr_attributes(self, init_function):
        repr_attributes = []
        kwargs = inspect.signature(init_function)
        for param in kwargs.parameters.values():
            repr_attributes.append(param.name)

        return repr_attributes
        
    def to_dict(self):
        attributes_dict = {}
        attributes_dict["class"] = f"{self.__module__}.{self.__class__.__name__}"
        repr_attributes = self._get_repr_attributes(self.__init__)
    
        if 'args' and 'kwargs' in repr_attributes:
            repr_attributes.remove('args')
            repr_attributes.remove('kwargs')
            
        for attribute in repr_attributes:
            value = getattr(self, attribute)

            # hasattr ist slower, but more general
            #if not isinstance(value, (int, float, list, str, np.number)):
            if hasattr(value, 'to_dict'):
                dict_value = value.to_dict()

            elif (isinstance(value, np.integer)):
                dict_value = int(value)

            elif isinstance(value, device):
                dict_value = str(value)
                
            elif isinstance(value, Enum):
                dict_value = value.value
                
            elif isinstance(value, np.ndarray):
                dict_value = tuple(value)  

            elif isinstance(value, list):
                dict_value = []
                for v in value:
                    if hasattr(v, 'to_dict'):
                        dict_value.append(v.to_dict())
                    else:
                        dict_value.append(v)

            elif 'Tiltmaze' in str(value):
                dict_value = str(type(value))                
            
            else:
                dict_value = value
                
            attributes_dict[attribute] = dict_value
        return attributes_dict 
    
    def save_config(self, directory: str, file_name: str):
        config_dict = self.to_dict()
        json_handler = stacierl.util.JSONHandler()
        
        path = directory + '/' + file_name + '.json'
        
        json_handler.save_dict_to_file(config_dict, path) 

