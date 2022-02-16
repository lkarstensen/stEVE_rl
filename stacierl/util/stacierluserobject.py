from abc import ABC, abstractmethod
import numpy as np
from torch import device 
import inspect


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
        print(attributes_dict["class"])
        repr_attributes = self._get_repr_attributes(self.__init__)
        print(repr_attributes)
        # intermediate solution
        if 'args' and 'kwargs' in repr_attributes:
            repr_attributes.remove('args')
            repr_attributes.remove('kwargs')
            
        for attribute in repr_attributes:
            value = getattr(self, attribute)

            # hasattr ist slower, but more general
            #if not isinstance(value, (int, float, list, str, np.number)):
            if hasattr(value, 'to_dict'):
                value = value.to_dict()

            if (isinstance(value, np.integer)):
                value = int(value)

            if isinstance(value, device):
                value = str(value)

            attributes_dict[attribute] = value
        return attributes_dict    

