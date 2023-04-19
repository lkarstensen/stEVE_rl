from typing import Any, List, Tuple
from enum import Enum
from importlib import import_module
import inspect

import numpy as np
import eve
import torch
import yaml


class ConfigHandler:
    def __init__(self):
        self.object_registry = {}

    def save_config(self, stacierl_object: Any, file: str) -> None:
        obj_dict = self.object_to_config_dict(stacierl_object)
        self.save_config_dict(obj_dict, file)

    def load_config(self, file: str) -> Any:
        obj_dict = self.load_config_dict(file)
        obj = self.config_dict_to_object(obj_dict)
        return obj

    def object_to_config_dict(self, stacierl_object: Any) -> dict:
        self.object_registry = {}
        config_dict = self._everl_obj_to_dict(stacierl_object)
        self.object_registry = {}
        return config_dict

    def config_dict_to_object(
        self,
        config_dict: dict,
        object_registry: dict = None,
        class_str_replace: List[Tuple[str, str]] = None,
    ) -> Any:
        class_str_replace = class_str_replace or []
        self.object_registry = object_registry or {}
        obj = self._dict_to_obj(config_dict, class_str_replace)
        self.object_registry = {}
        return obj

    def load_config_dict(self, file: str) -> dict:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(file, "r", encoding="utf-8") as config:
            config_dict = yaml.load(config, Loader=Loader)
        return config_dict

    def save_config_dict(self, config_dict: dict, file: str) -> None:
        if not file.endswith(".yml"):
            file += ".yml"
        with open(file, "w", encoding="utf-8") as dumpfile:
            yaml.dump(
                config_dict,
                dumpfile,
                default_flow_style=False,
                sort_keys=False,
                indent=4,
            )

    def _everl_obj_to_dict(self, everl_object) -> dict:
        attributes_dict = {}
        attributes_dict[
            "class"
        ] = f"{everl_object.__module__}.{everl_object.__class__.__name__}"
        attributes_dict["_id"] = id(everl_object)
        if id(everl_object) in self.object_registry:
            return attributes_dict
        init_attributes = self._get_init_attributes(everl_object.__init__)

        if "args" in init_attributes:
            init_attributes.remove("args")
        if "kwargs" in init_attributes:
            init_attributes.remove("kwargs")
        if "kwds" in init_attributes:
            init_attributes.remove("kwds")

        for attribute in init_attributes:
            nested_object = getattr(everl_object, attribute)
            attributes_dict[attribute] = self._obj_to_native_datatypes(nested_object)
        self.object_registry[id(everl_object)] = attributes_dict
        return attributes_dict

    def _obj_to_native_datatypes(self, obj) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, torch.device):  # pylint: disable=no-member
            return str(obj)

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, eve.Env):
            return f"{obj.__module__}.{obj.__class__.__name__}"

        if isinstance(obj, list):
            return [self._obj_to_native_datatypes(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._obj_to_native_datatypes(v) for v in obj)

        if isinstance(obj, dict):
            return {k: self._obj_to_native_datatypes(v) for k, v in obj.items()}

        if hasattr(obj, "__module__"):
            search_string = obj.__module__ + str(type(obj).__bases__)

            if "everl." in search_string:
                return self._everl_obj_to_dict(obj)

            if isinstance(obj, torch.optim.lr_scheduler.LRScheduler):
                return self._everl_obj_to_dict(obj)

            raise NotImplementedError(
                f"Handling this class {obj.__class__} in not implemented "
            )

        return obj

    @staticmethod
    def _get_init_attributes(init_function):
        init_attributes = []
        kwargs = inspect.signature(init_function)
        for param in kwargs.parameters.values():
            init_attributes.append(param.name)

        return init_attributes

    def _dict_to_obj(
        self, obj_config_dict: dict, class_str_replace: List[Tuple[str, str]]
    ):
        if not ("class" in obj_config_dict.keys() and "_id" in obj_config_dict.keys()):
            return obj_config_dict

        obj_id = obj_config_dict.pop("_id")
        if obj_id in self.object_registry.keys():
            return self.object_registry[obj_id]

        class_str: str = obj_config_dict.pop("class")
        for str_replace in class_str_replace:
            class_str.replace(str_replace[0], str_replace[1])
        for attribute_name, value in obj_config_dict.items():
            if isinstance(value, dict):
                obj_config_dict[attribute_name] = self._dict_to_obj(
                    value, class_str_replace
                )
            if isinstance(value, (list, tuple)):
                for i, list_entry in enumerate(value):
                    if isinstance(list_entry, dict):
                        obj_config_dict[attribute_name][i] = self._dict_to_obj(
                            list_entry, class_str_replace
                        )

        constructor = self._get_class_constructor(class_str)
        obj = constructor(**obj_config_dict)
        self.object_registry[obj_id] = obj
        return obj

    def _get_class_constructor(self, class_str: str):
        module_path, class_name = class_str.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
