import json
from importlib import import_module
import inspect
import numpy as np
from torch import device
from enum import Enum
from enum import Enum
import os


class ConfigHandler:
    def __init__(self, indent: int = 4, sort_keys: bool = False):
        self.indent = indent
        self.sort_keys = sort_keys

    def save_config(self, object, filepath: str):
        config_dict = self._to_dict(object)
        if not filepath.endswith(".json"):
            filepath = filepath + ".json"

        self._save_dict_to_file(config_dict, filepath)

    def _save_dict_to_file(self, config_dict: dict, file: str):
        with open(file, "w") as jsonfile:
            json.dump(
                config_dict, jsonfile, indent=self.indent, sort_keys=self.sort_keys
            )

    def _to_dict(self, object) -> dict:
        attributes_dict = {}
        attributes_dict["class"] = f"{object.__module__}.{object.__class__.__name__}"
        init_attributes = self._get_init_attributes(object.__init__)

        if "args" in init_attributes:
            init_attributes.remove("args")

        if "kwargs" in init_attributes:
            init_attributes.remove("kwargs")

        for attribute in init_attributes:
            value = getattr(object, attribute)

            if isinstance(value, np.integer):
                dict_value = int(value)

            elif isinstance(value, device):
                dict_value = str(value)

            elif isinstance(value, Enum):
                dict_value = value.value

            elif isinstance(value, np.ndarray):
                dict_value = value.tolist()

            elif isinstance(value, list):
                dict_value = []
                for v in value:
                    if hasattr(v, "__module__"):
                        if "eve" in v.__module__ and "Space" in str(type(v)):
                            dict_value.append(str(type(v)))
                            continue

                        if "stacierl" in v.__module__ or "eve" in v.__module__:
                            dict_value.append(self._to_dict(v))
                        continue

                    dict_value.append(v)

            else:
                if hasattr(value, "__module__"):
                    if "eve" in value.__module__ and "Space" in str(type(value)):
                        dict_value = str(type(value))
                        attributes_dict[attribute] = dict_value
                        continue

                    if "stacierl" in value.__module__ or "eve" in value.__module__:
                        dict_value = self._to_dict(value)
                        attributes_dict[attribute] = dict_value
                        continue
                dict_value = value

            attributes_dict[attribute] = dict_value
        return attributes_dict

    @staticmethod
    def _get_init_attributes(init_function):
        init_attributes = []
        kwargs = inspect.signature(init_function)
        for param in kwargs.parameters.values():
            init_attributes.append(param.name)

        return init_attributes

    def load_config_data(self, json_file):
        if ".json" in json_file:
            with open(json_file) as j_file:
                config_dict = json.load(j_file)
        else:
            config_dict = json.loads(json_file)

        obj = self._dict_to_obj(config_dict)
        return obj

    def _get_class_constructor(self, class_str: str):
        try:
            module_path, class_name = class_str.rsplit(".", 1)
            module = import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(class_str)

    def _dict_to_obj(self, config_dict: dict):
        class_str = config_dict["class"]
        del config_dict["class"]

        config_keys = config_dict.keys()
        for key in config_keys:
            if "wrapped" in key:
                if isinstance(config_dict[key], dict):
                    wrapped_keys = config_dict[key].keys()
                    for wrapped_key in wrapped_keys:
                        config_dict[key][wrapped_key] = self._dict_to_obj(
                            config_dict[key][wrapped_key]
                        )
                if isinstance(config_dict[key], list):
                    for i in range(len(config_dict[key])):
                        if isinstance(config_dict[key][i], dict):
                            config_dict[key][i] = self._dict_to_obj(config_dict[key][i])
            elif isinstance(config_dict[key], dict):
                config_dict[key] = self._dict_to_obj(config_dict[key])

        constructor = self._get_class_constructor(class_str)
        obj = constructor(**config_dict)
        return obj
