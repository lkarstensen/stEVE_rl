from stacierl.util.confighandler import ConfigHandler
import json
from importlib import import_module


class JSONHandler(ConfigHandler):
    def __init__(self, indent: int = 4, sort_keys: bool = False):
        self.indent = indent
        self.sort_keys = sort_keys


    def load_config_data(self, json_file):
        if '.json' in json_file:
            with open(json_file) as j_file:
                config_dict = json.load(j_file)
        else:
            config_dict = json.loads(json_file)
        
        obj = self._dict_to_obj(config_dict)
        return obj


    def _get_class_constructor(self, class_str: str):
        try:
            module_path, class_name = class_str.rsplit('.', 1)
            module = import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(class_str)


    def _dict_to_obj(self, config_dict: dict):
        class_str = config_dict['class']
        del config_dict['class']

        config_keys = config_dict.keys()
        for key in config_keys:
            if 'wrapped' in key:
                if isinstance(config_dict[key], dict):
                    wrapped_keys = config_dict[key].keys()
                    for wrapped_key in wrapped_keys:
                        config_dict[key][wrapped_key] = self._dict_to_obj(config_dict[key][wrapped_key])
                if isinstance(config_dict[key], list):
                    for i in range(len(config_dict[key])):
                        if isinstance(config_dict[key][i], dict):
                            config_dict[key][i] = self._dict_to_obj(config_dict[key][i])
            elif isinstance(config_dict[key], dict):
                config_dict[key] = self._dict_to_obj(config_dict[key])

        constructor = self._get_class_constructor(class_str)
        obj = constructor(**config_dict)
        return obj

    
    def save_dict_to_file(self, config_dict: dict, file: str):        
        with open(file, 'w') as jsonfile:
            json.dump(config_dict, jsonfile, indent=self.indent, sort_keys=self.sort_keys)
  
          

