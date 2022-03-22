import stacierl
import tiltmaze
from typing import List

from stacierl.replaybuffer.wrapper import FilterElement, FilterMethod
from tiltmaze.util import TiltmazeUserObject

########################################################
# 0) UTILITY FUNCTION
########################################################

def object_to_filterelement(
                            tiltmaze_object: TiltmazeUserObject,
                            value_dict: List[FilterElement]=[]
                            ):
    filter_list = []
    obj_dict = tiltmaze_object.to_dict()
    env_attributes = tiltmaze.Env.__init__.__code__.co_varnames
    
    for attr in env_attributes:
        if attr.replace('_','') in obj_dict['class']:
            env_path = attr 
            
    filter_list.append(FilterElement(env_path + '.' + 'class', obj_dict['class'], FilterMethod.EXACT))
    
    for obj_filter in value_dict:
        filter_list.append(FilterElement(env_path + '.' + obj_filter.path, obj_filter.value, obj_filter.method))
        
    return filter_list

########################################################
# 1) USER INTERFACE BASED ON TILTMAZE OBJECTS
########################################################    

guidewire_filter = [
    FilterElement('tip_length', 25, FilterMethod.EXACT),
    FilterElement('tip_angle', 0, FilterMethod.NOTEQUAL),
]

step_reward_filter = [
    FilterElement('factor', -0.005, FilterMethod.LESSEQUAL)
]

guidewire_filter_element = object_to_filterelement(tiltmaze.simulation.Guidewire(), guidewire_filter)
success_filter_element = object_to_filterelement(tiltmaze.success.TargetReached())
reward_filter_element = object_to_filterelement(tiltmaze.reward.Step(), step_reward_filter)

db_filter1 = guidewire_filter_element + success_filter_element + [FilterElement('success', 0.5, FilterMethod.GREATEREQUAL)]

########################################################
# 2) USER INTERFACE BASED ON TILTMAZE CONFIG FILE
########################################################

db_filter2 = [
    FilterElement('simulation.class', 'tiltmaze.simulation.guidewire.Guidewire', FilterMethod.EXACT),
    FilterElement('simulation.tip_length', 25, FilterMethod.EXACT),
    FilterElement('simulation.tip_angle', 0, FilterMethod.NOTEQUAL),
    FilterElement('success.class', 'tiltmaze.success.targetreached.TargetReached', FilterMethod.EXACT),
    FilterElement('success', 0.5, FilterMethod.GREATEREQUAL),
]
