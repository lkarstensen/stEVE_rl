import stacierl
from stacierl.replaybuffer.wrapper import FilterElement, FilterMethod

replay_buffer = 1e6,
batch_size = 64

db_filter = [
    FilterElement('success', 0.5, FilterMethod.GREATEREQUAL),
    FilterElement('simulation.tip_length', 25, FilterMethod.EXACT)
]

mongo_query = {}
for test_elem in db_filter:

    # if path in document.fields:
    #   path = test_elem.path
    # else (more generic?):
    #   path = 'env_config_INFO' + test_elem.path

    value = test_elem.value
    path = test_elem.path

    if test_elem.method == FilterMethod.EXACT:
        mongo_dict = {path: value}
    elif test_elem.method == FilterMethod.GREATEREQUAL:
        mongo_dict = {path: {'$gte': value}}
    elif test_elem.method == FilterMethod.LESSEQUAL:
        mongo_dict = {path: {'$lte': value}}
    elif test_elem.method == FilterMethod.NOTEQUAL:
        mongo_dict = {path: {'$ne': value}}
        
    mongo_query.update(mongo_dict)
    
print(mongo_query)

"""
replay_buffer = stacierl.replaybuffer.VanillaStepDB(replay_buffer, batch_size)
# nb loaded episodes in wrapper?
replay_buffer = stacierl.replaybuffer.LoadFromDB(nb_loaded_episodes=100,
                                                 filter=db_filter, 
                                                 wrapped_replaybuffer=replay_buffer, 
                                                 host='10.15.16.238')
"""

