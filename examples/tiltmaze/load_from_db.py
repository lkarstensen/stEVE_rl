import stacierl
from stacierl.replaybuffer.wrapper import FilterElement, FilterMethod

replay_buffer = 1e6,
batch_size = 64

db_filter = [
    FilterElement('success', 0.5, FilterMethod.GREATEREQUAL),
    FilterElement('simulation.tip_length', 25, FilterMethod.EXACT)
]

replay_buffer = stacierl.replaybuffer.VanillaStepDB(replay_buffer, batch_size)
replay_buffer = stacierl.replaybuffer.LoadFromDB(nb_loaded_episodes=100,
                                                 filter=db_filter, 
                                                 wrapped_replaybuffer=replay_buffer, 
                                                 host='10.15.16.238')
