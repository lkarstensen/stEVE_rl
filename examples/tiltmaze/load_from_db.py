import stacierl
import tiltmaze
import eve

import math

from stacierl.replaybuffer.wrapper import filter_database
from stacierl.replaybuffer.wrapper import FilterElement, FilterMethod

from typing import Dict
from stacierl.replaybuffer import EpisodeSuccess
               
eve_tree = eve.vesseltree.AorticArch(seed=1234)
vessel_tree = tiltmaze.vesseltree.FromEve3d(
    eve3d_vessel_tree=eve_tree,
    lao_rao=20,
    cra_cau=-5,
    contour_approx_margin=2.0,
    scaling_artery_diameter=0.5,
)
simu = tiltmaze.simulation.Guidewire(
    tip_length=25,
    tip_angle=math.pi / 2,
    flex_length=30,
    flex_element_length=1,
    flex_rotary_spring_stiffness=7e5,
    flex_rotary_spring_damping=3e2,
    stiff_element_length=2,
    stiff_rotary_spring_stiffness=1e6,
    stiff_rotary_spring_damping=3e2,
    wire_diameter=2,
    friction=1.0,
    damping=0.000001,
    velocity_limits=(50, 1.5),
    normalize_action=True,
    dt_step=1 / 7.5,
    dt_simulation=0.0002,
    body_mass=0.01,
    body_moment=0.1,
    spring_stiffness=2.5e6,
    spring_damping=100,
    last_segment_kp_angle=3,
    last_segment_kp_translation=5,
)
target = tiltmaze.target.CenterlineRandom(5)
pathfinder = tiltmaze.pathfinder.BruteForceBFS()
start = tiltmaze.start.InsertionPoint()
imaging = tiltmaze.imaging.ImagingDummy((500, 500))

pos = tiltmaze.state.Position(n_points=6, resolution=1)
pos = tiltmaze.state.wrapper.Normalize(pos)
target_state = tiltmaze.state.Target()
target_state = tiltmaze.state.wrapper.Normalize(target_state)
rot = tiltmaze.state.Rotation()
state = tiltmaze.state.Combination([pos, target_state, rot])

target_reward = tiltmaze.reward.TargetReached(1.0)
step_reward = tiltmaze.reward.Step(-0.005)
path_length_reward = tiltmaze.reward.PathLengthDelta(0.001)
reward = tiltmaze.reward.Combination([target_reward, step_reward, path_length_reward])

done_target = tiltmaze.done.TargetReached()
done_steps = tiltmaze.done.MaxSteps(100)
done = tiltmaze.done.Combination([done_target, done_steps])

success = tiltmaze.success.TargetReached()
visu = tiltmaze.visualisation.VisualisationDummy()

randomizer = tiltmaze.randomizer.RandomizerDummy()
env = tiltmaze.Env(
    vessel_tree=vessel_tree,
    state=state,
    reward=reward,
    done=done,
    simulation=simu,
    start=start,
    target=target,
    imaging=imaging,
    pathfinder=pathfinder,
    visualisation=visu,
    success=success,
    randomizer=randomizer,
)
#db_filter = filter_database(env, success=0.0, episode_length=10)
db_filter = [FilterElement('episode_length', 100, FilterMethod.EXACT)]

replay_buffer = stacierl.replaybuffer.VanillaStepDB(1e6, 64)
replay_buffer = stacierl.replaybuffer.LoadFromDB(nb_loaded_episodes=10,
                                                 db_filter=db_filter, 
                                                 wrapped_replaybuffer=replay_buffer, 
                                                 host='127.0.1.1',
                                                 port=65430)
"""
def mongodb_query(stacierl_query) -> Dict:
    mongo_query = {}
    stacierl_episode = EpisodeSuccess()
    fields = list(vars(stacierl_episode).keys())
    filter_elem:FilterElement 
    for filter_elem in stacierl_query:
        path = filter_elem.path
        value = filter_elem.value

        if path in fields:
            path = filter_elem.path
        elif path == 'episode_length':
            path = 'episode_length_INFO'
        else:
            path = 'env_config_INFO.' + filter_elem.path
        if isinstance(value, tuple):
            for i in range(len(value)):
                updated_path = path + '.' + str(i) + '.' + str(0)

                if filter_elem.method == FilterMethod.EXACT:
                    mongo_dict = {updated_path: value[i]}
                elif filter_elem.method == FilterMethod.GREATEREQUAL:
                    mongo_dict = {updated_path: {'$gte': value[i]}}
                elif filter_elem.method == FilterMethod.LESSEQUAL:
                    mongo_dict = {updated_path: {'$lte': value[i]}}
                elif filter_elem.method == FilterMethod.NOTEQUAL:
                    mongo_dict = {updated_path: {'$ne': value[i]}}

                mongo_query.update(mongo_dict)
        else:
            if filter_elem.method == FilterMethod.EXACT:
                mongo_dict = {path: value}
            elif filter_elem.method == FilterMethod.GREATEREQUAL:
                mongo_dict = {path: {'$gte': value}}
            elif filter_elem.method == FilterMethod.LESSEQUAL:
                mongo_dict = {path: {'$lte': value}}
            elif filter_elem.method == FilterMethod.NOTEQUAL:
                mongo_dict = {path: {'$ne': value}}
            
            mongo_query.update(mongo_dict)

    return mongo_query

print(mongodb_query(db_filter))
"""