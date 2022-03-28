import stacierl
import tiltmaze
import eve

import math

from stacierl.replaybuffer.wrapper import filter_database

########################################################
# 3) USER INTERFACE BASED, HARD-CODED FILTER OPTIONS
########################################################                
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
from stacierl.replaybuffer.wrapper import FilterElement, FilterMethod
#db_filter = filter_database(env, success=0.0)
db_filter1 = [FilterElement('simulation.velocity_limits.0.0', 50.0, FilterMethod.EXACT),
             FilterElement('simulation.velocity_limits.1.0', 1.5, FilterMethod.EXACT)]
db_filter2 = [FilterElement('simulation.velocity_limits', (50.0, 1.5), FilterMethod.EXACT)]

replay_buffer = stacierl.replaybuffer.VanillaStepDB(1e6, 64)
replay_buffer = stacierl.replaybuffer.LoadFromDB(nb_loaded_episodes=10,
                                                 db_filter=db_filter1, 
                                                 wrapped_replaybuffer=replay_buffer, 
                                                 host='127.0.1.1',
                                                 port=65430)