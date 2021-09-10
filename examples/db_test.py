
from collections import namedtuple
import imp
import importlib
import sys
from typing import NamedTuple
from my_fast_learner import Test
from numpy.lib.function_base import insert
sys.path.append(".")
from my_socket.socketclient import SocketClient
from my_socket.wrapper.socketwrapper import SocketWrapper
import numpy
import stacierl.environment.tiltmaze as tiltmaze
from importlib import import_module
import re




env_factory = tiltmaze.LNK2(dt_step=2 / 3)
env = env_factory.create_env()
env = SocketWrapper(env,player="testzzz",host="10.15.16.73",port=65430)
print(env.observation_space.shape)
#create_member_dict(env)
#with open("myfile.json", "w") as jsonfile:
#    json.dump(create_member_dict(env),jsonfile,indent=4)

"""
client  = SocketClient(host="10.15.16.73")
episodes = client.get_episodes(id = "613966eadca42b3c9967386a")
for ep in episodes:
    print(ep.steps[0] )

"""










































#module = import_module("my_fast_learner")
#klass = getattr(module,"Test")
#print(klass)
"""
namedt = Test()
for i in range(10):
    env.reset()
    for i in range(5):
        env.step(action=numpy.array([-0.53464305,0.36145353]))
        env.episode.nameddd = Test()
"""  



        


