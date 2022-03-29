from .wrapper import Wrapper, FilterElement, FilterMethod
from ..replaybuffer_db import EpisodeSuccess, ReplayBufferDB, Batch
from stacie_sockets.stacie_socketclient import SocketClient, DBMethods
from stacie_sockets.socket_msg import Query_Msg, Init_Msg
from tiltmaze.env import Environment

from typing import Dict, List, Optional

def _dict_to_filter(obj_dict: Dict, path: List[str]=[], filter_list: list=[]):    
    keys = obj_dict.keys()
    for key in keys:
        path.append(key)
        if isinstance(obj_dict[key], dict):
            path, filter_list = _dict_to_filter(obj_dict[key], path, filter_list)
            path.pop()
        elif isinstance(obj_dict[key], list):
            for i in range(len(obj_dict[key])):
                path.append(str(i))
                if isinstance(obj_dict[key][i], dict):
                    path, filter_list = _dict_to_filter(obj_dict[key][i], path, filter_list)
                    path.pop()
                #else:
                #    filter_list.append(FilterElement('.'.join(path), obj_dict[key], FilterMethod.EXACT))
                #    path.pop()
        else:
            filter_list.append(FilterElement('.'.join(path), obj_dict[key], FilterMethod.EXACT))
            path.pop()  
            
    return path, filter_list


def filter_database(env: Environment, 
                    success: Optional[float] = None,
                    success_criterion: Optional[FilterMethod] = FilterMethod.GREATEREQUAL,
                    episode_length: Optional[int] = None,
                    episode_length_criterion: Optional[FilterMethod] = FilterMethod.GREATEREQUAL
                   ):
    
    db_query_list = []
    
    if success:
        db_query_list += [FilterElement('success', success, success_criterion)]
    if episode_length:
        db_query_list += [FilterElement('episode_length', episode_length, episode_length_criterion)]
    
    env_dict = env.to_dict()
    equal_objects = ['simulation', 'done', 'reward', 'state']
    for key in equal_objects:
        _, filter_elements = _dict_to_filter(env_dict[key], path=[key], filter_list=[])
        db_query_list += filter_elements

    return db_query_list


class LoadFromDB(Wrapper):
    def __init__(
        self,
        nb_loaded_episodes: int,
        db_filter: List[FilterElement],  
        wrapped_replaybuffer: ReplayBufferDB,
        host='127.0.1.1',
        port=65430,
    ) -> None:
        
        self.wrapped_replaybuffer = wrapped_replaybuffer
        self.host = host
        self.port = port
        self.socket = SocketClient()
        self.socket.start_connection(self.host, self.port)

        self.socket.send_init_msg(DBMethods.GET_EPISODES)
        query_msg = Query_Msg(db_filter, nb_loaded_episodes)
        self.socket.send_data(query_msg)
        episodes_msg = self.socket.receive_data()
        db_episodes = episodes_msg.episodes
       
        for episode in db_episodes:
            self.wrapped_replaybuffer.push(episode)

        print('%d episodes were successfully loaded to buffer'%len(db_episodes))

    @property
    def batch_size(self) -> int:
        return self.wrapped_replaybuffer.batch_size
    
    def push(self, episode: EpisodeSuccess) -> None:        
        self.wrapped_replaybuffer.push(episode)
        
    def sample(self) -> Batch:
        return self.wrapped_replaybuffer.sample()
    
    def copy(self):
        return self.__class__(
            self.wrapped_replaybuffer.copy(),
            self.host,
            self.port)
        
    def close(self) -> None:
        self.wrapped_replaybuffer.close()
        
    def __len__(self) -> int:
        return len(self.wrapped_replaybuffer)