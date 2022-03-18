from typing import List

from . import Wrapper, FilterElement
from ..replaybuffer_db import EpisodeSuccess, ReplayBufferDB, Batch
from stacie_sockets.stacie_socketclient import SocketClient
from stacie_sockets.socket_msg import Query_Msg

class LoadFromDB(Wrapper):
    def __init__(
        self,
        nb_loaded_episodes: int,
        db_filter: List[FilterElement],  
        wrapped_replaybuffer: ReplayBufferDB,
        host='10.15.16.73',
        port=65430,
    ) -> None:
        
        self.wrapped_replaybuffer = wrapped_replaybuffer
        self.host = host
        self.port = port
        self.socket = SocketClient(self.host, self.port)

        # load episodes from db in buffer, nb_loaded_episodes needs to be send too
        query_msg = Query_Msg(db_filter, nb_loaded_episodes)
        self.socket.send_data(query_msg)
        db_episodes = self.socket.receive_data()
        
        for episode in db_episodes:
            self.wrapped_replaybuffer.push(episode)

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