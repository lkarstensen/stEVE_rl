from typing import Dict

from . import Wrapper
from ..replaybuffer_db import EpisodeSuccess, ReplayBufferDB, Batch
from my_socket.socketclient import SocketClient

class SavetoDB(Wrapper):
    def __init__(
        self,
        wrapped_replaybuffer: ReplayBufferDB,
        env_config: Dict,
        host='10.15.16.73',
        port=65430,
    ) -> None:
        
        self.wrapped_replaybuffer = wrapped_replaybuffer
        self.env_config = env_config
        self.host = host
        self.port = port
        self.socket = SocketClient(self.host, self.port)

    @property
    def batch_size(self) -> int:
        return self.wrapped_replaybuffer.batch_size
    
    def push(self, episode: EpisodeSuccess) -> None:
        # save to db
        
        self.wrapped_replaybuffer.push(episode)
        
    def sample(self) -> Batch:
        return self.wrapped_replaybuffer.sample()
    
    def copy(self):
        return self.__class__(
            self.wrapped_replaybuffer.copy(),
            self.env_config,
            self.host,
            self.port)
        
    def close(self) -> None:
        self.wrapped_replaybuffer.close()
        
    def __len__(self) -> int:
        return len(self.wrapped_replaybuffer)