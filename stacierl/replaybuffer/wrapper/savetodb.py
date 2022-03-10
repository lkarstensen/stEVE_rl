from typing import Dict

from . import Wrapper
from ..replaybuffer_db import EpisodeSuccess, ReplayBufferDB, Batch
from my_socket.socketclient import SocketClient
from tiltmaze.env import Env

class SavetoDB(Wrapper):
    def __init__(
        self,
        wrapped_replaybuffer: ReplayBufferDB,
        env: Env,
        host='10.15.16.73',
        port=65430,
    ) -> None:
        
        self.wrapped_replaybuffer = wrapped_replaybuffer
        self.env = env
        self.host = host
        self.port = port
        self.socket = SocketClient(self.host, self.port)

    @property
    def batch_size(self) -> int:
        return self.wrapped_replaybuffer.batch_size
    
    def push(self, episode: EpisodeSuccess) -> None:
        self._save_to_database(episode)
        self.wrapped_replaybuffer.push(episode)
      
    # episode needs to be first element of list  
    def _save_to_database(self, episode) -> None:
        self.socket.send_init_msg("save")
        
        info_dict = {
            'episode_length': len(episode.dones),
            'env_config': self.env.to_dict()
            }
        self.socket.send_data([episode, info_dict])
        
        self.socket.recieve_confirm_message(self.socket)

        
    def sample(self) -> Batch:
        return self.wrapped_replaybuffer.sample()
    
    def copy(self):
        return self.__class__(
            self.wrapped_replaybuffer.copy(),
            self.env,
            self.host,
            self.port)
        
    def close(self) -> None:
        self.wrapped_replaybuffer.close()
        
    def __len__(self) -> int:
        return len(self.wrapped_replaybuffer)