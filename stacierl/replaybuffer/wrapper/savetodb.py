from . import Wrapper
from ..replaybuffer_db import EpisodeSuccess, ReplayBufferDB, Batch
from stacie_sockets.stacie_socketclient import SocketClient
from stacie_sockets.socket_msg import Episode_Msg, Episodes_Msg, Text_Msg, Query_Msg, Length_Msg
from tiltmaze.env import Env

class SavetoDB(Wrapper):
    def __init__(
        self,
        wrapped_replaybuffer: ReplayBufferDB,
        env: Env,
        host='127.0.1.1',
        port=65430,
    ) -> None:
        
        self.wrapped_replaybuffer = wrapped_replaybuffer
        self.env = env
        self.host = host
        self.port = port
        if hasattr(wrapped_replaybuffer,"socket"):
            self.socket = wrapped_replaybuffer.socket
        else:

            self.socket = SocketClient()
            self.socket.start_connection(self.host, self.port)
        
    @property
    def batch_size(self) -> int:
        return self.wrapped_replaybuffer.batch_size
    
    def push(self, episode: EpisodeSuccess) -> None:
      
        self._save_to_database(episode)
        self.wrapped_replaybuffer.push(episode)
      
    # episode needs to be first element of list  
    def _save_to_database(self, episode) -> None:
        self.socket.send_init_msg(self.socket.db_methods.SAVE)

        info_dict = {
            'episode_length': len(episode.dones),
            'env_config': self.env.to_dict()
            }
        self.socket.send_data(Episode_Msg([episode, info_dict]))
        self.socket.receive_confirm_message()
    

    def delete_episodes(self,
                        db_filter,
                        nb_loaded_episodes = 0):
        self.socket.send_init_msg(self.socket.db_methods.DELETE_EPISODES)
        query_msg = Query_Msg(db_filter,nb_loaded_episodes)
        self.socket.send_data(query_msg)
        self.socket.receive_confirm_message()

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