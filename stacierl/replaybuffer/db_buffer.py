from .replaybuffer import ReplayBuffer,Episode,Batch
import random
import numpy as np 
from typing import Dict
from my_socket.socketclient import SocketClient


class DBBuffer(ReplayBuffer):

    def __init__(self, capacity,mixed = False):
        self.mixed = mixed
        self.capacity = capacity/2 if mixed else capacity
        self.buffer = []
        self.dbbuffer = [] 
        self.position = 0
        self.sample_counter = 1
        self.init_with_db()
        



    def init_with_db(self):
        counter = 0
        con = SocketClient(host="10.15.16.73")
        doc_limit = 0
        if self.mixed:
            doc_limit = 20000
        #__raw__ = {"steps": {"$elemMatch":{"info":True}},"episode_length":{"$gt":20}} 
        #__raw__={"steps": {"$elemMatch":{"extra_info":1}},
        
        episodes = con.get_episodes(player="fastlearner")
        #episodes = episodes + episodes
        # If you want to sort:
        episodes.sort(key = lambda episode: episode.episode_reward)
        for episode in episodes:
            steps = episode.steps
            for i in range(len(steps)-1):
                if counter == self.capacity:
                    print("capacity_limit reached")
                    break
                self.dbbuffer.append((self.dict_state_to_flat_np_state(steps[i]["next_state"]),steps[i+1]["action"],steps[i+1]["reward"],\
                    self.dict_state_to_flat_np_state(steps[i+1]["next_state"]),steps[i+1]["done"]))
                counter+= 1
            else:
                continue
            break
        print("DataBaseBuffer:")
        print(f"Filled with {len(self.dbbuffer)} DataBaseSteps | Capacity {self.capacity}")





    def dict_state_to_flat_np_state(self,state: Dict[str, np.ndarray]) -> np.ndarray:
        keys = sorted(state.keys())

        flat_state = np.array([])
        for key in keys:
            flat_state = np.append(flat_state, state[key].flatten())
        return flat_state

    def push(self, episode: Episode):

        for i in range(len(episode)):

            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (
                episode.states[i],
                episode.actions[i],
                episode.rewards[i],
                episode.next_states[i],
                episode.dones[i],
            )
            self.position = int((self.position + 1) % self.capacity)  # as a ring buffer


    def sample(self, batch_size: int) -> Batch:
        if self.mixed:
            batch = random.sample(self.buffer,int(batch_size/2))
            batch + random.sample(self.dbbuffer,int(batch_size/2))
        else:
            
            if self.sample_counter * batch_size < len(self.dbbuffer):
                sample_area  = self.dbbuffer[:self.sample_counter*batch_size]
                batch = random.sample(sample_area, batch_size)
                self.sample_counter +=  1  
            else:
                
                batch = random.sample(self.dbbuffer, batch_size)
        

        
            
            
        
        batch = map(np.stack, zip(*batch))  # stack for each element

       
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return Batch(*batch)

    def __len__(
        self,
    ):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer) + len(self.dbbuffer) if self.mixed else len(self.dbbuffer)

    def get_length(self):
        return len(self.buffer) + len(self.dbbuffer) if self.mixed else len(self.dbbuffer)

    def copy(self):
        copy = self.__class__(self.capacity)
        if self.mixed:
            for i in range(len(self.buffer)):
                copy.buffer.append(self.buffer[i])
        for i in range(len(self.dbbuffer)):
            copy.dbbuffer.append(self.dbbuffer[i])
        copy.position = self.position
        return copy