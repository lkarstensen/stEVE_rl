import random
import numpy as np
from .replaybuffer import ReplayBuffer
from .vanilla import Episode, Batch, Vanilla
import torch.multiprocessing as mp


class VanillaShared(ReplayBuffer):
    def __init__(self, capacity):
        self._task_queue = mp.SimpleQueue()
        self._result_queue = mp.SimpleQueue()
        self._request_lock = mp.Lock()
        self._shutdown_event = mp.Event()
        process = mp.Process(target=self.run, args=[capacity])
        process.start()

    def push(self, episode: Episode):

        self._task_queue.put(["push", episode])

    def sample(self, batch_size: int) -> Batch:
        self._request_lock.acquire()
        if not self._shutdown_event.is_set():
            self._task_queue.put(["sample", batch_size])
            batch = self._result_queue.get()
            self._request_lock.release()
            return batch
        else:
            return Batch([], [], [], [], [], [], [])

    def run(self, capacity):
        self._internal_replay_buffer = Vanilla(capacity)
        while not self._shutdown_event.is_set():
            task = self._task_queue.get()
            if task[0] == "push":
                self._internal_replay_buffer.push(task[1])
            elif task[0] == "sample":
                batch = self._internal_replay_buffer.sample(task[1])
                self._result_queue.put(batch)
            elif task[0] == "length":
                self._result_queue.put(len(self._internal_replay_buffer))
            elif task[0] == "shutdown":
                break

    def __len__(
        self,
    ):
        if not self._shutdown_event.is_set():
            self._request_lock.acquire()
            self._task_queue.put(["length"])
            length = self._result_queue.get()
            self._request_lock.release()
            return length
        else:
            return 0

    def get_length(self):
        return len(self)

    def copy(self):
        return self

    def close(self):
        self._request_lock.acquire()
        self._shutdown_event.set()
        self._task_queue.put("shutdown")
        self._request_lock.release()
