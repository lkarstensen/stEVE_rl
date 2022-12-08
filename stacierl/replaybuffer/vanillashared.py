import random
from stacierl.replaybuffer.vanillaepisode import VanillaEpisode
import numpy as np
from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillastep import VanillaStep
import torch.multiprocessing as mp
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
        counter_lock: mp_lock,
        counter: mp.Value,
    ):
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size

        self._added_self_to_counter = False
        self._counter_lock = counter_lock
        self.access_counter = counter

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if not self._added_self_to_counter:
            with self._counter_lock:
                self.access_counter.value += 1
            self._added_self_to_counter = True

        if not self._shutdown_event.is_set():
            self._task_queue.put(["push", episode.to_replay()])

    def sample(self) -> Batch:
        if not self._added_self_to_counter:
            with self._counter_lock:
                self.access_counter.value += 1
            self._added_self_to_counter = True
        self._request_lock.acquire()
        if not self._shutdown_event.is_set():
            self._task_queue.put(["sample"])
            batch = self._result_queue.get()
            self._request_lock.release()
            return batch
        else:
            return Batch([], [], [], [], [])

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

    def copy(self):
        copy = VanillaSharedBase(
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            self._counter_lock,
            self.access_counter,
        )
        return self

    def close(self):
        self._request_lock.acquire()
        self._shutdown_event.set()
        self._task_queue.put("shutdown")
        self._request_lock.release()


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
            mp.Lock(),
            mp.Value("i", 0),
        )
        self.capacity = capacity
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        self._internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        while not self._shutdown_event.is_set():
            task = self._task_queue.get()
            if task[0] == "push":
                self._internal_replay_buffer.push(task[1])
            elif task[0] == "sample":
                batch = self._internal_replay_buffer.sample()
                self._result_queue.put(batch)
            elif task[0] == "length":
                self._result_queue.put(len(self._internal_replay_buffer))
            elif task[0] == "shutdown":
                break
        self._internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
            self._counter_lock,
            self.access_counter,
        )

    def close(self):
        super().close()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        self._internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        while not self._shutdown_event.is_set():
            task = self._task_queue.get()
            if task[0] == "push":
                self._internal_replay_buffer.push(task[1])
            elif task[0] == "sample":
                batch = self._internal_replay_buffer.sample()
                self._result_queue.put(batch)
            elif task[0] == "length":
                self._result_queue.put(len(self._internal_replay_buffer))
            elif task[0] == "shutdown":
                break
        self._internal_replay_buffer.close()
