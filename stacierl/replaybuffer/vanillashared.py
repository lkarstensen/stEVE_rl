from time import sleep
from multiprocessing.synchronize import Lock as mp_lock
from multiprocessing.synchronize import Event as mp_event
import torch
import torch.multiprocessing as mp

from .replaybuffer import ReplayBuffer, Episode, Batch
from .vanillaepisode import VanillaEpisode
from .vanillastep import VanillaStep


class VanillaSharedBase(ReplayBuffer):
    def __init__(
        self,
        push_queue: mp.SimpleQueue,
        sample_queue: mp.SimpleQueue,
        task_queue: mp.SimpleQueue,
        result_queue: mp.SimpleQueue,
        request_lock: mp_lock,
        shutdown_event: mp_event,
        batch_size: int,
    ):
        self._push_queue = push_queue
        self._task_queue = task_queue
        self._sample_queue = sample_queue
        self._result_queue = result_queue
        self._request_lock = request_lock
        self._shutdown_event = shutdown_event
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):

        if not self._shutdown_event.is_set():
            self._push_queue.put(episode.to_replay())

    def sample(self) -> Batch:

        if self._shutdown_event.is_set():
            return Batch([], [], [], [], [])

        return self._sample_queue.get()

    def __len__(
        self,
    ):
        if self._shutdown_event.is_set():  #
            return 0

        with self._request_lock:
            self._task_queue.put(["length"])
            length = self._result_queue.get()
        return length

    def copy(self):
        return self

    def close(self) -> None:
        ...


class VanillaStepShared(VanillaSharedBase):
    def __init__(self, capacity, batch_size, sample_device: torch.device):
        super().__init__(
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.SimpleQueue(),
            mp.Lock(),
            mp.Event(),
            batch_size,
        )
        self.capacity = capacity
        self.sample_device = sample_device
        self._process = mp.Process(target=self.run)
        self._process.start()

    def run(self):
        internal_replay_buffer = VanillaStep(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)

    def loop(self, internal_replay_buffer: ReplayBuffer):
        while not self._shutdown_event.is_set():
            if (
                self._sample_queue.empty()
                and len(internal_replay_buffer) > self.batch_size
            ):
                batch = internal_replay_buffer.sample()
                if self.sample_device != torch.device("mps"):
                    batch = batch.to(self.sample_device)
                self._sample_queue.put(batch)
            elif not self._task_queue.empty():
                task = self._task_queue.get()
                if task[0] == "length":
                    self._result_queue.put(len(internal_replay_buffer))
                elif task[0] == "shutdown":
                    break
            elif not self._push_queue.empty():
                batch = self._push_queue.get()
                internal_replay_buffer.push(batch)
            else:
                sleep(0.0001)
        internal_replay_buffer.close()

    def copy(self):
        return VanillaSharedBase(
            self._push_queue,
            self._sample_queue,
            self._task_queue,
            self._result_queue,
            self._request_lock,
            self._shutdown_event,
            self.batch_size,
        )

    def close(self):
        self._shutdown_event.set()
        self._process.join()
        self._process.close()


class VanillaEpisodeShared(VanillaStepShared):
    def run(self):
        # os.nice(15)
        internal_replay_buffer = VanillaEpisode(self.capacity, self._batch_size)
        self.loop(internal_replay_buffer)
