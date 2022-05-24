import torch.multiprocessing as tmp


class SharedState:
    def __init__(self, nenvs):
        self.nenvs = nenvs
        self._observation_queues = [tmp.Queue(1) for _ in range(nenvs)]
        self._action_queues = [tmp.Queue(1) for _ in range(nenvs)]

    def close(self):
        for q in self._observation_queues:
            q.close()
            q.join_thread()
        for q in self._action_queues:
            q.close()
            q.join_thread()
