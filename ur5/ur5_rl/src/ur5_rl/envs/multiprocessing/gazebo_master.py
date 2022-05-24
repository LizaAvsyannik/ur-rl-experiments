import gym
import torch
from copy import deepcopy

from .shared_state import SharedState
from .gazebo_worker import GazeboEnvWorker


class GazeboMaster(gym.Env):
    def __init__(self, nenvs, env_cls, gazebo_ports, launch_files, **kwargs):
        self.nenvs = nenvs
        self.__state = SharedState(nenvs)
        self.__env_cls = env_cls
        self.__workers = [GazeboEnvWorker(env_cls,
                                          self.__state._observation_queues[i],
                                          self.__state._action_queues[i],
                                          str(i),
                                          gazebo_ports[i],
                                          launch_files,
                                          **kwargs.get('env_kwargs')) for i in range(nenvs)]

    def start(self):
        for worker in self.__workers:
            worker.start()

    def join(self):
        self.__state.close()
        for worker in self.__workers:
            worker.join()

    def terminate(self):
        self.__state.close()
        for worker in self.__workers:
            worker.terminate()

    def state_dim(self):
        return self.__env_cls.state_dim()

    def action_dim(self):
        return self.__env_cls.action_dim()
    
    def _get_observations(self):
        """ Returns CPU tensors of shape:
            - (nenvs, obs_space_dim) - observations
            - (nenvs, 1) - rewards
            - (nenvs, 1) - dones
        """
        observations = []
        rewards = []
        dones = []
        infos = {}
        for q in self.__state._observation_queues:
            obs, r, done, info = q.get()
            observations.append(obs.clone())
            rewards.append(deepcopy(r))
            dones.append(deepcopy(done))
            for k, v in info.items():
                if k not in infos:
                    infos[k] = [deepcopy(v)]
                else:
                    infos[k].append(deepcopy(v))
        return torch.vstack(observations), torch.FloatTensor(rewards), torch.ByteTensor(dones), infos

    def step(self, actions):
        """ Receives as input a CPU tensor of shape
            (nenvs, action_space_dim) - actions
        """
        for i, action in enumerate(actions):
            self.__state._action_queues[i].put(action)
        return self._get_observations()

    def reset(self):
        """ Receives as input a CPU tensor of shape
            (nenvs, action_space_dim) - actions
        """
        for i in range(self.nenvs):
            self.__state._action_queues[i].put('reset')
        return self._get_observations()[0]
