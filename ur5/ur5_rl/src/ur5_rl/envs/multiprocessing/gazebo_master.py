import gym
import torch

from .shared_state import SharedState
from .gazebo_worker import GazeboEnvWorker


class GazeboMaster(gym.Env):
    def __init__(self, nenvs, ros_master_ports, gazebo_ports, launch_files, **env_kwargs):
        self.nenvs = nenvs
        self.__state = SharedState(nenvs)
        self.__workers = [GazeboEnvWorker(self.__state._observation_queues[i],
                                          self.__state._action_queues[i],
                                          'ur5',
                                          ros_master_ports[i],
                                          gazebo_ports[i],
                                          launch_files,
                                          **env_kwargs) for i in range(nenvs)]

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
    
    def get_observations(self):
        """ Returns CPU tensors of shape:
            - (nenvs, obs_space_dim) - observations
            - (nenvs, 1) - rewards
            - (nenvs, 1) - dones
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        for obs, r, done, info in map(lambda q: q.get(), self.__state._observation_queues):
            observations.append(obs)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return torch.vstack(observations), torch.vstack(rewards), torch.vstack(dones), infos

    def step(self, actions):
        """ Receives as input a CPU tensor of shape
            (nenvs, action_space_dim) - actions
        """
        for i, action in enumerate(actions):
            self.__state._action_queues[i].put(action)
        return self.get_observations()

    def reset(self):
        """ Receives as input a CPU tensor of shape
            (nenvs, action_space_dim) - actions
        """
        for i in range(self.nenvs):
            self.__state._action_queues[i].put('reset')
        return self.get_observations()[0]
