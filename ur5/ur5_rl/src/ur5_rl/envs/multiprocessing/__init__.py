import gym
from .gazebo_master import GazeboMaster

gym.envs.register(
    id='UR5MultiEnvGoal-v0',
    entry_point='ur5_rl.envs.multiprocessing:GazeboMaster',
    kwargs = {'nenvs': 0,
              'ros_master_ports': [],
              'gazebo_ports': [],
              'launch_files': [],
              'env_kwargs': {}}
)