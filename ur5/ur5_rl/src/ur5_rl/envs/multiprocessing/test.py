from click import launch
import yaml
import rospy
import torch

from ur5_rl.envs.task_envs.ur5_env_reach_goal import UR5EnvGoal
from ur5_rl.envs.multiprocessing.gazebo_master import GazeboMaster

if __name__ == '__main__':
    NENVS = 4
    gazebo_ports = [10450 + i for i in range(NENVS)]
    launch_files = ['/home/ros/catkin_ws/src/ur-rl-experiments/ur5/ur5_gazebo/launch/ur5_cubes.launch']

    params_path = '/home/ros/catkin_ws/src/ur-rl-experiments/ur5/ur5_rl/config/reach_goal_params.yaml'
    with open(params_path, "r") as params_stream:
        env_kwargs = yaml.safe_load(params_stream)
        env_kwargs['pub_topic_name'] = f'/{env_kwargs["controllers_list"][0]}/command'
        kwargs = {'nenvs': NENVS,
                  'env_cls': UR5EnvGoal,
                  'gazebo_ports': gazebo_ports,
                  'launch_files': launch_files,
                  'env_kwargs': env_kwargs}
        master = GazeboMaster(**kwargs)
        master.start()
        print(master.reset())
        for i in range(10000):
            if i % 500 == 0:
                master.reset()
            master.step(6 * torch.rand(size=(NENVS, 6)) - 3)
