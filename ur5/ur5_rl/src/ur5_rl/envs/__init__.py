from importlib.metadata import entry_points
from .robot_gazebo_env_goal import RobotGazeboEnv
from .controllers_publishers import JointGroupPublisher

import gym

gym.envs.register(
    id='UR5EnvGoal-v0',
    entry_point='ur5_rl.envs.task_envs:UR5EnvGoal',     
    kwargs = {'controllers_list': [], 
     'joint_names': [],
     'joint_limits': [], 
     'link_names': [],
     'target_limits': [], 
     'pub_topic_name': ''}
)
