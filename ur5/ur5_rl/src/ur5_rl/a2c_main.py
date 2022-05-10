 #!/usr/bin/env python3

import gym
import rospy
import numpy as np
import torch
from torch import optim
from ur5_rl.envs.task_envs import UR5EnvGoal

from ur5_rl.algorithms.a2c import A2C, A2CModel, A2CPolicy
from ur5_rl.run_rl_utils import run_policy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
obs_dim = 15 
n_act = 6 

def read_params():
    n_episodes = rospy.get_param("/n_episodes")
    n_steps = rospy.get_param("/n_steps")

    controllers_list = rospy.get_param("/controllers_list")
    joint_names = rospy.get_param("/joint_names")

    joint_limits = {}
    joint_limits['lower'] =  list(map(lambda x: x * np.pi , rospy.get_param("/joint_limits/lower")))
    joint_limits['upper'] =  list(map(lambda x: x * np.pi , rospy.get_param("/joint_limits/upper")))

    target_limits  = {}
    target_limits['lower'] = rospy.get_param("/target_limits/lower")
    target_limits['upper'] = rospy.get_param("/target_limits/upper")
    target_limits['target_size'] = rospy.get_param("/target_limits/target_size")

    return n_episodes, n_steps, controllers_list, \
           joint_names, joint_limits, target_limits 


def main():
    rospy.init_node('ur_gym', anonymous=False, log_level=rospy.DEBUG)

    rospy.logdebug('Reading parameters...')
    n_episodes, n_steps, controllers_list, \
        joint_names, joint_limits, target_limits = read_params() 
    rospy.logdebug('Finished reading parameters')

    kwargs = {'controllers_list': controllers_list, 'joint_limits': joint_limits, \
              'target_limits': target_limits, 'pub_topic_name': f'{controllers_list[0]}/command'}
    env = gym.make('UR5EnvGoal-v0', **kwargs)
    
    model = A2CModel(obs_dim, n_act).to(DEVICE)
    policy = A2CPolicy(model, DEVICE)
    optimizer = optim.Adam(model.parameters())
    a2c = A2C(policy, optimizer)

    rospy.loginfo('Starting training loop')
    for ep in range(n_episodes):
        trajectory = run_policy(env, policy, DEVICE, n_steps)
        step_results = a2c.step(trajectory)
        print('[{}/{}] rewards: {:.3f}, value loss : {:.3f}, policy loss : {:.3f}, policy entropy : {:.3f}'.format(
                 ep, n_episodes,torch.sum(trajectory['rewards']).item(), step_results['value_loss'], step_results['policy_loss'],  step_results['entropy']))

    # env.reset()
    # print(env.step([0.0, -2.33, 1.57, 0.0, 0.0, -0.2]))


if __name__ == '__main__':
    main()