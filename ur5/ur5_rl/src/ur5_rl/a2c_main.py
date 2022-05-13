 #!/usr/bin/env python3

import wandb

import gym
import rospy
import numpy as np
import torch
from torch import optim
from ur5_rl.envs.task_envs import UR5EnvGoal

from ur5_rl.algorithms.a2c import A2C, A2CModel, A2CPolicy, MergeTimeBatch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = 'cpu'
obs_dim = 15
n_act = 6

WANDB_RUN_NAME = 'my-lapka-first-steps'
WANDB_MODEL_CHECKPOINT_NAME = 'my-lapka-first-checkpoint'

def read_params():
    config = {}
    config['n_episodes'] = rospy.get_param("/n_episodes")
    config['n_steps_per_episode'] = rospy.get_param("/n_steps_per_episode")
    config['learning_rate'] = rospy.get_param("/learning_rate")
    config['ckpt_freq'] = rospy.get_param("/ckpt_freq")

    controllers_list = rospy.get_param("/controllers_list")
    joint_names = rospy.get_param("/joint_names")
    link_names = rospy.get_param("/link_names")

    joint_limits = {}
    joint_limits['lower'] =  list(map(lambda x: x * np.pi , rospy.get_param("/joint_limits/lower")))
    joint_limits['upper'] =  list(map(lambda x: x * np.pi , rospy.get_param("/joint_limits/upper")))

    target_limits  = {}
    target_limits['lower'] = rospy.get_param("/target_limits/lower")
    target_limits['upper'] = rospy.get_param("/target_limits/upper")
    target_limits['target_size'] = rospy.get_param("/target_limits/target_size")

    return config, controllers_list, \
           joint_names, link_names, joint_limits, target_limits 


def run_episode(env, policy, n_steps=2):
    obs = env.reset()
    trajectory = {'observations': [], 'actions': [], 'log_probs': [], 'entropy': [],  'values': [],
                  'rewards': [], 'done': []}
    
    for _ in range(n_steps):
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim))
        step_results = {'observations': obs}

        policy_result = policy.act(obs)
        step_results.update(policy_result)
        
        obs, reward, done, _ = env.step(policy_result['actions'][0])
        step_results['rewards'] = torch.Tensor([reward])
        step_results['done'] = torch.ByteTensor([done])
        
        for k, v in step_results.items():
            trajectory[k].append(v)

        if done:
            break

    return trajectory


def add_value_targets(trajectory, gamma=0.99): # compute the returns
    rewards = trajectory['rewards']
    targets = torch.zeros_like(torch.vstack(rewards))
    ret = 0
    for t in reversed(range(len(rewards))):
        ret = rewards[t] + gamma * ret
        targets[t] = ret
    trajectory['value_targets'] = targets.to(DEVICE)


def run_policy(env, policy, postprocessor, n_steps=2):
    total_steps = 0
    trajectory = run_episode(env, policy, n_steps=n_steps)
    total_steps += len(trajectory['observations'])
    add_value_targets(trajectory)
    postprocessor(trajectory)
    return trajectory


def main():
    rospy.init_node('ur_gym', anonymous=False, log_level=rospy.DEBUG)

    rospy.logdebug('Reading parameters...')
    config, controllers_list, joint_names, link_names, joint_limits, target_limits = read_params()
    rospy.logdebug('Finished reading parameters')

    wandb.init(project="my-rl-lapka", entity="liza-avsyannik", name=WANDB_RUN_NAME, config=config)

    kwargs = {'controllers_list': controllers_list, 'joint_names': joint_names, 'joint_limits': joint_limits, 'link_names': link_names, 
              'target_limits': target_limits, 'pub_topic_name': f'{controllers_list[0]}/command'}
    env = gym.make('UR5EnvGoal-v0', **kwargs)
    
    model = A2CModel(obs_dim, n_act).to(DEVICE)
    policy = A2CPolicy(model, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    a2c = A2C(policy, optimizer)
    postprocessor = MergeTimeBatch(DEVICE)

    rospy.loginfo('Starting training loop')
    for ep in range(wandb.config.n_episodes):
        env.reset()
    #     trajectory = run_policy(env, policy, postprocessor, wandb.config.n_steps_per_episode)
    #     step_results = a2c.step(trajectory)
    #     wandb.log({'Number of steps': trajectory['rewards'].shape[0],
    #                'Mean reward': torch.mean(trajectory['rewards']),
    #                'Value loss': step_results['value_loss'],
    #                'Policy loss': step_results['policy_loss'],
    #                'Policy entropy': step_results['entropy']},
    #                step=ep)

    #     if (ep + 1) % wandb.config.ckpt_freq == 0:
    #         model_artifact = wandb.Artifact(name=WANDB_MODEL_CHECKPOINT_NAME,
    #                                         type='model')
    #         torch.save(model, f'{ep}.pth')
    #         model_artifact.add_file(f'{ep}.pth')
    #         wandb.log_artifact(model_artifact)


    # wandb.finish()
    # env.close()


if __name__ == '__main__':
    main()
