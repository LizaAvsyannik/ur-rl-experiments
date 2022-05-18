#!/usr/bin/env python3

import wandb
import time

import gym
import rospy
import numpy as np
import torch
from torch import optim
from ur5_rl.envs.task_envs import UR5EnvGoal

from ur5_rl.algorithms.a2c import A2C, A2CModel, A2CPolicy, MergeTimeBatch

WANDB_RUN_NAME = 'changed_policy_loss'
WANDB_MODEL_CHECKPOINT_NAME = 'changed_policy_loss-checkpoint'


def read_params():
    config = {}
    config['n_episodes'] = rospy.get_param("/n_episodes")
    config['n_steps_per_episode'] = rospy.get_param("/n_steps_per_episode")
    config['learning_rate'] = rospy.get_param("/learning_rate")
    config['ckpt_freq'] = rospy.get_param("/ckpt_freq")
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['ckpt_file'] = rospy.get_param('ckpt_file', '')

    controllers_list = rospy.get_param("/controllers_list")
    joint_names = rospy.get_param("/joint_names")
    link_names = rospy.get_param("/link_names")

    joint_limits = {}
    joint_limits['lower'] = list(
        map(lambda x: x * np.pi, rospy.get_param("/joint_limits/lower")))
    joint_limits['upper'] = list(
        map(lambda x: x * np.pi, rospy.get_param("/joint_limits/upper")))

    target_limits = {}
    target_limits['radius'] = rospy.get_param("/target_limits/radius")
    target_limits['target_size'] = rospy.get_param(
        "/target_limits/target_size")

    return config, controllers_list, \
        joint_names, link_names, joint_limits, target_limits


def run_episode(env, policy, n_steps):
    obs = env.reset()
    trajectory = {}

    for _ in range(n_steps):
        obs = obs.unsqueeze(0).to(policy.device)
        step_results = {'observations': obs}

        policy_result = policy.act(obs)
        step_results.update(policy_result)

        obs, reward, done, info = env.step(policy_result['actions'])
        step_results['rewards'] = torch.Tensor([reward]).to(policy.device)
        step_results['done'] = torch.ByteTensor([done]).to(policy.device)
        step_results['distances'] = torch.Tensor(
            [info['distance']]).to(policy.device)

        for k, v in step_results.items():
            if k not in trajectory:
                trajectory[k] = [v]
            else:
                trajectory[k].append(v)

        if done:
            break

    return trajectory


def add_value_targets(trajectory, gamma=0.99):  # compute the returns
    rewards = trajectory['rewards']
    targets = torch.zeros_like(torch.vstack(rewards))
    ret = 0
    for t in reversed(range(len(rewards))):
        ret = rewards[t] + gamma * ret
        targets[t] = ret
    trajectory['value_targets'] = targets


def run_policy(env, policy, postprocessor, n_steps=100):
    trajectory = run_episode(env, policy, n_steps=n_steps)
    add_value_targets(trajectory)
    postprocessor(trajectory)
    return trajectory


def main():
    rospy.init_node('ur_gym', anonymous=False, log_level=rospy.INFO)

    rospy.loginfo('Reading parameters...')
    config, controllers_list, joint_names, link_names, joint_limits, target_limits = read_params()
    rospy.loginfo('Finished reading parameters')

    wandb_run = wandb.init(project="my-rl-lapka", entity="liza-avsyannik",
                           name=WANDB_RUN_NAME, config=config)

    kwargs = {'controllers_list': controllers_list, 'joint_limits': joint_limits, 'link_names': link_names,
              'target_limits': target_limits, 'pub_topic_name': f'/{controllers_list[0]}/command'}
    env = gym.make('UR5EnvGoal-v0', **kwargs)

    start_ep = 0
    model = A2CModel(env.state_dim(), env.action_dim()).to(wandb.config.device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    if wandb.config.ckpt_file:
        rospy.loginfo(f'Loading model from {wandb.config.ckpt_file}')
        saved_state = torch.load(wandb.config.ckpt_file)
        start_ep = saved_state['episode']
        model.load_state_dict(saved_state['model_state'])
        optimizer.load_state_dict(saved_state['optimizer_state'])

    policy = A2CPolicy(model, wandb.config.device)
    a2c = A2C(policy, optimizer, action_norm_coef=3e-2, entropy_coef=1e-2)
    postprocessor = MergeTimeBatch(wandb.config.device)

    rospy.loginfo('Starting training loop')
    for ep in range(start_ep, wandb.config.n_episodes):
        env.reset()
        trajectory = run_policy(
            env, policy, postprocessor, wandb.config.n_steps_per_episode)
        step_results = a2c.step(trajectory)

        # time.sleep(3)
        wandb.log({'Number of steps': trajectory['rewards'].shape[0],
                   'Mean reward': torch.mean(trajectory['rewards']),
                   'Last step reward': trajectory['rewards'][-1],
                   'Mean distance': torch.mean(trajectory['distances']),
                   'Last step distance': trajectory['distances'][-1],
                   'Value loss': step_results['value_loss'],
                   'Policy loss': step_results['policy_loss'],
                   'Action norm': step_results['action_norm'],
                   'Policy entropy': step_results['entropy'],
                   'Gradient norm': step_results['grad_norm']},
                  step=ep)

        if (ep + 1) % wandb.config.ckpt_freq == 0:
            rospy.loginfo('Saving model checkpoint')
            model_artifact = wandb.Artifact(name=WANDB_MODEL_CHECKPOINT_NAME,
                                            type='model')
            torch.save({'episode': ep,
                        'url': wandb_run.url,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        f'/home/ros/catkin_ws/{wandb_run.id}-{ep}.pth')
            model_artifact.add_file(f'/home/ros/catkin_ws/{wandb_run.id}-{ep}.pth')
            wandb.log_artifact(model_artifact)

    wandb_run.finish()
    env.close()


if __name__ == '__main__':
    main()
