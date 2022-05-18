#!/usr/bin/env python3

import wandb

import gym
import rospy
import numpy as np
import torch
from torch import optim
from math import ceil
from tqdm import tqdm
from ur5_rl.envs.task_envs import UR5EnvGoal

from ur5_rl.algorithms.ppo import PPOAgent, Policy, PPO, make_ppo_runner

WANDB_RUN_NAME = 'lapka-ppo-fixed'
WANDB_MODEL_CHECKPOINT_NAME = 'lapka-ppo-fixed'


def read_params():
    config = {}
    config['n_episodes'] = rospy.get_param("/n_episodes")
    config['n_steps_per_episode'] = rospy.get_param("/n_steps_per_episode")
    config['learning_rate'] = rospy.get_param("/learning_rate")
    config['ckpt_freq'] = rospy.get_param("/ckpt_freq")
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['ckpt_file'] = rospy.get_param('ckpt_file', '')
    config['num_runner_epochs'] = rospy.get_param('num_runner_epochs')
    config['num_runner_minibatches'] = rospy.get_param('num_runner_minibatches')

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
    model = PPOAgent(env.state_dim(), env.action_dim()).to(wandb.config.device)
    
    if wandb.config.ckpt_file:
        rospy.loginfo(f'Loading model from {wandb.config.ckpt_file}')
        saved_state = torch.load(wandb.config.ckpt_file)
        start_ep = saved_state['episode']
        model.load_state_dict(saved_state['model_state'])
        optimizer.load_state_dict(saved_state['optimizer_state'])

    policy = Policy(model, wandb.config.device)
    runner = make_ppo_runner(env, policy,
                             num_runner_steps=wandb.config.n_steps_per_episode,
                             num_epochs=wandb.config.num_runner_epochs,
                             num_minibatches=wandb.config.num_runner_minibatches)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, eps=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda epoch: (wandb.config.n_episodes - epoch) / \
                                                            (wandb.config.n_episodes - epoch + 1))
    ppo = PPO(policy, optimizer)

    rospy.loginfo('Starting training loop')
    MINIBATCHES_IN_EPOCH = wandb.config.num_runner_epochs * wandb.config.num_runner_minibatches

    for ep in tqdm(range(start_ep, wandb.config.n_episodes), desc='Episode'):
        for mb_idx in range(MINIBATCHES_IN_EPOCH):
            trajectory = runner.get_next()
            losses = ppo.step(trajectory)

        wandb.log({'Number of steps': trajectory['rewards'].shape[0],
                   'Mean reward': torch.mean(trajectory['rewards']),
                   'Last step reward': trajectory['rewards'][-1],
                   'Mean distance': torch.mean(trajectory['distances']),
                   'Last step distance': trajectory['distances'][-1],
                   'Value loss': losses['loss/value'],
                   'Policy loss': losses['loss/policy'],
                   'Action norm': torch.mean(torch.norm(trajectory['actions'].detach().cpu(), dim=-1)),
                   'Policy entropy': losses['policy/entropy'],
                   'Gradient norm': losses['policy/grad_norm']},
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

        sched.step()
        

    wandb_run.finish()
    env.close()


if __name__ == '__main__':
    main()
