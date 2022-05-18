#!/usr/bin/env python3

import wandb
from tqdm import tqdm

import gym
import rospy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from ur5_rl.envs.task_envs import UR5EnvGoal

from ur5_rl.algorithms.dqn import DQNAgent, compute_td_loss
from ur5_rl.algorithms.replay_buffer import ReplayBuffer
from ur5_rl.dqn_utils import is_enough_ram, wait_for_keyboard_interrupt, linear_decay

WANDB_RUN_NAME = 'dqn-vel-1'
WANDB_MODEL_CHECKPOINT_NAME = 'dqn-vel-1-checkpoint'


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
    
    return config, controllers_list, joint_names, \
           link_names, joint_limits, target_limits


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer. 
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for step in range(n_steps):
        q = agent.get_qvalues([s])
        a = agent.sample_actions(q)[0]
        sp, r, done, _ = env.step(a)
        exp_replay.add(s, a, r, sp, done)
        if done:
            s = env.reset()
        else:
            s = sp

    return sum_rewards, s


def create_replay_buffer(state, agent, env, n_steps=1, replay_buffer_size=10 ** 4):
    exp_replay = ReplayBuffer(replay_buffer_size)
    env.reset()
    for i in range(replay_buffer_size // n_steps):
        if not is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available. 
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
                )
            break
        play_and_record(state, agent, env, exp_replay, n_steps=n_steps)
        if len(exp_replay) == replay_buffer_size:
            break
    return exp_replay


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


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

    agent = DQNAgent(env.state_dim(), len(env.actions) ** env.action_dim(), epsilon=1).to(wandb.config.device)
    optimizer = optim.Adam(agent.parameters(), lr=wandb.config.learning_rate)

    if wandb.config.ckpt_file:
        rospy.loginfo(f'Loading model from {wandb.config.ckpt_file}')
        saved_state = torch.load(wandb.config.ckpt_file)
        start_ep = saved_state['episode']
        agent.load_state_dict(saved_state['model_state'])
        optimizer.load_state_dict(saved_state['optimizer_state'])

    target_network = DQNAgent(env.state_dim(), len(env.actions) ** env.action_dim()).to(wandb.config.device)
    target_network.load_state_dict(agent.state_dict())

    exp_replay = create_replay_buffer(env.reset(), agent, env, n_steps=wandb.config.n_steps_per_episode)

    timesteps_per_epoch = 1
    batch_size = 16
    total_steps = 3 * 10**6
    decay_steps = 10**6

    init_epsilon = 1
    final_epsilon = 0.1

    refresh_target_network_freq = 5000
    eval_freq = 5000

    max_grad_norm = 50

    n_lives = 5

    rospy.loginfo('Starting training loop')
    state = env.reset()
    for ep in tqdm(range(start_ep, total_steps)):

        if not is_enough_ram():
            print('less that 100 Mb RAM available, freezing')
            print('make sure everything is ok and use KeyboardInterrupt to continue')
            wait_for_keyboard_interrupt()
        
        agent.epsilon = linear_decay(init_epsilon, final_epsilon, ep, decay_steps)

        # play
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        s, a, rw, ns, done = exp_replay.sample(batch_size)

        loss = compute_td_loss(s, a, rw, ns, done, agent, target_network, device=wandb.config.device)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if ep % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())
        
        wandb.log({'Loss': loss.data.cpu().item(),
                   'Gradient norm': grad_norm},
                  step=ep)

        if ep % eval_freq == 0:
            wandb.log({'Mean evaluation reward': evaluate(env, agent, n_games=3 * n_lives, greedy=True, t_max=100)},
                        step=ep)

        if (ep + 1) % wandb.config.ckpt_freq == 0:
            rospy.loginfo('Saving model checkpoint')
            model_artifact = wandb.Artifact(name=WANDB_MODEL_CHECKPOINT_NAME,
                                            type='model')
            torch.save({'episode': ep,
                        'url': wandb_run.url,
                        'model_state': agent.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        f'/home/ros/catkin_ws/{wandb_run.id}-{ep}.pth')
            model_artifact.add_file(f'/home/ros/catkin_ws/{wandb_run.id}-{ep}.pth')
            wandb.log_artifact(model_artifact)
        
    env.close()


if __name__ == '__main__':
    main()
