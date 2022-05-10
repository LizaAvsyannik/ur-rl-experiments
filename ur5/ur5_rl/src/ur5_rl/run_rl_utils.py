import torch


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
    trajectory['value_targets'] = targets


def run_policy(env, policy, postprocessor, n_steps=2):
    total_steps = 0
    trajectory = run_episode(env, policy, n_steps=n_steps)
    total_steps += len(trajectory['observations'])
    add_value_targets(trajectory)
    postprocessor(trajectory)
    return trajectory