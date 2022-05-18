import torch
from torch import nn
import numpy as np
from math import ceil
from ur5_rl.algorithms.runner import EnvRunner

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPOAgent(nn.Module):
    def __init__(self, state_dim, n_actions, eps=1e-8):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.eps = eps
        self.policy_nn = nn.Sequential(nn.Linear(self.state_dim, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 64),
                                       nn.ReLU(),
                                       nn.Linear(64, self.n_actions),
                                       nn.Tanh())
        self.value_nn = nn.Sequential(nn.Linear(self.state_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.var_nn = nn.Sequential(nn.Linear(self.state_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.n_actions),
                                    nn.Sigmoid())
        self.__initialize_net_weights(self.policy_nn)
        self.__initialize_net_weights(self.value_nn)
        self.__initialize_net_weights(self.var_nn)

    def __initialize_net_weights(self, net):
        for p in net.parameters():
            if p.ndim < 2:
                nn.init.zeros_(p)
            else:
                nn.init.orthogonal_(p, 2 ** 0.5)
                
    def forward(self, observations):
        policy_mean = self.policy_nn(observations)
        var = self.var_nn(observations)
        value = self.value_nn(observations)
        return policy_mean, var + self.eps, value


class Policy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def estimate_v(self, inputs):
        with torch.no_grad():
            self.model.eval()
            return self.model.value_nn(inputs.unsqueeze(0))
    
    def act(self, inputs, training=False):
        # Should return a dict.
        if isinstance(inputs, np.ndarray):
            inputs = torch.Tensor(inputs).to(self.device)
        if training:
            self.model.train()
            mean, cov, value = self.model(inputs)
            print(mean, cov)
            return {'distribution': torch.distributions.MultivariateNormal(mean, torch.diag_embed(cov)),
                    'values': value}
        else:
            with torch.no_grad():
                self.model.eval()
                mean, cov, value = self.model(inputs.unsqueeze(0))
                dist = torch.distributions.MultivariateNormal(mean.squeeze(), torch.diag_embed(cov.squeeze()))
                actions = dist.sample()
                return {'actions': actions.cpu().numpy(),
                        'log_probs': dist.log_prob(actions).cpu().numpy(),
                        'values': value.squeeze().cpu().numpy()}


class PPO:
    def __init__(self, policy, optimizer,
                 cliprange=0.2,
                 value_loss_coef=0.25,
                 entropy_coef=0.1,
                 max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def policy_loss(self, trajectory, act):
        """ Computes and returns policy loss on a given trajectory. """
        dist = act['distribution']
        log_probs = dist.log_prob(trajectory['actions'])
        entropy = torch.mean(dist.entropy())
        quotient = torch.exp(log_probs - trajectory['log_probs'])
        A = trajectory['advantages']
        J = quotient * A
        J_clipped = torch.clamp(quotient,
                                1 - self.cliprange,
                                1 + self.cliprange) * A
        J_stacked = torch.cat((J.unsqueeze(-1), J_clipped.unsqueeze(-1)), dim=-1)
        return -torch.mean(torch.min(J_stacked, dim=-1).values), entropy
      
    def value_loss(self, trajectory, act):
        """ Computes and returns value loss on a given trajectory. """
        old_values = trajectory['values']
        predicted_values = act['values']
        target_values = trajectory['value_targets'].unsqueeze(-1)
        l_simple = (predicted_values - target_values) ** 2
        clipped_diff = torch.clamp(predicted_values - old_values,
                                   -self.cliprange,
                                   self.cliprange)
        l_clipped = (old_values + clipped_diff - target_values) ** 2
        l_stacked = torch.cat((l_simple, l_clipped), dim=-1)
        return torch.mean(torch.max(l_stacked, dim=-1).values)
      
    def loss(self, trajectory):
        act = self.policy.act(trajectory["observations"], training=True)
        policy_loss, entropy = self.policy_loss(trajectory, act)
        value_loss = self.value_loss(trajectory, act)
        return {'loss/total': policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy,
                'loss/policy': policy_loss.detach().cpu().item(),
                'loss/value': value_loss.detach().cpu().item(),
                'policy/entropy': entropy}
    
    def step(self, trajectory):
        """ Computes the loss function and performs a single gradient step. """
        self.optimizer.zero_grad()
        loss_dict = self.loss(trajectory)
        loss_dict['loss/total'].backward()
        loss_dict['policy/grad_norm'] = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        loss_dict['loss/total'] = loss_dict['loss/total'].detach().cpu().item()
        return loss_dict


class GAE:
    """ Generalized Advantage Estimator. """
    def __init__(self, policy, gamma=0.99, lambda_=0.95):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        
    def __call__(self, trajectory):
        N = trajectory['values'].shape[0]
        rewards = trajectory['rewards']
        values = trajectory['values'].squeeze()
        resets = trajectory['resets'].byte()
        latest_obs = torch.Tensor(trajectory['state']['latest_observation']).to(self.policy.device)
        # training to avoid creating tensor
        latest_v = self.policy.estimate_v(latest_obs).squeeze().unsqueeze(0)
        values = torch.cat((values, latest_v), dim=0)
        advantages = [torch.zeros_like(latest_v)]
        advantage_zeros = torch.zeros_like(advantages[0])
        delta_zeros = torch.zeros_like(rewards[0])
        for t in range(N - 1, -1, -1):
            r_t = rewards[t]
            V_t = values[t]
            V_next = values[t + 1]
            delta_t = r_t + torch.where(resets[t], delta_zeros, self.gamma * V_next) - V_t
            advantage = delta_t.unsqueeze(0)
            advantage += torch.where(resets[t].unsqueeze(0),
                                     advantage_zeros,
                                     self.gamma * self.lambda_ * advantages[-1])
            advantages.append(advantage)
        advantages = advantages[::-1]
        trajectory['advantages'] = torch.cat(advantages[:-1], dim=0).squeeze()
        trajectory['value_targets'] = trajectory['advantages'] + values[:-1]   


class TrajectorySampler:
    """ Samples minibatches from trajectory for a number of epochs. """
    def __init__(self, runner, num_epochs, num_minibatches, transforms=None):
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None
        self.trajectory_length = 0
        self.permutation = None
        self.sample_trajectory()
        
    def sample_trajectory(self):
        self.trajectory = self.runner.get_next()
        self.trajectory_length = self.trajectory['actions'].shape[0]
        self.shuffle_trajectory()
                    
    def choose_minibatch(self, idx):
        mb_size = ceil(self.trajectory_length / self.num_minibatches)
        permutation_slice = self.permutation[idx*mb_size:(idx+1)*mb_size]
        minibatch = {}
        for k, v in self.trajectory.items():
            if k != 'state':
                minibatch[k] = v[permutation_slice]
        return minibatch
    
    def shuffle_trajectory(self):
        """ Shuffles all elements in trajectory.

        Should be called at the beginning of each epoch.
        """
        self.permutation = torch.randperm(self.trajectory_length)
    
    def get_next(self):
        """ Returns next minibatch.  """
        if self.minibatch_count == self.num_minibatches:
            self.minibatch_count = 0
            self.epoch_count += 1
            self.shuffle_trajectory()
        if self.epoch_count == self.num_epochs:
            self.epoch_count = 0
            self.sample_trajectory()
        minibatch = self.choose_minibatch(self.minibatch_count)
        self.minibatch_count += 1
        for tf in self.transforms:
            tf(minibatch)
        return minibatch


class NormalizeAdvantages:
    """ Normalizes advantages to have zero mean and variance 1. """
    def __call__(self, trajectory):
        adv = trajectory["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        trajectory["advantages"] = adv


class AsArray:
    """ 
    Converts lists of interactions to ndarray.
    """
    def __call__(self, trajectory):
        # Modify trajectory inplace. 
        for k, v in filter(lambda kv: kv[0] != "state",
                           trajectory.items()):
            if not torch.is_tensor(v[0]):
                trajectory[k] = np.asarray(v)


class AsTensor:
    """ 
    Converts lists of interactions to DEVICE torch.Tensor.
    """
    def __call__(self, trajectory):
        # Modify trajectory inplace. 
        for k, v in filter(lambda kv: kv[0] != "state",
                           trajectory.items()):
            if not torch.is_tensor(v[0]):
                trajectory[k] = torch.Tensor(v).to(DEVICE)
            else: 
                trajectory[k] = torch.vstack(v).to(DEVICE)


class FlattenTrajectory:
    def __call__(self, trajectory):
        n_steps, n_envs = trajectory['values'].shape[:2]
        for k, v in filter(lambda kv: kv[0] != "state",
                           trajectory.items()):
            trajectory[k] = trajectory[k].reshape(n_steps * n_envs, -1).squeeze()


def make_ppo_runner(env, policy, num_runner_steps=2048,
                    gamma=0.99, lambda_=0.95, 
                    num_epochs=10, num_minibatches=32):
    """ Creates runner for PPO algorithm. """
    runner_transforms = [AsArray(),
                         AsTensor(),  # changed this to better suit torch
                         GAE(policy, gamma=gamma, lambda_=lambda_)]
    runner = EnvRunner(env, policy, num_runner_steps,
                       transforms=runner_transforms)
  
    sampler_transforms = [NormalizeAdvantages()]
    sampler = TrajectorySampler(runner, num_epochs=num_epochs, 
                                num_minibatches=num_minibatches,
                                transforms=sampler_transforms)
    return sampler