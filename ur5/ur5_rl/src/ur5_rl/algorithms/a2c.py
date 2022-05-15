import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import typing as tp
import rospy


class A2CModel(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.__backbone = nn.Sequential(nn.Linear(self.state_dim, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 1024),
                                        nn.ReLU())
        self.__muhead = nn.Sequential(nn.Linear(1024, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, n_actions))
        self.__varhead = nn.Sequential(nn.Linear(1024, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, n_actions))
        self.__vhead = nn.Sequential(nn.Linear(1024, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))
        self.__initialize_weights()

    def __initialize_weights(self):
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.orthogonal_(p, 2 ** 0.5)
            else:
                nn.init.zeros_(p)

    def forward(self, inputs):
        internal = self.__backbone(inputs)
        mus = self.__muhead(internal)
        vars = self.__varhead(internal)
        v = self.__vhead(internal)
        return mus, vars, v


class A2CPolicy:
    def __init__(self, model, device, eps=1e-7):
        self.__model = model
        self.device = device
        self.eps = eps

    @property
    def model(self):
        return self.__model

    def act(self, inputs):
        # Implement policy by calling model, sampling actions and computing their log probs
        # Should return a dict containing keys ['actions', 'logits', 'log_probs', 'values'].
        mus, vars, values = self.__model(inputs.to(self.device))
        cov = torch.diag_embed(vars ** 2 + self.eps)
        dist = MultivariateNormal(mus, cov)
        actions = dist.rsample()
        return {'actions': actions.detach().cpu().numpy(),
                'actions_raw': actions,
                'log_probs': dist.log_prob(actions),
                'entropy': dist.entropy(),
                'values': values}
        
        
class MergeTimeBatch:
    def __init__(self, device):
        self.device = device
    
    """ Merges first two axes typically representing time and env batch. """
    def __call__(self, trajectory):
        # Modify trajectory inplace.
        trajectory['actions_raw'] = torch.vstack(trajectory['actions_raw'])
        trajectory['value_targets'] = trajectory['value_targets'].ravel()
        for k, v in trajectory.items():
            if k not in ['actions', 'actions_raw', 'value_targets']:
                trajectory[k] = torch.vstack(v).ravel()
        return trajectory


class A2C:
    def __init__(self,
                 policy,
                 optimizer,
                 value_loss_coef=0.25,
                 action_norm_coef=0.01,
                 entropy_coef=0.01,
                 max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = optimizer
        self.value_loss_coef = value_loss_coef
        self.action_norm_coef = action_norm_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def __advantage(self, values, value_targets):
        return values - value_targets
        
    def policy_loss(self, trajectory):
        advantage = self.__advantage(trajectory['values'].detach(),
                                     trajectory['value_targets'].detach())
        return torch.dot(trajectory['log_probs'], advantage) / advantage.shape[0]

    def value_loss(self, trajectory):
        advantage = self.__advantage(trajectory['values'],
                                     trajectory['value_targets'].detach())
        return torch.dot(advantage, advantage) / advantage.shape[0]

    def loss(self, trajectory):
        rospy.logdebug("Calculating episodes loss")
        policy_loss = self.policy_loss(trajectory)
        value_loss = self.value_loss(trajectory)
        action_norm = torch.mean(torch.norm(trajectory['actions_raw'], dim=-1))
        entropy = trajectory['entropy'].mean()
        a2c_loss = policy_loss + self.value_loss_coef * value_loss + self.action_norm_coef * action_norm - self.entropy_coef * entropy
        return {'a2c': a2c_loss, 'policy_loss': policy_loss, 'value_loss': value_loss, 'action_norm': action_norm, 'entropy': entropy}

    def step(self, trajectory):
        self.optimizer.zero_grad()
        loss_dict = self.loss(trajectory)
        loss = loss_dict['a2c']
        loss.backward()
        loss_dict['grad_norm'] = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return dict([(k, v.detach().cpu().item()) for k, v in loss_dict.items()])
