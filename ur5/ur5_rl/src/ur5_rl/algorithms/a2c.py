import torch
from torch import nn
from torch.distributions import MultivariateNormal
import rospy

from .nn_utils import LinearBlock


class A2CModel(nn.Module):
    def __init__(self, state_dim, n_actions, eps=1e-8):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.eps = eps
        self.__munet = nn.Sequential(LinearBlock(state_dim, 64),
                                     nn.GELU(),
                                     LinearBlock(64, 128),
                                     nn.GELU(),
                                     LinearBlock(128, n_actions, n_layers=2),
                                     nn.Tanh())
        self.__varnet = nn.Sequential(LinearBlock(state_dim, 64),
                                      nn.GELU(),
                                      LinearBlock(64, 128),
                                      nn.GELU(),
                                      LinearBlock(128, n_actions, n_layers=2),
                                      nn.Sigmoid())
        self.__vnet = nn.Sequential(LinearBlock(state_dim, 64),
                                    nn.ReLU(),
                                    LinearBlock(64, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1))
        self.__initialize_net_weights(self.__munet, scale=2**0.5)
        self.__initialize_net_weights(self.__varnet, scale=2**0.5)
        self.__initialize_net_weights(self.__vnet, scale=2**0.5)

    def __initialize_net_weights(self, net, scale):
        for p in net.parameters():
            if p.ndim < 2:
                nn.init.zeros_(p)
            else:
                nn.init.orthogonal_(p, scale)

    def forward(self, inputs):
        mus = self.__munet(inputs)
        vars = self.__varnet(inputs)
        v = self.__vnet(inputs)
        return mus, vars + self.eps, v


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
        print(mus, vars)
        cov = torch.diag_embed(3 * vars + self.eps)
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
        trajectory['rewards'] = trajectory['rewards'].ravel()
        trajectory['value_targets'] = trajectory['value_targets'].ravel()
        for k, v in trajectory.items():
            if k not in ['actions', 'actions_raw', 'rewards', 'value_targets']:
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
        advantage = self.__advantage(trajectory['value_targets'].detach(),
                                     trajectory['values'].detach())
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
        # a2c_loss = value_loss + self.action_norm_coef * action_norm - self.entropy_coef * entropy
        return {'a2c': a2c_loss, 'policy_loss': policy_loss, 'value_loss': value_loss, 'action_norm': action_norm, 'entropy': entropy}

    def step(self, trajectory):
        self.optimizer.zero_grad()
        loss_dict = self.loss(trajectory)
        loss = loss_dict['a2c']
        loss.backward()
        loss_dict['grad_norm'] = nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return dict([(k, v.detach().cpu().item()) for k, v in loss_dict.items()])
