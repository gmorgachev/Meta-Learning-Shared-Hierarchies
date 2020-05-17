import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from src.env_pool import EnvPool


class A2CAlgo:
    def __init__(self, params, device, n_actions, gamma, max_grad_norm,
                 entropy_coef=0.01, lr=0.01, value_loss_coef=0.5):
        self.params = list(params)
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def step(self,
             actions,
             rewards,
             is_not_done,
             logits,
             state_values):

        actions = torch.tensor(np.array(actions),
                               dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards),
                               dtype=torch.float32).to(self.device)
        is_not_done = torch.tensor(np.array(is_not_done),
                                   dtype=torch.float32).to(self.device)

        rewards = EnvPool.discount_with_dones(rewards, is_not_done, self.gamma)

        rollout_length = rewards.shape[1] - 1
        probas = F.softmax(logits, dim=2)
        logprobas = F.log_softmax(logits, dim=2)

        actions_one_hot = self.to_one_hot(actions, self.n_actions).view(
            actions.shape[0], actions.shape[1], self.n_actions)
        logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

        actor_loss = 0

        for t in reversed(range(rollout_length)):
            advantage = self.get_advantage(
                rewards,
                t,
                state_values).detach()

            actor_loss += -(logprobas_for_actions[:, t] * advantage).mean()

        entropy_reg = -(probas * logprobas).sum(-1).mean()
        actor_loss /= rollout_length
        critic_loss = (state_values - rewards).pow(2).mean()
        loss = actor_loss + \
               self.value_loss_coef * critic_loss + \
               -self.entropy_coef * entropy_reg

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()

        return loss.data.numpy(), grad_norm.data.numpy(), entropy_reg.data.numpy(), state_values.data.numpy(),\
               actor_loss.data.numpy(), critic_loss.data.numpy()

    def get_advantage(self, rewards, t, values):
        return rewards[:, t] + self.gamma*values[:, t+1] - values[:, t]

    def to_one_hot(self, y, n_dims=None):
        """ Take an integer tensor and convert it to 1-hot matrix. """
        y_tensor = y.to(dtype=torch.int64).reshape(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1).to(self.device)

        return y_one_hot