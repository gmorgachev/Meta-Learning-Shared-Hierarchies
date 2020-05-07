import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


class A2CAlgo:
    def __init__(self, agent, device, n_actions, gamma, max_grad_norm,
                 entropy_coef=0.01, lr=0.01, value_loss_coef=0.5):
        self.opt = torch.optim.AdamW(agent.parameters(), lr=lr)
        self.device = device
        self.n_actions = n_actions
        self.agent = agent.to(self.device)
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def step(self,
             states,
             actions,
             rewards,
             is_not_done,
             prev_memory_states):

        states = torch.tensor(np.asarray(states),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions),
                               dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards),
                               dtype=torch.float32).to(self.device)
        is_not_done = torch.tensor(np.array(is_not_done),
                                   dtype=torch.float32).to(self.device)
        rollout_length = rewards.shape[1] - 1
        memory = [m.detach() for m in prev_memory_states]

        logits = []  # append logit sequence here
        state_values = []  # append state values here
        for t in range(rewards.shape[1]):
            obs_t = states[:, t]
            memory, (logits_t, values_t) = self.agent(memory, obs_t)
            logits.append(logits_t)
            state_values.append(values_t)

        logits = torch.stack(logits, dim=1)
        state_values = torch.stack(state_values, dim=1)
        probas = F.softmax(logits, dim=2)
        logprobas = F.log_softmax(logits, dim=2)

        actions_one_hot = self.to_one_hot(actions, self.n_actions).view(
            actions.shape[0], actions.shape[1], self.n_actions)
        logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

        actor_loss = 0
        critic_loss = 0
        advantange = 0

        for t in reversed(range(rollout_length)):
            advantage = self.get_advantage(
                is_not_done,
                rewards,
                t,
                advantange,
                state_values).detach()

            critic_loss += (state_values[:, t] - rewards[:, t]).pow(2).mean()
            actor_loss += -(logprobas_for_actions[:, t] * advantage).mean()

        entropy_reg = -(probas * logprobas).sum(-1).mean()
        loss = actor_loss / rollout_length + \
               self.value_loss_coef * critic_loss / rollout_length + \
               -self.entropy_coef * entropy_reg

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()

        return loss.data.numpy(), grad_norm, entropy_reg.data.numpy()

    def get_advantage(self, masks, rewards, t, next_advantage, values, gae_lambda=0.95):
        return rewards[:, t] + \
               masks[:, t]*(
                (self.gamma*values[:, t+1]
                 - values[:, t]
                 - gae_lambda*next_advantage))

    @staticmethod
    def to_one_hot(y, n_dims=None):
        """ Take an integer tensor and convert it to 1-hot matrix. """
        y_tensor = y.to(dtype=torch.int64).reshape(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)

        return y_one_hot