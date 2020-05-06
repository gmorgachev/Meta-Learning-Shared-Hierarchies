import numpy as np
import torch

from torch import nn
from torch.nn import functional as F


class A2CAlgo:
    def __init__(self, agent, device, n_actions):
        self.opt = torch.optim.Adam(agent.parameters())
        self.device = device
        self.n_actions = n_actions
        self.agent = agent.to(self.device)

    def train_on_rollout(self,
                         states,
                         actions,
                         rewards,
                         is_not_done,
                         prev_memory_states,
                         gamma=0.99,
                         max_grad_norm=90):
        # shape: [batch_size, time, c, h, w]
        states = torch.tensor(np.asarray(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)  # shape: [batch_size, time]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)  # shape: [batch_size, time]
        is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32).to(self.device)  # shape: [batch_size, time]
        rollout_length = rewards.shape[1] - 1

        # predict logits, probas and log-probas using an agent.
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

        # select log-probabilities for chosen actions, log pi(a_i|s_i)
        actions_one_hot = self.to_one_hot(actions, self.n_actions).view(
            actions.shape[0], actions.shape[1], self.n_actions)
        logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

        J_hat = 0
        value_loss = 0
        cumulative_returns = state_values[:, -1].detach()

        for t in reversed(range(rollout_length)):
            r_t = rewards[:, t]
            V_t = state_values[:, t]
            V_next = state_values[:, t + 1].detach()
            logpi_a_s_t = logprobas_for_actions[:, t]
            cumulative_returns = r_t + gamma * cumulative_returns
            value_loss += torch.mean((r_t + gamma * V_next - V_t) ** 2)
            mask = is_not_done[:, t]
            advantage = mask * (rewards[:, t:] * gamma ** torch.arange(rewards.shape[-1] - t)).sum(1) \
                        - V_t \
                        - cumulative_returns * (gamma ** rewards.shape[-1] - t)
            advantage = advantage.detach()
            J_hat += torch.mean(logpi_a_s_t * advantage)

        entropy_reg = -torch.mean((probas * logprobas).sum(-1))

        loss = -J_hat / rollout_length + \
               value_loss / rollout_length + \
               -entropy_reg

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
        self.opt.step()
        self.opt.zero_grad()

        return loss.data.numpy(), grad_norm, entropy_reg.data.numpy()

    @staticmethod
    def to_one_hot(y, n_dims=None):
        """ Take an integer tensor and convert it to 1-hot matrix. """
        y_tensor = y.to(dtype=torch.int64).reshape(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)

        return y_one_hot