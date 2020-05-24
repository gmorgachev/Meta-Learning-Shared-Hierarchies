import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class MLSHAgent(nn.Module):
    def __init__(self, n_subpolicies, n_actions, obs_shape):
        super(self.__class__, self).__init__()
        self.n_actions = n_actions
        self.master_policy = MasterPolicy(n_subpolicies, obs_shape)
        self.subpolicies = nn.ModuleList([
            SubPolicy(n_actions, obs_shape) for _ in range(n_subpolicies)
        ])

    def forward_master(self, obs):
        return  self.master_policy(obs)

    def forward_sub(self, idxs, obs):
        assert len(idxs) == obs.shape[0]
        logits = []
        state_value = []
        for idx, ob in zip(idxs, obs):
            logit, v = self.subpolicies[idx](ob.clone())
            logits.append(logit)
            state_value.append(v)

        logits = torch.stack(logits)
        state_value = torch.cat(state_value)

        return logits, state_value

    def sample_actions(self, agent_outputs):
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)

        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step_master(self, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)
        l, s = self.forward_master(obs_img)
        return l, s

    def step(self, idxs, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)
        l, s = self.forward_sub(idxs, obs_img)

        return l, s


class SubPolicy(nn.Module):
    def __init__(self, n_actions, obs_shape):
        super(SubPolicy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        state_value = self.critic(x)

        return logits, state_value


class MasterPolicy(nn.Module):
    def __init__(self, n_subpolicies, obs_shape):
        super(MasterPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, n_subpolicies)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        state_value = self.critic(x)

        return logits, state_value
