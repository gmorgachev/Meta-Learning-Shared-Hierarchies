import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .model import Observer


class MLSHAgent(nn.Module):
    def __init__(self, n_subpolicies, n_actions, config):
        super(self.__class__, self).__init__()
        self.n_actions = n_actions
        self.observer = Observer(config)
        self.master_policy = MasterPolicy(n_subpolicies, config)
        self.subpolicies = nn.ModuleList([
            SubPolicy(n_actions, config) for _ in range(n_subpolicies)
        ])

    def forward_master(self, prev_state, obs):
        new_state = self.observer(prev_state, obs)
        logits, state_value = self.master_policy(new_state[0].detach())

        return new_state, (logits, state_value)

    def forward_sub(self, idxs, prev_state, obs):
        assert len(idxs) == obs.shape[0]
        new_state = self.observer(prev_state, obs)
        logits = []
        state_value = []
        for idx, state in zip(idxs, new_state[0]):
            logit, v = self.subpolicies[idx](state.unsqueeze(0).clone())
            logits.append(logit)
            state_value.append(v)

        logits = torch.cat(logits)
        state_value = torch.cat(state_value)

        return new_state, (logits, state_value)

    def get_initial_state(self, batch_size):
        return torch.zeros((batch_size, self.observer.rnn.hidden_size)),\
               torch.zeros((batch_size, self.observer.rnn.hidden_size))

    def sample_actions(self, agent_outputs):
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step_master(self, prev_state, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)
        _, (l, s) = self.forward_master(prev_state, obs_img)
        return prev_state, (l, s)

    def step(self, idxs, prev_state, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)

        (h, c), (l, s) = self.forward_sub(idxs, prev_state, obs_img)
        return (h, c), (l, s)


class SubPolicy(nn.Module):
    def __init__(self, n_actions, config):
        super(SubPolicy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(config["emb_dim"], 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(config["emb_dim"], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        state_value = self.critic(x)[:, 0]

        return logits, state_value


class MasterPolicy(nn.Module):
    def __init__(self, n_subpolicies, config):
        super(MasterPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(config["emb_dim"], 64),
            nn.Tanh(),
            nn.Linear(64, n_subpolicies)
        )

        self.critic = nn.Sequential(
            nn.Linear(config["emb_dim"], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        state_value = self.critic(x)[:, 0]

        return logits, state_value
