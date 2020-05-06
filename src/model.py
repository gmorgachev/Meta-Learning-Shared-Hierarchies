import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Observer(nn.Module):
    def __init__(self, config):
        super(self.__class__, self).__init__()
        self.img_observer = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2), padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            Flatten()
        )

        self.hid = nn.Linear(576, config["hidden_dim"])
        self.rnn = nn.LSTMCell(config["hidden_dim"], config["hidden_dim"])

    def forward(self, prev_state, obs):
        x = self.img_observer(obs.permute(0, 3, 1, 2))
        x = self.hid(x)
        new_state = self.rnn(x, prev_state)

        return new_state


class SimpleRecurrent(nn.Module):
    def __init__(self, obs_shape, n_actions, config):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()
        self.observer = Observer(config)

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

    def forward(self, prev_state, obs):
        new_state = self.observer(prev_state, obs)
        logits = self.actor(new_state[0])
        state_value = self.critic(new_state[0])[:, 0]

        return new_state, (logits, state_value)

    def get_initial_state(self, batch_size):
        return torch.zeros((batch_size, self.observer.rnn.hidden_size)),\
               torch.zeros((batch_size, self.observer.rnn.hidden_size))

    def sample_actions(self, agent_outputs):
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(self, prev_state, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)
        (h, c), (l, s) = self.forward(prev_state, obs_img)
        return (h, c), (l, s)
