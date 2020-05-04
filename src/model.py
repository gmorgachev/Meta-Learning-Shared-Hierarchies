import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SimpleRecurrent(nn.Module):
    def __init__(self, obs_shape, n_actions, reuse=False):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()
        self.observer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(),
            Flatten()
        )

        self.hid = nn.Linear(128, 128)
        self.rnn = nn.LSTMCell(128, 128)

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, prev_state, obs):
        x = self.observer(obs.permute(0, 3, 1, 2))
        x = self.hid(x)
        new_state = self.rnn(x, prev_state)
        x = torch.cat([new_state[0], x], -1)
        logits = self.actor(x)
        state_value = self.critic(x)[:, 0]

        return new_state, (logits, state_value)

    def get_initial_state(self, batch_size):
        return torch.zeros((batch_size, 128)),\
               torch.zeros((batch_size, 128))

    def sample_actions(self, agent_outputs):
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(self, prev_state, obs_t):
        obs_img = torch.tensor(obs_t, dtype=torch.float32)
        (h, c), (l, s) = self.forward(prev_state, obs_img)
        return (h, c), (l, s)
