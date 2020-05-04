import torch
from torch import nn
from torch.nn import functional as F


class Observation(nn.Module):
    def __init__(self, h, w, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class BasePolicy(nn.Module):
    def __init__(self, obs):
        super().__init__()
        self.obs_network = obs

    def init_weights(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            nn.init.uniform_(param, -0.1, 0.1)


class MasterPolicy(BasePolicy):
    def __init__(self, obs):
        super(MasterPolicy, self).__init__(obs)

    def act(self):
        pass

    def reset(self):
        pass


class Policy(BasePolicy):
    def __init__(self, obs):
        super(Policy, self).__init__(obs)

    def act(self):
        pass