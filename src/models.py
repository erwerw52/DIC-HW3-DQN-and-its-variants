import torch
import torch.nn as nn
import copy


class DQN(nn.Module):
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DoubleDQN(nn.Module):
    """Same architecture as DQN; Double DQN logic lives in the trainer."""
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    """
    Splits Q(s,a) into V(s) + A(s,a).
    Q = V + (A - mean(A))  to keep identifiability.
    """
    def __init__(self, input_dim=64, hidden1=150, hidden2=100, output_dim=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden2, 1)
        self.advantage_stream = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        feat = self.shared(x)
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


def make_target_network(online_model):
    target = copy.deepcopy(online_model)
    target.load_state_dict(online_model.state_dict())
    return target
