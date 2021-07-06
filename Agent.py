import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Input 8-dim obeseravtion, output an action out of four choices.


class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.last = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.fc1(state)
        return self.last(x)


class PolicyGradientAgent():
    def __init__(self, network, optimizer=optim.SGD, lr=1e-4):
        self.network = network
        self.optimizer = optimizer(self.network.parameters(), lr=lr)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def save(self, PATH):
        Agent_Dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
