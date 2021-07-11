import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class DQN(nn.Module):
    def __init__(self, nState, nAction, device="cuda"):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(nState, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 128),
            nn.Sigmoid(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, nAction),
        )
        self.last = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, x):  # Return expected value of each action
        x = torch.FloatTensor(x).to(self.device)
        x = self.fc1(x)
        x = self.last(x)
        return x


class QAgent():
    def __init__(self, nState=8, nAction=4, optimizer=optim.SGD, optimizerConfig={"lr": 1e-3}, device="cuda"):
        self.nAction = nAction
        self.nState = nState
        self.network = DQN(nState, nAction, device).to(device)
        self.optimizer = optimizer(
            self.network.parameters(), **optimizerConfig)
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.steps_done = 0
        self.GAMMA = 0.9
        self.BATCH_SIZE = 512

    def forward(self, state):
        return self.network(state)

    def learn(self, batch, targetNet):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)  # torch.Size([Batchsize, 8])
        action_batch = torch.cat(batch.action)  # torch.Size([Batchsize, 8])
        reward_batch = torch.cat(batch.reward)  # torch.Size([Batchsize])

        # Output of DQN is reward of each action, not probability
        state_action_values = self.network(
            state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = targetNet(
            non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = ((
            next_state_values * self.GAMMA) + reward_batch).float()
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-10, 10)
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(
            torch.FloatTensor(state).to("cuda"))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.nAction)]], device=self.device, dtype=torch.long)

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
