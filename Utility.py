import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
# Input 8-dim obeseravtion, output an action out of four choices.


class PolicyGradientNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.last = nn.Softmax(dim = -1)

    def forward(self, state):
        x = self.fc1(state)
        # print(x.shape)
        # print(x)
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


def TestAction(env, agent, actions_list):
    agent.network.eval()
    test_total_reward = []
    for actions in actions_list:
        state = env.reset()
        total_reward = 0
        done = False
        done_count = 0
        for action in actions:
            state, reward, done, _ = env.step(action)
            done_count += 1
            total_reward += reward
            if done:
                break
        print(f"Your reward is : %.2f" % total_reward)
        test_total_reward.append(total_reward)
    print(f"Your final reward is : %.2f" % np.mean(test_total_reward))
    return total_reward


def GenerateAction(env, agent, NUM_OF_TEST=5, quite=False):
    agent.network.eval()
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.sample(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        test_total_reward.append(total_reward)
        action_list.append(actions)
    distribution = {}
    for actions in action_list:
        for action in actions:
            if action not in distribution.keys():
                distribution[action] = 1
            else:
                distribution[action] += 1
    if not quite:
        print(f"Final reward is : %.2f" % np.mean(test_total_reward))
        print("Action list's distribution: ", distribution)
        np.save("Action_List_test.npy", np.array(action_list))
    return action_list
