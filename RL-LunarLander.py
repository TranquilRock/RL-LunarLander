from pyvirtualdisplay import Display
import time
import random
import gym

from numpy.lib.utils import source
from tqdm import tqdm
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from SaveLandingVideo import saveLandingVideo
from SetSeed import *
sourcePath = "./"
seed = 0xC8763

"""
step() returns：
- observation / state
- reward
- done
- 其餘資訊
Reward
- 小艇墜毀得到 -100 分
- 小艇在黃旗幟之間成功著地則得 100~140 分
- 噴射主引擎（向下噴火）每次 -0.3 分
- 小艇最終完全靜止則再得 100 分
- 小艇每隻腳碰觸地面 +10 分
"""

# To allow env.render on Terminal where Display var is disabled
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

env = gym.make('LunarLander-v2')
fixEnvironment(env, seed)
fixTorch(seed)
fixNumpy(seed)
initial_state = env.reset()
print("Environment: ", env.observation_space)
print("Action Space: ", env.action_space)
print("Initial State: ", initial_state)
start = time.time()

# Input 8-dim obeseravtion, output an action out of four choices.


class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            # nn.Linear(32, 32),
            # nn.Tanh(),
            # nn.Linear(32, 32),
            # nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
        )
        self.last = nn.Softmax()

    def forward(self, state):
        return self.last(self.fc1(state), dim=-1)


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


network = PolicyGradientNetwork().cuda()
agent = PolicyGradientAgent(network)
agent.network.train()

EPISODE_PER_BATCH = 5
NUM_BATCH = 400

avg_total_rewards, avg_final_rewards = [], []
progress_bar = tqdm(range(NUM_BATCH))
for batch in progress_bar:
    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []
    for episode in range(EPISODE_PER_BATCH):
        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        reward = 0.0
        done = False
        while not done:
            action, log_prob = agent.sample(state)  # at , log(at|st)
            next_state, tReward, done, _ = env.step(action)
            reward = reward * 0.99 + tReward  # accumulative decaying reward, DQN?
            # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            log_probs.append(log_prob)
            seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward)
        final_rewards.append(reward)
        total_rewards.append(total_reward)
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    progress_bar.set_description(
        f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
    rewards = (rewards - np.mean(rewards)) / \
        (np.std(rewards) + 1e-9)  # Normalisze Reward
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
end = time.time()
print(f"Total training time is {end-start} sec")
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.savefig(sourcePath + "TotalRewards.png")
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.savefig(sourcePath + "FinalRewards.png")


def GenerateAction(env, agent, NUM_OF_TEST=5):
    agent.network.eval()
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        total_reward = 0
        done = False
        while not done:
            action, _ = agent.sample(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print(total_reward)
        test_total_reward.append(total_reward)
        action_list.append(actions)
        print("length of actions is ", len(actions))
    print(f"Your final reward is : %.2f" % np.mean(test_total_reward))
    print("Action list: ", action_list)
    print("Action list's shape: ", np.shape(action_list))
    distribution = {}
    for actions in action_list:
        for action in actions:
            if action not in distribution.keys():
                distribution[action] = 1
            else:
                distribution[action] += 1
    print("Action list's distribution: ", end="")
    print(distribution)
    PATH = "Action_List_test.npy"
    np.save(PATH, np.array(action_list))
    return action_list


actions_list = GenerateAction(env, agent)
for i, action in enumerate(actions_list):
    saveLandingVideo(f"{i+1}.mp4", env=gym.make(
        'LunarLander-v2'), actions=action)


def TestAction(env, actions_list):
    agent.network.eval()
    test_total_reward = []
    for actions in actions_list:
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
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


TestAction(env, actions_list=actions_list)
