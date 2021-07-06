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
from Utility import *
sourcePath = "./"
seed = 0xC8763

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

network = PolicyGradientNetwork()
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
    if batch % 100 == 0:
        saveLandingVideo(f"Training.mp4", env=gym.make(
            'LunarLander-v2'), actions=GenerateAction(env, agent, NUM_OF_TEST=1,quite = True)[0])
        exit()
end = time.time()
print(f"Total training time is {end-start} sec")
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.savefig(sourcePath + "TotalRewards.png")
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.savefig(sourcePath + "FinalRewards.png")
actions_list = GenerateAction(env, agent)
for i, action in enumerate(actions_list):
    saveLandingVideo(f"{i+1}.mp4", env=gym.make(
        'LunarLander-v2'), actions=action)
TestAction(env = env, agent = agent, actions_list=actions_list)
