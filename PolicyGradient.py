from pyvirtualdisplay import Display
from IPython import display
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
import time
import random
import sys
# ===================
from util.SaveLandingVideo import saveLandingVideo
from util.SetSeed import *
from util.Utility import *
from agent.PolicyAgent import *
def main(argv):
    env = gym.make('LunarLander-v2')
    sourcePath = "./"
    seed = 0xC8763
    fixEnvironment(env, seed)
    fixTorch(seed)
    fixNumpy(seed)
    # To allow env.render on Terminal where Display var is disabled

    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()


    initial_state = env.reset()
    if "-i" in argv:
        print("Environment: ", env.observation_space)
        print("Action Space: ", env.action_space)
        print("Initial State: ", initial_state)
    start = time.time()

    network = PolicyGradientNetwork()
    agent = PolicyGradientAgent(network, lr=1e-6)
    agent.network.train()
    if "-l" in argv:
        agent.load(sourcePath + "ll.ckpt")

    EPISODE_PER_BATCH = 20
    NUM_BATCH = 10000
    Gamma = 0.95
    avg_total_rewards, avg_final_rewards = [], []
    progress_bar = tqdm(range(1, NUM_BATCH+1))
    for batch in progress_bar:
        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []
        for episode in range(EPISODE_PER_BATCH):
            state = env.reset()
            total_reward = 0
            seq_rewards = []
            reward = 0.0
            done = False
            while not done:
                action, log_prob = agent.sample(state)  # at , log(at|st)
                state, reward, done, _ = env.step(action)
                # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                log_probs.append(log_prob)
                seq_rewards.append(reward)
                total_reward += reward
            for i in range(len(seq_rewards) - 2, -1, -1):
                seq_rewards[i] += Gamma * seq_rewards[i + 1]
            rewards.extend(seq_rewards)
            final_rewards.append(reward)  # used to plot
            total_rewards.append(total_reward)  # used to plot
        agent.save(sourcePath + "ll.ckpt")
        avg_total_reward = sum(total_rewards) / len(total_rewards)  # used to plot
        avg_final_reward = sum(final_rewards) / len(final_rewards)  # used to plot
        avg_total_rewards.append(avg_total_reward)  # used to plot
        avg_final_rewards.append(avg_final_reward)  # used to plot
        progress_bar.set_description(
            f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
        # /(np.std(rewards) + 1e-9)  # Normalisze Reward
        rewards = (rewards - np.mean(rewards))
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        if batch % 1000 == 0:
            saveLandingVideo(f"Training.mp4", env=gym.make(
                'LunarLander-v2'), Agent=agent)
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
        saveLandingVideo(sourcePath + f"{i+1}.mp4", env=gym.make(
            'LunarLander-v2'), Agent=agent)
    TestAction(env=env, agent=agent, actions_list=actions_list)
