# This code was a modification of pytorch tutorial
# Link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from numpy.lib.function_base import average
from util.SetSeed import *
from util.ReplayMemory import *
from util.SaveLandingVideo import saveLandingVideo
from agent.QAgent import *
# ===============================================
import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
import sys


def main(argv):
    noDisplay = "-q" in argv
    if noDisplay:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()
    else:
        from IPython import display
        plt.ion()

    env = gym.make('LunarLander-v2')
    sourcePath = "./"
    seed = 0xC8763
    fixEnvironment(env, seed)
    fixTorch(seed)
    fixNumpy(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TARGET_UPDATE = 10  # Frequency of updating TARGET NETWORK

    # Get number of actions from gym action space
    state = env.reset()
    nActions = env.action_space.n
    policyQAgent = QAgent(len(state), nActions, optim.RMSprop, {
                          "lr": 1e-5}, device=device)
    targetNet = DQN(len(state), nActions).to(device)
    targetNet.load_state_dict(policyQAgent.network.state_dict())
    targetNet.eval()
    if "-l" in argv:
        policyQAgent.load(sourcePath + "Qmodel.ckpt")
    memory = ReplayMemory(100000)
    # =====================================Training ============================
    num_episodes = 100000
    maxReward = 0.0
    progressBar = tqdm(range(num_episodes))
    for i in progressBar:
        currentState = env.reset()
        currentState = torch.tensor(currentState).view(1, 8)
        totalReward = 0.0
        finalReward = 0.0
        loss = []
        for t in count():
            action = policyQAgent.select_action(currentState)
            nextState, reward, done, _ = env.step(action.item())
            if not noDisplay:
                plt.imshow(env.render(mode='rgb_array'),
                           interpolation='none')
            nextState = torch.tensor(nextState).view(
                1, 8) if not done else None
            reward = torch.tensor([reward], device=device)
            memory.push(currentState, action, nextState, reward)
            currentState = nextState

            transitions = memory.sample(
                min(policyQAgent.BATCH_SIZE, len(memory)))
            batch = Transition(*zip(*transitions))
            loss.append(policyQAgent.learn(batch, targetNet))

            totalReward = totalReward*policyQAgent.GAMMA + (reward.item())
            if done:
                finalReward = reward.item()
                break
        policyQAgent.save(sourcePath + "Qmodel.ckpt")
        progressBar.set_description(
            f"Total: {totalReward: 4.1f}, Final: {finalReward: 4.1f},Loss: {average(loss)}")
        
        if totalReward > maxReward:
            maxReward = totalReward
            saveLandingVideo(sourcePath + "Train.mp4", env , policyQAgent)
        elif i % TARGET_UPDATE == 0:
            tau = 5e-3
            for eval_param , param in zip(targetNet.parameters(), policyQAgent.network.parameters()):
                eval_param.data.copy_(
                    param.data * tau + (1 - tau) * eval_param.data)
            # targetNet.load_state_dict(policyQAgent.network.state_dict())
    if noDisplay:
        pass
    else:
        env.render()
        env.close()
        plt.ioff()



