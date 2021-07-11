# This code was a modification of pytorch tutorial
# Link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from util.SetSeed import *
from util.ReplayMemory import *
from util.SaveLandingVideo import saveLandingVideo
from agent.QAgent import *
#===============================================
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
    policyQAgent = QAgent(len(state), nActions, optim.RMSprop,{"lr":1e-4},device=device)
    targetNet = DQN(len(state), nActions).to(device)
    targetNet.load_state_dict(policyQAgent.network.state_dict())
    targetNet.eval()
    if "-l" in argv:
        policyQAgent.load(sourcePath + "Qmodel.ckpt")
    memory = ReplayMemory(300000)
    #=====================================Training ============================
    num_episodes = 100000
    progressBar = tqdm(range(num_episodes))
    for i in progressBar:
        currentState = env.reset()
        currentState = torch.tensor(currentState).view(1, 8)
        totalReward = []
        finalReward = 0.0
        for t in count():
            action = policyQAgent.select_action(currentState)
            nextState, reward, done, _ = env.step(action.item())
            if noDisplay:
                pass
            else:
                plt.imshow(env.render(mode='rgb_array'),
                        interpolation='none')
            nextState = torch.tensor(nextState).view(1, 8)
            reward = torch.tensor([reward], device=device)
            if done:
                nextState = None
            memory.push(currentState, action, nextState, reward)
            currentState = nextState
            
            if len(memory) >= policyQAgent.BATCH_SIZE:
                transitions = memory.sample(policyQAgent.BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                print(batch.shape)
                exit()
                policyQAgent.learn(batch,targetNet)
            
            totalReward.append(reward.item())
            if done:
                finalReward = reward.item()
                break
        if i % TARGET_UPDATE == 0:
            targetNet.load_state_dict(policyQAgent.network.state_dict())
        policyQAgent.save(sourcePath + "Qmodel.ckpt")
        progressBar.set_description(
            f"Total: {sum(totalReward): 4.1f}, Final: {finalReward: 4.1f}")
    if noDisplay:
        pass
    else:
        env.render()
        env.close()
        plt.ioff()
