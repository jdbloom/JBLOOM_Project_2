# Testing
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

import gym

## Telling the code to use the Nvidia GPU
os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

env = gym.make('BreakoutNoFrameskip-v4')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(state_size, action_size)
