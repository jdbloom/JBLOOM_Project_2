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
import matplotlib
from matplotlib import pyplot as plt
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n
        self.memory = deque(maxlen = 1000000)
        self.thirty_ep_reward = deque(maxlen = 100000)

        #print(self.state_size, self.action_size)
        # Discount Factor
        self.gamma = 0.99
        # Exploration Rate: at the beginning do 100% exploration
        self.epsilon = 1.0
        # Decay epsilon so we can shift from exploration to exploitation
        self.epsilon_decay = 0.995
        # Set floor for how low epsilon can go
        self.epsilon_min = 0.01
        # Set the learning rate
        self.learning_rate = 0.00015
        # batch_size
        self.batch_size = 32

        self.epsilon_decay_frames = 1.0/500000

        self.qnetwork = DQN(self.state_size[0], self.state_size[1], self.action_size).to(self.device)
        print('initial weights:')
        print(self.qnetwork.head.weight)
        self.q_prime = DQN(self.state_size[0], self.state_size[1], self.action_size).to(self.device)
        self.q_prime.load_state_dict(self.qnetwork.state_dict())

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr = self.learning_rate)

        self.loss = 0

        self.file_path = 'trained_models_2/./Q_Network_Parameters_'

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            file_number_to_load = 1933
            load_file_path = self.file_path+str(file_number_to_load)+'.pth'
            self.qnetwork.load_state_dict(torch.load(load_file_path, map_location = lambda storage, loc: storage))

            #for name, param in self.qnetwork.named_parameters():
            #    print(name, '\t\t', param.shape)
            print('loaded weights')
            print(self.qnetwork.head.weight)
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.curr_state = self.env.reset()
        ###########################
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        observation = observation[np.newaxis,:]
        observation = torch.tensor(observation, dtype = torch.float32).to(self.device)
        observation = observation.permute(0 , 3, 1, 2)
        if not test:
            if np.random.rand()<=self.epsilon:
                action = random.randrange(self.action_size)
            else:
                action = torch.argmax(self.qnetwork(observation)).item()
        else:
            action = torch.argmax(self.qnetwork(observation)).item()
        ###########################
        return action

    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        action = np.array(action, dtype = np.uint8)
        reward = np.array(reward, dtype = np.float32)
        done = np.array(done, dtype = np.float32)
        self.memory.append((state, action, reward, next_state, done))
        ###########################


    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        minibatch = random.sample(self.memory, self.batch_size)
        ###########################
        return minibatch
    def learn(self):
        minibatch = self.replay_buffer(self.batch_size)

        states, actions, rewards, next_states, dones = list(zip(*minibatch))

        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.from_numpy(np.stack(actions)).to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.from_numpy(np.stack(dones)).to(self.device)

        states = states.permute(0 , 3, 1, 2).float()
        next_states = next_states.permute(0, 3, 1, 2).float()
        actions = actions.unsqueeze(1)
        qfun = self.qnetwork(states)

        #print('input...\n',states[1][1].shape)
        #fig = plt.figure()
        #plt.imshow(states[0,0,:,:].cpu())
        #plt.title('State')
        #plt.savefig('state.png')
        #plt.close()

        state_action_values = qfun.gather(1, actions.long()).squeeze()

        next_state_values = self.q_prime(next_states).max(1).values.detach()

        TD_error = rewards + self.gamma*next_state_values*(1-dones)

        self.loss = F.smooth_l1_loss(state_action_values, TD_error)

        self.optimizer.zero_grad()
        self.loss.backward()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(0, self.epsilon - self.epsilon_decay_frames)


        for param in self.qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #print(torch.sum(self.qnetwork.conv1.weight.data))

    def train(self, n_episodes = 100000):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Initializing counters and lists for average reward over 30 episodes:
        ep_counter = 0.0
        time_steps = 0.0
        thirty_reward = 0.0
        ep_epsilon = []
        thirty_ep_reward = []
        thirty_ep_ep = []

        naming_counter = 0
        log = open('trained_models_2/log.txt', 'w+')
        log.write('Beginning of Log\n')
        log.close()

        frames = 0.0
        for e in range(n_episodes):

            running_loss = 0.0
            ep_counter += 1
            state = self.env.reset()
            done = False
            render = os.path.isfile('.makePicture')

            # Counters for Reward Averages per episode:
            ep_reward = 0.0
            counter = 0.0



            while not done:
                frames += 1
                counter += 1
                time_steps += 1

                if render: self.env.env.render()
                action = self.make_action(state, False)
                next_state, reward, done, _ = self.env.step(action)
                reward = np.clip(reward, -1, 1)

                self.push(state, action, reward, next_state, done)

                state = next_state
                #if done:
                #    reward = -1

                if frames > 500000:
                    if len(self.memory) > self.batch_size:
                        self.learn()
                        if frames%5000 == 0:
                            print('------------ UPDATING TARGET -------------')
                            self.q_prime.load_state_dict(self.qnetwork.state_dict())

                running_loss+= self.loss
                ep_reward+=reward
                thirty_reward += reward

            ep_epsilon.append(self.epsilon)
            # Print average reward for the episode:
            print('Episode ', e, 'had a reward of: ', ep_reward)
            print('Epsilon: ', self.epsilon)

            # Loging the average reward over 30 episodes
            if ep_counter%30 == 0:
                print('Frame: ', frames)
                thirty_ep_reward.append(thirty_reward/30)
                thirty_ep_ep.append(e)
                print('The Avereage Reward over 30 Episodes: ', thirty_reward/30.0)
                with open('trained_models_2/log.txt', 'a+') as log:
                    log.write(str(naming_counter)+' had a reward of '+ str(thirty_reward/30.0)+' over 30 ep\n')

                time_steps = 0.0
                thirty_reward = 0.0
                # Save network weights after we have started to learn
                if e > 3000:

                    print('saving... ', naming_counter)
                    save_file_path = self.file_path+str(naming_counter)+'.pth'
                    torch.save(self.qnetwork.state_dict(), save_file_path)
                    naming_counter += 1


                fig = plt.figure()
                plt.plot(ep_epsilon)
                plt.title('Epsilon decay')
                plt.xlabel('Episodes')
                plt.ylabel('Epsilon Value')
                plt.savefig('trained_models_2/epsilon.png')
                plt.close()

                fig = plt.figure()
                plt.plot(thirty_ep_ep, thirty_ep_reward)
                plt.title('Average Reward per 30 Episodes')
                plt.xlabel('Episodes')
                plt.ylabel('Average Reward')
                plt.savefig('trained_models_2/reward.png')
                plt.close()



            #################################
