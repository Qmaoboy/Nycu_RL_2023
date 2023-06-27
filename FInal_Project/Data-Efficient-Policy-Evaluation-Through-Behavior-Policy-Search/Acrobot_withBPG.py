# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
import PIL
import gym
from itertools import count
from collections import namedtuple
import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./Acrobot_event")

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()

        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()

        ########## YOUR CODE HERE (5~10 lines) ##########
        self.shared=nn.Sequential(torch.nn.Linear(self.observation_dim, self.hidden_size),
                                  nn.LeakyReLU(),
                                  nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.LeakyReLU())
        self.actor=torch.nn.Sequential( nn.Linear(self.hidden_size, self.action_dim))
        self.critic=torch.nn.Sequential(nn.Linear(self.hidden_size, 1))

        ########## END OF YOUR CODE ##########

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        x=self.shared(state)
        action_prob=self.actor(x)
        action_prob=nn.functional.log_softmax(action_prob-action_prob.max(-1,keepdims=True).values, dim=-1)

        ########## END OF YOUR CODE ##########

        return action_prob


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """

        ########## YOUR CODE HERE (3~5 lines) ##########
        state=torch.tensor(state)
        action_prob= self.forward(state)
        m=Categorical(action_prob.exp())
        action=m.sample()
        ########## END OF YOUR CODE ##########

        # save to action buffer

        return action.item()


def IS(H:list, model_b: nn.Module, model :nn.Module, up:bool):
    state=torch.tensor(H[0])
    act=H[1]
    reward=torch.tensor(H[2])
    g=[]
    last=0
    gamma=0.95
    for r in reward.flip(0):
        last=r+gamma*last
        g.append(last)
    g=torch.tensor(g).flip(0)
    g=g-g.mean()

    pi=model(state)[torch.arange(len(act)),act]#(len)
    pi_b=model_b(state)[torch.arange(len(act)),act]
    if up:
        return g*(pi-pi_b.detach()).exp()
    else:
        return g*(pi.detach()-pi_b).exp()



def get_behavior(B:list, model_b: nn.Module, model :nn.Module, optim:torch.optim.Adam):
    n=64
    for i in range(n):
        loss=0
        for H in B:
            ratio=IS(H, model_b, model, up=False)
            loss+= torch.mean(ratio**2)

        optim.zero_grad()
        # print(loss.item())
        loss.backward()
        optim.step()



def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode,
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode,
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """

    # Instantiate the policy model and the optimizer
    model_b = Policy()
    model = Policy()
    model.load_state_dict(model_b.state_dict())

    optimizer = optim.Adam(list(model.parameters())+list(model_b.parameters()), lr=lr)

    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # EWMA reward for tracking the learning progress
    ewma_reward = 500
    ma_t=0
    D=[]#H
    B=[]
    k=8
    n=8
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward =0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()

        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process

        ########## YOUR CODE HERE (10-15 lines) ##########
        H=[]
        optimizer.zero_grad()
        for t in range(9999):
            action=model_b.select_action(state)

            state_, reward, done, _ = env.step(action)
            ep_reward+=reward
            H.append([state, action, reward, state_])
            state=state_
            if done:
                break

        H=list(zip(*H))#[[s,...], [a,...],...]
        B.append(H)



        if i_episode%k==0:
            get_behavior(B, model_b, model, optimizer)
            D.extend(B)
            B=[]


        #off policy
        if i_episode%(k*n)==0:
            for _ in range(k*n*4):
                loss=0
                for H in D:
                    ratio=IS(H, model_b, model, up=True)
                    loss+= -ratio.mean()
                optimizer.zero_grad()
                loss.backward()
                # print(loss.item()/len(D))
                optimizer.step()
            model_b.load_state_dict(model.state_dict())
            D=[]

        ma_t=ma_t*0.95+t*0.05

        ########## END OF YOUR CODE ##########

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('reward', ewma_reward, i_episode)
        writer.add_scalar('length', ma_t, i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/Acrobot_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """
    model = Policy()

    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    render = True
    max_episode_len = 10000
    frames = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        frames.append(env.render(mode='rgb_array'))
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 frames.append(env.render(mode='rgb_array'))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    imageio.mimsave('gif/Acrobot.gif', frames, fps=30)


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    lr = 0.00003
    env = gym.make('Acrobot-v1')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train(lr)
    test(f'Acrobot_{lr}.pth')
