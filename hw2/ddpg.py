# Spring 2023, 535515 Reinforcement Learning
# HW2: DDPG

import sys
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_3")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.ac=nn.Sequential(
            nn.Linear(num_inputs,hidden_size), ## 3 * 128
            nn.ReLU(),
            # nn.Linear(hidden_size,hidden_size), ## 128 * 128
            # nn.ReLU(),
            nn.Linear(hidden_size,num_outputs), ## 128 * 1
            nn.ReLU(),
            nn.Sigmoid(),
        )
        for idx,m in enumerate(self.modules()):
            # print(idx,m)
            # print("########################################")
            if idx==4:
                print(m)
                nn.init.uniform_(m.weight,a=-3e-3,b=3e-3)

        ##############################################################
    #     self.fc1 = nn.Linear(num_inputs, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)
    #     self.fc3 = nn.Linear(hidden_size, num_outputs)
    #     self.relu = nn.ReLU()
    #     self.tanh = nn.Tanh()
    #     self.init_weights(3e-3)

    # def init_weights(self, init_w):
    #         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
    #         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
    #         self.fc3.weight.data.uniform_(-init_w, init_w)
        ########## END OF YOUR CODE ##########
    def forward(self, inputs):

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network

        scale=torch.tensor(self.action_space.high-self.action_space.low)
        act_prob=self.ac(inputs)
        return act_prob*scale+torch.tensor(self.action_space.low)
        # return act_prob
        ##############################################################
        # out = self.fc1(inputs)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # out = self.tanh(out)
        # return out
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        # self.input_layer=nn.Sequential(
        #     nn.Linear(num_inputs,hidden_size),
        #     nn.ReLU(),
        # )
        self.critic_model=nn.Sequential(
            nn.Linear(num_inputs+num_outputs,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size,hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size,1),
        )
        for idx,m in enumerate(self.modules()):
            if idx==6:
                print(m)
                nn.init.uniform_(m.weight,a=-3e-3,b=3e-3)



        ##############################################################
    #     self.fc1 = nn.Linear(num_inputs, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size+num_outputs, hidden_size)
    #     self.fc3 = nn.Linear(hidden_size, 1)
    #     self.relu = nn.ReLU()
    #     self.init_weights(3e-3)
    # def init_weights(self, init_w):
    #         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
    #         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
    #         self.fc3.weight.data.uniform_(-init_w, init_w)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        # kk=self.input_layer(inputs)
        state_action=torch.cat((inputs,actions),dim=1)
        # print(state_action.shape)
        Q_value=self.critic_model(state_action)
        return Q_value ## return Q value for update actor

        ##############################################################
        # out = self.fc1(inputs)
        # out = self.relu(out)
        # # debug()
        # out = self.fc2(torch.cat([out,actions],1))
        # out = self.relu(out)
        # out = self.fc3(out)
        # return out
        ########## END OF YOUR CODE ##########


class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-6, lr_c=1e-5):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))
        mu = mu.data
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed

        if action_noise!=None:
            actrion_=mu+torch.tensor(action_noise)
        else :
            actrion_=mu
        action_prob=torch.clamp(actrion_,min=torch.tensor(self.action_space.low),max=torch.tensor(self.action_space.high))  ### 對 Probability 做 clip
        # action_prob=torch.clamp(actrion_,min=-2.0,max=2.0)  ### 對 Probability 做 clip
        return action_prob
        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))

        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic

        ### critic update

        reward_batch=reward_batch.unsqueeze(1)
        target_Q=reward_batch+self.gamma*self.critic_target(next_state_batch,self.actor_target(next_state_batch)).detach()

        Current_Q=self.critic(state_batch,action_batch)

        policy_loss=F.mse_loss(Current_Q.to(torch.float64),target_Q.to(torch.float64)) ## policy loss
        # policy_loss=F.huber_loss(y.to(torch.float64), q.to(torch.float64),delta=0.5) ## policy loss

        self.critic_optim.zero_grad()
        policy_loss.backward()
        self.critic_optim.step()

        ### actor update

        value_loss=-self.critic(state_batch,self.actor(state_batch)).mean()# valuee loss


        self.actor_optim.zero_grad()
        value_loss.backward()
        self.actor_optim.step()

        ########## END OF YOUR CODE ##########

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix)
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

def train():
    num_episodes = 200
    gamma = 0.99
    tau = 0.005
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 50000
    batch_size = 128
    updates_per_step =1
    print_freq = 10
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0


    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a=3e-4, lr_c=1e-3)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    for i_episode in range(num_episodes):

        ounoise.scale = noise_scale
        ounoise.reset()

        state = torch.Tensor([env.reset()])

        episode_reward = 0
        while True:
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            action = agent.select_action(state,action_noise=ounoise.noise()).numpy()[0]
            # action*=2
            next_state, reward, done, _ = env.step(action)

            # print(next_state)
            # print(reward)
            reward/=10
            episode_reward+=reward
            next_state=torch.tensor([next_state])
            action=torch.tensor([action],dtype=torch.float32)

            memory.push(state,action,torch.tensor([done]),next_state,torch.tensor([reward]))
            total_numsteps+=1

            if total_numsteps%updates_per_step==0 and total_numsteps>batch_size*updates_per_step:
                sample_tran=memory.sample(batch_size)
                batch=Transition([],[],[],[],[])
                for i in sample_tran:
                    batch.state.append(i.state)
                    batch.action.append(i.action)
                    batch.mask.append(i.mask)
                    batch.next_state.append(i.next_state)
                    batch.reward.append(i.reward)
                value_loss,policy_loss=agent.update_parameters(batch)
            state=next_state
            if done:
                break
            ########## END OF YOUR CODE ##########
        # print(total_numsteps)

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])

                env.render()

                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state

                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            writer.add_scalar("Loss/value_loss",value_loss,i_episode)
            writer.add_scalar("Loss/Policy_loss",policy_loss,i_episode)
            writer.add_scalar("Reward/ewma_reward",ewma_reward,i_episode)

    agent.save_model('Pendulum-v1', '.pth')


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    #env = gym.make('LunarLanderContinuous-v2')
    env = gym.make('Pendulum-v1')
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    train()


