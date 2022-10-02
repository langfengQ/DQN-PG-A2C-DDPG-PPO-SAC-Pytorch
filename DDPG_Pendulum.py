'''
DDPG Pendulum

Author: Lang Feng
Creation Date : Fri., Dec. 3, 2021, UTC+8
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np
import argparse
from itertools import count

parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=9924)
parser.add_argument('--buffer-max-size', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--total-episode', type=int, default=1000)
parser.add_argument('--exploration-noise', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--render', default=True ,action='store_false')
parser.add_argument('--render_interval', type=int, default=5)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class Replay_Buffer(object):
    def __init__(self):
        self.buffer = []
        self.max_size = args.buffer_max_size
        self.index = 0

    def save(self, data):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.index)] = data
            self.index = (self.index + 1) % self.max_size
        else:
            self.buffer.append(data)

    def sample(self):
        random_index = np.random.choice(len(self.buffer), args.batch_size, replace=False)
        state = [self.buffer[i]['state'] for i in random_index]
        action = [self.buffer[i]['action'] for i in random_index]
        reward = [self.buffer[i]['reward'] for i in random_index]
        next_state = [self.buffer[i]['next_state'] for i in random_index]
        done = [1 - int(self.buffer[i]['done']) for i in random_index]

        return state, action, reward, next_state, done


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = Replay_Buffer()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)

        return action.cpu().data.numpy().flatten()

    def update(self, ):
        state, action, reward, next_state, done = self.replay_buffer.sample()
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)

        done = torch.Tensor(done).unsqueeze(1).to(device)

        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + done * args.gamma * target_Q.detach()
        Q = self.critic(state, action)

        loss_critic = F.mse_loss(Q, target_Q)

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        loss_actor = - self.critic(state, self.actor(state)).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data = args.tau * param.data + (1 - args.tau) * target_param.data
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data = args.tau * param.data + (1 - args.tau) * target_param.data


if __name__ == '__main__':
    agent = DDPG(state_dim, action_dim, max_action)
    total_step = 0

    for episode in range(args.total_episode):
        episode_reward = 0
        state = env.reset()
        for t in count():
            action = agent.select_action(state)
            action = (action + np.random.normal(0, args.exploration_noise, size=(action_dim))).clip(min_action,
                                                                                                    max_action)

            next_state, reward, done, _ = env.step(action)

            data = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            agent.replay_buffer.save(data)

            if len(agent.replay_buffer.buffer) >= args.batch_size: 
                agent.update()

            if args.render and episode % args.render_interval == 0: env.render()

            state = next_state
            total_step += 1
            episode_reward += reward
            if done:
                break
        print('Total step: {}  Episode: {}  Episode reward: {:.2f}'.format(total_step, episode, episode_reward))