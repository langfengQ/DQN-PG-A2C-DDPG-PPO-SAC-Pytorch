import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gym
import numpy as np
import argparse
from itertools import count
import random

parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=9924)
parser.add_argument('--buffer-max-size', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--total-episode', type=int, default=1000)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--render', action='store_true')
parser.add_argument('--render_interval', type=int, default=20)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim):
        super(PolicyNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mu_head = nn.Linear(300, 1)
        self.log_std_head = nn.Linear(300, 1)

        self.action_scale = (max_action - min_action) / 2.0
        self.action_bias = (max_action + min_action) / 2.0

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def gaussian_sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON), dim=1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.l1_1 = nn.Linear(state_dim + action_dim, 400)
        self.l2_1 = nn.Linear(400, 300)
        self.l3_1 = nn.Linear(300, 1)

        self.l1_2 = nn.Linear(state_dim + action_dim, 400)
        self.l2_2 = nn.Linear(400, 300)
        self.l3_2 = nn.Linear(300, 1)

    def forward(self, s, a):
        x0 = torch.cat((s, a), dim=1)

        x1 = F.relu(self.l1_1(x0))
        x1 = F.relu(self.l2_1(x1))
        x1 = self.l3_1(x1)

        x2 = F.relu(self.l1_2(x0))
        x2 = F.relu(self.l2_2(x2))
        x2 = self.l3_2(x2)
        return x1, x2


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


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.policynetwork = PolicyNetwork(state_dim).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policynetwork.parameters(), lr=1e-4)

        self.qnetwork = QNetwork(state_dim, action_dim).to(device)
        self.target_qnetwork = QNetwork(state_dim, action_dim).to(device)
        self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())
        self.optimizer_qnetwork = torch.optim.Adam(self.qnetwork.parameters(), lr=1e-3)

        self.replay_buffer = Replay_Buffer()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.policynetwork.gaussian_sample(state)

        return action.cpu().data.numpy().flatten()

    def update(self, ):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.Tensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob = self.policynetwork.gaussian_sample(state)
            target_Q1, target_Q2 = self.target_qnetwork(next_state, next_action)
            target_Q = reward + done * args.gamma * (torch.min(target_Q1, target_Q2) - args.alpha * next_log_prob)

        Q1, Q2 = self.qnetwork(state, action)
        Q1_loss = F.mse_loss(Q1, target_Q)
        Q2_loss = F.mse_loss(Q2, target_Q)
        Q_loss = Q1_loss + Q2_loss

        self.optimizer_qnetwork.zero_grad()
        Q_loss.backward()
        self.optimizer_qnetwork.step()

        action_new, log_prob = self.policynetwork.gaussian_sample(state)
        Q1_new, Q2_new = self.qnetwork(state, action_new)
        policy_loss = (args.alpha * log_prob - torch.min(Q1_new, Q2_new)).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

        for param, target_param in zip(self.qnetwork.parameters(), self.target_qnetwork.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


if __name__ == '__main__':
    agent = SAC(state_dim, action_dim, max_action)
    total_step = 0

    for episode in range(args.total_episode):
        episode_reward = 0
        state = env.reset()
        # print(len(agent.replay_buffer.buffer))
        for t in count():
            action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            data = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            agent.replay_buffer.save(data)

            if len(agent.replay_buffer.buffer) >= args.batch_size: agent.update()

            if args.render and episode % args.render_interval == 0: env.render()

            state = next_state
            total_step += 1
            episode_reward += reward
            if done:
                break
        print('Total step: {}  Episode: {}  Episode reward: {:.2f}'.format(total_step, episode, episode_reward))



