import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
from itertools import count
import math

from torch.distributions import Categorical

parser = argparse.ArgumentParser()

parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', type=int, default=9924)
parser.add_argument('--total-episode', type=int, default=2000)
parser.add_argument('--buffer-max-size', type=int, default=300)
parser.add_argument('--max-step', type=int, default=400)
parser.add_argument('--render', action='store_true')
parser.add_argument('--render-interval', type=int, default=20)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--time-step', type=int, default=50)
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class Replay_Buffer(object):
    def __init__(self):
        self.buffer = [[]]
        self.max_size = args.buffer_max_size
        self.index = 0

    def save(self, data):
        self.buffer[int(self.index)].append(data)

    def sample(self):
        random_index = np.random.choice(len(self.buffer), args.batch_size, replace=False)
        state, action, reward, next_state, done = [], [], [], [], []
        time_step = args.time_step
        for i in random_index: time_step = min(time_step, len(self.buffer[i]))

        for i in random_index:
            random_sub_index = np.random.randint(0, len(self.buffer[i]) - time_step + 1)
            state_, action_, reward_, next_state_, done_ = [], [], [], [], []
            for j in range(random_sub_index, time_step + random_sub_index):
                state_.append(self.buffer[i][j]['state'])
                action_.append(self.buffer[i][j]['action'])
                reward_.append(self.buffer[i][j]['reward'])
                next_state_.append(self.buffer[i][j]['next_state'])
                done_.append(1- int(self.buffer[i][j]['done']))
            state.append(state_)
            action.append(action_)
            reward.append(reward_)
            next_state.append(next_state_)
            done.append(done_)

        return state, action, reward, next_state, done
    
    def new_epsilon(self):
        self.index = (self.index + 1) % self.max_size

        if len(self.buffer) < self.max_size:
            self.buffer.append([])
        else:
            del self.buffer[int(self.index)][:]


class DRQN(nn.Module):
    def __init__(self):
        super(DRQN, self).__init__()

        self.l1 = nn.Linear(state_dim, self.channel)
        self.lstm = nn.LSTM(self.channel, self.channel, batch_first=True)
        self.l2 = nn.Linear(self.channel, action_dim)

    @property
    def channel(self):
        return 128

    def reset_lstm(self, batch_size):
        self.hc = (torch.zeros((1, batch_size, self.channel), dtype=torch.float32).to(device), 
        torch.zeros((1, batch_size, self.channel), dtype=torch.float32).to(device))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x, self.hc = self.lstm(x, self.hc)
        x = self.l2(x)
        return x
        

class Agent():
    def __init__(self):
        self.replay_buffer = Replay_Buffer()

        self.drqn = DRQN().to(device)
        self.target_drqn = DRQN().to(device)
        self.target_drqn.load_state_dict(self.drqn.state_dict())

        self.optim = torch.optim.Adam(self.drqn.parameters(), lr=5e-4)
        self.loss_func = nn.MSELoss()


    def select_action(self, state, epsilon):
        state = torch.FloatTensor(state).view(1, 1, -1).to(device)
        q = self.drqn(state)

        if np.random.random() > epsilon:
            action = q.max(dim=2)[1].item()
        else:
            action = np.random.randint(0, action_dim)

        return action

    def update(self):
        for _ in range(10):
            state, action, reward, next_state, done = self.replay_buffer.sample()

            state = torch.FloatTensor(state).to(device)
            action = torch.LongTensor(action).unsqueeze(2).to(device)
            reward = torch.FloatTensor(reward).unsqueeze(2).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).unsqueeze(2).to(device)

            with torch.no_grad():
                self.target_drqn.reset_lstm(batch_size=args.batch_size)
                next_q = self.target_drqn(next_state)
                target_q = reward + done * args.gamma * torch.max(next_q, dim=2)[0].unsqueeze(2)

            self.drqn.reset_lstm(batch_size=args.batch_size)
            q_temp = self.drqn(state)
            q = q_temp.gather(2, action)

            loss = self.loss_func(q, target_q)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
    def copy_drqn(self):
        self.target_drqn.load_state_dict(self.drqn.state_dict())

    @classmethod
    def get_decay(cls, episode):
        decay = max(0.05, math.pow(0.99, episode))
        return decay

if __name__ == '__main__':
    agent = Agent()
    total_step = 0
    
    for episode in range(args.total_episode):
        episode_reward = 0
        state = env.reset()

        agent.drqn.reset_lstm(batch_size=1)
        for t in count():
            action = agent.select_action(state, Agent.get_decay(episode))
            
            next_state, reward, done, _ = env.step(action)

            data = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            agent.replay_buffer.save(data)

            if args.render and episode % args.render_interval == 0 : env.render()

            state = next_state
            total_step += 1
            episode_reward += reward
            if done or t > args.max_step:
                break

        if len(agent.replay_buffer.buffer) - 5 >= args.batch_size:
            if episode % args.update_interval == 0:
                agent.copy_drqn()
            agent.update()

        agent.replay_buffer.new_epsilon()
        print('Total step: {}  Episode: {}  Episode reward: {:.2f}'.format(total_step, episode, episode_reward))
