import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
from itertools import count
from collections import namedtuple
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler

from torch.distributions import Categorical


parser = argparse.ArgumentParser(description="reinfore_example")
parser.add_argument('--gamma', type=float, default=0.99, help='discound factor')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--lr_a', type=float, default=1e-3, help='learning rate of actor network')
parser.add_argument('--lr_c', type=float, default=1e-3, help='learning rate of critic network')
parser.add_argument('--batch-size', type=int, default=36, help='batch size')
parser.add_argument('--log-interval', type=int, default=20, help='log internal')
parser.add_argument('--use-value-gradient', action='store_true', help='whether use value gradient when updating policy network')
args = parser.parse_args()

env = gym.make('CartPole-v1')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
env.seed(args.seed)
torch.manual_seed(args.seed)
Trans = namedtuple('trans', ['state', 'action', 'a_prob', 'reward', 'next_state'])



class AC_Net(nn.Module):
    def __init__(self, is_actor):
        super(AC_Net, self).__init__()
        self.affine1 = nn.Linear(num_state, 256)
        # self.dropout = nn.Dropout(p=0)
        self.action_head = nn.Linear(256, num_action)
        self.value_head = nn.Linear(256, 1)

        self.is_actor = is_actor

    def forward(self, x):
        x = self.affine1(x)

        # x = self.dropout(x)
        x = nn.ReLU(inplace=True)(x)
        # x = F.relu(x)
        if self.is_actor:
            action = self.action_head(x)
            return F.softmax(action, dim=1)
        else:
            value = self.value_head(x)
            return value


class PPO(object):
    clip_param = 0.2
    ppo_updata_time = 10

    def __init__(self):
        self.actor = AC_Net(is_actor=True)
        self.critic = AC_Net(is_actor=False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.buffer = list()
        # self.counter = 0
    
    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)

        m = Categorical(probs)
        action = m.sample()

        return action.item(), probs[:, action.item()].item()


    def save_transition(self, trans):
        self.buffer.append(trans)
        # self.counter += 1

    def learner(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer] 

        old_a_probs = torch.tensor([t.a_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        Gt = []
        R = 0
        for r in reward:
            R = r + args.gamma * R
            Gt.insert(0, R)

        Gt = torch.tensor(Gt, dtype=torch.float).view(-1, 1)
        for i in range(self.ppo_updata_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), args.batch_size, False):
                
                new_a_probs = self.actor(state[index, :]).gather(1, action[index, :])

                v = self.critic(state[index, :])
                advantage = Gt[index, :] - v.detach()

                ratio = new_a_probs / old_a_probs[index, :]

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
                self.actor_optimizer.step()

                value_loss = F.mse_loss(v, Gt[index, :])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
                self.critic_optimizer.step()

        del self.buffer[:]


if __name__=='__main__':
    ppo = PPO()
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action, a_probs = ppo.select_action(state)

            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            trans = Trans(state, action, a_probs, reward, state_)
            ppo.save_transition(trans)

            if args.render:
                env.render()
            state = state_
            if done:
                if len(ppo.buffer) > args.batch_size:
                    ppo.learner()
                break

        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        # finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode: {}, Last reward: {:.2f}, Average reward: {:.2f}'.format(i_episode, ep_reward, running_reward))
        
        if running_reward > env.spec.reward_threshold:
            print("Solved Running reward is now {} and " 
            "the last episode runs to {} time step!".format(running_reward, t))
            break
        
