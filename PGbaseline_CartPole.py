import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import argparse
from itertools import count

from torch.distributions import Categorical


parser = argparse.ArgumentParser(description="reinfore_example")
parser.add_argument('--gamma', type=float, default=0.99, help='discound factor')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--seed', type=int, default=543, help='seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--log-interval', type=int, default=10, help='log internal')
parser.add_argument('--use-value-gradient', action='store_true', help='whether use value gradient when updating policy network')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_log_probs = list()
        self.saved_rewards = list()

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        # x = nn.ReLU(inplace=True)
        x = F.relu(x)
        action = self.action_head(x)
        value = self.value_head(x)

        return F.softmax(action, dim=1), value

policy_net = Policy()
eps = np.finfo(np.float32).eps.item()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    probs, state_value = policy_net.forward(state)

    m = Categorical(probs)

    action = m.sample()
    policy_net.saved_log_probs.append((m.log_prob(action), state_value))

    return action.item()



def learner():
    returns = list()
    policy_loss = list()
    value_loss = list()
    G = 0


    for r in policy_net.saved_rewards[::-1]:
        G = r + args.gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, state_value), G in zip(policy_net.saved_log_probs, returns):
        advantage = G - state_value
        if args.use_value_gradient:
            pass
        else:
            advantage = advantage.detach()
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(G, state_value))
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss / 2

    loss.backward()
    optimizer.step()

    del policy_net.saved_rewards[:]
    del policy_net.saved_log_probs[:]


if __name__=='__main__':
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            action = select_action(state)

            state_, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy_net.saved_rewards.append(reward)
            ep_reward += reward
            state = state_
            if done:
                break

        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        learner()   
        # finish_episode() 
        if i_episode % args.log_interval == 0:
            print('Episode: {}, Last reward: {:.2f}, Average reward: {:.2f}'.format(i_episode, ep_reward, running_reward))
        
        if running_reward > env.spec.reward_threshold:
            print("Solved Running reward is now {} and " 
            "the last episode runs to {} time step!".format(running_reward, t))
            break
        
