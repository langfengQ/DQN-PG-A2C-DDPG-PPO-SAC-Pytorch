import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartPole-v0')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 0.0002
GAMMA = 0.98
SEED = 6666
env.seed(SEED)
torch.manual_seed(SEED)


# # >>>>>>>>>>>>>> Actor Network Establishment >>>>>>>>>>>>>>>>>>>> # #


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight.data, 0, 0.1)
        #         nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.softmax(out, dim=1)


class Actor(object):
    def __init__(self):
        self.actor_network = ActorNetwork(STATE_DIM, ACTION_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.actor_network.parameters(), LR)
        self.saved_log_probs = None

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        prob_weights = self.actor_network(state)

        m = torch.distributions.Categorical(prob_weights)
        action = m.sample()

        self.saved_log_probs = m.log_prob(action)

        return action.item()

    def learn(self, td_error):
        loss = -(self.saved_log_probs * td_error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# # >>>>>>>>>>>>>> Critic Network Establishment >>>>>>>>>>>>>>>>>>>> # #


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight.data, 0, 0.1)
        #         nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Critic(object):
    def __init__(self):
        self.critic_network = CriticNetwork(STATE_DIM).to(device)
        self.optimizer = torch.optim.Adam(self.critic_network.parameters(), LR)

        self.v = None
        self.v_ = None

    def td_error(self, state, state_, reward):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state_ = torch.FloatTensor(state_).unsqueeze(0).to(device)

        self.v = self.critic_network(state)
        self.v_ = self.critic_network(state_)

        with torch.no_grad():
            td_error = reward + GAMMA * self.v_ - self.v
        return td_error

    def learn(self, reward):
        loss = nn.MSELoss()(reward + GAMMA * self.v_, self.v)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# # >>>>>>>>>>>>>> Training >>>>>>>>>>>>>>>>>>>> # #

def main():
    actor  = Actor()
    critic = Critic()
    running_reward = 0
    for episode in range(10000):
        state = env.reset()
        total_reward = 0
        for step in range(2000):
            env.render()
            action = actor.choose_action(state)

            state_, reward, done, _ = env.step(action)
            total_reward += reward
            reward = reward / 100.
            # obtain td_error
            td_error = critic.td_error(state, state_, reward)
            # train actor network
            actor.learn(td_error)
            # train critic network
            critic.learn(reward)

            state = state_
            if done:
                break
        running_reward = running_reward * 0.9 + total_reward * 0.1

        if episode % 40 == 0:
            print('episode: ', episode, '; Average Reward:', running_reward)


if __name__ == '__main__':
    main()