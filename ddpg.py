import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import copy

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim = 256):
        super(Actor, self).__init__()

        self.model =nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), #generally we add 2-3 hidden layers
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), #generally we add 2-3 hidden layers
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(), #output between -1 and 1
        )

        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.model(state) #scale the output to action space
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), #generally we add 2-3 hidden layers
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
            return self.model(torch.cat([state, action], 1))
        
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action,reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
class OUNoise:
    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
        
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action,device):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        for param in self.actor_target.parameters():
            param.requires_grad = False

        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 0.001)

        self.noise = OUNoise(action_dim)

        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, add_noise = True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device) #converts numpy array to tensor and send to device
            action = self.actor(state).cpu().numpy() #gets the action and converts to numpy
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -self.actor.max_action, self.actor.max_action) #ensures action stays within valid range
    
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).reshape(-1, 1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()
    
def train_ddpg(env, agent, num_episodes, batch_size, replay_buffer_size):
    replay_buffer = ReplayBuffer(replay_buffer_size)
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards

