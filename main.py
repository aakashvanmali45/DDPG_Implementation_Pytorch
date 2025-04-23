import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from ddpg import DDPGAgent, train_ddpg

env = gym.make("Pendulum-v1", render_mode="human")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action,device)

num_episodes = 100
batch_size = 64
replay_buffer_size = 20000

rewards = train_ddpg(env, agent, num_episodes, batch_size, replay_buffer_size)

plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title('DDPG Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.savefig('ddpg_learning_curve.png')
plt.show()

state, _ = env.reset()
done = False
total_reward = 0

print("\nTesting the trained agent...")

while not done:
    action = agent.select_action(state, add_noise=False)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()
    
print(f"Test reward: {total_reward:.2f}")
env.close()