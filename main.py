import time
import mujoco
import mujoco.viewer
import time
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from cartpole_env import CartPoleEnv

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

# Usage:
# policy = PolicyNetwork()


def train(env, policy, optimizer, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            state = torch.tensor(state, dtype=torch.float32)
            probs = policy(state)
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)

            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

            if done:
                break
            state = next_state
            env.render()

        # Calculate the return
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f'Episode {episode}, Total Reward: {sum(rewards)}')

    env.close()

if __name__ == "__main__":
    env = CartPoleEnv()
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    train(env, policy, optimizer)