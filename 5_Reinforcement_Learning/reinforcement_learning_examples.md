# Reinforcement Learning Examples (Q-Learning)

import numpy as np
import gym
import random
import matplotlib.pyplot as plt

print("--- Reinforcement Learning Examples (Q-Learning) ---")

# --- 1. Q-Learning on FrozenLake-v1 Environment ---
print("\n--- 1. Q-Learning on FrozenLake-v1 ---")

# Create the environment
# is_slippery=False makes it deterministic, easier to learn for demonstration
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')

# Initialize Q-table with zeros
# Q-table dimensions: (number of states, number of actions)
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros((n_states, n_actions))

print(f"Number of States: {n_states}")
print(f"Number of Actions: {n_actions}")
print("Initial Q-table (first 5 rows):\n", q_table[:5, :])

# Q-Learning parameters
learning_rate = 0.9       # Alpha (α)
discount_factor = 0.9     # Gamma (γ)
epsilon = 1.0             # Epsilon for epsilon-greedy policy
max_epsilon = 1.0         # Exploration probability at start
min_epsilon = 0.01        # Minimum exploration probability
decay_rate = 0.005        # Exponential decay rate for epsilon

n_episodes = 10000        # Total episodes for training
max_steps_per_episode = 100 # Max steps per episode

rewards_per_episode = []

# Q-Learning Training Loop
print("\nStarting Q-Learning training...")
for episode in range(n_episodes):
    state, info = env.reset() # Reset environment for new episode
    done = False
    truncated = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > epsilon: # Exploit (choose best action from Q-table)
            action = np.argmax(q_table[state, :])
        else: # Explore (choose random action)
            action = env.action_space.sample()

        # Take action and observe new state and reward
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q-table using Bellman equation
        # Q(s,a) = Q(s,a) + α * [R + γ * max(Q(s',a')) - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * \
                                 (reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state
        rewards_current_episode += reward

        if done or truncated:
            break

    # Decay epsilon (exploration rate)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(rewards_current_episode)

print("Q-Learning training finished.")

# Calculate and plot rewards per episode
sum_rewards = np.zeros(n_episodes)
for t in range(n_episodes):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)]) # Sum of rewards over last 100 episodes
plt.figure(figsize=(10, 6))
plt.plot(sum_rewards)
plt.title('Rewards per Episode (Sum over 100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards')
plt.grid(True)
plt.show()

print("\nFinal Q-table (first 5 rows):\n", q_table[:5, :])

# Test the trained agent
print("\nTesting the trained agent (first 5 episodes):")
for episode in range(5):
    state, info = env.reset()
    done = False
    truncated = False
    print(f"\n--- Episode {episode + 1} ---")
    print(env.render())
    total_reward = 0
    for step in range(max_steps_per_episode):
        action = np.argmax(q_table[state, :]) # Choose best action from learned Q-table
        new_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(env.render())
        state = new_state
        if done or truncated:
            print(f"Episode finished after {step + 1} steps. Total Reward: {total_reward}")
            break

env.close()
