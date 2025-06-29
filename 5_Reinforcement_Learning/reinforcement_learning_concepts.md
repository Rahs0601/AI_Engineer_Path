# Reinforcement Learning Concepts

Reinforcement Learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. RL is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

## Key Concepts:

### 1. Agent and Environment

*   **Agent**: The learner or decision-maker. It interacts with the environment.
*   **Environment**: The world with which the agent interacts. It provides states and rewards to the agent in response to its actions.

### 2. State, Action, and Reward

*   **State (S)**: A complete description of the current situation in the environment. The agent observes the state.
*   **Action (A)**: A move made by the agent in a given state. Actions change the state of the environment.
*   **Reward (R)**: A scalar feedback signal from the environment to the agent, indicating how good or bad the agent's last action was. The agent's goal is to maximize the cumulative reward over time.

### 3. Policy

*   **Policy (π)**: The agent's strategy for choosing actions given a state. It maps states to actions.
    *   **Deterministic Policy**: `a = π(s)` (always chooses the same action for a given state).
    *   **Stochastic Policy**: `π(a|s)` (gives a probability distribution over actions for a given state).

### 4. Value Function

*   **Value Function (V(s))**: Predicts the expected cumulative reward an agent can expect to get starting from a given state `s` and following a particular policy `π`.
*   **Action-Value Function (Q(s, a))**: Predicts the expected cumulative reward an agent can expect to get starting from state `s`, taking action `a`, and then following a particular policy `π` thereafter.
    *   `Q(s, a)` is crucial for many RL algorithms, as it helps in deciding which action to take in a given state.

### 5. Model of the Environment

*   **Model-based RL**: The agent learns or is given a model of the environment (e.g., transition probabilities between states, expected rewards). It can then plan actions by simulating the environment.
*   **Model-free RL**: The agent learns directly from interactions with the environment, without explicitly learning a model of the environment. This is more common in complex environments.

### 6. Exploration vs. Exploitation

*   **Exploration**: The agent tries new actions to discover more about the environment and potentially find better rewards.
*   **Exploitation**: The agent uses its current knowledge to choose actions that it believes will yield the highest reward.
*   **Trade-off**: A fundamental challenge in RL is balancing exploration (to find optimal strategies) and exploitation (to maximize immediate rewards).
    *   **ε-greedy policy**: A common strategy where the agent explores with probability `ε` and exploits with probability `1-ε`.

### 7. Learning Paradigms

*   **On-policy learning**: The agent learns about the policy that it is currently following (e.g., SARSA).
*   **Off-policy learning**: The agent learns about an optimal policy independently of the agent's actions (e.g., Q-learning).

### 8. Common Reinforcement Learning Algorithms

*   **Q-Learning**: A model-free, off-policy RL algorithm that learns the optimal action-value function `Q(s, a)`. It updates Q-values based on the Bellman equation.
    *   `Q(s, a) = Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]`
        *   `α` (alpha): Learning rate
        *   `γ` (gamma): Discount factor (determines the importance of future rewards)
*   **SARSA (State-Action-Reward-State-Action)**: A model-free, on-policy RL algorithm. Similar to Q-learning, but the update rule uses the Q-value of the *next action taken* rather than the maximum possible Q-value.
*   **Deep Q-Networks (DQN)**: Combines Q-learning with deep neural networks to handle environments with large or continuous state spaces.
*   **Policy Gradients**: A family of algorithms that directly learn a policy function that maps states to actions, rather than learning value functions.
*   **Actor-Critic Methods**: Combine value-based and policy-based methods. An "actor" learns the policy, and a "critic" learns the value function to guide the actor.

## Resources:

*   **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto** (The classic textbook)
*   **David Silver's Reinforcement Learning Course (UCL)** (Lecture videos available online)
*   **OpenAI Gym**: A toolkit for developing and comparing reinforcement learning algorithms.
