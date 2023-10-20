import numpy as np
import matplotlib.pyplot as plt

# Number of arms/actions (k)
k = 10

# Probability of randomly choosing a random action
epsilon = 0.1

# True action values (unknown in practice)
true_action_values = np.random.normal(0, 1, k)

# Initialize Q(a) and N(a)
Q = np.zeros(k)
N = np.zeros(k)

# Store rewards over time for plotting
reward_history = []

# Number of rounds/iterations
num_rounds = 1000

for t in range(1, num_rounds + 1):
    if np.random.rand() < epsilon:
        # Explore: Choose a random action with probability epsilon
        A = np.random.choice(k)
    else:
        # Exploit: Choose the action with the highest estimated value (breaking ties randomly)
        max_Q = np.max(Q)
        best_actions = np.where(Q == max_Q)[0]
        A = np.random.choice(best_actions)

    # Simulate the reward for the chosen action (true values are unknown)
    reward = np.random.normal(true_action_values[A], 1)

    # Update Q(a) and N(a)
    N[A] += 1
    Q[A] = Q[A] + (reward - Q[A]) / N[A]

    # Track the reward for each round
    reward_history.append(reward)

# Calculate the cumulative average reward
cumulative_average_reward = np.cumsum(reward_history) / np.arange(1, num_rounds + 1)

# Plot the cumulative average reward
plt.plot(range(1, num_rounds + 1), cumulative_average_reward)
plt.xlabel("Number of Rounds")
plt.ylabel("Cumulative Average Reward")
plt.title("Simple Bandit Algorithm")
plt.show()
