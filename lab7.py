import numpy as np
import matplotlib.pyplot as plt

# Function for non-stationary bandit environment
def nonstationary_bandit(action):
    means = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.45, 0.55]
    return means[action - 1] + np.random.randn()

# Simulate reward for an action
action = 2
reward = nonstationary_bandit(action)
print(f"Reward for action {action}: {reward}")

# Function for binary bandit (type A)
def binary_bandit_A(action):
    probabilities = [0.1, 0.2]
    return 1 if np.random.rand() < probabilities[action - 1] else 0

# Function for binary bandit (type B)
def binary_bandit_B(action):
    probabilities = [0.8, 0.9]
    return 1 if np.random.rand() < probabilities[action - 1] else 0

# Exploration vs Exploitation experiment with different epsilon values
steps = 50
rewards = np.zeros((steps, 2, 3))
total_value = np.zeros((steps, 3))

for idx, epsilon in enumerate([0.01, 0.1, 0.3]):
    rewards[:, :, idx] = 0
    step = 0

    while step < steps:
        # Exploration phase
        if np.random.rand() < epsilon or step == 0:
            action = np.random.randint(2) + 1
            value = binary_bandit_A(action)
            total_value[step, idx] = value + (total_value[step - 1, idx] if step > 0 else 0)
            rewards[step, :, idx] = [value, action]
        else:
            # Exploitation phase
            action_1 = np.sum(rewards[:step, 0, idx] == 1) if np.sum(rewards[:step, 1, idx] == 1) else 0
            action_2 = np.sum(rewards[:step, 0, idx] == 1) if np.sum(rewards[:step, 1, idx] == 2) else 0
            action = 1 if action_2 > action_1 else 2

            value = binary_bandit_A(action)
            total_value[step, idx] = value + total_value[step - 1, idx]
            rewards[step, :, idx] = [value, action]

        step += 1

# Plot results for Bandit A with different epsilon values
plt.plot(total_value[:, 0], label='ε = 0.01')
plt.plot(total_value[:, 1], label='ε = 0.1')
plt.plot(total_value[:, 2], label='ε = 0.3')
plt.ylim([0, 50])
plt.xlabel('Time Steps')
plt.ylabel('Total Successes')
plt.legend()
plt.title('Binary Bandit A - Exploration vs Exploitation (ε = 0.01, 0.1, 0.3)')
plt.show()

# Rewards vs Time Steps plot for Bandit A
plt.figure(figsize=(6, 4))
for idx, epsilon in enumerate([0.01, 0.1, 0.3]):
    plt.plot(range(steps), rewards[:, 0, idx], label=f"ε = {epsilon}")

plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.ylim(-1, 2)
plt.title('Rewards vs Time Steps for Bandit A')
plt.legend()
plt.show()

# Repeat for Binary Bandit B
steps = 50
rewards = np.zeros((steps, 2, 3))
total_value = np.zeros((steps, 3))

for idx, epsilon in enumerate([0.01, 0.1, 0.3]):
    rewards[:, :, idx] = 0
    step = 0

    while step < steps:
        # Exploration phase
        if np.random.rand() < epsilon or step == 0:
            action = np.random.randint(2) + 1
            value = binary_bandit_B(action)
            total_value[step, idx] = value + (total_value[step - 1, idx] if step > 0 else 0)
            rewards[step, :, idx] = [value, action]
        else:
            # Exploitation phase
            action_1 = np.sum(rewards[:step, 0, idx] == 1) if np.sum(rewards[:step, 1, idx] == 1) else 0
            action_2 = np.sum(rewards[:step, 0, idx] == 1) if np.sum(rewards[:step, 1, idx] == 2) else 0
            action = 1 if action_2 > action_1 else 2

            value = binary_bandit_B(action)
            total_value[step, idx] = value + total_value[step - 1, idx]
            rewards[step, :, idx] = [value, action]

        step += 1

# Plot results for Bandit B with different epsilon values
plt.plot(total_value[:, 0], label='ε = 0.01')
plt.plot(total_value[:, 1], label='ε = 0.1')
plt.plot(total_value[:, 2], label='ε = 0.3')
plt.ylim([0, 50])
plt.xlabel('Time Steps')
plt.ylabel('Total Successes')
plt.legend()
plt.title('Binary Bandit B - Exploration vs Exploitation (ε = 0.01, 0.1, 0.3)')
plt.show()

# Rewards vs Time Steps plot for Bandit B
plt.figure(figsize=(6, 4))
for idx, epsilon in enumerate([0.01, 0.1, 0.3]):
    plt.plot(range(steps), rewards[:, 0, idx], label=f"ε = {epsilon}")

plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.ylim(-1, 2)
plt.title('Rewards vs Time Steps for Bandit B')
plt.legend()
plt.show()

# Multi-armed bandit simulation (10 arms)
steps = 10000
rewards = np.zeros((steps, 2, 3))
total_value = np.zeros((steps, 3))
cnt_actions = np.zeros((10, 3))

for idx, epsilon in enumerate([0.01, 0.1, 0.3]):
    total_value[:, idx] = 0
    rewards[:, :, idx] = 0
    cnt_actions[:, idx] = 0
    step = 1

    while step <= steps:
        # Exploration phase
        if np.random.rand() < epsilon or step == 1:
            action = np.random.randint(1, 11)
            value = nonstationary_bandit(action)
            total_value[step - 1, idx] = value + (total_value[step - 2, idx] if step > 1 else 0)
            rewards[step - 1, :, idx] = [value, action]
        else:
            # Exploitation phase
            actions = np.zeros((10, 2))
            for s in range(step - 1):
                actions[int(rewards[s, 1, idx]) - 1, 0] += rewards[s, 0, idx]
                actions[int(rewards[s, 1, idx]) - 1, 1] += 1

            action = np.argmax(np.divide(actions[:, 0], np.where(actions[:, 1] == 0, 1, actions[:, 1]))) + 1
            value = nonstationary_bandit(action)
            total_value[step - 1, idx] = value + total_value[step - 2, idx]
            rewards[step - 1, :, idx] = [value, action]
            cnt_actions[:, idx] = actions[:, 1]

        step += 1

# Plotting Results for 10 Armed Bandit
plt.figure(figsize=(10, 6))

# Plot rewards vs time steps
plt.subplot(2, 2, 1)
plt.plot(rewards[:, 1, 1])
plt.xlabel('Time Steps')
plt.ylabel('Actions')
plt.title('10 Armed Bandit - ε = 0.1')

plt.tight_layout()
plt.show()

# Plot number of times actions taken
plt.subplot(2, 2, 2)
plt.scatter(range(1, 11), cnt_actions[:, 1])
plt.xlabel('Actions')
plt.ylabel('Number of Times Taken')
plt.title('10 Armed Bandit - ε = 0.1')

# Plot total value received vs time steps
plt.figure(figsize=(8, 6))
plt.plot(total_value[:, 1])
plt.xlabel('Time Steps')
plt.ylabel('Total Value Received')
plt.title('Total Value Received vs Time Steps')
plt.tight_layout()
plt.show()

avg_reward = np.cumsum(rewards[:, 0, 1]) / np.arange(1, steps + 1)

# Plot average reward vs time steps
plt.figure(figsize=(8, 6))
plt.plot(avg_reward)
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('10 Armed Bandit - (ε = 0.1)')
plt.tight_layout()
plt.show()