import gymnasium as gym
import numpy as np

# ==========================================
# 1. CREATE THE ENVIRONMENT
# ==========================================

env = gym.make("CartPole-v1")


# ==========================================
# RANDOM ACTION EXPERIMENT
# ==========================================

print("\n--- RANDOM ACTION EXPERIMENT ---")

random_rewards = []
random_steps = []

for episode in range(10):
    observation, info = env.reset()

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    random_rewards.append(total_reward)
    random_steps.append(steps)

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

print("\n--- RANDOM AGENT SUMMARY ---")
print(f"Average Reward: {np.mean(random_rewards):.2f}")
print(f"Average Steps: {np.mean(random_steps):.2f}")
bins = [
    np.linspace(-2.4, 2.4, 6),
    np.linspace(-3, 3, 6),
    np.linspace(-0.2095, 0.2095, 12),
    np.linspace(-3.5, 3.5, 12)
]

def discretize_state(observation):
    return tuple(
        np.digitize(observation[i], bins[i])
        for i in range(len(observation))
    )

# ==========================================
# 3. CREATE THE Q-TABLE
# ==========================================

q_table = np.zeros((7, 7, 13, 13, 2))

# ==========================================
# 4. SET LEARNING PARAMETERS
# ==========================================

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

num_episodes = 5000

# ==========================================
# 5. TRAIN THE Q-LEARNING AGENT
# ==========================================

for episode in range(num_episodes):

    observation, info = env.reset()
    state = discretize_state(observation)

    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:

        # Exploration or exploitation
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Apply action
        next_observation, reward, terminated, truncated, info = env.step(action)

        next_state = discretize_state(next_observation)

        # Q-learning update
        old_value = q_table[state][action]

        if terminated:
            next_max = 0
        else:
            next_max = np.max(q_table[next_state])

        new_value = old_value + learning_rate * (
            reward + discount_factor * next_max - old_value
        )

        q_table[state][action] = new_value

        state = next_state
        total_reward += reward

    # Reduce exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        print(
            f"Episode {episode + 1}: "
            f"Reward = {total_reward}, "
            f"Epsilon = {epsilon:.3f}"
        )

env.close()

# ==========================================
# 6. SAVE THE LEARNED Q-TABLE
# ==========================================

np.save("cartpole_q_table.npy", q_table)

print("Q-learning training completed!")
print("Q-table saved successfully!")

# ==========================================
# 7. TEST THE TRAINED AGENT
# ==========================================

test_env = gym.make("CartPole-v1", render_mode="human")

q_learning_rewards = []

for episode in range(5):

    observation, info = test_env.reset()

    total_reward = 0
    steps = 0

    terminated = False
    truncated = False

    while not terminated and not truncated:

        state = discretize_state(observation)

        # Choose the best learned action
        action = np.argmax(q_table[state])

        observation, reward, terminated, truncated, info = test_env.step(action)

        total_reward += reward
        steps += 1

    print(f"Test Episode {episode + 1}")
    print(f"Total Reward: {total_reward}")
    print(f"Number of Steps: {steps}")
    print("-------------------------")

test_env.close()


# ==========================================
# 8. VISUALIZE RANDOM VS Q-LEARNING
# ==========================================

import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent

# Rewards from the five Q-learning test episodes

q_learning_rewards.append(total_reward)
# Calculate average rewards
random_average = np.mean(random_rewards)
q_learning_average = np.mean(q_learning_rewards)

# Create comparison graph
plt.figure(figsize=(8, 5))

methods = ["Random Agent", "Q-Learning Agent"]
average_rewards = [random_average, q_learning_average]

bars = plt.bar(
    methods,
    average_rewards,
    color=["steelblue", "mediumseagreen"],
    width=0.55
)

# Display values above bars
for bar, value in zip(bars, average_rewards):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}",
        ha="center",
        fontsize=11
    )

plt.ylabel("Average Reward")
plt.title("CartPole: Random Actions vs Q-Learning")
plt.ylim(0, max(average_rewards) * 1.2)
plt.tight_layout()

# Save graph
plt.savefig(folder / "CartPole_Reward_Comparison.png", dpi=300)
plt.close()

print("\nCartPole comparison graph saved successfully!")
print("Chapter 18 visualization completed!")
