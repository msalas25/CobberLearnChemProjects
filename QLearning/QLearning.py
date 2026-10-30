
# Chapter 15: Q-Learning with FrozenLake

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Get the number of states and actions
num_states = env.observation_space.n
num_actions = env.action_space.n

print("--- FROZENLAKE ENVIRONMENT ---")
print("Number of States:", num_states)
print("Number of Actions:", num_actions)

# Create a Q-table filled with zeros
q_table = np.zeros((num_states, num_actions))

print("\n--- INITIAL Q-TABLE ---")
print(q_table)


# STEP 4: TRAIN THE Q-LEARNING AGENT

# Training settings

# Updated training settings
episodes = 100000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.9999
min_epsilon = 0.01


# Track successful episodes
rewards = []

print("\n--- TRAINING THE AGENT ---")

for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    steps = 0

    while not done and steps < 100:
        steps += 1
        # Explore or use the best learned action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Perform the action
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Update the Q-table using the Bellman equation
        if done:
            target = reward
        else:
            target = reward + discount_factor * np.max(q_table[new_state])

        q_table[state, action] += learning_rate * (
            target - q_table[state, action]
        )

        state = new_state
        total_reward += reward

    rewards.append(total_reward)

    # Gradually reduce random exploration
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Show training progress
    if (episode + 1) % 2000 == 0:
        recent_success = np.mean(rewards[-2000:]) * 100

        print(
            f"Episode {episode + 1}: "
            f"Success Rate = {recent_success:.2f}%"
        )

print("\n--- FINAL Q-TABLE ---")
print(np.round(q_table, 3))

print("\nTraining completed!")

# Save the trained Q-table
np.save("QLearning_qtable.npy", q_table)

print("Q-table saved successfully!")
# STEP 5: VISUALIZE AND TEST THE LEARNED POLICY

print("\n--- LEARNED POLICY ---")

# Map actions to directional arrows
arrows = ["←", "↓", "→", "↑"]

# Standard FrozenLake layout
lake = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

# Display the best learned action for each state
for row in range(4):
    policy_row = []

    for col in range(4):
        state = row * 4 + col
        tile = lake[row][col]

        if tile in ("S", "H", "G"):
            policy_row.append(tile)
        else:
            best_action = np.argmax(q_table[state])
            policy_row.append(arrows[best_action])

    print("  ".join(policy_row))

# Test the agent without exploration
test_episodes = 100
successes = 0

print("\n--- TESTING THE TRAINED AGENT ---")

for episode in range(test_episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Always choose the best learned action
        action = np.argmax(q_table[state])

        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = new_state

        if done and reward == 1:
            successes += 1

success_rate = (successes / test_episodes) * 100

print(f"Successful Episodes: {successes}/{test_episodes}")
print(f"Success Rate: {success_rate:.2f}%")

if success_rate >= 70:
    print("Great! The agent learned a successful policy.")
elif success_rate < 30:
    print("The agent needs more training or adjustment.")
else:
    print("The agent is learning, but there is room for improvement.")

print("\nStep 5 completed!")


# STEP 6: VISUALIZE LEARNING PROGRESS

# Calculate the success rate for every 2000 episodes
window_size = 2000

episode_numbers = []
success_rates = []

for i in range(window_size, len(rewards) + 1, window_size):
    recent_rewards = rewards[i - window_size:i]
    success_rate_window = np.mean(recent_rewards) * 100

    episode_numbers.append(i)
    success_rates.append(success_rate_window)

# Create the learning progress graph
plt.figure(figsize=(10, 6))

plt.plot(
    episode_numbers,
    success_rates,
    marker="o",
    markersize=3,
    label="Training Success Rate"
)

plt.title("Q-Learning Progress in FrozenLake")
plt.xlabel("Training Episodes")
plt.ylabel("Success Rate (%)")

plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the graph
plt.savefig("QLearning_progress.png", dpi=300)

# Display the graph
plt.show()

print("\nLearning progress graph saved successfully!")

env.close()
