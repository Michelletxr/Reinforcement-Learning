import sys
import gym
from contextlib import closing
import time
import numpy as np
from six import StringIO, b
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
# We re-initialize the Q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))
# List of outcomes to plot
outcomes = []
actions_descriptions = np.random.choice(["LEFT", "DOWN", "RIGHT", "UP"])

print('Q-table before training:')
print(qtable)
# Training
print("====================================TRAINING=================================")
# Hyperparameters
episodes = 1000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
for _ in range(episodes):
    state = env.reset()[0]
    done = False

    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")
    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        print('estado {}: {}'.format(state, qtable[state]))
        if np.max(qtable[state]) > 0:
          policy = np.max(qtable[state])
          #print("policy: ", policy)
          action = list(qtable[state]).index(policy)
        # If there's no best action (only zeros), take a random one
        else:
          action = env.action_space.sample() 
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, value, info = env.step(action)
        # Update Q(s,a)
        #print("ação", action)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        # Update our current state
        state = new_state
        #print("novo estado:", state)
        #print("novo estado:", state)
        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table after training:')
print(qtable)
# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

# Evaluation
print()
print("====================================EVALUATION=================================")
episodes = 1
nb_success = 0
for _ in range(episodes):
    state = env.reset()[0]
    done = False
    sequence = []
    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])
        # If there's no best action (only zeros), take a random one
        else:
          action = env.action_space.sample()
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, value, info = env.step(action)
        sequence.append(action)
        print(done)
        # Update our current state
        state = new_state
        # When we get a reward, it means we solved the game
        nb_success += reward
        print(env.render())
    print(f"Sequence = {sequence}")
# Let's check our success rate!
print (f"Success rate = {nb_success/episodes*100}%")


print("\n\n")
print("====================================Epsilon-Greedy=================================")
qtable = np.zeros((env.observation_space.n, env.action_space.n))
# Hyperparameters
episodes = 1000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease
# List of outcomes to plot
outcomes = []
print('Q-table before training:')
print(qtable)

# Training
print("====================================TRAINING=================================")
for _ in range(episodes):
    state = env.reset()[0]
    done = False
    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")
    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()
        # If random number < epsilon, take a random action
        if rnd < epsilon:
          action = env.action_space.sample()
        # Else, take the action with the highest value in the current state
        else:
          action = np.argmax(qtable[state])
        # Implement this action and move the agent in the desired direction
        new_state, reward, done, value, info = env.step(action)
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])

        # Update our current state
        state = new_state
        # If we have a reward, it means that our outcome is a success
        if reward:
          outcomes[-1] = "Success"
    # Update epsilon
    epsilon = max(epsilon - epsilon_decay, 0)
print()
print('===========================================')
print('Q-table after training:')
print(qtable)
# Plot outcomes
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
ax.set_facecolor('#efeeea')
plt.bar(range(len(outcomes)), outcomes, color="#0A047A", width=1.0)
plt.show()

