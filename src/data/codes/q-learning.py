"""
Q-learning on a grid world
# pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 4, 4
NUM_STATES = ROWS * COLS
NUM_ACTIONS = 4  # 0=Up, 1=Down, 2=Left, 3=Right
START_STATE = 0
GOAL_STATE = 15
TRAP_STATES = {5, 10}

alpha = 0.1
gamma = 0.9
epsilon = 0.2
N_EPISODES = 500
MAX_STEPS = 50

RNG = np.random.default_rng(42)


def idx_to_rc(s: int):
    r, c = divmod(s, COLS)
    return r, c


def rc_to_idx(r: int, c: int) -> int:
    return r * COLS + c


def get_next_state(state: int, action: int) -> int:
    r, c = idx_to_rc(state)
    if action == 0:
        r = max(r - 1, 0)
    elif action == 1:
        r = min(r + 1, ROWS - 1)
    elif action == 2:
        c = max(c - 1, 0)
    elif action == 3:
        c = min(c + 1, COLS - 1)
    return rc_to_idx(r, c)


Q = np.zeros((NUM_STATES, NUM_ACTIONS))
episode_rewards = np.zeros(N_EPISODES)
episode_steps = np.zeros(N_EPISODES, dtype=int)

for ep in range(N_EPISODES):
    state = START_STATE
    total_reward = 0.0
    for step in range(MAX_STEPS):
        if RNG.random() < epsilon:
            action = int(RNG.integers(0, NUM_ACTIONS))
        else:
            action = int(np.argmax(Q[state]))

        next_state = get_next_state(state, action)
        if next_state == GOAL_STATE:
            reward = 10.0
        elif next_state in TRAP_STATES:
            reward = -5.0
        else:
            reward = 0.0

        td_target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (td_target - Q[state, action])
        state = next_state
        total_reward += reward
        if state == GOAL_STATE:
            break
    episode_rewards[ep] = total_reward
    episode_steps[ep] = step + 1

print("Learned Q:\n", Q)

labels = ["Up", "Down", "Left", "Right"]
policy = []
for s in range(NUM_STATES):
    if s == GOAL_STATE:
        policy.append("Goal")
    elif s in TRAP_STATES:
        policy.append("Trap")
    else:
        policy.append(labels[int(np.argmax(Q[s]))])

print("Learned Policy:")
for r in range(ROWS):
    row = [policy[rc_to_idx(r, c)].ljust(8) for c in range(COLS)]
    print(" ".join(row))

print(f"\nAvg Reward/ep = {episode_rewards.mean():.4f}")
print(f"Avg Steps/ep  = {episode_steps.mean():.4f}")

fig, ax = plt.subplots()
ax.plot(episode_rewards, linewidth=1.5)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("Reward per Episode")
ax.grid(True)

fig2, ax2 = plt.subplots()
ax2.plot(episode_steps, linewidth=1.5)
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps")
ax2.set_title("Steps per Episode")
ax2.grid(True)

path = [START_STATE]
state = START_STATE
for _ in range(20):
    if state == GOAL_STATE:
        break
    action = int(np.argmax(Q[state]))
    state = get_next_state(state, action)
    path.append(state)
    if state in TRAP_STATES:
        break
print("Learned path:", path)
plt.show()
