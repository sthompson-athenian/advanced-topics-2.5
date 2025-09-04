#!/usr/bin/env python
# coding: utf-8
from random import random, randint, seed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class Bandit:
    def __init__(self, num_arms, initial_radius=0):
        self.num_arms = num_arms
        self.arm_means = np.random.uniform(-initial_radius, initial_radius, self.num_arms)

    def pull_arm(self, arm: int, shuffle=False):
        res = np.random.normal(self.arm_means[arm], 1)
        if shuffle:
            self.shuffle()
        return res

    def shuffle(self, sd=0.01):
        increment = np.random.normal(0, sd, self.num_arms)
        self.arm_means += increment

    # Determine which arm would have been the optimal pick
    def optimal(self):
        return np.argmax(self.arm_means)


def violin():
    bandit = Bandit(num_arms=4, initial_radius=0)
    num_pulls = 10000

    # rewards no shuffle
    rewards_no_shuffling = []
    for arm in range(bandit.num_arms):
        rewards_no_shuffling.append([bandit.pull_arm(arm) for _ in range(num_pulls)])

    # rewards with shuffle
    rewards_with_shuffling = []
    for arm in range(bandit.num_arms):
        rewards_with_shuffling.append([bandit.pull_arm(arm, True) for _ in range(num_pulls)])

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(1, bandit.num_arms + 1)

    # no shuffle
    parts1 = ax.violinplot(rewards_no_shuffling, positions=positions - 0.15, widths=0.25,
                           showmeans=True, showextrema=False, showmedians=False)

    # with shuffle
    parts2 = ax.violinplot(rewards_with_shuffling, positions=positions + 0.15, widths=0.25,
                           showmeans=True, showextrema=False, showmedians=False)

    for pc in parts1['bodies']:
        pc.set_facecolor('blue')
        pc.set_alpha(0.5)
    for pc in parts2['bodies']:
        pc.set_facecolor('orange')
        pc.set_alpha(0.5)

    # legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, label='No Shuffle'),
        Patch(facecolor='orange', alpha=0.5, label='With Shuffle')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # labels
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Arm {i}" for i in range(bandit.num_arms)])
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward Distributions of {bandit.num_arms}-Armed Bandit")
    ax.grid(alpha=0.3)

    plt.show()


def RL(arms: int, trials: int, epsilon: float, alpha=None):
    estimates = [0 for _ in range(arms)]  # Estimates for each arm
    arm_running_totals = [0 for _ in range(arms)]  # Total reward for each arm
    trials_per_arm = [0 for _ in range(arms)]  # Number of times each arm has been pulled

    optimal_count = 0  # Total optimal choices made
    optimal_record = []  # Record of % optimal choice at each step

    reward_sum = 0  # Total rewards from all arms
    avg_reward_history = []  # Record of average reward at each step

    rlb = Bandit(arms)
    for step in range(trials):
        # Make explore / exploit decision
        if random() > epsilon:
            # Exploit
            # Note: selects first if there's a tie (ideally, should be random)
            arm_index = abs(np.argmax(estimates))  # abs fixes unsigned int issue
        else:
            # Explore
            arm_index = randint(0, arms - 1)

        # Determine if the optimal choice was made
        if arm_index == rlb.optimal():
            optimal_count += 1
        # Track our current optimal choice percentage
        optimal_record.append(optimal_count / (step + 1))

        res = rlb.pull_arm(arm_index, False)

        if alpha is not None:
            if step == 0:
                estimates[arm_index] = res
            else:
                estimates[arm_index] = estimates[arm_index] + alpha * (res - estimates[arm_index])
        else:
            trials_per_arm[arm_index] += 1
            arm_running_totals[arm_index] += res
            estimates[arm_index] = arm_running_totals[arm_index] / trials_per_arm[arm_index]

        reward_sum += res
        avg_reward_history.append(reward_sum / (step + 1))  # Compute new average overall reward

        rlb.shuffle()
    return optimal_record, avg_reward_history


def run_trials():
    np.random.seed(42)
    seed(42)
    sa_optimal_record, sa_avg_reward_history = RL(10, 100000, 0.1)
    np.random.seed(42)
    seed(42)
    ssp_optimal_record, ssp_avg_reward_history = RL(10, 100000, 0.1, 0.1)

    # Create x values
    x = np.arange(1, len(sa_avg_reward_history) + 1)

    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot - Reward History
    ax1.plot(x, sa_avg_reward_history, label='Sample Average')
    ax1.plot(x, ssp_avg_reward_history, label='Step-Size Parameter')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward History Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot - Optimal Action
    ax2.plot(x, sa_optimal_record, label='Sample Average')
    ax2.plot(x, ssp_optimal_record, label='Step-Size Parameter')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal action')
    ax2.set_title('% Optimal Action Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()


run_trials()
