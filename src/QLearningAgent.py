import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from Game2048 import Game2048


class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.q_table = defaultdict(float)
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_table[(state.tobytes(), action)] for action in self.actions]
        max_q = max(q_values)
        return random.choice(
            [
                action
                for action, q_value in zip(self.actions, q_values)
                if q_value == max_q
            ]
        )

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[(state.tobytes(), action)]
        max_next_q = max(
            [
                self.q_table[(next_state.tobytes(), next_action)]
                for next_action in self.actions
            ]
        )
        self.q_table[(state.tobytes(), action)] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )


def train_2048_agent(episodes=1000000):
    game = Game2048()
    agent = QLearningAgent(actions=["left", "right", "up", "down"])
    rewards = []

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0

        while not game.is_game_over():
            action = agent.choose_action(state)
            if game.move(action):
                next_state = game.get_state()
                reward = np.sum(next_state) - np.sum(state)
                agent.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            else:
                reward = -1
                agent.learn(state, action, reward, state)

        rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return agent, rewards


def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Evolution of Total Reward during Training")
    plt.legend()
    plt.show()


agent, rewards = train_2048_agent(episodes=10000)
plot_rewards(rewards)
