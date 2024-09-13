import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

from drive_game2 import DrivingEnv

class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.env = env
        self.n_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        state_size = len(env.get_state())
        self.model = DQN(state_size, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = gamma

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_dqn(env, episodes=2000, max_steps=1000, render_interval=10):
    agent = DQNAgent(env)
    scores = []

    for episode in range(episodes):
        state = env.reset()
        score = 0

        for step in range(max_steps):
            if episode % render_interval == 0:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            score += reward

            if done:
                break

            if episode % render_interval == 0:
                pygame.event.pump()

        agent.update_epsilon()
        scores.append(score)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

    return agent, scores

if __name__ == "__main__":
    env = DrivingEnv()
    agent, scores = train_dqn(env, episodes=3000, render_interval=50)

    # Test the trained agent
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        score += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        pygame.event.pump()
        time.sleep(0.03)

    print(f"Final Score: {score}")
    env.close()

## ------ run good ---------
# @ python drive_torch_dqn_nocnn.py
# Episode: 0, Score: 201, Epsilon: 0.99
# Episode: 10, Score: 538, Epsilon: 0.95
# Episode: 20, Score: 357, Epsilon: 0.90
# Episode: 30, Score: 200, Epsilon: 0.86
# Episode: 40, Score: 784, Epsilon: 0.81
# Episode: 50, Score: 1000, Epsilon: 0.77
# Episode: 60, Score: 265, Epsilon: 0.74
# Episode: 70, Score: 309, Epsilon: 0.70
# Episode: 80, Score: 160, Epsilon: 0.67
# Episode: 90, Score: 260, Epsilon: 0.63
# Episode: 100, Score: 511, Epsilon: 0.60
# Episode: 110, Score: 576, Epsilon: 0.57
# Episode: 120, Score: 233, Epsilon: 0.55
# Episode: 130, Score: 194, Epsilon: 0.52
# Episode: 140, Score: 211, Epsilon: 0.49
# Episode: 150, Score: 218, Epsilon: 0.47
# Episode: 160, Score: 319, Epsilon: 0.45
# Episode: 170, Score: 531, Epsilon: 0.42
# Episode: 180, Score: 171, Epsilon: 0.40
# Episode: 190, Score: 181, Epsilon: 0.38
# Episode: 200, Score: 166, Epsilon: 0.37
# Episode: 210, Score: 157, Epsilon: 0.35
# Episode: 220, Score: 326, Epsilon: 0.33
# Episode: 230, Score: 185, Epsilon: 0.31
# Episode: 240, Score: 306, Epsilon: 0.30
# Episode: 250, Score: 425, Epsilon: 0.28
# Episode: 260, Score: 240, Epsilon: 0.27
# Episode: 270, Score: 221, Epsilon: 0.26
# Episode: 280, Score: 256, Epsilon: 0.24
# Episode: 290, Score: 202, Epsilon: 0.23
# Episode: 300, Score: 179, Epsilon: 0.22
# Episode: 310, Score: 223, Epsilon: 0.21
# Episode: 320, Score: 161, Epsilon: 0.20
# Episode: 330, Score: 211, Epsilon: 0.19
# Episode: 340, Score: 222, Epsilon: 0.18
# Episode: 350, Score: 243, Epsilon: 0.17
# Episode: 360, Score: 452, Epsilon: 0.16
# Episode: 370, Score: 311, Epsilon: 0.16
# Episode: 380, Score: 381, Epsilon: 0.15
# Episode: 390, Score: 335, Epsilon: 0.14
# Episode: 400, Score: 344, Epsilon: 0.13
# Episode: 410, Score: 213, Epsilon: 0.13
# Episode: 420, Score: 457, Epsilon: 0.12
# Episode: 430, Score: 308, Epsilon: 0.12
# Episode: 440, Score: 258, Epsilon: 0.11
# Episode: 450, Score: 232, Epsilon: 0.10
# Episode: 460, Score: 403, Epsilon: 0.10
# Episode: 470, Score: 305, Epsilon: 0.09
# Episode: 480, Score: 410, Epsilon: 0.09
# Episode: 490, Score: 503, Epsilon: 0.09
# Episode: 500, Score: 166, Epsilon: 0.08
# Episode: 510, Score: 267, Epsilon: 0.08
# Episode: 520, Score: 183, Epsilon: 0.07
# Episode: 530, Score: 267, Epsilon: 0.07
# Episode: 540, Score: 197, Epsilon: 0.07
# Episode: 550, Score: 182, Epsilon: 0.06
# Episode: 560, Score: 404, Epsilon: 0.06
# Episode: 570, Score: 283, Epsilon: 0.06
# Episode: 580, Score: 162, Epsilon: 0.05
# Episode: 590, Score: 179, Epsilon: 0.05
# Episode: 600, Score: 757, Epsilon: 0.05
# Episode: 610, Score: 391, Epsilon: 0.05
# Episode: 620, Score: 253, Epsilon: 0.04
# Episode: 630, Score: 212, Epsilon: 0.04
# Episode: 640, Score: 542, Epsilon: 0.04
# Episode: 650, Score: 191, Epsilon: 0.04
# Episode: 660, Score: 169, Epsilon: 0.04
# Episode: 670, Score: 157, Epsilon: 0.03
# Episode: 680, Score: 164, Epsilon: 0.03
# Episode: 690, Score: 573, Epsilon: 0.03
# Episode: 700, Score: 285, Epsilon: 0.03
# Episode: 710, Score: 546, Epsilon: 0.03
# Episode: 720, Score: 375, Epsilon: 0.03
# Episode: 730, Score: 200, Epsilon: 0.03
# Episode: 740, Score: 198, Epsilon: 0.02
# Episode: 750, Score: 1000, Epsilon: 0.02
# Episode: 760, Score: 207, Epsilon: 0.02
# Episode: 770, Score: 816, Epsilon: 0.02
# Episode: 780, Score: 182, Epsilon: 0.02
# Episode: 790, Score: 517, Epsilon: 0.02
# Episode: 800, Score: 192, Epsilon: 0.02
# Episode: 810, Score: 869, Epsilon: 0.02
# Episode: 820, Score: 172, Epsilon: 0.02
# Episode: 830, Score: 300, Epsilon: 0.02
# Episode: 840, Score: 678, Epsilon: 0.01
# Episode: 850, Score: 154, Epsilon: 0.01
# Episode: 860, Score: 267, Epsilon: 0.01
# Episode: 870, Score: 224, Epsilon: 0.01
# Episode: 880, Score: 331, Epsilon: 0.01
# Episode: 890, Score: 437, Epsilon: 0.01
# Episode: 900, Score: 157, Epsilon: 0.01
# Episode: 910, Score: 341, Epsilon: 0.01
# Episode: 920, Score: 348, Epsilon: 0.01
# Episode: 930, Score: 196, Epsilon: 0.01
# Episode: 940, Score: 224, Epsilon: 0.01
# Episode: 950, Score: 225, Epsilon: 0.01
# Episode: 960, Score: 161, Epsilon: 0.01
# Episode: 970, Score: 559, Epsilon: 0.01
# Episode: 980, Score: 980, Epsilon: 0.01
# Episode: 990, Score: 244, Epsilon: 0.01
# Final Score: 578
# 
