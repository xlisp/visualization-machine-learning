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

import pygame
import numpy as np
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        self.SCREEN_WIDTH = 288
        self.SCREEN_HEIGHT = 512
        self.BIRD_WIDTH = 34
        self.BIRD_HEIGHT = 24
        self.PIPE_WIDTH = 52
        self.PIPE_GAP = 100
        self.GRAVITY = 1
        self.JUMP_STRENGTH = 10
        self.PIPE_VELOCITY = 3

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Load images
        self.background_img = pygame.image.load('background.png').convert()
        self.bird_img = pygame.image.load('bird.png').convert_alpha()
        self.pipe_img = pygame.image.load('pipe.png').convert_alpha()
        
        self.reset()

    def get_state(self):
        next_pipe = self.pipes[0]
        return np.array([
            self.bird_y,
            self.bird_velocity,
            next_pipe[0] - 50,  # x distance to next pipe
            next_pipe[1] - self.bird_y,  # y distance to next pipe's top
            next_pipe[1] + self.PIPE_GAP - self.bird_y  # y distance to next pipe's bottom
        ])

    def step(self, action):
        reward = 0
        self.bird_velocity += self.GRAVITY
        if action == 1:
            self.bird_velocity = -self.JUMP_STRENGTH
        self.bird_y += self.bird_velocity

        # Move pipes
        for pipe in self.pipes:
            pipe[0] -= self.PIPE_VELOCITY

        # Remove passed pipes and add new ones
        if self.pipes[0][0] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            self.score += 1
            reward = 1
            new_pipe = [self.SCREEN_WIDTH, random.randint(100, self.SCREEN_HEIGHT - 100 - self.PIPE_GAP)]
            self.pipes.append(new_pipe)

        # Check for collisions
        if (self.bird_y < 0 or self.bird_y > self.SCREEN_HEIGHT or
            (0 < self.pipes[0][0] < self.BIRD_WIDTH + 50 and
             (self.bird_y < self.pipes[0][1] or self.bird_y > self.pipes[0][1] + self.PIPE_GAP))):
            reward = -1
            return self.get_state(), reward, True, False, {}

        return self.get_state(), reward, False, False, {}

    def reset(self):
        self.bird_y = self.SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = [[self.SCREEN_WIDTH, random.randint(100, self.SCREEN_HEIGHT - 100 - self.PIPE_GAP)]]
        self.score = 0
        return self.get_state()

    def render(self):
        self.screen.blit(self.background_img, (0, 0))
        
        # Draw pipes
        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, (pipe[0], pipe[1] - self.PIPE_GAP - 320))
            self.screen.blit(pygame.transform.flip(self.pipe_img, False, True), (pipe[0], pipe[1] + self.PIPE_GAP))

        # Draw bird
        self.screen.blit(self.bird_img, (50, self.bird_y))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train_dqn(env, episodes=1000, max_steps=1000, render_interval=10):
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
    env = FlappyBirdEnv()
    agent, scores = train_dqn(env, episodes=1000, render_interval=50)

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

