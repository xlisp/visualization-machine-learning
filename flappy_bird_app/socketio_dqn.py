import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import socketio

# Socket.IO client setup to interact with the FastAPI game server
sio = socketio.Client()

# Connect to the game server
sio.connect('http://localhost:8000')  # Adjust this if your server runs on a different address or port

# Neural network architecture for DQN
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

# DQN Agent class
class DQNAgent:
    def __init__(self, input_size, n_actions, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.model = DQN(input_size, n_actions).to(self.device)
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

# Global variables for game state
current_game_state = None
current_reward = None
game_over = False

# Handle game state received from server
@sio.on('game_update')
def on_game_update(data):
    global current_game_state, current_reward, game_over
    current_game_state = data['game_state']
    current_reward = data['reward']
    game_over = data['game_state']['isGameOver']

# Handle action acknowledgment from server
@sio.on('action_ack')
def on_action_ack(data):
    print(f"Action acknowledged by server: {data}")

def train_dqn(episodes=2000, max_steps=1000):
    global current_game_state, current_reward, game_over

    input_size = 3  # Example size for birdY, pipeX, pipeY; adjust based on your state
    n_actions = 2   # Number of actions (flap or no-flap)
    
    agent = DQNAgent(input_size, n_actions)
    scores = []

    for episode in range(episodes):
        sio.emit('game_reset')  # Reset game state for each episode
        state = None
        score = 0

        # Wait until game state is received
        while current_game_state is None:
            pass

        state = [current_game_state['birdY'], current_game_state['pipes'][0]['x'], current_game_state['pipes'][0]['y']]  # Example state

        for step in range(max_steps):
            action = agent.get_action(state)

            # Send the action to the game via Socket.IO
            sio.emit('game_action', {'action': 'flap' if action == 1 else 'no-flap', 'state': current_game_state})

            # Wait for the next game state to be received
            while current_game_state is None:
                pass

            next_state = [current_game_state['birdY'], current_game_state['pipes'][0]['x'], current_game_state['pipes'][0]['y']]
            reward = current_reward
            done = game_over

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            score += reward

            if done:
                break

        agent.update_epsilon()
        scores.append(score)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {score}, Epsilon: {agent.epsilon:.2f}")

    return agent, scores

if __name__ == "__main__":
    train_dqn()

