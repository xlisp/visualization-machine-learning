import requests
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

from requests.adapters import HTTPAdapter

# Define the neural network for DQN
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

# Define the DQN agent that interacts with the FastAPI backend
class DQNAgent:
    def __init__(self, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=0.995):
        self.n_actions = 2  # Assume two actions: no flap (0) and flap (1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # State size based on the game state (birdY, pipes info, score, etc.)
        self.state_size = 1 + 2  # birdY + a simplified pipes state (2 values per pipe)
        self.model = DQN(self.state_size, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.gamma = gamma

    def send_action_to_backend(self, action):
        # Send the action (0 or 1) to FastAPI
        url = "http://localhost:8000/game/action/"
        response = requests.post(url, json={"action": action})
        return response.json()

    def get_state_from_backend(self):
        # Get the current game state from FastAPI
        url = "http://localhost:8000/game/state/"
        #response = requests.get(url)
        # Create a new requests session
        session = requests.Session()

        # Disable retries by setting max_retries to 0
        adapter = HTTPAdapter(max_retries=0)
        session.mount('http://', adapter)

        # Use the session for your requests
        response = session.get('http://localhost:8000/game/state/')


        game_state = response.json()

        # Simplify the game state to a vector (e.g., birdY and first pipe positions)
        ## birdY = game_state["birdY"] ---
        try:
            birdY = game_state["birdY"]
        except KeyError:
            birdY = 0  # Default value or some error handling
            print("Warning: birdY is missing in the game state")

        pipes = game_state["pipes"]
        if len(pipes) > 0:
            first_pipe_x = pipes[0]['x']
            first_pipe_y = pipes[0]['y']
        else:
            first_pipe_x = 0
            first_pipe_y = 0
        return np.array([birdY, first_pipe_x, first_pipe_y]), game_state['isGameOver'], game_state['score']

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

    def train_on_backend(self, episodes=2000):
        for episode in range(episodes):
            state, done, score = self.get_state_from_backend()  # Get initial state from backend
            episode_reward = 0

            for step in range(1000):  # Max steps per episode
                action = self.get_action(state)  # Get action from agent
                self.send_action_to_backend(action)  # Send action to FastAPI

                next_state, done, reward = self.get_state_from_backend()  # Get updated state from backend
                episode_reward += reward

                # Remember this experience for replay buffer
                self.remember(state, action, reward, next_state, done)

                # Train the agent with experience replay
                self.train()

                if done:
                    break

                state = next_state  # Update the current state

            # Update epsilon for exploration-exploitation trade-off
            self.update_epsilon()

            # Log progress every 10 episodes
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}")

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train_on_backend(episodes=2000)

## run------ flappy_bird_app  master @ uvicorn api:app --reload
# run 2 ---- flappy_bird_app  master @  python  dqn.py
## Episode 0, Total Reward: 0, Epsilon: 0.995
## requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /game/action/ (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x1088f4ed0>: Failed to establish a new connection: [Errno 61] Connection refused'))
