from flappy_bird_cl2 import FlappyBirdEnv
import numpy as np

# To implement a DQN (Deep Q-Network) agent using PyTorch to play the custom `FlappyBirdEnv` environment, we will need to create an agent that can interact with the environment and learn from the feedback it receives (rewards) via experience replay and Q-learning.

# Here’s a high-level plan to structure the DQN:

# 1. **Neural Network Architecture**: A model that approximates the Q-value function, which takes the observation (state) and outputs Q-values for each action.
# 2. **Experience Replay**: A mechanism to store the agent's experiences (state, action, reward, next state) in a buffer, and sample random batches to train the model.
# 3. **Training the Agent**: Use the Bellman equation to train the model by minimizing the temporal difference error.
# 4. **Exploration vs Exploitation (ε-greedy policy)**: Balance between exploring the environment (choosing random actions) and exploiting the learned policy (choosing the best-known action).
# 5. **Target Network**: A separate network to stabilize training by providing a more stable target for Q-learning.

# Here's the code to integrate DQN with the FlappyBird environment:

# ### Step 1: DQN Network
# The DQN will be a convolutional neural network that takes the screen observation as input and outputs Q-values for the possible actions (jump or not jump).

# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the Q-Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
# ```

# ### Step 2: DQN Agent and Experience Replay

# ```python
class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.model = DQN(input_shape, n_actions).to(device)
        self.target_model = DQN(input_shape, n_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Decay rate per episode
        self.epsilon_min = 0.01
        self.batch_size = 32

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(n_actions)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.model(state)
            return q_values.max(1)[1].item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
# ```

# ### Step 3: Training Loop

# The main loop will initialize the environment, agent, and start the training process:

# ```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = FlappyBirdEnv()

input_shape = (3, env.SCREEN_HEIGHT, env.SCREEN_WIDTH)
n_actions = env.action_space.n
agent = DQNAgent(input_shape, n_actions)

num_episodes = 500
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

        env.render()

    agent.update_target_model()
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

env.close()
# ```

# ### Explanation:
# - **DQN Network**: A CNN that processes the image observation (screen) and outputs Q-values for the actions.
# - **DQNAgent**: Responsible for selecting actions, storing experiences, and training the network.
# - **Training Loop**: The agent interacts with the environment, collects experiences, and uses the DQN to learn over time. The target network is updated periodically to stabilize learning.

# Let me know if you want further refinements or help with the training process!
