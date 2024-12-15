import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class GoNetwork(nn.Module):
    def __init__(self, board_size=9):
        super(GoNetwork, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers to process the board state
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Fully connected layers for move selection
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * board_size * board_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, board_size * board_size)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        conv_out = self.conv_layers(x)
        move_probabilities = self.fc_layers(conv_out)
        return move_probabilities

class GoQLearningAgent:
    def __init__(self, board_size=9, learning_rate=0.001, epsilon=0.1):
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural network for Q-learning
        self.q_network = GoNetwork(board_size).to(self.device)
        self.target_network = GoNetwork(board_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Hyperparameters
        self.epsilon = epsilon
        self.gamma = 0.99
        self.batch_size = 32
        self.memory = []
    
    def select_move(self, board_state, valid_moves):
        # Convert board state to tensor
        board_tensor = torch.FloatTensor(board_state).to(self.device)
        
        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Use Q-network to select best move
        with torch.no_grad():
            q_values = self.q_network(board_tensor)
            valid_q_values = q_values[valid_moves]
            best_move = valid_moves[torch.argmax(valid_q_values)]
        
        return best_move
    
    def train(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Sample batch for training when memory is sufficiently large
        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            
            # Prepare batch tensors
            states = torch.FloatTensor([exp[0] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
            next_states = torch.FloatTensor([exp[3] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)
            
            # Compute current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss and update network
            loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
