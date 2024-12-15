import pygame
import numpy as np
import torch
from go_game import GoGame
from go_reinforcement_learning import GoQLearningAgent

class GoTrainingEnvironment:
    def __init__(self, board_size=9):
        self.game = GoGame(board_size)
        self.agent = GoQLearningAgent(board_size)
        self.board_size = board_size
    
    def get_valid_moves(self):
        valid_moves = []
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.game.is_valid_move(x, y):
                    valid_moves.append(y * self.board_size + x)
        return valid_moves
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            # Reset game state
            self.game.board = np.zeros((self.board_size, self.board_size), dtype=int)
            self.game.current_player = 1
            
            done = False
            while not done:
                # Get current board state
                current_state = self.game.board.copy()
                
                # Get valid moves
                valid_moves = self.get_valid_moves()
                
                # Agent selects move
                action = self.agent.select_move(current_state, valid_moves)
                
                # Convert action to board coordinates
                x = action % self.board_size
                y = action // self.board_size
                
                # Place stone
                if self.game.place_stone(x, y):
                    # Simple reward mechanism (can be made more sophisticated)
                    reward = 1 if self.game.current_player == -1 else -1
                    
                    # Get next state
                    next_state = self.game.board.copy()
                    
                    # Check if game is over (simplified)
                    done = len(self.get_valid_moves()) == 0
                    
                    # Train agent
                    self.agent.train(current_state, action, reward, next_state, done)
                
                # Optional: visualize training (can be slow)
                self.game.draw_board()
                pygame.time.delay(100)  # Small delay for visualization
            
            # Periodic updates
            if episode % 10 == 0:
                print(f"Episode {episode} completed")
                # Optional: save model
                torch.save(self.agent.q_network.state_dict(), f'go_model_ep{episode}.pth')
        
        # Save final model
        torch.save(self.agent.q_network.state_dict(), 'final_go_model.pth')

if __name__ == "__main__":
    training_env = GoTrainingEnvironment()
    training_env.train()

