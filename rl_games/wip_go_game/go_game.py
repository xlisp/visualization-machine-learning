import pygame
import numpy as np
import sys

class GoGame:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.game_over = False
        
        # Initialize Pygame
        pygame.init()
        self.cell_size = 50
        self.margin = 50
        self.width = self.height = self.board_size * self.cell_size + 2 * self.margin
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Go Game')
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BOARD_COLOR = (210, 180, 140)
        
    def draw_board(self):
        # Clear screen
        self.screen.fill(self.BOARD_COLOR)
        
        # Draw grid lines
        for x in range(self.board_size + 1):
            pygame.draw.line(self.screen, self.BLACK, 
                             (self.margin + x * self.cell_size, self.margin),
                             (self.margin + x * self.cell_size, self.height - self.margin))
            pygame.draw.line(self.screen, self.BLACK, 
                             (self.margin, self.margin + x * self.cell_size),
                             (self.width - self.margin, self.margin + x * self.cell_size))
        
        # Draw stones
        for y in range(self.board_size):
            for x in range(self.board_size):
                if self.board[y][x] != 0:
                    stone_color = self.BLACK if self.board[y][x] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, stone_color, 
                                       (self.margin + x * self.cell_size, 
                                        self.margin + y * self.cell_size), 
                                       self.cell_size // 2 - 2)
        
        pygame.display.flip()
    
    def get_board_coords(self, mouse_pos):
        x, y = mouse_pos
        board_x = (x - self.margin) // self.cell_size
        board_y = (y - self.margin) // self.cell_size
        return board_x, board_y
    
    def is_valid_move(self, x, y):
        # Check if the position is within board and empty
        if (0 <= x < self.board_size and 0 <= y < self.board_size and 
            self.board[y][x] == 0):
            return True
        return False
    
    def place_stone(self, x, y):
        if self.is_valid_move(x, y):
            self.board[y][x] = self.current_player
            self.current_player *= -1  # Switch players
            return True
        return False
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get mouse position and convert to board coordinates
                    mouse_pos = pygame.mouse.get_pos()
                    board_x, board_y = self.get_board_coords(mouse_pos)
                    
                    # Try to place a stone
                    if self.place_stone(board_x, board_y):
                        self.draw_board()
            
            self.draw_board()
        
        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = GoGame()
    game.run()
