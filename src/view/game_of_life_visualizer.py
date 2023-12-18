import pygame
import torch
from src.model.game_of_life import GameOfLife

class GameOfLifeVisualizer:
    def __init__(self, cell_size=50, width=500, height=500, frame_rate=10):
        pygame.init()
        self.cell_size = cell_size
        self.grid_size = width // cell_size
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.frame_rate = frame_rate
        pygame.display.set_caption("Game of Life")

    def draw_board(self, board):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if board[x, y]:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)
                else:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
        pygame.display.flip()

    def display_board(self, board: torch.Tensor):
        self.draw_board(board=board)
        self.keep_window_open()
    
    def animate_evolution(self, init_board, epochs_number):
        game = GameOfLife(self.grid_size, device='cpu')
        game.set_board(init_board)
        
        for _ in range(epochs_number):
            self.screen.fill((0, 0, 0))  # Fill the screen with black

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            self.draw_board(game.get_board().cpu().numpy())  # Draw the current board state

            game.step()  # Update the game state
            self.clock.tick(self.frame_rate)  # Control the frame rate
        self.keep_window_open()
            
    def keep_window_open(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            self.clock.tick(self.frame_rate)

