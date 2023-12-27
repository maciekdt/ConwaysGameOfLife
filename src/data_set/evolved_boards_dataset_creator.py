import torch
from torch.utils.data import DataLoader
from src.model.game_of_life import GameOfLife
from src.data_set.evolved_boards_dataset import EvolvedBoardsDataset
import random

class EvolvedBoardsDatasetCreator:
    def __init__(self, board_size, device='cpu'):
        self.board_size = board_size
        self.device = device
        self.game = GameOfLife(board_size, device)
        
    def generate_dataset(self, num_samples, num_evolution_steps, file_path):
        initial_states = []
        evolved_states = []
        i = 0
        while i < num_samples:
            initial_state = self.generate_initial_state()
            self.game.set_board(initial_state)
            self.game.evaluate_board(num_evolution_steps)
            evolved_state = self.game.get_board()
            if evolved_state.sum() == 0 and random.random() < 0.9:
                continue

            initial_states.append(initial_state.cpu().unsqueeze(0))
            evolved_states.append(evolved_state.cpu())
            i+=1
            # logging
            if(i%100 == 0):
                print(f"Generated [{i}/{num_samples}]")

        torch.save(EvolvedBoardsDataset(initial_states, evolved_states), file_path)
        print(f"Dataset generated and saved to {file_path}")
        
    
    def generate_initial_state(self):
        mean_size = 4
        std_dev = 1
        while True:
            kernel_size = int(max(2, min(6, random.gauss(mean_size, std_dev))))
            board = torch.zeros(self.board_size, self.board_size, dtype=torch.float, device=self.device)

            row_start = torch.randint(0, self.board_size - kernel_size + 1, (1,)).item()
            col_start = torch.randint(0, self.board_size - kernel_size + 1, (1,)).item()

            board[row_start:row_start + kernel_size, col_start:col_start + kernel_size] = torch.randint(0, 2, (kernel_size, kernel_size), dtype=torch.float)
            if board.sum() > 0:
                break
        return board
    

    def load_dataset(self, file_path):
        return torch.load(file_path)

    def get_data_loader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 