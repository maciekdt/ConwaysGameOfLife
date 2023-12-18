import torch
from torch.utils.data import DataLoader
from src.model.game_of_life import GameOfLife
from src.data_set.evolved_boards_dataset import EvolvedBoardsDataset

class EvolvedBoardsDatasetCreator:
    def __init__(self, board_size, device='cpu'):
        self.board_size = board_size
        self.device = device
        self.game = GameOfLife(board_size, device)
        
    def generate_dataset(self, num_samples, num_evolution_steps, file_path):
        initial_states = []
        evolved_states = []

        for i in range(num_samples):
            initial_state = torch.randint(0, 2, (self.board_size, self.board_size), dtype=torch.int, device=self.device)
            self.game.set_board(initial_state)
            self.game.evaluate_board(num_evolution_steps)
            evolved_state = self.game.get_board()

            initial_states.append(initial_state.cpu())
            evolved_states.append(evolved_state.cpu())
            if(i%100 == 0 & i>1):
                print(f"Generated [{i}/{num_samples}]")

        torch.save(EvolvedBoardsDataset(initial_states, evolved_states), file_path)
        print(f"Dataset generated and saved to {file_path}")
    

    def load_dataset(self, file_path):
        return torch.load(file_path)

    def get_data_loader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 