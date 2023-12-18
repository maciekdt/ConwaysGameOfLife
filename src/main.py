import os
import torch
from view.game_of_life_visualizer import GameOfLifeVisualizer

dataset_dir = 'datasets'
os.makedirs(dataset_dir, exist_ok=True)
dataset_path = os.path.join(dataset_dir, 'train_dataset.pt')

dataset = torch.load(dataset_path)
initial_state, evolved_state = dataset[0]

visualizer = GameOfLifeVisualizer()
#initial_board = torch.randint(0, 2, (10, 10), dtype=torch.bool)
#visualizer.animate_evolution(init_board=initial_board, epochs_number=100)
visualizer.display_board(evolved_state)