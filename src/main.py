from view.game_of_life_visualizer import GameOfLifeVisualizer
import torch

visualizer = GameOfLifeVisualizer()
initial_board = torch.randint(0, 2, (10, 10), dtype=torch.bool)
#visualizer.animate_evolution(init_board=initial_board, epochs_number=100)
visualizer.display_board(initial_board)