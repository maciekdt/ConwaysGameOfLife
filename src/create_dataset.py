from data_set.evolved_boards_dataset_creator import EvolvedBoardsDatasetCreator
import os

creator = EvolvedBoardsDatasetCreator(board_size=10)
dataset_path = os.path.join('datasets', 'train_dataset.pt')
creator.generate_dataset(num_samples=1000, num_evolution_steps=100, file_path=dataset_path)
