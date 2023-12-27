from data_set.evolved_boards_dataset_creator import EvolvedBoardsDatasetCreator
import os

creator = EvolvedBoardsDatasetCreator(board_size=7)
#dataset_path = os.path.join('datasets', 'train_dataset.pt')
creator.generate_dataset(num_samples=100000, num_evolution_steps=5, file_path='datasets/train_dataset.pt')
creator.generate_dataset(num_samples=10000, num_evolution_steps=5, file_path='datasets/test_dataset.pt')
creator.generate_dataset(num_samples=10000, num_evolution_steps=5, file_path='datasets/val_dataset.pt')
