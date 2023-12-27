import torch

#For GPU suport set device='cuda'
class GameOfLife:
    def __init__(self, size, device='cpu'):
        self.size = size
        self.device = device
        self.board = torch.zeros(size, size, dtype=torch.float).to(device)

    def step(self):
        padded_board = torch.nn.functional.pad(self.board, (1, 1, 1, 1), mode='constant', value=0)
        
        kernel_matrix = [[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]]
        
        kernel = torch.tensor(kernel_matrix, dtype=torch.float).to(self.device)
        
        neighbors = torch.nn.functional.conv2d(padded_board.unsqueeze(0).unsqueeze(0).float(),
                                               kernel.unsqueeze(0).unsqueeze(0),
                                               padding=0).squeeze(0).squeeze(0).int()

        self.board = ((self.board == 1) & ((neighbors == 2) | (neighbors == 3))) | ((self.board == 0) & (neighbors == 3))
        self.board = self.board.float()

    def set_board(self, new_board):
        if new_board.shape == (self.size, self.size):
            self.board = new_board
        else:
            raise ValueError("New board must be of the same size as the current board")

    def get_board(self):
        return self.board
    
    def evaluate_board(self, epochs_number):
        for _ in range(epochs_number):
            self.step()
        

