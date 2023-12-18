from torch.utils.data import Dataset

class EvolvedBoardsDataset(Dataset):
    def __init__(self, initial_states, evolved_states):
        self.initial_states = initial_states
        self.evolved_states = evolved_states

    def __len__(self):
        return len(self.initial_states)

    def __getitem__(self, idx):
        return self.initial_states[idx], self.evolved_states[idx]