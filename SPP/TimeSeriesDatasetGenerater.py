import torch
from torch.utils.data import Dataset

class TimeSeriesDatasetGenerater(Dataset):
    def __init__(self, data, targets, sequence_length, sequence_stride=1, sampling_rate=1, batch_size=128, shuffle=False, seed=None, start_index=None, end_index=None):
        super().__init__()

        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.start_index = start_index
        self.end_index = end_index

        self.indices = []

        for i in range(self.start_index, self.end_index, self.sampling_rate):
            for j in range(0, len(self.data) - self.sequence_length + 1, self.sequence_stride):
                self.indices.append((i, j))

        if self.shuffle:
            torch.manual_seed(self.seed)
            torch.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]

        inputs = self.data[i:i + self.sequence_length]
        targets = self.targets[i + self.sequence_length]

        return inputs, targets.unsqueeze(dim=-1)
# Create a TimeSeriesDataset object
#dataset = TimeSeriesDataset(data, targets, sequence_length, sequence_stride, sampling_rate, batch_size, shuffle, seed, start_index, end_index)

# Create a PyTorch DataLoader object
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
