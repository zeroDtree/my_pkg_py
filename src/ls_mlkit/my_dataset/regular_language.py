import torch
from torch.utils.data import Dataset, DataLoader
import exrex
from torch.utils.data import random_split


class RegularLanguageDataset(Dataset):
    def __init__(self, regex_pattern, max_len=10, data_size=100, limit=100):
        self.regex_pattern = regex_pattern
        generator = exrex.generate(self.regex_pattern, limit=limit)
        self.data = list()
        for s in generator:
            if len(s) <= max_len:
                self.data.append(s)
            if len(self.data) > data_size:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        string = self.data[idx]
        return string


def get_regular_language_dataset(regex_pattern, max_len=10, data_size=100, limit=100, test_ratio=0.2, **kwargs):
    dataset = RegularLanguageDataset(regex_pattern, max_len=max_len, data_size=data_size, limit=limit)
    num_samples = len(dataset)
    train_size = int((1 - test_ratio) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )
    return train_dataset, test_dataset, test_dataset


if __name__ == '__main__':
    regex_pattern = r"a*"
    max_len = 20
    dataset_size = 100
    limit = 100
    dataset = RegularLanguageDataset(regex_pattern, max_len=max_len, data_size=dataset_size, limit=limit)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch)
