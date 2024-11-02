from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import random_split


def transform_label_to_integer(df, label_col_index):
    labels = df.iloc[:, label_col_index]
    label_set = list()
    for e in labels:
        if e not in label_set:
            label_set.append(e)
    label_to_integer = dict()
    i = 0
    for e in label_set:
        label_to_integer[e] = i
        i += 1
    for i in range(0, len(labels)):
        df.iloc[i, label_col_index] = label_to_integer[df.iloc[i, label_col_index]]


def transform_dataframe_to_tensor(df, shuffle=True) -> torch.Tensor:
    ar = np.array(df).astype(float)
    x = torch.from_numpy(ar)
    x = x.to(torch.float)
    if shuffle:
        random_indices = torch.randperm(
            len(x), generator=torch.Generator().manual_seed(0)
        )
        x = x[random_indices]
    return x


class IrisDataset(Dataset):
    def __init__(self, path="./data/Iris.csv", label_col_index=5):
        super().__init__()
        df = pd.read_csv(path, header=None, index_col=None)
        df = df.drop(index=0).reset_index(drop=True)
        transform_label_to_integer(df, label_col_index)
        x = transform_dataframe_to_tensor(df)
        self.x = x[:, 1:-1]
        self.y = x[:, -1].to(torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def get_iris_dataset(test_ratio=0.2, **kwargs):

    dataset = IrisDataset()
    num_samples = len(dataset)
    train_size = int((1 - test_ratio) * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)
    )
    print(
        f"total_size = {len(dataset)}, (train_size, test_size) = {len(train_dataset)}, {len(test_dataset)}"
    )
    return train_dataset, test_dataset, test_dataset


if __name__ == "__main__":
    pass
