import numpy as np
import torch
from torch.utils.data import Dataset


def extract_plants(PATH, IMG_SIZE=128):
    train_data = np.load(PATH)
    x_train = np.array([i[0] for i in train_data]).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    y_train = np.array([i[1] for i in train_data])
    return x_train, y_train


class PlantsData(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.FloatTensor(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
