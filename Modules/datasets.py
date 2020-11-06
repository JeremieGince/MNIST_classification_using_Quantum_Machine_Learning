import pennylane.numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy.stats import zscore
from scipy.signal import lfilter
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class MoonDataset(Dataset):
    def __init__(self):
        # Fixing the dataset and problem
        self.X, self.y = make_moons(n_samples=200, noise=0.1)
        self.y_ = torch.unsqueeze(torch.tensor(self.y), 1)  # used for one-hot encoded labels
        self.y_hot = torch.scatter(torch.zeros((200, 2)), 1, self.y_, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y_hot[idx]

    def show(self):
        c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in self.y]  # colours for each class
        plt.axis("off")
        plt.scatter(self.X[:, 0], self.X[:, 1], c=c)
        plt.show()