import pennylane.numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy.stats import zscore
from scipy.signal import lfilter
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn import datasets as sk_datasets
import matplotlib.pyplot as plt


class MoonDataset(Dataset):
    def __init__(self, trainingSize: float = 0.6, testSize: float = 0.2, validationSize: float = 0.2):
        # Fixing the dataset and problem
        self.X, self.y = make_moons(n_samples=200, noise=0.1)
        self.y_ = torch.unsqueeze(torch.tensor(self.y), 1)  # used for one-hot encoded labels
        self.y_hot = torch.scatter(torch.zeros((200, 2)), 1, self.y_, 1)
        if trainingSize + testSize + validationSize != 1:
            raise ValueError("The sum of sizes of the training data, test data and validation data must be 1.")
        self.Xtrain, self.Xtest, self.yTrain_hot, self.yTest_hot = train_test_split(self.X, self.y_hot,
                                                                                    test_size=testSize + validationSize,
                                                                                    shuffle=False)
        self.Xtest, self.Xval, self.yTest_hot, self.yVal_hot = train_test_split(self.Xtest, self.yTest_hot,
                                                                                test_size=testSize,
                                                                                shuffle=False)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y_hot[idx]

    def getTrainData(self):
        return self.Xtrain, self.yTrain_hot

    def getTestData(self):
        return self.Xtest, self.yTest_hot

    def getValidationData(self):
        return self.Xval, self.yVal_hot

    def show(self):
        c = ["#1f77b4" if y_ == 0 else "#ff7f0e" for y_ in self.y]  # colours for each class
        plt.axis("off")
        plt.scatter(self.X[:, 0], self.X[:, 1], c=c)
        plt.show()


class MNISTDataset(Dataset):
    def __init__(self,  trainingSize: float = 0.6, testSize: float = 0.2, validationSize: float = 0.2):
        self.digits = sk_datasets.load_digits()
        # Fixing the dataset and problem
        self.X, self.y = self.digits.images[:, np.newaxis, :, :], self.digits.target
        self.y_ = torch.unsqueeze(torch.tensor(self.y, dtype=int), 1)  # used for one-hot encoded labels
        self.y_hot = torch.scatter(torch.zeros((self.y.size, np.unique(self.y).size)), 1, self.y_, 1).numpy()

        if trainingSize + testSize + validationSize != 1:
            raise ValueError("The sum of sizes of the training data, test data and validation data must be 1.")
        self.X_train, self.X_test, self.y_train_hot, self.y_test_hot = train_test_split(
            self.X,
            self.y_hot,
            test_size=testSize + validationSize,
            shuffle=False
        )
        self.X_test, self.X_val, self.y_test_hot, self.y_val_hot = train_test_split(
            self.X_test,
            self.y_test_hot,
            test_size=testSize,
            shuffle=False
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y_hot[idx]

    def getTrainData(self):
        return self.X_train, self.y_train_hot

    def getTestData(self):
        return self.X_test, self.y_test_hot

    def getValidationData(self):
        return self.X_val, self.y_val_hot

    def show(self):
        print(self.digits["DESCR"])

        _, axes = plt.subplots(2, 5)
        images_and_labels = list(zip(self.digits.images, self.digits.target))
        for ax, (image, label) in zip(axes[0, :], images_and_labels[:5]):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Training: %i' % label)
        for ax, (image, label) in zip(axes[1, :], images_and_labels[5:10]):
            ax.set_axis_off()
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title('Training: %i' % label)

        plt.show()


if __name__ == '__main__':
    mnist = MNISTDataset()
    mnist.show()
