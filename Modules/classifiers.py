import torch
import torch.optim as optim
from torch.autograd import Variable
import pennylane as qml
import pennylane.numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import warnings

from Modules.layers import QuantumPseudoLinearLayer


class QuantumClassifier(torch.nn.Module):
    def __init__(self, input_shape, output_shape, **hp):
        super(QuantumClassifier, self).__init__()
        self.hp = hp
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_qubits = hp.get("nb_qubits", 2)
        self.seq_0 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.input_shape), self.nb_qubits),
            *[QuantumPseudoLinearLayer(nb_qubits=self.nb_qubits, **hp)
              for _ in range(hp.get("nb_q_layer", 2))],
        )
        self.seq_1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.get_output_shape_seq_0()), self.output_shape),
            torch.nn.Softmax(),
        )
    def get_output_shape_seq_0(self):
        ones = torch.Tensor(np.ones((1, *self.input_shape))).float()
        for param in self.parameters():
            print(param.device)
        return self.seq_0(ones).shape

    def forward(self, x):
        return self.seq_1(self.seq_0(x))


class ClassicalClassifier(torch.nn.Module):
    def __init__(self, input_shape, output_shape, **hp):
        super(ClassicalClassifier, self).__init__()
        # print(input_shape)
        self.hp = hp
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden_neurons = hp.get("nb_hidden_neurons", 1_000)
        self.linear_block = torch.nn.Sequential(
            torch.nn.Linear(self.nb_hidden_neurons, self.nb_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )
        self.seq = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(self.input_shape), self.nb_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            *[self.linear_block for _ in range(hp.get("nb_hidden_layer", 1))],
            torch.nn.Linear(self.nb_hidden_neurons, self.output_shape),
            torch.nn.Softmax(),
        )

    def forward(self, x):
        # x = torch.Tensor(x).float()
        return self.seq(x)
