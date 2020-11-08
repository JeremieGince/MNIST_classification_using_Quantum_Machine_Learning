import pennylane as qml
from pennylane import numpy as np
import torch
from pennylane.templates import RandomLayers

np.random.seed(0)
class Quantum_convolution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dev = qml.device("default.qubit", wires=4)
        # Random circuit parameters
        rand_params = np.random.uniform(high=2 * np.pi, size=(1, 4))
        self.rand_params = torch.tensor(rand_params)
        self.qnode = qml.QNode(self.circuit,dev,interface='torch')


    @staticmethod
    def circuit(phi=None,random_layer_param = None):
        # Encoding of 4 classical input values
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(random_layer_param, wires=list(range(4)))

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    def forward(self,x):
        out = np.zeros((14, 14, 4))

        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = self.qnode(
                    phi=[
                        x[j, k, 0],
                        x[j, k + 1, 0],
                        x[j + 1, k, 0],
                        x[j + 1, k + 1, 0]
                    ],
                    random_layer_param = self.rand_params

                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(4):
                    out[j // 2, k // 2, c] = q_results[c]
        return out
# wires = 4
# # set the random seed
# np.random.seed(42)
#
# dev = qml.device("default.qubit", wires=wires)
#
# @qml.qnode(dev)
# def circuit(params):
