import pennylane as qml
from pennylane import numpy as np
import torch
from pennylane.templates import RandomLayers


class QuantumConvolutionLayer(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.kernel_size = hp.get("kernel_size", (2, 2))
        self.nb_qubits = hp.get("nb_qubits", 2)
        dev = qml.device("default.qubit", wires=self.nb_qubits)

        # Random circuit parameters
        np.random.seed(0)
        rand_params = np.random.uniform(high=2 * np.pi, size=self.kernel_size)
        self.weights = torch.tensor(rand_params).float()

        self.get_circuit = lambda phi, layer_param: self.circuit(phi, layer_param)
        self.qnode = qml.QNode(self.get_circuit, dev, interface='torch')

    def circuit(self, phi=None, layer_params=None):
        # Encoding of 4 classical input values
        for j in range(self.nb_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(layer_params, wires=list(range(self.nb_qubits)))

        # Measurement producing n classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(self.nb_qubits)]

    def forward(self, x):
        print(x.shape)
        x = torch.reshape(x, (*x.shape[1:], x.shape[0]))
        print(x.shape)
        out = np.zeros((x.shape[0]//self.kernel_size[0], x.shape[1]//self.kernel_size[1], self.nb_qubits, x.shape[-1]))

        for i in range(x.shape[-1]):
            # Loop over the coordinates of the top-left pixel of 2X2 squares
            # for j in range(0, x.shape[0], self.kernel_size[0]):
            #     for k in range(0, x.shape[1], self.kernel_size[1]):
            for j in range(0, x.shape[0]-1):
                for k in range(0, x.shape[1]-1):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = self.qnode(
                        # phi=[
                        #     x[j+jj, k+kk, i]
                        #     for jj in range(self.kernel_size[0])
                        #     for kk in range(self.kernel_size[1])
                        # ],
                        x[j:j+self.kernel_size[0], k:k+self.kernel_size[1]].flatten(),
                        self.weights
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(self.nb_qubits):
                        out[j // 2, k // 2, c, i] = q_results[c]
        out = torch.reshape(torch.Tensor(out).float(), (out.shape[-1], *out.shape[:-1]))
        print(out.shape)
        return out


if __name__ == '__main__':
    import time

    s = time.time()
    q_conv_layer = QuantumConvolutionLayer()
    q_conv_layer.cuda()
    inputs = torch.Tensor(np.ones((32, 8, 8))).float().cuda()
    outputs = q_conv_layer(inputs)

    print(f"elapse time: {time.time()-s:.2f}")
