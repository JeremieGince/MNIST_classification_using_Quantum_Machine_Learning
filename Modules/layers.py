import pennylane as qml
from pennylane import numpy as np
import torch
from pennylane.templates import RandomLayers
import time
from scipy.ndimage.filters import generic_filter


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

        get_circuit = lambda phi, layer_param: self.circuit(phi, layer_param)
        self.qnode = qml.QNode(get_circuit, dev, interface='torch')

        self.output_shape = None

    def circuit(self, phi=None, layer_params=None):
        # Encoding of 4 classical input values
        for j in range(self.nb_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(layer_params, wires=list(range(self.nb_qubits)))

        # Measurement producing n classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(self.nb_qubits)]

    def forward(self, x):
        if self.output_shape is None:
            self.output_shape = (
                x.shape[0],
                x.shape[1] * self.nb_qubits,
                x.shape[2] // self.kernel_size[0],
                x.shape[3] // self.kernel_size[1]
            )
        print((*[1 for _ in range(len(x.shape) - len(self.kernel_size))], *self.kernel_size))
        out1 = generic_filter(x, self.q_filter,
                              size=(*[1 for _ in range(len(x.shape) - len(self.kernel_size))], *self.kernel_size))
        print(f"out1.shape: {out1.shape}")

        # out = torch.Tensor(np.array(list(map(self.convolve, x)))).float()
        output_shape = (x.shape[0], x.shape[1]*self.nb_qubits, x.shape[2] // self.kernel_size[0], x.shape[3] // self.kernel_size[1])
        out = torch.zeros(output_shape).to(x.device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i] = self.convolve(x[i, j])

        x_flat = self.flatten_x(x)
        return out

    def flatten_x(self, x):
        out = torch.zeros(self.output_shape).to(x.device)
        out[:x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
        return out.flatten()

    def unflatten_x(self, x):
        pass

    def flatten_q_output(self, q_out):
        out = torch.zeros(self.output_shape).to(x.device)
        out[0, 0:2] = q_out
        return out.flatten()

    def convolve(self, x):
        # x = torch.squeeze(x)
        out = torch.zeros((self.nb_qubits, x.shape[0] // self.kernel_size[0], x.shape[1] // self.kernel_size[1])).to(x.device)
        for j in range(0, x.shape[0] - 1):
            for k in range(0, x.shape[1] - 1):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = self.qnode(
                    x[j:j + self.kernel_size[0], k:k + self.kernel_size[1]].flatten(),
                    self.weights
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(self.nb_qubits):
                    out[c, j // self.kernel_size[0], k // self.kernel_size[1]] = q_results[c]
        return out

    def q_filter(self, phi):
        return self.qnode(phi, self.weights)


class QuantumPseudoLinearLayer(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.nb_qubits = hp.get("nb_qubits", 2)
        self.n_layers = hp.get("n_layers", 6)
        dev = qml.device("default.qubit", wires=self.nb_qubits)
        get_circuit = lambda inputs, weights: self.circuit(inputs, weights)
        self.qnode = qml.QNode(get_circuit, dev, interface='torch')
        weight_shapes = {"weights": (self.n_layers, self.nb_qubits)}
        self.seq = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.nb_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(self.nb_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.nb_qubits)]

    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    import time

    s = time.time()
    q_conv_layer = QuantumConvolutionLayer(kernel_size=(2, 2), nb_qubits=2)
    inputs = torch.Tensor(np.ones((32, 1, 8, 8))).float()
    outputs = q_conv_layer(inputs)
    print(outputs.shape)

    print(f"elapse time: {time.time()-s:.2f}")