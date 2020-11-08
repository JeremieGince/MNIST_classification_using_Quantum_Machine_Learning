import pennylane as qml
from pennylane import numpy as np
import torch
from pennylane.templates import RandomLayers
import time

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

    def circuit(self, phi=None, layer_params=None):
        # Encoding of 4 classical input values
        for j in range(self.nb_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(layer_params, wires=list(range(self.nb_qubits)))

        # Measurement producing n classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(self.nb_qubits)]

    def forward(self, x):
        # out = torch.Tensor(np.array(list(map(self.convolve, x)))).float()
        start = time.time()
        # print("batch start")
        out = torch.zeros((x.shape[0],x.shape[1]*self.nb_qubits  ,x.shape[2] // self.kernel_size[0], x.shape[3] // self.kernel_size[1])).to(x.device)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[i] = self.convolve(x[i,j])
        # print(f"Elapse time :  {time.time() - start}")
        return out

    def convolve(self, x):
        # x = torch.squeeze(x)
        out = torch.zeros((self.nb_qubits,x.shape[0] // self.kernel_size[0], x.shape[1] // self.kernel_size[1])).to(x.device)
        for j in range(0, x.shape[0] - 1):
            for k in range(0, x.shape[1] - 1):
                # Process a squared 2x2 region of the image with a quantum circuit
                q_results = self.qnode(
                    x[j:j + self.kernel_size[0], k:k + self.kernel_size[1]].flatten(),
                    self.weights
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(self.nb_qubits):
                    out[c,j // self.kernel_size[0], k // self.kernel_size[1]] = q_results[c]
        return out


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
    q_conv_layer = QuantumConvolutionLayer(kernel_size=(2, 2), nb_qubits=1)
    inputs = torch.Tensor(np.ones((32, 8, 8))).float()
    outputs = q_conv_layer(inputs)
    print(outputs.shape)

    print(f"elapse time: {time.time()-s:.2f}")