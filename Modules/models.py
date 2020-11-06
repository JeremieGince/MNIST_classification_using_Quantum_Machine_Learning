import torch
import pennylane as qml

N_QBITS = 2
quantum_device = qml.device("default.qubit", wires=N_QBITS)


class BaseModel(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.hp = hp

    def set_hp(self, **hp):
        raise NotImplementedError()

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X, y):
        raise NotImplementedError()


class HybridModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.n_layers = 6

        weight_shapes = {"weights": (self.n_layers, N_QBITS)}

        self.clayer_1 = torch.nn.Linear(2, 4)
        self.qlayer_1 = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.qlayer_2 = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.clayer_2 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, 2, dim=1)
        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_2(x)
        return self.softmax(x)

    @qml.qnode(quantum_device)
    def qnode(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(N_QBITS))
        qml.templates.BasicEntanglerLayers(weights, wires=range(N_QBITS))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QBITS)]


class ClassicalModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.clayer_1 = torch.nn.Linear(2, 4)
        self.clayer_2 = torch.nn.Linear(2, 2)
        self.clayer_3 = torch.nn.Linear(2, 2)
        self.clayer_4 = torch.nn.Linear(4, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.clayer_1(x)
        x_1, x_2 = torch.split(x, 2, dim=1)
        x_1 = self.clayer_2(x_1)
        x_2 = self.clayer_3(x_2)
        x = torch.cat([x_1, x_2], axis=1)
        x = self.clayer_4(x)
        return self.softmax(x)

