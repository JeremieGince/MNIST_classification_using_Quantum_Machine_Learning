import torch
from torch.nn import Conv2d
from Modules.layers import QuantumConvolutionLayer


class QuantumBackbone(torch.nn.Module):
    def __init__(self, input_shape, output_shape, **hp):
        super().__init__()
        self.hp = hp

        self.seq = torch.nn.Sequential(
            *[QuantumConvolutionLayer(kernel_size=self.hp.get("kernel_size", (2, 2)),
                                      nb_qubits=self.hp.get("nb_qubits", 2))
              for _ in self.hp.get("nb_q_conv_layer", 3)]
        )

    def forward(self, x):
        return self.seq(x)


class ClassicalBackbone(torch.nn.Module):
    def __init__(self, input_shape, output_shape, **hp):
        super().__init__()
        self.hp = hp

        self.conv_block = lambda oi, oc: torch.nn.Sequential(
            Conv2d(oi, oc, kernel_size=self.hp.get("kernel_size", 2), stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.seq = torch.nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block(32, 64),
            *[self.conv_block(64, 64) for _ in range(self.hp.get("nb_q_conv_layer", 0))],
            self.conv_block(64, self.hp.get("out_channels", 2))
        )

    def forward(self, x):
        return self.seq(x)
