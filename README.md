# MNIST_classification_using_Quantum_Machine_Learning
This is an API used to create a machine learning classifier using quantum computing.

## Background information about convolution and quantum computations
Image classification rely mostly on the notion of convolution. In computer science, the notion of convolution can be described by a small matrix, called kernel, sliding on a bigger matrix. Depending on the main objective, the bigger matrix can be an image, time series, etc. In machine learning, convolutions are used in convolutional neural networks (CNNs for short). The kernel is made of different weights, they can be specific (like in a Gaussian kernel, in Sobel kernels and many more known and useful kernels) or they can be randomly initialized. For more info, one can visit https://en.wikipedia.org/wiki/Kernel_(image_processing)

In quantum computation, instead of working with classical information stored in *bits*, we work with *qubits*. which can be seen as the quantum equivalent. Instead of being in a single state, either 0 or 1, qubits can be in a superpostion of states of 0 and 1, i.e. qubits have a certain probability of being 0 and another certain probability of being 1. This lead to interesting properties, such as multiple simultaenous computation that can be used to *easily* crack cryptography (easy relative to a classical computer!). For more general information about quantum computing, see https://en.wikipedia.org/wiki/Quantum_computing

## Quantum convolution
To create a quantum convolution layer, we initialize a quantum circuit with parameters like the number of qubits we want or the kernel size. Then, some quantum operations are done on the kernel depending on the weights. This then gives us a certain number of outputs, depending on the number of qubits, because the value of each qubit is measured. The folowing image shows how a quantum convolution layer works with only 4 qubits<sup>1</sup>
![Quantum convolution scheme](https://pennylane.ai/qml/_images/circuit.png?raw=true "Quantum convolution layer")
<sup>1</sup> Taken from https://pennylane.ai/qml/_images/circuit.png
