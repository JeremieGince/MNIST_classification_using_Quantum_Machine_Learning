import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pennylane as qml
import pennylane.numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

N_QBITS = 2
quantum_device = qml.device("default.qubit", wires=N_QBITS)


class BaseModel(torch.nn.Module):
    def __init__(self, **hp):
        super().__init__()
        self.hp = hp

        self.default_use_cuda = torch.cuda.is_available()

    def set_hp(self, **hp):
        raise NotImplementedError()

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        optimizer_parameters = kwargs.get("optimizer_parameters",
                                          {
                                              "lr": 0.01,
                                              "momentum": 0.9,
                                              "nesterov": True,
                                              "weight_decay": 10 ** -6
                                          })
        kwargs["optimizer_parameters"] = optimizer_parameters
        optimizer = kwargs.get("optimizer",
                               optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), **optimizer_parameters)
                               )
        kwargs["optimizer"] = optimizer

        criterion = kwargs.get("criterion", nn.MSELoss())
        kwargs["criterion"] = criterion

        batch_size = kwargs.get("batch_size", 32)

        train_data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True, drop_last=True
        )

        if X_val is not None and y_val is not None:
            val_data_loader = torch.utils.data.DataLoader(
                list(zip(X_val, y_val)), batch_size=batch_size, shuffle=True, drop_last=True
            )
        else:
            val_data_loader = None

        history = {
            'epochs': [],
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        if kwargs.get("use_cuda", self.default_use_cuda):
            self.cuda()

        epochs_id = range(kwargs.get("epochs", 100))
        progress = tqdm.tqdm(
            epochs_id,
            unit="epoch"
        )
        kwargs["progress"] = progress
        for epoch in epochs_id:
            history["epochs"].append(epoch)

            train_loss, train_acc = self.do_epoch(train_data_loader, **kwargs)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if val_data_loader is not None:
                val_loss, val_acc = self.do_epoch(val_data_loader, backprop=False, **kwargs)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            progress.update()
            progress.set_postfix_str(' '.join([str(_n)+': '+str(_v[-1] if _v else None) for _n, _v in history.items()]))

        progress.close()
        return history

    def do_epoch(self, data_loader, **kwargs):
        if kwargs.get("scheduler", False):
            kwargs["scheduler"].step()

        epoch_mean_loss = 0
        epoch_mean_acc = 0
        for j, (inputs, targets) in enumerate(data_loader):
            if kwargs.get("use_cuda", self.default_use_cuda):
                inputs = inputs.cuda()
                targets = targets.cuda()

            # inputs = Variable(inputs)
            # targets = Variable(targets)

            n = j + 1
            batch_loss = self.do_batch(inputs, targets, **kwargs)
            epoch_mean_loss = (n * epoch_mean_loss + batch_loss) / (n + 1)
            batch_acc = self.score(inputs, targets)
            epoch_mean_acc = (n * epoch_mean_acc + batch_acc) / (n + 1)

        return epoch_mean_acc, epoch_mean_loss

    def do_batch(self, inputs, targets, **kwargs):
        # Use model.zero_grad() instead of optimizer.zero_grad()
        # Otherwise, variables that are not optimized won't be cleared
        self.zero_grad()
        output = self(inputs)

        loss = kwargs["criterion"](output, targets)

        if kwargs.get("backprop", True):
            loss.backward()
            kwargs["optimizer"].step()

        return loss

    def predict(self, X):
        y_pred = self(X)
        return torch.argmax(y_pred, axis=1).detach().numpy()

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    @staticmethod
    def show_history(history: dict, **kwargs):
        epochs = history['epochs']

        fig, axes = plt.subplots(2, 1)

        axes[0].set_title('Train accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, history['train_acc'], label='Train')
        axes[0].plot(epochs, history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_title('Train loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, history['train_loss'], label='Train')
        axes[1].plot(epochs, history['val_loss'], label='Validation')
        plt.tight_layout()
        if kwargs.get("saving", True):
            os.makedirs(f"figures", exist_ok=True)
            plt.savefig(f"figures/training_history_{kwargs.get('name', '')}.png", dpi=300)
        plt.show()


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
        x = self.clayer_1(x.float())
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


if __name__ == '__main__':
    from Modules.datasets import MoonDataset
    c_model = ClassicalModel()

    moon_dataset = MoonDataset()
    c_model.fit(moon_dataset.X, moon_dataset.y_hot, batch_size=5)
    print(c_model.score(moon_dataset.X, moon_dataset.y_hot))
