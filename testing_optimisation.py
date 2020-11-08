import numpy as np
import requests
import time
import pandas
import collections

from io import BytesIO
from http.client import HTTPConnection
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

from Modules.models import ClassicalModel
from Modules.hp_optimizers import optimize_parameters, GPOParamGen
from Modules.datasets import MNISTDataset


if __name__ == '__main__':
    mnist_dataset = MNISTDataset()

    bounds_params = {
        "nb_hidden_neurons": [1, 30, 1]
    }

    gpo = GPOParamGen(bounds_params, max_itr=30)
    hp = optimize_parameters(ClassicalModel, *mnist_dataset.getTrainData(), gpo, fit_kwargs={"epochs": 20})
    print(f"\n predicted hp: {hp} \n")
    gpo.show_expectation()

    c_model = ClassicalModel(**hp)
    print(c_model)
    history = c_model.fit(
        *mnist_dataset.getTrainData(),
        *mnist_dataset.getValidationData(),
        batch_size=32,
        verbose=True
    )
    print(c_model.score(*mnist_dataset.getTestData()))
    c_model.show_history(history)

