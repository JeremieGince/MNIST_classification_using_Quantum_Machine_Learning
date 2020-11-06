import torch


class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set_hp(self, **kwargs):
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()