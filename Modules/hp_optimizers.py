import pennylane.numpy as np
from sklearn.model_selection import KFold


class ParamGen:
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False):
        self.bounds_names = list(bounds_dict.keys())
        self.bounds_dict = bounds_dict
        self.bounds_as_list_of_possible_values = bounds_as_list_of_possible_values
        self.default_interval = default_interval
        self.history = []

    def reset(self):
        return

    def __len__(self):
        raise NotImplementedError()

    def get_param(self):
        raise NotImplementedError()

    def add_score_info(self, param, score):
        self.history.append((param, score))


class GridParamGen(ParamGen):
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False):
        super(GridParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)
        if bounds_as_list_of_possible_values:
            xx = np.meshgrid(*[bounds_dict[p] for p in self.bounds_names])
        else:
            xx = np.meshgrid(*[
                np.arange(
                    bounds_dict[p][0],
                    bounds_dict[p][1] + (bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval),
                    bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval
                ) for p in self.bounds_names
            ])
        self.params = list(zip(*[_xx.ravel() for _xx in xx]))
        self.idx = 0

    def reset(self):
        self.idx = 0

    def __len__(self):
        return len(self.params)

    def get_param(self):
        param = self.params[self.idx]
        self.idx += 1
        return {self.bounds_names[i]: param[i] for i in range(len(param))}


class RandomParamGen(ParamGen):
    def __len__(self):
        pass

    def get_param(self):
        pass


class GPOParamGen(ParamGen):
    def __len__(self):
        pass

    def get_param(self):
        pass


def optimize_parameters(model_cls, X, y, param_gen: ParamGen, n_splits: int = 2):
    best_parameters = None
    best_cross_val_score = -1

    param_gen.reset()
    for i in range(len(param_gen)):
        params = param_gen.get_param()
        clf = model_cls(**params)
        kf = KFold(n_splits=n_splits, shuffle=False)

        mean_score = 0
        n = 1
        for train_index, test_index in kf.split(X):
            sub_X_train, sub_X_test = X[train_index], X[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]

            clf.fit(sub_X_train, sub_y_train)

            score = np.mean(clf.predict(sub_X_test) == sub_y_test)
            mean_score = (n * mean_score + score) / (n + 1)
            n += 1

        param_gen.add_score_info(params, mean_score)

        if mean_score > best_cross_val_score:
            best_parameters = params
            best_cross_val_score = mean_score

    return best_parameters, best_cross_val_score
