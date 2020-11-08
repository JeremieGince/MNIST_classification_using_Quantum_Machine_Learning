import pennylane.numpy as np
from sklearn.model_selection import KFold
from Modules.models import BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import tqdm
import warnings
import matplotlib.pyplot as plt


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

    def get_trial_param(self):
        raise NotImplementedError()

    def get_best_param(self):
        return max(self.history, key=lambda t: t[-1])

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

    def get_trial_param(self):
        param = self.params[self.idx]
        self.idx += 1
        return {self.bounds_names[i]: param[i] for i in range(len(param))}


class RandomParamGen(ParamGen):
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False):
        super(RandomParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)

    def __len__(self):
        pass

    def get_trial_param(self):
        pass


class GPOParamGen(ParamGen):
    def __init__(self, bounds_dict, default_interval=1, bounds_as_list_of_possible_values=False, **kwargs):
        super(GPOParamGen, self).__init__(bounds_dict, default_interval, bounds_as_list_of_possible_values)
        if bounds_as_list_of_possible_values:
            self.xx = np.meshgrid(*[bounds_dict[p] for p in self.bounds_names])
        else:
            self.xx = np.meshgrid(*[
                np.arange(
                    bounds_dict[p][0],
                    bounds_dict[p][1] + (bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval),
                    bounds_dict[p][2] if len(bounds_dict[p]) > 2 else default_interval
                ) for p in self.bounds_names
            ])
        self.xx = np.array(list(zip(*[_x.ravel() for _x in self.xx])))

        self.xi = kwargs.get("xi", 0.1)
        self.Lambda = kwargs.get("lambda", 1.0)
        self.bandwidth = kwargs.get("bandwidth", 1.0)

        self.max_itr = kwargs.get("max_itr", 30)
        self.X, self.y = [], []
        self.gpr = GaussianProcessRegressor(RBF(length_scale=self.bandwidth), alpha=self.Lambda, optimizer=None)

    def __len__(self):
        return self.max_itr

    def get_trial_param(self):
        if len(self.X) > 0:
            eis = self.expected_improvement()
            idx = np.argmax(eis)
        else:
            idx = np.random.randint(self.xx.shape[0])

        t_param = self.xx[idx]
        return {self.bounds_names[i]: t_param[i] for i in range(len(t_param))}

    def get_best_param(self):
        f_hat = self.gpr.predict(self.xx)
        b_param = self.xx[np.argmax(f_hat)]
        return {self.bounds_names[i]: b_param[i] for i in range(len(b_param))}

    def add_score_info(self, param, score):
        super().add_score_info(param, score)

        self.X.append([param[p] for p in self.bounds_names])
        self.y.append(score)
        self.gpr.fit(np.array(self.X), np.array(self.y))

    def expected_improvement(self):
        f_hat = self.gpr.predict(np.array(self.X))
        best_f = np.max(f_hat)

        f_hat, std_hat = self.gpr.predict(self.xx, return_std=True)
        improvement = f_hat - best_f - self.xi

        Z = improvement / std_hat
        ei = improvement * norm.cdf(Z) + std_hat * norm.pdf(Z)
        return ei

    def show_expectation(self):
        _xx = np.squeeze(self.xx)
        if len(_xx.shape) > 1:
            raise NotImplementedError("Show of higher than one dimensional of hp is not implemented yet.")
        f_hat, std_hat = self.gpr.predict(self.xx, return_std=True)

        plt.figure(1)
        plt.plot(_xx, f_hat)
        plt.plot(np.squeeze(np.array(self.X)), np.squeeze(np.array(self.y)), 'x')
        plt.fill_between(_xx, f_hat, f_hat + std_hat, alpha=0.4)
        plt.fill_between(_xx, f_hat, f_hat - std_hat, alpha=0.4)
        plt.xlabel("hp space")
        plt.ylabel("Expected outcome (f)")
        plt.title("EI")
        plt.show()


def optimize_parameters(model_cls, X, y, param_gen: ParamGen, n_splits: int = 2, fit_kwargs={}):
    warnings.simplefilter("ignore", UserWarning)

    param_gen.reset()
    progress = tqdm.tqdm(range(len(param_gen)), unit='itr', postfix="optimisation")
    for i in progress:
        params = param_gen.get_trial_param()

        kf = KFold(n_splits=n_splits, shuffle=False)

        mean_score = 0
        for train_index, test_index in kf.split(X):
            sub_X_train, sub_X_test = X[train_index], X[test_index]
            sub_y_train, sub_y_test = y[train_index], y[test_index]

            clf: BaseModel = model_cls(**params)
            clf.fit(sub_X_train, sub_y_train, **fit_kwargs)

            score = clf.score(sub_X_test, sub_y_test)
            mean_score = (i * mean_score + score) / (i + 1)

        progress.set_postfix_str(f"mean_score: {mean_score:.2f}")
        param_gen.add_score_info(params, mean_score)
        progress.update()

    progress.close()
    return param_gen.get_best_param()

