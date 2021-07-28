import numpy as np
import pandas as pd
from sklearn import base, utils
import scipy.stats as sps


class CLFColumnSampler:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = base.check_X_y(X, y, copy=True, ensure_min_samples=100)
        values, counts = np.unique(y, return_counts=True)
        self.class_v_percs_ = np.array([values, counts / len(y)]).transpose()
        self.dist_sep_class_ = {}
        self.num_columns_ = X.shape[1]
        df = pd.DataFrame(X).join(pd.Series(y, name='target'))
        for c in np.unique(y):
            X_c = df[df['target'] == c].drop('target', axis=1).values
            hist = {index: np.histogram(X_c[:, index], density=True) for index in range(X_c.shape[1])}
            dist = {index: sps.rv_histogram(hist[index]) for index in hist.keys()}
            self.dist_sep_class_[c] = dist
        return self

    def sample(self, n=100):
        utils.check_scalar(n, name='n', min_val=10, target_type=int)
        sample = pd.DataFrame()
        target = pd.Series(name='target')
        numbers = np.ceil(self.class_v_percs_[:, 1] * n)
        if sum(numbers) != n:
            numbers[np.argmax(numbers)] -= 1
        class_v_nums = np.array([self.class_v_percs_[:, 0], numbers]).transpose()
        for i in range(class_v_nums.shape[0]):
            label_sample = pd.DataFrame()
            label, number = class_v_nums[i]
            dists = self.dist_sep_class_[label]
            for col, dist in dists.items():
                label_sample = pd.concat([label_sample, pd.Series(dist.rvs(size=int(number)), name=str(col))], axis=1)
            target_sample = pd.Series(np.full(shape=int(number), fill_value=label), name='target')
            sample = sample.append(label_sample, ignore_index=True)
            target = target.append(target_sample, ignore_index=True)
        full = pd.concat([sample, target], axis=1)
        full = full.sample(frac=1)
        sample, target = full.drop('target', axis=1), full.target
        return sample.values, target.values
