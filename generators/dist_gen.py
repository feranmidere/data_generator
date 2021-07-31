import numpy as np
import pandas as pd
from sklearn import utils
import joblib as jb
from . import dist_base


class SimpleClassificationSynthesiser(dist_base.ClassificationSynthesiser):
    def __init__(self, n_jobs=1, discrete_columns=[]):
        self.discrete_columns = discrete_columns
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if hasattr(X, 'columns'):
            self.columns = list(X.columns)
        else:
            self.columns = list(range(X.shape[1]))
        self.discrete_columns_indices = list(
            map(self.columns.index, self.discrete_columns))
        self.numerical_columns = [
            col for col in self.columns if col not in self.discrete_columns]
        self.numerical_columns_indices = list(
            map(self.columns.index, self.numerical_columns))
        values, counts = np.unique(y, return_counts=True)
        self.class_v_percs_ = np.array([values, counts / len(y)]).transpose()
        self.num_columns_ = X.shape[1]
        df = pd.DataFrame(X).join(pd.Series(y, name='target'))
        with jb.Parallel(n_jobs=self.n_jobs) as parallel:
            dist_list = parallel(jb.delayed(self.model_dist_by_class)(
                df, c, self.numerical_columns_indices) for c in np.unique(y))
        self.dist_sep_class_ = {
            c: dist for c, dist in zip(
                np.unique(y), dist_list)}
        return self

    def sample(self, n=100):
        utils.check_scalar(n, name='n', min_val=10, target_type=int)
        numbers = np.floor(self.class_v_percs_[:, 1] * n)
        if sum(numbers) != n:
            numbers[np.argmax(numbers)] += 1
        assert sum(numbers) == n, str(sum(numbers)) + ', ' + str(n)
        class_v_nums = np.array(
            [self.class_v_percs_[:, 0], numbers]).transpose()
        with jb.Parallel(n_jobs=self.n_jobs) as parallel:
            label_samples = parallel(
                jb.delayed(self.generate_sample_class)(
                    class_v_nums,
                    i,
                    self.dist_sep_class_,
                    self.n_jobs) for i in range(
                    class_v_nums.shape[0]))
        full = pd.DataFrame(np.concatenate(label_samples, axis=0))
        full = full.sample(frac=1)
        sample, target = full.drop(
            full.shape[1] - 1, axis=1), full[full.shape[1] - 1]
        return sample.values, target.values
