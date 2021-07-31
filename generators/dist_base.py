import numpy as np
import scipy.stats as sps
import joblib as jb#


class BaseSynthesiser:
    def __init__(self, n_jobs=1, discrete_columns=None):
        self.discrete_columns = discrete_columns
        self.n_jobs = n_jobs

    def gen_col_sample(self, col, dist, number):
        if isinstance(dist, tuple):
            p, dist_ = dist
            result = np.random.choice(dist_, size=int(number), p=p)
        else:
            result = dist.rvs(size=(int(number)))
        return result


class ClassificationSynthesiser(BaseSynthesiser):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_dist_by_class(self, df, c, numerical_columns_indices):
        X_c = df[df['target'] == c].drop('target', axis=1).values
        dist = {}
        for index in range(X_c.shape[1]):
            if index in numerical_columns_indices:
                hist = np.histogram(X_c[:, index].astype(float), density=True)
                dist[index] = sps.rv_histogram(hist)
            else:
                counts = df.iloc[:, index].value_counts().sort_index()
                dist[index] = (counts.values / counts.sum(), counts.index)
        return dist

    def generate_sample_class(self, class_counts, i, dist_sep_class, n_jobs):
        label, number = class_counts[i]
        dists = dist_sep_class[label]
        with jb.Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                jb.delayed(self.gen_col_sample)(
                    col,
                    dist,
                    number) for col,
                dist in dists.items())
        label_sample = np.array(results).T
        target_sample = np.full(shape=(int(number), 1), fill_value=label)
        full_label_sample = np.concatenate([label_sample, target_sample], axis=1)
        return full_label_sample


class RegressionSynthesiser(BaseSynthesiser):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_dist(self, df, numerical_columnns_indices):
        X = df.drop('target', axis=1).values
        dist = {}
        for index in range(X.shape[1]):
            if index in numerical_columnns_indices:
                hist = np.histogram(X[:, index].astype(float), density=True)
                dist[index] = sps.rv_histogram(hist)
            else:
                counts = df.iloc[:, index].value_counts().sort_index()
                dist[index] = (counts.values / counts.sum(), counts.index)
        return dist

    def generate_sample(self, dists, n_jobs, number):
        with jb.Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                jb.delayed(self.gen_col_sample)(
                    col,
                    dist,
                    number) for col,
                dist in dists.items())
        label_sample = np.array(results).T
        return label_sample
