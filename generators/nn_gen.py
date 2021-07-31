import numpy as np
import pandas as pd
from sklearn import utils
import joblib as jb
import scipy.stats as sps
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from . import dist_base


class SequentialRegressionSynthesiser(dist_base.RegressionSynthesiser):
    def __init__(
            self,
            n_jobs=1,
            discrete_columns=[],
            optimizer='adam',
            callbacks=None,
            loss='mse',
            epochs=75,
            batch_size=32,
            validation_split=.35):
        self.n_jobs = n_jobs
        self.discrete_columns = discrete_columns
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

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
        self.dist = self.model_dist(df, self.numerical_columns_indices)
        self.model = keras.Sequential([
            layers.Dense(100, activation='relu'),
            layers.Dropout(.2),
            layers.BatchNormalization(),
            layers.Dense(100, activation='relu'),
            layers.Dropout(.2),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=self.callbacks,
            verbose=False)
        return self

    def sample(self, n=100):
        utils.check_scalar(n, name='n', target_type=int)
        sample = self.generate_sample(self.dist, self.n_jobs, n)
        target = self.model.predict(sample)
        full = pd.DataFrame(np.concatenate(
            [sample, target], axis=1)).sample(frac=1)
        sample, target = full.drop(
            full.shape[1] - 1, axis=1), full[full.shape[1] - 1]
        return sample.values, target.values
