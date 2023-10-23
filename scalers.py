import numpy as np
import typing


class MinMaxScaler:
    def fit(self, data):
        self.min = np.amin(data, axis=0)
        self.max = np.amax(data, axis=0)

    def transform(self, data):
        data = (data - self.min)/(self.max - self.min)
        return data


class StandardScaler:
    def fit(self, data):
        self.math = np.mean(data, axis=0)
        self.disp = np.sqrt(np.var(data, axis=0))

    def transform(self, data):
        return (data - self.math)/(self.disp)
