from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import special, optimize
from sklearn.base import BaseEstimator, RegressorMixin


class PsychometricCurve(BaseEstimator, RegressorMixin):
    mean_lims: Tuple[float, float]
    var_lims: Tuple[float, float]
    guess_rate_lims: Tuple[float, float]
    lapse_rate_lims: Tuple[float, float]

    def __init__(self, mean_lims=(0, 20), var_lims=(0.1, 5),
                 guess_rate_lims=(0.01, 0.05), lapse_rate_lims=(0.01, 0.05)):
        self.mean_: Union[None, float] = None
        self.var_: Union[None, float] = None
        self.guess_rate_: Union[None, float] = None
        self.lapse_rate_: Union[None, float] = None

        self.set_params(mean_lims=mean_lims,
                        var_lims=var_lims,
                        guess_rate_lims=guess_rate_lims,
                        lapse_rate_lims=lapse_rate_lims)

    def fit(self, x: np.array, y: np.array):
        popt, pcov = optimize.curve_fit(f=self._fit_func, xdata=x, ydata=y,
                                        p0=[np.mean(lims) for lims in [self.mean_lims, self.var_lims,
                                                                       self.guess_rate_lims, self.lapse_rate_lims]],
                                        bounds=([self.mean_lims[0], self.var_lims[0],
                                                 self.guess_rate_lims[0], self.lapse_rate_lims[0]],
                                                [self.mean_lims[1], self.var_lims[1],
                                                 self.guess_rate_lims[1], self.lapse_rate_lims[1]]))

        self.mean_ = popt[0]
        self.var_ = popt[1]
        self.lapse_rate_ = popt[2]
        self.guess_rate_ = popt[3]

        return self

    def predict(self, x: np.array) -> np.ndarray:
        return self._fit_func(x,
                              mean=self.mean_, var=self.var_, guess_rate=self.guess_rate_, lapse_rate=self.lapse_rate_)

    @staticmethod
    def _fit_func(x: np.array, mean: float, var: float, guess_rate: float, lapse_rate: float):
        return guess_rate + (1.0 - guess_rate - lapse_rate) * 0.5 * \
               (1.0 + special.erf((x - mean) / np.sqrt(2.0 * var ** 2.0)))

    def plot(self, x: np.array, y: np.ndarray = None, show: bool = True):
        fig, ax = plt.subplots()

        ax.plot(x, self.predict(x), label='y_pred')

        if y is not None:
            ax.scatter(x, y, label='y')

        ax.legend()
        ax.set_xlabel('N events')
        ax.set_ylabel('Prop fast')
        fig.tight_layout()

        if show:
            fig.show()

        return fig


if __name__ == "__main__":
    x = np.linspace(start=12, stop=16, num=6)
    y = (x > x.mean()).astype(float)
    y[2] = y[2] + np.abs(np.random.rand())
    y[3] = y[3] - np.abs(np.random.rand())

    pc = PsychometricCurve().fit(x, y)
    pc.plot(x, y)
    pc.score(x, y)
