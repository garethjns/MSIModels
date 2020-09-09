import unittest

import matplotlib.pyplot as plt
import numpy as np

from msi_models.models.psychometric_curve import PsychometricCurve


class TestPsychometricCurve(unittest.TestCase):
    def setUp(self):
        self._sut = PsychometricCurve()
        self._x = np.linspace(start=12, stop=16, num=6)
        self._y = (self._x > self._x.mean()).astype(float)

    def test_fit_func(self):
        # Arrange
        x = np.linspace(start=12, stop=16, num=6)

        # Act
        y = PsychometricCurve._fit_func(x, guess_rate=0.01, lapse_rate=0.01, mean=14, var=1)

        # Assert
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(x), len(y))
        for yp, yt in zip(y, [0.03229513, 0.12276828, 0.34768669, 0.65231331, 0.87723172, 0.96770487]):
            self.assertAlmostEqual(yp, yt)

    def test_fit_with_simple_data(self):
        # Act
        self._sut.fit(self._x, self._y)

        # Assert
        self.assertIsNotNone(self._sut.mean_)
        self.assertIsNotNone(self._sut.var_)
        self.assertIsNotNone(self._sut.guess_rate_)
        self.assertIsNotNone(self._sut.lapse_rate_)
        self.assertGreater(self._sut.score(self._x, self._y), 0.98)

    def test_fit_with_noisy_data(self):
        # Arrange
        self._y[2] = self._y[2] + np.abs(np.random.rand() / 10)
        self._y[3] = self._y[3] - np.abs(np.random.rand() / 10)

        # Act
        self._sut.fit(self._x, self._y)

        # Assert
        self.assertIsNotNone(self._sut.mean_)
        self.assertIsNotNone(self._sut.var_)
        self.assertIsNotNone(self._sut.guess_rate_)
        self.assertIsNotNone(self._sut.lapse_rate_)
        self.assertGreater(self._sut.score(self._x, self._y), 0.90)

    def test_plot_with_y(self):
        # Arrange
        self._y[2] = self._y[2] + np.abs(np.random.rand())
        self._y[3] = self._y[3] - np.abs(np.random.rand())
        self._sut.fit(self._x, self._y)

        # Act
        fig = self._sut.plot(self._x, self._y, show=False)

        # Assert
        self.assertIsInstance(fig, plt.Figure)
