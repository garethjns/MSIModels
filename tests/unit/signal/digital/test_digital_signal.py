import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from signal.digital.digital_siginal import DigitalSignal
from signal.digital.conversion import pts_to_ms, ms_to_pts


class TestFunctions(unittest.TestCase):
    def test_ms_to_pts(self):
        self.assertEqual(ms_to_pts(t_ms=1000,
                                   fs=1000), 1000)

        self.assertEqual(ms_to_pts(t_ms=500,
                                   fs=1000), 500)

        self.assertEqual(ms_to_pts(t_ms=1000,
                                   fs=500), 500)

        self.assertEqual(ms_to_pts(t_ms=20,
                                   fs=800), 16)

    def test_pts_to_ms(self):
        self.assertEqual(pts_to_ms(t_pts=1000,
                                   fs=1000), 1000)

        self.assertEqual(pts_to_ms(t_pts=500,
                                   fs=1000), 500)


class TestDigitalSignal(unittest.TestCase):

    @classmethod
    @patch.multiple(DigitalSignal, __abstractmethods__=set())
    def setUpClass(cls) -> None:
        cls.dt_1 = DigitalSignal(fs=100,
                                 duration=1000)
        cls.dt_2 = DigitalSignal(fs=100,
                                 duration=500)

    def test_eq(self):
        self.assertTrue(self.dt_1 == self.dt_1)
        self.assertFalse(self.dt_1 == self.dt_2)

    def test_unique(self):
        self.assertEqual(len(np.unique([self.dt_1, self.dt_1])), 1)
        self.assertEqual(len(np.unique([self.dt_1, self.dt_2])), 2)

    def test_duration_in_pts(self):
        self.assertEqual(self.dt_1.duration_pts, 100)

    @patch.multiple(DigitalSignal, __abstractmethods__=set())
    def test_with_defaults(self):
        dt = DigitalSignal()

        self.assertEqual(len(dt.x), 20)
        self.assertEqual(len(dt.x_pts), 20)

    def test_plot(self):
        self.dt_1.plot()
        plt.show()