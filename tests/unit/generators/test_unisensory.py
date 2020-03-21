import unittest
from typing import Generator

import numpy as np

from msi_models.models.generators.unisensory import unisensory_binary


class TestUnisensoryBinary(unittest.TestCase):

    def test_is_generator(self):
        gen = unisensory_binary()

        self.assertIsInstance(gen, Generator)

    def test_defaults_return_expected_shapes(self):
        gen = unisensory_binary()
        x, y = next(gen)

        self.assertIsInstance(x, dict)
        self.assertEqual(1, len(x.values()))
        x_ = list(x.values())[0]
        self.assertIsInstance(x_, np.ndarray)
        self.assertEqual((2, 665, 1), x_.shape)

        self.assertIsInstance(y, dict)
        self.assertEqual(2, len(y.values()))
        y_ = list(y.values())[0]
        self.assertIsInstance(y_, np.ndarray)
        self.assertEqual((2,), y_.shape)

    def test_alternates_return_expected_shapes(self):
        gen = unisensory_binary(n=3,
                                events=10,
                                duration=1100,
                                fs=1200)
        x, y = next(gen)

        self.assertIsInstance(x, dict)
        self.assertEqual(1, len(x.values()))
        x_ = list(x.values())[0]
        self.assertIsInstance(x_, np.ndarray)
        self.assertEqual((3, 1320, 1), x_.shape)

        self.assertIsInstance(y, dict)
        self.assertEqual(2, len(y.values()))
        y_ = list(y.values())[0]
        self.assertIsInstance(y_, np.ndarray)
        self.assertEqual((3,), y_.shape)
