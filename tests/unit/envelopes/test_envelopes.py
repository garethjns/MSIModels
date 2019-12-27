import unittest

import numpy as np

from signal.envelopes.templates import ConstantEnvelope, CosEnvelope, CosRiseEnvelope


ONES = np.ones(shape=(1000, ))
ZEROS = np.zeros(shape=(800, ))


class TestConstantEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = ConstantEnvelope()

    def test_envelope_on_ones(self) -> None:
        self.assertTrue(np.all(self.env(ONES) == ONES))

    def test_envelope_on_zeros(self) -> None:
        self.assertTrue(np.all(self.env(ZEROS) == ZEROS))


class TestCosEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = CosEnvelope()

    def test_envelope_on_ones(self) -> None:
        self.assertLess(np.sum(self.env(ONES)), np.sum(ONES))

    def test_envelope_on_zeros(self) -> None:
        self.assertTrue(np.all(np.round(self.env(ZEROS)) == ZEROS))


class TestCosRiseEnvelope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = CosRiseEnvelope(fs=1000,
                                  rise=200)

    def test_envelope_on_ones(self) -> None:
        self.assertLess(np.sum(self.env(ONES)), np.sum(ONES))

    def test_envelope_on_zeros(self) -> None:
        self.assertTrue(np.all(self.env(ZEROS) == ZEROS))