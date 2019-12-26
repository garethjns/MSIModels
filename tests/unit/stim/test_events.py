import copy
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from stim.events import Event, NoiseEvent, SineEvent, CompoundEvent


class TestEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev_1 = Event(fs=10000,
                         duration=20,
                         mag=1,
                         clip=np.inf)

    def test_length_as_expected(self):
        self.assertEqual(len(self.ev_1.y), 200)

    @patch.multiple(Event, _envelope_f=partial(np.ones, shape=(200,)))
    def test_y_no_envelope(self):
        ev = Event(fs=10000,
                   duration=20,
                   mag=1,
                   clip=np.inf)

        self.assertListEqual(list(ev.y), list(np.ones(shape=(200,))))

    def test_plot(self):
        self.ev_1.plot()
        plt.show()


class TestNoiseEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev_1 = NoiseEvent(fs=10000,
                              duration=20,
                              mag=1,
                              clip=np.inf,
                              cache=True)

    def test_invalid_noise_dist_raises_error(self):
        self.assertRaises(ValueError, lambda: NoiseEvent(fs=10000,
                                                         duration=20,
                                                         mag=1,
                                                         clip=np.inf,
                                                         dist='invalid'))

    def test_dist_statistics(self):
        ev_norm = NoiseEvent(fs=10000,
                             duration=20,
                             mag=1,
                             clip=np.inf,
                             dist='normal')

        ev_uniform = NoiseEvent(fs=10000,
                                duration=20,
                                mag=1,
                                clip=np.inf,
                                dist='uniform')

        self.assertAlmostEqual(float(np.mean(ev_norm.y)), 0.0, 0)
        self.assertAlmostEqual(float(np.mean(ev_norm.y)), float(np.mean(ev_uniform.y)), 0)

    def test_consistent_y_each_regen(self):
        y_1 = copy.deepcopy(self.ev_1.y)

        self.ev_1.clear()

        y_2 = copy.deepcopy(self.ev_1.y)

        self.assertTrue(np.all(y_1 == y_2))

    def test_no_seed_different_y(self):
        ev_1 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf)

        ev_2 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_same_seed_same_y(self):
        ev_1 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf,
                          seed=1)

        ev_2 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf,
                          seed=1)

        self.assertTrue(ev_1 == ev_2)
        self.assertTrue(np.all(ev_1.y == ev_2.y))

    def test_diff_seed_diff_y(self):
        ev_1 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf,
                          seed=1)

        ev_2 = NoiseEvent(fs=10000,
                          duration=20,
                          mag=1,
                          clip=np.inf,
                          seed=2)

        self.assertFalse(ev_1 == ev_2)
        self.assertFalse(np.all(ev_1.y == ev_2.y))

    def test_plot(self):
        self.ev_1.plot()
        plt.show()


class TestSineEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev_1 = SineEvent(fs=10000,
                             duration=20,
                             mag=1,
                             clip=np.inf)

    def test_plot(self):
        self.ev_1.plot()
        plt.show()


class TestCompoundEvent(unittest.TestCase):
    def test_sine_noise_multiply(self):
        sine_event = SineEvent(fs=10000)
        noise_event = NoiseEvent(mag=0.3,
                                 fs=10000)

        compound_event = sine_event * noise_event

        self.assertAlmostEqual(np.mean(sine_event.y), np.mean(compound_event.y), 1)
        self.assertGreater(np.std(compound_event.y), np.std(sine_event.y))

    def test_construct_from_list_of_2(self):
        sine_event = SineEvent()
        compound_event = CompoundEvent(events=[sine_event, sine_event])

        self.assertAlmostEqual(np.mean(sine_event.y), np.mean(compound_event.y), 0)
        self.assertGreater(np.std(compound_event.y), np.std(sine_event.y))

    def test_construct_from_list_of_3(self):
        sine_event = SineEvent()
        noise_event = NoiseEvent()
        compound_event = CompoundEvent(events=[sine_event, sine_event, noise_event])

        self.assertAlmostEqual(np.mean(sine_event.y), np.mean(compound_event.y), 0)
        self.assertGreater(np.std(compound_event.y), np.std(sine_event.y))

    def test_incompatible_events_fs_raises_error(self):
        sine_event = SineEvent()
        noise_event = NoiseEvent(fs=100)

        self.assertRaises(ValueError, lambda: CompoundEvent(events=[sine_event, sine_event, noise_event]))

    def test_incompatible_events_duration_raises_error(self):
        sine_event = SineEvent(duration=100)
        noise_event = NoiseEvent()

        self.assertRaises(ValueError, lambda: CompoundEvent(events=[sine_event, sine_event, noise_event]))
