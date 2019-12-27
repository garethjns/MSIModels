import copy
import unittest
from functools import partial
from unittest.mock import patch, MagicMock

import numpy as np
from matplotlib import pyplot as plt

from signal.events.event import Event, CompoundEvent
from signal.events.noise import NoiseEvent
from signal.events.tonal import SineEvent


class TestEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev_1 = Event(fs=10000,
                         duration=20,
                         mag=1,
                         clip=np.inf)

    def test_length_as_expected(self):
        self.assertEqual(len(self.ev_1.y), 200)

    def test_y_no_envelope(self):
        ev = Event(fs=10000,
                   duration=20,
                   mag=1,
                   clip=np.inf)
        ev.envelope = lambda x: x

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

        ev_norm.plot()
        ev_uniform.plot()
        plt.show()

        self.assertAlmostEqual(float(np.mean(ev_norm.y)), 0.0, -1)
        self.assertLess(float(np.mean(ev_norm.y)), float(np.mean(ev_uniform.y)), 0)
        self.assertLess(float(np.std(ev_uniform.y)), float(np.std(ev_norm.y)))

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

    @staticmethod
    def _mock_compound_event(fs, start, duration_pts) -> MagicMock:
        return MagicMock(fs=fs,
                         start=start,
                         duration_pts=duration_pts,
                         x=np.linspace(start, duration_pts, duration_pts)
)

    def test_combiner_with_matching_events(self):
        sine_event = SineEvent(fs=10000)
        noise_event = NoiseEvent(mag=0.3,
                                 fs=10000)

        mock_event = self._mock_compound_event(fs=1000,
                                               start=0,
                                               duration_pts=200)

        y = CompoundEvent._combiner(mock_event, [sine_event, noise_event])

        self.assertEqual(len(mock_event.x), mock_event.duration_pts)
        self.assertEqual(len(y), mock_event.duration_pts)

        plt.plot(mock_event.x, y)
        plt.show()

    def test_combiner_with_non_matching_events(self):
        sine_event = SineEvent(fs=1000,
                               duration=500,
                               start=500)
        noise_event = NoiseEvent(mag=0.3,
                                 duration=1000,
                                 fs=1000)

        mock_event = self._mock_compound_event(fs=1000,
                                               start=0,
                                               duration_pts=1000)

        y = CompoundEvent._combiner(mock_event, [sine_event, noise_event])

        self.assertEqual(len(mock_event.x), mock_event.duration_pts)
        self.assertEqual(len(y), mock_event.duration_pts)

        plt.plot(mock_event.x, y)
        plt.show()

    def test_combiner_with_multiple_non_matching_events(self):
        sine_event_1 = SineEvent(fs=1000,
                                 duration=1000,
                                 start=0)
        sine_event_2 = SineEvent(fs=1000,
                                 duration=1000,
                                 start=300)
        noise_event = NoiseEvent(mag=0.3,
                                 duration=1000,
                                 fs=1000,
                                 start=600)

        mock_event = self._mock_compound_event(fs=1000,
                                               start=0,
                                               duration_pts=1600)

        y = CompoundEvent._combiner(mock_event, [sine_event_1, sine_event_2, noise_event])

        self.assertEqual(len(mock_event.x), mock_event.duration_pts)
        self.assertEqual(len(y), mock_event.duration_pts)

        plt.plot(mock_event.x, y)
        plt.show()

    def test_combiner_with_equal_weighted_events(self):
        constant_event_1 = Event(fs=1000,
                                 duration=1000,
                                 start=0)
        constant_event_2 = Event(fs=1000,
                                 duration=1000,
                                 start=0)

        mock_event = self._mock_compound_event(fs=1000,
                                               start=0,
                                               duration_pts=1000)

        y = CompoundEvent._combiner(mock_event, [constant_event_1, constant_event_2])

        self.assertAlmostEqual(float(max(y)), 1, 3)
        self.assertEqual(len(mock_event.x), mock_event.duration_pts)
        self.assertEqual(len(y), mock_event.duration_pts)

        plt.plot(mock_event.x, y)
        plt.show()

    def test_combiner_with_unequal_weighted_events(self):
        constant_event_1 = Event(fs=1000,
                                 duration=1000,
                                 start=0)
        constant_event_2 = Event(fs=1000,
                                 duration=1000,
                                 start=0)

        mock_event = self._mock_compound_event(fs=1000,
                                               start=0,
                                               duration_pts=1000)

        y = CompoundEvent._combiner(mock_event, [constant_event_1, constant_event_2],
                                    weights=[0.5, 1])

        self.assertAlmostEqual(float(max(y)), 1.5, 3)
        self.assertEqual(len(mock_event.x), mock_event.duration_pts)
        self.assertEqual(len(y), mock_event.duration_pts)

        constant_event_1.plot()
        constant_event_2.plot()
        plt.plot(mock_event.x, y)
        plt.show()

    def test_sine_noise_multiply(self):
        sine_event = SineEvent(fs=10000)
        noise_event = NoiseEvent(mag=0.3,
                                 fs=10000)

        compound_event = sine_event * noise_event

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 1)

    def test_construct_from_list_of_2(self):
        sine_event = SineEvent()
        compound_event = CompoundEvent(events=[sine_event, sine_event])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        # This should be same as weighting is equal proportions by default
        self.assertAlmostEqual(float(np.std(compound_event.y)), float(np.std(sine_event.y)))

    def test_construct_from_list_of_2_set_weights(self):
        sine_event = SineEvent()
        compound_event = CompoundEvent(events=[sine_event, sine_event],
                                       weights=[1, 1])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)
        # This should be greater as weights are a total of 2
        self.assertGreater(np.std(compound_event.y), np.std(sine_event.y))

    def test_construct_from_list_of_3(self):
        sine_event = SineEvent()
        noise_event = NoiseEvent()
        compound_event = CompoundEvent(events=[sine_event, sine_event, noise_event])

        self.assertAlmostEqual(float(np.mean(sine_event.y)), float(np.mean(compound_event.y)), 0)

    def test_construct_from_offset_list_of_3(self):
        sine_event_1 = SineEvent(start=100,
                                 duration=1000)
        sine_event_2 = SineEvent(start=300,
                                 duration=1000)
        sine_event_3 = SineEvent(start=500,
                                 duration=1000)

        compound_event = CompoundEvent(events=[sine_event_1, sine_event_2, sine_event_3])

        sine_event_1.plot()
        sine_event_2.plot()
        sine_event_3.plot()
        compound_event.plot(show=True)

    def test_incompatible_events_fs_raises_error(self):
        sine_event = SineEvent()
        noise_event = NoiseEvent(fs=100)

        self.assertRaises(ValueError, lambda: CompoundEvent(events=[sine_event, sine_event, noise_event]))