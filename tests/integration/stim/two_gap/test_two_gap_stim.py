import copy
import unittest

import numpy as np

from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events, template_noisy_sine_events


class TestTwoGapStim(unittest.TestCase):
    _sut = TwoGapStim

    def test_construction_from_single_event_template_works(self):
        stim = self._sut(template_sine_events())
        self.assertIsInstance(stim, TwoGapStim)

    def test_single_event_template_return_expected_shape(self):
        stim = self._sut(template_sine_events())
        self.assertEqual(len(stim.y.y), 1300)

    def test_single_event_template_increased_fs_return_expected_shape(self):
        stim = self._sut(template_sine_events(fs=2000))
        self.assertEqual(len(stim.y.y), 2600)

    def test_single_event_template_decreased_fs_return_expected_shape(self):
        stim = self._sut(template_sine_events(fs=500))
        self.assertEqual(len(stim.y.y), 650)

    def test_single_event_template_modified_duration_return_expected_shape(self):
        stim = self._sut(template_sine_events(duration=2000,
                                              duration_tol=4))
        self.assertEqual(len(stim.y.y), 2000)

    def test_construction_from_compound_event_template_works(self):
        stim = self._sut(template_noisy_sine_events())
        self.assertIsInstance(stim, TwoGapStim)

    def test_compound_event_template_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events())
        self.assertEqual(len(stim.y.y), 1300)

    def test_compound_event_template_increased_fs_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(fs=2000))
        self.assertEqual(len(stim.y.y), 2600)

    def test_compound_event_template_decreased_fs_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(fs=500))
        self.assertEqual(len(stim.y.y), 650)

    def test_compound_event_template_modified_duration_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(duration=2000,
                                                    duration_tol=4))
        self.assertEqual(len(stim.y.y), 2000)

    def test_cached_stim_returns_consistent_ys(self):
        stim = self._sut(template_sine_events(cache=True))

        y_first_call = copy.copy(stim.y.y)
        y_second_call = copy.copy(stim.y.y)
        y_mask_first_call = copy.copy(stim.y_mask.y)
        y_mask_second_call = copy.copy(stim.y_mask.y)

        self.assertTrue(stim.params.cache)
        for y1, y2 in zip(y_first_call, y_second_call):
            self.assertAlmostEqual(y1, y2, 7)
        for y1, y2 in zip(y_mask_first_call, y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)

    def test_uncached_stim_returns_consistent_ys(self):
        stim = self._sut(template_sine_events(cache=False))

        y_first_call = copy.copy(stim.y.y)
        y_second_call = copy.copy(stim.y.y)
        y_mask_first_call = copy.copy(stim.y_mask.y)
        y_mask_second_call = copy.copy(stim.y_mask.y)

        # Event and background noise will differ
        self.assertFalse(stim.params.cache)
        for y1, y2 in zip(np.abs(y_first_call), np.abs(y_second_call)):
            self.assertAlmostEqual(y1, y2, 1)
        for y1, y2 in zip(y_mask_first_call, y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)
