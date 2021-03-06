import copy
import unittest

import numpy as np

from msi_models.stim.multi_two_gap.multi_two_gap_stim import MultiTwoGapStim
from msi_models.stim.multi_two_gap.multi_two_gap_template import MultiTwoGapTemplate


class TestMultiTwoGapStim(unittest.TestCase):
    _sut = MultiTwoGapStim

    def test_construction_from_unmatched_template_works(self):
        stim = self._sut(MultiTwoGapTemplate['unmatched_async'].set_options().build())
        self.assertIsInstance(stim, MultiTwoGapStim)

    def test_construction_from_matched_template_works(self):
        stim = self._sut(MultiTwoGapTemplate['matched_async'].set_options().build())
        self.assertIsInstance(stim, MultiTwoGapStim)

    def test_construction_from_sync_template_works(self):
        stim = self._sut(MultiTwoGapTemplate['matched_sync'].set_options().build())
        self.assertIsInstance(stim, MultiTwoGapStim)

    def test_cached_stim_returns_consistent_ys(self):
        stim = self._sut(MultiTwoGapTemplate['unmatched_async'].set_options(cache=True).build())

        left_y_first_call = copy.copy(stim.y_objs[0].y)
        left_y_second_call = copy.copy(stim.y_objs[0].y)
        right_y_first_call = copy.copy(stim.y_objs[1].y)
        right_y_second_call = copy.copy(stim.y_objs[1].y)

        left_y_mask_first_call = copy.copy(stim.y_mask_objs[0].y)
        left_y_mask_second_call = copy.copy(stim.y_mask_objs[0].y)
        right_y_mask_first_call = copy.copy(stim.y_mask_objs[1].y)
        right_y_mask_second_call = copy.copy(stim.y_mask_objs[1].y)

        self.assertTrue(np.all(stim.params.cache))
        for y1, y2 in zip(left_y_first_call, left_y_second_call):
            self.assertAlmostEqual(y1, y2, 7)
        for y1, y2 in zip(left_y_mask_first_call, left_y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)
        for y1, y2 in zip(right_y_first_call, right_y_second_call):
            self.assertAlmostEqual(y1, y2, 7)
        for y1, y2 in zip(right_y_mask_first_call, right_y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)

    def test_uncached_stim_returns_consistent_ys(self):
        stim = self._sut(MultiTwoGapTemplate['matched_sync'].set_options(cache=False, seed=123).build())

        left_y_first_call = copy.copy(stim.y_objs[0].y)
        left_y_second_call = copy.copy(stim.y_objs[0].y)
        right_y_first_call = copy.copy(stim.y_objs[1].y)
        right_y_second_call = copy.copy(stim.y_objs[1].y)

        left_y_mask_first_call = copy.copy(stim.y_mask_objs[0].y)
        left_y_mask_second_call = copy.copy(stim.y_mask_objs[0].y)
        right_y_mask_first_call = copy.copy(stim.y_mask_objs[1].y)
        right_y_mask_second_call = copy.copy(stim.y_mask_objs[1].y)

        self.assertTrue(~np.any(stim.params.cache))
        # Event and background noise will differ
        for y1, y2 in zip(np.abs(left_y_first_call), np.abs(left_y_second_call)):
            self.assertAlmostEqual(y1, y2, 1)
        for y1, y2 in zip(left_y_mask_first_call, left_y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)
        for y1, y2 in zip(np.abs(right_y_first_call), np.abs(right_y_second_call)):
            self.assertAlmostEqual(y1, y2, 1)
        for y1, y2 in zip(right_y_mask_first_call, right_y_mask_second_call):
            self.assertAlmostEqual(y1, y2, 7)
