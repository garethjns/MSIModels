import unittest

from msi_models.stim.two_gap.two_gap_stim import TwoGapStim
from msi_models.stim.two_gap.two_gap_templates import template_sine_events, template_noisy_sine_events


class TestTwoGapStim(unittest.TestCase):
    _sut = TwoGapStim

    def test_construction_from_single_event_template__works(self):
        stim = self._sut(template_sine_events())
        self.assertIsInstance(stim, TwoGapStim)

    def test_single_event_template_return_expected_shape(self):
        stim = self._sut(template_sine_events())
        self.assertEqual(len(stim.y.y), 1000)

    def test_single_event_template_increased_fs_return_expected_shape(self):
        stim = self._sut(template_sine_events(fs=2000))
        self.assertEqual(len(stim.y.y), 2000)

    def test_single_event_template_decreased_fs_return_expected_shape(self):
        stim = self._sut(template_sine_events(fs=500))
        self.assertEqual(len(stim.y.y), 500)

    def test_single_event_template_modified_duration_return_expected_shape(self):
        stim = self._sut(template_sine_events(duration=2000,
                                              duration_tol=4))
        self.assertEqual(len(stim.y.y), 2000)

    def test_construction_from_compound_event_template_works(self):
        stim = self._sut(template_noisy_sine_events())
        self.assertIsInstance(stim, TwoGapStim)

    def test_compound_event_template_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events())
        self.assertEqual(len(stim.y.y), 1000)

    def test_compound_event_template_increased_fs_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(fs=2000))
        self.assertEqual(len(stim.y.y), 2000)

    def test_compound_event_template_decreased_fs_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(fs=500))
        self.assertEqual(len(stim.y.y), 500)

    def test_compound_event_template_modified_duration_return_expected_shape(self):
        stim = self._sut(template_noisy_sine_events(duration=2000,
                                                    duration_tol=4))
        self.assertEqual(len(stim.y.y), 2000)