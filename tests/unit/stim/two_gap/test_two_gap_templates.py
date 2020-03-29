import unittest

from msi_models.stim.two_gap.two_gap_params import TwoGapParams
from msi_models.stim.two_gap.two_gap_templates import template_noisy_sine_events, template_sine_events


class TestTemplateSineEvents(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = template_sine_events

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut()
        self.assertIsInstance(params, TwoGapParams)

    def test_example_compatible_params_raise_no_errors_with_different_fs(self):
        params = self._sut(fs=500)
        self.assertIsInstance(params, TwoGapParams)

    def test_example_compatible_params_raise_no_errors_with_different_duration(self):
        params = self._sut(duration=800)
        self.assertIsInstance(params, TwoGapParams)


class TestTemplateNoisySineEvents(unittest.TestCase):
    def setUp(self) -> None:
        self._sut = template_noisy_sine_events

    def test_compatible_params_raise_no_validation_errors(self):
        params = self._sut()
        self.assertIsInstance(params, TwoGapParams)

    def test_example_compatible_params_raise_no_errors_with_different_fs(self):
        params = self._sut(fs=500)
        self.assertIsInstance(params, TwoGapParams)

    def test_example_compatible_params_raise_no_errors_with_different_duration(self):
        params = self._sut(duration=800)
        template_noisy_sine_events(duration=100)
        self.assertIsInstance(params, TwoGapParams)
